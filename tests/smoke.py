"""Runs every part of the package that does not need a DICOM on disk.

Not a substitute for real data — `dicom.headers`, `dicom.ordering` and
`dicom.laterality` are exercised here only on synthetic frames. What this does cover
is everything that can be wrong without any file being involved: shapes, the presence
mask actually changing an output, gradients reaching the head, the fingerprint
detecting a perturbation, and the loop learning a signal that is there.

    python -m tests.smoke
"""

from __future__ import annotations

import copy
import json
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import torch
from torch import nn

from dataclasses import replace
from rsna.config import Config, PixelRules, TARGETS, pool_parts
from rsna.dicom import (
    CacheMismatch, annotate, build_cache, hdr_vec, load_cache, order_slices,
    plan_cache, read_slot, sample_indices,
)
from rsna.model import (ENCODERS, HuggingFaceViT, TimmBackbone, build_model,
                        check_fingerprint, fingerprint, find_encoder, spec_for)
from rsna.model.fingerprint import WeightsError
from rsna.train import assign_folds, augment, build_targets, fit, predict, take_window
from stub_backbone import StubBackbone, StubTimmConv, StubTimmViT
from synthetic_dicom import series_record, write_series

PASSED, FAILED = [], []


def check(name: str, condition: bool, detail: str = "") -> None:
    (PASSED if condition else FAILED).append(name)
    print(f"  {'ok  ' if condition else 'FAIL'}  {name}{'  — ' + detail if detail else ''}")


def test_config() -> None:
    print("\nconfig")
    c = Config()
    check("round-trips through a dict", Config.from_dict(c.to_dict()) == c)
    check("cache tag changes with the rules",
          c.cache_tag() != c.replace(rules=PixelRules(order="dominant_axis")).cache_tag())
    try:
        Config.from_dict({**c.to_dict(), "unknown_field": 1})
        check("refuses an unknown field", False)
    except ValueError:
        check("refuses an unknown field", True)


def test_headers() -> None:
    print("\ndicom.headers")
    check("parses a multi-value header", (hdr_vec("1|2|3", 3) == [1, 2, 3]).all())
    check("rejects a short one", hdr_vec("1|2", 3) is None)
    df = pd.DataFrame([
        dict(SeriesDescription="SAG PD FS", SequenceName="", ScanOptions="FS",
             ScanningSequence="SE", RepetitionTime="3000", EchoTime="30", PixelSpacing="0.4|0.4"),
        dict(SeriesDescription="COR T1", SequenceName="", ScanOptions="SAT_GEMS",
             ScanningSequence="SE", RepetitionTime="500", EchoTime="12", PixelSpacing="0.5|0.5"),
        dict(SeriesDescription="ax stir", SequenceName="", ScanOptions="",
             ScanningSequence="SE", RepetitionTime="4000", EchoTime="80", PixelSpacing="0.6|0.6"),
    ])
    out = annotate(df)
    check("recovers fat suppression", list(out["fatsat"]) == [True, False, True],
          "GE writes SAT_GEMS for spatial saturation, which is not fat sat")
    check("recovers weighting", list(out["weight"]) == ["PD", "T1", "T2"])
    check("derives the fluid axis", list(out["fluid"]) == [True, False, True])


def test_folds() -> None:
    print("\ntrain.folds")
    c = Config()
    train = pd.DataFrame({
        "StudyInstanceUID": [f"s{i}" for i in range(10)],
        "Report": ["same text"] * 4 + [f"text {i}" for i in range(6)],
        **{t: [np.nan] * 10 for t in TARGETS},
    })
    folds = assign_folds(train, c)
    check("identical reports share a fold", len(set(folds.iloc[:4])) == 1,
          "otherwise a study is scored on a target it trained on")
    check("one fold per study", len(folds) == len(train))

    derived = pd.DataFrame({t: np.full(10, 0.7) for t in TARGETS},
                           index=[f"s{i}" for i in range(10)])
    derived.index.name = "StudyInstanceUID"
    gold = train.copy()
    gold.loc[0, TARGETS] = 1.0
    y, w = build_targets(list(train["StudyInstanceUID"]), gold, derived, c)
    check("expert labels outweigh derived ones",
          w[0][0] == c.gold_weight and w[1][0] == 1.0)
    check("expert labels win on value", y[0][0] == 1.0 and y[1][0] == 0.7)

    confidence = pd.DataFrame({t: np.full(10, 0.2) for t in TARGETS},
                              index=[f"s{i}" for i in range(10)])
    _, weighted = build_targets(list(train["StudyInstanceUID"]), gold, derived, c,
                                confidence=confidence)
    check("uses source confidence for weak-label weights",
          weighted[1][0] == 0.25 + 0.75 * 0.2)
    check("expert labels ignore the confidence table", weighted[0][0] == c.gold_weight)

    # The published baseline writes `0.25 + 0.75 * conf`. We write
    # `floor + (1 - floor) * conf` so the floor can be tuned — the teams disagree about
    # it, pilkwang uses 0.25 and prvsiyan's V52 uses 0.15. At the default floor the two
    # must be the *same* number, not a close one: a run meant to reproduce a published
    # score cannot differ from it by a rounding decision.
    literal = np.float32(0.25) + np.float32(0.75) * confidence.to_numpy(np.float32)
    check("the default floor reproduces the published formula bit for bit",
          np.array_equal(weighted[1:].view(np.int32), literal[1:].view(np.int32)),
          "0.25 and 0.75 are both exact in binary, so 1 - 0.25 is exactly 0.75")

    _, lowered = build_targets(list(train["StudyInstanceUID"]), gold, derived,
                               c.replace(weight_floor=0.15), confidence=confidence)
    check("a different floor moves the weights",
          abs(lowered[1][0] - (0.15 + 0.85 * 0.2)) < 1e-6)

    # -- assertedness: weighting a table that reports no confidence ------------- #
    from rsna.data.labels import silence_of
    from rsna.train.folds import assertedness

    grid = np.linspace(0.0, 1.0, 41)
    check("assertedness at silence 0.5 is exactly prvsiyan's certainty",
          np.allclose(assertedness(grid, 0.5), np.clip(2 * np.abs(grid - 0.5), 0, 1)),
          "it is that formula with the anchor made explicit, not a rival to it")

    shifted = assertedness(grid, 0.25)
    check("the silence is the floor of the curve",
          abs(assertedness(np.array([0.25]), 0.25)[0]) < 1e-12
          and shifted.min() == 0.0)
    check("a weak assertion outranks the silence",
          assertedness(np.array([0.35]), 0.25)[0] > assertedness(np.array([0.25]), 0.25)[0],
          "the defect certainty has on this table: 0.35 is nearer 0.5 than 0.25 is")
    check("a confident denial weighs as much as a confident claim",
          abs(assertedness(np.array([0.0]), 0.25)[0]
              - assertedness(np.array([1.0]), 0.25)[0]) < 1e-12,
          "the two sides are normalised separately because they are not the same length")

    derived_soft = pd.DataFrame({t: np.r_[np.full(5, 0.25), np.full(5, 0.97)]
                                 for t in TARGETS},
                                index=[f"s{i}" for i in range(10)])
    told = c.replace(weights="assertedness", silence=0.25)
    _, w4 = build_targets(list(train["StudyInstanceUID"]), gold, derived_soft, told)
    check("silent studies fall to the floor, asserted ones do not",
          abs(w4[1][0] - c.weight_floor) < 1e-6 and w4[9][0] > 0.9,
          "and no confidence column was involved")

    try:
        build_targets(list(train["StudyInstanceUID"]), gold, derived_soft,
                      c.replace(weights="assertedness"))
        check("refuses assertedness without a silence level", False)
    except ValueError:
        check("refuses assertedness without a silence level", True)

    check("the registry knows each table's silence",
          silence_of("llm_labels_v4_blend.csv") == 0.25
          and silence_of("report_labels_v2.csv") == 0.28
          and silence_of("/tmp/never-seen.csv") is None,
          "measured from the cells pilkwang's verdict column calls UNK")

    for bad, why in [({"weight_floor": 1.5}, "a floor outside [0, 1]"),
                     ({"weights": "nonsense"}, "an unknown weights mode")]:
        try:
            build_targets(list(train["StudyInstanceUID"]), gold, derived,
                          c.replace(**bad), confidence=confidence)
            check(f"refuses {why}", False)
        except ValueError:
            check(f"refuses {why}", True)


def _model(cfg: Config, **kw):
    return build_model(cfg, backbone=StubBackbone(dim=32, patch=8, **kw))


def test_model() -> None:
    print("\nmodel")
    cfg = Config(img=56, slices=3, group=3)
    m = _model(cfg)
    imgs = torch.randint(0, 256, (4, cfg.n_slot, cfg.group, cfg.img, cfg.img), dtype=torch.uint8)
    full = torch.ones(4, cfg.n_slot)
    partial = full.clone()
    partial[0, 2:] = 0

    out = m(imgs, full, cfg.img)
    check("emits one logit per target", out.shape == (4, len(TARGETS)))
    check("the presence mask changes the output",
          not torch.allclose(out, m(imgs, partial, cfg.img)),
          "an absent slot must be excluded, not fed zeros")

    torch.nn.functional.binary_cross_entropy_with_logits(
        m(imgs, full, cfg.img), torch.rand(4, len(TARGETS))).backward()
    check("gradients reach the head",
          all(p.grad is not None and p.grad.abs().sum() > 0 for p in m.head.parameters()))
    check("early encoder blocks stay frozen",
          not any(p.requires_grad for p in m.backbone.encoder.layer[0].parameters()))

    check("names its encoder in the config",
          (cfg.encoder, cfg.encoder_variant, cfg.pool) == ("dinov2", "small", "cls_mean"),
          "a checkpoint fitted on one encoder cannot be read by another")
    check("reports a missing encoder instead of guessing",
          find_encoder(cfg, root="/nonexistent") is None)

    f = fingerprint(m, cfg)
    check("fingerprint is deterministic", np.array_equal(f, fingerprint(m, cfg)))
    m.head.out.bias.data += 0.05
    try:
        check_fingerprint(m, cfg, f)
        check("fingerprint catches perturbed weights", False)
    except WeightsError:
        check("fingerprint catches perturbed weights", True,
              "weights read through the wrong preprocessing predict, they do not raise")


def test_augment() -> None:
    print("\ntrain.augment")
    cfg = Config(img=56, slices=6, group=3)
    rows = torch.randint(0, 256, (4, cfg.n_slot, cfg.slices, cfg.img, cfg.img), dtype=torch.uint8)
    g = take_window(rows, cfg.group, cfg)
    check("takes one group of channels", g.shape[2] == cfg.group)
    a = augment(g, cfg)
    check("preserves shape and dtype", a.shape == g.shape and a.dtype == g.dtype)
    check("actually perturbs", not torch.equal(a, g))
    check("stays in byte range", int(a.min()) >= 0 and int(a.max()) <= 255)


def test_loop() -> None:
    print("\ntrain.loop")
    cfg = Config(img=56, slices=6, group=3, epochs=6, batch_studies=4, eval_batch=4,
                 lr_head=5e-3, unfreeze_last=2)
    n, rng = 64, np.random.default_rng(0)
    cache = rng.integers(0, 120, (n, cfg.n_slot, cfg.slices, cfg.img, cfg.img), dtype=np.uint8)
    mask = np.ones((n, cfg.n_slot), np.float32)
    y = np.zeros((n, len(TARGETS)), np.float32)
    y[:, 0] = rng.random(n) > 0.5
    # A signal that is genuinely there: the first slot is brighter when the target is 1.
    positive = y[:, 0] > 0.5
    cache[positive, 0] = np.clip(cache[positive, 0].astype(int) + 120, 0, 255).astype(np.uint8)
    w = np.ones_like(y)

    model = _model(cfg, n_layer=2)
    result = fit(model, cache, mask, y, w, np.arange(48), np.arange(48, 64), cfg, "cpu")
    check("loss decreases", result.history[-1].loss < result.history[0].loss,
          f"{result.history[0].loss:.3f} -> {result.history[-1].loss:.3f}")
    check("recovers a planted signal", result.best_holdout_auc > 0.95,
          f"holdout AUC {result.best_holdout_auc:.3f}")
    check("keeps the best state", result.state_dict is not None and len(result.state_dict) > 0)

    p = predict(model, cache, mask, np.arange(48, 64), cfg, "cpu")
    check("predictions are probabilities", bool((p >= 0).all() and (p <= 1).all()))
    check("predictions vary between studies", len(set(np.round(p[:, 0], 4))) > 1)


def test_pixels() -> None:
    """The pixel path, on real .dcm files written for the purpose."""

    import tempfile

    import pydicom

    print("\ndicom.pixels")
    tmp = Path(tempfile.mkdtemp())
    cfg = Config(img=32, group=3, slices=3, crop_mm=16.0)

    check("samples across the central band",
          list(sample_indices(12, 3, cfg.band)) == [2, 5, 8],
          "the outermost slices of a knee series are soft tissue outside the joint")

    directory = write_series(tmp / "ordered", n_slices=12, size=64)
    record = series_record(directory)
    ordered, resolved = order_slices(record["dir"], record["files"], cfg)
    check("recovers order from geometry", resolved)

    def first_pixel(name):
        return int(pydicom.dcmread(directory / name).pixel_array.mean())

    by_name = [first_pixel(f) for f in record["files"]]
    by_geometry = [first_pixel(f) for f in ordered]
    check("file-name order is not physical order", by_name != sorted(by_name),
          f"{by_name} — a SOP Instance UID is unique, not ordered")
    check("geometric order is physical order", by_geometry == sorted(by_geometry))

    record["ordered"] = ordered
    out = read_slot(record, cfg)
    check("returns bytes at the requested size",
          out.shape == (cfg.group, cfg.img, cfg.img) and out.dtype == torch.uint8)
    means = [float(out[i].float().mean()) for i in range(out.shape[0])]
    check("slices come back in order", means == sorted(means), str([round(m) for m in means]))
    check("normalises to the full byte range", int(out.min()) == 0 and int(out.max()) == 255,
          "1st-99th percentile: MR intensity has no absolute scale")

    # Corrupt exactly the files the sampler will reach, so the test does not depend on
    # the arbitrary file-name order: with a corrupt slice, geometry cannot be read and
    # read_slot falls back to that order.
    partial = series_record(write_series(tmp / "partial", n_slices=12, size=64))
    sampled = sample_indices(len(partial["files"]), cfg.group, cfg.band)
    for position in sampled[:2]:
        (tmp / "partial" / partial["files"][int(position)]).write_bytes(b"not a dicom")
    failures = []
    out = read_slot(partial, cfg, decode_failures=failures)
    check("survives unreadable slices", out is not None and len(failures) == 1,
          "filled from the nearest slice that decoded, and reported once")
    check("a filled slice is a real slice, not black",
          out is not None and int(out.max()) > 0)

    dead = series_record(write_series(tmp / "dead", n_slices=4, corrupt={0, 1, 2, 3}))
    failures = []
    check("reports a dead series absent, not black",
          read_slot(dead, cfg, decode_failures=failures) is None and len(failures) == 1,
          "the presence mask can express absent; it cannot express black")


def test_laterality() -> None:
    """Mirroring a knee, and the one axis where it is not a pixel flip."""

    from rsna.dicom import normalise_laterality

    print("\ndicom.laterality")
    img = np.arange(4 * 3 * 3, dtype=np.uint8).reshape(4, 3, 3)

    check("a left knee is never touched",
          np.array_equal(normalise_laterality(img, "Sagittal", "L", True), img)
          and np.array_equal(normalise_laterality(img, "Coronal", "L", True), img))
    check("coronal mirrors the pixels",
          np.array_equal(normalise_laterality(img, "Coronal", "R"), img[..., ::-1]))
    check("sagittal mirrors the SLICE ORDER, not the pixels",
          np.array_equal(normalise_laterality(img, "Sagittal", "R", True), img[::-1]),
          "its left-right axis is the through-plane one, so the mirror is a reversal")
    check("and only when the rule asks",
          np.array_equal(normalise_laterality(img, "Sagittal", "R", False), img),
          "off by default: thirteen packages were fitted without it, and a manifest "
          "written before the field existed reads back as the default")

    # The cache tag is the only guard here — the fingerprint is computed on synthetic
    # pixels that never pass through normalise_laterality, so it cannot see this change.
    plain = Config()
    flipped = plain.replace(rules=PixelRules(sagittal_flip=True))
    check("turning it on changes the cache tag",
          plain.cache_tag() != flipped.cache_tag(),
          f"{plain.cache_tag()} vs {flipped.cache_tag()}")
    check("and leaving it off does not",
          plain.cache_tag() == Config().replace(
              rules=PixelRules(sagittal_flip=False)).cache_tag(),
          "so the thirteen existing caches stay valid")


def test_cache() -> None:
    """Decode a two-study corpus, reload it, and refuse a stale one."""

    import tempfile

    print("\ndicom.cache")
    tmp = Path(tempfile.mkdtemp())
    cfg = Config(img=32, group=3, slices=3, crop_mm=16.0)

    check("sizes the cache from free memory", plan_cache(4407, cfg, n_test=1322) >= 1,
          "coverage gives way before resolution when the budget binds")

    slot_map, lat_map = {}, {}
    for study in ("A", "B"):
        slot_map[study] = {}
        for name in ("SAG_FLUID_FS", "COR_FLUID_FS"):
            directory = write_series(tmp / f"{study}_{name}", n_slices=10, size=64)
            record = series_record(directory)
            record["SeriesInstanceUID"] = f"{study}_{name}"
            slot_map[study][name] = record
        # B is a right knee, so it must come back mirrored onto the left convention.
        lat_map[study] = "R" if study == "B" else "L"

    path = tmp / "cache.npy"
    studies, cache, mask = build_cache(slot_map, {}, lat_map, cfg, cache_slices=3,
                                       path=path, order_cache=tmp / "order.json")
    check("fills only the slots that exist",
          mask.sum() == 4 and mask.shape == (2, cfg.n_slot),
          "two studies x two of six slots")
    check("writes bytes of the right shape",
          cache.shape == (2, cfg.n_slot, 3, cfg.img, cfg.img) and cache.dtype == np.uint8)

    left, right = np.asarray(cache[0, 1]), np.asarray(cache[1, 1])
    check("mirrors a right knee onto the left convention",
          not np.array_equal(left, right) and np.array_equal(left, right[:, :, ::-1]),
          "five of the twelve targets are named for a side")

    reloaded_studies, reloaded, _ = load_cache(path, cfg, 3)
    check("reloads from disk unchanged",
          reloaded_studies == studies
          and np.array_equal(np.asarray(reloaded), np.asarray(cache)))

    for name, bad in (("a different pixel rule",
                       lambda: load_cache(path, cfg.replace(rules=PixelRules(order="dominant_axis")), 3)),
                      ("a different slice count", lambda: load_cache(path, cfg, 6))):
        try:
            bad()
            check(f"refuses {name}", False)
        except CacheMismatch:
            check(f"refuses {name}", True,
                  "same shape, different pixels — nothing downstream could tell")


def test_windows() -> None:
    """Which slices reach the encoder — a property of the config, not a utility."""

    print("\nconfig.windows")
    cfg = Config(slices=12, group=3)
    check("training windows are disjoint", cfg.windows(overlap=False) == [0, 3, 6, 9],
          "one drawn at random per step")
    check("inference windows overlap", cfg.windows(overlap=True) == list(range(10)),
          "all of them, logits averaged")
    check("a short run keeps the central windows",
          cfg.windows(overlap=True, limit=4) == [3, 4, 5, 6],
          "the ends of a stack are soft tissue outside the joint")
    check("a cache smaller than one window still yields one",
          Config(slices=2, group=3).windows() == [0])
    check("windows travel with the weights",
          Config.from_dict(cfg.to_dict()).windows() == cfg.windows(),
          "so inference cannot use a split training never saw")


def test_submission() -> None:
    """The file Kaggle scores."""

    import tempfile

    from rsna.infer import benchmark_submission, blend_members, write_submission

    print("\ninfer.submission")
    tmp = Path(tempfile.mkdtemp())
    studies = [f"study{i}" for i in range(5)]

    frame = pd.read_csv(benchmark_submission(studies, tmp / "bench.csv"))
    check("benchmark is 0.5 everywhere", bool((frame[TARGETS].to_numpy() == 0.5).all()),
          "what a crash after the decode pass should leave behind")

    rng = np.random.default_rng(0)
    predictions = rng.random((5, len(TARGETS))) * 1000  # arbitrary scale
    frame = pd.read_csv(write_submission(predictions, studies, tmp / "a.csv"))
    values = frame[TARGETS].to_numpy()
    check("predictions are written as ranks",
          bool((values > 0).all() and (values <= 1).all() and np.isclose(values.max(), 1.0)),
          "the metric reads order only, and ranks make members blendable")
    check("ranking preserves the order",
          bool((np.argsort(values[:, 0]) == np.argsort(predictions[:, 0])).all()))

    blended = blend_members([predictions, rng.random((5, len(TARGETS)))], [0.6, 0.4])
    check("blending stays in rank space",
          bool((blended > 0).all() and (blended <= 1).all()))

    for name, bad in (("a wrong shape",
                       lambda: write_submission(np.zeros((4, 12)), studies, tmp / "x.csv")),
                      ("non-finite predictions",
                       lambda: write_submission(np.full((5, 12), np.nan), studies, tmp / "x.csv")),
                      ("duplicate studies",
                       lambda: benchmark_submission(["a", "a"], tmp / "x.csv"))):
        try:
            bad()
            check(f"refuses {name}", False)
        except ValueError:
            check(f"refuses {name}", True)


def test_figures() -> None:
    """Every figure builds under the default config.

    They are only pictures, but they are how the pipeline gets read, and they break
    silently: a figure that assumed `group` slices kept working until `slices` stopped
    equalling it. Building each one is cheap and catches exactly that.
    """

    import tempfile

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from rsna import viz

    print("\nviz")
    tmp = Path(tempfile.mkdtemp())
    cfg = Config(img=32, group=3, slices=6, crop_mm=16.0)
    record = series_record(write_series(tmp / "viz", n_slices=16, size=64))
    record["plane"] = "Coronal"

    viz.verify_against_read_slot(record, cfg)
    check("the illustrated chain equals read_slot", True,
          "otherwise the figures show a preprocessing nobody runs")

    built = []
    for name, kwargs in (("figure_stack", {}), ("figure_steps", {"side": "R"}),
                         ("figure_cache", {"side": "R"}), ("figure_channels", {"side": "R"}),
                         ("figure_windows", {"side": "R", "overlap": False}),
                         ("figure_windows", {"side": "R", "overlap": True})):
        try:
            figure = getattr(viz, name)(record, cfg, **kwargs)
            plt.close(figure)
            built.append(name)
        except Exception as exc:
            check(f"{name} builds", False, f"{type(exc).__name__}: {exc}")
    check("every figure builds", len(built) == 6, ", ".join(sorted(set(built))))


def test_encoders() -> None:
    """The seam every architecture goes through.

    `Model` and `build_model` ask a spec questions rather than testing the encoder's
    name, so these are the questions — a new family that answers them is usable without
    either of those functions changing. That is the whole claim of the design, and it
    is only worth making if something checks it.

    The timm specs run against stand-ins rather than timm itself, which is fair for the
    plumbing — prefix counts, block flattening, a feature map becoming tokens — and not
    fair for the loading, which `TimmBackbone.load` alone does and nothing here covers.
    """

    print("\nmodel.encoders")
    check("the registry names dinov2", ENCODERS["dinov2"] is HuggingFaceViT)
    check("and the timm families", ENCODERS["coatnet"] is TimmBackbone)
    try:
        spec_for(Config(encoder="not_an_encoder"))
        check("refuses an unregistered encoder", False)
    except ValueError:
        check("refuses an unregistered encoder", True)

    # The spec aliases the backbone instead of owning it. That is what keeps the state
    # dict's keys where they were, and `load_member` refuses a single unexpected name —
    # so nothing may quietly turn the spec into a module or break the alias.
    pinned = build_model(Config(img=32, slices=3, group=3),
                         backbone=StubBackbone(dim=64, patch=8))
    check("the spec contributes nothing to the state dict",
          not any(k.startswith("spec") for k in pinned.state_dict()),
          "the thirteen packages fitted before it existed still load key for key")
    clone = copy.deepcopy(pinned)
    check("and the alias survives a copy",
          pinned.spec.module is pinned.backbone
          and clone.spec.module is clone.backbone,
          "a spec left pointing at the original would run the wrong weights, silently")

    hf = HuggingFaceViT(StubBackbone(dim=64, patch=8))
    check("hf: one prefix token, and it is a class token",
          (hf.n_prefix, hf.has_class_token, hf.dim) == (1, True, 64))
    check("hf: blocks in depth order", len(hf.blocks()) == 12)
    check("hf: takes three channels", hf.in_channels == 3)

    vit = TimmBackbone(StubTimmViT(dim=64, n_layer=6, patch=8))
    check("timm vit: width from num_features, prefix from num_prefix_tokens",
          (vit.dim, vit.n_prefix, vit.has_class_token) == (64, 1, True))
    check("timm vit: blocks in depth order", len(vit.blocks()) == 6)
    check("timm vit: forward_features emits tokens",
          tuple(vit.tokens(torch.zeros(2, 3, 32, 32)).shape) == (2, 1 + 16, 64))

    conv = TimmBackbone(StubTimmConv(dim=64, stages=(2, 2)))
    check("timm conv: no class token to read",
          (conv.n_prefix, conv.has_class_token) == (0, False),
          "so the head is told, rather than given a corner patch as a summary")
    check("timm conv: stages flatten into blocks", len(conv.blocks()) == 4,
          "unfreeze_last counts the same thing on a staged and on a flat model")
    check("timm conv: the feature map becomes tokens",
          tuple(conv.tokens(torch.zeros(2, 3, 32, 32)).shape) == (2, 16, 64))

    check("a missing class token narrows the slot feature",
          (pool_parts("cls_mean", True), pool_parts("cls_mean", False)) == (2, 1))
    try:
        pool_parts("nonsense")
        check("refuses an unknown pooling", False)
    except ValueError:
        check("refuses an unknown pooling", True)

    # -- a family the model has never seen, driven end to end ------------------- #
    cfg = Config(img=32, slices=3, group=3, encoder="coatnet",
                 encoder_variant="stub", unfreeze_last=3)
    model = build_model(cfg, backbone=StubTimmConv(dim=64, stages=(2, 2))).eval()
    check("a conv encoder sizes the head from what it actually emits",
          model.head.proj[1].in_features == 64,
          "one part, not two: there is no class token to concatenate")
    frozen = [p.requires_grad for stage in model.backbone.stages
              for p in stage.blocks[0].parameters()]
    check("unfreeze_last counts from the end", frozen[0] is False and frozen[-1] is True)

    imgs = torch.randint(0, 256, (2, cfg.n_slot, cfg.window_size, cfg.img, cfg.img),
                         dtype=torch.uint8)
    mask = torch.ones(2, cfg.n_slot)
    out = model(imgs, mask, cfg.img)
    check("and it produces one logit per target", tuple(out.shape) == (2, len(TARGETS)))

    # -- stem "none": the encoder takes the window itself ----------------------- #
    raw = Config(img=32, slices=12, group=12, stem="none", encoder="coatnet",
                 encoder_variant="stub")
    check("stem 'none' asks the encoder for the whole window",
          (raw.encoder_channels, raw.window_size) == (12, 12))
    native = build_model(raw, backbone=StubTimmConv(dim=64, in_chans=12)).eval()
    check("no stem is built for it", native.stem is None)
    check("and the normalisation is sized to the window",
          tuple(native.mean.shape) == (1, 12, 1, 1),
          "the grey equivalent of the ImageNet statistics, not the RGB spread")
    wide = torch.randint(0, 256, (2, raw.n_slot, 12, raw.img, raw.img), dtype=torch.uint8)
    check("a twelve-channel encoder runs",
          tuple(native(wide, torch.ones(2, raw.n_slot), raw.img).shape)
          == (2, len(TARGETS)))

    try:
        build_model(raw.replace(encoder="dinov2"), backbone=StubBackbone(dim=64, patch=8))
        check("refuses a stem the encoder cannot take", False)
    except ValueError:
        check("refuses a stem the encoder cannot take", True,
              "three-channel backbone, twelve-channel window — caught at build, not "
              "inside the first convolution")

    # -- against timm itself, where it is installed ----------------------------- #
    # The stubs above follow timm's contract. This checks the contract is what timm
    # actually does — a stub can only ever confirm what I believed when I wrote it.
    # Skipped rather than failed when timm is absent: no fitted package needs it, and
    # the DINOv2 path must stay installable without it.
    try:
        import timm
    except ImportError:
        print("  skip  timm is not installed, so the real CoAtNet check did not run")
    else:
        real = TimmBackbone(
            timm.create_model("coatnet_pico_rw_224", pretrained=False, num_classes=0))
        check("timm: a real CoAtNet reports no class token",
              (real.n_prefix, real.has_class_token) == (0, False))
        check("timm: its stages flatten into a block list", len(real.blocks()) > 4,
              f"{len(real.blocks())} blocks across its stages")
        tok = real.tokens(torch.zeros(1, 3, 224, 224))
        check("timm: forward_features returns a map, and it becomes tokens",
              tok.ndim == 3 and tok.shape[-1] == real.dim, f"{tuple(tok.shape)}")
        wide = TimmBackbone(timm.create_model("coatnet_pico_rw_224", pretrained=False,
                                              num_classes=0, in_chans=12))
        check("timm: in_chans is read back off the rebuilt convolution",
              wide.in_channels == 12,
              "read from the model, not from what config asked for — so a backbone "
              "built for the wrong width is caught rather than agreed with")

def test_encoder_unchanged() -> None:
    """The DINOv2 path, pinned before it is refactored.

    Not a claim that these numbers are right — a claim that they are what the code
    produced on 2026-09-23, so a change of plumbing that quietly changes the model
    cannot pass unnoticed. Regenerate `tests/golden/encoder_dinov2.json` only when a
    change to the model is *intended*, and say so in the commit that does it.

    The real encoder is not in git, so this runs against the stub. The other half of
    the check is that every existing package still verifies its own fingerprint, which
    needs the weights and is done with `scripts.rescore`.
    """

    print("\nmodel.encoder (characterisation)")
    golden_file = Path(__file__).with_name("golden") / "encoder_dinov2.json"
    if not golden_file.is_file():
        check("golden file is present", False, str(golden_file))
        return
    golden = json.loads(golden_file.read_text())

    cases = {"cls_mean": Config(img=56, slices=3, group=3),
             "cls_mean_focal": Config(img=56, slices=3, group=3, pool="cls_mean_focal"),
             "compress": Config(img=56, slices=12, group=3, stem="compress")}
    for name, cfg in cases.items():
        want = golden[name]
        torch.manual_seed(0)
        model = build_model(cfg, backbone=StubBackbone(dim=64, patch=8)).eval()
        g = torch.Generator().manual_seed(1)
        imgs = torch.randint(0, 256, (3, cfg.n_slot, cfg.window_size, cfg.img, cfg.img),
                             generator=g, dtype=torch.uint8)
        mask = torch.ones(3, cfg.n_slot)
        mask[0, 2:] = 0
        with torch.no_grad():
            got = model(imgs, mask, cfg.img).flatten().tolist()

        worst = max(abs(a - b) for a, b in zip(got, want["out"]))
        check(f"{name}: output unchanged", worst < 1e-9, f"worst element {worst:.2e}")

        names = sorted(n for n, p in model.named_parameters() if p.requires_grad)
        missing, extra = set(want["trainable"]) - set(names), set(names) - set(want["trainable"])
        check(f"{name}: the same parameters train", not missing and not extra,
              f"lost {sorted(missing)[:2]} gained {sorted(extra)[:2]}" if (missing or extra) else "")

        fp = fingerprint(model, cfg).flatten().tolist()
        worst_fp = max(abs(a - b) for a, b in zip(fp, want["fingerprint"]))
        check(f"{name}: fingerprint unchanged", worst_fp < 1e-9, f"worst {worst_fp:.2e}")


def _optimiser_groups(model, config):
    """The groups `rsna.train.loop.fit` builds, so the test moves when the loop does."""

    groups = [
        {"params": [p for p in model.backbone.parameters() if p.requires_grad],
         "lr": config.lr_backbone},
        {"params": list(model.head.parameters()), "lr": config.lr_head},
    ]
    if model.stem is not None:
        groups.append({"params": list(model.stem.parameters()), "lr": config.lr_head})
    return groups


def test_stems() -> None:
    """Both ways of turning cached slices into three channels."""

    print("\nmodel.stems")
    for stem, channels, passes in (("window", 3, 10), ("compress", 12, 1)):
        cfg = Config(stem=stem, slices=12, group=3, img=56)
        check(f"{stem}: encoder takes {channels} channels", cfg.window_size == channels)
        check(f"{stem}: {passes} pass(es) per slot at inference",
              len(cfg.windows()) == passes,
              "one window and no averaging is the point of compress")

        # `requires_grad` says a parameter *may* move; only the optimiser makes it. A
        # stem outside every param group trains to nothing while the run reports it as
        # trainable and finishes without a word — which is exactly what happened.
        model = _model(cfg)
        named = dict(model.named_parameters())
        seen = {id(q) for g in _optimiser_groups(model, cfg) for q in g["params"]}
        orphans = [n for n, q in named.items() if q.requires_grad and id(q) not in seen]
        check(f"{stem}: every trainable parameter reaches the optimiser",
              not orphans, ", ".join(orphans[:3]))

        model = _model(cfg)
        imgs = torch.randint(0, 256, (2, cfg.n_slot, cfg.window_size, cfg.img, cfg.img),
                             dtype=torch.uint8)
        full = torch.ones(2, cfg.n_slot)
        partial = full.clone()
        partial[0, 3:] = 0
        out = model(imgs, full, cfg.img)
        check(f"{stem}: emits one logit per target", out.shape == (2, len(TARGETS)))
        check(f"{stem}: the presence mask still changes the output",
              not torch.allclose(out, model(imgs, partial, cfg.img)))

        f = fingerprint(model, cfg)
        check(f"{stem}: fingerprint is deterministic",
              np.array_equal(f, fingerprint(model, cfg)))

    from rsna.model.stems import build_stem
    check("window builds no stem", build_stem(Config(stem="window")) is None)
    blank = build_stem(Config(stem="compress", slices=8))(
        torch.zeros(2, 8, 32, 32))
    check("an absent slot stays blank through the stem", bool((blank == 0).all()),
          "the projection has a bias, so zeros in would not give zeros out")
    try:
        build_stem(Config(stem="nonsense"))
        check("refuses an unknown stem", False)
    except ValueError:
        check("refuses an unknown stem", True)


def test_experiments() -> None:
    """Named configs, and that they say what they claim."""

    import experiments

    print("\nexperiments")
    names = experiments.available()
    check("both approaches are registered",
          {"window_reference", "depth_compress"} <= set(names), ", ".join(names))
    baseline, compress = experiments.load("window_reference"), experiments.load("depth_compress")
    check("window_reference is the ported approach", baseline.config.stem == "window")
    check("depth_compress compresses the stack", compress.config.stem == "compress")
    check("an experiment defines the whole run",
          all(getattr(compress, k) for k in ("split", "labels", "encoder"))
          and compress.fold == 0,
          "--experiment alone is enough to reproduce it")
    check("an approach travels in the weights",
          Config.from_dict(compress.config.to_dict()).stem == "compress",
          "so load_member rebuilds the right model without being told")
    try:
        experiments.load("does_not_exist")
        check("refuses an unknown experiment", False)
    except SystemExit:
        check("refuses an unknown experiment", True)


def test_eval() -> None:
    print("\neval")
    import tempfile
    from types import SimpleNamespace

    import rsna.eval as ev

    rng = np.random.default_rng(0)
    n_target = len(TARGETS)

    # -- metrics, on arrays whose answer is known -------------------------------- #
    y = np.zeros((40, n_target), np.float32)
    y[:20] = 1.0
    perfect = y.copy()
    check("a perfect ranking scores 1", ev.macro_auc(y, perfect) == 1.0)
    check("an inverted ranking scores 0", ev.macro_auc(y, 1.0 - perfect) == 0.0)
    one_class = np.ones((40, n_target), np.float32)
    check("one class present scores NaN",
          np.isnan(ev.macro_auc(one_class, perfect)))

    scaled = ev.rank_normalise(perfect * 17.0 + 3.0)
    check("rank normalising does not move the AUC",
          abs(ev.macro_auc(y, scaled) - 1.0) < 1e-9,
          "AUC reads ranks, so the scale must not matter")
    check("rank normalising lands in [0, 1]",
          bool(scaled.min() >= 0.0 and scaled.max() <= 1.0))
    spread = ev.rank_normalise(rng.random((40, n_target)))
    check("rank normalising spans the range when nothing ties",
          bool(spread.min() == 0.0 and spread.max() == 1.0),
          "ties share the mean rank, which is what keeps the AUC unchanged")

    noisy = rng.random((40, n_target)).astype(np.float32)
    lo, hi = ev.bootstrap_macro(y, noisy, n_boot=200, seed=0)
    check("the interval brackets the estimate",
          lo <= ev.macro_auc(y, noisy) <= hi, f"{lo:.3f}-{hi:.3f}")

    # -- a record survives a round trip through disk ----------------------------- #
    tmp = Path(tempfile.mkdtemp())
    for fold in range(3):
        truth = (rng.random((12, n_target)) < 0.4).astype(np.float32)
        truth[:, 2] = 0.0                      # one target with a single class
        pred = np.clip(truth * 0.6 + rng.random((12, n_target)) * 0.4, 0, 1)
        history = [SimpleNamespace(epoch=e, loss=1.0 / (e + 2),
                                   holdout_auc=0.5 + 0.01 * e,
                                   annotation_auc=float("nan")) for e in range(8)]
        ev.write_run_record(
            tmp / f"pkg-f{fold}", "unit", fold, "train_series", "labels.csv",
            SimpleNamespace(best_epoch=7, best_holdout_auc=0.57, history=history,
                            holdout_true=truth, holdout_pred=pred),
            [f"uid-{fold}-{i}" for i in range(12)])

    records = ev.read_sweep(tmp)
    check("reads back every fold", len(records) == 3)
    check("keeps the predictions", records[0].p.shape == (12, n_target))
    check("keeps the history", len(records[0].history) == 8)

    y_all, p_all = ev.pool(records)
    check("pooling gathers every study", len(y_all) == 36,
          "each study predicted once, by a model that never saw it")

    # -- the report is a pure function of those files ---------------------------- #
    report = ev.build(records)
    check("counts the studies", report["n_studies"] == 36)
    scored = [r for r in report["targets"] if np.isfinite(r["auc"])]
    check("every target carries an interval",
          all(r["lo"] <= r["auc"] <= r["hi"] for r in scored),
          "a point estimate on 12 studies without its width invites reading a "
          "difference the run never measured")
    wide, narrow = ev.auc_interval(0.8, 5, 75), ev.auc_interval(0.8, 400, 400)
    check("fewer positives widen the interval",
          (wide[1] - wide[0]) > 3 * (narrow[1] - narrow[0]),
          f"{wide[1] - wide[0]:.3f} vs {narrow[1] - narrow[0]:.3f}")
    check("reports one row per target", len(report["targets"]) == n_target)
    check("flags a target with one class",
          any(f["level"] == "error" and "one class" in f["text"]
              for f in report["flags"]))
    check("flags a run that ended on its best epoch",
          any("final epoch" in f["text"] for f in report["flags"]))
    check("summarises without a browser", "out-of-fold" in ev.summary_text(report))

    page = ev.render_html(report)
    check("the page carries its charts", page.count("<svg") >= 2 and "track" in page,
          "inline SVG and CSS, so they stay crisp and the page stays one file")
    check("the page fetches nothing", "http://" not in page and "https://" not in page
          and "<img" not in page)
    check("the page names the experiment", "unit" in page)

    written = ev.write(report, tmp / "report.html")
    check("writes the numbers beside the page", written.with_suffix(".json").is_file())
    check("says why there is no expert section", "--holdout-gold" in page)

    # -- the expert-labelled studies, held out of every fold --------------------- #
    gold_uids = [f"gold-{i}" for i in range(20)]
    gold_truth = (rng.random((20, n_target)) < 0.4).astype(np.float32)
    top = Path(tempfile.mkdtemp())
    for fold in range(3):
        truth = (rng.random((12, n_target)) < 0.4).astype(np.float32)
        pred = rng.random((12, n_target)).astype(np.float32)
        gold_pred = np.clip(gold_truth * 0.5 + rng.random((20, n_target)) * 0.5, 0, 1)
        order = list(range(20))
        if fold == 1:                      # one fold writes them in another order
            order = order[::-1]
        history = [SimpleNamespace(epoch=e, loss=1.0 / (e + 2),
                                   holdout_auc=0.5 + 0.01 * e,
                                   annotation_auc=0.5 + 0.02 * e) for e in range(8)]
        ev.write_run_record(
            top / f"pkg-f{fold}", "unit", fold, "train_series", "labels.csv",
            SimpleNamespace(best_epoch=7, best_holdout_auc=0.57, history=history,
                            holdout_true=truth, holdout_pred=pred,
                            gold_true=gold_truth[order],
                            gold_pred=gold_pred[order].astype(np.float32)),
            [f"uid-{fold}-{i}" for i in range(12)],
            gold_uids=[gold_uids[i] for i in order])

    gold_records = ev.read_sweep(top)
    check("reads the expert predictions back", gold_records[0].n_gold == 20)

    gy, gp, guids = ev.ensemble_gold(gold_records)
    check("averages the folds instead of stacking them", len(gy) == 20,
          "all three models saw the same studies; stacking would count each one "
          "three times and report an interval far too narrow")
    check("aligns the folds by study id, not by row",
          np.array_equal(gy, gold_truth[[gold_uids.index(u) for u in guids]]),
          "one fold wrote them reversed; a row-wise merge would score one patient "
          "against another's truth")

    gold_report = ev.build(gold_records)
    check("the report carries the expert section",
          gold_report["gold"] is not None and gold_report["gold"]["n"] == 20)
    gold_page = ev.render_html(gold_report)
    check("and a third curve for it", gold_page.count("<svg") >= 4,
          "loss, holdout, expert — plus the fold spread")

    try:
        ev.read_sweep(tmp / "pkg-f0" / "nothing-here")
        check("refuses a directory with no record", False)
    except (FileNotFoundError, OSError):
        check("refuses a directory with no record", True)


def test_annotation_side() -> None:
    """Recovering which compartment was clicked, without asking the annotator.

    The first bundle was drawn `--tagged-only`, and that filter selected a manufacturer.
    Admitting untagged studies raised the question of how the side gets known — and the
    answer is that it already is: clicking the lateral meniscus says which end of the
    stack is lateral, because the lateral meniscus sits distinctly nearer one end.
    Asking for it again would only restate the click.

    What this pins is the recovery, since a sign error here annotates the opposite
    meniscus 40 mm away and nothing downstream would report it.
    """

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from tools.annotate.to_mm import lateral_end_from_click, rows, side_at

    print("\nannotate.to_mm")

    a = {"n": 40}
    check("a click in the low half puts lateral at the first end",
          lateral_end_from_click(a, {"slice": 8}) == "first")
    check("and one in the high half at the last",
          lateral_end_from_click(a, {"slice": 31}) == "last")
    check("an export with no stack length recovers nothing rather than guessing",
          lateral_end_from_click({}, {"slice": 8}) is None)

    # +x is the patient's left, so the lateral end of a left knee carries the larger x.
    # Comparing the two ends rather than reading the sign of one is what survives the
    # isocentre being set on the knee instead of the midline: 161/161 against 159/161.
    left = {"x_first": 135.6, "x_last": 13.6}
    right = {"x_first": -32.0, "x_last": -157.7}
    check("the lateral end further toward the patient's left is a left knee",
          side_at(left, "first") == "L" and side_at(right, "last") == "R")
    check("and an isocentre-shifted study still resolves",
          side_at({"x_first": 50.0, "x_last": -30.0}, "first") == "L",
          "both ends would not agree on a sign; they do agree on an order")
    check("no geometry, no side",
          side_at({"x_first": None, "x_last": None}, "first") is None)

    # Where the tag exists the click restates it, so the two can disagree — and that is
    # the one error this dataset cannot survive: a point on the wrong compartment.
    def one(side, slice_):
        return rows({"annotations": [{
            "study": "s", "series": "x", "n": 40, "x_first": 135.6, "x_last": 13.6,
            "transform": {"row0": 0, "col0": 0, "scale": 1},
            "side": side, "side_from": "DICOM Laterality tag", "skipped": False,
            "points": {"lat_centre": {"slice": slice_, "sop": "u", "row": 1.0, "col": 2.0,
                                      "ipp": None, "iop": None, "ps": None}}}]}, "t")[0]

    check("a click agreeing with the tag is marked so", one("L", 8)["side_agrees"] is True)
    check("one on the other end is flagged", one("L", 31)["side_agrees"] is False,
          "the tag says lateral is the first end; the click landed near the last")
    check("with no tag the click supplies the side",
          one(None, 31)["side"] == "R" and one(None, 31)["side_tagged"] is None)
    check("and the tag still wins where it exists",
          one("L", 31)["side"] == "L" and one("L", 31)["side_from_click"] == "R",
          "both are kept; merging them would hide the disagreement")


def test_series_choice() -> None:
    """Which sagittal series an annotation bundle shows, and what `--prefer-3d` may not do.

    A 3D acquisition has to be in the annotation set — the classifier meets one in 3.4 % of
    studies and the landmark model would otherwise never have seen the appearance. But
    reaching for depth must not reach across the weighting: a study can hold a 28-slice PD
    beside a 140-slice T1, and a T1 fills no slot, so showing it would train the landmark
    model on pixels the ROI branch never receives. The first version of the flag did
    exactly that on four studies.
    """

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import pandas as pd
    from rsna.landmark.series import (LANDMARKS, PICKERS, pick_axial,
                                  pick_coronal, pick_sagittal)

    print("\nannotate.bundle")

    def sag(*rows):
        return pd.DataFrame([{"plane": "Sagittal", "weight": w, "fatsat": f,
                              "n_slices": n, "SeriesInstanceUID": uid}
                             for w, f, n, uid in rows])

    # The annotator is one HTML file serving every landmark, and no JavaScript engine is
    # installed here to run it — so what can be checked is that its wiring and the
    # registry still agree. Each of these guards a failure that is silent in a browser:
    # a help section for a point that no bundle names, a point whose section is missing
    # so the annotator sees the *other* landmark's instructions, or the storage key
    # quietly renamed out from under a session that has not been exported yet.
    tool = (Path(__file__).resolve().parents[1]
            / "tools/annotate/annotate.html").read_text()
    sections = set(re.findall(r'data-lm="([a-z_]+)"', tool))
    check("the annotator has a help section for every landmark, and no orphans",
          sections == set(LANDMARKS), f"{sorted(sections)}")
    check("and it takes the point from the bundle rather than from its own source",
          "m.landmark" in tool and "KEY = TARGET.id" in tool,
          "one tool, many points: the manifest names which one")
    check("the meniscus keeps the storage key its sessions were saved under",
          '"rsna-landmarks-v1"' in tool,
          "renaming it would hide annotation work still sitting unexported in a browser")
    figures = Path(__file__).resolve().parents[1] / "tools/annotate/figures"
    check("every landmark ships the picture of what its point means",
          all((figures / f"{k}.jpg").exists() for k in LANDMARKS),
          "a tool that ships without the definition of the point it collects gets two "
          "annotators' readings, averaged")

    def axi(*rows):
        return pd.DataFrame([{"plane": "Axial", "weight": w, "fatsat": f,
                              "n_slices": n, "SeriesInstanceUID": uid}
                             for w, f, n, uid in rows])

    # The patellofemoral joint wants the opposite of what the meniscus wants, and for
    # the same reason read the other way: a tear is signal inside fibrocartilage, which
    # fat suppression flattens; subchondral oedema is only visible once the fat is gone.
    mixed = axi(("PD", False, 40, "pd"), ("PD", True, 32, "pdfs"), ("T1", False, 60, "t1"))
    check("the axial picker prefers fat suppression, where the meniscus refuses it",
          pick_axial(mixed)["SeriesInstanceUID"] == "pdfs"
          and pick_sagittal(sag(("PD", True, 32, "pdfs"),
                                ("PD", False, 28, "pd")))["SeriesInstanceUID"] == "pd",
          "PD FS covers 72.2 % of studies axially and T2 FS 28.2 %, 98.5 % together")
    check("and falls back to T2 fat-suppressed before anything unsuppressed",
          pick_axial(axi(("T2", True, 30, "t2fs"),
                         ("PD", False, 44, "pd")))["SeriesInstanceUID"] == "t2fs")
    check("every landmark names a plane that has a picker",
          all(d["plane"] in PICKERS for d in LANDMARKS.values()),
          ", ".join(f"{k} on {v['plane']}" for k, v in LANDMARKS.items()))
    check("each point says what the annotator must be told about left and right",
          {k: v["side_cue"] for k, v in LANDMARKS.items()}
          == {"lat_centre": "stack-end", "pf_centre": "none",
              "mcl_centre": "image-side"},
          "three situations, not degrees of one: a sagittal stack has a lateral END, a "
          "coronal picture has a medial SIDE, an axial midline point has neither")
    check("and only the sagittal stack can recover a side from the click",
          [k for k, v in LANDMARKS.items() if v["side_cue"] == "stack-end"]
          == ["lat_centre"],
          "the click's distance to an end means nothing when the ends are front and back")
    check("the coronal picker prefers fat suppression, like the axial one",
          pick_coronal(pd.DataFrame([
              {"plane": "Coronal", "weight": "PD", "fatsat": False, "n_slices": 40,
               "SeriesInstanceUID": "pd"},
              {"plane": "Coronal", "weight": "PD", "fatsat": True, "n_slices": 30,
               "SeriesInstanceUID": "pdfs"}]))["SeriesInstanceUID"] == "pdfs",
          "a sprain is oedema, and oedema needs the fat gone")

    both = sag(("PD", True, 28, "pd"), ("T1", False, 140, "t1"), ("GRE", False, 92, "gre"))
    check("--prefer-3d does not cross the weighting to reach a deeper series",
          pick_sagittal(both, prefer_deep=True)["SeriesInstanceUID"] == "pd",
          "28-slice PD over a 140-slice T1: a T1 fills no slot")

    deep_pd = sag(("PD", True, 320, "pd3d"), ("PD", False, 30, "pd2d"))
    check("but it does take a deep series within the preference",
          pick_sagittal(deep_pd, prefer_deep=True)["SeriesInstanceUID"] == "pd3d")
    check("and without the flag the 2D one wins on fat suppression",
          pick_sagittal(deep_pd)["SeriesInstanceUID"] == "pd2d",
          "short-TE without FS shows signal inside the fibrocartilage")

    tie = sag(("PD", False, 29, "nofs"), ("PD", True, 29, "fs"))
    check("with no 3D present the flag changes nothing",
          pick_sagittal(tie, prefer_deep=True)["SeriesInstanceUID"]
          == pick_sagittal(tie)["SeriesInstanceUID"] == "nofs",
          "reordering two equally shallow series would rebuild the bundle for nothing")

    check("no PD falls back rather than returning nothing",
          pick_sagittal(sag(("T2", True, 30, "t2")))["SeriesInstanceUID"] == "t2")
    check("and no sagittal at all returns nothing",
          pick_sagittal(pd.DataFrame({"plane": ["Coronal"], "weight": ["PD"],
                                      "fatsat": [False], "n_slices": [30],
                                      "SeriesInstanceUID": ["c"]})) is None)


def test_annotation_side_rule() -> None:
    """The annotation bundle must threshold the knee, not the corner of the image.

    Two rules for the same question lived in this repository and the annotation tooling
    picked the weaker one: `tools/atlas/study.py` thresholded `ImagePositionPatient[0]`,
    which is the top-left corner and sits half a field of view — about 90 mm — from the
    anatomy. On study …96541786246 that reads -24 mm where the centre reads +66 mm and
    the tibia measures +84 mm on the axial series. A left knee was badged as right, the
    annotator followed the badge, and the click landed on the medial meniscus. The
    trained model found it: 49.7 mm of error against 1.8 mm median, and it was the only
    study in 304 where the two rules disagree.
    """

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from tools.atlas.study import side_of

    print("\nannotate.side")

    def headers(ipp_x, cols=512, spacing=0.35, laterality=None):
        return pd.DataFrame([{
            "StudyInstanceUID": "s", "Laterality": laterality,
            "ImagePositionPatient": f"{ipp_x}|-90.0|40.0",
            "ImageOrientationPatient": "1.0|0.0|0.0|0.0|1.0|0.0",
            "PixelSpacing": f"{spacing}|{spacing}", "Rows": cols, "Columns": cols}])

    # Corner at -24 mm, centre at -24 + 512*0.35/2 = +65.6 mm: the real study's numbers.
    check("the side comes from the centre of the image, not its corner",
          side_of(headers(-24.0))[0] == "L",
          "the corner reads -24 mm and would say R; the centre reads +66 mm")
    check("and says so, so a bundle records which rule badged it",
          "centre" in side_of(headers(-24.0))[1], side_of(headers(-24.0))[1])
    check("a knee genuinely near the midline gets no badge at all",
          side_of(headers(-100.0))[0] is None,
          "centre at -10 mm, inside the dead zone the rule was measured to be "
          "no better than chance in")
    check("the DICOM tag still wins over any geometry",
          side_of(headers(-24.0, laterality="R"))[0] == "R")


def test_excluded_list() -> None:
    """A withdrawn annotation must stay withdrawn, and a mistyped id must not pass.

    Study ids are forty digits. The first version of the exclusion file carried one
    typed from memory after reading only its last eleven characters: it matched nothing,
    the run printed no warning, and the annotation it was meant to remove stayed in the
    output. So a listed id that appears in no export is now an error.
    """

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from tools.annotate.to_mm import main as to_mm_main

    print("\nannotate.excluded")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        point = {"slice": 3, "sop": "u", "row": 1.0, "col": 2.0,
                 "ipp": None, "iop": None, "ps": None}
        export = {"annotations": [
            {"study": "A" * 40, "series": "s", "n": 30, "transform":
             {"row0": 0, "col0": 0, "scale": 1}, "side": "L", "skipped": False,
             "points": {"lat_centre": dict(point)}},
            {"study": "B" * 40, "series": "s", "n": 30, "transform":
             {"row0": 0, "col0": 0, "scale": 1}, "side": "L", "skipped": False,
             "points": {"lat_centre": dict(point)}}]}
        (tmp / "e.json").write_text(json.dumps(export))
        out = tmp / "out.csv"

        (tmp / "drop.csv").write_text("study,reason\n" + "A" * 40 + ",withdrawn\n")
        argv = sys.argv
        try:
            sys.argv = ["to_mm", str(tmp / "e.json"), "--excluded", str(tmp / "drop.csv"),
                        "-o", str(out)]
            to_mm_main()
            kept = pd.read_csv(out)["study"].tolist()
            check("a listed study is dropped", kept == ["B" * 40], str(kept))

            (tmp / "typo.csv").write_text("study,reason\n" + "Z" * 40 + ",withdrawn\n")
            sys.argv = ["to_mm", str(tmp / "e.json"), "--excluded", str(tmp / "typo.csv"),
                        "-o", str(out)]
            try:
                to_mm_main()
                check("an id that matches nothing is refused", False)
            except SystemExit:
                check("an id that matches nothing is refused", True,
                      "silence there keeps the annotation it was meant to remove")

            sys.argv = ["to_mm", str(tmp / "e.json"), "--excluded", str(tmp / "drop.csv"),
                        "--no-exclude", "-o", str(out)]
            to_mm_main()
            check("and the list can be turned off on purpose",
                  len(pd.read_csv(out)) == 2)
        finally:
            sys.argv = argv


def test_landmark_geometry() -> None:
    """A point in the patient, into the model's frame and back out.

    Every other part of this pipeline fails loudly. A coordinate bug does not: a
    transposed row and column still trains, still produces a loss curve that falls, and
    still predicts confidently into the opposite compartment. So the two conversions are
    written as an inverse pair and checked as one, on an oblique stack rather than an
    axis-aligned one — an axis-aligned test passes with the row and column swapped.
    """

    from rsna.dicom.geometry import normal_of, patient_mm, pixel_of, through_plane
    from rsna.landmark import (LandmarkConfig, Sampled, choose_depth, decode, encode,
                               place, project, resample_plane)
    from rsna.landmark.sample import to_native, to_resampled

    print("\nlandmark.geometry")

    # A sagittal stack as this corpus actually stores them: the normal is mostly -x but
    # tilted into y, which is what makes the depth axis a projection rather than a read.
    iop = np.array([-0.22489, 0.974384, 0.0, 0.0, 0.0, -1.0])
    # Deliberately anisotropic. Every series in this corpus has square pixels, so a
    # swapped PixelSpacing pairing would be invisible on real data and pass in
    # production while being wrong.
    spacing = np.array([0.417, 0.283])
    n = normal_of(iop)
    check("the normal is orthogonal to both orientation vectors",
          abs(n @ iop[:3]) < 1e-12 and abs(n @ iop[3:]) < 1e-12)

    p = patient_mm([10.0, -20.0, 30.0], iop, spacing, 123.5, 77.25)
    row, col = pixel_of([10.0, -20.0, 30.0], iop, spacing, p)
    check("patient_mm and pixel_of invert each other",
          abs(row - 123.5) < 1e-9 and abs(col - 77.25) < 1e-9,
          "PixelSpacing is (rows, columns) and IOP is (columns, rows) — the one "
          "mismatch this module exists to get right once")

    config = LandmarkConfig()

    # -- the depth rule ------------------------------------------------------ #
    keep, stride = choose_depth(30, 3.3, config)
    check("a 2D series is kept slice for slice", stride == 1 and len(keep) == 30)
    keep, stride = choose_depth(320, 0.4, config)
    check("a 3D series is thinned by an integer stride",
          stride == 8 and len(keep) == 40 and set(np.diff(keep)) == {8},
          "40 slices at 3.2 mm — inside the 2D spread, and no slice invented")
    keep, stride = choose_depth(50, 3.3, config)
    check("a 2D series two slices too long is cropped, not thinned",
          stride == 1 and len(keep) == config.slices,
          "halving 3.3 mm to 6.6 mm to save two peripheral slices is the worse trade")
    check("and the crop takes the middle", keep[0] == 1 and keep[-1] == 48)
    check("placement centres by default, and clips",
          place(30, config) == 9 and place(30, config, 999) == 18
          and place(60, config, 5) == 0)

    # -- the in-plane transform ---------------------------------------------- #
    _, t = resample_plane(np.zeros((512, 400), np.uint8), 0.3, config)
    back = to_resampled(t, *to_native(t, 130.5, 77.25))
    check("the in-plane transform inverts", np.allclose(back, (130.5, 77.25)))

    # -- a synthetic study, fully specified ---------------------------------- #
    slices, step = 30, 3.3
    ipp0 = np.array([135.6, -80.0, 40.0])
    start = place(slices, config)
    volume = np.zeros((config.slices, config.img, config.img), np.uint8)
    valid = np.zeros(config.slices, bool); valid[start:start + slices] = True
    ipp = np.full((config.slices, 3), np.nan)
    ipp[start:start + slices] = ipp0 + np.arange(slices)[:, None] * step * n
    t_mm = np.full(config.slices, np.nan)
    t_mm[start:start + slices] = [through_plane(x, n) for x in ipp[start:start + slices]]
    s = Sampled(volume=volume, valid=valid, t_mm=t_mm, ipp=ipp, iop=iop, normal=n,
                transform=t, spacing=spacing, native_spacing_mm=step, native_n=slices,
                stride=1, sop=[None] * config.slices)

    # a point on slice 7, a little off the centre of the image
    truth = patient_mm(ipp[start + 7], iop, spacing, 300.0, 250.0)
    tp, rm, cm = project(truth, s)
    check("the depth of a point on slice 7 is that slice's own depth",
          abs(tp - t_mm[start + 7]) < 1e-9)
    check("in-plane projection does not depend on which slice is the origin",
          np.allclose(project(truth, s)[1:],
                      project(truth, replace(s, ipp=np.roll(ipp, 0)))[1:]))

    heat = encode({"lat_centre": truth}, s, config)
    check("the heatmap has one channel per named landmark",
          heat.shape == (len(config.points), config.slices, config.grid, config.grid))
    check("and is zero on the padding", heat[0][~valid].max() == 0.0)
    got, conf = decode(heat, s, config)
    # Not exactly zero any more: the decode averages over a 3-sigma neighbourhood, and
    # truncating a Gaussian biases its centroid. 0.08 mm against a 12 mm tolerance is
    # the price of not returning the middle of the volume under a background floor.
    check("encode and decode return the same place",
          float(np.linalg.norm(got[0] - truth)) < 0.2,
          f"{np.linalg.norm(got[0] - truth):.4f} mm on a 4 mm sigma, all of it the "
          f"truncation of the averaging window")
    check("and report a confidence", 0 < conf[0] <= 1)

    # The case the first version of this test missed. A predicted map has a background
    # floor; the exact Gaussian encode() writes does not. With 196 608 cells against a
    # peak of about 180, a floor of 0.018 carries 99 % of the mass, and a soft-argmax
    # over the whole volume returns its centre whatever the model predicted — which is
    # what happened: 31 mm of error while the loss fell by a factor of six.
    floored = heat + 0.018
    got_f, conf_f = decode(floored, s, config)
    check("a background floor does not drag the answer to the middle",
          float(np.linalg.norm(got_f[0] - truth)) < 1.0,
          f"{np.linalg.norm(got_f[0] - truth):.2f} mm under a floor that holds 99 % "
          f"of the volume's mass")
    check("and the confidence falls when the mass is spread",
          conf_f[0] < conf[0] / 2,
          f"{conf_f[0]:.3f} against {conf[0]:.3f} on a clean map")

    flat = np.ones_like(heat)
    check("a flat heatmap still decodes rather than dividing by zero",
          np.all(np.isfinite(decode(flat, s, config)[0])))
    check("an empty one decodes to nothing rather than to the origin",
          np.all(np.isnan(decode(np.zeros_like(heat), s, config)[0])))

    # -- the config ---------------------------------------------------------- #
    check("the landmark config round-trips",
          LandmarkConfig.from_dict(config.to_dict()) == config)
    try:
        LandmarkConfig.from_dict({**config.to_dict(), "sigma_px": 3})
        check("refuses a field this pipeline does not define", False)
    except ValueError:
        check("refuses a field this pipeline does not define", True,
              "a stale experiment should fail loudly, not be read half-way")


def test_roi() -> None:
    """Cropping a branch's input around a landmark, and the three traps in it.

    The crop is small and every mistake in it is silent: a ROI taken a hundred slices
    away still looks like a knee, a box with anisotropic pixels still trains, and a
    window centred the wrong way round still fills its slots.
    """

    from rsna.dicom.geometry import normal_of, patient_mm, through_plane
    from rsna.roi import RoiSpec, bowtie_direction, choose_slices, crop_plane, extract

    print("\nroi")

    spec = RoiSpec()
    check("the default box gives isotropic pixels and whole patches",
          abs(spec.box_w_mm / spec.out_w - spec.box_h_mm / spec.out_h) < 1e-9
          and spec.out_w % 14 == 0 and spec.out_h % 14 == 0,
          f"{spec.box_w_mm}x{spec.box_h_mm} mm on {spec.out_w}x{spec.out_h} px "
          f"= {spec.mm_per_px:.4f} mm/px, patch 14 = {14 * spec.mm_per_px:.2f} mm")
    try:
        RoiSpec(box_h_mm=26.0)
        check("a box with anisotropic pixels is refused", False)
    except ValueError:
        check("a box with anisotropic pixels is refused", True,
              "it would stretch every study by the same wrong factor, and no "
              "augmentation undoes that")
    try:
        # 121 px with a box height to match, so only the patch rule can catch it
        RoiSpec(out_h=121, box_h_mm=48.0 * 121 / 224)
        check("...and so is an output that is not whole patches", False)
    except ValueError as exc:
        check("...and so is an output that is not whole patches", "patches" in str(exc),
              "121 px gives 8 patches of 14 and drops nine rows in silence")

    # -- the window ---------------------------------------------------------- #
    t = np.arange(20) * 3.0
    keep = choose_slices(t, t_point=30.0, toward_bowtie=1.0, spec=spec)
    taken = (t[keep] - 30.0)
    check("the depth window is asymmetric, short toward the bowtie",
          taken.max() <= spec.lateral_mm + 1e-6 and taken.min() >= -spec.medial_mm - 1e-6
          and abs(taken.min()) > taken.max(),
          f"{taken.min():+.0f} to {taken.max():+.0f} mm around the landmark")
    flipped = choose_slices(t, 30.0, toward_bowtie=-1.0, spec=spec)
    check("and turns over with the direction",
          abs((t[flipped] - 30.0).max()) > abs((t[flipped] - 30.0).min()))

    fine = np.arange(200) * 0.4
    thin = choose_slices(fine, t_point=40.0, toward_bowtie=1.0, spec=spec)
    check("too many slices are thinned, not averaged",
          len(thin) == spec.slots and len(set(thin.tolist())) == spec.slots
          and all(i in range(len(fine)) for i in thin),
          "every kept slice is still an acquired one")

    # -- the direction ------------------------------------------------------- #
    # A knee that runs out sooner on one side: that side is the lateral one. The rule
    # agrees with the DICOM laterality on 298 of 298 annotated studies, worst margin
    # 1.9x, which is what lets an asymmetric crop work without the tag.
    vol = np.zeros((20, 40, 40), np.uint8)
    vol[4:14] = 200                     # tissue only over slices 4..13
    ts = np.arange(20) * 3.0
    d, ratio = bowtie_direction(vol, ts, t_point=ts[11])
    check("the bowtie is the side where the knee ends sooner", d > 0 and ratio > 1.5,
          f"tissue ends 6 mm above the landmark and 21 mm below, ratio {ratio:.1f}x")
    d2, _ = bowtie_direction(vol, ts, t_point=ts[6])
    check("and it follows the landmark, not the stack", d2 < 0)

    # -- the crop ------------------------------------------------------------ #
    img = np.full((300, 300), 100, np.uint8)
    out = crop_plane(img, row=5.0, col=5.0, spacing=(0.4, 0.4), spec=spec)
    check("a box running off the image is padded, not clamped",
          out.shape == (spec.out_h, spec.out_w) and (out == 0).any() and (out > 0).any(),
          "clamping would keep the size and change the scale, which is worse")

    # -- the index trap ------------------------------------------------------ #
    # An annotation records the index of the stack the annotator scrolled. On a 3D
    # series that is a subsampled one, and using it on the acquired stack lands a
    # hundred slices away — on an image that still looks like a knee.
    iop = np.array([0.0, 1.0, 0.0, 0.0, 0.0, -1.0])
    ps = np.array([0.4, 0.4])
    n = normal_of(iop)
    geom = [{"sop": f"s{i}", "ipp": (np.array([50.0, -60.0, 30.0]) + i * 0.4 * n).tolist(),
             "iop": iop.tolist(), "ps": ps.tolist()} for i in range(200)]
    deep = np.zeros((200, 200, 200), np.uint8)
    deep[60:140] = 180                                  # the knee, slices 60..139
    point = patient_mm(geom[100]["ipp"], iop, ps, 100.0, 100.0)
    stack = extract(deep, geom, point, spec)
    tp = through_plane(point, n)
    check("a deep series is thinned by an integer stride", stack.stride > 1,
          f"0.40 mm spacing, stride {stack.stride} -> "
          f"{stack.native_spacing_mm * stack.stride:.2f} mm")
    check("and the slices kept are the ones near the landmark in millimetres",
          bool(np.all(np.abs(stack.t_mm[stack.valid] - tp)
                      <= max(spec.lateral_mm, spec.medial_mm) + 1e-6)),
          f"{np.abs(stack.t_mm[stack.valid] - tp).max():.1f} mm at worst, not the "
          f"hundred an index would have given")
    check("the crop comes out at the spec's size",
          stack.volume.shape == (spec.slots, spec.out_h, spec.out_w))
    check("and the filled slots carry pixels", stack.volume[stack.valid].max() > 0)

    sparse = [g for i, g in enumerate(geom) if i % 14 == 0][:12]
    vol2 = deep[::14][:12]
    few = extract(vol2, sparse, point, spec)
    check("a coarse series pads instead of inventing slices",
          int(few.valid.sum()) < spec.slots and not few.valid[-1],
          f"{int(few.valid.sum())} acquired slices in {spec.slots} slots at "
          f"{few.native_spacing_mm:.1f} mm")


def test_expert() -> None:
    """What the expert reads, and the padding that must never reach its encoder.

    Four studies in five carry one of the two sagittal PD series and not the other, so
    the other's slots are padding. Letting them through is not merely wasted work: a
    ResNet has twenty BatchNorm layers whose statistics are taken over the whole batch,
    and 44 % of a batch being black moved the features of the *real* images by 83 % in
    relative norm. It trains, it converges, and it is quietly handicapped throughout.
    """

    from rsna.expert import ExpertConfig, ExpertNet
    from rsna.roi import SPECS, RoiSpec

    print("\nexpert")

    config = ExpertConfig()
    spec = RoiSpec()
    check("the expert reads the ROI the spec cuts, for every target of its group",
          config.rois[0] == spec.name
          and set(config.targets) == {"Lateral Meniscus", "Lateral OA"},
          f"{len(config.targets)} targets on one encoder: they share a compartment, so "
          f"they share the features that describe it")
    check("a run written before the model took several targets or planes still reads",
          ExpertConfig.from_dict({"target": "Lateral Meniscus",
                                  "roi": "lateral_meniscus"}).targets
          == ("Lateral Meniscus",),
          "an experiment file that no longer reproduces its own run turns a published "
          "number into a rumour")

    class Trunk(nn.Module):
        """A stand-in with the one property that matters: it sees the whole batch."""

        num_features = 8

        def __init__(self):
            super().__init__()
            self.norm = nn.BatchNorm2d(3)
            self.fc = nn.Linear(3, 8)

        def forward(self, x):
            return self.fc(self.norm(x).mean(dim=(2, 3)))

    model = ExpertNet.__new__(ExpertNet)
    nn.Module.__init__(model)
    model.config = config
    model.trunk = Trunk()
    Attention = __import__("rsna.expert", fromlist=["Attention"]).Attention
    model.attend = Attention(8, heads=len(config.targets))
    model.drop = nn.Dropout(0.0)
    model.head = nn.Linear(8, len(config.targets))
    nn.init.zeros_(model.head.weight)
    nn.init.constant_(model.head.bias, config.prior_logit)
    model.train()

    b, s, slots = 4, 2, spec.slots
    imgs = torch.rand(b, s, slots, 32, 48)
    mask = torch.ones(b, s, slots, dtype=torch.bool)
    mask[:, 1] = False                    # the second series is absent everywhere
    mask[0, 0, 3:] = False                # and one study is short by two slices
    imgs = imgs * mask[..., None, None]

    noisy = imgs.clone()
    pad = ~mask[..., None, None].expand_as(imgs)
    noisy[pad] = torch.rand(int(pad.sum()))
    with torch.no_grad():
        a, c = model([(imgs, mask)]), model([(noisy, mask)])
    check("one logit per target", tuple(a.shape) == (b, len(config.targets)))
    check("padding never reaches the encoder",
          float((a - c).abs().max()) < 1e-6,
          "filling it with noise instead of zeros changes nothing, which is only true "
          "if it was never encoded")

    # One scorer per target, not one shared. The heads were always separate; the pooling
    # was not, and that is what two targets on one encoder actually compete over. A tear
    # is signal inside the fibrocartilage, which non-fat-suppressed PD shows;
    # osteoarthritis is bone, which the fat-suppressed and the T1 show. Sharing the
    # scorer cost the meniscus 0.0084 and left osteoarthritis at 0.7866.
    two = Attention(8, heads=2)
    with torch.no_grad():
        nn.init.constant_(two.score.bias, 0.0)
        two.score.weight.copy_(torch.stack([torch.ones(8), -torch.ones(8)]))
        feat = torch.rand(1, 3, 8)
        pooled = two(feat, torch.ones(1, 3, dtype=torch.bool))
    check("each target weighs the series its own way",
          tuple(pooled.shape) == (1, 2, 8)
          and float((pooled[0, 0] - pooled[0, 1]).abs().max()) > 1e-3,
          "opposite scorers must not land on the same pooled vector, or the per-target "
          "attention is decoration")
    with torch.no_grad():
        one_head = Attention(8, heads=1)
        flat = one_head(feat, torch.ones(1, 3, dtype=torch.bool))
        w = one_head.score(feat).squeeze(-1).softmax(dim=1)
    check("and with a single target it is still the shared mean",
          float((flat[:, 0] - torch.einsum("bs,bsd->bd", w, feat)).abs().max()) < 1e-6,
          "the single-target runs reproduce: same shape, same initialisation, same "
          "arithmetic as before the split")

    check("a study with no series at all falls back to the prior",
          float((model([(imgs, torch.zeros_like(mask))]).detach()
                 - config.prior_logit).abs().max()) < 1e-5,
          "a masked softmax over nothing would otherwise give NaN")

    # A window is live when its *centre* slot is real. Five slots give three windows,
    # centred on slots 1, 2 and 3.
    win, live = model.windows(imgs, mask)
    check("windows follow the slots, and the short study loses one",
          live.shape[2] == slots - config.group + 1
          and int(live[0, 0].sum()) == 2 and int(live[1, 0].sum()) == 3,
          "slices 0-2 of five: the window centred on slot 3 has no centre to sit on")
    check("and the absent series has no live window at all", int(live[:, 1].sum()) == 0)
    check("an edge window repeats the edge slice instead of reaching into padding",
          bool(torch.equal(win[0, 0, 1, 2], imgs[0, 0, 2])),
          "the window centred on slot 2 would take slot 3, which is padding; it takes "
          "slot 2 again, as a slice at the edge of an acquisition has no neighbour")
    check("a full stack keeps its real neighbours",
          bool(torch.equal(win[1, 0, 1, 2], imgs[1, 0, 3])),
          "clamping only bites at the edge of the acquired run, not everywhere")

    # Two ROI specs do not share a tensor: 48x27 mm and 48x30 mm at the same millimetres
    # per pixel are 126 and 140 rows. The trunk runs once per group and the features meet
    # at the attention, which is what lets each box be the size of what it frames.
    # The patellofemoral box moves along the OTHER axis. `box_offset_mm` shifts columns
    # and `box_rise_mm` shifts rows, and a crop that confused them would be displaced
    # sideways by the amount it should have risen — silently, since both produce a
    # plausible-looking crop of the same knee.
    # Every spec must name a landmark some annotation pass actually collected. Getting
    # this wrong does not crash: the build filters the table to zero rows and writes an
    # empty cache, reports `nan %` coverage, and the training stage reads it.
    from rsna.landmark.series import LANDMARKS as _POINTS
    unknown = {s.name: s.landmark for s in SPECS.values() if s.landmark not in _POINTS}
    check("every ROI spec hangs off a landmark this project collects",
          not unknown, f"{unknown or 'all of them'}")
    check("and the axial specs hang off the axial point",
          all(s.landmark == "pf_centre" for s in SPECS.values() if s.plane == "Axial"),
          "a spec left on the default asked for lat_centre and silently cut nothing")

    pf = SPECS["pf_oa"]
    check("the patellofemoral box rises instead of shifting sideways",
          pf.plane == "Axial" and pf.box_rise_mm > 0 and pf.box_offset_mm == 0,
          f"{pf.box_w_mm:.0f}x{pf.box_h_mm:.0f} mm raised {pf.box_rise_mm:.0f} mm "
          f"toward the front")
    check("and its pixels are isotropic and whole patches",
          abs(pf.box_w_mm / pf.out_w - pf.box_h_mm / pf.out_h) < 1e-9
          and pf.out_w % pf.patch == 0 and pf.out_h % pf.patch == 0,
          f"{pf.out_w}x{pf.out_h} px at {pf.mm_per_px:.3f} mm/px, against a 0.31 mm/px "
          f"median acquisition — finer would resample past what was acquired")
    check("its depth is symmetric, having no bowtie to be short toward",
          pf.symmetric_depth,
          "the landmark is mid-patella by definition and the joint runs as far above "
          "it as below")

    # The ramp has to be bigger than the box, or both crops run off the edge and the
    # zero padding swamps the very difference being measured.
    flat = np.tile(np.arange(200, dtype=np.uint8)[:, None], (1, 200))
    spacing = (1.0, 1.0)
    from rsna.roi.extract import crop_plane
    tiny = pf.replace(out_w=14, out_h=12, patch=1)
    low = crop_plane(flat, 100.0, 100.0, spacing, tiny, to_lateral=1.0, to_front=-1.0)
    high = crop_plane(flat, 100.0, 100.0, spacing, tiny, to_lateral=1.0, to_front=1.0)
    check("and the rise follows the series' own row direction, not a fixed sign",
          float(low.mean()) < float(high.mean()),
          f"on a ramp brightening with the row index, a series whose front is -row "
          f"gives {low.mean():.0f} and one whose front is +row gives {high.mean():.0f}")

    cor = SPECS["lateral_meniscus_coronal"]
    wide = ExpertConfig(rois=("lateral_meniscus", "lateral_meniscus_coronal"))
    big = ExpertNet(wide, pretrained=False)
    sag_spec = SPECS["lateral_meniscus"]
    pair = [(torch.rand(2, 2, sag_spec.slots, sag_spec.out_h, sag_spec.out_w),
             torch.ones(2, 2, sag_spec.slots, dtype=torch.bool)),
            (torch.rand(2, 2, cor.slots, cor.out_h, cor.out_w),
             torch.ones(2, 2, cor.slots, dtype=torch.bool))]
    with torch.no_grad():
        both = big(pair)
    check("two ROI groups of different shapes go through one encoder",
          tuple(both.shape) == (2, len(wide.targets)),
          f"{sag_spec.out_h}x{sag_spec.out_w} and {cor.out_h}x{cor.out_w} rows, one trunk")
    pair[1] = (pair[1][0], torch.zeros(2, 2, cor.slots, dtype=torch.bool))
    with torch.no_grad():
        one = big(pair)
    check("and a study missing one of them still answers",
          bool(torch.isfinite(one).all()),
          "88.3 % of studies have a coronal PD against 99.8 % sagittal, so this is the "
          "common case, not the edge one")


def main() -> int:
    for test in (test_config, test_headers, test_folds, test_pixels, test_cache,
                 test_laterality, test_model, test_stems, test_encoders, test_encoder_unchanged,
                 test_experiments,
                 test_augment, test_loop, test_windows,
                 test_submission, test_figures, test_eval, test_annotation_side, test_series_choice, test_annotation_side_rule,
                 test_excluded_list, test_roi, test_expert,
                 test_landmark_geometry):
        test()
    print(f"\n{len(PASSED)} passed, {len(FAILED)} failed")
    if FAILED:
        print("failed: " + ", ".join(FAILED))
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
