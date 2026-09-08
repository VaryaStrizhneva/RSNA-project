"""Runs every part of the package that does not need a DICOM on disk.

Not a substitute for real data — `dicom.headers`, `dicom.ordering` and
`dicom.laterality` are exercised here only on synthetic frames. What this does cover
is everything that can be wrong without any file being involved: shapes, the presence
mask actually changing an output, gradients reaching the head, the fingerprint
detecting a perturbation, and the loop learning a signal that is there.

    python -m tests.smoke
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import torch

from rsna.config import Config, PixelRules, TARGETS
from rsna.dicom import (
    CacheMismatch, annotate, build_cache, hdr_vec, load_cache, order_slices,
    plan_cache, read_slot, sample_indices,
)
from rsna.model import build_model, check_fingerprint, fingerprint, find_encoder
from rsna.model.fingerprint import WeightsError
from rsna.train import assign_folds, augment, build_targets, fit, predict, take_window
from stub_backbone import StubBackbone
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


def main() -> int:
    for test in (test_config, test_headers, test_folds, test_pixels, test_cache,
                 test_model, test_augment, test_loop, test_windows,
                 test_submission, test_figures):
        test()
    print(f"\n{len(PASSED)} passed, {len(FAILED)} failed")
    if FAILED:
        print("failed: " + ", ".join(FAILED))
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
