"""Fit one model end to end, and write a weights package.

This is the equivalent of the public notebook's `main()`, except that it trains and
stops: the scored run does inference only, from the package this writes. Everything it
calls lives in `src/rsna`, so a Kaggle notebook and this script cannot drift apart.

    # plumbing check: a handful of studies, tiny images, two epochs
    python -m scripts.train --experiment depth_compress --split test_series --limit 3 \
        --img 112 --slices 6 --epochs 2 --fake-labels --out out/package

    # the real thing, once the DICOMs are local
    python -m scripts.train --experiment window_baseline --out out/package-baseline

The stages are printed as they complete, because when this breaks it will break at one
of them and the log should say which.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.rsna.config import TARGETS, Config
from src.rsna.dicom import annotate, build_cache, laterality_of, pick_slots, walk
from src.rsna.model import build_model
from src.rsna.model.fingerprint import fingerprint
from src.rsna.model import Member, write_package
from src.rsna.train import assign_folds, build_targets, fit

import experiments

T0 = time.time()


def log(message: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {message}", flush=True)


def plane_map_for(root: Path, split: str) -> dict:
    """Series -> anatomical plane, from the competition CSV."""

    csv = root / f"{split.replace('_series', '')}_series.csv"
    table = pd.read_csv(csv)
    return dict(zip(table["SeriesInstanceUID"], table["Anatomical_Plane"]))


def fake_targets(studies: list[str], config: Config, seed: int = 0):
    """Targets with a signal that is actually there, for a plumbing run.

    A run on three studies cannot measure anything, but it can prove the chain carries
    gradients: the first slot is brightened where the first target is 1, so a model that
    fits will drive the loss down and one that is wired wrong will not.
    """

    rng = np.random.default_rng(seed)
    y = np.zeros((len(studies), len(TARGETS)), np.float32)
    y[:, 0] = (np.arange(len(studies)) % 2).astype(np.float32)
    y[:, 1] = rng.random(len(studies)) > 0.5
    return y, np.ones_like(y)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default="window_baseline",
                        help=f"Named config to fit. One of: "
                             f"{', '.join(experiments.available())}")
    parser.add_argument("--data-root", default="data/raw", type=Path)
    parser.add_argument("--split", default=None)
    parser.add_argument("--labels", default=None)
    parser.add_argument("--encoder", default=None,
                        help="Local encoder directory, or a mounted one on Kaggle.")
    parser.add_argument("--out", default="out/package", type=Path)
    parser.add_argument("--cache", default=None, type=Path,
                        help="Where to keep the decoded cache. Omit to hold it in memory.")
    parser.add_argument("--limit", type=int, default=0, help="Use only the first N studies.")
    parser.add_argument("--fake-labels", action="store_true",
                        help="Plant a learnable signal instead of reading a label table.")
    parser.add_argument("--img", type=int, default=None)
    parser.add_argument("--slices", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--fold", type=int, default=None, help="Which fold is held out.")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    # The experiment defines the run; the flags only override it, for a quick check.
    experiment = experiments.load(args.experiment)
    overrides = {k: v for k, v in (("img", args.img), ("slices", args.slices),
                                   ("epochs", args.epochs),
                                   ("batch_studies", args.batch)) if v is not None}
    config = experiment.config.replace(**overrides)

    # A flag overrides the experiment, so a quick check can shrink a real run without
    # editing its definition.
    split = args.split or experiment.split
    labels = args.labels or experiment.labels
    encoder = args.encoder or experiment.encoder
    fold = args.fold if args.fold is not None else experiment.fold
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    log(f"experiment: {args.experiment}  (stem={config.stem}, "
        f"{config.window_size} channels to the encoder, "
        f"{len(config.windows())} pass(es) per slot at inference)")
    log(f"config: {config.img}px, {config.slices} slices, group {config.group}, "
        f"{config.n_slot} slots, {config.epochs} epochs, batch {config.batch_studies}")
    log(f"run: split={split}, fold={fold}, encoder={encoder}")
    log(f"     labels={labels}")
    log(f"device: {device}")

    # -- headers -------------------------------------------------------------- #
    headers = annotate(walk(args.data_root, split))
    if headers.empty or "SeriesDescription" not in headers:
        raise SystemExit(f"no series under {args.data_root / split}")
    plane_map = plane_map_for(args.data_root, split)
    headers["plane"] = headers["SeriesInstanceUID"].map(plane_map)
    log(f"{len(headers)} series across {headers['StudyInstanceUID'].nunique()} studies")

    if args.limit:
        keep = sorted(headers["StudyInstanceUID"].unique())[: args.limit]
        headers = headers[headers["StudyInstanceUID"].isin(keep)]
        log(f"limited to {len(keep)} studies")

    # -- geometry ------------------------------------------------------------- #
    sides, stats = laterality_of(headers, config)
    log(f"laterality: {stats['from_tag']} tagged, {stats['from_geometry']} from geometry, "
        f"{stats['unresolved']} unresolved, {stats['disagree']} disagree")

    slots = pick_slots(headers, plane_map, config)
    filled = sum(len(s) for s in slots.values())
    log(f"slots: {filled}/{len(slots) * config.n_slot} filled")

    # -- pixels --------------------------------------------------------------- #
    failures: list = []
    studies, cache, mask = build_cache(slots, plane_map, sides, config,
                                       cache_slices=config.slices, path=args.cache,
                                       log=log, decode_failures=failures)
    if failures:
        log(f"{len(failures)} series had a slice that would not decode")

    # -- targets -------------------------------------------------------------- #
    if args.fake_labels:
        y, w = fake_targets(studies, config)
        # Plant the signal the fake targets claim: without it there is nothing to learn
        # and a working pipeline is indistinguishable from a broken one.
        bright = y[:, 0] > 0.5
        cache[bright, 0] = np.clip(cache[bright, 0].astype(int) + 90, 0, 255).astype(np.uint8)
        folds = pd.Series((np.arange(len(studies)) % max(config.n_folds, 2)),
                          index=studies, name="fold")
        log("targets: planted signal (plumbing run, the score means nothing)")
    else:
        train_csv = pd.read_csv(args.data_root / "train.csv", dtype={"StudyInstanceUID": str})
        derived = pd.read_csv(labels, dtype={"StudyInstanceUID": str}).set_index(
            "StudyInstanceUID")
        y, w = build_targets(studies, train_csv, derived, config)
        folds = assign_folds(train_csv, config).reindex(studies)
        supervised = int((w.sum(1) > 0).sum())
        log(f"targets: {supervised}/{len(studies)} studies supervised from {labels}")

    holdout = np.array([i for i, s in enumerate(studies) if folds.get(s, -1) == fold])
    train_idx = np.array([i for i in range(len(studies)) if i not in set(holdout.tolist())])
    if len(holdout) == 0 or len(train_idx) < config.batch_studies:
        cut = max(1, len(studies) // 5)
        holdout, train_idx = np.arange(cut), np.arange(cut, len(studies))
        log("fold split too small; fell back to a positional split")
    log(f"train {len(train_idx)} / holdout {len(holdout)} studies")

    # -- fit ------------------------------------------------------------------- #
    model = build_model(config, source=encoder).to(device)
    log(f"model: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters, "
        f"{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.1f}M trainable")

    result = fit(model, cache, mask, y, w, train_idx, holdout, config, device,
                 on_epoch=lambda r: log(f"  epoch {r.epoch + 1}/{config.epochs}  "
                                        f"loss {r.loss:.4f}  holdout {r.holdout_auc:.4f}"))
    if np.isfinite(result.best_holdout_auc):
        log(f"best epoch {result.best_epoch + 1}, holdout AUC {result.best_holdout_auc:.4f}")
    else:
        log(f"no epoch was selectable (holdout AUC is NaN — too few studies, or a "
            f"target with one class); kept the last epoch, {result.best_epoch + 1}")

    # -- package --------------------------------------------------------------- #
    model.load_state_dict(result.state_dict)
    member = Member(name=f"{args.experiment}_fold{fold}_{config.img}px",
                    config=config,
                    state_dict=result.state_dict,
                    fingerprint=fingerprint(model, config, device),
                    fold=fold,
                    holdout_auc=result.best_holdout_auc,
                    epoch=result.best_epoch)
    path = write_package(args.out, [member],
                         note=f"{args.experiment}; {split}, {len(studies)} studies, "
                              f"{'planted signal' if args.fake_labels else labels}")
    log(f"package written to {path}")
    print(json.dumps(json.loads((path / 'manifest.json').read_text())["members"][0],
                     indent=1))


if __name__ == "__main__":
    main()
