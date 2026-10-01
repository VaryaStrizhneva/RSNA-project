"""Fit the landmark model and write what it found.

    python -m scripts.train_landmark --experiment landmark_reference
    python -m scripts.train_landmark --experiment landmark_reference --fold 0 --epochs 5

Reads the sampled cache rather than the DICOM — 304 studies at 48x256x256 is under a
gigabyte, so a step costs no decode. Writes one directory per run holding the weights of
every fold, the per-epoch history, and the out-of-fold error of every study in
millimetres, which is the number the model is judged on.

Everything here is wiring. The parts worth testing live in `rsna.landmark`.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsna.landmark import LandmarkConfig, cache                      # noqa: E402
from rsna.landmark.loop import build_targets, fit, folds             # noqa: E402
from rsna.landmark.network import LandmarkNet                        # noqa: E402

EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experiment", default="landmark_reference")
    ap.add_argument("--cache", default="/data/mgr/rsna-knee/landmark-cache", type=Path)
    ap.add_argument("--out", default=None, type=Path)
    ap.add_argument("--fold", type=int, default=None,
                    help="fit only this fold; all of them by default")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--weight", type=float, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    spec = json.loads((EXPERIMENTS / f"{args.experiment}.json").read_text())
    config = LandmarkConfig.from_dict(spec.get("config", {}))
    run = dict(spec.get("run", {}))
    for key in ("epochs", "batch", "lr", "weight"):
        if getattr(args, key) is not None:
            run[key] = getattr(args, key)

    out = args.out or Path("out") / args.experiment
    out.mkdir(parents=True, exist_ok=True)
    log(f"{args.experiment}: {json.dumps(run)}")

    volumes, records = cache.load(args.cache, config)
    log(f"{len(records)} studies, {volumes.nbytes / 1e9:.2f} GB")
    targets = build_targets(records, volumes, config)

    splits = folds(len(records), run.get("folds", 5), run.get("seed", 0))
    history, errors, studies = {}, [], []
    for k, val in enumerate(splits):
        if args.fold is not None and k != args.fold:
            continue
        train = np.setdiff1d(np.arange(len(records)), val)
        log(f"fold {k}: {len(train)} train, {len(val)} held out")
        model = LandmarkNet(config)
        result = fit(model, volumes, records, targets, train, val, config,
                     device=args.device, epochs=run.get("epochs", 60),
                     batch=run.get("batch", 8), lr=run.get("lr", 3e-4),
                     weight=run.get("weight", 200.0), seed=run.get("seed", 0), log=log)
        torch.save({"config": config.to_dict(), "fold": k, "state": result.state},
                   out / f"fold{k}.pt")
        history[str(k)] = [vars(e) for e in result.history]
        errors.append(result.errors)
        studies += result.studies
        b = result.best
        log(f"fold {k} best: epoch {b.epoch}, median {b.median_mm:.1f} mm, "
            f"p90 {b.p90_mm:.1f} mm, within 12 mm {100 * b.within_12:.0f} %")

    e = np.concatenate(errors)
    log(f"\nOUT OF FOLD over {len(e)} studies")
    log(f"  median {np.nanmedian(e):.1f} mm   p90 {np.nanpercentile(e, 90):.1f} mm   "
        f"within 12 mm {100 * np.nanmean(e < 12):.0f} %")
    log("  the constant predictor, with the side known perfectly: p90 21.2 mm")

    (out / "history.json").write_text(json.dumps(
        {"experiment": args.experiment, "config": config.to_dict(), "run": run,
         "history": history,
         "errors": {s: float(x) for s, x in zip(studies, e)}}, indent=1))
    log(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
