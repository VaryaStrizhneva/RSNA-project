"""Run the trained landmark models over a whole split and write what they found.

    python -m scripts.predict_landmark --run out/landmark_reference --split train_series

Argument parsing and logging only: the chain itself is `rsna.infer.predict_landmarks`,
which the scored Kaggle notebook calls too. Keeping the logic there rather than here is
what stops a notebook and a command line from drifting apart — and the config comes from
the weights, not from this file, so a run is always read the way it was fitted.

The output is one row per study per point — the landmark in patient millimetres, the
confidence, and the agreement between folds. That table is the only thing the ROI side
needs, which is what keeps the two models independent of each other.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rsna.dicom.headers import annotate, walk                        # noqa: E402
from rsna.infer import predict_landmarks                             # noqa: E402


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="out/landmark_reference", type=Path)
    ap.add_argument("--data-root", default="data/raw", type=Path)
    ap.add_argument("--split", default="train_series")
    ap.add_argument("--dicom-root", default=None, type=Path,
                    help="where the pixels are, if not under --data-root")
    ap.add_argument("--out", default=None, type=Path)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    root = args.dicom_root or args.data_root
    series = annotate(walk(Path(root), args.split))
    # The plane is the competition's own label, not something recovered from the
    # header — `scripts/train.py` reads it the same way, so both paths agree.
    planes = pd.read_csv(args.data_root / f"{args.split.replace('_series', '')}_series.csv")
    series["plane"] = series["SeriesInstanceUID"].map(
        dict(zip(planes["SeriesInstanceUID"], planes["Anatomical_Plane"])))
    log(f"{len(series)} series over {series.StudyInstanceUID.nunique()} studies, "
        f"{int(series.plane.isna().sum())} with no plane")

    table, failed = predict_landmarks(args.run, series, device=args.device,
                                      batch=args.batch, workers=args.workers,
                                      limit=args.limit, log=log)

    out = args.out or Path("out/landmarks") / f"{args.split}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    log(f"\n{len(table)} rows, {len(failed)} studies failed -> {out}")
    for point, sel in table.groupby("point"):
        log(f"  {point}: confidence median {sel.confidence.median():.3f}, "
            f"fold spread median {sel.spread_mm.median():.1f} mm, "
            f"p90 {sel.spread_mm.quantile(0.9):.1f} mm")
    for study, why in failed[:10]:
        log(f"  FAILED {study[-11:]}: {why}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
