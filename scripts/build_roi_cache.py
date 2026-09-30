"""Cut every study's region of interest once, from the landmarks a model predicted.

    python -m scripts.build_roi_cache --landmarks out/landmarks/train_series.csv \
        --spec lateral_meniscus --out /data/mgr/rsna-knee/roi-cache

Two orders of magnitude smaller than the classifier's cache — 1.2 GB against 36 — which
is what cropping buys and the reason a ROI branch can hold its whole training set in
memory.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rsna.config import Config                                  # noqa: E402
from rsna.dicom.headers import annotate, walk                   # noqa: E402
from rsna.roi import SPECS, cache                               # noqa: E402


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--landmarks", default="out/landmarks/train_series.csv", type=Path)
    ap.add_argument("--spec", default="lateral_meniscus")
    ap.add_argument("--data-root", default="data/raw", type=Path)
    ap.add_argument("--dicom-root", default=None, type=Path)
    ap.add_argument("--split", default="train_series")
    ap.add_argument("--out", default="/data/mgr/rsna-knee/roi-cache", type=Path)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    spec = SPECS[args.spec]
    log(f"{spec.name}: {spec.box_w_mm:.0f}x{spec.box_h_mm:.0f} mm on "
        f"{spec.out_w}x{spec.out_h} px, {spec.lateral_mm:g}/{spec.medial_mm:g} mm over "
        f"{spec.slots} slots")

    series = annotate(walk(args.dicom_root or args.data_root, args.split))
    planes = pd.read_csv(args.data_root / f"{args.split.replace('_series', '')}_series.csv")
    series["plane"] = series["SeriesInstanceUID"].map(
        dict(zip(planes["SeriesInstanceUID"], planes["Anatomical_Plane"])))
    log(f"{len(series)} series over {series.StudyInstanceUID.nunique()} studies")

    cache.build(args.landmarks, series, args.out, spec, Config(),
                workers=args.workers, log=log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
