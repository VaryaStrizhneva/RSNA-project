"""Write a submission whose expert columns come from experts.

    python -m scripts.predict_experts \\
        --package out/ref/pkg-f0 \\
        --expert out/expert_acl --expert out/expert_medial_oa \\
        --landmark out/landmark_both --landmark out/landmark_pf \\
        --dicom-root /data/mgr/rsna-knee/extracted --split test_series

Argument parsing and logging only: the chain itself is `rsna.infer.run_expert_submission`,
which the scored Kaggle notebook calls too. Keeping the logic there is what stops a
notebook and a command line from drifting apart.

Which expert owns which column, and which landmark each region hangs off, are read off
the weights rather than passed here — so a run is always applied the way it was fitted,
and a landmark model nothing depends on is skipped without being mentioned twice.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rsna.infer import run_expert_submission                         # noqa: E402

T0 = time.time()


def log(message: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {message}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--package", required=True, type=Path,
                    help="the wide model, which fills every column an expert does not")
    ap.add_argument("--expert", action="append", required=True, type=Path,
                    help="an expert run directory; repeatable")
    ap.add_argument("--landmark", action="append", required=True, type=Path,
                    help="a landmark run directory; repeatable, unused ones are skipped")
    ap.add_argument("--data-root", default=Path("data/raw"), type=Path)
    ap.add_argument("--dicom-root", default=None, type=Path,
                    help="where the pixels are, if not under --data-root")
    ap.add_argument("--split", default="test_series")
    ap.add_argument("--encoder", default="models/dinov2-small")
    ap.add_argument("--out", default=Path("submission.csv"), type=Path)
    ap.add_argument("--scratch", default=None, type=Path,
                    help="where the regions are cut; a temp directory by default. "
                         "Files already there are reused and never deleted.")
    ap.add_argument("--blend", type=float, default=None,
                    help="weight the expert carries against the wide model, both as "
                         "ranks; omit to hand it the column outright, 0.5 to average")
    ap.add_argument("--keep-cache", action="store_true",
                    help="leave the regions on disk — 10 GB for 1300 studies")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    out = run_expert_submission(
        expert_runs=args.expert, landmark_runs=args.landmark, package=args.package,
        data_root=args.dicom_root or args.data_root, split=args.split,
        encoder=args.encoder, out=args.out, scratch=args.scratch,
        device=args.device, workers=args.workers, keep_cache=args.keep_cache,
        blend=args.blend, log=log)
    log(f"done -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
