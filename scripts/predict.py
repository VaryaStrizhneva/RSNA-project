"""Read a weights package over a decoded cache and write a submission.

Argument parsing and logging only: the chain itself is `rsna.infer.run_submission`,
which the scored Kaggle notebook calls too. Keeping the logic there rather than here is
what stops a notebook and a command line from drifting apart — and the config comes
from the package, not from this file, so a member is always read the way it was fitted.

    python -m scripts.predict --package out/submit-window --split test_series
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from src.rsna.infer import run_submission

T0 = time.time()


def log(message: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {message}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--data-root", default="data/raw", type=Path)
    parser.add_argument("--split", default="test_series")
    parser.add_argument("--encoder", default="models/dinov2-small")
    parser.add_argument("--out", default="submission.csv", type=Path)
    parser.add_argument("--cache", default=None, type=Path)
    parser.add_argument("--max-windows", type=int, default=None,
                        help="Read each member over at most this many windows.")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    run_submission(args.package, args.data_root, args.split, encoder=args.encoder,
                   out=args.out, cache=args.cache, max_windows=args.max_windows,
                   device=device, log=log)


if __name__ == "__main__":
    main()
