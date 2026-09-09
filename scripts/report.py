"""Turn a finished sweep into one HTML page and ten lines of text.

Reads only what training wrote beside the weights - `history.json` and `holdout.csv` -
so it costs a second, needs no GPU, and says the same thing every time it runs.

    # every fold under a sweep directory
    python -m scripts.report out/sweep-window-20

    # one fold, somewhere else
    python -m scripts.report out/pkg-window-f0 --out out/one-fold.html
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.rsna.eval import build, read_sweep, summary_text, write


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path,
                        help="A sweep directory, or one package directory.")
    parser.add_argument("--out", type=Path, default=None,
                        help="Where to write the page. Defaults to <run>/report.html")
    args = parser.parse_args()

    records = read_sweep(args.run)
    print(f"{len(records)} fold(s) under {args.run}: "
          f"{', '.join(str(r.fold) for r in records)}")

    report = build(records)
    path = write(report, args.out or (args.run / "report.html"))
    print()
    print(summary_text(report))
    print()
    print(f"written to {path}")


if __name__ == "__main__":
    main()
