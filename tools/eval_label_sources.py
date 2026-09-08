"""Score report-derived label tables against the expert labels. Reads only.

Prints, for every registered source and any extra table passed on the command
line, the macro AUC and the twelve per-target AUCs on the 58 expert-labelled
studies. It writes nothing — it exists to compare sources, not to produce one.

    python -m tools.eval_label_sources
    python -m tools.eval_label_sources --table path/to/another_table.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.rsna.data.label_eval import bootstrap_macro, load_gold, macro_auc, per_target_auc
from src.rsna.data.labels import LABEL_SOURCES, load_label_table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/raw")
    parser.add_argument(
        "--table",
        action="append",
        default=[],
        metavar="PATH",
        help="Extra label table to score. Repeatable.",
    )
    parser.add_argument("--bootstrap", type=int, default=1000,
                        help="Resamples for the confidence interval; 0 to skip.")
    args = parser.parse_args()

    gold = load_gold(args.data_root)
    print(f"Ground truth: {len(gold)} expert-labelled studies")
    print("Positives per target: " + ", ".join(f"{t}={n}" for t, n in gold.sum().items()))
    print()

    tables: dict[str, tuple[pd.DataFrame, bool]] = {
        source.name: (load_label_table(source), source.contaminated)
        for source in LABEL_SOURCES
    }
    for path in args.table:
        tables[Path(path).stem] = (load_label_table(path), False)

    rows = []
    for name, (table, contaminated) in tables.items():
        covered = len(table.index.intersection(gold.index))
        row = {
            "source": name,
            "macro_auc": macro_auc(gold, table),
            "covers": f"{covered}/{len(gold)}",
        }
        if args.bootstrap:
            lo, hi = bootstrap_macro(gold, table, n=args.bootstrap)
            row["ci95"] = f"{lo:.3f}-{hi:.3f}"
        row["note"] = "CONTAMINATED: reproduces the official labels" if contaminated else ""
        rows.append(row)

    print("Macro AUC per source")
    print(pd.DataFrame(rows).to_string(index=False, float_format="%.4f"))

    print("\nPer-target AUC")
    matrix = pd.DataFrame({n: per_target_auc(gold, t) for n, (t, _) in tables.items()})
    print(matrix.to_string(float_format="%.3f"))

    print(
        "\nA 58-study macro AUC carries an interval about seven points wide."
        "\nDifferences smaller than that are not evidence of anything."
    )


if __name__ == "__main__":
    main()
