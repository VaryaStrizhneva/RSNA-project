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
from src.rsna.data.labels import (
    LABEL_SOURCES,
    load_confidence_table,
    load_label_table,
    load_verdict_table,
)
from src.rsna.data.metadata import TARGET_COLUMNS, clean_report_text


def build_audit_rows(gold: pd.DataFrame, data_root: Path, sources: list[tuple]) -> pd.DataFrame:
    """One row per source, study and target for manual disagreement review."""

    train = pd.read_csv(data_root / "train.csv", dtype={"StudyInstanceUID": str})
    reports = train.set_index("StudyInstanceUID")["Report"].map(clean_report_text)
    rows = []

    for name, source, contaminated in sources:
        table = load_label_table(source)
        aligned = table.reindex(gold.index)
        ranks = table[TARGET_COLUMNS].rank(pct=True).reindex(gold.index)
        confidence = load_confidence_table(source)
        if confidence is not None:
            confidence = confidence.reindex(gold.index)
        verdict = load_verdict_table(source)
        if verdict is not None:
            verdict = verdict.reindex(gold.index)
        target_auc = per_target_auc(gold, table)
        source_macro = macro_auc(gold, table)

        for study in gold.index:
            for target in TARGET_COLUMNS:
                expert = int(gold.at[study, target])
                score = aligned.at[study, target]
                rank_score = ranks.at[study, target]
                if pd.isna(rank_score):
                    disagreement = pd.NA
                elif expert == 1:
                    disagreement = 1.0 - float(rank_score)
                else:
                    disagreement = float(rank_score)

                rows.append({
                    "source": name,
                    "contaminated": contaminated,
                    "StudyInstanceUID": study,
                    "target": target,
                    "expert": expert,
                    "score": score,
                    "score_rank": rank_score,
                    "rank_disagreement": disagreement,
                    "error_type": "missed_positive" if expert == 1 else "possible_false_positive",
                    "confidence": (
                        confidence.at[study, target]
                        if confidence is not None and study in confidence.index else pd.NA
                    ),
                    "verdict": (
                        verdict.at[study, target]
                        if verdict is not None and study in verdict.index else pd.NA
                    ),
                    "target_auc_on_58": target_auc[target],
                    "source_macro_auc_on_58": source_macro,
                    "report": reports.get(study, ""),
                })

    return (
        pd.DataFrame(rows)
        .sort_values(["rank_disagreement", "source", "StudyInstanceUID", "target"],
                     ascending=[False, True, True, True], na_position="last")
        .reset_index(drop=True)
    )


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
    parser.add_argument("--audit-out", type=Path, default=None,
                        help="Write a CSV of expert-vs-source disagreements for inspection.")
    args = parser.parse_args()

    gold = load_gold(args.data_root)
    print(f"Ground truth: {len(gold)} expert-labelled studies")
    print("Positives per target: " + ", ".join(f"{t}={n}" for t, n in gold.sum().items()))
    print()

    sources = [(source.name, source, source.contaminated) for source in LABEL_SOURCES]
    for path in args.table:
        sources.append((Path(path).stem, Path(path), False))

    tables: dict[str, tuple[pd.DataFrame, bool]] = {
        name: (load_label_table(source), contaminated)
        for name, source, contaminated in sources
    }

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

    if args.audit_out is not None:
        audit = build_audit_rows(gold, Path(args.data_root), sources)
        args.audit_out.parent.mkdir(parents=True, exist_ok=True)
        audit.to_csv(args.audit_out, index=False)
        print(f"\nAudit CSV written to {args.audit_out} ({len(audit)} rows)")


if __name__ == "__main__":
    main()
