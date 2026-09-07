"""Try combinations of report-derived label tables, and write one out.

Blending is a choice with two parts, and this script separates them.

*Which sources.* Averaging near-identical sources adds nothing, so the candidates
below range from single sources up to all four, and the Spearman matrix printed
first shows which of them are actually distinct readings.

*Whether to weight per target.* Tempting, since no source wins everywhere. The
`--selection` check answers it honestly: it compares picking the best source per
target in sample against picking it out of sample, and the gap is what the pick
is really worth.

    python -m scripts.blend_labels
    python -m scripts.blend_labels --selection
    python -m scripts.blend_labels --write default --out data/processed/report_labels_blend.csv
"""

from __future__ import annotations

import argparse
import itertools

import numpy as np
import pandas as pd
from scipy import stats

from src.data.label_eval import bootstrap_macro, load_gold, macro_auc, per_target_auc
from src.data.labels import (
    DISTINCT_SOURCES,
    LABEL_SOURCES,
    blend,
    load_label_table,
    to_ranks,
)
from src.data.metadata import TARGET_COLUMNS

#: Candidate blends. Keys are names for `--write`; values are source names.
#: Contaminated sources are deliberately absent: they cannot be scored here.
CANDIDATES: dict[str, list[str]] = {
    "pilkwang_only": ["pilkwang"],
    "steven_only": ["steven_v4_blend"],
    "default": DISTINCT_SOURCES,
    "three": ["pilkwang", "steven_v4_blend", "steven_v2"],
    "all_four": ["pilkwang", "steven_v4_blend", "steven_v2", "steven_full"],
    "steven_family": ["steven_v4_blend", "steven_v2", "steven_full"],
}


def spearman_matrix(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Mean rank correlation between sources, averaged over the twelve targets.

    Computed on all 4,407 studies rather than the 58, because how alike two
    sources are is a property of the sources, not of the annotated subset.
    """

    names = list(tables)
    out = pd.DataFrame(index=names, columns=names, dtype=float)
    ranked = {n: t.rank(pct=True) for n, t in tables.items()}

    for left, right in itertools.product(names, names):
        a, b = ranked[left], ranked[right]
        shared = a.index.intersection(b.index)
        out.loc[left, right] = np.mean(
            [stats.spearmanr(a.loc[shared, t], b.loc[shared, t],
                             nan_policy="omit").statistic for t in TARGET_COLUMNS]
        )
    return out


def selection_is_worth_it(gold: pd.DataFrame, ranked: dict, n_splits: int = 300,
                          seed: int = 0) -> tuple[float, float]:
    """Per-target source picking, scored in sample and out of sample.

    Out of sample, half the studies choose the source per target and the other
    half score that choice, repeated and averaged. The difference between the two
    numbers is selection optimism, and it does not transfer to the test set.
    """

    rng = np.random.default_rng(seed)
    matrix = pd.DataFrame({n: per_target_auc(gold, t) for n, t in ranked.items()})
    in_sample = float(matrix.max(axis=1).mean())

    positions = np.arange(len(gold))
    scores = []
    for _ in range(n_splits):
        shuffled = rng.permutation(positions)
        fit, test = shuffled[: len(positions) // 2], shuffled[len(positions) // 2 :]
        held = []
        for target in TARGET_COLUMNS:
            best, best_auc = None, -np.inf
            for name, frame in ranked.items():
                aligned = frame[target].reindex(gold.index)
                value = _auc(gold[target].iloc[fit], aligned.iloc[fit])
                if np.isfinite(value) and value > best_auc:
                    best, best_auc = name, value
            if best is None:
                continue
            value = _auc(gold[target].iloc[test],
                         ranked[best][target].reindex(gold.index).iloc[test])
            if np.isfinite(value):
                held.append(value)
        if held:
            scores.append(np.mean(held))

    return in_sample, float(np.mean(scores))


def _auc(truth, scores):
    from src.data.label_eval import auc

    return auc(truth, scores)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/raw")
    parser.add_argument("--write", metavar="NAME",
                        help=f"Blend to write out. One of: {', '.join(CANDIDATES)}")
    parser.add_argument("--out", default="data/processed/report_labels_blend.csv")
    parser.add_argument("--selection", action="store_true",
                        help="Also check whether per-target source picking survives validation.")
    parser.add_argument("--bootstrap", type=int, default=800)
    args = parser.parse_args()

    gold = load_gold(args.data_root)
    tables = {
        s.name: load_label_table(s) for s in LABEL_SOURCES if not s.contaminated
    }

    print("How alike are the sources? (mean Spearman over targets, all 4,407 studies)")
    print(spearman_matrix(tables).to_string(float_format="%.3f"))
    print("\nSources correlating above ~0.95 are one reading, not two.")

    print("\nCandidate blends, scored on the expert-labelled studies")
    rows = []
    for name, members in CANDIDATES.items():
        table = blend({m: tables[m] for m in members})
        lo, hi = bootstrap_macro(gold, table, n=args.bootstrap)
        rows.append({
            "blend": name,
            "sources": " + ".join(members),
            "macro_auc": macro_auc(gold, table),
            "ci95": f"{lo:.3f}-{hi:.3f}",
        })
    print(pd.DataFrame(rows).sort_values("macro_auc", ascending=False)
          .to_string(index=False, float_format="%.4f"))

    if args.selection:
        ranked = {n: to_ranks(t) for n, t in tables.items()}
        in_sample, out_of_sample = selection_is_worth_it(gold, ranked)
        print("\nIs picking the best source per target worth it?")
        print(f"  in sample        {in_sample:.4f}   <- optimistic, this is a fit")
        print(f"  out of sample    {out_of_sample:.4f}   <- what the pick is worth")
        print(f"  optimism         {in_sample - out_of_sample:.4f}")

    if args.write:
        if args.write not in CANDIDATES:
            raise SystemExit(f"unknown blend {args.write!r}; choose from {', '.join(CANDIDATES)}")
        members = CANDIDATES[args.write]
        table = blend({m: tables[m] for m in members})
        table.round(6).to_csv(args.out)
        print(f"\nWrote blend {args.write!r} ({' + '.join(members)}) "
              f"to {args.out} - {len(table)} studies")


if __name__ == "__main__":
    main()
