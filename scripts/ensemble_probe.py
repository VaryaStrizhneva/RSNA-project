"""Several already-trained runs on one column, averaged as ranks and priced out of fold.

    python -m scripts.ensemble_probe              # the shipped pairs, k = 2
    python -m scripts.ensemble_probe --k 3        # and whether a third adds
    python -m scripts.ensemble_probe --nested     # is choosing the partner worth it?

**No training and no GPU.** Every column has between 4 and 16 runs on disk from earlier
experiments, each with a `holdout.csv` of out-of-fold predictions over the same studies
and the same report-hash folds, so the whole measurement is a read of CSVs.

**Ranks, never probabilities.** Two runs trained with different crops and encoders are
not calibrated against each other, and the metric is an AUC, so only the ordering
carries anything. `chain.py` averages the same way at inference.

**The rule is fixed in advance: the shipped run plus the next best by out-of-fold.** No
search over pairs — `--nested` is there to show why. Picking the partner per column by
what ensembles best on these very studies reports the search's luck as skill; selecting
on four folds and judging on the fifth puts it at **-0.0006 against the fixed rule, 9
signs of 35**, i.e. significantly worse.

**Every delta is paired per fold**, which is the only one of our three predictors that
has ever had the right sign (the 58-study gold inverted one, and absolute out-of-fold
missed it too).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, kendalltau, rankdata
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from rsna.config import Config                                       # noqa: E402
from rsna.train import assign_folds                                  # noqa: E402

#: What the published bundle actually mounts, one run per column. **Read off the
#: dataset, not off the inventory**: the first version of this file had MCL's base as
#: `expert_mcl_cnx_lr1e-4` and the partner dataset was built with a second copy of the
#: real one, which the graft cell would have collapsed into a single run without raising.
SHIPPED = {
    "ACL": "expert_acl_cnxnano",
    "MCL": "expert_mcl_cnxnano",
    "Lateral Meniscus": "expert_lm_wide_s11_lr1e-4",
    "Medial Meniscus": "expert_mm_wide_s11_lr1e-4",
    "Medial OA": "expert_medial_oa_wide_lr1e-4",
    "Lateral OA": "expert_lateral_oa_wide_lr1e-4",
    "PF OA": "expert_pf_oa_wide_lr1e-4",
    "Baker's": "expert_baker_cnxnano_b16",
}


def inventory(out: Path) -> dict[str, list[tuple[float, str, str]]]:
    """Column -> (out-of-fold AUC, run name, roi name), best first."""

    found: dict[str, list] = {}
    for path in sorted(glob.glob(str(out / "expert_*" / "history.json"))):
        run = os.path.dirname(path)
        try:
            meta = json.load(open(path))
        except Exception:
            continue
        targets = tuple(meta.get("config", {}).get("targets", []))
        auc = meta.get("oof", {}).get("mean")
        if len(targets) != 1 or not auc or not os.path.exists(f"{run}/holdout.csv"):
            continue
        roi = (meta["config"].get("rois") or ["?"])[0]
        found.setdefault(targets[0], []).append((auc, os.path.basename(run), roi))
    return {k: sorted(v, reverse=True) for k, v in found.items()}


def readings(out: Path, run: str) -> tuple[pd.Series, pd.Series]:
    frame = pd.read_csv(out / run / "holdout.csv").set_index("study")
    truth = [c for c in frame if c.startswith("true:")][0]
    pred = [c for c in frame if c.startswith("pred:")][0]
    return frame[truth], frame[pred]


def per_fold(y: pd.Series, p: pd.Series, folds: np.ndarray) -> np.ndarray:
    """One AUC per fold. Pooling the five into one ranking costs 0.005 to 0.016 and
    80 % of the pairs it scores are cross-fold, so the comparison is made fold by fold."""

    out = []
    for k in sorted(set(folds)):
        mask = folds == k
        binary = (y[mask] > 0.5).astype(int)
        if binary.nunique() > 1:
            out.append(roc_auc_score(binary, p[mask]))
    return np.array(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "out")
    ap.add_argument("--data-root", type=Path, default=ROOT / "data/raw")
    ap.add_argument("--k", type=int, default=2,
                    help="runs a column. The third is worth +0.0034 and the fourth "
                         "+0.0005, so 3 is where it stops.")
    ap.add_argument("--nested", action="store_true",
                    help="price choosing the partner per column, by selecting on four "
                         "folds and judging on the fifth")
    args = ap.parse_args()

    train = pd.read_csv(args.data_root / "train.csv", dtype={"StudyInstanceUID": str})
    folds = assign_folds(train, Config())
    cat = inventory(args.out)

    deltas, signs, total = [], 0, 0
    print(f"{'column':18s} {'shipped':30s} {'partner':30s} {'one':7s} "
          f"{f'k={args.k}':7s} {'delta':8s} signs tau cuts")
    for column, base in SHIPPED.items():
        if column not in cat:
            print(f"{column:18s} no run with a holdout on disk")
            continue
        rows = cat[column]
        base_roi = next(r[2] for r in rows if r[1] == base)
        chosen = [r for r in rows if r[1] != base][:args.k - 1]
        if not chosen:
            continue
        y, p0 = readings(args.out, base)
        index = y.index
        fold = folds.reindex(index).values
        ranks = [rankdata(p0)]
        for _, name, _ in chosen:
            ranks.append(rankdata(readings(args.out, name)[1].reindex(index)))
        alone = per_fold(y, p0, fold)
        ens = per_fold(y, pd.Series(np.mean(ranks, axis=0), index=index), fold)
        d = ens - alone
        deltas.append(d.mean())
        signs += int((d > 0).sum())
        total += len(d)
        tau = kendalltau(ranks[0][:3000], ranks[1][:3000]).statistic
        cuts = len({base_roi} | {r[2] for r in chosen})
        print(f"{column:18s} {base.replace('expert_',''):30s} "
              f"{', '.join(n.replace('expert_','') for _, n, _ in chosen)[:30]:30s} "
              f"{alone.mean():.4f}  {ens.mean():.4f}  {d.mean():+.4f}  "
              f"{(d > 0).sum()}/{len(d)}  {tau:.2f} {cuts}")
    p = binomtest(signs, total, 0.5, alternative="greater").pvalue
    print(f"\nmean {np.mean(deltas):+.4f} a column, signs {signs}/{total}, p={p:.5f}, "
          f"macro over {len(deltas)} columns {np.mean(deltas) * len(deltas) / 12:+.4f}")
    print("Out of fold is agreement with the label extractor. Transfer it onto the true "
          "scale as a relative error cut before believing it: on the three columns the "
          "solo probes priced, +0.0088 a column becomes about +0.0050 of macro, and "
          "less again after the 0.5 blend with the host.")

    if not args.nested:
        return 0

    print("\n--- is choosing the partner worth anything, net of the bias of choosing? ---")
    rows_out = []
    for column, base in SHIPPED.items():
        if column not in cat:
            continue
        cands = [n for _, n, _ in cat[column] if n != base]
        fixed_partner = cands[0]
        y, p0 = readings(args.out, base)
        index = y.index
        fold = folds.reindex(index).values
        rk = {base: rankdata(p0)}
        for n in cands:
            rk[n] = rankdata(readings(args.out, n)[1].reindex(index))

        def score(name, mask):
            pair = pd.Series((rk[base] + rk[name]) / 2, index=index)
            binary = (y[mask] > 0.5).astype(int)
            return roc_auc_score(binary, pair[mask]) if binary.nunique() > 1 else np.nan

        for k in sorted(set(fold)):
            train_mask, test_mask = fold != k, fold == k
            best = max(cands, key=lambda n: score(n, train_mask))
            rows_out.append((column, best, score(best, test_mask),
                             score(fixed_partner, test_mask)))
    df = pd.DataFrame(rows_out, columns=["column", "pick", "chosen", "fixed"])
    for column, g in df.groupby("column"):
        top = g["pick"].value_counts()
        print(f"{column:18s} chosen {g.chosen.mean():.4f}  fixed {g.fixed.mean():.4f}  "
              f"{g.chosen.mean() - g.fixed.mean():+.4f}  "
              f"({top.iloc[0]}/5 {top.index[0].replace('expert_', '')})")
    d = (df.chosen - df.fixed).values
    print(f"\nover {len(d)} held-out folds: selection - fixed rule = {d.mean():+.4f}, "
          f"signs {(d > 0).sum()}/{len(d)}, "
          f"p={binomtest(int((d > 0).sum()), len(d), 0.5, alternative='greater').pvalue:.3f}")
    print("Significantly WORSE. Take the k best by out-of-fold and do not search.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
