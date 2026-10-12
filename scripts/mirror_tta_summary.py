"""The mirror averaging, read across runs as a paired per-fold comparison.

    python -m scripts.mirror_tta_summary out/expert_lm_d32_cnxnano out/expert_acl_cnxnano ...

`mirror_tta.py` writes one probability per study per orientation; this reads those files
and asks the only question that matters, which is whether the averaging beats the single
orientation **on the same fold**. Fold-to-fold spread on these columns runs 0.018 to
0.057 and would swamp a 0.003 effect read unpaired -- the same trap the encoder
comparison walked into before it was paired, where pairing took the standard error from
+/-0.006 to 0.0016.

The last line projects the macro. It uses the one calibration this project has actually
measured end to end rather than a guess: the encoder change lifted eight columns by
0.0056 in the mean and moved the leaderboard 0.004, so a sum of per-column gains divided
by the twelve targets reproduced that to within 7 %. The projection is therefore
`sum(delta) / 12`, and it is a projection and not a measurement -- row #25 of the
experiment log is a +0.0396 read locally that the leaderboard scored at 0.000.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+", type=Path)
    args = ap.parse_args()

    deltas, rows = [], []
    for run in args.runs:
        tta = pd.read_csv(run / "mirror_tta.csv", dtype={"study": str})
        hold = pd.read_csv(run / "holdout.csv", dtype={"study": str}).set_index("study")
        targets = [c.split(":", 1)[1] for c in tta.columns if c.startswith("plain:")]
        for target in targets:
            y = (hold.loc[tta["study"], f"true:{target}"].to_numpy(float) > 0.5).astype(int)
            p, m = tta[f"plain:{target}"].to_numpy(), tta[f"mirror:{target}"].to_numpy()
            per = []
            for fold in sorted(tta["fold"].unique()):
                s = (tta["fold"] == fold).to_numpy()
                if len(set(y[s].tolist())) < 2:
                    continue
                per.append((roc_auc_score(y[s], (p[s] + m[s]) / 2),
                            roc_auc_score(y[s], p[s])))
            per = np.array(per)
            d = per[:, 0] - per[:, 1]
            rows.append({"run": run.name, "target": target,
                         "plain": roc_auc_score(y, p),
                         "mirror": roc_auc_score(y, m),
                         "mean": roc_auc_score(y, (p + m) / 2),
                         "fold_sd": per[:, 1].std(ddof=1),
                         "paired_d": d.mean(), "paired_se": d.std(ddof=1) / np.sqrt(len(d)),
                         "folds_won": int((d > 0).sum()), "folds": len(d)}
                        )
            deltas.extend(d.tolist())

    table = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    d = np.array(deltas)
    overall = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
    print(f"\npooled over {len(d)} paired folds: {d.mean():+.4f} "
          f"+/- {d.std(ddof=1) / np.sqrt(len(d)):.4f}, t = {overall:.2f}")
    print(f"spread between folds of the plain score: "
          f"{table['fold_sd'].min():.3f} to {table['fold_sd'].max():.3f} "
          f"-- which is why this is paired")
    gain = table.groupby("target")["mean"].first() - table.groupby("target")["plain"].first()
    print(f"\nsum of per-column gains {gain.sum():+.4f} over {len(gain)} columns "
          f"-> projected macro {gain.sum() / 12:+.4f} (a projection, not a measurement)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
