"""Does averaging over the in-plane jitter the folds trained under buy anything?

    python -m scripts.jitter_tta out/expert_lm_d32_cnxnano out/expert_mcl_cnxnano

The sibling question to `scripts.mirror_tta`, and it covers every expert rather than the
three sagittal ones -- in-plane jitter is a nuisance on any plane, so coronal and axial
regions are in scope too.

**Why it should work less well, stated before the measurement.** The mirror paid
(+0.0024 a column, t = 3.39) because the depth axis is a *discrete* symmetry whose two
signs each cover about half the corpus: the reversed reading is a genuinely different
reading of the same anatomy, which is why it scored the same (0.8955 against 0.8958)
while agreeing only at Spearman 0.92. A random affine has no such structure. It averages
over noise the encoder was explicitly trained to ignore, so the honest prior is that it
buys variance reduction and little else. This project has fifteen levers measured at
zero; this is a cheap test of a sixteenth, not a prediction.

**The views are fixed, not drawn.** The transforms span the training range
(`augment_batch` uses +/-0.14 rad, 0.94 to 1.06 of scale, +/-0.06 of shift) at about half
its radius, deterministically, so the submission does not depend on a seed and the
measurement can be repeated. The photometric half of the training augmentation is not
reproduced: brightness and noise carry nothing about the anatomy.

The identity view is scored by the same code path as the others and checked against the
run's own `holdout.csv` -- a mismatch there means nothing else in the table is readable.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from rsna.config import Config                                          # noqa: E402
from rsna.infer.experts import (JITTER_VIEWS as VIEWS,                 # noqa: E402
                                inplane_jitter, load_experts, score_experts)
from rsna.train import assign_folds                                     # noqa: E402
from scripts.mirror_tta import load_groups                              # noqa: E402

CACHE = Path("/data/mgr/rsna-knee/roi-cache")

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--roi-cache", type=Path, default=CACHE)
    ap.add_argument("--data-root", type=Path, default=ROOT / "data/raw")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    train_csv = pd.read_csv(args.data_root / "train.csv",
                            dtype={"StudyInstanceUID": str})
    from sklearn.metrics import roc_auc_score

    rows = []
    for run in args.runs:
        models, config, specs = load_experts(run, args.device, log=lambda m: None)
        groups, names = load_groups(specs, args.roi_cache)
        holdout = pd.read_csv(run / "holdout.csv", dtype={"study": str})
        meta = json.loads((run / "history.json").read_text()).get("run", {})
        folds = assign_folds(train_csv, Config(weights=meta.get("weights", "uniform")))

        at = {s: i for i, s in enumerate(names)}
        keep = [s for s in holdout["study"] if s in at]
        rec = np.array([at[s] for s in keep], int)
        fold_of = folds.reindex(keep).to_numpy()
        nt = len(config.targets)

        preds = {k: np.full((len(keep), nt), np.nan) for k in ("id",) + tuple(v[0] for v in VIEWS)}
        for fold in sorted(set(fold_of.tolist())):
            if fold < 0 or fold >= len(models):
                continue
            sel = np.flatnonzero(fold_of == fold)
            sub = [(v[rec[sel]], m[rec[sel]]) for v, m in groups]
            one = [models[int(fold)]]
            preds["id"][sel] = score_experts(one, sub, batch=config.batch,
                                             device=args.device)
            for name, a, sc, dx, dy in VIEWS:
                preds[name][sel] = score_experts(
                    one, sub, batch=config.batch, device=args.device,
                    transform=inplane_jitter(a, sc, dx, dy))
        print(f"{run.name}: {config.encoder}, {[sp.plane for sp in specs]}, "
              f"{len(keep)} studies, {len(VIEWS) + 1} views")

        stack = np.stack([preds["id"]] + [preds[v[0]] for v in VIEWS])
        for j, target in enumerate(config.targets):
            y = (holdout.set_index("study").loc[keep, f"true:{target}"].to_numpy(float)
                 > 0.5).astype(int)
            was = holdout.set_index("study").loc[keep, f"pred:{target}"].to_numpy(float)
            if len(set(y.tolist())) < 2:
                continue
            singles = {k: roc_auc_score(y, preds[k][:, j]) for k in preds}
            avg = stack[:, :, j].mean(axis=0)
            per = []
            for fold in sorted(set(fold_of.tolist())):
                s = fold_of == fold
                if len(set(y[s].tolist())) < 2:
                    continue
                per.append((roc_auc_score(y[s], avg[s]),
                            roc_auc_score(y[s], preds["id"][s, j])))
            per = np.array(per)
            d = per[:, 0] - per[:, 1]
            rows.append({"run": run.name, "target": target,
                         "run_auc": roc_auc_score(y, was), **singles,
                         "avg": roc_auc_score(y, avg), "paired_d": d.mean(),
                         "paired_se": d.std(ddof=1) / np.sqrt(len(d)),
                         "won": int((d > 0).sum()), "folds": len(d)})
            if abs(rows[-1]["run_auc"] - singles["id"]) > 0.002:
                print(f"  ** the identity view does not reproduce the run on {target} "
                      f"({rows[-1]['run_auc']:.4f} against {singles['id']:.4f})")

    table = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print("\n" + table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    d = table["paired_d"].to_numpy()
    print(f"\nmean paired delta over {len(d)} columns: {d.mean():+.4f}, "
          f"{int((d > 0).sum())} positive")
    print(f"sum of per-column gains {(table['avg'] - table['id']).sum():+.4f} "
          f"-> projected macro {(table['avg'] - table['id']).sum() / 12:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
