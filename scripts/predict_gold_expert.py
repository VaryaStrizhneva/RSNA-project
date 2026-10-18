"""An expert run's probabilities on the 58 radiologist-read studies, written down.

    python -m scripts.predict_gold_expert out/expert_lm_d32_cnxnano

The training loop computes this score every epoch and keeps the scalar in
`history.json`; the per-study probabilities it was computed from are discarded. That is
enough to read one run and useless for comparing two, because **comparing two label
tables is the one question the weak-label AUC cannot answer**: each run is scored
against its own table, and those tables disagree about which studies are positive —
`pilkwang` calls 1114 studies torn on Lateral Meniscus where `v4_blend` calls 690. The
gold is the only label set they share.

So this reloads the saved folds, predicts on the gold studies from the ROI cache the run
was fitted on, averages the folds as probabilities — matching `score_experts`, and
matching how the loop pooled them — and writes `gold_predictions.csv` beside the run.

**What the result is worth.** The gold is 58 studies. The project has already read a
+0.0396 there and seen 0.000 on the leaderboard (rows #24/#25), so a difference under
about 0.05 means nothing. And a run fitted on a table whose construction used those same
58 studies — `riad_hybrid` chose its column per finding on them — cannot be compared here
at all. The number is written anyway, and both caveats travel with it in the header.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from rsna.config import Config                                       # noqa: E402
from rsna.data.labels import TARGET_COLUMNS                          # noqa: E402
from rsna.infer.experts import load_experts, score_experts           # noqa: E402
from rsna.roi import cache                                           # noqa: E402

CACHE = Path("/data/mgr/rsna-knee/roi-cache")


def gold_studies(data_root: Path) -> pd.DataFrame:
    """The studies a radiologist read, with their labels. Complete rows only."""

    train = pd.read_csv(data_root / "train.csv", dtype={"StudyInstanceUID": str})
    train = train.set_index("StudyInstanceUID")[TARGET_COLUMNS]
    return train[train.notna().all(axis=1)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run", type=Path)
    ap.add_argument("--roi-cache", type=Path, default=CACHE)
    ap.add_argument("--data-root", type=Path, default=ROOT / "data/raw")
    ap.add_argument("--device", default="cpu",
                    help="cpu by default: 58 studies take seconds and the GPU is "
                         "usually busy with the training this is meant to read")
    args = ap.parse_args()

    models, config, specs = load_experts(args.run, args.device, log=print)
    gold = gold_studies(args.data_root)

    groups, names = [], None
    for spec in specs:
        volumes, mask, records = cache.load(args.roi_cache, spec)
        cut = [r["study"] for r in records]
        if names is None:
            names = cut
        elif cut != names:
            raise SystemExit(f"{spec.name} was cut over a different set of studies")
        groups.append((volumes, mask))

    at = {s: i for i, s in enumerate(names)}
    keep = [s for s in gold.index if s in at]
    rows = np.array([at[s] for s in keep], int)
    print(f"{len(keep)} of {len(gold)} gold studies are in the cache")

    sub = [(v[rows], m[rows]) for v, m in groups]
    scores = score_experts(models, sub, batch=config.batch, device=args.device)

    out = pd.DataFrame({"study": keep})
    for j, target in enumerate(config.targets):
        out[f"true:{target}"] = (gold.loc[keep, target].to_numpy(float) > 0.5).astype(int)
        out[f"pred:{target}"] = scores[:, j]
    path = args.run / "gold_predictions.csv"
    out.to_csv(path, index=False)

    from sklearn.metrics import roc_auc_score
    for target in config.targets:
        y, p = out[f"true:{target}"], out[f"pred:{target}"]
        if y.nunique() > 1:
            print(f"  {target}: AUC {roc_auc_score(y, p):.4f} "
                  f"({int(y.sum())} positive of {len(y)})")
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
