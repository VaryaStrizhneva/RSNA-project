"""Does an expert care which way its sagittal stack runs?

    python -m scripts.mirror_tta out/expert_lm_d32_cnxnano

**What the question is.** `choose_slices` returns the kept slices *in stack order*, and a
sagittal stack's order is anatomical only up to the knee's side: increasing
through-plane position runs medial-to-lateral on a left knee and lateral-to-medial on a
right one. `toward_bowtie` fixes the window's *extent* -- 8 mm one way, 24 the other --
and nothing fixes its *direction*. Measured on the cached crops, the per-slot lit-area
profile of the left knees correlates +0.773 with the right knees **reversed** and only
+0.223 with them as they are; the coronal regions, whose depth axis is
anterior-posterior and never a laterality axis, give +0.97 and -0.90, which is the
control that says the measurement is reading what it claims to.

So the depth axis is a nuisance axis, the encoder has seen both of its signs in roughly
equal number, and reversing it at inference is **in distribution**. The same is not true
of a coronal or axial region: there laterality is in-plane and `crop_plane` already
mirrors it so that lateral is on the right, so a reversal would be an input the model
has never been shown. Only `plane == "Sagittal"` regions are reversed here.

**Three numbers come out, and the third is the one that decides anything.**

`plain` reproduces the run. It is checked against the run's own `holdout.csv` rather
than trusted: a mismatch means this inference path differs from the training loop's and
no other number here is worth reading.

`mean` is the test-time augmentation -- the average of the two orientations. It can only
buy variance reduction.

`mirror` alone is the diagnostic. If it matches `plain`, the encoder is invariant to the
nuisance: it paid for that invariance in capacity but it got it, and canonicalising the
axis would buy back only what the invariance cost. If it collapses, the encoder keyed on
the orientation -- it has been solving two problems with half the studies each -- and
the fix is not this averaging but ordering the slots by signed anatomical distance in
`choose_slices`, which costs a cache rebuild and a retrain.

**One artefact, bounded rather than assumed away.** `extract` centres the acquired run
in the slot grid, so where the padding is odd a reversal shifts the run by one slot.
`ExpertNet.windows` clamps a window's outer channels to the ends of the run and pools
over windows, so the set of windows is unchanged by that shift -- unless the shift moves
the run across the first or last admissible centre, which would change how many windows
are live. The count is compared for both orientations and reported.
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

from rsna.config import Config                                       # noqa: E402
from rsna.infer.experts import load_experts, mirror_groups, score_experts           # noqa: E402
from rsna.roi import cache                                           # noqa: E402

CACHE = Path("/data/mgr/rsna-knee/roi-cache")


def load_groups(specs: list, roi_cache: Path) -> tuple[list, list]:
    """One `(volumes, mask)` per region, and the study order they share.

    Assembled the same way `rsna.infer.chain` assembles it, and refusing the same thing:
    two regions cut over different studies are two alignments and one model cannot read
    both.
    """

    groups, names = [], None
    for spec in specs:
        volumes, mask, records = cache.load(roi_cache, spec)
        cut = [r["study"] for r in records]
        if names is None:
            names = cut
        elif cut != names:
            raise SystemExit(f"{spec.name} was cut over a different set of studies")
        groups.append((volumes, mask))
    return groups, names


def live_windows(mask: np.ndarray, group: int) -> np.ndarray:
    """How many windows `ExpertNet.windows` would call real, per study.

    A window is real when its centre slot holds an acquired slice, and centres run from
    `group // 2` to `slots - group // 2`.
    """

    half = group // 2
    centres = mask[:, :, half:mask.shape[2] - half]
    return centres.astype(bool).sum(axis=(1, 2))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run", type=Path)
    ap.add_argument("--roi-cache", type=Path, default=CACHE)
    ap.add_argument("--data-root", type=Path, default=ROOT / "data/raw")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    models, config, specs = load_experts(args.run, args.device, log=print)
    planes = [sp.plane for sp in specs]
    print(f"{args.run.name}: {config.encoder}, regions {list(config.rois)} "
          f"on {planes}")
    if "Sagittal" not in planes:
        raise SystemExit("no sagittal region: the depth axis here is not a laterality "
                         "axis and reversing it would be an input the model never saw")

    groups, names = load_groups(specs, args.roi_cache)

    flipped = mirror_groups(groups, specs)
    for (_, m), (_, fm), spec in zip(groups, flipped, specs):
        if spec.plane != "Sagittal":
            continue
        a, b = live_windows(m, config.group), live_windows(fm, config.group)
        moved = int((a != b).sum())
        print(f"  {spec.name}: live windows differ on {moved} of {len(a)} studies "
              f"({100 * moved / len(a):.2f} %) after the reversal")

    # The truth, and which fold scored each study, exactly as the run had them. The
    # thresholded truth is read off the run's own holdout rather than rebuilt, so this
    # cannot disagree with the table the run was fitted against.
    holdout = pd.read_csv(args.run / "holdout.csv", dtype={"study": str})
    train_csv = pd.read_csv(args.data_root / "train.csv", dtype={"StudyInstanceUID": str})
    run_meta = json.loads((args.run / "history.json").read_text()).get("run", {})
    base = Config(weights=run_meta.get("weights", "uniform"))
    from rsna.train import assign_folds
    folds = assign_folds(train_csv, base)

    at = {s: i for i, s in enumerate(names)}
    keep = [s for s in holdout["study"] if s in at]
    if len(keep) != len(holdout):
        print(f"  {len(holdout) - len(keep)} scored studies are not in the cache")
    rows = np.array([at[s] for s in keep], int)
    fold_of = folds.reindex(keep).to_numpy()

    nt = len(config.targets)
    plain = np.full((len(keep), nt), np.nan, np.float64)
    mirror = np.full((len(keep), nt), np.nan, np.float64)
    for fold in sorted(set(fold_of.tolist())):
        if fold < 0 or fold >= len(models):
            continue
        sel = np.flatnonzero(fold_of == fold)
        idx = rows[sel]
        one = [models[int(fold)]]
        plain[sel] = score_experts(one, [(v[idx], m[idx]) for v, m in groups],
                                   batch=config.batch, device=args.device)
        mirror[sel] = score_experts(one, [(v[idx], m[idx]) for v, m in flipped],
                                    batch=config.batch, device=args.device)
        print(f"  fold {fold}: {len(sel)} studies scored both ways")

    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score

    print(f"\n{'target':22s} {'run':>7s} {'plain':>7s} {'mirror':>7s} {'mean':>7s} "
          f"{'delta':>7s} {'rho':>6s}")
    for j, target in enumerate(config.targets):
        y = (holdout.set_index("study").loc[keep, f"true:{target}"].to_numpy(float)
             > 0.5).astype(int)
        was = holdout.set_index("study").loc[keep, f"pred:{target}"].to_numpy(float)
        if len(set(y.tolist())) < 2:
            continue
        auc = lambda p: roc_auc_score(y, p)                            # noqa: E731
        a_run, a_plain = auc(was), auc(plain[:, j])
        a_mir, a_mean = auc(mirror[:, j]), auc((plain[:, j] + mirror[:, j]) / 2)
        rho = spearmanr(plain[:, j], mirror[:, j]).statistic
        print(f"{target:22s} {a_run:7.4f} {a_plain:7.4f} {a_mir:7.4f} {a_mean:7.4f} "
              f"{a_mean - a_plain:+7.4f} {rho:6.3f}")
        if abs(a_run - a_plain) > 0.002:
            print(f"  ** this path does not reproduce the run on {target} "
                  f"({a_run:.4f} against {a_plain:.4f}); nothing above is readable")

    out = pd.DataFrame({"study": keep, "fold": fold_of})
    for j, target in enumerate(config.targets):
        out[f"plain:{target}"] = plain[:, j]
        out[f"mirror:{target}"] = mirror[:, j]
    path = args.run / "mirror_tta.csv"
    out.to_csv(path, index=False)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
