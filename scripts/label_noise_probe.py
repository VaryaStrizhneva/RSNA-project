"""Does the model's disagreement with a label point at a bad label, or just a hard case?

    python -m scripts.label_noise_probe

**The question this answers, and why it comes first.** Dropping the studies a model gets
wrong raises every score measured on what is left, by construction, whether or not a
single label was actually wrong. So before a filtered retrain is worth 2 h 30 of GPU,
the filter has to be shown to detect **label noise** rather than **difficulty**. That
needs a witness to the label's quality that the model had no part in, and there are two.

**Witness one, the other extractors.** `pilkwang` and `flight_v4hybrid` read the same
radiology reports as `steven_v4_blend` with different machinery. Where two extractors
read the same report to opposite verdicts, that cell is genuinely ambiguous. Binarised at
each table's own silence anchor, because the raw scales are not comparable -- pilkwang's
five verdict levels run 0.08/0.28/0.68/0.82/0.94 and v4_blend's do not.

**Witness two, the radiologist.** 58 studies carry a human reading. Tiny, but the
question here is about 58 x 8 label cells rather than about an AUC, which is a different
amount of evidence than row #25's +/-0.05.

**The confound, reported alongside.** A model disagrees most where the label sits near
its silence anchor, and that is difficulty, not noise -- `assertedness` already
downweights exactly those cells. So every bin also reports the mean label confidence. An
effect that disappears once confidence is held fixed is an effect `assertedness` has
already taken.

Out-of-fold predictions throughout: a model judging a study it trained on would be
scoring its own memory.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from rsna.data.labels import (LABEL_SOURCES, load_label_table,     # noqa: E402
                              silence_of)

#: The shipped expert per column, and the run whose out-of-fold predictions to read.
RUNS = {
    "Lateral Meniscus": "expert_lm_d32_cnxnano",
    "Medial Meniscus": "expert_mm_cnxnano",
    "ACL": "expert_acl_cnxnano",
    "MCL": "expert_mcl_cnxnano",
    "Medial OA": "expert_medial_oa_cnxnano_b16",
    "Lateral OA": "expert_lateral_oa_cnxnano_b16",
    "PF OA": "expert_pf_oa_cnxnano",
    "Baker's": "expert_baker_cnxnano_b16",
}

TRAINED_ON = "steven_v4_blend"
WITNESSES = ("pilkwang", "flight_v4hybrid")


def source(name: str):
    for s in LABEL_SOURCES:
        if s.name == name:
            return s
    raise SystemExit(f"no label source named {name}")


def verdicts(name: str) -> tuple[pd.DataFrame, float]:
    """A table binarised at its own silence anchor, and that anchor."""

    src = source(name)
    table = load_label_table(src)
    anchor = silence_of(src)
    if anchor is None:
        # No explicit "the report does not say" score; the corpus median is the only
        # anchor available and it is announced as a weaker one.
        anchor = float(np.nanmedian(table.to_numpy(float)))
        print(f"  {name}: no silence anchor; using the corpus median {anchor:.3f}")
    return (table > anchor).astype(float).where(table.notna()), anchor


def main() -> int:
    print(f"trained on {TRAINED_ON}; witnesses {list(WITNESSES)}\n")
    own, own_anchor = verdicts(TRAINED_ON)
    other = {w: verdicts(w)[0] for w in WITNESSES}
    truth = load_label_table(source(TRAINED_ON))

    train = pd.read_csv(ROOT / "data/raw/train.csv", dtype={"StudyInstanceUID": str})
    gold = train.set_index("StudyInstanceUID")[list(RUNS)]
    gold = gold[gold.notna().all(axis=1)]

    frames = []
    for target, run in RUNS.items():
        path = ROOT / "out" / run / "holdout.csv"
        if not path.exists():
            print(f"  {run}: no holdout.csv, skipped")
            continue
        hold = pd.read_csv(path, dtype={"study": str}).set_index("study")
        col = f"pred:{target}"
        if col not in hold.columns:
            print(f"  {run}: no {col}, skipped")
            continue
        studies = [s for s in hold.index if s in truth.index]
        y = truth.loc[studies, target].to_numpy(float)
        p = hold.loc[studies, col].to_numpy(float)
        mine = own.loc[studies, target].to_numpy(float)
        # How far the label is from "the report does not say" -- the confound.
        confidence = np.where(y >= own_anchor,
                              (y - own_anchor) / (1 - own_anchor),
                              (own_anchor - y) / own_anchor)
        conflict = np.zeros(len(studies))
        seen = np.zeros(len(studies))
        for w, table in other.items():
            v = table.reindex(studies)[target].to_numpy(float)
            ok = np.isfinite(v) & np.isfinite(mine)
            conflict[ok] += (v[ok] != mine[ok]).astype(float)
            seen[ok] += 1
        frames.append(pd.DataFrame(
            {"study": studies, "target": target, "y": y, "p": p,
             "confidence": confidence,
             # Disagreement judged on the label's own side of the anchor, so a confident
             # positive the model calls 0.02 scores high and a silent cell does not.
             "d_model": np.abs(mine - p),
             "conflict": np.where(seen > 0, conflict / np.maximum(seen, 1), np.nan)}))

    all_ = pd.concat(frames, ignore_index=True)
    all_ = all_[all_["conflict"].notna()]
    print(f"\n{len(all_)} label cells over {all_['target'].nunique()} columns, "
          f"{all_['study'].nunique()} studies")
    print(f"the witnesses contradict {TRAINED_ON} on "
          f"{100 * (all_['conflict'] > 0).mean():.1f} % of cells\n")

    print("by decile of the model's disagreement with the label:")
    all_["bin"] = pd.qcut(all_["d_model"], 10, labels=False, duplicates="drop")
    grouped = all_.groupby("bin").agg(
        cells=("d_model", "size"), d_model=("d_model", "mean"),
        label_confidence=("confidence", "mean"),
        witnesses_disagree=("conflict", "mean"))
    print(grouped.to_string(float_format=lambda v: f"{v:.3f}"))

    # Hold the confound fixed: only cells the table asserts strongly, where
    # `assertedness` would already be giving near-full weight.
    firm = all_[all_["confidence"] > 0.75].copy()
    print(f"\nholding label confidence fixed (>0.75, {len(firm)} cells):")
    firm["bin"] = pd.qcut(firm["d_model"], 5, labels=False, duplicates="drop")
    print(firm.groupby("bin").agg(
        cells=("d_model", "size"), d_model=("d_model", "mean"),
        label_confidence=("confidence", "mean"),
        witnesses_disagree=("conflict", "mean")).to_string(
            float_format=lambda v: f"{v:.3f}"))

    g = all_[all_["study"].isin(gold.index)].copy()
    if len(g):
        g["gold"] = [gold.loc[s, t] > 0.5 for s, t in zip(g["study"], g["target"])]
        g["table_wrong"] = (g["y"] > own_anchor) != g["gold"]
        print(f"\nagainst the radiologist, {len(g)} cells on "
              f"{g['study'].nunique()} studies:")
        g["bin"] = pd.qcut(g["d_model"], 4, labels=False, duplicates="drop")
        print(g.groupby("bin").agg(
            cells=("d_model", "size"), d_model=("d_model", "mean"),
            table_wrong_vs_gold=("table_wrong", "mean")).to_string(
                float_format=lambda v: f"{v:.3f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
