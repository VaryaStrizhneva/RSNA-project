"""Splitting the corpus without scoring a model on what it trained on.

Targets are derived from the reports, and **some reports are byte-identical across
studies** — templates read out for an unremarkable knee. Every study in such a group
receives the same target vector, so splitting the group across a fold boundary scores
the model against a target whose source it has already trained on. Validation rises,
the leaderboard does not.

Grouping is therefore on the report text, not on the study id.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from ..config import TARGETS, Config


def report_group(report: object) -> int:
    """A stable integer per distinct report text.

    Hashing the text rather than assigning group ids by first appearance keeps the
    grouping identical between runs, machines and splits — a fold that moves between
    runs makes two experiments incomparable for a reason nobody would look for.
    """

    text = "" if report is None or (isinstance(report, float) and pd.isna(report)) else str(report)
    return int(hashlib.md5(text.encode()).hexdigest()[:8], 16)


def assign_folds(train: pd.DataFrame, config: Config) -> pd.Series:
    """Study id -> fold index, grouped so identical reports never straddle a boundary."""

    groups = train["Report"].map(report_group)
    folds = groups % config.n_folds
    return pd.Series(folds.values, index=train["StudyInstanceUID"].values, name="fold")


def fold_report(train: pd.DataFrame, folds: pd.Series, config: Config) -> pd.DataFrame:
    """Fold sizes, plus how many studies share a report with another study.

    Printed rather than assumed: if duplicate reports were rare the grouping would be
    pointless, and if they were dominant the folds would be badly unbalanced.
    """

    groups = train.set_index("StudyInstanceUID")["Report"].map(report_group)
    sizes = groups.value_counts()
    shared = int(sizes[sizes > 1].sum())

    rows = []
    for fold in range(config.n_folds):
        members = folds[folds == fold].index
        gold = train.set_index("StudyInstanceUID").loc[members, TARGETS].notna().all(axis=1)
        rows.append({"fold": fold, "studies": len(members), "expert_labelled": int(gold.sum())})

    out = pd.DataFrame(rows)
    out.attrs["studies_sharing_a_report"] = shared
    out.attrs["distinct_reports"] = int(sizes.size)
    return out


def build_targets(
    studies: list[str],
    train: pd.DataFrame,
    derived: pd.DataFrame,
    config: Config,
    confidence: pd.DataFrame | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Targets and per-target loss weights, for studies in the given order.

    Expert labels win wherever they exist and carry `config.gold_weight`; there are
    only 58 of them and they are the best labels in the corpus, so they stay in
    training rather than being held out as a test set.

    Report-derived labels are weighted between 0.25 and 1.0 by the source's own
    confidence when it reports one. A table with no confidence column is treated as
    uniformly confident rather than rejected — but note that a study covered by no
    source at all gets weight zero and drops out of training entirely, which is
    visible in the returned weights and should be counted by the caller.
    """

    gold = train.set_index("StudyInstanceUID")[TARGETS]
    gold = gold[gold.notna().all(axis=1)]

    y = np.zeros((len(studies), len(TARGETS)), np.float32)
    w = np.zeros_like(y)

    for i, study in enumerate(studies):
        if study in gold.index:
            y[i] = gold.loc[study].values
            w[i] = config.gold_weight
        elif study in derived.index:
            y[i] = derived.loc[study, TARGETS].values
            if confidence is not None and study in confidence.index:
                w[i] = 0.25 + 0.75 * confidence.loc[study, TARGETS].values
            else:
                w[i] = 1.0
    return y, w
