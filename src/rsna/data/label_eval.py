"""Scoring label tables against the expert-labelled studies.

Everything here answers one
question — how well does a column of scores order the 58 studies that carry
official labels — and everything here is limited by the same thing: those 58
studies are the entire ground truth, and for MCL they contain nine positives.
Point estimates from this module are noisy, which is why the interval helpers
exist alongside them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .metadata import TARGET_COLUMNS, get_expert_labeled_rows, load_metadata


def load_gold(data_root: str = "data/raw") -> pd.DataFrame:
    """The expert labels, indexed by study, as integers."""

    train = load_metadata(data_root).train
    gold = get_expert_labeled_rows(train)
    return gold.set_index("StudyInstanceUID")[TARGET_COLUMNS].astype(int)


def auc(truth: pd.Series, scores: pd.Series) -> float:
    """AUC for one target, tolerating a source that did not cover every study.

    An uncovered study is scored at the source's mean rank, so it neither helps
    nor hurts. pilkwang's table is one study short of the corpus.
    """

    if truth.nunique() < 2:
        return float("nan")
    filled = scores.fillna(scores.mean())
    if filled.isna().all():
        return float("nan")
    return roc_auc_score(truth, filled)


def per_target_auc(gold: pd.DataFrame, table: pd.DataFrame) -> pd.Series:
    """AUC of every target for one table."""

    aligned = table.reindex(gold.index)
    return pd.Series({t: auc(gold[t], aligned[t]) for t in TARGET_COLUMNS})


def macro_auc(gold: pd.DataFrame, table: pd.DataFrame) -> float:
    """The competition metric: unweighted mean of the twelve AUCs."""

    return float(per_target_auc(gold, table).mean())


def bootstrap_macro(
    gold: pd.DataFrame, table: pd.DataFrame, n: int = 2000, seed: int = 0
) -> tuple[float, float]:
    """Percentile interval for the macro AUC, resampling studies with replacement.

    Reported next to every point estimate because a 58-study AUC has an interval
    roughly seven points wide, and differences smaller than that are not real.
    """

    rng = np.random.default_rng(seed)
    aligned = table.reindex(gold.index)
    positions = np.arange(len(gold))

    draws = []
    for _ in range(n):
        take = rng.choice(positions, size=len(positions), replace=True)
        truth, scores = gold.iloc[take], aligned.iloc[take]
        values = [auc(truth[t], scores[t]) for t in TARGET_COLUMNS]
        with np.errstate(invalid="ignore"):
            draws.append(np.nanmean(values))

    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))
