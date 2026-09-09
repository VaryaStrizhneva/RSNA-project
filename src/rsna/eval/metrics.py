"""Numbers from arrays. No file reading, no drawing, no printing.

Kept apart from the rest of the package so it can be tested with sixteen synthetic
studies: every claim a report makes is one of these functions, and a function that
takes two arrays and returns a float is a claim that can be checked.
"""

from __future__ import annotations

import numpy as np

from ..config import TARGETS

# Below this many positives (or negatives) in a column, its AUC is a coin toss dressed
# as a measurement. Reported, but flagged, and never quietly averaged into a headline.
MIN_POSITIVES = 5


def per_target_auc(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """AUC per target. NaN where the column holds one class only."""

    from sklearn.metrics import roc_auc_score

    out = np.full(y.shape[1], np.nan)
    for j in range(y.shape[1]):
        column = y[:, j]
        if 0 < column.sum() < len(column):
            out[j] = roc_auc_score(column, p[:, j])
    return out


def macro_auc(y: np.ndarray, p: np.ndarray) -> float:
    """The competition metric: the unweighted mean over the targets that scored."""

    scores = per_target_auc(y, p)
    return float(np.nanmean(scores)) if not np.all(np.isnan(scores)) else float("nan")


def rank_normalise(p: np.ndarray) -> np.ndarray:
    """Map each column onto [0, 1] by rank.

    Necessary before pooling folds. AUC reads ranks alone, so a model's output scale is
    arbitrary — and five models fitted on five different subsets have five different
    scales. Concatenating them raw computes an AUC over a mixture of scales, which is
    not the same quantity as any of them and is silently wrong rather than noisy.
    """

    from scipy.stats import rankdata

    out = np.empty(p.shape, np.float64)
    denominator = max(p.shape[0] - 1, 1)
    for j in range(p.shape[1]):
        out[:, j] = (rankdata(p[:, j]) - 1.0) / denominator
    return out


def pool(records: list) -> tuple[np.ndarray, np.ndarray]:
    """Every fold's holdout, stacked, each fold rank-normalised first.

    This is the out-of-fold set: each study was predicted exactly once, by a model that
    never saw it. It is the only assembly in which the per-target numbers have enough
    studies behind them to mean anything.
    """

    ys = [r.y for r in records if r.n]
    ps = [rank_normalise(r.p) for r in records if r.n]
    if not ys:
        return (np.zeros((0, len(TARGETS))), np.zeros((0, len(TARGETS))))
    return np.concatenate(ys), np.concatenate(ps)


def bootstrap_macro(y: np.ndarray, p: np.ndarray, n_boot: int = 2000,
                    seed: int = 0, level: float = 0.95) -> tuple[float, float]:
    """Percentile confidence interval for the macro AUC, resampling studies.

    Studies are resampled, not cells: the twelve targets of one study are correlated,
    and resampling cells would treat them as twelve independent observations and return
    an interval several times too narrow.
    """

    if len(y) < 3:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        take = rng.integers(0, len(y), len(y))
        value = macro_auc(y[take], p[take])
        if np.isfinite(value):
            draws.append(value)
    if len(draws) < n_boot // 10:
        return (float("nan"), float("nan"))
    lo = float(np.percentile(draws, 100 * (1 - level) / 2))
    hi = float(np.percentile(draws, 100 * (1 + level) / 2))
    return lo, hi


def auc_interval(auc: float, positives: int, negatives: int,
                 z: float = 1.96) -> tuple[float, float]:
    """Hanley-McNeil interval for one AUC.

    An AUC is a rank statistic over `positives x negatives` pairs, but those pairs are
    not independent: they are built from `positives + negatives` observations. Hanley
    and McNeil's variance accounts for that, which a naive binomial does not, and it is
    what makes a column of nine positives report the width it deserves.
    """

    if not np.isfinite(auc) or positives < 1 or negatives < 1:
        return (float("nan"), float("nan"))
    q1 = auc / (2.0 - auc)
    q2 = 2.0 * auc * auc / (1.0 + auc)
    variance = (auc * (1 - auc)
                + (positives - 1) * (q1 - auc * auc)
                + (negatives - 1) * (q2 - auc * auc)) / (positives * negatives)
    half = z * float(np.sqrt(max(variance, 0.0)))
    return (max(auc - half, 0.0), min(auc + half, 1.0))


def target_table(y: np.ndarray, p: np.ndarray) -> list[dict]:
    """One row per target: how many positives, what it scored, and how wide that is.

    No column here is constant across runs. The label table's own AUC per target used
    to sit alongside, as a ceiling; it was dropped because it never moves — it is a
    property of the labels, recorded in `docs/data.md`, not a result of a run — and
    because subtracting it compared two figures measured on different studies against
    different truths.
    """

    aucs = per_target_auc(y, p)
    rows = []
    for j, target in enumerate(TARGETS):
        positives = int(y[:, j].sum())
        negatives = int(len(y) - positives)
        lo, hi = auc_interval(float(aucs[j]), positives, negatives)
        rows.append({
            "target": target,
            "positives": positives,
            "n": int(len(y)),
            "auc": float(aucs[j]),
            "lo": lo,
            "hi": hi,
            # An interval straddling 0.5 means the run showed nothing on this target,
            # whichever side of chance the point estimate happens to fall.
            "flat": bool(np.isfinite(lo) and lo <= 0.5 <= hi),
            "thin": positives < MIN_POSITIVES or negatives < MIN_POSITIVES,
        })
    return rows


def flags(records: list, rows: list[dict], fold_aucs: np.ndarray) -> list[dict]:
    """Everything that looks wrong, as checkable conditions rather than impressions.

    Each returns a level and a sentence naming the number that triggered it. The point
    is to make a bad run *say* it is bad: a macro AUC of 0.69 over five folds that
    range from 0.55 to 0.81 is not a result, and nothing about the headline shows it.
    """

    out: list[dict] = []

    def add(level, text):
        out.append({"level": level, "text": text})

    finite = fold_aucs[np.isfinite(fold_aucs)]
    if len(finite) >= 2:
        spread = float(finite.std())
        if spread > 0.05:
            add("error", f"The folds disagree by {spread:.3f} (from {finite.min():.3f} "
                         f"to {finite.max():.3f}). The holdouts are too small to "
                         f"separate this run from another one; do not read the mean as "
                         f"a score.")
        elif spread > 0.02:
            add("warn", f"Fold spread {spread:.3f} — enough to hide a difference of a "
                        f"couple of points between two configurations.")

    for record in records:
        epochs = record.history
        if not epochs:
            continue
        if record.best_epoch == len(epochs) - 1 and len(epochs) > 1:
            add("warn", f"Fold {record.fold} scored best at its final epoch "
                        f"({len(epochs)}): it had not finished improving. Train longer "
                        f"before concluding anything about this configuration.")
        losses = [e["loss"] for e in epochs if e.get("loss") is not None]
        if len(losses) >= 4 and losses[-1] < min(losses[:-3]) * 0.98:
            add("info", f"Fold {record.fold}: training loss was still falling at the "
                        f"end ({losses[-1]:.4f}). Undertrained rather than overfitted.")
        if len(epochs) >= 6 and record.best_epoch < len(epochs) // 3:
            add("warn", f"Fold {record.fold} peaked at epoch {record.best_epoch + 1} of "
                        f"{len(epochs)} and got worse after: overfitting, or a holdout "
                        f"too small for the peak to be real.")

    for row in rows:
        if not np.isfinite(row["auc"]):
            add("error", f"{row['target']}: no AUC — the out-of-fold set holds one "
                         f"class only ({row['positives']} positives of {row['n']}).")
            continue
        if row["thin"]:
            add("warn", f"{row['target']}: {row['positives']} positives of {row['n']}, "
                        f"so {row['auc']:.3f} spans {row['lo']:.3f}-{row['hi']:.3f}. "
                        f"Read it as no measurement rather than as a score.")
        if row["flat"] and not row["thin"]:
            # Deliberately not "below chance": with this many positives the interval
            # covers 0.5, so a point estimate under it is not evidence of an inversion.
            add("warn", f"{row['target']}: {row['auc']:.3f}, interval "
                        f"{row['lo']:.3f}-{row['hi']:.3f} covers 0.5. Nothing was "
                        f"measured here — not a signal, and not an inversion either.")
        elif row["auc"] < 0.35:
            add("error", f"{row['target']}: {row['auc']:.3f}, far enough below chance "
                         f"that the ordering looks inverted rather than absent.")

    order = {"error": 0, "warn": 1, "info": 2}
    return sorted(out, key=lambda f: order[f["level"]])
