"""Writing the file Kaggle scores.

**Ranks, not probabilities.** The metric is a macro AUC, invariant under any strictly
increasing transform of a column, so calibration is worth nothing — and per-column
percentile ranks are what make two members blendable at all, whatever scale each emits.

**The benchmark goes first.** A run that dies after the decode pass has spent the
expensive half; a valid 0.5 file left behind scores 0.500 instead of nothing. The cost
is that a failed run then looks successful, which is why `docs/pipeline_pitfalls.md`
§10 says to check the predictions are not all identical before submitting.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..config import TARGETS

SUBMISSION_NAME = "submission.csv"


def benchmark_submission(studies, path: str | Path = SUBMISSION_NAME) -> Path:
    """Write 0.5 for every study — the file a crash should leave behind."""

    frame = pd.DataFrame({"StudyInstanceUID": list(studies)})
    for target in TARGETS:
        frame[target] = 0.5
    return _write(frame, path)


def write_submission(predictions: np.ndarray, studies, path: str | Path = SUBMISSION_NAME,
                     rank: bool = True) -> Path:
    """Write one prediction matrix, as per-column percentile ranks by default."""

    predictions = np.asarray(predictions, dtype=float)
    if predictions.shape != (len(studies), len(TARGETS)):
        raise ValueError(f"expected {(len(studies), len(TARGETS))} predictions, "
                         f"got {predictions.shape}")
    if not np.isfinite(predictions).all():
        raise ValueError("predictions contain NaN or infinity")

    frame = pd.DataFrame(predictions, columns=TARGETS)
    if rank:
        frame = frame.rank(method="average", pct=True)
    frame.insert(0, "StudyInstanceUID", list(studies))
    return _write(frame, path)


def _write(frame: pd.DataFrame, path: str | Path) -> Path:
    expected = ["StudyInstanceUID"] + TARGETS
    if list(frame.columns) != expected:
        raise ValueError(f"submission schema drift: {list(frame.columns)}")
    if not frame["StudyInstanceUID"].is_unique:
        raise ValueError("duplicate study ids in the submission")
    path = Path(path)
    frame.to_csv(path, index=False)
    return path
