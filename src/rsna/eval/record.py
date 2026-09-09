"""What a training run leaves behind for a report to read.

Two files beside the weights, both text, both small:

* `history.json` — one entry per epoch, plus what the run was.
* `holdout.csv` — the held-out studies, their targets and the predictions of the
  selected epoch.

Everything a report says is derived from these. That is the point: a report needs no
GPU, no pixel cache and no checkpoint, so it can be regenerated after a config change,
on another machine, or a month later, and it always says the same thing about the same
run. The alternative — re-running inference to draw a chart — costs a forward pass per
fold and quietly reports a *different* model whenever the code around it has moved.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import TARGETS

HISTORY = "history.json"
HOLDOUT = "holdout.csv"
UID = "StudyInstanceUID"


@dataclass
class RunRecord:
    """One fold: what it was, how it went, and what it predicted."""

    experiment: str
    fold: int
    split: str
    labels: str
    best_epoch: int
    best_holdout_auc: float
    history: list[dict]
    uids: list[str]
    y: np.ndarray            # (n_studies, 12) binary targets
    p: np.ndarray            # (n_studies, 12) predicted probabilities

    @property
    def n(self) -> int:
        return len(self.uids)


def _finite(x) -> float | None:
    """NaN is not JSON. An unmeasurable score is null, which is what it means."""

    x = float(x)
    return x if np.isfinite(x) else None


def write_run_record(path: str | Path, experiment: str, fold: int, split: str,
                     labels: str, result, uids: list[str]) -> Path:
    """Write `history.json` and `holdout.csv` for one fitted fold."""

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    (path / HISTORY).write_text(json.dumps({
        "experiment": experiment,
        "fold": fold,
        "split": split,
        "labels": labels,
        "best_epoch": int(result.best_epoch),
        "best_holdout_auc": _finite(result.best_holdout_auc),
        "epochs": [{"epoch": int(e.epoch),
                    "loss": _finite(e.loss),
                    "holdout_auc": _finite(e.holdout_auc),
                    "annotation_auc": _finite(e.annotation_auc)}
                   for e in result.history],
    }, indent=1) + "\n", encoding="utf-8")

    if result.holdout_pred is not None and result.holdout_true is not None:
        frame = pd.DataFrame({UID: uids})
        for j, target in enumerate(TARGETS):
            frame[f"true:{target}"] = np.asarray(result.holdout_true)[:, j]
            frame[f"pred:{target}"] = np.asarray(result.holdout_pred)[:, j]
        frame.to_csv(path / HOLDOUT, index=False)
    return path


def read_run_record(path: str | Path) -> RunRecord:
    """Read back one fold. Raises if the directory holds no record."""

    path = Path(path)
    meta_file = path / HISTORY
    if not meta_file.is_file():
        raise FileNotFoundError(f"no {HISTORY} under {path}")
    meta = json.loads(meta_file.read_text(encoding="utf-8"))

    uids: list[str] = []
    y = np.zeros((0, len(TARGETS)), np.float32)
    p = np.zeros((0, len(TARGETS)), np.float32)
    if (path / HOLDOUT).is_file():
        frame = pd.read_csv(path / HOLDOUT, dtype={UID: str})
        uids = frame[UID].tolist()
        y = frame[[f"true:{t}" for t in TARGETS]].to_numpy(np.float32)
        p = frame[[f"pred:{t}" for t in TARGETS]].to_numpy(np.float32)

    return RunRecord(
        experiment=meta.get("experiment", path.name),
        fold=int(meta.get("fold", -1)),
        split=meta.get("split", ""),
        labels=meta.get("labels", ""),
        best_epoch=int(meta.get("best_epoch", -1)),
        best_holdout_auc=float(meta["best_holdout_auc"])
        if meta.get("best_holdout_auc") is not None else float("nan"),
        history=meta.get("epochs", []),
        uids=uids, y=y, p=p,
    )


def read_sweep(root: str | Path) -> list[RunRecord]:
    """Every fold under a directory, ordered by fold.

    Looks one level down as well as at the root itself, so both a sweep directory of
    `pkg-f0/ pkg-f1/ ...` and a single package directory work.
    """

    root = Path(root)
    found = []
    if (root / HISTORY).is_file():
        found.append(read_run_record(root))
    for child in sorted(root.iterdir()) if root.is_dir() else []:
        if child.is_dir() and (child / HISTORY).is_file():
            found.append(read_run_record(child))
    if not found:
        raise FileNotFoundError(
            f"no run record under {root}. A package written before the report existed "
            f"has no {HISTORY}; refit it, or point at a newer sweep.")
    return sorted(found, key=lambda r: r.fold)
