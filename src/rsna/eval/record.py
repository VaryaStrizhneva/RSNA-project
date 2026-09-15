"""What a training run leaves behind for a report to read.

Three files beside the weights, all text, all small:

* `history.json` — one entry per epoch, plus what the run was.
* `holdout.csv` — the held-out studies, their targets and the predictions of the
  selected epoch.
* `gold.csv` — the same for the expert-labelled studies, when `--holdout-gold` kept
  them out of training. Only there when that was asked for.

Everything a report says is derived from these. That is the point: a report needs no
GPU, no pixel cache and no checkpoint, so it can be regenerated after a config change,
on another machine, or a month later, and it always says the same thing about the same
run. The alternative — re-running inference to draw a chart — costs a forward pass per
fold and quietly reports a *different* model whenever the code around it has moved.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import TARGETS

HISTORY = "history.json"
HOLDOUT = "holdout.csv"
GOLD = "gold.csv"
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
    #: The expert-labelled studies, when `--holdout-gold` kept them out of training.
    #: Held apart from the fields above because they are a different measurement: the
    #: same studies in every fold, and compared against the truth rather than against
    #: a label table.
    gold_uids: list[str] = field(default_factory=list)
    gold_y: np.ndarray | None = None
    gold_p: np.ndarray | None = None

    @property
    def n(self) -> int:
        return len(self.uids)

    @property
    def n_gold(self) -> int:
        return len(self.gold_uids)


def _finite(x) -> float | None:
    """NaN is not JSON. An unmeasurable score is null, which is what it means."""

    x = float(x)
    return x if np.isfinite(x) else None


def _predictions(frame_uids: list[str], truth, pred) -> "pd.DataFrame":
    frame = pd.DataFrame({UID: frame_uids})
    for j, target in enumerate(TARGETS):
        frame[f"true:{target}"] = np.asarray(truth)[:, j]
        frame[f"pred:{target}"] = np.asarray(pred)[:, j]
    return frame


def write_run_record(path: str | Path, experiment: str, fold: int, split: str,
                     labels: str, result, uids: list[str],
                     gold_uids: list[str] | None = None) -> Path:
    """Write `history.json`, `holdout.csv` and, when there is one, `gold.csv`."""

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
        _predictions(uids, result.holdout_true,
                     result.holdout_pred).to_csv(path / HOLDOUT, index=False)

    gold_pred = getattr(result, "gold_pred", None)
    gold_true = getattr(result, "gold_true", None)
    if gold_uids and gold_pred is not None and gold_true is not None:
        _predictions(list(gold_uids), gold_true,
                     gold_pred).to_csv(path / GOLD, index=False)
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

    gold_uids: list[str] = []
    gold_y = gold_p = None
    if (path / GOLD).is_file():
        frame = pd.read_csv(path / GOLD, dtype={UID: str})
        gold_uids = frame[UID].tolist()
        gold_y = frame[[f"true:{t}" for t in TARGETS]].to_numpy(np.float32)
        gold_p = frame[[f"pred:{t}" for t in TARGETS]].to_numpy(np.float32)

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
        gold_uids=gold_uids, gold_y=gold_y, gold_p=gold_p,
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
