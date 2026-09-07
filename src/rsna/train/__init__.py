"""Fitting: folds, targets, augmentation, the loop.

The public baseline has all of this inline inside a 9,000-character `main()`, mixed
with orchestration. The reasoning is ported; the structure is ours.
"""

from .augment import augment, take_group
from .folds import assign_folds, build_targets, fold_report
from .loop import EpochResult, FitResult, fit, macro_auc, predict

__all__ = [
    "augment", "take_group",
    "assign_folds", "build_targets", "fold_report",
    "EpochResult", "FitResult", "fit", "macro_auc", "predict",
]
