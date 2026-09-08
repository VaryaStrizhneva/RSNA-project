"""Report-derived label tables: loading, ranking, and blending.

Only a handful of the 4,407 training studies carry expert labels, so the targets
used for training are derived from the radiology reports. Several such tables have
been published; this module treats them as interchangeable *sources* and combines
them.

Everything here works in **rank space**. The competition metric is a macro AUC,
which is invariant under any strictly increasing transform of a column, so the
absolute values a source emits carry no information beyond the order they induce.
Ranking also puts sources on a common scale: some emit five distinct values, others
two thousand.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .metadata import TARGET_COLUMNS

EXTERNAL_ROOT = Path("data/external")


@dataclass(frozen=True)
class LabelSource:
    """One published table of report-derived labels."""

    name: str
    path: Path
    #: True when the table reproduces the official expert labels on the annotated
    #: studies. Such a table may be trained on, but scoring it against those same
    #: studies measures nothing.
    contaminated: bool


LABEL_SOURCES = [
    LabelSource(
        "pilkwang",
        EXTERNAL_ROOT / "pilkwang-rsna-knee-llm-labels" / "report_labels_v2.csv",
        contaminated=False,
    ),
    LabelSource(
        "steven_v4_blend",
        EXTERNAL_ROOT / "stevenleehans-rsna-knee-llm-report-labels" / "llm_labels_v4_blend.csv",
        contaminated=False,
    ),
    LabelSource(
        "steven_v2",
        EXTERNAL_ROOT / "stevenleehans-rsna-knee-llm-report-labels" / "llm_labels_v2.csv",
        contaminated=False,
    ),
    LabelSource(
        "steven_full",
        EXTERNAL_ROOT / "stevenleehans-rsna-knee-llm-report-labels" / "llm_labels_full.csv",
        contaminated=False,
    ),
    LabelSource(
        "yunus_merged",
        EXTERNAL_ROOT / "yunusgmsoy-rsna-knee-llm-labels-4-source-merged" / "report_labels_v5.csv",
        contaminated=True,
    ),
]


def load_label_table(source: LabelSource | str | Path) -> pd.DataFrame:
    """Read one label table, keeping the study id and the twelve targets.

    Returns a frame indexed by StudyInstanceUID. Targets are coerced to numeric;
    a source that leaves a study out is simply absent from the index rather than
    filled, so callers can see the gap.
    """

    path = source.path if isinstance(source, LabelSource) else Path(source)
    frame = pd.read_csv(path, dtype={"StudyInstanceUID": str})

    missing = [col for col in TARGET_COLUMNS if col not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing target columns: {missing}")

    out = frame.set_index("StudyInstanceUID")[TARGET_COLUMNS]
    return out.apply(pd.to_numeric, errors="coerce")

