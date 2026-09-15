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
    #: The score this table gives a finding its report never mentions — its "I do not
    #: know" value, which is *not* the middle of the range and differs between tables.
    #:
    #: Measured, not assumed: pilkwang publishes a `__verdict` column, so the cells
    #: where a report is silent are known exactly; each table's value below is the mode
    #: of what it assigns to those same cells. The agreement is noted because the two
    #: extractors do not draw the line in quite the same place.
    #:
    #: A weighting that decides how hard a silence should pull needs this number, and
    #: it belongs to the table rather than to a run — repeating it in each experiment
    #: is how two runs come to disagree about a property of the same file.
    silence: float | None = None


LABEL_SOURCES = [
    LabelSource(
        "pilkwang",
        EXTERNAL_ROOT / "pilkwang-rsna-knee-llm-labels" / "report_labels_v2.csv",
        contaminated=False,
        silence=0.28,  # 100% — it is the UNK score itself
    ),
    LabelSource(
        "steven_v4_blend",
        EXTERNAL_ROOT / "stevenleehans-rsna-knee-llm-report-labels" / "llm_labels_v4_blend.csv",
        contaminated=False,
        silence=0.25,  # 63% of pilkwang's silent cells
    ),
    LabelSource(
        "steven_v2",
        EXTERNAL_ROOT / "stevenleehans-rsna-knee-llm-report-labels" / "llm_labels_v2.csv",
        contaminated=False,
        silence=0.50,  # 63%
    ),
    LabelSource(
        "steven_full",
        EXTERNAL_ROOT / "stevenleehans-rsna-knee-llm-report-labels" / "llm_labels_full.csv",
        contaminated=False,
        silence=0.50,  # 85%
    ),
    LabelSource(
        "yunus_merged",
        EXTERNAL_ROOT / "yunusgmsoy-rsna-knee-llm-labels-4-source-merged" / "report_labels_v5.csv",
        contaminated=True,
        silence=0.246,  # 52%; a four-source mean, so less sharply defined
    ),
]


def silence_of(source: LabelSource | str | Path) -> float | None:
    """The table's "the report does not say" score, or None if we have not measured it.

    Matched on the file name rather than the full path so that a table mounted
    somewhere else — a Kaggle input, another volume — is still recognised.
    """

    if isinstance(source, LabelSource):
        return source.silence
    name = Path(source).name
    for known in LABEL_SOURCES:
        if known.path.name == name:
            return known.silence
    return None


def load_label_frame(source: LabelSource | str | Path) -> pd.DataFrame:
    """Read a raw label-source CSV, indexed by study id."""

    path = source.path if isinstance(source, LabelSource) else Path(source)
    frame = pd.read_csv(path, dtype={"StudyInstanceUID": str})
    if "StudyInstanceUID" not in frame.columns:
        raise ValueError(f"{path} is missing StudyInstanceUID")
    return frame.set_index("StudyInstanceUID")


def load_label_table(source: LabelSource | str | Path) -> pd.DataFrame:
    """Read one label table, keeping the study id and the twelve targets.

    Returns a frame indexed by StudyInstanceUID. Targets are coerced to numeric;
    a source that leaves a study out is simply absent from the index rather than
    filled, so callers can see the gap.
    """

    path = source.path if isinstance(source, LabelSource) else Path(source)
    frame = load_label_frame(source)

    missing = [col for col in TARGET_COLUMNS if col not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing target columns: {missing}")

    out = frame[TARGET_COLUMNS]
    return out.apply(pd.to_numeric, errors="coerce")


def load_confidence_table(source: LabelSource | str | Path) -> pd.DataFrame | None:
    """Read per-target confidence columns when a source provides them.

    The returned columns are renamed to the plain target names so callers can pass
    the frame directly next to `load_label_table`. Sources without confidence
    columns return None and are treated as uniformly weighted weak labels.
    """

    path = source.path if isinstance(source, LabelSource) else Path(source)
    frame = load_label_frame(source)
    columns = {target: f"{target}__conf" for target in TARGET_COLUMNS}
    present = [column for column in columns.values() if column in frame.columns]
    if not present:
        return None

    missing = [column for column in columns.values() if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} has partial confidence columns; missing: {missing}")

    out = frame[[columns[target] for target in TARGET_COLUMNS]].copy()
    out.columns = TARGET_COLUMNS
    return out.apply(pd.to_numeric, errors="coerce").clip(0.0, 1.0)


def load_verdict_table(source: LabelSource | str | Path) -> pd.DataFrame | None:
    """Read YES/NO/UNK verdict columns when a source provides them."""

    path = source.path if isinstance(source, LabelSource) else Path(source)
    frame = load_label_frame(source)
    columns = {target: f"{target}__verdict" for target in TARGET_COLUMNS}
    present = [column for column in columns.values() if column in frame.columns]
    if not present:
        return None

    missing = [column for column in columns.values() if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} has partial verdict columns; missing: {missing}")

    out = frame[[columns[target] for target in TARGET_COLUMNS]].copy()
    out.columns = TARGET_COLUMNS
    return out.astype("string")

