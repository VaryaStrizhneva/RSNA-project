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


#: The Steven tables are near-duplicates of one another (mean Spearman 0.97-0.99
#: across targets), so blending all three mostly averages one extractor with
#: itself and drowns the second opinion. These two are the genuinely distinct
#: readings available (mean Spearman 0.84), and blending them scores the same as
#: the best single source while not staking everything on one extractor.
DISTINCT_SOURCES = ["pilkwang", "steven_v4_blend"]


def default_tables() -> dict[str, pd.DataFrame]:
    """Load the sources that `blend` is meant to be called with."""

    return {s.name: load_label_table(s) for s in LABEL_SOURCES if s.name in DISTINCT_SOURCES}


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


def to_ranks(table: pd.DataFrame) -> pd.DataFrame:
    """Rank each target column to [0, 1], averaging ties.

    Missing values are placed at the column mean rank rather than dropped, so that
    a study a source could not read neither helps nor hurts it in a blend.
    """

    ranks = table.rank(pct=True, na_option="keep")
    return ranks.fillna(ranks.mean())


def blend(
    tables: dict[str, pd.DataFrame],
    weights: dict[str, dict[str, float]] | None = None,
) -> pd.DataFrame:
    """Combine several label tables into one, in rank space.

    Parameters
    ----------
    tables:
        Source name -> label table, as returned by `load_label_table`.
    weights:
        Optional target -> {source: weight}. Missing entries default to equal
        weight over the sources that cover the study. Without weights this is a
        plain mean of ranks, which is the choice that assumes the least.
    """

    if not tables:
        raise ValueError("no tables to blend")

    ranked = {name: to_ranks(table) for name, table in tables.items()}
    index = sorted(set().union(*(frame.index for frame in ranked.values())))

    out = pd.DataFrame(index=pd.Index(index, name="StudyInstanceUID"),
                       columns=TARGET_COLUMNS, dtype=float)

    for target in TARGET_COLUMNS:
        cols, ws = {}, {}
        for name, frame in ranked.items():
            weight = 1.0 if weights is None else weights.get(target, {}).get(name, 0.0)
            if weight == 0.0:
                continue
            # Name each column after its source: several sources contribute the same
            # target, and duplicate column labels would misalign the weighting below.
            cols[name] = frame[target].reindex(index)
            ws[name] = weight

        if not cols:
            raise ValueError(f"no source contributes to {target!r}")

        stacked = pd.DataFrame(cols)
        weight_frame = pd.DataFrame(
            [ws] * len(stacked), index=stacked.index, columns=list(cols)
        ).where(stacked.notna())
        out[target] = (stacked * weight_frame).sum(axis=1) / weight_frame.sum(axis=1)

    return out
