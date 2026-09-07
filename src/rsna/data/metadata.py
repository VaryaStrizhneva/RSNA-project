from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import unicodedata

import pandas as pd


TARGET_COLUMNS = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]


SERIES_SLOT_SPECS = [
    {"slot": "sagittal_fluid", "Anatomical_Plane": "Sagittal", "Fluid_Sensitive": 1},
    {"slot": "sagittal_nonfluid", "Anatomical_Plane": "Sagittal", "Fluid_Sensitive": 0},
    {"slot": "coronal_fluid", "Anatomical_Plane": "Coronal", "Fluid_Sensitive": 1},
    {"slot": "coronal_nonfluid", "Anatomical_Plane": "Coronal", "Fluid_Sensitive": 0},
    {"slot": "axial_fluid", "Anatomical_Plane": "Axial", "Fluid_Sensitive": 1},
    {"slot": "axial_nonfluid", "Anatomical_Plane": "Axial", "Fluid_Sensitive": 0},
]


@dataclass(frozen=True)
class MetadataBundle:
    """Competition CSVs loaded at study and series level."""

    root: Path
    train: pd.DataFrame
    train_series: pd.DataFrame
    test: pd.DataFrame
    test_series: pd.DataFrame
    sample_submission: pd.DataFrame


def load_metadata(root: str | Path = "data/raw") -> MetadataBundle:
    """Load the five competition metadata CSVs.

    Parameters
    ----------
    root:
        Directory containing train.csv, train_series.csv, test.csv,
        test_series.csv, and sample_submission.csv.
    """

    root = Path(root)
    return MetadataBundle(
        root=root,
        train=_read_csv(root / "train.csv"),
        train_series=_read_csv(root / "train_series.csv"),
        test=_read_csv(root / "test.csv"),
        test_series=_read_csv(root / "test_series.csv"),
        sample_submission=_read_csv(root / "sample_submission.csv"),
    )


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing metadata file: {path}")
    return pd.read_csv(path, dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str})


def clean_report_text(text: object) -> str:
    """Return report text normalized for later label extraction.

    The reports are multilingual. Some local displays may show smth as
    "TÃ©cnica". If the optional ftfy package is installed, use it to repair that
    text; otherwise keep a Unicode-normalized string.
    """

    if pd.isna(text):
        return ""
    value = unicodedata.normalize("NFKC", str(text)).strip()
    try:
        from ftfy import fix_text
    except ImportError:
        return value
    return fix_text(value)


def add_clean_reports(train: pd.DataFrame) -> pd.DataFrame:
    out = train.copy()
    out["Report_clean"] = out["Report"].map(clean_report_text)
    return out


def get_expert_labeled_rows(train: pd.DataFrame) -> pd.DataFrame:
    """Rows where all official target columns are present."""

    missing = [col for col in TARGET_COLUMNS if col not in train.columns]
    if missing:
        raise ValueError(f"Missing target columns in train.csv: {missing}")

    mask = train[TARGET_COLUMNS].notna().all(axis=1)
    return train.loc[mask].copy()


def missing_label_summary(train: pd.DataFrame) -> pd.DataFrame:
    rows = []
    n_rows = len(train)
    for target in TARGET_COLUMNS:
        present = int(train[target].notna().sum())
        rows.append(
            {
                "target": target,
                "labeled": present,
                "missing": int(n_rows - present),
                "labeled_rate": present / n_rows if n_rows else 0.0,
            }
        )
    return pd.DataFrame(rows)


def label_summary(train: pd.DataFrame) -> pd.DataFrame:
    """Count explicit expert labels and positive rates per target."""

    rows = []
    for target in TARGET_COLUMNS:
        values = pd.to_numeric(train[target], errors="coerce")
        labeled = values.dropna()
        positive = int((labeled == 1).sum())
        negative = int((labeled == 0).sum())
        rows.append(
            {
                "target": target,
                "labeled": int(len(labeled)),
                "positive": positive,
                "negative": negative,
                "positive_rate": positive / len(labeled) if len(labeled) else None,
            }
        )
    return pd.DataFrame(rows).sort_values("positive_rate", ascending=False)


def series_summary(series: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Summaries describing the MRI series available per study."""

    series_per_study = series.groupby("StudyInstanceUID").size()
    return {
        "anatomical_plane": _value_counts(series, "Anatomical_Plane"),
        "fluid_sensitive": _value_counts(series, "Fluid_Sensitive"),
        "fat_suppression": _value_counts(series, "Fat_Suppression"),
        "series_per_study_describe": series_per_study.describe().to_frame("value"),
        "series_per_study_counts": series_per_study.value_counts().sort_index().to_frame("n_studies"),
        "plane_by_fluid": _cross_counts(series, "Anatomical_Plane", "Fluid_Sensitive"),
    }


def _value_counts(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    return (
        frame[column]
        .value_counts(dropna=False)
        .sort_index()
        .rename_axis(column)
        .reset_index(name="count")
    )


def _cross_counts(frame: pd.DataFrame, left: str, right: str) -> pd.DataFrame:
    return (
        frame.groupby([left, right], dropna=False)
        .size()
        .rename("count")
        .reset_index()
        .sort_values([left, right])
    )


def build_series_slots(
    series: pd.DataFrame,
    study_ids: Iterable[str] | None = None,
    slot_specs: list[dict[str, object]] | None = None,
) -> pd.DataFrame:
    """Build one selected series per study/slot.

    The result has one row per requested study and slot. If a matching series is
    missing, `present` is False and `SeriesInstanceUID` is NA. If several series
    match a slot, the first one after stable sorting is used for now; the DICOM
    loader can later upgrade this by choosing the series with most valid slices.
    """

    specs = slot_specs or SERIES_SLOT_SPECS
    frame = series.copy()
    frame["StudyInstanceUID"] = frame["StudyInstanceUID"].astype(str)
    frame["SeriesInstanceUID"] = frame["SeriesInstanceUID"].astype(str)

    if study_ids is None:
        ids = sorted(frame["StudyInstanceUID"].unique())
    else:
        ids = [str(study_id) for study_id in study_ids]

    frame = frame.sort_values(["StudyInstanceUID", "Anatomical_Plane", "Fluid_Sensitive", "SeriesInstanceUID"])
    by_study = {study_id: group for study_id, group in frame.groupby("StudyInstanceUID", sort=False)}

    rows = []
    for study_id in ids:
        group = by_study.get(study_id)
        for spec in specs:
            selected = _select_slot(group, spec) if group is not None else None
            rows.append(
                {
                    "StudyInstanceUID": study_id,
                    "slot": spec["slot"],
                    "Anatomical_Plane": spec["Anatomical_Plane"],
                    "Fluid_Sensitive": spec["Fluid_Sensitive"],
                    "present": selected is not None,
                    "SeriesInstanceUID": selected["SeriesInstanceUID"] if selected is not None else pd.NA,
                    "Fat_Suppression": selected.get("Fat_Suppression", pd.NA) if selected is not None else pd.NA,
                }
            )

    return pd.DataFrame(rows)


def _select_slot(group: pd.DataFrame, spec: dict[str, object]) -> pd.Series | None:
    matches = group[
        (group["Anatomical_Plane"] == spec["Anatomical_Plane"])
        & (pd.to_numeric(group["Fluid_Sensitive"], errors="coerce") == int(spec["Fluid_Sensitive"]))
    ]
    if matches.empty:
        return None
    return matches.iloc[0]


def slot_coverage(slots: pd.DataFrame) -> pd.DataFrame:
    """Count how often each slot was present across studies."""

    return (
        slots.groupby("slot")["present"]
        .agg(["sum", "count", "mean"])
        .rename(columns={"sum": "present_count", "count": "study_count", "mean": "present_rate"})
        .reset_index()
        .sort_values("present_rate")
    )
