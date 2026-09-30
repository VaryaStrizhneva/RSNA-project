"""Reading a study the way a radiologist would, not the way the model does.

The training cache keeps twelve slices from the middle 60% of each series, cropped to
130 mm and resampled to 336 px. That is a deliberate reduction, and it is exactly what
an atlas must not inherit: a finding thrown away by the band would look, in the atlas,
like a finding that is not there.

So everything here goes to the DICOM. Full stack, every slice, native resolution, no
crop. The only thing reused from the package is the slice *ordering*, because getting
that wrong would put the slices in an order no anatomy explains — and that code is
already tested.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from rsna.config import Config                      # noqa: E402
from rsna.dicom.headers import annotate             # noqa: E402
from rsna.dicom.ordering import order_slices        # noqa: E402

ROOT = Path("/data/mgr/rsna-knee/extracted")
TRAIN_SERIES = ROOT / "train_series"

#: Everything the atlas wants to say about a series in its caption.
HEADER_TAGS = ["SeriesDescription", "SequenceName", "ScanOptions", "ScanningSequence",
               "RepetitionTime", "EchoTime", "PixelSpacing", "SliceThickness",
               "Rows", "Columns", "Laterality", "ImagePositionPatient",
               "ImageOrientationPatient"]


@dataclass
class Series:
    """One acquired series, in the order the scanner laid it down in the patient."""

    uid: str
    plane: str
    weight: str          # T1 / T2 / PD / GRE / UNK, recovered from the header
    fatsat: bool
    volume: np.ndarray   # (slice, row, col) float32, rescaled, NOT normalised
    mm_per_px: float
    thickness: float
    description: str
    tr: float | None
    te: float | None
    ordered: bool        # False when no usable geometry and the file order was kept
    positions: list[float]   # the through-plane coordinate of each slice

    @property
    def n(self) -> int:
        return len(self.volume)

    @property
    def label(self) -> str:
        fs = " FS" if self.fatsat else ""
        return f"{self.plane} {self.weight}{fs}"

    @property
    def width_mm(self) -> float:
        return self.volume.shape[-1] * self.mm_per_px


def _tag(ds, name, default=None):
    v = getattr(ds, name, default)
    if isinstance(v, (list, pydicom.multival.MultiValue)):
        return "|".join(str(x) for x in v)
    return v


def series_headers(study: str) -> pd.DataFrame:
    """One header per series of a study, annotated with the recovered axes.

    The competition's own `Fluid_Sensitive` and `Fat_Suppression` columns are equal on
    every training series, so they carry one axis rather than two; `annotate` recovers
    both from TR, TE and the description. See docs/data.md.
    """

    rows = []
    directory = TRAIN_SERIES / study
    for series in sorted(directory.iterdir()):
        files = sorted(series.glob("*.dcm"))
        if not files:
            continue
        ds = pydicom.dcmread(str(files[0]), force=True, stop_before_pixels=True)
        row = {"StudyInstanceUID": study, "SeriesInstanceUID": series.name,
               "n_slices": len(files), "dir": str(series)}
        row.update({t: _tag(ds, t) for t in HEADER_TAGS})
        rows.append(row)

    table = annotate(pd.DataFrame(rows))
    plane = pd.read_csv(ROOT / "train_series.csv")
    table["plane"] = table["SeriesInstanceUID"].map(
        dict(zip(plane["SeriesInstanceUID"], plane["Anatomical_Plane"])))
    return table


def load_series(row, config: Config | None = None) -> Series:
    """Every slice of one series, ordered along the through-plane axis.

    No band, no crop, no resampling: `sample_indices` and the 130 mm crop are training
    decisions, and an atlas that repeated them could not show what they discard.
    """

    config = config or Config()
    directory = Path(row["dir"])
    names = sorted(p.name for p in directory.glob("*.dcm"))
    names, ordered = order_slices(str(directory), names, config)

    planes, positions = [], []
    for name in names:
        try:
            ds = pydicom.dcmread(str(directory / name), force=True)
            a = ds.pixel_array.astype(np.float32)
            slope = float(getattr(ds, "RescaleSlope", 1) or 1)
            intercept = float(getattr(ds, "RescaleIntercept", 0) or 0)
            planes.append(a * slope + intercept)
            positions.append(_through_plane(ds))
        except Exception:
            continue

    if not planes:
        raise ValueError(f"{directory} decoded no slices")

    shape = planes[0].shape
    planes = [p for p in planes if p.shape == shape]
    spacing = row.get("px")
    if spacing is None or not np.isfinite(spacing):
        spacing = 0.33

    return Series(
        uid=row["SeriesInstanceUID"], plane=str(row.get("plane", "?")),
        weight=str(row.get("weight", "UNK")), fatsat=bool(row.get("fatsat", False)),
        volume=np.stack(planes), mm_per_px=float(spacing),
        thickness=float(row.get("SliceThickness") or 3.0),
        description=str(row.get("SeriesDescription") or ""),
        tr=_float(row.get("RepetitionTime")), te=_float(row.get("EchoTime")),
        ordered=ordered, positions=positions[:len(planes)])


def _float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _through_plane(ds) -> float:
    """The coordinate the stack is ordered on: position projected on the slice normal."""

    try:
        iop = np.asarray(ds.ImageOrientationPatient, dtype=float)
        ipp = np.asarray(ds.ImagePositionPatient, dtype=float)
        return float(np.dot(ipp, np.cross(iop[:3], iop[3:])))
    except Exception:
        return float("nan")


def patient_x(ds_or_row) -> float | None:
    """The x of the image's top-left **corner**, not of the knee.

    Kept because it is what an imported member may have been fitted under, and because
    `rsna.dicom.laterality.side_from_corner_x` names the same quantity. It is the wrong
    thing to threshold: the corner sits half a field of view from the centre, which is
    about 90 mm here, and that is more than enough to put a knee on the wrong side of
    zero. Measured on one study of this corpus, the corner gives -24 mm and the centre
    +66 mm — a left knee called right. `side_of` uses the centre.
    """

    try:
        return float(str(ds_or_row["ImagePositionPatient"]).split("|")[0])
    except Exception:
        return None


def sagittal_orientation(series: Series | None, side: str | None) -> tuple[str, str]:
    """Which end of a sagittal stack is medial, and which is lateral.

    DICOM patient coordinates are LPS: +x is the patient's **left**. A left knee
    therefore sits at positive x and a right knee at negative x, and *medial* means
    "toward the midline" — toward smaller x on the left, toward larger x on the right.

    `order_slices` sorts on the projection onto the slice normal, and over this corpus
    every sagittal normal points along -x, so every stack runs in order of **decreasing
    patient x** regardless of which knee it is. Measured directly on 600 sagittal series
    drawn at random: n_x is negative on **600 of 600**, median -0.991, worst -0.871. The
    0.8% that are frankly oblique keep the sign, so the medial-to-lateral order survives
    even where the plane is tilted. The consequence is that the anatomical
    meaning of "slice 0" is **reversed between left and right knees**, and nothing in
    the training pipeline puts it back: `normalise_laterality` mirrors coronal and axial
    images and returns sagittal ones untouched.

    That does not affect a model that averages over the stack, which is why it has gone
    unnoticed. It matters here, because an atlas that labelled the wrong end "lateral"
    would teach the wrong thing.
    """

    if side == "L":
        return "lateral", "medial"
    if side == "R":
        return "medial", "lateral"
    return "?", "?"


def stack_orientation(plane: str, side: str | None) -> tuple[str, str]:
    """What the two ends of a stack are, anatomically, for any plane.

    Measured on this corpus rather than assumed: a coronal stack runs in order of
    **increasing y** and an axial one of **increasing z**, and in LPS that is anterior
    to posterior and inferior to superior. Both are the same on a left and a right knee
    — only the sagittal axis flips, because only the sagittal through-plane direction
    is the left-right one, and left-right is the axis "medial" is defined against.
    """

    if plane == "Sagittal":
        return sagittal_orientation(None, side)
    if plane == "Coronal":
        return "anterior", "posterior"
    if plane == "Axial":
        return "inferior", "superior"
    return "?", "?"


def side_of(headers: pd.DataFrame) -> tuple[str | None, str]:
    """Which knee, and where that was established from.

    Delegates the geometry to `rsna.dicom.laterality.side_from_geometry` rather than
    thresholding an x of its own. It did the latter once, on the image **corner**, and
    the corpus contains a study where that reads -24 mm while the centre reads +66 mm
    and the tibia measures +84 mm on the axial: a left knee shown as right, annotated on
    the strength of the badge, and landing on the opposite meniscus. One rule, the one
    that was measured.
    """

    tagged = [str(x).strip().upper()[:1] for x in headers["Laterality"].dropna()]
    tagged = [x for x in tagged if x in ("L", "R")]
    if tagged:
        return tagged[0], "DICOM Laterality tag"

    from rsna.config import Config
    from rsna.dicom.laterality import side_from_geometry

    study = str(headers["StudyInstanceUID"].iloc[0]) if "StudyInstanceUID" in headers \
        else "_"
    frame = headers if "StudyInstanceUID" in headers \
        else headers.assign(StudyInstanceUID=study)
    side = side_from_geometry(frame, Config()).get(study)
    if side is None:
        return None, "geometry gives no answer away from the midline"
    return side, "geometry (centre of the image)"
