"""Which knee was scanned, and how to put every study on one convention.

Five of the twelve targets are named for a side — the two menisci, the two
tibiofemoral compartments, and the medial collateral ligament. Medial and lateral are
defined relative to the body midline, so which side of the *image* they fall on
depends on which knee was scanned. Unless that is normalised, those five targets see
their axis reversed on a large minority of the corpus and learn noise, while the seven
side-agnostic targets look fine — so the pipeline appears to work.

`Laterality` (0020,0060) is Type 2C and may legitimately be absent. In this corpus it
is missing on exactly half the studies, and by whole vendors rather than scattered
series, so treating an untagged study as left-sided leaves half the corpus
un-normalised.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Config
from .headers import hdr_vec


def side_from_geometry(headers: pd.DataFrame, config: Config) -> dict[str, str | None]:
    """Study -> 'L' / 'R' / None, from where the image sits in the patient.

    DICOM patient coordinates are LPS: +x is the patient's left, so the centre of a
    right knee sits at negative x. The **centre** is used rather than
    `ImagePositionPatient` itself, which is the corner of the image and is offset by
    half a field of view — enough to change the sign on a knee near the midline.

    The median over a study's series is thresholded, not a single series: one header
    is read per series, and on a sagittal stack that slice can sit anywhere across the
    joint. Studies whose centre falls inside `lat_min_offset_mm` of the midline are
    left unresolved rather than guessed — measured against the tagged half, the rule
    is right 97% of the time overall and no better than chance inside 20 mm.
    """

    centres: dict[str, list[float]] = {}
    for row in headers.itertuples(index=False):
        ipp = hdr_vec(getattr(row, "ImagePositionPatient", None), 3)
        iop = hdr_vec(getattr(row, "ImageOrientationPatient", None), 6)
        spacing = hdr_vec(getattr(row, "PixelSpacing", None), 2)
        rows, cols = getattr(row, "Rows", None), getattr(row, "Columns", None)
        if ipp is None or iop is None or spacing is None or not rows or not cols:
            continue
        try:
            centre = (ipp[:3]
                      + iop[:3] * spacing[1] * float(cols) / 2
                      + iop[3:6] * spacing[0] * float(rows) / 2)
        except (TypeError, ValueError):
            continue
        centres.setdefault(row.StudyInstanceUID, []).append(float(centre[0]))

    out: dict[str, str | None] = {}
    for study, xs in centres.items():
        median = float(np.median(xs))
        if abs(median) < config.lat_min_offset_mm:
            out[study] = None
        else:
            out[study] = "R" if median < 0 else "L"
    return out


def side_from_corner_x(headers: pd.DataFrame, config: Config) -> dict[str, str | None]:
    """The laterality rule an imported member may have been fitted under.

    Thresholds the median raw `ImagePositionPatient` x — the image *corner*, not its
    centre — with a 5 mm dead zone instead of 20 mm, so it also commits on studies the
    centre rule leaves unresolved.

    Neither difference changes a shape. Each decides whether a study is mirrored, and a
    study mirrored one way at training and the other at inference presents the five
    side-defined targets with their axis reversed.
    """

    out: dict[str, str | None] = {}
    for study, group in headers.groupby("StudyInstanceUID"):
        xs = []
        for row in group.itertuples(index=False):
            ipp = hdr_vec(getattr(row, "ImagePositionPatient", None), 3)
            if ipp is not None and np.isfinite(ipp).all():
                xs.append(float(ipp[0]))
        if not xs:
            out[study] = None
            continue
        x = float(np.median(xs))
        out[study] = None if abs(x) < config.lat_legacy_offset_mm else ("R" if x < 0 else "L")
    return out


def laterality_of(headers: pd.DataFrame, config: Config) -> tuple[dict[str, str | None], dict[str, int]]:
    """Study -> side: the tag where it exists, geometry where it does not.

    Returns the mapping and a small tally, which is worth logging: if geometry and the
    tag disagree often, one of them is wrong and every side-defined target is affected.
    """

    if config.rules.laterality == "corner_x":
        geometric = side_from_corner_x(headers, config)
    else:
        geometric = side_from_geometry(headers, config)

    sides: dict[str, str | None] = {}
    stats = {"from_tag": 0, "from_geometry": 0, "unresolved": 0, "disagree": 0}

    for study, group in headers.groupby("StudyInstanceUID"):
        tagged = [str(x).strip().upper() for x in group["Laterality"].dropna()]
        if config.rules.laterality == "corner_x" and "ImageLaterality" in group.columns:
            tagged += [str(x).strip().upper() for x in group["ImageLaterality"].dropna()]
        # The tag is sometimes an empty string rather than absent, which is not NaN.
        tagged = [x[0] for x in tagged if x and x[0] in ("L", "R")]

        side = tagged[0] if tagged else None
        if side is not None:
            stats["from_tag"] += 1
            if geometric.get(study) is not None and geometric[study] != side:
                stats["disagree"] += 1
        else:
            side = geometric.get(study)
            stats["from_geometry"] += side is not None
            stats["unresolved"] += side is None
        sides[study] = side

    return sides, stats


def normalise_laterality(image: np.ndarray, plane: str, side: str | None) -> np.ndarray:
    """Map every knee onto a left-knee convention.

    Coronal and axial views mirror under a horizontal flip. Sagittal stacks do not:
    their left-right image axis is the through-plane direction, so mirroring them is
    handled by the slice order, not here.
    """

    if side != "R":
        return image
    if plane in ("Coronal", "Axial"):
        return image[..., ::-1]
    return image
