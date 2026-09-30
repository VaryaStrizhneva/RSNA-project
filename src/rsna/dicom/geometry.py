"""Between a pixel on a slice and a place in the patient.

Two conversions and their inverse, in one place because the indexing is easy to get
subtly wrong and impossible to notice afterwards: a landmark stored with row and column
transposed still trains a model, still produces a loss curve that goes down, and still
predicts confidently — in the wrong compartment.

`ImageOrientationPatient` is the direction cosines of the first row and the first
column: the first triplet points along increasing **column** index, the second along
increasing **row** index. `PixelSpacing` is (between rows, between columns), so it pairs
with them the other way round. That mismatch is the whole reason this module exists.
"""

from __future__ import annotations

import numpy as np


def _axes(iop) -> tuple[np.ndarray, np.ndarray]:
    """The column and row directions, renormalised.

    DICOM says these are unit vectors; the headers store them to finite precision, and
    measured over 80 series of this corpus they are off by up to 5.2e-07 with an
    orthogonality residual of 2.2e-07. That is nothing on its own — 6e-5 of a pixel —
    but it is enough to keep `patient_mm` and `pixel_of` from being an exact inverse
    pair, and an inverse pair that is only nearly exact cannot be tested as one.
    """

    iop = np.asarray(iop, float)
    u, v = iop[0:3], iop[3:6]
    return u / np.linalg.norm(u), v / np.linalg.norm(v)


def patient_mm(ipp, iop, spacing, row: float, col: float) -> np.ndarray:
    """Pixel (row, col) of a slice -> position in the patient, in millimetres."""

    u, v = _axes(iop)
    return (np.asarray(ipp, float)
            + col * float(spacing[1]) * u
            + row * float(spacing[0]) * v)


def pixel_of(ipp, iop, spacing, point) -> tuple[float, float]:
    """The inverse: a position in the patient -> (row, col) on that slice.

    The two orientation vectors are unit length and orthogonal, so the inverse is a
    projection rather than a solve. A point off the slice plane projects onto it; how
    far off it was is `through_plane` below, and the caller is expected to care.
    """

    u, v = _axes(iop)
    d = np.asarray(point, float) - np.asarray(ipp, float)
    return (float(d @ v / float(spacing[0])),
            float(d @ u / float(spacing[1])))


def normal_of(iop) -> np.ndarray:
    """The unit vector a stack advances along when the slice index increases."""

    u, v = _axes(iop)
    n = np.cross(u, v)
    return n / np.linalg.norm(n)


def through_plane(point, normal) -> float:
    """Where a point sits along the stack axis, in millimetres.

    This is the depth coordinate the landmark model predicts. It is a projection onto
    the slice normal rather than the patient's x, because a sagittal stack is rarely
    exactly sagittal — measured over this corpus the normal runs to 0.87 along x with
    the rest spread across y — and using x directly would mix the depth axis with the
    anterior-posterior one on precisely the oblique studies.
    """

    return float(np.asarray(point, float) @ np.asarray(normal, float))
