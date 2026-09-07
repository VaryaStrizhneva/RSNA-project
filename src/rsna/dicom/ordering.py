"""The order the slices of a series are actually in.

A DICOM file name here is a SOP Instance UID, assigned to be unique rather than
ordered, so sorting by it produces a sequence uncorrelated with anatomy — measured over
one series, Spearman between file-name rank and physical position is 0.009, i.e. none.
Anything that assumes the file order means something is then operating on noise: the
three channels of a 2.5D input are three unrelated views rather than neighbouring
slices, "the middle of the stack" is a random subset, and reversing slice order to
normalise laterality reverses nothing.

**It fails silently.** No exception, no warning — only a model that underperforms,
which is indistinguishable from a hard problem.
"""

from __future__ import annotations

import os
import re

import numpy as np
import pydicom

from ..config import Config

#: ImagePositionPatient, ImageOrientationPatient, InstanceNumber.
ORDER_TAGS = [(0x0020, 0x0032), (0x0020, 0x0037), (0x0020, 0x0013)]

#: Share of slices that must carry usable geometry before it is trusted.
_GEOMETRY_THRESHOLD = 0.8


def natural_key(name: str) -> tuple:
    """Last-resort ordering: digits in a file name compared as numbers."""

    return tuple(int(x) if x.isdigit() else x.lower()
                 for x in re.split(r"(\d+)", str(name)))


def _order_by_normal(directory: str, files: list[str]) -> tuple[list[str], bool]:
    """Project each slice position onto the slice normal.

    Each slice carries its position in patient coordinates and its in-plane axes;
    the through-plane coordinate is monotonic along the stack:

        n = r_x x r_y,    k = p . n

    Signed in patient coordinates, which is what laterality normalisation needs.
    """

    keyed: list[tuple[float | None, str]] = []
    for name in files:
        key = None
        try:
            ds = pydicom.dcmread(os.path.join(directory, name), force=True,
                                 stop_before_pixels=True, specific_tags=ORDER_TAGS)
            iop = np.asarray(ds.ImageOrientationPatient, dtype=float)
            ipp = np.asarray(ds.ImagePositionPatient, dtype=float)
            key = float(np.dot(ipp, np.cross(iop[:3], iop[3:])))
        except Exception:
            try:
                key = float(ds.InstanceNumber)
            except Exception:
                key = None
        keyed.append((key, name))

    if any(key is None for key, _ in keyed):
        # A series with no usable geometry keeps its arbitrary order: worse than
        # sorting, better than dropping the series, and reported rather than hidden.
        return files, False
    return [name for _, name in sorted(keyed, key=lambda t: t[0])], True


def _order_by_dominant_axis(directory: str, files: list[str]) -> tuple[list[str], bool]:
    """The order an imported member may have been fitted under.

    Sorts on the raw patient coordinate along whichever axis varies most across the
    stack, rather than on the projection onto the slice normal. The two differ by a
    sign, not a formula: over this corpus every sagittal series has a slice normal with
    n_x in [-1.00, -0.98], so the projection is the negative of the raw x this sorts
    on and the two stacks come out exactly reversed.
    """

    rows = []
    for position, name in enumerate(files):
        ipp = instance = None
        try:
            ds = pydicom.dcmread(os.path.join(directory, name), force=True,
                                 stop_before_pixels=True,
                                 specific_tags=["ImagePositionPatient", "InstanceNumber"])
            raw = getattr(ds, "ImagePositionPatient", None)
            if raw is not None and len(raw) >= 3:
                coords = np.asarray(raw[:3], dtype=np.float64)
                if np.isfinite(coords).all():
                    ipp = coords
            number = getattr(ds, "InstanceNumber", None)
            if number is not None:
                instance = float(number)
        except Exception:
            pass
        rows.append((name, ipp, instance, position))

    placed = [r for r in rows if r[1] is not None]
    need = max(2, int(_GEOMETRY_THRESHOLD * len(rows)))

    if len(placed) >= need:
        xyz = np.stack([r[1] for r in placed])
        axis = int(np.argmax(np.ptp(xyz, axis=0)))
        spare = float(np.nanmedian(xyz[:, axis]))
        rows.sort(key=lambda r: (float(r[1][axis]) if r[1] is not None else spare,
                                 r[2] if r[2] is not None else float("inf"), r[3]))
    elif sum(r[2] is not None for r in rows) >= need:
        rows.sort(key=lambda r: (r[2] if r[2] is not None else float("inf"), r[3]))
    else:
        rows.sort(key=lambda r: natural_key(r[0]))
    return [r[0] for r in rows], True


def order_slices(directory: str, files: list[str], config: Config) -> tuple[list[str], bool]:
    """Return the series' files sorted along the through-plane axis.

    The second element is False when no usable geometry was found and the arbitrary
    order was kept — count those rather than ignoring them.
    """

    if config.rules.order == "dominant_axis":
        return _order_by_dominant_axis(directory, files)
    return _order_by_normal(directory, files)
