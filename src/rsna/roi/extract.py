"""A landmark in patient millimetres, a series, and the crop that follows.

Three things here are easy to get wrong and impossible to notice afterwards.

**The slice is found by millimetres, never by an index.** An annotation records the
index of the stack the annotator scrolled, which on a 3D series is a subsampled one —
five slots out of three hundred and twenty. Using that index on the acquired stack lands
a hundred slices away, outside the joint, on an image that still looks like a knee. The
point's own coordinates do not have that problem.

**The depth window is asymmetric, and its direction comes from the pixels.** The
landmark sits a slice or two off the bowtie, so there is much less meniscus toward the
periphery than toward the notch. Which way that is can be read off the volume: the side
where the imaged knee runs out sooner is the lateral one. Measured against the known
laterality on 298 annotated studies, that rule is right **298 times out of 298**, with a
worst-case separation of 1.9x — so an asymmetric crop needs no `Laterality` tag.

**Nothing is interpolated between slices.** Where the spacing is coarse there are fewer
slices than slots and the rest is padding; where it is fine there are more and some are
dropped. The same rule the landmark sampler follows, for the same reason.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..dicom.geometry import normal_of, pixel_of, through_plane
from .config import RoiSpec


@dataclass(frozen=True)
class RoiStack:
    """One study's ROI for one series, as a branch receives it."""

    volume: np.ndarray        #: (slots, out_h, out_w) uint8 — padding is zero
    valid: np.ndarray         #: (slots,) bool
    t_mm: np.ndarray          #: (slots,) through-plane position, NaN where padded
    sop: list                 #: SOPInstanceUID per slot, None where padded
    toward_bowtie: float      #: +1 or -1, the direction of the peripheral end
    native_spacing_mm: float
    stride: int               #: 1 unless the series was thinned
    centre_slot: int          #: the slot holding the slice nearest the landmark


def bowtie_direction(volume: np.ndarray, t: np.ndarray,
                     t_point: float, floor: float = 0.55) -> tuple[float, float]:
    """Which way the peripheral end lies, and by how clear a margin.

    Returns +1 when the bowtie is toward increasing `t`. The rule is that the imaged
    knee ends sooner on the lateral side — median 19.8 mm from the landmark against
    74.5 mm toward the medial edge, a factor of 3.8 — so the shorter side wins. The
    margin comes back with it: a ratio near 1 means the study did not answer, and a
    caller that wants to be careful can refuse it rather than crop backwards.
    """

    area = (volume > 40).mean(axis=(1, 2))
    lit = area >= floor * area.max()
    if lit.sum() < 2:
        return 1.0, 1.0
    ts = np.asarray(t)[lit]
    up, down = abs(ts.max() - t_point), abs(ts.min() - t_point)
    lo = max(min(up, down), 1e-6)
    return (1.0 if up < down else -1.0), max(up, down) / lo


def choose_slices(t: np.ndarray, t_point: float, toward_bowtie: float,
                  spec: RoiSpec) -> np.ndarray:
    """Indices of the acquired slices the ROI keeps, in stack order.

    Everything inside the asymmetric window; thinned by even spacing when more than
    `slots` fall in it. Thinning drops slices, it never averages or interpolates them.
    """

    d = (np.asarray(t) - t_point) * toward_bowtie
    inside = np.flatnonzero((d >= -spec.medial_mm - 1e-6) & (d <= spec.lateral_mm + 1e-6))
    if len(inside) > spec.slots:
        pick = {int(round(x)) for x in np.linspace(0, len(inside) - 1, spec.slots)}
        inside = inside[sorted(pick)]
    return inside


def crop_plane(image: np.ndarray, row: float, col: float, spacing,
               spec: RoiSpec) -> np.ndarray:
    """The box around (row, col), at the spec's fixed millimetres per pixel."""

    import cv2

    hw = spec.box_w_mm / 2 / float(spacing[1])
    hh = spec.box_h_mm / 2 / float(spacing[0])
    r0, r1 = int(round(row - hh)), int(round(row + hh))
    c0, c1 = int(round(col - hw)), int(round(col + hw))

    # Pad rather than clamp: a box that runs off the acquisition keeps its physical
    # size and gains a black margin, so a millimetre is the same number of pixels on
    # every study.
    canvas = np.zeros((r1 - r0, c1 - c0), image.dtype)
    sr0, sc0 = max(r0, 0), max(c0, 0)
    sr1, sc1 = min(r1, image.shape[0]), min(c1, image.shape[1])
    if sr1 > sr0 and sc1 > sc0:
        canvas[sr0 - r0:sr1 - r0, sc0 - c0:sc1 - c0] = image[sr0:sr1, sc0:sc1]

    interp = cv2.INTER_AREA if canvas.shape[1] >= spec.out_w else cv2.INTER_LINEAR
    return cv2.resize(canvas, (spec.out_w, spec.out_h), interpolation=interp)


def extract(volume: np.ndarray, geometry: list[dict], point_mm,
            spec: RoiSpec) -> RoiStack:
    """Crop one series around one landmark.

    `geometry` is one dict per slice of `volume`, in the same order, each carrying
    `sop`, `ipp`, `iop` and `ps`. `volume` must already be windowed to uint8.
    """

    usable = [i for i, g in enumerate(geometry) if g and g.get("ipp") is not None]
    if not usable:
        raise ValueError("no slice carries the geometry a crop needs")

    iop = np.asarray(geometry[usable[0]]["iop"], float)
    ps = np.asarray(geometry[usable[0]]["ps"], float)
    n = normal_of(iop)
    point = np.asarray(point_mm, float)
    t_point = through_plane(point, n)
    t = np.array([through_plane(g["ipp"], n) if g and g.get("ipp") is not None else np.nan
                  for g in geometry])

    steps = np.abs(np.diff(t[np.isfinite(t)]))
    native = float(np.median(steps)) if len(steps) else float("nan")
    stride = 1
    if np.isfinite(native) and 0 < native < spec.decimate_to_mm / 2:
        stride = max(1, int(round(spec.decimate_to_mm / native)))
    thinned = np.arange(0, len(geometry), stride)

    toward, _ = bowtie_direction(volume[thinned], t[thinned], t_point)
    keep = thinned[choose_slices(t[thinned], t_point, toward, spec)]

    # In-plane position does not depend on which slice supplies the origin: both
    # orientation vectors are orthogonal to the normal, so a displacement along the
    # stack projects to zero on each.
    row, col = pixel_of(geometry[usable[0]]["ipp"], iop, ps, point)

    out = np.zeros((spec.slots, spec.out_h, spec.out_w), np.uint8)
    valid = np.zeros(spec.slots, bool)
    t_mm = np.full(spec.slots, np.nan)
    sop: list = [None] * spec.slots
    start = (spec.slots - len(keep)) // 2
    centre = start + int(np.argmin(np.abs(t[keep] - t_point))) if len(keep) else 0

    for j, i in enumerate(keep):
        slot = start + j
        out[slot] = crop_plane(volume[i], row, col, ps, spec)
        valid[slot] = True
        t_mm[slot] = t[i]
        sop[slot] = geometry[i].get("sop")

    return RoiStack(volume=out, valid=valid, t_mm=t_mm, sop=sop, toward_bowtie=toward,
                    native_spacing_mm=native, stride=stride, centre_slot=centre)
