"""A point in the patient, and the heatmap that stands for it.

The model does not regress three numbers. It writes a probability over a grid and the
coordinates are read back out of it — because a coordinate is a global, non-spatial
output and convolutions are poor at those, while a heatmap gives dense supervision:
every cell of the grid receives a target from a single click. With ~300 annotations
that difference is most of what makes the problem trainable.

This module is the **only** place where millimetres and grid indices meet. Everything
else works in one or the other. A transposed row and column here still trains, still
produces a loss curve that falls, and still predicts confidently into the opposite
compartment — so `encode` and `decode` are written as an exact inverse pair and tested
as one.
"""

from __future__ import annotations

import numpy as np

from ..dicom.geometry import _axes, pixel_of, through_plane
from .config import LandmarkConfig
from .sample import Sampled, to_native, to_resampled


def project(point_mm, s: Sampled) -> tuple[float, float, float]:
    """A patient point -> (through-plane mm, row mm, column mm) in the model's frame.

    The in-plane position does not depend on which slice is used as the origin: the two
    orientation vectors are orthogonal to the normal, so a displacement along the stack
    axis projects to zero on both. Any valid slot gives the same answer, which is why
    the first one is taken rather than the nearest.
    """

    k = int(np.argmax(s.valid))
    row_n, col_n = pixel_of(s.ipp[k], s.iop, s.spacing, point_mm)
    row_r, col_r = to_resampled(s.transform, row_n, col_n)
    mm = s.mm_per_px_rc
    return (through_plane(point_mm, s.normal), row_r * mm[0], col_r * mm[1])


def unproject(t_ipp: np.ndarray, row_mm: float, col_mm: float, s: Sampled) -> np.ndarray:
    """The inverse: an in-plane position in millimetres, on a given plane origin.

    `t_ipp` is a patient origin, not a scalar depth — the decoding averages the origins
    of the slots the heatmap puts weight on, which is exact and needs no assumption that
    the stack advances by pure translation along its normal.
    """

    mm = s.mm_per_px_rc
    row_n, col_n = to_native(s.transform, row_mm / mm[0], col_mm / mm[1])
    u, v = _axes(s.iop)
    return (np.asarray(t_ipp, float)
            + col_n * float(s.spacing[1]) * u
            + row_n * float(s.spacing[0]) * v)


def _cell_centres(s: Sampled, config: LandmarkConfig) -> tuple[np.ndarray, np.ndarray]:
    """Where each heatmap cell sits, in millimetres, along rows and columns."""

    mm = s.mm_per_px_rc
    step = (np.arange(config.grid) + 0.5) * config.stride
    return step * mm[0], step * mm[1]


def encode(points_mm: dict, s: Sampled, config: LandmarkConfig) -> np.ndarray:
    """Landmarks -> (K, slices, grid, grid), a Gaussian in millimetres.

    The width is in millimetres, not in cells, so a study with 2.2 mm between slices and
    one with 5.5 mm get a target of the same physical size — the one place where the
    varying depth spacing has to be handled rather than tolerated.
    """

    out = np.zeros((len(config.points), config.slices, config.grid, config.grid),
                   np.float32)
    rows_mm, cols_mm = _cell_centres(s, config)
    t = np.where(s.valid, s.t_mm, np.nan)

    for k, name in enumerate(config.points):
        p = points_mm.get(name)
        if p is None or not np.all(np.isfinite(p)):
            continue
        tp, rp, cp = project(p, s)
        d2 = ((t - tp) ** 2)[:, None, None] \
            + ((rows_mm - rp) ** 2)[None, :, None] \
            + ((cols_mm - cp) ** 2)[None, None, :]
        g = np.exp(-d2 / (2.0 * config.sigma_mm ** 2))
        out[k] = np.where(np.isfinite(g), g, 0.0)
    return out


def decode(heat: np.ndarray, s: Sampled, config: LandmarkConfig,
           window: float = 3.0) -> tuple[np.ndarray, np.ndarray]:
    """(K, slices, grid, grid) -> K points in patient millimetres, and a confidence.

    A soft-argmax **around the peak**, not over the whole volume. The difference is not
    a refinement: a predicted map has a background floor, and this volume has 196 608
    cells against a peak that spans about 180. At a floor of 0.018 — which is what a
    sigmoid gives at the bias this head starts from — the background carries 99 % of the
    mass and a whole-volume soft-argmax returns the centre of the volume whatever the
    model predicted. Measured: the error sat at 31 mm, worse than predicting the median
    position, while the loss fell by a factor of six.

    So the peak is located first, then averaged over the cells within `window` sigmas of
    it, in millimetres. The depth is recovered by averaging the slots' patient origins
    rather than their indices, so the irregular and study-dependent spacing never enters.

    The confidence is the share of the volume's mass that falls inside that window. A
    model that found nothing spreads its mass everywhere and scores near zero, and a
    branch cropping around a point that was never found should be told to ignore it.
    """

    heat = np.asarray(heat, np.float64)
    rows_mm, cols_mm = _cell_centres(s, config)
    valid = np.asarray(s.valid, bool)
    ipp = np.where(valid[:, None], np.nan_to_num(s.ipp), 0.0)
    t = np.where(valid, np.nan_to_num(s.t_mm), 0.0)

    points = np.full((len(config.points), 3), np.nan)
    conf = np.zeros(len(config.points))
    for k in range(heat.shape[0]):
        w = np.clip(heat[k], 0, None) * valid[:, None, None]
        total = w.sum()
        if total <= 0:
            continue

        # Everything within `window` sigmas of the peak, measured in millimetres so the
        # neighbourhood is the same physical size whatever the slice spacing is.
        ks, ki, kj = np.unravel_index(int(np.argmax(w)), w.shape)
        d2 = ((t - t[ks]) ** 2)[:, None, None] \
            + ((rows_mm - rows_mm[ki]) ** 2)[None, :, None] \
            + ((cols_mm - cols_mm[kj]) ** 2)[None, None, :]
        near = d2 <= (window * config.sigma_mm) ** 2
        w = w * near
        inside = w.sum()
        if inside <= 0:
            continue
        conf[k] = float(inside / total)
        w = w / inside
        origin = np.einsum("s,sc->c", w.sum(axis=(1, 2)), ipp)
        points[k] = unproject(origin,
                              float(w.sum(axis=(0, 2)) @ rows_mm),
                              float(w.sum(axis=(0, 1)) @ cols_mm), s)
    return points, conf
