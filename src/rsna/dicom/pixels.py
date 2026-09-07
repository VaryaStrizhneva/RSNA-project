"""One series -> a small stack of comparable images.

Reading is the expensive half of this pipeline — a study is on the order of 150 files
— so the caller reads once at the largest configuration it needs and derives smaller
ones from the buffer rather than re-reading.

Three things happen here that all have to happen, in this order: slices are sampled
across a central band of the *ordered* stack, cropped to a constant physical extent,
and normalised by percentile. Skip any one and the encoder sees images that are not
comparable to each other.
"""

from __future__ import annotations

import os

import numpy as np
import pydicom
import torch
import torch.nn.functional as F

from ..config import Config


def sample_indices(n: int, n_slice: int, band: tuple[float, float]) -> np.ndarray:
    """Which slices of an ordered stack of `n` to read.

    Spread over a central band: the outermost slices of a knee series are mostly soft
    tissue outside the joint. The band is configurable rather than literal because how
    much of the stack is worth reading depends on how many slices are taken — at three
    the middle is all that fits, while at sixteen the ends are worth having, and a
    Baker cyst sits at the posteromedial end of a sagittal stack.
    """

    lo, hi = int(band[0] * (n - 1)), int(band[1] * (n - 1))
    if hi > lo:
        idx = np.unique(np.linspace(lo, hi, n_slice).astype(int))
    else:
        idx = np.array([n // 2])
    while len(idx) < n_slice:
        idx = np.append(idx, idx[-1])
    return idx[:n_slice]


def _read_plane(path: str) -> np.ndarray | None:
    """One slice as float, rescaled, or None when it will not decode."""

    try:
        ds = pydicom.dcmread(path, force=True)
        a = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, "RescaleSlope", 1) or 1)
        intercept = float(getattr(ds, "RescaleIntercept", 0) or 0)
        return a * slope + intercept
    except Exception:
        # Deliberately no shape: see read_slot. Inventing one is how a single
        # unreadable file erases a whole series.
        return None


def read_slot(record: dict, config: Config, n_slice: int | None = None,
              out_size: int | None = None,
              decode_failures: list | None = None) -> torch.Tensor | None:
    """`n_slice` physically spread slices from one series, at `out_size` pixels.

    Returns uint8 `[n_slice, out, out]`, normalised per series to its 1st-99th
    percentile. Percentiles rather than min/max because MR intensity has no absolute
    scale, and a single bright vessel would otherwise compress the whole dynamic range.

    Returns None when nothing in the series decodes — which the presence mask can
    express, unlike a black slot, which it cannot.
    """

    n_slice = config.group if n_slice is None else n_slice
    out_size = config.img if out_size is None else out_size
    files = record.get("ordered") or record["files"]
    directory, spacing = record["dir"], record.get("px")

    if not files:
        return None

    def failed():
        if decode_failures is not None:
            decode_failures.append(record.get("SeriesInstanceUID", directory))

    idx = sample_indices(len(files), n_slice, config.band)
    planes = [_read_plane(os.path.join(directory, files[int(i)])) for i in idx]

    got = [k for k, p in enumerate(planes) if p is not None]

    if config.rules.decode_fill == "zero":
        # What an imported member may have been fitted with: a failure becomes a zero
        # plane at the resize target, which the shape check below then propagates to the
        # whole slot. Kept only because such weights were learned against slots blacked
        # out this way.
        if not got:
            failed()
        planes = [np.zeros((out_size, out_size), np.float32) if p is None else p
                  for p in planes]
        got = list(range(len(planes)))

    if not got:
        failed()
        return None

    if len(got) < len(planes):
        # Fill from the nearest slice that did decode — the same convention the sampler
        # already uses when the band holds fewer distinct slices than were asked for.
        failed()
        for k, plane in enumerate(planes):
            if plane is None:
                planes[k] = planes[min(got, key=lambda j: abs(j - k))]

    # Slices of one series can still differ in matrix size — multi-echo and some
    # reformats do — and those are genuinely not stackable.
    shape = planes[0].shape
    planes = [p if p.shape == shape else np.zeros(shape, np.float32) for p in planes]
    volume = np.stack(planes)

    # Constant physical extent, then resize. PixelSpacing varies 3.4x across the corpus,
    # so a fixed-pixel resize would hand the encoder images whose physical scale differs
    # by that factor.
    if spacing and np.isfinite(spacing) and spacing > 0:
        want = int(round(config.crop_mm / spacing))
        h, w = shape
        if 16 < want < min(h, w):
            cy, cx = h // 2, w // 2
            half = want // 2
            volume = volume[:, max(0, cy - half):cy + half, max(0, cx - half):cx + half]

    lo, hi = np.percentile(volume, [1, 99])
    volume = np.clip((volume - lo) / max(hi - lo, 1e-6), 0, 1)

    tensor = torch.from_numpy(np.ascontiguousarray(volume)).unsqueeze(0)
    tensor = F.interpolate(tensor, size=(out_size, out_size), mode="bilinear",
                           align_corners=False)

    # uint8, not float32. These buffers queue between the reader threads and the
    # encoder, and at this size a float32 slot-series is several megabytes. Intensity is
    # already normalised into [0, 1], so eight bits cost nothing that a bilinear resize
    # has not already cost, and the queue is a quarter the size.
    return (tensor.squeeze(0) * 255).round().clamp(0, 255).to(torch.uint8)
