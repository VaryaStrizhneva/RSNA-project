"""The pixel path, replayed with every intermediate kept.

`read_slot` returns only its result, so illustrating what it does means running the
chain a second time. Two implementations of one computation is a divergence waiting to
happen, which is what `verify_against_read_slot` exists to prevent: it asserts the
replay ends exactly where the real one ends, and every figure calls it first.
"""

from __future__ import annotations

import os

import numpy as np
import pydicom
import torch
import torch.nn.functional as F

from ..config import Config
from ..dicom.ordering import order_slices
from ..dicom.pixels import read_slot, sample_indices


def load_stack(record: dict, config: Config) -> tuple[np.ndarray, list[int]]:
    """Every slice of a series in physical order, plus which ones get cached."""

    if "ordered" not in record:
        record["ordered"], _ = order_slices(record["dir"], record["files"], config)
    files = record["ordered"]

    planes = []
    for name in files:
        ds = pydicom.dcmread(os.path.join(record["dir"], name), force=True)
        a = ds.pixel_array.astype(np.float32)
        a = a * float(getattr(ds, "RescaleSlope", 1) or 1) + float(
            getattr(ds, "RescaleIntercept", 0) or 0)
        planes.append(a)

    sampled = [int(i) for i in sample_indices(len(files), config.slices, config.band)]
    return np.stack(planes) if planes else np.zeros((0, 1, 1), np.float32), sampled


def preprocessing_steps(record: dict, config: Config,
                        side: str | None = None) -> list[tuple[str, np.ndarray]]:
    """The chain `read_slot` runs, with every intermediate kept.

    Returns (label, volume) pairs — each volume is `[group, h, w]`, so a caller can
    show whichever slice it likes and see the same transformation applied to all of
    them. The percentile step is deliberately computed over the whole stack, as
    `read_slot` does: normalising slice by slice would stretch a nearly empty slice
    across the full range and make it look as contrasted as one full of anatomy.
    """

    stack, sampled = load_stack(record, config)
    steps: list[tuple[str, np.ndarray]] = []

    volume = stack[sampled]
    steps.append((f"1. raw, rescaled\n{volume.shape[1]}x{volume.shape[2]} "
                  f"[{volume.min():.0f}, {volume.max():.0f}]", volume.copy()))

    spacing = record.get("px")
    if spacing and np.isfinite(spacing) and spacing > 0:
        want = int(round(config.crop_mm / spacing))
        h, w = volume.shape[1:]
        if 16 < want < min(h, w):
            cy, cx, half = h // 2, w // 2, want // 2
            volume = volume[:, max(0, cy - half):cy + half, max(0, cx - half):cx + half]
    steps.append((f"2. crop to {config.crop_mm:.0f} mm\n{volume.shape[1]}x{volume.shape[2]} "
                  f"@ {spacing:.3f} mm/px", volume.copy()))

    lo, hi = np.percentile(volume, [1, 99])
    volume = np.clip((volume - lo) / max(hi - lo, 1e-6), 0, 1)
    steps.append((f"3. percentile 1-99\n[{volume.min():.2f}, {volume.max():.2f}]",
                  volume.copy()))

    tensor = torch.from_numpy(np.ascontiguousarray(volume)).unsqueeze(0)
    tensor = F.interpolate(tensor, size=(config.img, config.img), mode="bilinear",
                           align_corners=False)
    volume = tensor.squeeze(0).numpy()
    steps.append((f"4. resize\n{config.img}x{config.img}", volume.copy()))

    volume = np.clip(np.round(volume * 255), 0, 255).astype(np.uint8)
    steps.append((f"5. quantise\nuint8 [{volume.min()}, {volume.max()}]", volume.copy()))

    if side == "R":
        plane = record.get("plane", "")
        if plane in ("Coronal", "Axial"):
            volume = volume[..., ::-1]
            steps.append((f"6. mirror\nright knee, {plane.lower()}", volume.copy()))
        else:
            steps.append(("6. no mirror\nright knee but sagittal", volume.copy()))
    else:
        steps.append((f"6. no mirror\nside {side or 'unresolved'}", volume.copy()))

    return steps


def verify_against_read_slot(record: dict, config: Config) -> None:
    """The chain above must end exactly where the pipeline ends.

    Without this, the figures could quietly illustrate a preprocessing nobody runs.
    """

    steps = preprocessing_steps(record, config, side=None)
    mine = steps[4][1]
    theirs = read_slot(record, config, n_slice=config.slices)
    if theirs is None:
        raise AssertionError("read_slot returned nothing for this series")
    if not np.array_equal(mine, theirs.numpy()):
        diff = np.abs(mine.astype(int) - theirs.numpy().astype(int)).max()
        raise AssertionError(
            f"the illustrated chain diverges from read_slot by up to {diff} grey levels")
