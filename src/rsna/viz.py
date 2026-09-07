"""Figures that show the pixel path doing its work.

Everything here is for looking, not for deciding. It lives in the package rather than
in a notebook so that the notebook holds no logic and cannot drift from what actually
runs — and `verify_against_read_slot` asserts exactly that: the chain reproduced here,
step by step, ends where `read_slot` ends.
"""

from __future__ import annotations

import os

import numpy as np
import pydicom
import torch
import torch.nn.functional as F

from .config import Config
from .dicom.ordering import order_slices
from .dicom.pixels import read_slot, sample_indices


def load_stack(record: dict, config: Config) -> tuple[np.ndarray, list[int]]:
    """Every slice of a series in physical order, plus which ones get sampled."""

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

    sampled = [int(i) for i in sample_indices(len(files), config.group, config.band)]
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
            steps.append((f"6. no mirror\nright knee but sagittal", volume.copy()))
    else:
        steps.append((f"6. no mirror\nside {side or 'unresolved'}", volume.copy()))

    return steps


def verify_against_read_slot(record: dict, config: Config) -> None:
    """The chain above must end exactly where the pipeline ends.

    Without this, the figures could quietly illustrate a preprocessing nobody runs.
    """

    steps = preprocessing_steps(record, config, side=None)
    mine = steps[4][1]
    theirs = read_slot(record, config)
    if theirs is None:
        raise AssertionError("read_slot returned nothing for this series")
    if not np.array_equal(mine, theirs.numpy()):
        diff = np.abs(mine.astype(int) - theirs.numpy().astype(int)).max()
        raise AssertionError(
            f"the illustrated chain diverges from read_slot by up to {diff} grey levels")


def figure_stack(record: dict, config: Config, max_shown: int = 24):
    """Every slice in physical order; the sampled ones outlined."""

    import matplotlib.pyplot as plt

    stack, sampled = load_stack(record, config)
    n = len(stack)
    shown = list(range(n)) if n <= max_shown else sorted(
        set(np.linspace(0, n - 1, max_shown).astype(int)) | set(sampled))

    cols = min(8, len(shown))
    rows = int(np.ceil(len(shown) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(1.7 * cols, 1.8 * rows))
    for ax, i in zip(np.atleast_1d(axes).ravel(), shown):
        lo, hi = np.percentile(stack, [1, 99])
        ax.imshow(np.clip((stack[i] - lo) / max(hi - lo, 1e-6), 0, 1), cmap="gray")
        ax.set_xticks([]); ax.set_yticks([])
        picked = i in sampled
        ax.set_title(f"{i}{'  ←' if picked else ''}", fontsize=8,
                     color="tab:red" if picked else "black")
        for spine in ax.spines.values():
            spine.set_edgecolor("tab:red" if picked else "0.85")
            spine.set_linewidth(2.0 if picked else 0.5)
    for ax in np.atleast_1d(axes).ravel()[len(shown):]:
        ax.axis("off")
    fig.suptitle(f"{n} slices in physical order — sampled: {sampled}", fontsize=10)
    fig.tight_layout()
    return fig


def figure_steps(record: dict, config: Config, side: str | None = None, slice_index: int = 1):
    """One panel per transformation, on a single slice."""

    import matplotlib.pyplot as plt

    steps = preprocessing_steps(record, config, side)
    fig, axes = plt.subplots(1, len(steps), figsize=(2.6 * len(steps), 3.2))
    for ax, (title, volume) in zip(np.atleast_1d(axes), steps):
        image = volume[min(slice_index, len(volume) - 1)]
        ax.imshow(image, cmap="gray")
        ax.set_title(title, fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    return fig


def figure_channels(record: dict, config: Config, side: str | None = None):
    """The three slices as three channels, in the right order and in file order.

    The composite is the point: three neighbouring slices are nearly grey, because the
    channels agree. Three arbitrary slices are not, and the colour fringing is the
    model's input being noise along its third dimension.
    """

    import matplotlib.pyplot as plt

    correct = preprocessing_steps(record, config, side)[-1][1]

    shuffled = dict(record)
    shuffled["ordered"] = list(record["files"])  # file-name order: a SOP UID sort
    wrong = preprocessing_steps(shuffled, config, side)[-1][1]

    fig, axes = plt.subplots(2, 4, figsize=(11, 6))
    for row, (volume, label) in enumerate(((correct, "geometric order"),
                                           (wrong, "file-name order"))):
        for c in range(3):
            axes[row, c].imshow(volume[c], cmap="gray", vmin=0, vmax=255)
            axes[row, c].set_title(f"{label} — channel {c}", fontsize=8)
        axes[row, 3].imshow(np.transpose(volume, (1, 2, 0)))
        axes[row, 3].set_title(f"{label} — as RGB", fontsize=8)
        for ax in axes[row]:
            ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    return fig
