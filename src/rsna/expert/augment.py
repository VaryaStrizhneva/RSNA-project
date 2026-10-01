"""Jitter that the label does not depend on.

Smaller than the landmark model's, and for a reason: there the target was a position and
every geometric transform had to move with it. Here the label is one bit about the whole
crop, so a transform only has to leave the anatomy readable.

No reversal of the depth axis. On the landmark model it was the most valuable
augmentation, because the task was to find which end was lateral. Here the crop is
already oriented — the ROI extractor put the bowtie on a known side — and reversing it
would teach the model to ignore an ordering that now carries real information.
"""

from __future__ import annotations

import torch
from torch import nn


def augment_batch(volume: torch.Tensor, rng: torch.Generator) -> torch.Tensor:
    """(batch, series, slot, h, w) -> the same, jittered."""

    b, dev = volume.shape[0], volume.device
    u = lambda lo, hi: torch.rand(b, device=dev, generator=rng) * (hi - lo) + lo  # noqa: E731

    shape = volume.shape
    x = volume.reshape(b, -1, *shape[-2:])

    # Small in-plane moves only: the landmark is already within 3.7 mm at the ninetieth
    # percentile, and a crop shifted much further stops containing what it was cut for.
    angle, scale = u(-0.14, 0.14), u(0.94, 1.06)
    cos, sin = torch.cos(angle) / scale, torch.sin(angle) / scale
    theta = torch.zeros(b, 2, 3, device=dev, dtype=x.dtype)
    theta[:, 0, 0], theta[:, 0, 1], theta[:, 0, 2] = cos, -sin, u(-0.06, 0.06)
    theta[:, 1, 0], theta[:, 1, 1], theta[:, 1, 2] = sin, cos, u(-0.06, 0.06)
    grid = nn.functional.affine_grid(theta, x.shape, align_corners=False)
    x = nn.functional.grid_sample(x, grid, mode="bilinear", align_corners=False)

    # Five manufacturers and two field strengths, so what generalisation there is across
    # scanners comes from here as much as from the data.
    view = (-1,) + (1,) * (x.ndim - 1)
    x = x * u(0.85, 1.15).view(view) + u(-0.12, 0.12).view(view)
    x = x.clamp(0, 1).pow(u(0.8, 1.25).view(view))
    x = x + torch.randn(x.shape, device=dev, dtype=x.dtype, generator=rng) * 0.02
    return x.clamp(0, 1).reshape(shape)
