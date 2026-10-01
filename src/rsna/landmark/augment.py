"""Jitter the input, move the target with it.

Everything here transforms the volume and its heatmap by the *same* operation, so no
augmentation can be applied to one and forgotten on the other — which is the way a
landmark pipeline quietly teaches a model to predict a few millimetres off.

The depth operations carry most of the weight, and for a measured reason. On the 304
annotations a constant offset from the *lateral* end of the stack reaches p90 12.6 mm
while a constant slot index reaches 42.7 mm: almost all of the depth problem is deciding
which end is lateral. Reversing the stack takes that decision away from position and
hands it to anatomy — after a reversal the fibular head is at the other end, and a model
that had learned "lateral is where the index is small" is wrong on half its examples.

`decimate` exists because the corpus cannot teach the spacing it will meet. 148 of the
264 annotated 2D series sit in a 0.5 mm band around 3.3, with ten studies below 2.75.
Thinning a stack does not invent anything — it throws slices away — and the 3D series
can produce any spacing in the corpus range from the same annotation, since the label is
in millimetres and does not depend on how the stack was sampled.
"""

from __future__ import annotations

import torch
from torch import nn


def reverse(volume: torch.Tensor, heat: torch.Tensor):
    """Flip the depth axis of both. The landmark in the patient does not move."""

    return volume.flip(1), heat.flip(2)


def shift_depth(volume: torch.Tensor, heat: torch.Tensor, by: int):
    """Slide the stack within its slots, so slot index carries no information."""

    return volume.roll(by, dims=1), heat.roll(by, dims=2)


def decimate(volume: torch.Tensor, heat: torch.Tensor, stride: int):
    """Keep every `stride`-th slice, re-centred in the slots it had.

    The heatmap is summed over the slices that collapse into one rather than sampled,
    so a target whose peak fell on a discarded slice is not lost — it moves onto the
    slice that replaced it, which is what physically happened.
    """

    if stride <= 1:
        return volume, heat
    s = volume.shape[1]
    keep = torch.arange(0, s, stride, device=volume.device)
    v = torch.zeros_like(volume)
    h = torch.zeros_like(heat)
    n = len(keep)
    start = (s - n) // 2
    v[:, start:start + n] = volume[:, keep]
    for j, i in enumerate(keep):
        h[:, :, start + j] = heat[:, :, i:i + stride].sum(dim=2)
    return v, h


def _affine(x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """Apply one 2x3 affine per batch item to a (batch, channel, h, w) tensor."""

    grid = nn.functional.affine_grid(theta, x.shape, align_corners=False)
    return nn.functional.grid_sample(x, grid, mode="bilinear", align_corners=False)


def in_plane(volume: torch.Tensor, heat: torch.Tensor, angle: torch.Tensor,
             shift: torch.Tensor, scale: torch.Tensor):
    """Rotate, translate and scale both, about the centre of the image.

    `shift` is in normalised units, so the same numbers move the volume and the heatmap
    by the same physical distance despite their different resolutions — the one thing
    that has to be right for a stride-4 target to stay on its anatomy.
    """

    b = volume.shape[0]
    cos, sin = torch.cos(angle) / scale, torch.sin(angle) / scale
    theta = torch.zeros(b, 2, 3, device=volume.device, dtype=volume.dtype)
    theta[:, 0, 0], theta[:, 0, 1], theta[:, 0, 2] = cos, -sin, shift[:, 0]
    theta[:, 1, 0], theta[:, 1, 1], theta[:, 1, 2] = sin, cos, shift[:, 1]

    v = _affine(volume, theta)
    k, s = heat.shape[1], heat.shape[2]
    h = _affine(heat.reshape(b, k * s, *heat.shape[-2:]), theta).reshape_as(heat)
    return v, h


def intensity(volume: torch.Tensor, brightness: torch.Tensor, contrast: torch.Tensor,
              gamma: torch.Tensor, noise: float, generator=None) -> torch.Tensor:
    """Brightness, contrast, gamma and noise. The target is untouched: none of it moves.

    Five manufacturers and two field strengths are in the corpus, and the annotation set
    holds one to five studies from three of them. What generalisation there is across
    scanners will come from here as much as from the data.
    """

    shape = (-1,) + (1,) * (volume.ndim - 1)
    x = volume * contrast.view(shape) + brightness.view(shape)
    x = x.clamp(0, 1).pow(gamma.view(shape))
    if noise > 0:
        x = x + torch.randn(volume.shape, device=volume.device,
                            dtype=volume.dtype, generator=generator) * noise
    return x.clamp(0, 1)
