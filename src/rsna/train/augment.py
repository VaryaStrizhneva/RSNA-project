"""Jitter that no label depends on.

Neither flip is available here, for different reasons.

A **horizontal** flip would reintroduce the nuisance axis that laterality
normalisation removes — it would undo, once per batch, what the header pass was run to
establish.

A **vertical** flip is not a nuisance axis at all. A knee is acquired in a canonical
orientation and no study in this corpus looks like its own vertical mirror. An
augmentation covers directions along which the label does not change; this one moves
the input off the distribution the encoder will be asked about. Where a finding sits in
the frame is information rather than noise.

What is left is a few degrees of rotation and a few per cent of scale and translation:
enough to prevent memorising the exact framing, which is what an augmentation is for,
while leaving the anatomy where it was.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from ..config import Config


def take_group(rows: torch.Tensor, group_index: int, config: Config) -> torch.Tensor:
    """Slice `config.group` consecutive channels out of cached slices."""

    start = group_index * config.group
    return rows[:, :, start:start + config.group]


def augment(imgs: torch.Tensor, config: Config,
            generator: torch.Generator | None = None) -> torch.Tensor:
    """A small rigid jitter and an intensity scale, applied to a whole bag at once."""

    # A bag arrives as [study, slot, group, h, w]: five axes, not four. The warp is a
    # 2-D operation, so the two leading axes are folded together and restored after —
    # every slot image is an independent acquisition and gets its own jitter.
    lead = imgs.shape[:-3]
    x = imgs.reshape(-1, *imgs.shape[-3:]).float()
    n, device = x.shape[0], x.device

    def rand(*shape):
        return torch.rand(*shape, device=device, generator=generator)

    rot = (rand(n) - 0.5) * 2 * (config.aug_rot_deg * math.pi / 180)
    # Zoom in only. `border` padding repeats the edge row outward, and the edge of this
    # crop is where the popliteal fossa sits; zooming out would fabricate tissue exactly
    # where a Baker cyst is looked for.
    scale = 1.0 + rand(n) * config.aug_scale
    tx = (rand(n) - 0.5) * 2 * config.aug_shift
    ty = (rand(n) - 0.5) * 2 * config.aug_shift

    cos, sin = torch.cos(rot) / scale, torch.sin(rot) / scale
    theta = torch.zeros(n, 2, 3, device=device, dtype=torch.float32)
    theta[:, 0, 0], theta[:, 0, 1], theta[:, 0, 2] = cos, -sin, tx
    theta[:, 1, 0], theta[:, 1, 1], theta[:, 1, 2] = sin, cos, ty

    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)

    intensity = 1.0 + (rand(n, 1, 1, 1) - 0.5) * 2 * config.aug_intensity
    x = (x * intensity).clamp(0, 255)
    return x.reshape(*lead, *x.shape[-3:]).to(imgs.dtype)
