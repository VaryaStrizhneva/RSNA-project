"""Turning cached slices into the three channels a pretrained encoder expects.

A pretrained vision encoder takes three channels, because it was trained on RGB. The
cache holds more slices than that, so something has to reduce them — and *how* is a
design choice, not a detail.

The default elsewhere is to take three contiguous slices and run the encoder once per
window. This is the alternative: hand the encoder a **learned mixture** of the whole
stack, once.

Ported from `bend-the-knee-to-the-dinosaurs`, cell 22, where it sits behind a `stem`
switch whose default is off — so it is one competitor's idea, tried, not a settled
practice. See `notebooks/external/`.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class GatedDepthBlock(nn.Module):
    """One residual mixing step along the slice axis.

    A gated 1x1 convolution: `v(z) * silu(g(z))` lets each output slice choose how much
    of each input slice to admit, per pixel. The residual is scaled by a learned
    `gamma` initialised small, so the block starts as almost the identity and the stack
    reaches the encoder barely touched — mixing is learned rather than imposed.
    """

    def __init__(self, n_slice: int, dropout: float = 0.0, gamma_init: float = 0.1):
        super().__init__()
        self.norm = nn.GroupNorm(1, n_slice)
        self.value = nn.Conv2d(n_slice, n_slice, 1)
        self.gate = nn.Conv2d(n_slice, n_slice, 1)
        self.out = nn.Conv2d(n_slice, n_slice, 1)
        self.gamma = nn.Parameter(torch.full((n_slice, 1, 1), gamma_init))
        self.drop = nn.Dropout2d(dropout) if dropout else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.norm(x)
        return x + self.gamma * self.drop(self.out(self.value(z) * F.silu(self.gate(z))))


class DepthCompress(nn.Module):
    """`n_slice` slices -> three channels, by a learned projection.

    The encoder then runs **once** per slot instead of once per window, and training
    sees exactly what inference sees. What it gives up is the free regularisation the
    window scheme gets: a different window each step, and an average over windows at
    test time.

    An absent slot arrives as zeros. The projection has a bias, so zeros in do not give
    zeros out — `keep` puts them back, because the presence mask downstream assumes a
    blank slot stays blank.
    """

    def __init__(self, n_slice: int, out_channels: int = 3, depth: int = 1,
                 dropout: float = 0.0, imagenet: bool = True):
        super().__init__()
        self.blocks = nn.ModuleList(
            [GatedDepthBlock(n_slice, dropout) for _ in range(depth)])
        self.proj = nn.Conv2d(n_slice, out_channels, 1, bias=True)
        self.imagenet = imagenet
        if imagenet:
            self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, -1, 1, 1))
            self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, -1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, n_slice, h, w) in [0, 1]. Returns (batch, 3, h, w)."""

        keep = (x.amax(dim=1, keepdim=True) > 0).to(x.dtype)
        for block in self.blocks:
            x = block(x)
        x = self.proj(x)
        if self.imagenet:
            x = (x - self.mean) / self.std
        return x * keep


def build_stem(config) -> nn.Module | None:
    """The stem `config.stem` asks for, or None when the encoder is fed raw slices."""

    if config.stem == "window":
        return None
    if config.stem == "compress":
        return DepthCompress(config.slices, 3, depth=config.stem_depth)
    raise ValueError(f"unknown stem {config.stem!r}; expected 'window' or 'compress'")
