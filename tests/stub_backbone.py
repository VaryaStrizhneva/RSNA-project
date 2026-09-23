"""A stand-in for DINOv2 with the same interface, so the model can be exercised
without the real encoder present: same call signature, same `last_hidden_state`
shape (batch, 1 + patches, dim), same `encoder.layer` / `layernorm` structure that
`build_model` freezes against."""

from __future__ import annotations

import torch
import torch.nn as nn


class _Config:
    def __init__(self, hidden_size):
        self.hidden_size = hidden_size


class _Out:
    def __init__(self, last_hidden_state):
        self.last_hidden_state = last_hidden_state


class StubBackbone(nn.Module):
    def __init__(self, dim=64, n_layer=12, patch=14):
        super().__init__()
        self.config = _Config(dim)
        self.patch = patch
        self.embed = nn.Conv2d(3, dim, kernel_size=patch, stride=patch)
        self.encoder = nn.Module()
        self.encoder.layer = nn.ModuleList(
            [nn.Sequential(nn.Linear(dim, dim), nn.GELU()) for _ in range(n_layer)])
        self.layernorm = nn.LayerNorm(dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, pixel_values):
        x = self.embed(pixel_values).flatten(2).transpose(1, 2)
        for layer in self.encoder.layer:
            x = layer(x)
        x = torch.cat([self.cls.expand(x.shape[0], -1, -1), x], dim=1)
        return _Out(self.layernorm(x))


class StubTimmViT(nn.Module):
    """A stand-in for a timm vision transformer: `forward_features` returns tokens,
    `num_prefix_tokens` says how many lead, `num_features` is the width, `blocks` is
    the depth list. Enough to drive `TimmBackbone` without timm installed."""

    def __init__(self, dim=64, n_layer=6, patch=8, n_prefix=1, in_chans=3):
        super().__init__()
        self.num_features = dim
        self.num_prefix_tokens = n_prefix
        self.embed = nn.Conv2d(in_chans, dim, kernel_size=patch, stride=patch)
        self.blocks = nn.ModuleList(
            [nn.Sequential(nn.Linear(dim, dim), nn.GELU()) for _ in range(n_layer)])
        self.norm = nn.LayerNorm(dim)
        self.prefix = nn.Parameter(torch.zeros(1, n_prefix, dim))

    def forward_features(self, x):
        x = self.embed(x).flatten(2).transpose(1, 2)
        for block in self.blocks:
            x = block(x)
        x = torch.cat([self.prefix.expand(x.shape[0], -1, -1), x], dim=1)
        return self.norm(x)


class StubTimmConv(nn.Module):
    """A stand-in for a timm convolutional or hybrid backbone — CoAtNet's shape.

    `forward_features` returns a feature *map*, not tokens, and there is no class
    token. Its stages are themselves sequences of blocks, which is what
    `TimmBackbone.blocks` has to flatten for `unfreeze_last` to count the same thing
    here as on a flat model.
    """

    def __init__(self, dim=64, stages=(2, 2), in_chans=3):
        super().__init__()
        self.num_features = dim
        self.stem = nn.Conv2d(in_chans, dim, kernel_size=8, stride=8)
        self.stages = nn.ModuleList()
        for n in stages:
            stage = nn.Module()
            stage.blocks = nn.ModuleList(
                [nn.Conv2d(dim, dim, 3, padding=1) for _ in range(n)])
            self.stages.append(stage)
        self.norm = nn.GroupNorm(1, dim)

    def forward_features(self, x):
        x = self.stem(x)
        for stage in self.stages:
            for block in stage.blocks:
                x = block(x)
        return self.norm(x)
