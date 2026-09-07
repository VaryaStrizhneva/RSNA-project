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
