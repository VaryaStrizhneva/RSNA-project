"""One encoder over the slices, one head per landmark, one heatmap out.

Three decisions, each forced by something measured rather than chosen by taste.

**A convolutional encoder with a feature pyramid, not the classifier's token encoder.**
A heatmap wants a dense spatial output at a fine stride; a ViT at patch 14 gives an
18x18 grid over 256 pixels, which is 10 mm a cell against a 4 mm target width. The timm
pyramid gives stride 4 — 64x64 at 2.8 mm a cell — for free. This is why the landmark
model does not go through `rsna.model.encoders`: that registry answers "drive this
backbone as a whole-image encoder and give me one vector", which is a different question
from "give me multi-scale spatial features", and only a convolutional model can answer
the second.

**Depth context spans the whole stack, not a local neighbourhood.** Measured on the 304
annotations: a constant offset from the *lateral end* of the stack gets p90 to 12.6 mm,
while a constant slot index gets 42.7 mm. Almost the entire depth problem is deciding
which end is lateral — and the evidence for that is the fibular head, which can sit
twenty slices from the meniscus. A stack of 3x3x3 convolutions sees seven slices and
would never reach it. Dilated convolutions along depth reach 63 slots, which is the
whole input.

**No absolute position anywhere.** The depth path is convolutional and order-aware but
carries no positional embedding, and training shifts the stack within its slots at
random. "The joint is at slot 24" is true of the padding pattern as well as of the
anatomy, and a model allowed to read it would stop looking at knees.
"""

from __future__ import annotations

import torch
from torch import nn

from .config import LandmarkConfig


class Pyramid(nn.Module):
    """Top-down fusion of a timm feature pyramid down to one stride-4 map."""

    def __init__(self, channels: list[int], width: int):
        super().__init__()
        self.lateral = nn.ModuleList(nn.Conv2d(c, width, 1) for c in channels)
        self.smooth = nn.ModuleList(
            nn.Sequential(nn.Conv2d(width, width, 3, padding=1),
                          nn.GroupNorm(8, width), nn.GELU())
            for _ in channels[:-1])

    def forward(self, feats: list[torch.Tensor]) -> torch.Tensor:
        x = self.lateral[-1](feats[-1])
        for i in range(len(feats) - 2, -1, -1):
            x = nn.functional.interpolate(x, size=feats[i].shape[-2:],
                                          mode="bilinear", align_corners=False)
            x = self.smooth[i](x + self.lateral[i](feats[i]))
        return x


class DepthContext(nn.Module):
    """Every slice sees the whole stack, without learning where it sits in it.

    Dilations 1, 2, 4, 8 give a receptive field of 31 slots, and two passes cover the
    48. The summary is a mean over the slice, so what travels along the depth axis is
    "what kind of slice is this" — which is exactly what deciding the lateral end needs.
    """

    def __init__(self, width: int):
        super().__init__()
        self.path = nn.Sequential(*[
            layer
            for d in (1, 2, 4, 8)
            for layer in (nn.Conv1d(width, width, 3, padding=d, dilation=d),
                          nn.GroupNorm(8, width), nn.GELU())])

    def forward(self, summary: torch.Tensor) -> torch.Tensor:
        """(batch, slice, width) -> (batch, slice, width)"""

        return self.path(summary.transpose(1, 2)).transpose(1, 2)


class LandmarkNet(nn.Module):
    """(batch, slice, 3, img, img) -> (batch, landmark, slice, grid, grid)."""

    def __init__(self, config: LandmarkConfig, width: int = 64, pretrained: bool = True):
        super().__init__()
        import timm

        self.config = config
        self.trunk = timm.create_model(config.encoder_variant, pretrained=pretrained,
                                       features_only=True, out_indices=(1, 2, 3, 4))
        channels = self.trunk.feature_info.channels()
        reduction = self.trunk.feature_info.reduction()
        if reduction[0] != config.stride:
            raise ValueError(f"{config.encoder_variant} gives stride {reduction[0]} at its "
                             f"first kept stage; the config asks for {config.stride}")
        self.pyramid = Pyramid(channels, width)
        self.depth = DepthContext(width)
        self.head = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1), nn.GroupNorm(8, width), nn.GELU(),
            nn.Conv2d(width, len(config.points), 1))

        # A heatmap head starts by predicting "nothing here", which is true of almost
        # every cell. Without this the first steps spend themselves pushing a symmetric
        # initialisation down to the background level.
        nn.init.constant_(self.head[-1].bias, -4.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s = x.shape[:2]
        f = self.pyramid(self.trunk(x.flatten(0, 1)))          # (b*s, w, g, g)
        w, g = f.shape[1], f.shape[-1]

        ctx = self.depth(f.mean(dim=(-2, -1)).view(b, s, w))    # (b, s, w)
        f = f + ctx.reshape(b * s, w, 1, 1)

        out = self.head(f).view(b, s, -1, g, g)
        return out.permute(0, 2, 1, 3, 4).contiguous()          # (b, k, s, g, g)
