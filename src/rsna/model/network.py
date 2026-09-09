"""Encoder plus head, trained end to end."""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import POOL_PARTS, TARGETS, Config
from .heads import SlotHead
from .stems import build_stem


class Model(nn.Module):
    """A study is a bag of slot images.

    The bag is flattened for the encoder and folded back before the head, so the
    encoder never sees the study structure and the head never sees pixels.
    """

    def __init__(self, backbone: nn.Module, dim: int, config: Config,
                 pool: str = "cls_mean", prior: bool = False):
        super().__init__()
        if pool not in POOL_PARTS:
            raise ValueError(f"unknown pooling {pool!r}; expected one of {sorted(POOL_PARTS)}")
        self.backbone = backbone
        self.pool = pool
        # None for stem="window": the slices already are the three channels.
        self.stem = build_stem(config)
        self.head = SlotHead(dim * POOL_PARTS[pool], config.n_slot, len(TARGETS),
                             prior=prior, config=config)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, imgs: torch.Tensor, mask: torch.Tensor,
                img_size: int | None = None) -> torch.Tensor:
        """imgs: (batch, slot, channel, h, w) uint8, where `channel` is
        `config.window_size` — three raw slices, or the whole cache for a stem that
        compresses it. mask: (batch, slot)."""

        batch, slots = imgs.shape[:2]
        x = imgs.reshape(batch * slots, *imgs.shape[2:]).float().div_(255.0)

        if img_size is not None and img_size != x.shape[-1]:
            # The cache is held at the highest resolution any configuration needs and
            # the rest downsample from it, so every configuration sees the same pixels
            # through a different sampling grid rather than a different crop.
            x = F.interpolate(x, size=(img_size, img_size), mode="bilinear",
                              align_corners=False)

        if self.stem is not None:
            # The stem mixes the stack down to three channels and applies the ImageNet
            # normalisation itself, so the buffers below are not used on this path.
            x = self.stem(x)
        else:
            x = (x - self.mean) / self.std

        out = self.backbone(pixel_values=x).last_hidden_state
        patch = out[:, 1:]
        parts = [out[:, 0], patch.mean(1)]

        if self.pool == "cls_mean_focal":
            # The upper tail of each channel over the patch grid, taken per channel
            # rather than by selecting whole patches: a finding occupies a small part
            # of the field, so a plain mean over 256 patches dilutes it by two orders
            # of magnitude. This keeps the top eighth of each channel's responses.
            k = max(1, patch.shape[1] // 8)
            parts.append(patch.topk(k, dim=1).values.mean(1))

        feat = torch.cat(parts, dim=1).reshape(batch, slots, -1)
        return self.head(feat, mask)


def find_encoder(config: Config, root: str | Path = "/kaggle/input") -> Path | None:
    """Locate a mounted encoder checkpoint, or None.

    Matched on the directory holding a `config.json` whose path names the encoder,
    preferring one that also names the variant. Searching by content rather than by an
    expected path means the notebook keeps working whatever Kaggle calls the mount.
    """

    root = Path(root)
    if not root.is_dir():
        return None

    hits = []
    for current, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in ("train_series", "test_series")]
        if "config.json" in files and config.encoder in current.lower():
            hits.append(Path(current))

    for hit in hits:
        if config.encoder_variant in str(hit).lower():
            return hit
    return hits[0] if hits else None


def build_model(config: Config, backbone: nn.Module | None = None,
                source: str | Path | None = None) -> Model:
    """Load the encoder and open the last `config.unfreeze_last` blocks for training.

    The early blocks of a self-supervised transformer are generic edge and texture
    filters; the late blocks carry semantics. Opening only the late ones is the
    cautious choice — there may not be enough supervision here to improve the early
    ones, and there is certainly enough to damage them.

    Which encoder, which pooling and whether the head carries the anatomical prior all
    come from `config`, so a weights package states them rather than the call site
    guessing.

    `backbone` is injectable so this can be exercised without the real encoder present:
    the tests build a stub with the same interface. `source` overrides the search.
    """

    if backbone is None:
        from transformers import AutoModel

        path = Path(source) if source is not None else find_encoder(config)
        if path is None:
            raise FileNotFoundError(
                f"{config.encoder}/{config.encoder_variant} is not mounted and no "
                f"source was given")
        backbone = AutoModel.from_pretrained(str(path))

    n_layer = len(backbone.encoder.layer)
    for param in backbone.parameters():
        param.requires_grad = False
    for block in backbone.encoder.layer[max(0, n_layer - config.unfreeze_last):]:
        for param in block.parameters():
            param.requires_grad = True
    for param in backbone.layernorm.parameters():
        param.requires_grad = True

    dim = backbone.config.hidden_size
    return Model(backbone, dim, config, pool=config.pool, prior=config.prior)
