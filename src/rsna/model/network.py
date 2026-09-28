"""Encoder plus head, trained end to end."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import TARGETS, Config, pool_parts
from .encoders import EncoderSpec, find_encoder, spec_for
from .heads import SlotHead
from .stems import IMAGENET_MEAN, IMAGENET_STD, build_stem

__all__ = ["Model", "build_model", "find_encoder"]


def _channel_stats(n_channels: int) -> tuple[list[float], list[float]]:
    """The normalisation for `n_channels` inputs.

    Three channels get the ImageNet statistics unchanged, which is what every weights
    package fitted so far carries. More than three means the backbone's first
    convolution was rebuilt by averaging the pretrained RGB filters, so the matching
    input is the grey equivalent: the mean of the three. The spread between the RGB
    statistics is a property of colour photographs and means nothing on a slice.
    """

    if n_channels == 3:
        return list(IMAGENET_MEAN), list(IMAGENET_STD)
    grey_mean = sum(IMAGENET_MEAN) / 3
    grey_std = sum(IMAGENET_STD) / 3
    return [grey_mean] * n_channels, [grey_std] * n_channels


class Model(nn.Module):
    """A study is a bag of slot images.

    The bag is flattened for the encoder and folded back before the head, so the
    encoder never sees the study structure and the head never sees pixels.

    What the encoder *is* reaches this class only through an `EncoderSpec`, which is a
    plain object rather than a module: the backbone is registered here under its own
    name, so a checkpoint fitted before the spec existed still loads key for key.
    """

    def __init__(self, backbone: nn.Module, spec: EncoderSpec, config: Config,
                 pool: str = "cls_mean", prior: bool = False):
        super().__init__()
        self.backbone = backbone
        self.spec = spec  # plain attribute: owns no parameters, so invisible to nn
        self.pool = pool
        self.n_prefix = spec.n_prefix
        self.has_class_token = spec.has_class_token
        # None for stem="window" and stem="none": the slices already are the channels.
        self.stem = build_stem(config)
        self.head = SlotHead(spec.dim * pool_parts(pool, spec.has_class_token),
                             config.n_slot, len(TARGETS), prior=prior, config=config)
        mean, std = _channel_stats(spec.in_channels)
        self.register_buffer("mean", torch.tensor(mean).view(1, -1, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, -1, 1, 1))

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

        out = self.spec.tokens(x)
        patch = out[:, self.n_prefix:]
        parts = [out[:, 0]] if self.has_class_token else []
        parts.append(patch.mean(1))

        if self.pool == "cls_mean_focal":
            # The upper tail of each channel over the patch grid, taken per channel
            # rather than by selecting whole patches: a finding occupies a small part
            # of the field, so a plain mean over 256 patches dilutes it by two orders
            # of magnitude. This keeps the top eighth of each channel's responses.
            k = max(1, patch.shape[1] // 8)
            parts.append(patch.topk(k, dim=1).values.mean(1))

        feat = torch.cat(parts, dim=1).reshape(batch, slots, -1)
        return self.head(feat, mask)


def build_model(config: Config, backbone: nn.Module | None = None,
                source: str | Path | None = None) -> Model:
    """Load the encoder and open the last `config.unfreeze_last` blocks for training.

    The early blocks of a self-supervised transformer are generic edge and texture
    filters; the late blocks carry semantics. Opening only the late ones is the
    cautious choice — there may not be enough supervision here to improve the early
    ones, and there is certainly enough to damage them.

    Which encoder, which pooling and whether the head carries the anatomical prior all
    come from `config`, so a weights package states them rather than the call site
    guessing. Which *family* the encoder belongs to is a registry lookup on
    `config.encoder`; see `rsna.model.encoders`.

    `backbone` is injectable so this can be exercised without the real encoder present:
    the tests build a stub with the same interface. `source` overrides the search.
    """

    kind = spec_for(config)

    if backbone is None:
        path = Path(source) if source is not None else find_encoder(config)
        if path is None:
            raise FileNotFoundError(
                f"{config.encoder}/{config.encoder_variant} is not mounted and no "
                f"source was given")
        backbone = kind.load(config, path)

    spec = kind(backbone)

    if spec.in_channels != config.encoder_channels:
        raise ValueError(
            f"{config.encoder} takes {spec.in_channels} channels but stem "
            f"{config.stem!r} delivers {config.encoder_channels}")

    blocks = spec.blocks()
    for param in backbone.parameters():
        param.requires_grad = False
    for block in blocks[max(0, len(blocks) - config.unfreeze_last):]:
        for param in block.parameters():
            param.requires_grad = True
    for param in spec.always_trainable():
        param.requires_grad = True

    return Model(backbone, spec, config, pool=config.pool, prior=config.prior)
