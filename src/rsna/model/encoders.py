"""Driving a pretrained backbone without knowing which one it is.

Two encoders that both "produce features" still disagree about nearly everything a
caller has to know: what they are called, how they are loaded, how many channels they
take, whether their output is a token sequence or a feature map, how many of the
leading tokens are not patches, and where the blocks are that one might freeze. The
model needs answers to exactly those questions and to nothing else.

So each family of backbone gets a **spec**: a small object that answers them. Adding an
architecture means writing one spec and naming it in `ENCODERS`, not threading another
branch through `build_model` and `Model.forward`.

A spec is deliberately **not** an `nn.Module`. It holds a reference to the backbone and
owns no parameters, so the backbone stays registered on the model under its own name
and the thirteen packages fitted before this file existed still load key for key. That
is a constraint, not a preference: `load_member` refuses a state dict with a single
unexpected name, which is what makes it worth trusting.

Two specs live here:

* `HuggingFaceViT` — what DINOv2 is loaded through today, `transformers.AutoModel`.
  Its behaviour is pinned by `tests/golden/encoder_dinov2.json` and must not move.
* `TimmBackbone` — `timm`'s normalised contract, which is how everything that is not
  on the HuggingFace hub as a bare `AutoModel` arrives: CoAtNet, DINOv3, ConvNeXt.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn as nn


class EncoderSpec:
    """What the rest of the model needs from a pretrained vision backbone.

    Subclasses answer for one family. Everything here is a question the model actually
    asks; a capability that no caller reads does not belong in this class.
    """

    #: Filenames that mark a directory as a checkpoint of this family, for `find`.
    markers: tuple[str, ...] = ("config.json",)

    def __init__(self, module: nn.Module):
        self.module = module

    # -- what the model is built around ------------------------------------- #

    @property
    def dim(self) -> int:
        """Width of one token. The head's input follows from it."""

        raise NotImplementedError

    @property
    def in_channels(self) -> int:
        """How many channels the backbone takes. Three for anything pretrained on RGB;
        a stem exists to reduce the slice stack to this number."""

        return 3

    @property
    def has_class_token(self) -> bool:
        """Whether token 0 summarises the image rather than covering part of it.

        A convolutional backbone has no such token, and taking its first patch as if it
        were one would be a quiet lie — so pooling asks instead of assuming.
        """

        return self.n_prefix > 0

    @property
    def n_prefix(self) -> int:
        """Leading tokens that are not patches — class token, registers."""

        raise NotImplementedError

    # -- what the model calls ----------------------------------------------- #

    def tokens(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, channel, h, w) -> (batch, token, dim), prefix tokens first."""

        raise NotImplementedError

    def blocks(self) -> list[nn.Module]:
        """The backbone's blocks in depth order, shallowest first.

        Freezing is expressed as a count from the end, so this is the only thing a
        family has to expose for `unfreeze_last` to mean the same everywhere.
        """

        raise NotImplementedError

    def always_trainable(self) -> list[nn.Parameter]:
        """Parameters opened whatever `unfreeze_last` says — the final normalisation,
        which sits after every block and is cheap to adapt."""

        return []

    # -- loading ------------------------------------------------------------- #

    @classmethod
    def load(cls, config, path: Path) -> nn.Module:
        """Build the backbone from a checkpoint directory."""

        raise NotImplementedError


ENCODERS: dict[str, type[EncoderSpec]] = {}


def register(*names: str):
    """Name a spec in the registry. One decorator line is the whole cost of adding an
    architecture that the existing specs already know how to drive."""

    def wrap(cls: type[EncoderSpec]) -> type[EncoderSpec]:
        for name in names:
            ENCODERS[name] = cls
        return cls
    return wrap


def spec_for(config) -> type[EncoderSpec]:
    """The spec `config.encoder` names."""

    try:
        return ENCODERS[config.encoder]
    except KeyError:
        raise ValueError(
            f"unknown encoder {config.encoder!r}; registered: "
            f"{', '.join(sorted(ENCODERS))}") from None


@register("dinov2")
class HuggingFaceViT(EncoderSpec):
    """A vision transformer loaded by `transformers.AutoModel`.

    The path DINOv2-small and -base have always taken here. Its numbers are pinned by a
    golden file; this class is a description of what the code already did, moved out of
    `build_model` so that it stops being the only thing `build_model` can do.
    """

    @property
    def dim(self) -> int:
        return self.module.config.hidden_size

    @property
    def n_prefix(self) -> int:
        # DINOv2 as `AutoModel` emits one class token and no registers.
        return 1

    def tokens(self, x: torch.Tensor) -> torch.Tensor:
        return self.module(pixel_values=x).last_hidden_state

    def blocks(self) -> list[nn.Module]:
        return list(self.module.encoder.layer)

    def always_trainable(self) -> list[nn.Parameter]:
        return list(self.module.layernorm.parameters())

    @classmethod
    def load(cls, config, path: Path) -> nn.Module:
        from transformers import AutoModel

        return AutoModel.from_pretrained(str(path))


@register("coatnet", "dinov3", "convnext")
class TimmBackbone(EncoderSpec):
    """Anything reached through `timm`.

    timm normalises what HuggingFace leaves to each model class: `num_features` is the
    feature width, `num_prefix_tokens` says how many leading tokens are not patches,
    `forward_features` stops before the classifier, and `in_chans` at creation rebuilds
    the first convolution for a different channel count. That last one is why a timm
    encoder can be fed the slice stack directly, with `stem = "none"`.

    `forward_features` returns a token sequence for a transformer and a feature map for
    anything with convolutions in it — CoAtNet is both, and returns a map. Flattening
    the map into tokens makes the two look the same to the pooling above, which is
    honest: a spatial position is a spatial position. What does *not* survive is the
    class token, and `has_class_token` says so rather than letting the head read a
    corner patch as a summary.
    """

    markers = ("config.json", "model.safetensors", "pytorch_model.bin")

    @property
    def dim(self) -> int:
        return self.module.num_features

    @property
    def in_channels(self) -> int:
        # Set by `load` from the config, because timm rebuilds the stem for it.
        return getattr(self.module, "_rsna_in_chans", 3)

    @property
    def n_prefix(self) -> int:
        return int(getattr(self.module, "num_prefix_tokens", 0))

    def tokens(self, x: torch.Tensor) -> torch.Tensor:
        out = self.module.forward_features(x)
        if out.ndim == 4:
            # (batch, dim, h, w) -> (batch, token, dim). channels_last models already
            # emit (batch, h, w, dim); `num_features` tells the two apart.
            if out.shape[1] == self.dim:
                out = out.flatten(2).transpose(1, 2)
            else:
                out = out.flatten(1, 2)
        return out

    def blocks(self) -> list[nn.Module]:
        """timm's own depth grouping where there is one, its stages otherwise.

        `group_matcher` is what timm's layer-wise learning-rate decay uses, so it is the
        library's own answer to "what is a block here" and covers models whose stages
        are not a flat list.
        """

        for attr in ("blocks", "stages", "layers"):
            found = getattr(self.module, attr, None)
            if found is not None:
                return [m for stage in found for m in self._flatten(stage)]
        raise ValueError(
            f"{type(self.module).__name__} exposes no blocks, stages or layers; "
            f"freezing by depth needs one of them")

    @staticmethod
    def _flatten(stage: nn.Module) -> list[nn.Module]:
        """A stage that is itself a sequence of blocks counts as its blocks, so
        `unfreeze_last` measures the same thing on a flat and a staged model."""

        inner = getattr(stage, "blocks", None)
        if isinstance(inner, (nn.ModuleList, nn.Sequential)):
            return list(inner)
        return [stage]

    def always_trainable(self) -> list[nn.Parameter]:
        norm = getattr(self.module, "norm", None) or getattr(self.module, "norm_pre", None)
        return list(norm.parameters()) if norm is not None else []

    @classmethod
    def load(cls, config, path: Path) -> nn.Module:
        import timm

        model = timm.create_model(config.encoder_variant, pretrained=True,
                                  pretrained_cfg_overlay={"file": str(path)},
                                  num_classes=0, in_chans=config.encoder_channels)
        model._rsna_in_chans = config.encoder_channels
        return model


def find_encoder(config, root: str | Path = "/kaggle/input") -> Path | None:
    """Locate a mounted encoder checkpoint, or None.

    Matched on a directory holding one of the spec's marker files whose path names the
    encoder, preferring one that also names the variant. Searching by content rather
    than by an expected path means the notebook keeps working whatever Kaggle calls the
    mount.
    """

    root = Path(root)
    if not root.is_dir():
        return None
    markers = set(spec_for(config).markers)

    hits = []
    for current, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in ("train_series", "test_series")]
        if markers & set(files) and config.encoder in current.lower():
            hits.append(Path(current))

    for hit in hits:
        if config.encoder_variant in str(hit).lower():
            return hit
    return hits[0] if hits else None
