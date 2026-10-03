"""Starting a trunk from weights that are not ImageNet.

One file matters so far: RadImageNet's ResNet-50, fitted on roughly 1.35 million
radiological images. It is published as the state dict of a `torchvision` ResNet-50 with
its classifier cut off and the remainder wrapped in a `Sequential`, so its keys read
`backbone.0.weight`, `backbone.4.0.conv1.weight`, and so on. timm names the same tensors
`conv1.weight` and `layer1.0.conv1.weight`. The two models are identical; only the
spelling differs, and `torchvision.models.resnet50().children()` fixes the translation:

    0 conv1   1 bn1   2 relu   3 maxpool   4 layer1   5 layer2   6 layer3   7 layer4

Checked rather than assumed: the remapped file loads into `timm.create_model("resnet50")`
with **zero missing and zero unexpected keys** once the classifier is excused, and
carries 23.6 M parameters against timm's own count.

A loader that silently tolerated a mismatch would be worse than none at all — a trunk
half-initialised from radiology and half from `kaiming_normal_` trains perfectly well and
answers a question nobody asked. So anything unexplained raises.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

#: `torchvision.models.resnet50().children()` index -> timm attribute, for the layers a
#: trunk keeps. The classifier is not among them: timm is built with `num_classes=0`.
_TORCHVISION_CHILDREN = {"0": "conv1", "1": "bn1", "4": "layer1", "5": "layer2",
                         "6": "layer3", "7": "layer4"}


def _remap(state: dict) -> dict:
    """Rename `backbone.<index>.…` to the timm attribute it stands for."""

    out, unknown = {}, set()
    for key, value in state.items():
        head, _, rest = key.partition(".")
        if head != "backbone":
            out[key] = value
            continue
        index, _, tail = rest.partition(".")
        name = _TORCHVISION_CHILDREN.get(index)
        if name is None:
            unknown.add(index)
            continue
        out[f"{name}.{tail}"] = value
    if unknown:
        raise ValueError(
            f"this file carries torchvision children {sorted(unknown)}, which are not "
            f"layers a trunk keeps; it is not the published RadImageNet encoder")
    return out


def load_encoder_weights(trunk: nn.Module, path: str | Path, log=None) -> None:
    """Start `trunk` from `path` instead of from ImageNet, or refuse to.

    Keys missing from the file are an error, with one exception: a classifier the trunk
    does not have, because timm is asked for `num_classes=0` and the published encoder
    was cut at the same place.
    """

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} does not exist. RadImageNet is not in git — it is 94 MB of someone "
            f"else's weights — so fetch it first:\n"
            f"  kaggle datasets download -d marwanmath/resnet-50-radimagenet-marwan "
            f"-p models/radimagenet --unzip")

    state = torch.load(path, map_location="cpu", weights_only=True)
    state = state.get("state_dict", state)
    missing, unexpected = trunk.load_state_dict(_remap(state), strict=False)
    missing = [k for k in missing if not k.startswith(("fc.", "classifier.", "head."))]
    if missing or unexpected:
        raise ValueError(
            f"{path.name} does not fit this trunk: {len(missing)} missing "
            f"({missing[:3]}), {len(unexpected)} unexpected ({list(unexpected)[:3]}). "
            f"A trunk half-initialised from radiology and half from scratch trains "
            f"fine and answers a question nobody asked.")
    if log is not None:
        total = sum(v.numel() for v in state.values())
        log(f"  trunk started from {path.name} ({total / 1e6:.1f} M parameters) "
            f"instead of ImageNet")
