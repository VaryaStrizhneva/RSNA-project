"""What a single-pathology expert is, end to end."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

from ..roi.config import SPECS, RoiSpec


@dataclass(frozen=True)
class ExpertConfig:
    """One model, one target."""

    #: Which pathologies, as named in `rsna.config.TARGETS`. Several on one encoder
    #: rather than one model each: they share a region of interest, so they share the
    #: features that describe it, and a lateral compartment is one thing to look at.
    targets: tuple[str, ...] = ("Lateral Meniscus", "Lateral OA")
    #: Which ROI specs cut its input, by name in `rsna.roi.SPECS`. Several because a
    #: compartment is read in more than one plane and they do not show the same thing:
    #: a sagittal crop shows the two horns in profile, a coronal one shows extrusion,
    #: which is a displacement past the tibial margin that a sagittal slice cannot give
    #: at all. Each spec brings its own box, its own depth and its own output size, so
    #: they do not share a tensor — the encoder is run once per spec and the features
    #: meet at the attention.
    rois: tuple[str, ...] = ("lateral_meniscus",)

    #: A timm model name. Convolutional on purpose; see `network.py`.
    encoder: str = "resnet18"
    #: Neighbouring slices per window, as the encoder's channels.
    group: int = 3
    dropout: float = 0.2
    #: Bias every head starts at, in logits. One value because the two targets sit at
    #: the same rate — Lateral Meniscus 25.1 %, Lateral OA 24.6 % — and log(0.25/0.75)
    #: is -1.1. A group whose rates differ would need one per target.
    prior_logit: float = -1.1

    epochs: int = 30
    batch: int = 32
    lr: float = 3e-4
    weight_decay: float = 1e-4
    folds: int = 5
    seed: int = 0

    @property
    def specs(self) -> list[RoiSpec]:
        return [SPECS[name] for name in self.rois]

    def to_dict(self) -> dict:
        out = asdict(self)
        out["targets"] = list(self.targets)
        out["rois"] = list(self.rois)
        return out

    @classmethod
    def from_dict(cls, data: dict) -> "ExpertConfig":
        data = dict(data)
        # Runs written before this model read several planes, or trained several
        # targets, name one of each. Reading them back matters: an experiment file that
        # no longer reproduces its own run turns a published number into a rumour.
        for old, new in (("roi", "rois"), ("target", "targets")):
            if old in data:
                data[new] = (data.pop(old),)
        for key in ("rois", "targets"):
            if key in data:
                data[key] = tuple(data[key])
        unknown = set(data) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"config records fields this pipeline does not define: "
                             f"{sorted(unknown)}")
        return cls(**data)

    def replace(self, **changes) -> "ExpertConfig":
        return replace(self, **changes)
