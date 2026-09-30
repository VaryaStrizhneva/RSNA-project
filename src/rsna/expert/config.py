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
    #: Which ROI spec cuts its input. `"wide"` means no crop at all — the control run
    #: that makes the comparison a comparison rather than a claim.
    roi: str = "lateral_meniscus"

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
    def spec(self) -> RoiSpec | None:
        return SPECS.get(self.roi)

    def to_dict(self) -> dict:
        out = asdict(self)
        out["targets"] = list(self.targets)
        return out

    @classmethod
    def from_dict(cls, data: dict) -> "ExpertConfig":
        data = dict(data)
        # Runs written before this model trained several targets at once name one.
        if "target" in data:
            data["targets"] = (data.pop("target"),)
        if "targets" in data:
            data["targets"] = tuple(data["targets"])
        unknown = set(data) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"config records fields this pipeline does not define: "
                             f"{sorted(unknown)}")
        return cls(**data)

    def replace(self, **changes) -> "ExpertConfig":
        return replace(self, **changes)
