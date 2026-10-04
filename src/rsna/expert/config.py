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
    #: Where the trunk starts from, when not from ImageNet. A path to a state dict, or
    #: "" for timm's own pretrained weights.
    #:
    #: The one that exists is RadImageNet — a ResNet-50 fitted on ~1.35 M radiological
    #: images rather than on photographs. The case for it is stronger here than for the
    #: wide model: a 48 x 27 mm crop of fibrocartilage in greyscale has none of the
    #: statistics that make ImageNet worth anything, no colour, no objects, no scene,
    #: while a whole knee at 336 px at least looks like a picture.
    #:
    #: Changing this usually changes `encoder` too, so a run that sets it is testing two
    #: things at once — capacity and pretraining. Which of the two paid needs the third
    #: run, the same architecture on ImageNet.
    encoder_weights: str = ""
    #: Neighbouring slices per window, as the encoder's channels.
    group: int = 3
    dropout: float = 0.2
    #: Bias every head starts at, in logits. One value because the two targets sit at
    #: the same rate — Lateral Meniscus 25.1 %, Lateral OA 24.6 % — and log(0.25/0.75)
    #: is -1.1. A group whose rates differ would need one per target.
    prior_logit: float = -1.1

    #: Decay of an exponential moving average of the weights, kept alongside training
    #: and evaluated beside it. 0 turns it off.
    #:
    #: It decides nothing: what a fold saves and ships is still its last epoch. The
    #: average is a passenger, measured every epoch on the held-out fold and on the gold,
    #: so one batch of runs answers whether it is worth preferring instead of a coin
    #: toss before them.
    #:
    #: The window is what is being chosen, not the number: it spans about 1/(1-decay)
    #: steps. At 109 steps an epoch, 0.999 averages over roughly nine epochs — a stretch
    #: where this model genuinely still moves (0.065 of held-out AUC across epochs 6-30)
    #: rather than the last five, where it has already converged under `OneCycleLR` and
    #: varies by 0.004. Averaging five copies of one point is what made plain SWA
    #: pointless here.
    ema_decay: float = 0.0

    epochs: int = 30
    batch: int = 32
    lr: float = 3e-4
    weight_decay: float = 1e-4
    folds: int = 5
    seed: int = 0

    #: Read **one series per box** instead of all of them, choosing the first that is
    #: present. The caches carry two or three; measured, the leading one is there for
    #: 83-84 % of studies and the others for 13-37 %, so the usual case is one series
    #: whatever this says. What it changes is the minority: as separate channels the
    #: extra series are fed to every study that has them, which cost the MCL expert
    #: **0.0118**; as a fallback they are read only where nothing better exists, which
    #: takes coverage from 84 % to 99.8 %.
    one_channel: bool = False
    #: Best-first priority per entry of `rois`, as indices into that spec's `series`.
    #: Empty keeps the declared order. A **read-time** decision, deliberately not a
    #: `RoiSpec` field: the cache cuts every series it was asked for, and putting the
    #: order in the spec would make `cache.load` refuse the cache it cut, since it
    #: compares the whole spec for equality.
    #:
    #: Measured from the trained experts' own attention over series: five of seven lean
    #: 0.80-0.93 on the leading fat-suppressed series, so the declared order is right
    #: for them. `medial_meniscus` is the exception at 0.44/0.56, wanting
    #: non-fat-suppressed first — which is what `Attention`'s docstring predicts, a
    #: meniscal tear being signal inside the fibrocartilage.
    channel_order: tuple[tuple[int, ...], ...] = ()

    @property
    def specs(self) -> list[RoiSpec]:
        return [SPECS[name] for name in self.rois]

    @property
    def orders(self) -> list[tuple[int, ...] | None]:
        """One priority per ROI, `None` where the declared order stands."""

        if not self.channel_order:
            return [None] * len(self.rois)
        if len(self.channel_order) != len(self.rois):
            raise ValueError(
                f"channel_order has {len(self.channel_order)} entries for "
                f"{len(self.rois)} rois; a silent mismatch would read the wrong series")
        return [tuple(o) if o else None for o in self.channel_order]

    def to_dict(self) -> dict:
        out = asdict(self)
        out["targets"] = list(self.targets)
        out["rois"] = list(self.rois)
        out["channel_order"] = [list(o) for o in self.channel_order]
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
        if "channel_order" in data:
            data["channel_order"] = tuple(tuple(o) for o in data["channel_order"])
        unknown = set(data) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"config records fields this pipeline does not define: "
                             f"{sorted(unknown)}")
        return cls(**data)

    def replace(self, **changes) -> "ExpertConfig":
        return replace(self, **changes)
