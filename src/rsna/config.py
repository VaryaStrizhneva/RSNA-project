"""Every decision that changes what a pixel is, in one place.

Two configurations that agree on array shapes but disagree on how a slice is chosen
produce different arrays of identical shape. Nothing downstream can detect that, so
the configuration is carried explicitly rather than read from module globals, and it
is written into a weights package so inference reproduces the reading a model was
fitted under.

Ported from pilkwang's public baseline notebook, which reaches these constants by
measurement over this corpus; see docs/references.md. The values are kept, the
mechanism is not: the original mutates globals (`adopt_config_globals`), this passes
an object.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Literal

TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
    "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture",
]

#: A slot is (name, plane, fluid, fat-suppressed). `fluid=None` means the weighting is
#: not conditioned on.
Slot = tuple[str, str, bool | None, bool]

#: Plane x recovered weighting x recovered fat suppression.
#:
#: The competition ships `Fluid_Sensitive` and `Fat_Suppression`, but they are equal on
#: every one of the 24,371 training series (see docs/data.md), so as delivered they
#: carry one axis rather than two. These slots use the axes recovered from the DICOM
#: headers by `dicom.headers.annotate` instead.
SLOTS_RECOVERED: list[Slot] = [
    ("SAG_FLUID_FS", "Sagittal", True, True),
    ("COR_FLUID_FS", "Coronal", True, True),
    ("AX_FLUID_FS", "Axial", True, True),
    ("SAG_FLUID_NOFS", "Sagittal", True, False),
    ("COR_T1", "Coronal", False, False),
    ("SAG_T1", "Sagittal", False, False),
]

#: Plane x the single axis the delivered flags carry. Kept as a switch so the slot
#: definition can be varied while everything else is held fixed. Under this scheme a
#: structural slot mixes T1 with non-fat-suppressed PD/T2, which have very different
#: tissue contrast.
SLOTS_PUBLIC: list[Slot] = [
    ("SAG_FLUID", "Sagittal", None, True),
    ("COR_FLUID", "Coronal", None, True),
    ("AX_FLUID", "Axial", None, True),
    ("SAG_STRUCT", "Sagittal", None, False),
    ("COR_STRUCT", "Coronal", None, False),
    ("AX_STRUCT", "Axial", None, False),
]

#: How many parts the per-slot feature is concatenated from. The encoder emits one
#: vector per token; a slot feature is a fixed summary of that grid.
POOL_PARTS = {"cls_mean": 2, "cls_mean_focal": 3}

#: Which slots a diagnosis is read on, as a fixed tilt on the attention logits rather
#: than a learned parameter. Indices are into `SLOTS_RECOVERED`. Because it is a
#: buffer in the state dict, it is part of a member's definition and must be
#: reproduced exactly for its weights to mean anything.
SLOT_PRIOR_TABLE = {
    "ACL": (0, 3, 5), "MCL": (1, 4),
    "Medial Meniscus": (0, 1, 3, 4), "Lateral Meniscus": (0, 1, 3, 4),
    "Medial OA": (1, 4, 5), "Lateral OA": (1, 4, 5),
    "PF OA": (0, 2, 5), "Effusion": (0, 2), "Synovitis": (0, 2),
    "Baker's": (0,), "Contusion": (0, 1, 2), "Fracture": (0, 1, 2, 4, 5),
}

#: exp(0.55) makes a preferred slot about 1.73x the weight of an unpreferred one:
#: it biases the softmax without ever excluding a slot.
SLOT_PRIOR_STRENGTH = 0.55

#: Two GPUs disagree by about 1e-5 on the same computation; a preprocessing change
#: moves the output by order one. The tolerance sits between them.
FINGERPRINT_TOL = 2e-3


OrderRule = Literal["normal", "dominant_axis"]
LateralityRule = Literal["centre", "corner_x"]
DecodeFillRule = Literal["nearest", "zero"]


@dataclass(frozen=True)
class PixelRules:
    """The choices that change which pixels a slot holds, without changing its shape.

    A model fitted under one reading and fed another gets pixels its weights never saw,
    with every dimension still agreeing. These are named so a weights package can state
    which reading it needs, and so an unrecognised value can be refused rather than
    defaulted.
    """

    #: `normal` projects the slice position onto the slice normal; `dominant_axis`
    #: sorts on the raw patient coordinate of the most-varying axis. On sagittal
    #: series the two orders come out exactly reversed.
    order: OrderRule = "normal"
    #: `centre` thresholds the x of the image centre, `corner_x` the x of the image
    #: corner — up to half a field of view apart, enough to reverse the side on a knee
    #: scanned near the midline.
    laterality: LateralityRule = "centre"
    #: Whether an empty structural slot may be filled by relaxing the weighting.
    #: Rejected by default: it puts one series in two slots and makes the presence mask
    #: assert a sequence that was never acquired.
    slot_fallback: bool = False
    #: What a slice that will not decode becomes. `nearest` copies the closest slice
    #: that did; `zero` blacks it out, which propagates to the whole slot.
    decode_fill: DecodeFillRule = "nearest"


@dataclass(frozen=True)
class Config:
    """One reading of the corpus, end to end."""

    # -- pixels ------------------------------------------------------------- #
    #: Side of the decoded square, in pixels.
    img: int = 336
    #: Field of view kept around the joint centre.
    #:
    #: The crop must be smaller than the smallest field of view in the corpus or it
    #: silently does nothing. Measured over every training series, the acquired field
    #: of view has median 160 mm and runs 70-320: a 160 mm crop is larger than the
    #: image in 60% of series and is skipped for all of them, leaving their physical
    #: scale unnormalised. 130 mm is below the field of view of 99.6% of series and
    #: still contains the joint.
    crop_mm: float = 130.0
    #: Slices stacked as the channels of one encoder input.
    group: int = 3
    #: Slices decoded per slot.
    slices: int = 3
    #: Fraction of the ordered stack sampled across, avoiding the empty ends.
    band: tuple[float, float] = (0.20, 0.80)

    # -- structure ---------------------------------------------------------- #
    slots: list[Slot] = field(default_factory=lambda: list(SLOTS_RECOVERED))
    rules: PixelRules = field(default_factory=PixelRules)

    # -- laterality --------------------------------------------------------- #
    #: Inside this distance from the midline the side is not readable from geometry,
    #: and the study is left unresolved rather than guessed. Measured against the
    #: tagged half of the corpus, the geometric rule is right 97% of the time overall
    #: and no better than chance inside 20 mm.
    lat_min_offset_mm: float = 20.0
    #: The dead zone the legacy corner-x rule was fitted with.
    lat_legacy_offset_mm: float = 5.0

    # -- fitting ------------------------------------------------------------ #
    seed: int = 2026
    epochs: int = 10
    batch_studies: int = 8
    lr_head: float = 1e-3
    #: The encoder is adapted, not retrained.
    lr_backbone: float = 8e-6
    unfreeze_last: int = 6
    weight_decay: float = 0.02
    eval_batch: int = 8
    #: Rigid jitter. No flip: a horizontal one would reintroduce the nuisance axis the
    #: laterality normalisation removes, and a vertical one moves the input off the
    #: distribution entirely — a knee is acquired in a canonical orientation, and where
    #: a finding sits in the frame is information (a Baker cyst is identified by lying
    #: in the popliteal fossa).
    aug_rot_deg: float = 8.0
    aug_scale: float = 0.08
    aug_shift: float = 0.05
    aug_intensity: float = 0.10
    #: Loss weight for the 58 expert-labelled studies, against 0.25-1.0 for
    #: report-derived ones.
    gold_weight: float = 3.0
    n_folds: int = 5

    @property
    def n_slot(self) -> int:
        return len(self.slots)

    @property
    def slot_names(self) -> list[str]:
        return [s[0] for s in self.slots]

    @property
    def n_group(self) -> int:
        return max(self.slices // self.group, 1)

    @property
    def mm_per_pixel(self) -> float:
        """Physical scale of one pixel. A feature narrower than two of these cannot
        survive sampling, whatever the encoder."""

        return self.crop_mm / self.img

    def cache_tag(self) -> str:
        """The name a decoded cache is stored under.

        It must name everything that decides the pixels, not only their dimensions —
        otherwise a second configuration attaches to the first one's file and trains
        against pixels it never asked for, with nothing reporting a mismatch.
        """

        import hashlib
        import json

        tag = (f"{self.img}px_{self.slices}sl_{int(self.crop_mm)}mm_"
               f"{self.band[0]:.2f}-{self.band[1]:.2f}")
        if self.rules != PixelRules():
            digest = hashlib.md5(
                json.dumps(asdict(self.rules), sort_keys=True).encode()
            ).hexdigest()[:6]
            tag += "_" + digest
        return tag

    def to_dict(self) -> dict:
        """Serialisable form, for the manifest that travels with a weights package."""

        out = asdict(self)
        out["slots"] = [list(s) for s in self.slots]
        out["band"] = list(self.band)
        return out

    @classmethod
    def from_dict(cls, data: dict) -> "Config":
        data = dict(data)
        data["slots"] = [tuple(s) for s in data.get("slots", SLOTS_RECOVERED)]
        data["band"] = tuple(data.get("band", (0.20, 0.80)))
        if isinstance(data.get("rules"), dict):
            data["rules"] = PixelRules(**data["rules"])
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"config records fields this pipeline does not define: "
                             f"{sorted(unknown)}")
        return cls(**{k: v for k, v in data.items() if k in known})

    def replace(self, **changes) -> "Config":
        return replace(self, **changes)
