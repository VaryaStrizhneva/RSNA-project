"""Every decision that changes what the landmark model sees.

Deliberately separate from `rsna.config.Config`, which describes the classifier's cache.
The two share no pixels: the landmark model reads the DICOM directly, keeps the acquired
slices rather than resampling across them, and normalises no laterality at all — its
answer is a point in the patient, which is the same point whichever way the stack was
stored. Folding these fields into `Config` would make every classifier manifest carry
settings it never used.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace


@dataclass(frozen=True)
class LandmarkConfig:
    """One reading of an annotated study, end to end."""

    # -- in plane ----------------------------------------------------------- #
    #: Side of the resampled slice, in pixels.
    img: int = 256
    #: Field of view kept, in millimetres. 180 is what the annotation bundle rendered,
    #: so the model's field of view is the one the annotator actually looked at.
    fov_mm: float = 180.0

    # -- through plane ------------------------------------------------------ #
    #: How many slice slots the input tensor has. The acquired slices are placed in it
    #: and the rest is padding: **nothing is interpolated between slices**. Measured
    #: over the 320 annotated studies, 277 of the 279 2D series hold 48 or fewer, and
    #: the two that do not exceed it by two slices at the periphery, outside the joint.
    slices: int = 48
    #: Spacing a deep series is decimated *towards*, in millimetres. Only 3D series are
    #: touched, and only by an integer stride, so no slice is duplicated or invented —
    #: 320 slices at 0.4 mm becomes 40 at 3.2 mm. Chosen as the corpus median so the
    #: decimated 3D series land inside the 2D spread (2.20-5.50 mm) rather than beside
    #: it.
    decimate_to_mm: float = 3.3

    # -- the target --------------------------------------------------------- #
    #: Names of the landmarks predicted, one output channel each. One encoder with K
    #: heads is 14.9 M parameters against 38.9 M for K separate models, and one forward
    #: pass instead of K.
    points: tuple[str, ...] = ("lat_centre",)
    #: Width of the Gaussian written into the target heatmap, in millimetres. A 40 mm
    #: ROI still contains the meniscus if the landmark is within 12 mm, and a third of
    #: the tolerance is a reasonable width for the thing being learnt.
    sigma_mm: float = 4.0
    #: How much coarser the heatmap is than the input, in plane. The depth axis is not
    #: downsampled: it has 48 slots to begin with.
    stride: int = 4

    # -- the encoder -------------------------------------------------------- #
    #: A timm model name, created with `features_only=True`. Not a key into
    #: `rsna.model.encoders`: that registry drives a backbone as a whole-image encoder
    #: producing one vector, and a heatmap needs a multi-scale feature map instead —
    #: a different capability that only a convolutional model has. See `network.py`.
    #:
    #: ResNet-18 is 11.2 M parameters with stages at strides 4, 8, 16 and 32. Bigger
    #: would memorise ~300 annotations; the design note puts the useful range at 4-15 M.
    encoder_variant: str = "resnet18"

    @property
    def mm_per_px(self) -> float:
        """Millimetres per pixel in the resampled slice, the same for every study."""

        return self.fov_mm / self.img

    @property
    def cell_mm(self) -> float:
        """Millimetres per heatmap cell, in plane."""

        return self.mm_per_px * self.stride

    @property
    def grid(self) -> int:
        """Side of the heatmap, in cells."""

        if self.img % self.stride:
            raise ValueError(f"img {self.img} is not a multiple of stride {self.stride}")
        return self.img // self.stride

    def to_dict(self) -> dict:
        out = asdict(self)
        out["points"] = list(self.points)
        return out

    @classmethod
    def from_dict(cls, data: dict) -> "LandmarkConfig":
        data = dict(data)
        if "points" in data:
            data["points"] = tuple(data["points"])
        unknown = set(data) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"config records fields this pipeline does not define: "
                             f"{sorted(unknown)}")
        return cls(**data)

    def replace(self, **changes) -> "LandmarkConfig":
        return replace(self, **changes)
