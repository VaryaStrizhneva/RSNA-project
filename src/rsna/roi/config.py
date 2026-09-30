"""What a region of interest is, for one pathology.

Every number here was chosen against a measurement or is marked as a judgement. The
distinction matters more than usual: a ROI that is slightly wrong does not fail, it
quietly trains a model on the wrong pixels.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace


@dataclass(frozen=True)
class RoiSpec:
    """One crop, in millimetres, around one landmark."""

    #: Which pathology this ROI is for, and which landmark it hangs off.
    name: str = "lateral_meniscus"
    landmark: str = "lat_centre"

    # -- in plane ----------------------------------------------------------- #
    #: Box width, along the **anterior-posterior** axis. Measured on this corpus, the
    #: image's horizontal axis runs anterior to posterior on 170 studies out of 170,
    #: so a box wider than tall follows the meniscus rather than the frame.
    box_w_mm: float = 48.0
    #: Box height, along the superior-inferior axis. 27 rather than a rounder number so
    #: the pixels come out isotropic at the output size below: 48/224 = 27/126.
    box_h_mm: float = 27.0
    out_w: int = 224
    out_h: int = 126
    #: Both output sides must divide by this. A convolutional encoder does not care;
    #: a ViT does, and it does not complain: a DINOv2 patch embedding is a stride-14
    #: convolution, so 121 rows give 8 patches and the last 9 rows — 7 % of the ROI —
    #: are dropped with nothing raised. Set it to 1 for an encoder that has no patch.
    patch: int = 14

    # -- through plane ------------------------------------------------------- #
    #: Millimetres kept toward the **bowtie**, the peripheral end. Short on purpose:
    #: the landmark sits a slice or two off it, so there is little meniscus that way.
    lateral_mm: float = 4.0
    #: Millimetres kept toward the intercondylar notch, where the rest of it is.
    medial_mm: float = 12.0
    #: Depth slots in the tensor. Measured over 320 studies at 4/12 mm: 240 fill all
    #: five exactly, 49 need padding and 31 have slices to spare and are thinned.
    slots: int = 5
    #: A 3D series is thinned towards this before anything else, by an integer stride —
    #: otherwise 16 mm at 0.4 mm spacing is forty slices for five slots.
    decimate_to_mm: float = 3.3

    #: Which series the ROI is taken from, as (plane, weighting, fat-suppressed).
    #: Several at once, each with its own presence mask: SAG_PD_FS covers 81.3 % of
    #: studies and SAG_PD_NOFS 36.3 %, but at least one of the two covers 99.7 %.
    series: tuple[tuple[str, str, bool], ...] = (
        ("Sagittal", "PD", True), ("Sagittal", "PD", False))

    @property
    def mm_per_px(self) -> float:
        return self.box_w_mm / self.out_w

    def __post_init__(self) -> None:
        if self.patch > 1 and (self.out_w % self.patch or self.out_h % self.patch):
            raise ValueError(
                f"output {self.out_w}x{self.out_h} is not a whole number of "
                f"{self.patch}-pixel patches: a patch embedding would keep "
                f"{self.out_h // self.patch * self.patch} of {self.out_h} rows and say "
                f"nothing.")
        if abs(self.box_w_mm / self.out_w - self.box_h_mm / self.out_h) > 1e-6:
            raise ValueError(
                f"anisotropic pixels: {self.box_w_mm}/{self.out_w} != "
                f"{self.box_h_mm}/{self.out_h}. A crop whose millimetres per pixel "
                f"differ by axis silently stretches every study by the same wrong "
                f"factor, which no augmentation undoes.")

    def to_dict(self) -> dict:
        out = asdict(self)
        out["series"] = [list(s) for s in self.series]
        return out

    @classmethod
    def from_dict(cls, data: dict) -> "RoiSpec":
        data = dict(data)
        if "series" in data:
            data["series"] = tuple(tuple(s) for s in data["series"])
        unknown = set(data) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"spec records fields this pipeline does not define: "
                             f"{sorted(unknown)}")
        return cls(**data)

    def replace(self, **changes) -> "RoiSpec":
        return replace(self, **changes)


#: The specs this pipeline knows. One per pathology group; six more to write.
SPECS = {"lateral_meniscus": RoiSpec()}
