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
    #: Which plane the crop is taken in. It decides what the axes *mean*, and so what
    #: the box and the depth window are for: on a sagittal slice the horizontal axis is
    #: anterior-posterior and the depth axis is medial-lateral; on a coronal slice they
    #: swap. Nothing else in this spec makes sense without it.
    plane: str = "Sagittal"

    # -- in plane ----------------------------------------------------------- #
    #: Box width, along the **anterior-posterior** axis. Measured on this corpus, the
    #: image's horizontal axis runs anterior to posterior on 170 studies out of 170,
    #: so a box wider than tall follows the meniscus rather than the frame.
    box_w_mm: float = 48.0
    #: Box height, along the superior-inferior axis. 27 rather than a rounder number so
    #: the pixels come out isotropic at the output size below: 48/224 = 27/126.
    box_h_mm: float = 27.0
    #: How far the box centre sits toward the **lateral** side of the landmark, in
    #: millimetres; negative is toward the midline. Zero on a sagittal crop, where the
    #: horizontal axis is anterior-posterior and there is no lateral to shift toward.
    #: A coronal crop needs it: measured on 70 studies, the landmark sits at 95 % of the
    #: knee's width from its medial edge, with 28 mm of skin in front of it and 85 mm of
    #: knee behind — so a box centred on it wastes half its width outside the patient.
    box_offset_mm: float = 0.0
    #: How far the box centre is moved toward the **top of the picture**, in
    #: millimetres; negative moves it down. Shifts the box along the *row* axis, where
    #: `box_offset_mm` shifts it along the column axis — two different directions, and a
    #: crop needs whichever one its plane gives it.
    #:
    #: What the top *is* depends on the plane, and both were measured rather than
    #: assumed: an axial series runs its rows toward the posterior on 3342 of 3342, so up
    #: is **anterior**; a coronal one runs them toward the inferior on 3815 of 3815, so
    #: up is **superior**. A sagittal crop has no use for it.
    #:
    #: The patellofemoral box needs it. Its landmark is the joint line, with the patella
    #: in front and the trochlea behind, and the patella is the half that gets clipped:
    #: centred, a 48 mm box reaches 24 mm forward and cuts the patella's anterior cortex
    #: on the studies where it sits high. Raising it 8 mm costs an empty strip past the
    #: skin on 60 % of studies, but a **median of 3.1 mm** of one — 6 % of the box, 14 %
    #: at the 90th centile — against losing the bone the osteophytes grow on.
    #:
    #: The collateral ligament box needs it the other way, hence the sign. Its landmark
    #: is the joint line; the femoral origin is two to three centimetres above and the
    #: tibial insertion five to seven below — but the field of view does not reach that
    #: far down. Measured over 209 annotated studies, a box descending 30 mm below the
    #: click stays inside the image on 100 % of them, 40 mm on 98.1 %, 50 mm on 89 % and
    #: 60 mm on **67 %**. So it is given most of its height downward, and no more than
    #: was actually imaged.
    box_rise_mm: float = 0.0
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

    @property
    def symmetric_depth(self) -> bool:
        """Whether the depth window needs a direction at all.

        A sagittal crop does: the landmark sits a slice or two off the bowtie, so the
        window is short one way, and which way has to be read off the pixels. A coronal
        crop does not: its depth axis is anterior-posterior and the landmark is the
        midpoint between the two horns by its own definition — measured, 43.8 mm of knee
        one way against 32.8 the other, against 3.7x on the sagittal axis.
        """

        return abs(self.lateral_mm - self.medial_mm) < 1e-9

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
#:
#: The `_d*` entries vary the **depth only** — the box is settled at 48 x 27 mm and they
#: all share it, so what they compare is how much of the compartment a branch should see
#: along the stack and at what density. They exist because that pair of numbers is the
#: one part of the ROI no measurement of ours can settle: an annotation is a point, and a
#: point says nothing about how far the meniscus extends around it.
SPECS = {
    "lateral_meniscus": RoiSpec(),
    #: The coronal view of the same compartment, hanging off the same landmark — no new
    #: annotation, because the point is in patient millimetres and the other series of
    #: the study are read in those same millimetres. What it adds is **extrusion**: the
    #: body displaced past the tibial margin, which a sagittal slice cannot show at all.
    #:
    #: That reuse is an **assumption, not a guarantee**, and the scanner says so: these
    #: series do not share a FrameOfReferenceUID — distinct on 14 of 14 series of one
    #: study, while StudyInstanceUID stayed identical across the same 14 under an
    #: anonymiser that regenerated both, so the original frames genuinely differed. In
    #: DICOM that is an explicit refusal to promise the two are registered; what holds
    #: them together is only that nothing moved the table between acquisitions.
    #:
    #: Measured instead of assumed, on the leg silhouette against air so that fat
    #: suppression cannot be read as displacement, fields of view intersected first:
    #: coronal PD fat-suppressed against coronal T1 agree to **0.82 mm in the median**,
    #: against 0.88 mm for a same-weighting sagittal pair as control — but the tail is
    #: worse, p90 5.6 mm and max 15.6 mm against the control's 3.7 and 6.2. A minority
    #: of studies did move, and 15 mm on a 48 x 30 mm box is most of a meniscus.
    #:
    #: The sagittal-to-coronal transfer itself is **not validated**: the only measures
    #: tried for it read 4.5 mm on a same-plane control that is known to be sub-millimetre,
    #: so they cannot resolve what they would need to. It would take a handful of coronal
    #: clicks compared against the projected point to settle, and it predates the T1 slot.
    "lateral_meniscus_coronal": RoiSpec(
        name="lateral_meniscus_coronal", plane="Coronal",
        box_w_mm=48.0, box_h_mm=30.0, out_w=224, out_h=140,
        box_offset_mm=-6.0,          # 18 mm toward the margin, 30 toward the root
        lateral_mm=14.0, medial_mm=14.0, slots=9,
        #: Three, not the sagittal pair. Mirroring the sagittal choice was the obvious
        #: thing and it was the wrong one: coronal PD fat-suppressed covers 84.0 % of
        #: studies but coronal PD without it only 13.3 %, so the second slot is empty
        #: four studies in five. Coronal T1 is on 64.1 %, and 273 studies (6.2 %) have
        #: it with no coronal PD at all — under the pair they contributed nothing from
        #: this plane. Reading all three takes coverage from 88.3 % to 94.5 %, and T1 is
        #: the sequence that shows bone, which is what osteoarthritis is.
        series=(("Coronal", "PD", True), ("Coronal", "PD", False),
                ("Coronal", "T1", False))),
    #: The patellofemoral joint, on the axial plane this pipeline had never opened.
    #: Its own landmark -- `pf_centre`, the middle of the joint space -- because the
    #: meniscus point is 60 mm away in another compartment and says nothing about this
    #: one.
    #:
    #: 56 x 48 mm, raised 8 mm toward the front. The width crosses both facets: the
    #: patella measures about 40 mm, so 56 leaves 8 mm either side. The height was
    #: chosen by eye on four studies and the rise with it -- centred, the box clipped
    #: the patella's anterior cortex.
    #:
    #: 196 x 168 px rather than 224: both divide by 14, so a patch embedding keeps every
    #: row, and 0.286 mm/px is already finer than the 0.31 mm/px median of the axial
    #: acquisitions. Going to 224 would resample past what was acquired.
    #:
    #: **The depth is a guess and is meant to be swept**, the way the meniscus depth
    #: was. Symmetric, because the landmark is at mid-patella by its definition and the
    #: joint runs as far above it as below -- there is no bowtie here to be short
    #: toward.
    "pf_oa": RoiSpec(
        name="pf_oa", landmark="pf_centre", plane="Axial",
        box_w_mm=56.0, box_h_mm=48.0, out_w=196, out_h=168,
        box_rise_mm=8.0,
        lateral_mm=16.0, medial_mm=16.0, slots=9,
        series=(("Axial", "PD", True), ("Axial", "T2", True))),
    #: The medial collateral ligament, on the coronal plane, hanging off its own point
    #: — `mcl_centre`, the ligament where it crosses the joint line. Nothing else this
    #: project has collected is within 50 mm of it: placing it from the lateral meniscus
    #: landmark and the limb's medial skin edge was tried and lands in the subcutaneous
    #: fat, because the thickness between skin and ligament varies from patient to
    #: patient. See docs/atlas/mcl.html.
    #:
    #: 32 x 80 mm, **taller than wide** — the inverse of the meniscus and patellar boxes,
    #: because the ligament is a long thin band rather than a thing to frame. Dropped
    #: 12 mm toward the tibia, so 28 mm above the joint line and 52 below: the femoral
    #: origin is two to three centimetres up, the tibial insertion five to seven down,
    #: and the field of view runs out before the latter on a third of studies.
    #:
    #: Shifted 5 mm **toward the knee** — that is what `box_offset_mm` does from a
    #: medial landmark, since lateral is the way back to the joint. Centred exactly on
    #: the click, half the width sits in subcutaneous fat; 5 mm buys bone and loses
    #: nothing the ligament occupies.
    #:
    #: 112 x 280 px: both divide by 14, the aspect matches 32:80 exactly so the pixels
    #: are square, and 0.286 mm/px is already finer than the 0.31 mm/px median coronal
    #: acquisition.
    #:
    #: **The depth is a guess and is meant to be swept**, like the two before it.
    "mcl": RoiSpec(
        name="mcl", landmark="mcl_centre", plane="Coronal",
        box_w_mm=32.0, box_h_mm=80.0, out_w=112, out_h=280,
        box_offset_mm=5.0, box_rise_mm=-12.0,
        lateral_mm=12.0, medial_mm=12.0, slots=7,
        #: Three, and the third is NOT fat-suppressed. The pair of fat-suppressed
        #: sequences leaves 197 studies (4.5 %) with no coronal at all, and a study with
        #: no pixels still gets scored: every window is masked, the attention falls back
        #: to zero and the head emits its bias, so they arrive as one block of ties.
        #: Measured on the patellofemoral run, 1.5 % uncovered cost 0.0032 of AUC.
        #:
        #: **96.4 % of those 197 have a coronal PD without fat suppression**, and adding
        #: it takes coverage to 99.8 %. It is the right sequence to fall back on rather
        #: than merely the common one: PD without suppression is where the band itself
        #: reads best, the surrounding fat giving it contrast, so the ligament's
        #: thickness and continuity are plain. What it shows less well is oedema, which
        #: is the low-grade sprain — it is a worse sequence than the fat-suppressed ones
        #: and a far better one than nothing.
        #:
        #: Coronal T1 was the other candidate and was refused: +1.0 point of coverage,
        #: and T1 does not show oedema at all, so it would add a slot in which the thing
        #: being looked for is largely invisible.
        series=(("Coronal", "PD", True), ("Coronal", "T2", True),
                ("Coronal", "PD", False))),
    #: What the first collateral run was actually cut under — the two fat-suppressed
    #: series only, 95.5 % of studies. Kept so `experiments/expert_mcl_fsonly.json`
    #: still reproduces the number it published.
    "mcl_fsonly": RoiSpec(
        name="mcl_fsonly", landmark="mcl_centre", plane="Coronal",
        box_w_mm=32.0, box_h_mm=80.0, out_w=112, out_h=280,
        box_offset_mm=5.0, box_rise_mm=-12.0,
        lateral_mm=12.0, medial_mm=12.0, slots=7,
        series=(("Coronal", "PD", True), ("Coronal", "T2", True))),
    #: The depth sweep for the patellofemoral box, which the annotations cannot settle:
    #: a point says nothing about how far the joint extends around it. What *was*
    #: measured, over the 200 annotated stacks, is how many slots each fills --
    #: 16/9 fills every slot on 56 % and pads 44 %, 16/7 fills 93 % but throws away
    #: acquired slices on 60 %. The reference keeps the 4.0 mm slot pitch, nearest the
    #: 3.6 mm median acquisition; these three say whether that mattered.
    "pf_oa_d12": RoiSpec(name="pf_oa_d12", landmark="pf_centre", plane="Axial",
                         box_w_mm=56.0, box_h_mm=48.0, out_w=196, out_h=168,
                         box_rise_mm=8.0, lateral_mm=12.0, medial_mm=12.0, slots=7,
                         series=(("Axial", "PD", True), ("Axial", "T2", True))),
    # the cheap one: same extent, coarser grid, five windows instead of seven
    "pf_oa_s7": RoiSpec(name="pf_oa_s7", landmark="pf_centre", plane="Axial",
                        box_w_mm=56.0, box_h_mm=48.0, out_w=196, out_h=168,
                        box_rise_mm=8.0, lateral_mm=16.0, medial_mm=16.0, slots=7,
                        series=(("Axial", "PD", True), ("Axial", "T2", True))),
    # the whole patella and then some, at the same pitch
    "pf_oa_d20": RoiSpec(name="pf_oa_d20", landmark="pf_centre", plane="Axial",
                         box_w_mm=56.0, box_h_mm=48.0, out_w=196, out_h=168,
                         box_rise_mm=8.0, lateral_mm=20.0, medial_mm=20.0, slots=11,
                         series=(("Axial", "PD", True), ("Axial", "T2", True))),
    # 50 % more extent at the same density
    "lateral_meniscus_d24": RoiSpec(name="lateral_meniscus_d24",
                                    lateral_mm=6.0, medial_mm=18.0, slots=7),
    # double the extent: does the far side of the compartment carry anything?
    "lateral_meniscus_d32": RoiSpec(name="lateral_meniscus_d32",
                                    lateral_mm=8.0, medial_mm=24.0, slots=9),
    # the same peripheral margin, everything extra spent toward the notch
    "lateral_meniscus_m20": RoiSpec(name="lateral_meniscus_m20",
                                    lateral_mm=4.0, medial_mm=20.0, slots=7),
    # the opposite test: more peripheral margin, nothing else changed
    "lateral_meniscus_l6": RoiSpec(name="lateral_meniscus_l6",
                                   lateral_mm=6.0, medial_mm=12.0, slots=5),
}
