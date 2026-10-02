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
    #: How far the box centre is moved along the **column** axis, in millimetres. What
    #: that axis *is* depends on the plane, and all three were measured: a coronal or
    #: axial series runs its columns toward the patient's left, so the shift is toward
    #: the **lateral** side and negative goes toward the midline; a sagittal series runs
    #: them toward the **posterior** on 5563 of 5563, so there the shift is backwards.
    #: A coronal crop needs it: measured on 70 studies, the landmark sits at 95 % of the
    #: knee's width from its medial edge, with 28 mm of skin in front of it and 85 mm of
    #: knee behind — so a box centred on it wastes half its width outside the patient.
    box_offset_mm: float = 0.0
    #: How far the box centre is moved toward the **top of the picture**, in
    #: millimetres; negative moves it down. Shifts the box along the *row* axis, where
    #: `box_offset_mm` shifts it along the column axis — two different directions, and a
    #: crop needs whichever one its plane gives it.
    #:
    #: What the top *is* depends on the plane, and all were measured rather than
    #: assumed: an axial series runs its rows toward the posterior on 3342 of 3342, so up
    #: is **anterior**; a coronal one runs them toward the inferior on 3815 of 3815 and a
    #: sagittal one on 5563 of 5563, so on both of those up is **superior**.
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

    #: A second landmark, when the region is defined **between two points** rather than
    #: around one. The crop then centres on their midpoint and the depth window spans
    #: from one to the other, inset by `depth_inset_mm` at each end — so its width
    #: follows the knee rather than being fixed.
    #:
    #: The cruciate needs this and nothing before it did. It sits in the notch, between
    #: the compartments, so no single point this project collects is near it — but the
    #: two meniscus points bracket it. Measured over the 294 studies carrying both, they
    #: are 50.8 mm apart in the median (p2.5 42.5, p97.5 64.3), so a window inset 8 mm
    #: at each end spans 35 mm in the median and ranges from 27 to 48. A fixed +/- 17 mm
    #: would be too wide on a small knee and too narrow on a large one.
    landmark2: str | None = None
    #: Millimetres dropped at each end of a two-landmark depth window. Ignored without
    #: `landmark2`.
    depth_inset_mm: float = 0.0

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
    #: The other compartment, and nothing else changes. Every number in `RoiSpec`'s
    #: defaults describes the size of a meniscus and the useful thickness around it, and
    #: a medial meniscus is of the same order — so the box, the depth and the series are
    #: taken verbatim from the lateral one.
    #:
    #: The asymmetry **did** look like it would have to flip. The depth is short toward
    #: the bowtie and long toward the notch because the landmark sits a slice or two off
    #: the peripheral end, and the medial compartment's periphery is the opposite edge
    #: of the knee. But `bowtie_direction` does not hardcode a side: it measures how far
    #: the imaged knee extends either way **from the point it is given**, and the nearer
    #: end wins. Measured on 70 studies carrying both points, it returns opposite
    #: directions for the two, **70 times out of 70**, with a comparable margin — median
    #: separation 3.15 against the lateral's 3.50, minimum 2.00 against 2.10. So it
    #: corrects itself and no field was needed.
    #: A little larger than the lateral box in both directions, because the medial
    #: meniscus is the larger of the two — a wide C against a nearly closed O — and its
    #: posterior horn is the broader one.
    #:
    #: 54 x 33 rather than the 52 x 30 that was asked for, because the sizes are
    #: quantised. Holding the lateral crop's resolution (48/224 = 3/14 mm per pixel) and
    #: requiring both output sides to divide by 14 leaves `box = 3 * patches`: the boxes
    #: available near 52 x 30 are 51 x 30, 54 x 30, 51 x 33 and 54 x 33, and nothing in
    #: between. Choosing the same millimetres per pixel is what makes this box
    #: comparable to the lateral one at all — what differs is extent, not sharpness.
    "medial_meniscus": RoiSpec(
        name="medial_meniscus", landmark="med_centre",
        box_w_mm=54.0, box_h_mm=33.0, out_w=252, out_h=154),
    #: The anterior cruciate ligament, defined **between** the two meniscus points
    #: rather than around one — the first region here that needs two. It sits in the
    #: intercondylar notch, between the compartments, so nothing this project collects
    #: is near it; but the two meniscus points bracket it, 50.8 mm apart in the median.
    #:
    #: The depth window starts 12 mm inside each of them and keeps everything between:
    #: measured over the 294 studies carrying both points, **26.8 mm in the median**,
    #: 18.5 on the narrowest knee and 40.3 on the widest, and never empty. A fixed
    #: half-extent cannot do that — it would be too wide on a small knee and too narrow
    #: on a large one.
    #:
    #: 12 rather than 8: at 8 the window ran to 48 mm on the widest knees, which is more
    #: than the notch is deep and spends slots on compartment rather than on cruciate.
    #:
    #: An adaptive window cannot fill a fixed number of slots, and 11 is the least bad of
    #: them. Measured over 120 studies, the window holds 4 to 12 acquired slices, median
    #: 8 — so 9 slots would fill on 42 % of studies but **throw acquired slices away on
    #: 26 %**, while 13 would fill on none. 11 pads 88 % and thins 3 %, and padding is
    #: the cheaper mistake: a window whose centre slot is padding never reaches the
    #: encoder, where a thinned stack has lost pixels that existed.
    #:
    #: 60 x 52 mm, 8 mm up and 4 mm back from the midpoint. "Up" and "back" are
    #: measurable and not figurative: a sagittal series runs its columns toward the
    #: posterior and its rows toward the inferior on 5563 of 5563, so up is superior and
    #: a positive column offset is backwards. The ligament runs from the back of the
    #: notch down and forwards, so its middle sits above the joint line the two meniscus
    #: points lie on.
    #:
    #: 210 x 182 px at 0.286 mm/px, not the meniscus crop's 0.214. The cruciate is a
    #: 10 mm structure, not a 1.5 mm tear, and the finer grid would cost 78 % more
    #: pixels to resolve something that does not need it. It also makes the box land on
    #: whole patches: at 4/14 mm per pixel the available sizes step by 4 mm, and 60 and
    #: 52 are both multiples of 4.
    #:
    #: **Expect little.** The rule the first three experts suggest — a crop helps when
    #: the lesion is small against the whole-knee view — puts this on the wrong side:
    #: the cruciate bundle is 26 px at the wide model's 0.387 mm/px, against 4 px for a
    #: meniscal tear that worked and 10 px for the collateral band that did not.
    "acl": RoiSpec(
        name="acl", landmark="lat_centre", landmark2="med_centre", plane="Sagittal",
        box_w_mm=60.0, box_h_mm=52.0, out_w=210, out_h=182,
        box_offset_mm=4.0, box_rise_mm=8.0, depth_inset_mm=12.0,
        lateral_mm=0.0, medial_mm=0.0, slots=11,
        series=(("Sagittal", "PD", True), ("Sagittal", "PD", False))),
    #: Tibiofemoral osteoarthritis, one spec per compartment, both hanging off the
    #: meniscus point that compartment already has — **no new annotation**. Projected
    #: onto a coronal slice the two points land one in each compartment, 57 mm apart on
    #: the study measured, and a box this size around each frames the joint line with
    #: bone above and below and the outer margin inside it.
    #:
    #: Coronal and not sagittal, because that is the plane the disease is read in: joint
    #: space narrowing is the height of the gap seen face on, and marginal osteophytes
    #: grow at the edges of the plateau, which the sagittal view cuts through rather
    #: than displays.
    #:
    #: 54 x 39 rather than the meniscus box's 54 x 33, and **not offset**. The meniscus
    #: coronal box is shifted 6 mm toward the periphery because it is looking for
    #: extrusion — the body displaced past the tibial margin. Osteoarthritis is not at
    #: the margin but across the compartment, and its subchondral oedema and sclerosis
    #: are *inside* the bone on both sides of the gap, so the box is centred and taller.
    #:
    #: Expect little. Lateral OA was tried on the meniscus crops and lost to the wide
    #: model by 0.028 with the interval excluding zero; the likeliest reason is not the
    #: region at all but that its pixels add **nothing** over the other eleven labels
    #: (comorbidity alone 0.8575 against the wide model's 0.8471, a negative margin).
    #: Medial OA is at +0.021, which is small but positive, and the crop costs one head
    #: on a region being built anyway.
    "medial_oa": RoiSpec(
        name="medial_oa", landmark="med_centre", plane="Coronal",
        box_w_mm=54.0, box_h_mm=39.0, out_w=252, out_h=182,
        lateral_mm=14.0, medial_mm=14.0, slots=9,
        series=(("Coronal", "PD", True), ("Coronal", "T2", True),
                ("Coronal", "PD", False))),
    "lateral_oa": RoiSpec(
        name="lateral_oa", landmark="lat_centre", plane="Coronal",
        box_w_mm=54.0, box_h_mm=39.0, out_w=252, out_h=182,
        lateral_mm=14.0, medial_mm=14.0, slots=9,
        series=(("Coronal", "PD", True), ("Coronal", "T2", True),
                ("Coronal", "PD", False))),
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
    #: The popliteal cyst, and the first region here cut for **coverage** rather than
    #: resolution. A cyst is 25 mm across, 64 px in the wide model's view — six times the
    #: 10 px where the collateral expert failed — so nothing is gained by zooming. What is
    #: gained is the slices the wide model's sampler throws away: its `band` keeps the
    #: central 60 % of each stack, which leaves a **median of 5.3 mm** of tissue medial of
    #: `med_centre` out of **26.4 mm acquired**, and 19.9 % of studies with 10 mm or more
    #: against 97.6 % acquired. A cyst hangs off the posteromedial capsule, between the
    #: semimembranosus and the medial head of the gastrocnemius, which is exactly there.
    #:
    #: Measured, not recalled. Over 72 reported-positive against 118 reported-negative
    #: studies, the frequency of bright voxels — fluid being the brightest thing on a
    #: fat-suppressed fluid-sensitive image, thresholded per study against its own
    #: distribution — peaks **44 mm posterior and 17 mm superior** to the landmark. The
    #: first draft of this box was put 12 mm *below* it by analogy with the collateral
    #: ligament, which the map contradicts.
    #:
    #: 90 x 75 mm at 252 x 210 holds **71 %** of that excess. Bigger holds more — 96 x 84
    #: holds 78.5 % — but at 0.429 mm/px, coarser than the wide model's own 0.387, and its
    #: posterior edge leaves the acquisition on a few per cent of studies. The box is
    #: forgiving, which matters once the landmark comes from a model rather than a click:
    #: moving it 6 mm in any direction costs under 2 points of capture.
    #:
    #: The depth window [-8, +20] mm holds **88 %** of the excess against 72 % for
    #: [-8, +12]; at the 3.3 mm median sagittal spacing that is 9 slices. `lateral_mm` is
    #: the distance toward the nearer end of the stack, which `bowtie_direction` finds on
    #: its own — from `med_centre` that end is the medial edge, so it is the field that
    #: opens the window onto the discarded slices.
    #:
    #: One number says whether the box is worth cutting at all: the bright-voxel fraction
    #: inside it, with nothing fitted, scores **0.6847**. The same count restricted to the
    #: slices the band keeps scores **0.5500**, and over the whole slice 0.6323. The signal
    #: is in what is thrown away, and it is specific to this region.
    #:
    #: Sagittal only, and fat-suppressed first: popliteal fat is abundant exactly here, so
    #: suppression is what separates the cyst from it. Coronal is cut worse than sagittal
    #: — the cyst sits at 0.78 of a coronal stack, outside the band on 43.3 % of studies —
    #: but its depth window would have to be offset 22 mm posteriorly, which this dataclass
    #: cannot express and `bowtie_direction` must not be asked to do. Axial needs no
    #: expert: the cyst falls at 0.50 of that stack, outside the band on 0.6 %.
    "baker": RoiSpec(
        name="baker", landmark="med_centre", plane="Sagittal",
        box_w_mm=90.0, box_h_mm=75.0, out_w=252, out_h=210,
        box_offset_mm=16.0, box_rise_mm=20.0,
        lateral_mm=20.0, medial_mm=8.0, slots=9,
        series=(("Sagittal", "PD", True), ("Sagittal", "PD", False))),
    #: The cyst again, larger in every direction the measurement allowed, and on the
    #: sequence the first spec forgot.
    #:
    #: **The forgotten slot.** `baker` took its two series straight from the meniscus
    #: specs — sagittal PD fat-suppressed, then PD without. That is right for a meniscus,
    #: where PD *is* the sequence, and wrong for a fluid collection sitting in popliteal
    #: fat: 766 studies (17.4 %) carry no sagittal PD fat-sat, and **478 of them do carry
    #: a sagittal T2 fat-sat** the spec never asked for, so they were read on a sequence
    #: where the fat is as bright as the cyst. Adding the slot moves 11 % of the corpus
    #: onto a suppressed sequence; it buys almost no coverage (99.8 % to 99.9 %), which
    #: is the opposite of why the collateral ligament got its third series.
    #:
    #: **The size.** Measured against the acquisition over 292 annotated studies, a box
    #: centred 16 mm behind and 20 mm above the landmark fits entirely inside the image
    #: on 100 % of studies at 90 x 75, **95.5 % at 110 x 95**, and 86.3 % at 120 x 100 —
    #: the posterior edge is what runs out, at a median of 81 mm of knee behind the
    #: point. 110 x 95 is the last size that does not pad an eighth of the corpus.
    #:
    #: It is deliberately past what the trivial probe prefers. That probe — the
    #: bright-voxel fraction in the box — falls monotonically as the box grows, from
    #: 0.716 at 64 x 56 to 0.664 at 126 x 105, because it is a **mean** and empty area
    #: dilutes it. The network is not a mean: it attends over windows, masked, once per
    #: target. What the probe measures that does transfer is the per-study capture, and
    #: that rises with size: at 90 x 75 no reported-positive study has under 20 % of its
    #: excess in the box, at 64 x 56 one in ten has under 10 %.
    #:
    #: **The depth.** [-8, +28] mm against the first spec's [-8, +20]: 93 % of the
    #: measured excess against 88 %, and the p10 of per-study capture rises from 0.26 to
    #: 0.29. Eleven slices at the 3.3 mm median spacing. This is the axis where growing
    #: is safest, because the attention can drop a window it does not want and the
    #: encoder's global average pool cannot drop a corner of an image.
    "baker_wide": RoiSpec(
        name="baker_wide", landmark="med_centre", plane="Sagittal",
        box_w_mm=110.0, box_h_mm=95.0, out_w=308, out_h=266,
        box_offset_mm=16.0, box_rise_mm=20.0,
        lateral_mm=28.0, medial_mm=8.0, slots=11,
        series=(("Sagittal", "PD", True), ("Sagittal", "T2", True),
                ("Sagittal", "PD", False))),
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
