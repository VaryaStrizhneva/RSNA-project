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
    #: **A floor is missing, and it bites on 11 studies.** The window is
    #: `separation - 2 x 12 mm`, which is calibrated on a median separation of 52.6 mm.
    #: When the landmark model puts the two points too close together -- the failure mode
    #: visible on `docs/atlas/landmarking/` -- the fixed inset amplifies the error instead
    #: of absorbing it: a separation of 25.6 mm leaves a **1.6 mm** window, and the box
    #: then holds a single acquired slice, so one live window of nine reaches the encoder.
    #:
    #: Measured over the corpus: separation median 52.6 mm, 1st centile 41.1, minimum 2.0.
    #: **11 studies of 4407 end with a window under 10 mm and 8 under 5 mm** -- 0.25 %, so
    #: this does not move an AUC and it did not explain `convnext_nano` losing 0.0051 on
    #: this column. It is still wrong, and the fix is a floor: below about 16 mm, take
    #: 16 mm centred on the midpoint rather than the derived width. Not done yet.
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
    #: **The inset this spec's own documentation describes.** `depth_inset_mm` reads 12
    #: on the shipped `acl` while the field's comment above says *"a window inset 8 mm at
    #: each end spans 35 mm in the median and ranges from 27 to 48"* -- and 35 mm is what
    #: 8 gives on the 51.4 mm separation measured here. So the prose records the intent
    #: and the 12 drifted in after it without the comment following.
    #:
    #: Restoring 8 is worth more than any other window change on the books, and it is the
    #: cheapest. Measured: the shipped window spans **27.4 mm** in the median and asks
    #: **11 slots** of it, a pitch of **2.74 mm** against a sagittal through-plane
    #: spacing of 3.30 -- finer than the acquisition, so it *cannot* be filled. Expected
    #: fill is 85 % and **measured fill is 7.82 of 11, 71 %**, with every slot full on
    #: only **11.3 %** of studies. That is the worst-filled window of the set: the
    #: meniscus boxes sit at 93 %, `pf_oa` at 87 %, the two OA boxes at 84 %.
    #:
    #: At 8 mm the window is 35.4 mm, the pitch 3.54 -- just above the acquisition, where
    #: the meniscus windows sat before their sweep -- and 11.7 slices are available for
    #: 11 slots, so it fills. About **3.5 more real slices per study**, against 1.8 for
    #: the OA enlargement and 2 for `pf_oa_wide_d40`.
    #:
    #: **Nothing else moves**: same box in plane, same resolution, same slot count, same
    #: series. On the atlas page the candidate rectangle therefore sits exactly on the
    #: shipped one, and what differs is which *slices* carry it -- which is the whole
    #: content of this change.
    "acl_inset8": RoiSpec(
        name="acl_inset8", landmark="lat_centre", landmark2="med_centre",
        plane="Sagittal",
        box_w_mm=60.0, box_h_mm=52.0, out_w=210, out_h=182,
        box_offset_mm=4.0, box_rise_mm=8.0, depth_inset_mm=8.0,
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
    #: **The medial compartment's frame and window, both enlarged.** 54 x 39 to **60 x 42 mm** -- the nearest expressible box to +10 % on
    #: each axis, since holding 3/14 mm per pixel with both output sides divisible by 14
    #: forces `box = 3 * k` and 59.4 and 42.9 are not expressible -- and the depth from
    #: 28 mm over 9 slots to **35 over 11**, the 3.50 mm pitch unchanged.
    #:
    #: **The pitch is deliberately not touched, and the measurement is why.** Coronal
    #: through-plane spacing is **3.50 mm** in the median (p5 3.00, p95 5.07), which is
    #: this grid's pitch exactly. Holding 28 mm and spending the two extra slots on
    #: density instead would put the pitch at 2.80 and drop the expected fill from 93 %
    #: to **78 %** -- padding, not slices. Lengthening to 35 mm at 3.50 keeps 92 % and
    #: buys about **1.8 more real slices** (10.16 of 11 against 8.33 of 9). Measured fill
    #: today is 7.55 of 9, so 0.78 of a slot is the window genuinely overrunning the
    #: acquisition and the rest is how far apart the slices are.
    #:
    #: **Unlike the patellofemoral box, nothing here is wasted.** Measured row by row and
    #: column by column over 1200 studies, tissue sits between 0.76 and 0.83 across the
    #: height and 0.74 and 0.89 across the width -- no dead band on any of the four
    #: edges, where `pf_oa`'s top edge carries 0.13 because its +8 mm rise pushes it past
    #: the skin. So an enlargement here reaches real pixels in every direction, and the
    #: box is centred where it should be.
    #:
    #: Memory is the practical risk and not the anatomy: this column already trains at
    #: **batch 16**, and nine windows at 280 x 196 px is about 1.5x what it costs now.
    "medial_oa_wide_d35": RoiSpec(
        name="medial_oa_wide_d35", landmark="med_centre", plane="Coronal",
        box_w_mm=60.0, box_h_mm=42.0, out_w=280, out_h=196,
        lateral_mm=17.5, medial_mm=17.5, slots=11,
        series=(("Coronal", "PD", True), ("Coronal", "T2", True),
                ("Coronal", "PD", False))),
    "lateral_oa": RoiSpec(
        name="lateral_oa", landmark="lat_centre", plane="Coronal",
        box_w_mm=54.0, box_h_mm=39.0, out_w=252, out_h=182,
        lateral_mm=14.0, medial_mm=14.0, slots=9,
        series=(("Coronal", "PD", True), ("Coronal", "T2", True),
                ("Coronal", "PD", False))),
    #: **The lateral compartment, the same change as the medial one.** 54 x 39 to **60 x 42 mm** -- the nearest expressible box to +10 % on
    #: each axis, since holding 3/14 mm per pixel with both output sides divisible by 14
    #: forces `box = 3 * k` and 59.4 and 42.9 are not expressible -- and the depth from
    #: 28 mm over 9 slots to **35 over 11**, the 3.50 mm pitch unchanged.
    #:
    #: **The pitch is deliberately not touched, and the measurement is why.** Coronal
    #: through-plane spacing is **3.50 mm** in the median (p5 3.00, p95 5.07), which is
    #: this grid's pitch exactly. Holding 28 mm and spending the two extra slots on
    #: density instead would put the pitch at 2.80 and drop the expected fill from 93 %
    #: to **78 %** -- padding, not slices. Lengthening to 35 mm at 3.50 keeps 92 % and
    #: buys about **1.8 more real slices** (10.16 of 11 against 8.33 of 9). Measured fill
    #: today is 7.55 of 9, so 0.78 of a slot is the window genuinely overrunning the
    #: acquisition and the rest is how far apart the slices are.
    #:
    #: **Unlike the patellofemoral box, nothing here is wasted.** Measured row by row and
    #: column by column over 1200 studies, tissue sits between 0.76 and 0.83 across the
    #: height and 0.74 and 0.89 across the width -- no dead band on any of the four
    #: edges, where `pf_oa`'s top edge carries 0.13 because its +8 mm rise pushes it past
    #: the skin. So an enlargement here reaches real pixels in every direction, and the
    #: box is centred where it should be.
    #:
    #: Memory is the practical risk and not the anatomy: this column already trains at
    #: **batch 16**, and nine windows at 280 x 196 px is about 1.5x what it costs now.
    "lateral_oa_wide_d35": RoiSpec(
        name="lateral_oa_wide_d35", landmark="lat_centre", plane="Coronal",
        box_w_mm=60.0, box_h_mm=42.0, out_w=280, out_h=196,
        lateral_mm=17.5, medial_mm=17.5, slots=11,
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
    #: **Taller, asymmetrically, and a longer window.** 80 to **112 mm** of height --
    #: +30 % of the current height added above and +10 % below, which also moves the box
    #: up: the centre shifts by (0.30 - 0.10)/2 x 80 = 8 mm, so `box_rise_mm` goes from
    #: **-12 to -4**. Width, offset and resolution are untouched, and 112 mm is
    #: expressible exactly (`box = 4 * k` at 2/7 mm per pixel, 392 px = 14 x 28).
    #:
    #: **The asymmetry is measured, and only one half of it is supported.** Row by row
    #: over 1200 studies the tissue fraction is **0.833 at the top edge**, peaks at 0.851
    #: around 16-20 mm, then falls steadily to **0.615 at the bottom** -- so the ligament
    #: does continue above the box, which is what the extra 24 mm up is for, while the
    #: 8 mm down goes toward the thinnest part of the crop. That direction is the one to
    #: drop first if this needs trimming.
    #:
    #: **The depth window is 36 mm over 11 slots, and the pitch is what chose it.**
    #: `choose_slices` thins a surplus with `np.linspace` over *list positions*, with no
    #: regard for which slice carries the landmark -- so a window with surplus can drop
    #: the very slice it was centred on. Measured over 120 studies, the shipped 24/7
    #: window keeps the landmark's own slice **98.3 %** of the time; an earlier draft of
    #: this spec at 32 mm over 9 slots, which has 10.1 slices available for 9, kept it
    #: only **90.0 %** -- one study in ten reframed onto a slice that is not the one the
    #: point was placed on.
    #:
    #: A pitch just *above* the acquisition's leaves no surplus to thin, and that fixes
    #: it without touching shared code:
    #:
    #: | window | landmark kept | fill | real slices |
    #: |---|---|---|---|
    #: | 24 mm / 7 slots, 4.00 | 98.3 % | 88 % | 6.17 |
    #: | 32 mm / 9 slots, 4.00 | **90.0 %** | 90 % | 8.13 |
    #: | 28 mm / 9 slots, 3.50 | 100 % | 82 % | 7.36 |
    #: | 32 mm / 11 slots, 3.20 | 100 % | 76 % | 8.34 |
    #: | **36 mm / 11 slots, 3.60** | **100 %** | 85 % | **9.40** |
    #:
    #: So 36/11 beats 32/9 on both counts at once -- never loses the landmark, and reads
    #: 9.40 real slices against 8.13 -- because 3.60 sits just over the 3.51 mm coronal
    #: spacing. **The proper fix is still a `keep_centre` rule in `choose_slices`**, which
    #: would have to be opt-in: that function serves every region, so changing it would
    #: alter the pixels of every cache and the input of all thirteen fitted packages, the
    #: trap `PixelRules.sagittal_flip` is deliberately parked on.
    #:
    #: Memory will be the question rather than the anatomy: 392 px of height against 280,
    #: and nine windows against five, is roughly **2.5x** what this column costs now.
    #: **Two series, not three, and that is not a detail.** The shipped MCL expert reads
    #: `mcl_fsonly` rather than `mcl`, because dropping the non-fat-suppressed coronal is
    #: worth **+0.0117** on this column -- recorded in `rsna.infer.chain` beside the 197
    #: studies the expert then abstains on. A first draft of this spec carried all three
    #: series and would have re-introduced a loss more than twice the size of the gain it
    #: was built to chase.
    "mcl_tall_d36": RoiSpec(
        name="mcl_tall_d36", landmark="mcl_centre", plane="Coronal",
        box_w_mm=32.0, box_h_mm=112.0, out_w=112, out_h=392,
        box_offset_mm=5.0, box_rise_mm=-4.0,
        lateral_mm=18.0, medial_mm=18.0, slots=11,
        series=(("Coronal", "PD", True), ("Coronal", "T2", True))),
    #: **The same box, grown about the centre it already had.** `mcl_tall_d36` + `lr =
    #: 1e-4` measured **−0.0020, 3 of 5 folds**, while the same bundle took +0.0053 and
    #: +0.0052 on the two meniscus columns. The one thing MCL did differently was *move*:
    #: its `box_rise_mm` went −12 to −4, lifting the frame 8 mm, where both meniscus
    #: bundles grew symmetrically about an unchanged centre. Repositioning is the
    #: category this project keeps measuring at zero — the Effusion expert lost 0.025 for
    #: bringing only better centring.
    #:
    #: So this holds the 112 mm height and puts the rise back to **−12**: +16 mm up and
    #: +16 mm down instead of +24 and +8. One variable against `mcl_tall_d36`, and the
    #: answer separates "the box was moved" from "the box was enlarged" — which are
    #: inseparable in any asymmetric growth, and were therefore confounded in the first
    #: attempt.
    "mcl_tall_d36_centred": RoiSpec(
        name="mcl_tall_d36_centred", landmark="mcl_centre", plane="Coronal",
        box_w_mm=32.0, box_h_mm=112.0, out_w=112, out_h=392,
        box_offset_mm=5.0, box_rise_mm=-12.0,
        lateral_mm=18.0, medial_mm=18.0, slots=11,
        series=(("Coronal", "PD", True), ("Coronal", "T2", True))),
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
    #: **A larger frame and a longer window, on the plane with the thinnest resolution
    #: margin of the set.** 56 x 48 to **64 x 56 mm** -- the nearest expressible box to
    #: +15 % on each axis, since holding 2/7 mm per pixel with both output sides
    #: divisible by 14 forces `box = 4 * k` and 64.4 and 55.2 are not expressible -- and
    #: the depth from 32 mm over 9 slots to **40 over 11**, the 4.00 mm pitch unchanged.
    #:
    #: **The pitch is deliberately not touched, and that is measured.** Axial
    #: through-plane spacing is **3.90 mm** in the median (p5 3.30, p95 5.70) against
    #: this grid's 4.00, so unlike the sagittal meniscus windows -- 3.30 against 4.00,
    #: a 21 % gap -- there is nothing to recover by densifying here. Going to 3.20 mm
    #: would add padding, not slices.
    #:
    #: **Why the longer window gains rather than pads.** Expected fill from the spacing
    #: alone is 8.43 of 9 at 32 mm and **10.29 of 11 at 40 mm** -- the same 94 %, so
    #: about two more *real* slices. Measured fill at 32/9 is 7.92, so 0.52 of a slot is
    #: the window genuinely overrunning the acquisition and the rest is simply how far
    #: apart the slices are. An earlier reading of this compared 7.92 against the
    #: *median* of what is available (9.2) rather than the mean of the minimum (8.43),
    #: which inflated the overrun and wrongly argued against lengthening.
    #:
    #: **What the enlargement will and will not reach.** Measured row by row over 1500
    #: studies, tissue runs from 7.1 mm below the top edge to the bottom edge itself,
    #: peaking at 0.936 and still 0.782 where the box stops -- so the anatomy continues
    #: past the bottom, and past both sides (columns 0.72 to 0.62). It does **not**
    #: continue past the top: the first 4 mm carry 0.189 of tissue and the next 4 carry
    #: 0.429, because `box_rise_mm = +8` puts the top edge 32 mm anterior of
    #: `pf_centre` while the patella's cortex is at 25 and the skin just beyond.
    #:
    #: `box_rise_mm` is **left at +8 anyway**, so half the added height grows into that
    #: air. Dropping it to 0 would trade 8 mm of air for 8 mm of tissue at no cost in
    #: pixels or memory, and is the obvious next single variable if this does not pay.
    "pf_oa_wide_d40": RoiSpec(
        name="pf_oa_wide_d40", landmark="pf_centre", plane="Axial",
        box_w_mm=64.0, box_h_mm=56.0, out_w=224, out_h=196,
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
    #: The same 32 mm at the acquisition's own rate, as `medial_meniscus_d32_s11` does
    #: for the other compartment. Both shipped meniscus windows run a **4.00 mm** pitch
    #: against a sagittal through-plane spacing of **3.30 mm** in the median, so
    #: `choose_slices` thins a surplus on about seven series in ten; eleven slots over
    #: the same window is 3.20 mm, which is the acquisition's.
    #:
    #: **This exists to give the question a second column, and for one reason.** The
    #: effect is expected at about +0.0009 -- the window count rises 20 % where
    #: `medial_meniscus_d32` raised it 132 % for +0.0056 -- against a paired standard
    #: error of 0.0020 on five folds. No amount of pooling we can afford fixes that:
    #: resolving +0.0009 by a t-test would need ~100 folds. What ten paired folds *can*
    #: do is a **sign test**, which ignores magnitude: `d32` came out positive 5 of 5
    #: (p = 0.031), and 9 or 10 of 10 here would read p = 0.011 or 0.001.
    #:
    #: `acl` is not a third column: its window is derived from the two meniscus points
    #: and already runs a 2.68 mm pitch, finer than the acquisition, so densifying it
    #: would add padding and nothing else.
    "lateral_meniscus_d32_s11": RoiSpec(name="lateral_meniscus_d32_s11",
                                        lateral_mm=8.0, medial_mm=24.0, slots=11),
    #: The medial compartment's winning combination, transposed: the large frame, the
    #: native pitch, and `lr = 1e-4` in the run that uses it. 48 x 27 to **57 x 33 mm**
    #: on 266 x 154 px, +18.8 % and +22.2 % of side — the nearest expressible box to the
    #: proportion the medial box actually received, since `box = 3 * k` quantises the
    #: choice and 56.0 and 31.9 are not expressible.
    #:
    #: **The component that carries that combination is measured negative here**, and it
    #: should be read knowing so. On the medial column `lr = 1e-4` was worth +0.0134
    #: against +0.0024 for the pitch and −0.0019 for the frame, so it is nearly the whole
    #: effect; on *this* column the same change was run on `lateral_meniscus_d32` itself
    #: and cost **−0.0104**. Not an extrapolation — one variable, same wd, same thirty
    #: epochs, the measurement that made `lr = 1e-4` across the board refusable in the
    #: first place.
    #:
    #: It is built anyway because adding the components up has already proved a poor
    #: predictor: on the medial side they summed to +0.0138 and the combination measured
    #: +0.0053, 2.6x over. What that failure does not do is change a sign.
    "lateral_meniscus_d32_wide_s11": RoiSpec(
        name="lateral_meniscus_d32_wide_s11",
        box_w_mm=57.0, box_h_mm=33.0, out_w=266, out_h=154,
        lateral_mm=8.0, medial_mm=24.0, slots=11),
    #: **The depth sweep the lateral compartment got and the medial one never did.**
    #: The shipped pair is lopsided and the asymmetry is an accident of order, not a
    #: finding: `lateral_meniscus_d32` carries 8 + 24 mm over 9 slots because that sweep
    #: ran and won, while `medial_meniscus` is still on the original 4 + 12 over 5.
    #: Measured on the lateral column with resnet18, the same table and the same
    #: weighting, the four windows read **m20 0.8320, l6 0.8373, d24 0.8387, d32
    #: 0.8476** -- so doubling the extent was worth +0.009 to +0.016, which is larger
    #: than any test-time trick this project has found.
    #:
    #: **The acquisition supports it, and that is not true on every plane.** A longer
    #: window at the same 4.0 mm pitch fills *better*, not worse, because more acquired
    #: slices fall inside it and `choose_slices` thins a surplus down to `slots` where a
    #: narrow window cannot be topped up: the 5-slot medial window finds 4.40 slices of
    #: 5 (88 %) and fills every slot on 53.3 %, while the 9-slot lateral window finds
    #: 8.41 of 9 (**93.4 %**) and fills every slot on **65.8 %**. The patellofemoral box
    #: is the counter-example and the reason to measure rather than assume -- its axial
    #: stacks find only 7.80 slices of 9, so lengthening *that* window buys padding.
    #:
    #: **The box in plane is deliberately untouched.** It is the only variable this spec
    #: changes, and in plane the medial box is already the larger of the two (54 x 33
    #: against 48 x 27). Coverage is not the difference either: both read the same two
    #: sagittal series, so both see 82.5 % and 37.5 %, 99.8 % with at least one, and the
    #: same 8 studies with neither.
    #:
    #: The anatomy argues the same way. The medial meniscus is the wide C against the
    #: lateral's nearly closed O, and its posterior horn is the broader of the two, so
    #: there is more of it to reach along the depth axis rather than less.
    "medial_meniscus_d32": RoiSpec(
        name="medial_meniscus_d32", landmark="med_centre",
        box_w_mm=54.0, box_h_mm=33.0, out_w=252, out_h=154,
        lateral_mm=8.0, medial_mm=24.0, slots=9),
    #: **The same 32 mm, sampled at the rate it was acquired.** `medial_meniscus_d32`
    #: won on five folds of five (+0.0056 paired, t = 2.82) and it still throws acquired
    #: slices away: the sagittal through-plane spacing is **3.30 mm in the median** (p5
    #: 3.00, p95 4.80) against this grid's 4.00 mm pitch, so `choose_slices` thins a
    #: surplus on about seven series in ten. Eleven slots over the same window is a
    #: 3.20 mm pitch, which is the acquisition's own.
    #:
    #: This is the side of the trade the `acl` spec already argued for and that the
    #: meniscus windows have been on the wrong side of throughout -- *padding is the
    #: cheaper mistake: a window whose centre slot is padding never reaches the encoder,
    #: where a thinned stack has lost pixels that existed.*
    #:
    #: **And it separates two mechanisms that `d32` left entangled.** Lengthening the
    #: window both showed more meniscus and gave more windows to pool, and the pooling
    #: half is visible: the cross-fold calibration cost fell from −0.0111 to −0.0015
    #: when the count went from 3 windows to 7. This spec holds the anatomy fixed and
    #: moves the count to 9, so a gain here is the variance term and not new signal --
    #: which is what decides whether enlarging the box *in plane* is worth a run.
    #:
    #: Extent is deliberately untouched, and so is everything in plane.
    "medial_meniscus_d32_s11": RoiSpec(
        name="medial_meniscus_d32_s11", landmark="med_centre",
        box_w_mm=54.0, box_h_mm=33.0, out_w=252, out_h=154,
        lateral_mm=8.0, medial_mm=24.0, slots=11),
    #: **The same window, a larger frame, and not one pixel coarser.** `d32` is the
    #: baseline this is one variable away from: depth, slots, landmark, plane, series and
    #: millimetres per pixel are all held, and only the in-plane extent moves -- 54 x 33
    #: to 63 x 39 mm, which is +16.7 % and +18.2 % of side and +37.9 % of area.
    #:
    #: The sizes are not free, and these are the available ones. Holding 3/14 mm per
    #: pixel and requiring both output sides to divide by 14 -- so a patch embedding
    #: keeps every row -- forces `box = 3 * patches`: 63 mm is 294 px and 39 mm is
    #: 182 px, both exactly, where 62 or 38 are not expressible at all. Same reasoning
    #: that fixed the original 54 x 33, and same purpose: **what differs from `d32` is
    #: extent, not sharpness.**
    #:
    #: **Resolution has room here and did not on the axial box.** Sagittal acquisitions
    #: are 0.332 mm/px in the median, and this crop is 0.214, so it oversamples by 1.55x
    #: -- enlarging the box at a fixed pixel count would still oversample by 1.35x and
    #: lose nothing that was acquired. `pf_oa` sits at 0.286 against an axial 0.312, a
    #: margin of 1.09x, which is why the same proposal was refused there.
    #:
    #: **What the argument is weaker about.** The depth extension reached *more
    #: meniscus* -- the posterior horn runs along that axis. In plane, 54 mm already
    #: covers a 35-40 mm meniscus with margin, so the 4.5 mm added front and back is fat
    #: pad and popliteal tissue and the 3 mm added above and below is condyle and
    #: plateau. If `d32` paid because it saw more of the structure, this will not pay; if
    #: it paid because context and pooled windows reduce variance, it may.
    "medial_meniscus_d32_wide": RoiSpec(
        name="medial_meniscus_d32_wide", landmark="med_centre",
        box_w_mm=63.0, box_h_mm=39.0, out_w=294, out_h=182,
        lateral_mm=8.0, medial_mm=24.0, slots=9),
    #: The large frame *and* the native pitch, together, with `lr = 1e-4` in the run
    #: that uses it. **Three changes at once, deliberately, and not an experiment**: a
    #: row that moves cannot say which of the three moved it. It is a bid for the best
    #: model on this column rather than for knowledge about it, which is a thing worth
    #: wanting and is recorded here as what it is.
    #:
    #: What is already known about each part, so the result can be read against it:
    #: the frame alone measured **−0.0019** over four paired folds and a fifth that
    #: diverged, so it is the component with no support; the pitch is under measurement
    #: as `medial_meniscus_d32_s11` and is expected at about +0.0009, below what five
    #: folds resolve; and **`lr = 1e-4` is the strongest of the three** — on this exact
    #: column with the original 5-slot box it took the out-of-fold figure from 0.8994 to
    #: **0.9127**, one variable, +0.0133. That last one has never been adopted because
    #: the same change costs the *lateral* compartment 0.0105, which is why it was
    #: refused across the board and why the per-column case was left standing.
    #:
    #: Memory is the practical risk: nine windows at 294 x 182 px is about 1.8x what
    #: `medial_meniscus_d32` peaked at, which is near the card rather than comfortably
    #: inside it. If it does not fit at batch 32 the batch becomes a fourth change and
    #: the comparison degrades further, so that is a decision and not a default.
    "medial_meniscus_d32_wide_s11": RoiSpec(
        name="medial_meniscus_d32_wide_s11", landmark="med_centre",
        box_w_mm=63.0, box_h_mm=39.0, out_w=294, out_h=182,
        lateral_mm=8.0, medial_mm=24.0, slots=11),
    #: The joint effusion, in the recess it actually fills.
    #:
    #: **Clinically there is no doubt where to look.** Fluid collects first and most in
    #: the suprapatellar recess, above the patella between the quadriceps tendon and the
    #: femur, and the standard grading is its antero-posterior depth there. So the anchor
    #: is `pf_centre` and the direction is up — not `med_centre`, which sits at the joint
    #: line 40 mm below the signal.
    #:
    #: **The plane follows from an asymmetry, and it is the mirror of Baker's.** What
    #: matters for a structure *above* the patella is whether "up" is an axis of the image
    #: or of the stack. Measured over 250 series per plane: in **axial** up is
    #: through-plane and the acquisition reaches a median **13 mm** above `pf_centre`, so
    #: the recess is largely not acquired; in **sagittal** up is in-plane with a median
    #: **70 mm** of headroom, and +42 mm is reachable on **100 %** of studies. The cyst is
    #: posteromedial and gets the opposite answer, which is why its box is axial.
    #:
    #: **And the capture curve says the opposite of Baker's too.** A difference map over
    #: 200 reported effusions against 200 without puts the sagittal peak at **+0.339,
    #: 42 mm above `pf_centre`** on 99 % coverage. But capture only climbs with size:
    #: 56 x 70 mm holds 43.6 %, 70 x 84 holds 66.9 %, and **84 x 84 holds 94.4 %**. A cyst
    #: sits and can be framed tightly; a recess fills and cannot. So this box is large on
    #: purpose, and at 0.353 mm/px it is barely finer than the wide model's 0.387 — its
    #: advantage is **where it is centred**, not how closely it looks.
    #:
    #: Depth is symmetric, so `symmetric_depth` holds and `bowtie_direction` is never
    #: asked: its "nearer end" heuristic was calibrated on a meniscus and means nothing
    #: for a recess.
    "effusion_sagittal": RoiSpec(
        name="effusion_sagittal", landmark="pf_centre", plane="Sagittal",
        box_w_mm=84.0, box_h_mm=84.0, out_w=238, out_h=238,
        box_offset_mm=0.0, box_rise_mm=28.0,
        lateral_mm=24.0, medial_mm=24.0, slots=15, decimate_to_mm=3.3,
        series=(("Sagittal", "PD", True), ("Sagittal", "T2", True))),
    #: The same recess seen transversely, as high as the acquisition allows.
    #:
    #: Its depth window is capped by the 13 mm of median headroom above `pf_centre`, which
    #: is the whole reason the sagittal box leads. Asymmetric, so `bowtie_direction` picks
    #: the sign — measured on 150 axial stacks it points **inferior 99.3 %** of the time,
    #: making `lateral_mm` the inferior reach and `medial_mm` the superior one.
    #:
    #: 84 x 56 mm holds **91.6 %** of the axial excess. Coverage is what argues for it:
    #: axial PD fat-sat reaches 72.2 % of studies and T2 fat-sat takes the pair to
    #: **98.6 %**, so the two boxes together cover **99.14 %** where the sagittal one alone
    #: covers 93.5 %.
    "effusion_axial": RoiSpec(
        name="effusion_axial", landmark="pf_centre", plane="Axial",
        box_w_mm=84.0, box_h_mm=56.0, out_w=294, out_h=196,
        box_offset_mm=0.0, box_rise_mm=0.0,
        lateral_mm=18.0, medial_mm=24.0, slots=11, decimate_to_mm=4.0,
        series=(("Axial", "PD", True), ("Axial", "T2", True))),
    #: The popliteal cyst, in the plane the first attempt never gave it.
    #:
    #: **Why axial.** The sagittal `baker` spec dismissed this plane in one line — *axial
    #: needs no expert, the cyst falls at 0.50 of that stack* — which argues that the
    #: **wide model** already sees it, not that an expert would read it no better. Two
    #: different questions, and the first was answered in place of the second. The axial
    #: plane is also the only one that shows the neck between the semimembranosus tendon
    #: and the medial head of gastrocnemius, which is what separates a true Baker's cyst
    #: from any other popliteal fluid.
    #:
    #: **Located, not guessed.** An axial difference map over 250 studies with a reported
    #: cyst against 250 without — bright-voxel frequency on fat-suppressed fluid-sensitive
    #: images, thresholded per slice at its own 97th percentile — puts the excess at
    #: **36 mm posterior** of `med_centre`, at the landmark's own medial position
    #: (**within ±6 mm at every height**, which is the axis the sagittal map could not
    #: resolve), spanning **−20 to +50 mm** in height and peaking at +10 to +30. The cyst
    #: is not at the joint line; it is above it, in the popliteal fossa.
    #:
    #: 56 x 56 mm holds **82.5 %** of that excess against 71 % for the sagittal 90 x 75,
    #: and stays wholly inside the image on **97.6 %** of studies against 98.8 % — so it
    #: is not a trade. The same box with its depth window shrunk to the joint line keeps
    #: 48.3 %: nearly half the signal is in the 40 mm a joint-centred window would miss.
    #:
    #: **Three signs, all read off the series rather than assumed.** `box_rise_mm` moves
    #: the box along the rows, which is *anterior* on an axial crop, so a **negative**
    #: value is what goes posterior. `box_offset_mm` is zero because the measurement said
    #: no medial correction was needed. And the depth window is asymmetric, so
    #: `bowtie_direction` picks its direction — measured on 150 axial stacks it points
    #: **inferior 99.3 %** of the time, which makes `lateral_mm` the inferior reach and
    #: `medial_mm` the superior one.
    #:
    #: Fat-suppressed only, and PD before T2: axial PD fat-sat covers 72.2 % of studies
    #: and T2 fat-sat takes the pair to **98.6 %**. Axial T1 exists on 12.4 % and is not
    #: worth a fallback level.
    "baker_axial": RoiSpec(
        name="baker_axial", landmark="med_centre", plane="Axial",
        box_w_mm=56.0, box_h_mm=56.0, out_w=196, out_h=196,
        box_offset_mm=0.0, box_rise_mm=-36.0,
        lateral_mm=20.0, medial_mm=50.0, slots=18, decimate_to_mm=4.0,
        series=(("Axial", "PD", True), ("Axial", "T2", True))),
    #: The whole knee, for the five targets no compartment box covers — effusion,
    #: synovitis, the popliteal cyst, contusion, fracture. Pipeline v3's wide branches.
    #:
    #: **Not a wide slot: a wide box.** The distinction is the point. Every leg of the
    #: public pipeline crops 130 mm around the *image* centre, and measured over all
    #: 4407 studies the knee sits a median 17 mm off it, with half the corpus having
    #: joint anatomy outside the window. Centred on the joint, 100 mm is enough — a
    #: half-knee is about 48 mm — and the 30 mm they spend on being off-centre buys
    #: resolution instead: 0.357 mm/px against their 0.387.
    #:
    #: **Two landmarks, for the centre only.** The midpoint of the two meniscus points
    #: is the joint centre, and both are already predicted for every study, so this
    #: needs no new annotation. The non-zero depth window is what tells `extract` to
    #: take the extent from the spec rather than from the gap between the points —
    #: 27 to 48 mm, which is a compartment and not a knee.
    #:
    #: **Depth is symmetric in both planes, and that is a decision.** Measured over 120
    #: series per plane, the joint centre sits 49 mm from each end of a sagittal stack
    #: and 60/41 mm in a coronal one. The coronal asymmetry is tempting and declined:
    #: an asymmetric window makes `symmetric_depth` false, which sends `extract` to
    #: `bowtie_direction` — a rule measured on the medial-lateral axis, applied to an
    #: anterior-posterior one. ±40 mm is what both coronal ends support, and since the
    #: window holds ~90% of the stack either way, the slot cap decides, not the window.
    #:
    #: **Coarser than a compartment box, deliberately.** 0.357 mm/px and 4.5 mm between
    #: slots, against 0.214 and 3.3 for a meniscus. These branches are for diffuse
    #: findings, which are large; the budget is better spent on reaching the whole knee
    #: than on resolving a tear the compartment boxes already resolve.
    "knee_sagittal": RoiSpec(
        name="knee_sagittal", landmark="lat_centre", landmark2="med_centre",
        plane="Sagittal", box_w_mm=100.0, box_h_mm=100.0, out_w=280, out_h=280,
        lateral_mm=45.0, medial_mm=45.0, slots=20, decimate_to_mm=4.5,
        series=(("Sagittal", "PD", True), ("Sagittal", "T2", True),
                ("Sagittal", "PD", False))),
    #: The same box in the plane osteoarthritis and the collaterals are read in, and the
    #: one the public pipeline never crops at all on two of its three layouts.
    "knee_coronal": RoiSpec(
        name="knee_coronal", landmark="lat_centre", landmark2="med_centre",
        plane="Coronal", box_w_mm=100.0, box_h_mm=100.0, out_w=280, out_h=280,
        lateral_mm=40.0, medial_mm=40.0, slots=20, decimate_to_mm=4.5,
        series=(("Coronal", "PD", True), ("Coronal", "T2", True),
                ("Coronal", "PD", False))),
    #: ---- bone contusion: wide in plane, coarse through it ----
    #:
    #: A contusion is marrow oedema, one to two centimetres across, in the subchondral
    #: bone of the condyles, the plateaus or the patella. It has no landmark of its own
    #: and never will, so both boxes hang off the **midpoint of the two meniscus
    #: centres** — the notch, the most central stable point there is.
    #:
    #: The design trades the two axes against each other on purpose: **0.225 mm/px in
    #: plane, 5-6 mm between slots**. A meniscal tear is a 1 mm line and needs every
    #: slice; a contusion is a blob and does not. Keeping both fine would have cost
    #: 22x a compartment box, where this costs 5x and 6x.
    #:
    #: Measured against the shipped osteoarthritis boxes, which look at the same bone:
    #: those stop at ±14 mm of depth, and a **posterolateral tibial plateau** contusion
    #: — one of the two sites of the pivot-shift injury — falls outside that.
    "contusion_coronal": RoiSpec(
        name="contusion_coronal", landmark="lat_centre", landmark2="med_centre",
        plane="Coronal", box_w_mm=88.2, box_h_mm=63.0, out_w=392, out_h=280,
        lateral_mm=30.0, medial_mm=30.0, slots=12, decimate_to_mm=5.0,
        series=(("Coronal", "PD", True), ("Coronal", "T2", True))),
    #: What the coronal plane cannot give: the posterior plateau, the patella's own
    #: subchondral bone, and the sulcus terminalis where the deep lateral femoral notch
    #: sign is read.
    "contusion_sagittal": RoiSpec(
        name="contusion_sagittal", landmark="lat_centre", landmark2="med_centre",
        plane="Sagittal", box_w_mm=88.2, box_h_mm=72.45, out_w=392, out_h=322,
        lateral_mm=36.0, medial_mm=36.0, slots=12, decimate_to_mm=6.0,
        series=(("Sagittal", "PD", True), ("Sagittal", "T2", True))),
    #: The medial meniscus seen from the front, beside the sagittal box the expert
    #: already reads. Three findings the sagittal plane serves badly, and all three are
    #: coronal readings: **medial extrusion**, which is *defined* as the gap between the
    #: tibial plateau margin and the meniscus' outer edge and needs both in one image;
    #: the **body in cross-section**, which on sagittal is the continuous band that is
    #: hardest to judge; and **root truncation**, a different lesion from a body tear.
    #:
    #: **Asymmetric in plane: 27 mm medial, 42 mm lateral.** `box_offset_mm` is +7.5
    #: toward the lateral side, so the medial edge sits where a centred 54 mm box put it
    #: (7.5 - 34.5 = -27) and every millimetre added on review went inward, toward the
    #: notch.
    #:
    #: Medial 27 mm: the plateau margin is 10 to 15 mm out, so this keeps it plus the
    #: medial gutter and the collateral ligament behind it — the tissue an extruded
    #: meniscus protrudes into.
    #:
    #: Lateral 42 mm: far enough for the middle of the intercondylar notch and the
    #: lateral tibial spine, which buys the **posterior root insertion** and
    #: **bucket-handle fragments** — the latter displace *into* the notch and so sit
    #: outside any box drawn around the compartment alone.
    #:
    #: **Measured, with laterality resolved per series.** A first pass compared image
    #: left against image right, which does not say which is medial, and reported the two
    #: margins swapped. Resolved properly over 220 coronal fat-suppressed studies, the
    #: room toward the notch is **92.2 mm at the 5th centile** against **40.2 mm** toward
    #: the skin. So growing laterally is free — 42 mm holds on **99.5 %** of studies and
    #: so would 45 — and the binding constraint is the medial 27 mm, which is why it was
    #: not moved. This is the mirror of `lateral_meniscus_coronal`'s -6 mm, which shifts
    #: *inward* because its landmark sits near the lateral skin instead.
    #:
    #: 36 mm superior-inferior, centred on the joint line: 18 mm up for the condyle's
    #: articular surface, 18 down for the plateau and its subchondral plate. Less than
    #: the OA box's 39, which needs marrow both sides; more than
    #: `lateral_meniscus_coronal`'s 30, because an edge is being read against an edge.
    #: Judgement.
    #:
    #: **Depth is symmetric at ±25 mm over 15 slots, and the symmetry is not a
    #: preference.** On a coronal stack the depth axis is anterior-posterior, but
    #: `lateral_mm`/`medial_mm` resolve through `bowtie_direction`, whose rule — "the
    #: imaged knee ends sooner on the lateral side, 19.8 mm against 74.5" — is a
    #: medial-lateral fact with no meaning on the AP axis. Every coronal spec here is
    #: symmetric, so `symmetric_depth` short-circuits that function and it is never
    #: called; breaking the symmetry to bias the window backwards would invoke it on the
    #: wrong axis, which is the bug `tools/atlas/contusion_map.py` already made once. So
    #: the posterior root is reached by making the window **wide**, not lopsided: 50 mm
    #: runs from the anterior horn to the root insertion with margin at both ends.
    #:
    #: **The weakness, stated rather than hidden.** `med_centre` is defined as *the
    #: centre of the medial meniscus*, so in an extruded knee it travels outward with the
    #: meniscus and the meniscus is always mid-box. Extrusion is a **relative**
    #: displacement, and centring on the thing that moved turns it into a measurement the
    #: model must make against the plateau margin *inside* the box. The information
    #: survives — the margin is in frame, at a position that shifts — but it is harder to
    #: read than from a plateau-anchored box. Anchoring there would need a landmark we
    #: have never annotated, for a gain our own rule says centring alone does not give.
    "medial_meniscus_coronal": RoiSpec(
        name="medial_meniscus_coronal", landmark="med_centre", plane="Coronal",
        box_w_mm=69.0, box_h_mm=36.0, out_w=322, out_h=168,
        box_offset_mm=7.5, box_rise_mm=0.0,
        lateral_mm=25.0, medial_mm=25.0, slots=15, decimate_to_mm=3.3,
        series=(("Coronal", "PD", True), ("Coronal", "T2", True),
                ("Coronal", "PD", False))),
}
