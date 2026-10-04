"""Which series of a study the landmark model reads.

This has to be the same rule at inference as at annotation, or the model meets a
sequence it never trained on. It lived in `tools/annotate/bundle.py`, which is the wrong
direction of dependency — the tools import from the package, never the reverse — and it
decides what the model *is*, so it belongs here.
"""

from __future__ import annotations

import pandas as pd


def pick_sagittal(headers: pd.DataFrame, prefer_deep: bool = False):
    """The sagittal series a meniscus is actually read on: PD first, then T2, then T1.

    `prefer_deep` takes a 3D acquisition where the study has one. It has to be asked
    for, because the default preference hides them: a 3D sagittal PD is almost always
    fat suppressed, so a study holding both a 320-slice PD FS and a 30-slice PD without
    FS returns the 30 — and the annotation set ends up with no 3D at all, which is what
    happened to the first bundle. The training pipeline's own `pick_slots` sorts on
    slice count, so the model *does* meet these series; the annotator should too.

    A 3D series is preferred only *within* the weighting, never across it. Taking the
    deepest sagittal outright reaches past a 28-slice PD to a 140-slice T1, and a T1
    fills no slot the classifier has.
    """

    sag = headers[headers["plane"] == "Sagittal"]
    if not len(sag):
        return None
    for weight in ("PD", "T2", "T1"):
        hit = sag[sag["weight"] == weight]
        if not len(hit):
            continue
        if prefer_deep:
            deep = hit[hit["n_slices"] > 100]
            if len(deep):
                return deep.sort_values("n_slices", ascending=False).iloc[0]
        # Prefer no fat suppression: short-TE without FS shows signal *inside* the
        # fibrocartilage best, which is the whole basis of meniscal grading.
        return hit.sort_values(["fatsat", "n_slices"], ascending=[True, False]).iloc[0]
    return sag.sort_values("n_slices", ascending=False).iloc[0]


def pick_axial(headers: pd.DataFrame, prefer_deep: bool = False):
    """The axial series the patellofemoral joint is read on: fat-suppressed first.

    The opposite preference to `pick_sagittal`, and the same reasoning turned round. A
    meniscal tear is signal *inside* fibrocartilage, which fat suppression flattens, so
    the meniscus wants it off. Patellofemoral osteoarthritis is cartilage loss and
    subchondral oedema, and oedema is only visible once the fat around it is suppressed,
    so this wants it on.

    PD fat-suppressed covers 72.2 % of studies and T2 fat-suppressed 28.2 %; at least
    one of the two covers **98.5 %**, and adding T1 to that gains nothing at all — so
    the pair is the whole of the preference, and what follows it is only a fallback for
    the 1.5 % that have neither.
    """

    ax = headers[headers["plane"] == "Axial"]
    if not len(ax):
        return None
    for weight, fat in (("PD", True), ("T2", True), ("PD", False), ("T2", False),
                        ("T1", False)):
        hit = ax[(ax["weight"] == weight) & (ax["fatsat"].astype(bool) == fat)]
        if not len(hit):
            continue
        if prefer_deep:
            deep = hit[hit["n_slices"] > 100]
            if len(deep):
                return deep.sort_values("n_slices", ascending=False).iloc[0]
        return hit.sort_values("n_slices", ascending=False).iloc[0]
    return ax.sort_values("n_slices", ascending=False).iloc[0]


def pick_coronal(headers: pd.DataFrame, prefer_deep: bool = False):
    """The coronal series a collateral ligament is read on: fat-suppressed first.

    Same preference as the axial patellofemoral rule and for the same reason: a ligament
    sprain is oedema in and around the band, and oedema is only visible once the fat
    around it is suppressed. PD fat-suppressed covers 84.0 % of studies and T2
    fat-suppressed 17.0 %; at least one of the two covers **95.5 %**.
    """

    cor = headers[headers["plane"] == "Coronal"]
    if not len(cor):
        return None
    for weight, fat in (("PD", True), ("T2", True), ("PD", False), ("T1", False)):
        hit = cor[(cor["weight"] == weight) & (cor["fatsat"].astype(bool) == fat)]
        if not len(hit):
            continue
        if prefer_deep:
            deep = hit[hit["n_slices"] > 100]
            if len(deep):
                return deep.sort_values("n_slices", ascending=False).iloc[0]
        return hit.sort_values("n_slices", ascending=False).iloc[0]
    return cor.sort_values("n_slices", ascending=False).iloc[0]


#: Which picker each landmark needs, by the plane it is annotated on.
PICKERS = {"Sagittal": pick_sagittal, "Axial": pick_axial, "Coronal": pick_coronal}


#: The points this project collects, and what each needs to be annotated correctly.
#:
#: `side_cue` says what the annotator has to be told about left and right, and the three
#: values are three different situations rather than degrees of one:
#:
#: * **stack-end** -- a sagittal stack runs along the left-right axis, so which *end* of
#:   it is lateral decides which meniscus is being pointed at. The tool badges the ends
#:   and lets the annotator declare them, and the click's own position recovers the side
#:   afterwards (161/161 on the first bundle).
#: * **image-side** -- a coronal stack runs front to back, so no end is lateral; but the
#:   image's horizontal axis *is* left-right, so which **side of the picture** is medial
#:   flips with the knee. Measured, 3815 of 3815 coronal series run their columns toward
#:   the patient's left, so medial is the image's right on a right knee and its left on a
#:   left one. The tool has to say which, and the click cannot recover it.
#: * **none** -- an axial stack runs bottom to top and `pf_centre` sits on the midline of
#:   its own joint, so nothing about left or right changes where the click goes. Showing
#:   a badge there would ask for a declaration that cannot be wrong, which teaches an
#:   annotator to stop reading badges.
#:
#: `click_near` may be None even on a sagittal stack: `acl_centre` sits in the notch,
#: near the middle of the left-right axis, so which end of the stack it is nearest says
#: nothing about the side. It is the meniscus points' distance from the periphery that
#: makes the trick work, not the plane.
#:
#: `click_near` says which end of a sagittal stack the point lands nearest, and it is
#: what lets the click recover the side without a second question. It is **not** the same
#: for the two menisci: a lateral point is near the lateral end, a medial one near the
#: medial end, so the same click position implies opposite sides. Reading `med_centre`
#: under the lateral rule would report every knee as the other one — silently, since both
#: answers are valid sides.
#:
#: `prefer_deep` has to be read at **inference** as well as at annotation, and has to be
#: the same both times. It decides which series of a study the model is shown, so a run
#: that predicts on a 320-slice acquisition for a point annotated on a 30-slice one is
#: asking the model about pixels it never trained on. The meniscus bundles were built
#: with it and the patellofemoral one without, so it lives here rather than being passed
#: separately to each script and eventually passed differently.
LANDMARKS = {
    "lat_centre": {
        "id": "lat_centre", "plane": "Sagittal", "colour": "#ff6b6b",
        "side_cue": "stack-end", "prefer_deep": True, "click_near": "lateral",
        "what": "the centre of the lateral meniscus",
    },
    "med_centre": {
        "id": "med_centre", "plane": "Sagittal", "colour": "#6ea8fe",
        "side_cue": "stack-end", "prefer_deep": True, "click_near": "medial",
        "what": "the centre of the medial meniscus",
    },
    "acl_centre": {
        "id": "acl_centre", "plane": "Sagittal", "colour": "#c58af9",
        "side_cue": "stack-end", "prefer_deep": True, "click_near": None,
        "what": "the mid-substance of the anterior cruciate ligament",
    },
    "pf_centre": {
        "id": "pf_centre", "plane": "Axial", "colour": "#ffb24d",
        "side_cue": "none", "prefer_deep": False, "click_near": None,
        "what": "the middle of the patellofemoral joint space",
    },
    "mcl_centre": {
        "id": "mcl_centre", "plane": "Coronal", "colour": "#7fc98b",
        "side_cue": "image-side", "prefer_deep": False, "click_near": None,
        "what": "the medial collateral ligament where it crosses the joint line",
    },
}
