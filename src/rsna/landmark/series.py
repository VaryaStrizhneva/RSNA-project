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


#: Which picker each landmark needs, by the plane it is annotated on.
PICKERS = {"Sagittal": pick_sagittal, "Axial": pick_axial}


#: The points this project collects, and what each needs to be annotated correctly.
#:
#: `prefer_deep` has to be read at **inference** as well as at annotation, and has to be
#: the same both times. It decides which series of a study the model is shown, so a run
#: that predicts on a 320-slice acquisition for a point annotated on a 30-slice one is
#: asking the model about pixels it never trained on. The meniscus bundles were built
#: with it and the patellofemoral one without, so it lives here rather than being passed
#: separately to each script and eventually passed differently.
#:
#: `laterality` is the one that changes the annotation tool rather than the model: a
#: sagittal stack runs along the left-right axis, so which end is lateral decides which
#: meniscus is being pointed at and the annotator has to be told. An axial stack runs
#: inferior to superior, and `pf_centre` sits on the midline of its joint, so nothing
#: about left or right changes where the click goes. Showing a lateral badge there would
#: be asking for a declaration that cannot be wrong, which teaches an annotator to stop
#: reading badges.
LANDMARKS = {
    "lat_centre": {
        "id": "lat_centre", "plane": "Sagittal", "colour": "#ff6b6b",
        "laterality": True, "prefer_deep": True,
        "what": "the centre of the lateral meniscus",
    },
    "pf_centre": {
        "id": "pf_centre", "plane": "Axial", "colour": "#ffb24d",
        "laterality": False, "prefer_deep": False,
        "what": "the middle of the patellofemoral joint space",
    },
}
