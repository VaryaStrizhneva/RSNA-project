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
