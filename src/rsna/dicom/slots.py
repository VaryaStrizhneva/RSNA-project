"""One series per slot per study.

A study holds four to fourteen series and the set differs from study to study, so
before anything can be batched the acquisition has to be mapped onto a fixed number of
positions. A slot is a predicate on (plane, weighting, fat suppression); a study
either has a series matching it or does not, and the presence mask says which.
"""

from __future__ import annotations

import pandas as pd

from ..config import Config


def pick_slots(series: pd.DataFrame, plane_map: dict[str, str],
               config: Config) -> dict[str, dict[str, pd.Series]]:
    """Study -> {slot name: the chosen series row}.

    Ties are broken toward the stack with the most slices: a thicker stack samples the
    joint more densely, which the band sampler benefits from.

    A slot with no series matching its predicate **stays empty**, and no substitute is
    admitted from a neighbouring predicate. Relaxing the weighting to fill a structural
    slot would, over the training corpus, put one series in two slots for 2,383 of
    4,407 studies and leave 56% of the T1 slot holding PD or T2. The presence mask
    would then assert a sequence that was never acquired, and the per-diagnosis
    attention would divide itself across two identical slots, giving one acquisition
    about twice the weight it carries in a study that holds both. The mask exists to
    say a slot is absent, which is what an absent slot is.

    `config.rules.slot_fallback` reproduces that relaxation anyway, for weights fitted
    under it: over half of such a member's training studies had a structural slot
    holding a series that is not T1, and leaving those slots empty would present it
    with a presence mask it never saw.
    """

    series = series.copy()
    series["plane"] = series["SeriesInstanceUID"].map(plane_map)

    out: dict[str, dict[str, pd.Series]] = {}
    for study, group in series.groupby("StudyInstanceUID"):
        chosen: dict[str, pd.Series] = {}
        for name, plane, fluid, fatsat in config.slots:
            selector = (group["plane"] == plane) & (group["fatsat"] == fatsat)
            # fluid=None means "do not condition on weighting" — the public scheme,
            # where the single delivered flag stands in for both axes at once.
            if fluid is not None:
                selector &= (group["fluid"] == fluid)
            candidates = group[selector]

            if len(candidates) == 0 and config.rules.slot_fallback and fluid is False:
                candidates = group[(group["plane"] == plane) & (~group["fatsat"])]

            if len(candidates):
                chosen[name] = candidates.sort_values("n_slices", ascending=False).iloc[0]
        out[study] = chosen
    return out


def slot_coverage(slots: dict[str, dict[str, pd.Series]], config: Config) -> pd.DataFrame:
    """How often each slot was filled. Ours to check, not ported."""

    rows = []
    total = max(len(slots), 1)
    for name in config.slot_names:
        present = sum(name in chosen for chosen in slots.values())
        rows.append({"slot": name, "present": present, "studies": total,
                     "rate": present / total})
    return pd.DataFrame(rows).sort_values("rate")
