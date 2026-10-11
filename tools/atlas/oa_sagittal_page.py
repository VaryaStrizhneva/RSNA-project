"""A sagittal box for the osteoarthritis compartments, before it is an expert.

    python -m tools.atlas.oa_sagittal_page --n 5

Writes `docs/atlas/oa_sagittal.html`. The two shipped OA experts read **coronal only**,
and a coronal slice cannot separate anterior from posterior: it has no such axis. MOAKS
scores each tibiofemoral compartment in anterior, central and posterior subregions, so
two thirds of the grading scheme is invisible to the box we train on today. This page
proposes one sagittal box, drawn on knees the report calls arthritic and knees it does
not, so the geometry can be rejected before five folds are spent on it.

**One box, two landmarks.** The same geometry hangs off `med_centre` for the medial
compartment and `lat_centre` for the lateral one. Nothing new is landmarked: the point is
3-D, and projecting it onto a sagittal stack's normal picks the slice by itself. That is
already how `medial_meniscus` and `lateral_meniscus_d32` work, so the mechanism is not
new either -- only the box is bigger, because osteoarthritis needs the subchondral bone
of both the condyle and the plateau, and a meniscus needs neither.

**The image does not constrain it.** Measured over 250 studies with a fat-suppressed
sagittal series, a box reaching 40 mm from either compartment landmark stays inside the
acquisition on **99.6 %** of them in all four directions, the tightest being 46.9 mm of
room below at the 5th centile. So unlike the collateral ligament box -- where descending
60 mm worked on 67 % of studies and the measurement set the height -- the only limits
here are anatomy and the pixel budget. Both numbers below are therefore judgement, and
marked as such.

**What it is not.** It brings no plane the *host* lacks: the 0.949 CoAtNet reads 26
fat-suppressed and 22 plain sagittal slices. What it brings is 0.214 mm/px on the
anterior-posterior axis against the host's 0.365, and against the current expert's
nothing at all. Whether that pays is the open question, and the honest prior is uneven:
`nartaa`'s label expansion moved **Lateral OA by +0.048** on their 58-study gate, which
reads as a column capped by its label rather than by its pixels, while **Medial OA does
not appear in their gains at all**. So the medial compartment is the better bet, and if
only one is built it should be that one.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from rsna.config import Config                                        # noqa: E402
from rsna.roi import RoiSpec                                          # noqa: E402
from tools.atlas import page as P                                     # noqa: E402
from tools.atlas.expert_page import (IN_COLOUR, _extract_for,         # noqa: E402
                                     _frames, _legend, landmark_table)

LABELS = ROOT / "data/external/stevenleehans-rsna-knee-llm-report-labels/llm_labels_v4_blend.csv"

#: 75.0 x 63.0 mm on 350 x 294 px. Both sides divide by the ViT patch -- 25 and 21
#: patches -- and the pixels come out isotropic at 75/350 = 63/294 = 0.2143 mm/px, the
#: same pitch every other expert reads. `RoiSpec.__post_init__` refuses anything else.
#:
#: **Grown from 66.0 x 48.0 on review.** The ask was 15 % a side; this is +13.6 % AP and
#: +12.5 % SI, and the shortfall is the lattice rather than a choice. Exactly 15 % wants
#: 75.9 x 55.2 mm, which at this pitch is 354.2 x 257.6 px -- neither divisible by 14.
#: The nearest lattice points either side were +18.2 % and +9.1 %, so this is the closest
#: the geometry allows while keeping one pitch for every expert.
#:
#: 75 mm anterior-posterior: the adult femoral condyle is 60 to 65 mm front to back, and
#: the landmark sits near its mid-AP on the joint line, so +/-37.5 mm clears both ends
#: with room for the posterior condyle, which curves back behind the tibial plateau.
#: Reaching the ends is the whole point -- MOAKS' anterior and posterior subregions *are*
#: the ends. Judgement, from anatomy; the measurement only says it fits.
#:
#: 63 mm superior-inferior, asymmetric: **36 mm above the joint line and 27 below.** The
#: femur is the taller half of what matters -- subchondral bone, the marrow lesions start
#: in, and the osteophytes that grow off the condylar margin -- while the tibial plateau
#: and its subchondral plate are done within 20 mm. The first draft was symmetric at 27
#: each way; the extra 9 mm went entirely upward on review, with the lower edge pinned.
#: Judgement.
#:
#: **The depth window is unchanged at 12/18 mm.** "Each side" was read as the drawn box's
#: four in-plane edges; the medial-lateral extent is a separate parameter and growing it
#: by 15 % would move 30 mm to 34.5, which at 3.3 mm sampling is the same nine slots.
#:
#: `box_offset_mm` is zero, which is also a choice. The room
#: measurement came back near-symmetric around the landmark -- median 80.3 mm of anterior
#: room against 84.6 mm posterior at `med_centre` -- so there is nothing to correct for,
#: unlike the coronal boxes where the landmark sits at 95 % of the knee's width and a
#: centred box wastes half of itself outside the patient.
GEOMETRY = dict(
    plane="Sagittal", box_w_mm=75.0, box_h_mm=63.0, out_w=350, out_h=294,
    # Grown upward only, on review. +9 mm of height with +4.5 mm of rise leaves the lower
    # edge exactly where it was -- -27 + 4.5 - 31.5 = -27 -- and takes the upper one from
    # 27 mm above the landmark to 36. `box_rise_mm` is positive toward the top of the
    # picture, and `extract` reads that direction from each series' own orientation
    # (`to_front = sign(row . +z)`, which is -1 because rows run inferior), so a positive
    # value is superior on sagittal and coronal alike. The room measurement clears it:
    # 40 mm above either compartment landmark stays inside the acquisition on 100 % of
    # 250 studies.
    box_rise_mm=4.5,
    # Depth is medial-lateral here. 30 mm spans the compartment, biased toward the notch
    # because the weight-bearing cartilage that wears is inward of the meniscal body.
    # Nine slots at 3.3 mm, the same depth sampling as the coronal box it joins, so the
    # two regions of one expert stay comparable.
    lateral_mm=12.0, medial_mm=18.0, slots=9, decimate_to_mm=3.3,
    # The coronal OA boxes' own series policy, unchanged on purpose: the experiment this
    # page is for has one variable, the added plane. `mcl_fsonly` showed that dropping
    # the non-fat-suppressed slot was worth +0.0117 on a fat-sensitive finding, so an
    # FS-only variant is worth a later run -- but not in the same run as this.
    series=(("Sagittal", "PD", True), ("Sagittal", "T2", True),
            ("Sagittal", "PD", False)),
)

CANDIDATES = [
    (RoiSpec(name="medial_oa_sagittal", landmark="med_centre", **GEOMETRY),
     "Medial OA", "the compartment that carries most of the load, and the one "
     "<code>nartaa</code>'s label expansion did <b>not</b> move &mdash; so the one whose "
     "ceiling may still be pixels rather than labels."),
    (RoiSpec(name="lateral_oa_sagittal", landmark="lat_centre", **GEOMETRY),
     "Lateral OA", "the same geometry on the other landmark. Their labels moved this "
     "column by +0.048, which is the reading that argues against spending a run here "
     "first."),
]


def budget(spec: RoiSpec) -> tuple[int, int]:
    """Pixels per study this box costs, and how many windows it makes."""

    windows = max(1, spec.slots - 2)
    return spec.out_w * spec.out_h * windows, windows


def facts(spec: RoiSpec, note: str, coronal_px: int) -> str:
    px, windows = budget(spec)
    rows = [
        ("plane", f"{spec.plane} &mdash; columns run <b>anterior to posterior</b> on "
                  f"5563 of 5563 series, rows <b>superior to inferior</b> on 5563 of "
                  f"5563, and the depth axis is <b>medial-lateral</b>"),
        # Derived from the spec, never typed: this line said "no offset and no rise"
        # for one build after the box was grown upward, which is exactly the kind of
        # caption that describes a box the model does not read.
        ("hung off", f"<code>{spec.landmark}</code> &mdash; no offset, and raised "
                     f"<b>{spec.box_rise_mm:+.1f} mm</b>, so the box reaches "
                     f"<b>{spec.box_rise_mm + spec.box_h_mm / 2:.0f} mm above</b> the "
                     f"joint line and {abs(spec.box_rise_mm - spec.box_h_mm / 2):.0f} mm "
                     f"below it"),
        ("box", f"{spec.box_w_mm:.0f} &times; {spec.box_h_mm:.0f} mm on "
                f"{spec.out_w} &times; {spec.out_h} px "
                f"(<b>{spec.box_w_mm / spec.out_w:.4f} mm/px</b>) &mdash; "
                f"{spec.box_w_mm:.0f} mm AP, {spec.box_h_mm:.0f} mm SI"),
        ("depth window", f"{spec.lateral_mm:.0f} mm toward the bowtie, "
                         f"{spec.medial_mm:.0f} mm toward the notch, every "
                         f"<b>{spec.decimate_to_mm:.1f} mm</b> over {spec.slots} slots"),
        ("series", ", ".join(f"{p} {w}{'-FS' if f else ''}" for p, w, f in spec.series)
                   + " &mdash; a fat-suppressed sagittal series exists on <b>93.5 %</b> "
                     "of studies, against 95.6 % coronal"),
        ("costs", f"{px / 1e6:.2f} Mpx per study over {windows} windows, against "
                  f"{coronal_px / 1e6:.2f} for the coronal box it joins &mdash; "
                  f"<b>{(px + coronal_px) / 1e6:.2f} Mpx</b> for the two-region expert, "
                  f"about 15 GiB at batch 32"),
        ("why", note),
    ]
    body = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in rows)
    return f'<table class="facts"><tbody>{body}</tbody></table>'


def gallery(spec: RoiSpec, studies: list[str], points: pd.DataFrame,
            config: Config, tag: str) -> str:
    blocks = []
    for study in studies:
        built = _extract_for(study, spec, points, config)
        if built is None:
            blocks.append(f"<p class='note'>&hellip;{P._esc(study[-12:])}: no "
                          f"fat-suppressed sagittal series, or no landmark.</p>")
            continue
        series, stems, stack, row, eight, geometry, point, second = built
        frames, flagged, marks = _frames(series, stems, stack, spec, geometry, point,
                                         eight, second)
        live = int(stack.valid.sum())
        blocks.append(
            f"<h4>&hellip;{P._esc(study[-12:])} <span class='meta'>{spec.plane} "
            f"{row['weight']}{'-FS' if bool(row['fatsat']) else ''} &middot; "
            f"{len(stems)} slices at {stack.native_spacing_mm:.1f} mm &middot; "
            f"{live} of {spec.slots} slots kept</span></h4>"
            + P.viewer(f"oa-{tag}-{study[-8:]}", frames, "posterior", "anterior",
                       series.mm_per_px, flagged=flagged,
                       caption=f"<b style='color:{IN_COLOUR}'>Green</b> = this slice "
                               f"reaches the expert. {_legend(marks)}"))
    return "".join(blocks)


def pick(labels: pd.DataFrame, points: pd.DataFrame, column: str,
         n: int) -> tuple[list, list]:
    """n studies the report calls arthritic in this compartment, and n it does not."""

    have = points.groupby("study").point.nunique()
    usable = labels[labels.index.isin(set(have[have >= 2].index))]
    yes = usable[usable[column] > 0.5].index.tolist()
    no = usable[usable[column] <= 0.5].index.tolist()
    rng = np.random.default_rng(7)
    return (list(rng.choice(yes, min(n, len(yes)), replace=False)),
            list(rng.choice(no, min(n, len(no)), replace=False)))


def build(out: Path, n: int) -> Path:
    config = Config()
    points = landmark_table()
    points = points[points.point.isin(("lat_centre", "med_centre"))]
    labels = pd.read_csv(LABELS, dtype={"StudyInstanceUID": str}).set_index(
        "StudyInstanceUID")
    coronal_px = 252 * 182 * 7           # the shipped box, same windows = slots - 2

    blocks = [f"<p class='note'>{__doc__.split(chr(10) + chr(10), 1)[1]}</p>"]
    for spec, title, note in CANDIDATES:
        column = next(c for c in labels.columns
                      if c.lower().startswith(title.split()[0].lower())
                      and "oa" in c.lower())
        yes, no = pick(labels, points, column, n)
        blocks.append(f"<h2>{title} &mdash; <code>{spec.name}</code></h2>")
        blocks.append(facts(spec, note, coronal_px))
        blocks.append(f"<h3>Reported in the radiology report ({len(yes)})</h3>")
        blocks.append(gallery(spec, yes, points, config, f"{spec.name}-y"))
        blocks.append(f"<h3>Not reported ({len(no)})</h3>")
        blocks.append(f"<p class='note'>Not proof of a normal compartment &mdash; a "
                      f"report is silent for many reasons. These are here so the box can "
                      f"be judged on knees where there is nothing to find.</p>")
        blocks.append(gallery(spec, no, points, config, f"{spec.name}-n"))

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(P.page("A sagittal box for osteoarthritis", "".join(blocks)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=5, help="studies per arm, per compartment")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "docs" / "atlas" / "oa_sagittal.html")
    args = ap.parse_args()
    path = build(args.out, args.n)
    print(f"{path}  ({path.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
