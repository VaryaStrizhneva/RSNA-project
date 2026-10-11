"""Two candidate boxes for bone contusion, drawn on reported and unreported knees.

    python -m tools.atlas.contusion_page -o docs/atlas/contusion.html

Nothing here is trained. The boxes are proposals, placed by the production cropper so
that what is drawn is what a branch would receive, and shown on studies whose report
mentions a contusion **and** on studies whose report does not — because a box that holds
the finding is only interesting if the two look different inside it.

**Why the boxes are shaped the way they are.** A contusion is marrow oedema, 1–2 cm
across, and it sits in the subchondral bone of the condyles, the plateaus or the patella.
So the box must be *large* in plane — the bony knee is 75 mm wide — but it does not need
every slice, unlike a meniscal tear where the finding is a 1 mm line. That asymmetry is
the whole design: **0.22 mm/px in plane, 5–6 mm between slices**. It buys a field four
times wider than any shipped box for ten times the pixels of one, where keeping the
through-plane sampling fine as well would have cost twenty-two.

**The grid-based study that was meant to place these could not.** `contusion_map.py`
sampled a millimetre grid in fixed patient axes, which works for an axial stack — where
the superior-inferior axis is the stack's normal — and silently does not for a coronal
one, where anterior-posterior is. It therefore measured one coronal slice per study,
thirty-one times over, and its antero-posterior profile is an artefact of the plane being
slightly oblique. What survives is that on the notch slice, **73 % of the excess sits at
the two far edges**, ±30–60 mm, peaking just above the joint line. That is consistent
with condylar margins and equally consistent with the soft-tissue swelling any injured
knee carries, and it is not enough to place a box on.
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

#: Both hang off the **midpoint of the two meniscus centres** — the intercondylar notch,
#: the most central point the landmark models give us. A contusion has no landmark of its
#: own and never will: it can sit anywhere in the subchondral bone, so the origin only has
#: to be stable, which the notch is. Non-zero extents, so `extract` centres on the
#: midpoint and keeps the window declared here rather than deriving it from the two points.
CANDIDATES = [
    (RoiSpec(name="contusion_coronal", landmark="lat_centre", landmark2="med_centre",
             # Millimetres derived from pixels, not the reverse: `RoiSpec` refuses
             # anisotropic pixels, and 392x280 at 0.225 mm/px is 88.2 x 63.0 mm exactly,
             # where picking round millimetres first gave 0.2245 against 0.2286.
             plane="Coronal", box_w_mm=88.2, box_h_mm=63.0, out_w=392, out_h=280,
             lateral_mm=30.0, medial_mm=30.0, slots=12, decimate_to_mm=5.0,
             series=(("Coronal", "PD", True), ("Coronal", "T2", True))),
     "Both condyles and both plateaus in one image, which is the plane a femorotibial "
     "contusion is read in. 88 mm spans the bony width; ±30 mm of depth spans the knee "
     "front to back, where the shipped osteoarthritis boxes stop at ±14 and would miss "
     "a <b>posterolateral plateau</b> contusion — one of the two sites of the pivot-shift "
     "injury."),
    (RoiSpec(name="contusion_sagittal", landmark="lat_centre", landmark2="med_centre",
             plane="Sagittal", box_w_mm=88.2, box_h_mm=72.45, out_w=392, out_h=322,
             lateral_mm=36.0, medial_mm=36.0, slots=12, decimate_to_mm=6.0,
             series=(("Sagittal", "PD", True), ("Sagittal", "T2", True))),
     "The antero-posterior extent the coronal plane cannot give: the <b>posterior "
     "plateau</b>, the <b>patella</b> with its own subchondral bone, and the sulcus "
     "terminalis where the <i>deep lateral femoral notch sign</i> is measured. ±36 mm "
     "of depth carries it across the full width of the knee, so both compartments are "
     "seen rather than one."),
]

CONDITION = """
<p>A bone contusion is <b>marrow oedema from trauma</b> — bruised bone. No fracture line,
no cortical break: the trabeculae are injured and the marrow fills with fluid and blood.
On a fat-suppressed sequence, where the marrow's own fat has been silenced, it is an
unmistakable bright cloud in bone that should be uniformly dark. On a T1 it is the
opposite, a <i>darkening</i> of normally bright fatty marrow, which is the reading that
separates a contusion from a fracture and the reason a non-fat-suppressed sequence belongs
in this box eventually.</p>

<p><b>It follows mechanisms, not chance.</b> That is what makes a box possible at all:</p>

<ul>
<li>The <b>pivot shift</b> that tears the ACL drives the lateral femoral condyle against
the tibia: contusions at the <b>sulcus terminalis</b> of the lateral condyle — the
<i>deep lateral femoral notch sign</i> — and at the <b>posterolateral tibial
plateau</b>. These two are the "kissing contusions" and they are the commonest pattern of
all.</li>
<li><b>Valgus impaction</b>, the mechanism of a medial collateral sprain, compresses the
<b>lateral</b> compartment: condyle and plateau together.</li>
<li><b>Patellar dislocation</b> leaves the <b>medial patellar facet</b> against the
anterior lateral condyle.</li>
<li>A <b>direct blow</b> — dashboard, fall onto the knee — hits the <b>patella</b> and
the anterior tibia.</li>
</ul>

<p>The lateral femoral condyle appears in three of those four. So the distribution is far
from uniform, but it is not a point either: the region that matters is the
<b>subchondral bone</b> — within a centimetre or two of the articular surface — of the
condyles, the plateaus and the patella. A band following the joint line, plus the patella
in front. Not the whole knee, and not one spot.</p>

<p><b>Why it is worth a box at all.</b> The wide model reads <b>0.8644</b> on this column,
seventh of twelve, so there is more headroom here than on Baker's (0.8967) which still
paid. And the finding is a large bright blob in dark bone — the easiest kind of thing to
see, if the field of view holds it and the resolution is enough to separate marrow from
cortex.</p>

<p><b>And the honest caution.</b> A knee with a reported contusion has been injured, so it
also carries effusion, soft-tissue swelling, and often a torn ligament. Any model reading
these boxes can reach the right answer through those instead, and a difference map cannot
tell the two apart. That is not a reason to skip the experiment; it is a reason not to
read a good score as proof the model found marrow oedema.</p>
"""


def budget(spec: RoiSpec) -> tuple[int, int]:
    """Pixels per study this box costs, and how many windows it makes."""

    windows = max(1, spec.slots - 2)
    return spec.out_w * spec.out_h * windows, windows


def facts(spec: RoiSpec, note: str) -> str:
    px, windows = budget(spec)
    rows = [
        ("plane", spec.plane),
        ("centred on", f"the midpoint of <code>{spec.landmark}</code> and "
                       f"<code>{spec.landmark2}</code>"),
        ("box", f"{spec.box_w_mm:.0f} &times; {spec.box_h_mm:.0f} mm on "
                f"{spec.out_w} &times; {spec.out_h} px "
                f"(<b>{spec.box_w_mm / spec.out_w:.3f} mm/px</b>)"),
        ("depth window", f"&plusmn;{spec.lateral_mm:.0f} mm, symmetric, sampled every "
                         f"<b>{spec.decimate_to_mm:.0f} mm</b> over {spec.slots} slots"),
        ("series", ", ".join(f"{p} {w}{'-FS' if f else ''}" for p, w, f in spec.series)),
        ("costs", f"{px / 1e6:.2f} Mpx per study over {windows} windows &mdash; "
                  f"{px / 229376:.0f}&times; <code>lateral_meniscus_d32</code>"),
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
            blocks.append(f"<p class='note'>&hellip;{P._esc(study[-12:])}: no series or "
                          f"no landmark.</p>")
            continue
        series, stems, stack, row, eight, geometry, point, second = built
        frames, flagged, marks = _frames(series, stems, stack, spec, geometry, point,
                                         eight, second)
        live = int(stack.valid.sum())
        lr = (("posterior", "anterior") if spec.plane == "Sagittal"
              else ("medial", "lateral"))
        blocks.append(
            f"<h4>&hellip;{P._esc(study[-12:])} <span class='meta'>{spec.plane} "
            f"{row['weight']}{'-FS' if bool(row['fatsat']) else ''} &middot; "
            f"{len(stems)} slices at {stack.native_spacing_mm:.1f} mm &middot; "
            f"{live} of {spec.slots} slots kept</span></h4>"
            + P.viewer(f"c-{tag}-{study[-8:]}", frames, lr[0], lr[1], series.mm_per_px,
                       flagged=flagged,
                       caption=f"<b style='color:{IN_COLOUR}'>Green</b> = this slice "
                               f"reaches the branch. {_legend(marks)}"))
    return "".join(blocks)


def pick(labels: pd.DataFrame, points: pd.DataFrame, n: int) -> tuple[list, list]:
    """n studies the report calls contused, and n it does not, both with the landmarks."""

    column = next(c for c in labels.columns if "ontusion" in c)
    have = points.groupby("study").point.nunique()
    both = set(have[have >= 2].index)
    usable = labels[labels.index.isin(both)]
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
    yes, no = pick(labels, points, n)

    total = sum(budget(s)[0] for s, _ in CANDIDATES)
    sections = []
    for spec, note in CANDIDATES:
        sections.append(f"""
<h2>{P._esc(spec.name)}</h2>
{facts(spec, note)}
<h3>Reported contused</h3>
{gallery(spec, yes, points, config, "y")}
<h3>Not reported</h3>
<p class="note">Same box, same drawing, on knees whose report says nothing about a
contusion. If the two groups look alike inside the box, the box is not where the finding
is &mdash; which is how the effusion box was found out, after it was trained rather than
before.</p>
{gallery(spec, no, points, config, "n")}
""")

    body = f"""
<style>
table.facts{{border-collapse:collapse;margin:12px 0 20px;width:100%}}
table.facts td{{padding:6px 10px;border-top:1px solid #222a35;vertical-align:top}}
table.facts td:first-child{{color:#8b97a8;width:30%}}
.lede{{color:#aab4c2;font-size:15px;line-height:1.6}}
ul{{color:#cdd3d9;line-height:1.7;max-width:76ch}}
</style>
<h1>Bone contusion &mdash; two boxes, proposed</h1>
<p class="lede">Nothing trained. Both boxes are placed by
<code>rsna.roi.extract</code>, so what is drawn is what a branch would receive, and both
are shown on reported and unreported knees.</p>

<h2>The condition</h2>
{CONDITION}

<h2>What the two boxes would cost</h2>
<p class="note"><b>{total / 1e6:.2f} Mpx per study</b> between them, which is
{total / 229376:.0f} times one shipped box and about a third of a whole-knee box at the
same resolution. At batch 32 that is roughly {total / 1e6 * 32 * 0.43:.0f} GiB against
the 44.3 this machine has &mdash; it fits, with batch 16 as the comfortable fallback.
The trade is deliberate: <b>fine in plane, coarse through plane</b>, because the lesion
is a centimetre-scale blob and not a millimetre-scale line.</p>
{''.join(sections)}
"""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(P.page("Bone contusion — two boxes, proposed", body))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--out", type=Path,
                    default=ROOT / "docs" / "atlas" / "contusion.html")
    ap.add_argument("-n", type=int, default=3, help="Studies per group per box.")
    args = ap.parse_args()
    path = build(args.out, args.n)
    print(f"{path}  ({path.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
