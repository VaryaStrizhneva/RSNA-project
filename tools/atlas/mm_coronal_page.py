"""A coronal box for the medial meniscus, before it is a region.

    python -m tools.atlas.mm_coronal_page --n 5

Writes `docs/atlas/mm_coronal.html`. The shipped medial meniscus expert reads **one
sagittal box** -- 54 x 33 mm, five slots, 16 mm of depth. Three findings that the
radiologist actually reports are poorly served by it, and all three are coronal readings:

**Medial extrusion.** The meniscus body displaced outward past the tibial plateau margin.
It is highly specific for meniscal damage and it is *defined* on a coronal image, as the
distance between two edges that only that plane puts side by side.

**The meniscal body.** Seen in cross-section on coronal, as a wedge. On sagittal the body
is the part that reads as a continuous band and is the hardest segment to judge.

**Root truncation.** The posterior root attaches at the posterior intercondylar area.
Medial root tears are common and they are a different lesion from a body tear.

**No new landmark.** The ideal anchor for extrusion is the tibial plateau margin, which we
have never annotated. It is not worth a session of clicks and a training run: the margin
sits 10 to 15 mm from `med_centre`, so a box wide enough holds both, and our own rule says
better centring alone does not pay.

**But centring on `med_centre` costs something, and it should be said plainly.** The
landmark is defined as *the centre of the medial meniscus*, so in an extruded knee it
travels outward with the meniscus and the meniscus is always in the middle of the box.
Extrusion is a **relative** displacement, and centring on the thing that moved turns it
from an absolute offset into a measurement the model must make against the plateau margin
*inside* the box. The information survives -- the margin is in frame, at a position that
shifts -- but it is harder to read than it would be from a plateau-anchored box. This is
the one real weakness of the design and it is a consequence of the landmark we have.
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
from rsna.roi.config import SPECS                                     # noqa: E402
from tools.atlas import page as P                                     # noqa: E402
from tools.atlas.expert_page import (IN_COLOUR, _extract_for,         # noqa: E402
                                     _frames, _legend, landmark_table)
from tools.atlas.oa_sagittal_page import budget, pick                 # noqa: E402

LABELS = ROOT / "data/external/stevenleehans-rsna-knee-llm-report-labels/llm_labels_v4_blend.csv"

#: Read from the registry, never redefined here. `expert_mcl_fsonly` shipped weights
#: whose stored spec named three series while the registry's entry had two, so a page
#: resolving by name described a box the model never read. One definition, in
#: `rsna.roi.config.SPECS`, is what stops that: every number below — 69 x 36 mm on
#: 322 x 168 px, +7.5 mm toward the notch, +/-25 mm over 15 slots — and the reasoning
#: behind each lives there, next to the box the trainer will actually cut.
CORONAL = SPECS["medial_meniscus_coronal"]


def facts(spec: RoiSpec, sagittal: RoiSpec) -> str:
    px, windows = budget(spec)
    spx, swin = budget(sagittal)
    rows = [
        ("plane", "Coronal &mdash; columns run toward the patient's <b>left</b>, rows "
                  "toward the <b>inferior</b> on 3815 of 3815 series, and the depth axis "
                  "is <b>anterior-posterior</b>"),
        ("hung off", "<code>med_centre</code> &mdash; shifted "
                     f"<b>{CORONAL.box_offset_mm:+.1f} mm toward the notch</b>, no rise, "
                     f"so the box reaches "
                     f"<b>{CORONAL.box_offset_mm + CORONAL.box_w_mm / 2:.0f} mm lateral</b> "
                     f"and {abs(CORONAL.box_offset_mm - CORONAL.box_w_mm / 2):.0f} mm "
                     f"medial"),
        ("box", f"{spec.box_w_mm:.0f} &times; {spec.box_h_mm:.0f} mm on "
                f"{spec.out_w} &times; {spec.out_h} px "
                f"(<b>{spec.box_w_mm / spec.out_w:.4f} mm/px</b>) &mdash; "
                f"{spec.box_w_mm:.0f} mm ML, {spec.box_h_mm:.0f} mm SI"),
        ("depth window", f"&plusmn;{spec.lateral_mm:.0f} mm anterior-posterior, "
                         f"<b>symmetric</b>, every {spec.decimate_to_mm:.1f} mm over "
                         f"{spec.slots} slots"),
        ("series", ", ".join(f"{p} {w}{'-FS' if f else ''}" for p, w, f in spec.series)
                   + " &mdash; a fat-suppressed coronal series exists on <b>95.6 %</b> "
                     "of studies"),
        ("fits the image", "<b>99.5 %</b> of 220 measured studies, laterality resolved "
                           "per series. Room at the 5th centile: 92.2 mm toward the "
                           "notch, 40.2 mm toward the skin &mdash; so the medial 27 mm "
                           "binds and the lateral 42 mm is free"),
        ("costs", f"{px / 1e6:.2f} Mpx per study over {windows} windows, beside the "
                  f"shipped sagittal box's {spx / 1e6:.2f} over {swin} &mdash; "
                  f"<b>{(px + spx) / 1e6:.2f} Mpx</b> for the two-region expert, "
                  f"about 8 GiB at batch 32"),
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
                          f"coronal series carrying this box, or no landmark.</p>")
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
            + P.viewer(f"mmc-{tag}-{study[-8:]}", frames, lr[0], lr[1], series.mm_per_px,
                       flagged=flagged,
                       caption=f"<b style='color:{IN_COLOUR}'>Green</b> = this slice "
                               f"reaches the expert. {_legend(marks)}"))
    return "".join(blocks)


def build(out: Path, n: int) -> Path:
    config = Config()
    points = landmark_table()
    points = points[points.point.isin(("lat_centre", "med_centre"))]
    labels = pd.read_csv(LABELS, dtype={"StudyInstanceUID": str}).set_index(
        "StudyInstanceUID")
    column = next(c for c in labels.columns if c.lower().startswith("medial men"))
    yes, no = pick(labels, points, column, n)
    sagittal = SPECS["medial_meniscus"]

    blocks = [f"<p class='note'>{__doc__.split(chr(10) + chr(10), 1)[1]}</p>",
              f"<h2>Medial Meniscus &mdash; <code>{CORONAL.name}</code></h2>",
              facts(CORONAL, sagittal),
              f"<h3>Reported torn in the radiology report ({len(yes)})</h3>",
              gallery(CORONAL, yes, points, config, "y"),
              f"<h3>Not reported ({len(no)})</h3>",
              "<p class='note'>Not proof of an intact meniscus &mdash; a report is "
              "silent for many reasons. These are here so the box can be judged where "
              "there is nothing to find.</p>",
              gallery(CORONAL, no, points, config, "n"),
              f"<h2>The box it would join</h2>",
              "<p class='note'>The shipped sagittal box, drawn on the same knees, so the "
              "two regions can be compared rather than imagined. It is unchanged by this "
              "proposal: the experiment has one variable, the added plane.</p>",
              gallery(sagittal, yes[:2] + no[:2], points, config, "sag")]

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(P.page("A coronal box for the medial meniscus", "".join(blocks)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=5, help="studies per arm")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "docs" / "atlas" / "mm_coronal.html")
    args = ap.parse_args()
    path = build(args.out, args.n)
    print(f"{path}  ({path.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
