"""What patellofemoral osteoarthritis is, and where to put the landmark for it.

    python -m tools.atlas.pf_page -o docs/atlas/pf_oa.html

Written before the annotation round rather than after, for the reason the meniscus page
exists: a landmark whose definition lives in the annotator's head gets two annotators'
readings averaged, and a point that drifts by a centimetre between the first fifty
studies and the last fifty puts that drift straight into the model's error floor.

Everything geometric on this page is measured on this corpus. Nothing here is a
radiological reading -- the positive and negative examples are labelled from the report
extraction, which is a weak label, and they are shown as illustrations of what the text
said, not as a diagnosis made from the pixels.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from rsna.config import Config                                          # noqa: E402
from tools.atlas import page as P                                       # noqa: E402
from tools.atlas.render import stack_to_jpegs, to_jpeg, window          # noqa: E402
from tools.atlas.study import (TRAIN_SERIES, load_series,               # noqa: E402
                               series_headers, side_of, stack_orientation)

LABELS = ROOT / "data/external/pilkwang-rsna-knee-llm-labels/report_labels_v2.csv"

#: The study the anatomy figure is drawn on, and the slice through the middle of its
#: patella. A right knee, so the image's left edge is the patient's lateral side.
FIGURE_STUDY = "16599146542"
FIGURE_SLICE = 28

#: The anterior band of the slice, as fractions of the acquired image, and the zoom the
#: figure is drawn at. The marker coordinates below are in the zoomed crop's pixels --
#: they were read off a printed grid rather than guessed, so they only mean anything
#: together with these three numbers.
CROP = (0.0, 0.45, 0.12, 0.88)      # row0, row1, col0, col1 as fractions
ZOOM = 3
POINT = (350, 292)                  # the joint space, mid-patella
BOX_MM = (56.0, 40.0)


def _crop_zoom(image: np.ndarray) -> np.ndarray:
    h, w = image.shape
    r0, r1, c0, c1 = CROP
    crop = image[int(h * r0):int(h * r1), int(w * c0):int(w * c1)]
    return cv2.resize(crop, (crop.shape[1] * ZOOM, crop.shape[0] * ZOOM),
                      interpolation=cv2.INTER_CUBIC)


def _text(img, t, x, y, colour):
    cv2.putText(img, t, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 5)
    cv2.putText(img, t, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, colour, 1)


def anatomy_figure(series, with_point: bool) -> str:
    """The joint, named -- and optionally the point and the box over it."""

    big = cv2.cvtColor(_crop_zoom(window(series.volume)[FIGURE_SLICE]),
                       cv2.COLOR_GRAY2BGR)
    _text(big, "anterior", 12, 34, (210, 210, 255))
    _text(big, "LATERAL  (this is a right knee)", 12, big.shape[0] - 16, (255, 170, 170))
    _text(big, "MEDIAL", big.shape[1] - 110, big.shape[0] - 16, (170, 255, 170))
    _text(big, "PATELLA", 288, 205, (255, 230, 120))
    _text(big, "trochlea (femur)", 262, 365, (255, 230, 120))
    _text(big, "joint space", 545, 300, (120, 255, 255))

    if with_point:
        cx, cy = POINT
        px = lambda mm: int(round(mm / series.mm_per_px * ZOOM))     # noqa: E731
        cv2.rectangle(big, (cx - px(BOX_MM[0]) // 2, cy - px(BOX_MM[1]) // 2),
                      (cx + px(BOX_MM[0]) // 2, cy + px(BOX_MM[1]) // 2),
                      (0, 200, 255), 2)
        cv2.drawMarker(big, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 36, 3)
        _text(big, "the point", cx + 24, cy - 14, (120, 120, 255))
    return to_jpeg(big, max_width=880)[0]


def find(tail: str) -> str:
    hits = [d.name for d in TRAIN_SERIES.iterdir() if d.name.endswith(tail)]
    if len(hits) != 1:
        raise ValueError(f"{tail!r} matches {len(hits)} studies")
    return hits[0]


def axial_of(headers: pd.DataFrame):
    """The axial series an annotator should be shown: fat-suppressed PD, then T2 FS.

    The opposite preference to the meniscus, and for the same reason stated the other
    way round. A meniscal tear is signal *inside* fibrocartilage, which fat suppression
    flattens; patellofemoral osteoarthritis is cartilage loss and subchondral oedema,
    and oedema is only visible once the fat around it is suppressed.
    """

    ax = headers[headers["plane"] == "Axial"]
    if not len(ax):
        return None
    for weight, fat in (("PD", True), ("T2", True), ("PD", False), ("T1", False)):
        hit = ax[(ax["weight"] == weight) & (ax["fatsat"].astype(bool) == fat)]
        if len(hit):
            return hit.sort_values("n_slices", ascending=False).iloc[0]
    return ax.sort_values("n_slices", ascending=False).iloc[0]


def case(uid: str, vid: str, config: Config, note: str) -> str:
    headers = series_headers(uid)
    row = axial_of(headers)
    if row is None:
        return ""
    series = load_series(row, config)
    side, how = side_of(headers)
    low, high = stack_orientation("Axial", side)
    frames, _ = stack_to_jpegs(series.volume)
    lat = "left edge" if side == "R" else "right edge" if side == "L" else "unknown edge"
    caption = (f"{P._esc(series.label)} &middot; {series.n} slices &middot; "
               f"{series.mm_per_px:.2f} mm/px &middot; {side or '?'} knee "
               f"({P._esc(how)}) &mdash; so <b>lateral is the {lat}</b> of the image, "
               f"and anterior is the top. {note}")
    return (f'<h3>&hellip;{P._esc(uid[-11:])}</h3>'
            + P.viewer(vid, frames, low, high, series.mm_per_px, caption=caption))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--out", type=Path, default=ROOT / "docs/atlas/pf_oa.html")
    ap.add_argument("--cases", type=int, default=2, help="examples per label class")
    args = ap.parse_args()

    config = Config()
    fig_uid = find(FIGURE_STUDY)
    fig_series = load_series(axial_of(series_headers(fig_uid)), config)
    plain = anatomy_figure(fig_series, with_point=False)
    marked = anatomy_figure(fig_series, with_point=True)

    lab = pd.read_csv(LABELS)
    pos = [u for u in lab[lab["PF OA"] > 0.9]["StudyInstanceUID"]
           if (TRAIN_SERIES / u).is_dir()]
    neg = [u for u in lab[lab["PF OA"] < 0.1]["StudyInstanceUID"]
           if (TRAIN_SERIES / u).is_dir()]

    body = [f"""
<header>
  <h1>PF OA &mdash; patellofemoral osteoarthritis</h1>
  <p class="lede">The joint between the kneecap and the groove it slides in. It is the
  next target because it is the one the wide model reads worst while having the most
  positives to learn from, and because it lives in a plane this pipeline has never
  opened. This page says what it is, where it is seen, and exactly where the landmark
  goes &mdash; written before the annotation round, so that the first fifty studies and
  the last fifty get the same point.</p>
</header>

<div class="card">
<h2>1 &middot; Why this one next</h2>
<p>The twelve-target model, per target, ranked by how badly it does. Macro ROC-AUC
weights every target the same, so the question is not only where the model is weak but
where there is enough signal to fix it.</p>
<table class="report">
<tr><th>target</th><th>prevalence</th><th>positives</th><th>AUC</th></tr>
<tr><td>MCL</td><td>16.4 %</td><td>282</td><td>0.8080</td></tr>
<tr class="danger"><td><b>PF OA</b></td><td><b>47.2 %</b></td><td><b>810</b></td>
    <td><b>0.8093</b></td></tr>
<tr><td>Lateral Meniscus <span class="tag">done</span></td><td>14.8 %</td><td>254</td>
    <td>0.8206</td></tr>
<tr><td>ACL</td><td>20.3 %</td><td>349</td><td>0.8241</td></tr>
<tr><td colspan="4" class="note">&hellip; and at the other end, the compartment we
    might have done next:</td></tr>
<tr><td>Medial Meniscus</td><td>41.0 %</td><td>704</td><td>0.8785</td></tr>
<tr><td>Medial OA</td><td>36.8 %</td><td>631</td><td>0.8821</td></tr>
</table>
<p class="note">Second worst, and <b>810 positives</b> &mdash; three times what the
lateral meniscus had. These are the wide model's numbers against report-extracted
labels over two folds: the best ranking signal available, not a truth.</p>
</div>

<div class="card">
<h2>2 &middot; What the joint is</h2>
<p>Not a tibiofemoral compartment at all. The patella is a bone inside the quadriceps
tendon; its back face is cartilage, and it slides in the <b>trochlea</b>, the groove at
the front of the femur. A median ridge on the patella sits in the groove and divides its
back face into a <b>lateral facet</b> (usually the larger) and a <b>medial facet</b>.</p>
<div class="stage"><img src="{plain}" alt="an axial slice through the patellofemoral joint, labelled"></div>
<p class="note">One acquired axial slice through the middle of the patella, enlarged
{ZOOM}&times;. Fat-suppressed proton density, so fluid and oedema are bright and fat is
dark.</p>
<p>What osteoarthritis does here: the cartilage thins and the joint space narrows,
osteophytes grow at the patellar edges, the bone under the cartilage goes bright with
oedema or dark with sclerosis, and the patella can sit tilted or shifted laterally
rather than centred in its groove.</p>
</div>

<div class="card">
<h2>3 &middot; Which images show it</h2>
<p>The axial plane, which is the view that puts the patella against the trochlea and
shows both facets at once. Measured over the corpus's 4407 studies:</p>
<table class="report">
<tr><th>axial series</th><th>of studies</th></tr>
<tr><td>PD fat-suppressed</td><td>72.2 %</td></tr>
<tr><td>T2 fat-suppressed</td><td>28.2 %</td></tr>
<tr><td><b>at least one of the two</b></td><td><b>98.5 %</b></td></tr>
<tr><td>&hellip; adding T1</td><td>98.5 % (nothing)</td></tr>
</table>
<p>Fat suppression is wanted here, which is the <em>opposite</em> of the meniscus rule
and for the same reason stated the other way: a meniscal tear is signal inside
fibrocartilage, which suppression flattens; subchondral oedema is only visible once the
fat around it is suppressed.</p>
<p><b>Orientation, measured not assumed.</b> Over 3342 axial PD fat-suppressed series,
the image columns run toward the patient's <b>left</b> and the rows toward the
<b>posterior</b> &mdash; 3342 of 3342, both of them, with no exceptions. So on every
axial slice in this corpus:</p>
<ul>
<li><b>anterior is the top</b> of the image, so the patella is always at the top;</li>
<li>the image's left edge is the patient's <b>right</b>;</li>
<li>so <b>lateral is the left edge on a right knee</b>, and the right edge on a left
one &mdash; the one thing that flips between knees.</li>
</ul>
<p class="note">Because the landmark below sits on the midline of the joint, that flip
does not change where you click. It matters for the box, not for the point.</p>
</div>

<div class="card">
<h2>4 &middot; Where the landmark goes</h2>
<p>One point per study: <b>the middle of the joint space between the patella and the
femur</b> &mdash; in the bright line, halfway between the patella's lateral and medial
edges &mdash; on the slice through the middle of the patella.</p>
<div class="stage"><img src="{marked}" alt="the same slice with the landmark and a candidate box"></div>
<p class="note">The red cross is the point. The orange box is a
{BOX_MM[0]:.0f}&times;{BOX_MM[1]:.0f} mm candidate region of interest around it, drawn
only to show what the point has to be able to carry &mdash; the box itself is not
settled, and will get its own page of sizes to choose from, the way the meniscus box
did.</p>
<h3>Why the joint and not the bone</h3>
<p>Centring on the joint line rather than on the patella puts the patella in front of
the point and the trochlea behind it, so one box holds both surfaces that
osteoarthritis attacks. A point on the middle of the patella would push the trochlea to
the edge of the box; a point in the trochlear groove would push the patella out of it.</p>
<h3>Which slice</h3>
<p>The patella spans ten to fifteen axial slices. Take one through its <b>middle</b>
&mdash; scroll until the patella is at its widest, which is also where the joint is best
seen. The slices where the patella is just appearing or just vanishing are the ones to
avoid: there the joint line is a sliver and the point moves a lot for a small scroll.</p>
<h3>What would make a click wrong</h3>
<ul>
<li>clicking <b>inside the patella</b> rather than in the dark-or-bright gap behind it;</li>
<li>clicking the <b>fat pad or the tendon</b> in front of the patella, which on a slice
below the patella is the structure that takes its place;</li>
<li>clicking on a slice <b>below the patella</b> entirely &mdash; the femoral condyles
and the notch are there, and they belong to the other compartments;</li>
<li>drifting to one facet. The point is on the ridge, between the two.</li>
</ul>
</div>

<div class="card">
<h2>5 &middot; Scroll a stack</h2>
<p>Each of these is a whole axial series. Scroll from the bottom: tibia and fibula
first, then the joint, then the patella appears at the top, widens, and disappears as
you pass above it. The slice to annotate is in the middle of that span.</p>
<p class="note">The labels below come from the report extraction, not from reading these
pixels, and they are a weak label. Shown as what the report said, not as a diagnosis
&mdash; and not by a radiologist.</p>"""]

    n = 0
    for uid in pos[:args.cases]:
        body.append('<h3 class="case-head">report says PF OA</h3>')
        body.append(case(uid, f"pf{n}", config, "The report mentions patellofemoral "
                                                "osteoarthritis."))
        n += 1
        print(f"  positive {uid[-11:]}", flush=True)
    for uid in neg[:args.cases]:
        body.append('<h3 class="case-head">report says no PF OA</h3>')
        body.append(case(uid, f"pf{n}", config, "The report does not mention it."))
        n += 1
        print(f"  negative {uid[-11:]}", flush=True)
    body.append("</div>")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(P.page("PF OA — patellofemoral osteoarthritis", "".join(body)))
    print(f"\n{args.out}  ({args.out.stat().st_size/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
