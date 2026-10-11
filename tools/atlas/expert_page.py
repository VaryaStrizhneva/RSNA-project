"""One page per expert: the condition it reads, and the box it reads it in.

    python -m tools.atlas.expert_page baker --out docs/atlas/experts

Replaces the development pages, which were one per question asked rather than one per
expert, and which accumulated three generations for the ACL and three for the MCL.

**Every placement on this page comes from `rsna.roi.extract`, not from a description of
it.** The box is not drawn where the prose says it should be: the series is handed to the
production cropper with the production `RoiSpec` and the predicted landmark, and what
comes back is shown. So the two panels are the series as it arrives and the stack as the
branch receives it, and a page that disagreed with the pipeline could not be built. That
matters here more than elsewhere — the sign of `box_rise_mm` was inverted once, and a
page that re-derived the geometry would have illustrated the error rather than exposed it.

The condition prose is written, not generated, and carries no radiological authority; the
measured figures beside it do. Each is labelled with where it comes from.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pydicom

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from rsna.config import Config                                   # noqa: E402
from rsna.expert import ExpertConfig                             # noqa: E402
from rsna.roi import RoiSpec                                     # noqa: E402
# Aliased: this module's own `LANDMARKS` is the path of the predicted-point
# CSV, and importing the registry under the same name silently replaced it
# with a `Path`, which then has no `.get`.
from rsna.landmark.series import LANDMARKS as POINT_DEFS        # noqa: E402
from rsna.dicom.geometry import normal_of, pixel_of, through_plane  # noqa: E402
from rsna.roi.extract import extract                             # noqa: E402
from tools.atlas import page as P                                # noqa: E402
from tools.atlas.render import to_jpeg, window                   # noqa: E402
from rsna.dicom.ordering import order_slices                     # noqa: E402
from tools.atlas.study import load_series, series_headers        # noqa: E402

LANDMARKS = Path("/data/mgr/rsna-knee/roi-infer/landmarks.csv")
CACHE = Path("/data/mgr/rsna-knee/roi-cache")
IN_COLOUR, OUT_COLOUR, MID_COLOUR = "#39d98a", "#5a6472", "#ffd166"

#: One entry per shipped expert. `run` is the directory its weights live in, which is
#: also where its measured AUC is read from; `rois` is taken from the run's own config
#: rather than repeated here, so the page cannot describe a box the model did not read.
EXPERTS: dict[str, dict] = {
    "baker": {
        "target": "Baker's",
        "run": "expert_baker_axial",
        "title": "Baker's cyst",
        "subtitle": "the popliteal cyst, read in the plane that shows its neck",
        "condition": """
<p>A Baker's cyst is not a cyst in the strict sense: it is the joint's own fluid, escaped
backwards through a one-way gap between the <b>semimembranosus tendon</b> and the
<b>medial head of gastrocnemius</b>, and collected behind the knee. The gap is the
diagnosis. Popliteal fluid that does not pass through it is something else — a
meniscal cyst, a ganglion, a bursa — and the distinction is what the radiologist is
making when they call one.</p>

<p>Which decides the plane. On a <b>sagittal</b> image the cyst is a bright pocket behind
the femoral condyle, and its neck runs medially, straight through the slice: the plane
that shows the lesion best is the plane that hides what makes it one. The <b>axial</b>
plane cuts across the neck, so the tendon, the muscle head and the fluid between them
appear in the same image. That is why this expert reads axial even though the lesion is
larger and more obvious sagittally.</p>

<p>It is fluid, so it is read on a <b>fat-suppressed</b> sequence, where fluid is bright
against a suppressed background. PD-FS and T2-FS both serve; the box takes whichever the
study carries.</p>
""",
        "placement": """
<p>The box hangs off <b>`med_centre`</b> — the centre of the medial meniscus, predicted
by the landmark model, not annotated per study. The cyst is not there: it sits well
posterior and somewhat superior to it. So the spec carries a constant
<b>`box_rise_mm` = &minus;36&nbsp;mm</b>, and the sign is the trap. `box_rise_mm` moves
along the crop's <i>rows</i>, and rows run toward the picture's "up", which is the
<b>anterior</b> on an axial image. DICOM puts the posterior at +y. So a
<b>negative</b> rise goes <b>posterior</b>, which is where the cyst is, and the same
formula applied to a coronal box would move it <i>inferior</i> instead.</p>

<p>The landmark's value here is not that it knows where the cyst is — a constant offset
from the medial meniscus cannot know that. It is that it <b>removes variance</b>: the
same 36&nbsp;mm measured from a predicted anatomical point lands in the same place across
knees of different sizes and positions in the bore, where 36&nbsp;mm from the image
centre would not.</p>
""",
        "study": None,
    },
    "lateral_meniscus": {
        "target": "Lateral Meniscus",
        "run": "expert_lm_d32",
        "title": "Lateral meniscus tear",
        "subtitle": "the column the wide model reads worst, and the box that reads it best",
        "condition": """
<p>The meniscus is a wedge of fibrocartilage between the femoral condyle and the tibial
plateau — a C in plan view, a triangle in section, thick at the rim and tapering to a free
edge at the centre of the joint. It spreads load and it has almost no blood supply, which
is why a tear through its inner two thirds does not heal.</p>

<p>On a sagittal image the plane cuts the C <b>lengthwise</b> at the periphery, where it
appears as one continuous dark band — the <b>bowtie</b>. Move inward and the band splits
into <b>two dark triangles</b> facing each other: the anterior and posterior horns, with
the joint's centre between them. That transition is the landmark's definition and it is
also the reading: a tear is a <b>bright line</b> crossing the uniformly dark triangle, or
a triangle whose shape is wrong — blunted, displaced, absent.</p>

<p>Which is why the sequence matters more here than the contrast between fat and water.
<b>Proton density</b> gives the best signal-to-noise inside a structure that is dark on
everything, so a 1–2 mm bright line within it is visible at all. The box takes PD-FS
first and plain PD second — the <i>only</i> box in the registry whose fallback is a
non-fat-suppressed sequence, because what it needs is resolution inside dark tissue and
not fluid conspicuity.</p>

<p><b>The lateral side is the harder one.</b> The lateral meniscus is more mobile, more
circular, and its posterior horn sits close to the popliteal hiatus — a normal gap in the
capsule that mimics a tear. The medial meniscus is tethered and more regular. That
asymmetry shows in our own numbers: the wide model reads <b>0.7736</b> on this column,
its worst of twelve, against 0.8705 on the medial one.</p>
""",
        "placement": """
<p>The box is centred on <b><code>lat_centre</code></b>, with <b>no offset at all</b> —
<code>box_offset_mm</code> and <code>box_rise_mm</code> are both zero. It is the only
placement in the registry that needs none, because the landmark is defined as the thing
the box is meant to contain: midway between the two horn apexes, on the joint line.
Baker's needs 36 mm of posterior offset because its landmark is not its lesion; here the
two coincide by construction.</p>

<p><b>48 × 27 mm</b>, wider than tall, which follows the anatomy: the meniscus spans the
joint front to back and is a few millimetres thick. At 224 × 126 px that is
<b>0.214 mm/px</b> — the finest crop of the eight, and deliberately so, since the finding
is a line 1–2 mm across.</p>

<p><b>The depth window is asymmetric: 8 mm one way, 24 mm the other</b>, for 32 mm in
all — which is where <code>d32</code> in the run's name comes from. The asymmetry is the
point. The landmark sits in the band where both horns are formed, and the meniscus
continues <i>inward</i> from there toward the notch while it <i>ends</i> almost
immediately outward. So the window reaches 24 mm toward the centre of the knee and only
8 mm toward the periphery, and <code>bowtie_direction</code> is what tells the cropper
which way is which — read per study from the lit extent of the stack, never assumed.</p>

<p>The 32 mm was swept, not chosen: the registry still carries
<code>lateral_meniscus</code> at 16 mm, <code>_d24</code> at 24, <code>_l6</code> and
<code>_m20</code> varying each half separately. <code>d32</code> won and is what
ships.</p>
""",
        "study": None,
    },
    "acl": {
        "target": "ACL",
        "run": "expert_acl",
        "title": "Anterior cruciate ligament tear",
        "subtitle": "a box with no landmark of its own, placed between two that have one",
        "condition": """
<p>The ACL runs from the inner wall of the lateral femoral condyle, downward and forward
through the <b>intercondylar notch</b>, to the tibial plateau in front. It stops the tibia
sliding forward on the femur, and it is the ligament that fails in a pivoting injury.</p>

<p>On a sagittal image an intact ligament is a <b>dark band running parallel to the roof
of the notch</b> — Blumensaat's line — taut and straight from one attachment to the other.
A tear shows as discontinuity, a wavy or absent band, or bright signal through what should
be uniformly dark. The reading is often easier from what surrounds it than from the
ligament itself: the tibia translated forward, the posterior cruciate buckled into a
tighter curve than it should be, bone bruising on the lateral condyle, a <b>Segond
fracture</b> of the lateral tibial rim.</p>

<p><b>Its obliquity is the difficulty.</b> The ligament crosses all three planes at once,
so no single sagittal slice holds it end to end; it is seen across a few neighbouring
slices and the reader assembles it. That is also why the coronal plane is the first thing
we would add to this box — it resolves the femoral attachment and separates a partial tear
from a complete one, which the sagittal plane struggles to do.</p>

<p>Read on proton density, PD-FS first and plain PD second, for the same reason as the
menisci: the structure is dark on every sequence, so what is wanted is signal-to-noise
inside dark tissue rather than fluid conspicuity.</p>
""",
        "placement": """
<p><b>This box has no landmark of its own.</b> Nobody ever clicked the ACL; the registry
carries an <code>acl_centre</code> definition and no model was fitted for it. Instead the
box is centred on the <b>midpoint of <code>lat_centre</code> and
<code>med_centre</code></b> — the two meniscus centres — and the midpoint of the two
compartments <i>is</i> the intercondylar notch, which is the corridor the ligament runs
through. The anatomy supplies the point that the annotation never did.</p>

<p>It is the only spec in the registry with a <code>landmark2</code>, and the only one
whose depth extent is <b>derived rather than declared</b>. Its
<code>lateral_mm</code> and <code>medial_mm</code> are both zero, and that zero is a
signal, not a width: <code>extract</code> reads it as "take the span between the two
points, inset at each end" — 12 mm here. The window then <b>follows the knee</b> instead
of fixing a number, which is what this box needs, since the distance between the menisci
is the width of the notch and varies from 27 to 48 mm across the 294 studies carrying both
points.</p>

<p>That coupling had to be broken to make the whole-knee boxes possible. For a while
<code>landmark2</code> meant both "centre between the two points" <i>and</i> "take your
depth from them", which is right for a 27–48 mm notch and wrong for a 90 mm knee. The two
are now separate: a zero extent derives the window from the points, a non-zero one keeps
the spec's own.</p>

<p>Because the derived window is symmetric by construction,
<b><code>bowtie_direction</code> is never consulted</b> — which matters, since that rule
was measured on the medial-lateral axis and would be answering a different question here.</p>

<p>The box is <b>60 × 52 mm</b>, the largest of the eight, because the ligament is long
and obliquely placed. It is nudged <b>4 mm posterior and 8 mm superior</b> of the midpoint
— on a sagittal crop a positive offset along columns goes backwards and a positive rise
goes up — which carries it toward the roof of the notch, where the femoral attachment is
and where a tear is most often read.</p>
""",
        "study": None,
    },
    "medial_meniscus": {
        "target": "Medial Meniscus",
        "run": "expert_medial_meniscus",
        "title": "Medial meniscus tear",
        "subtitle": "the same reading as the lateral side, on a structure that moves less",
        "condition": """
<p>Same tissue, same reading, easier target. Like the lateral meniscus it is a wedge of
fibrocartilage seen as a <b>bowtie</b> at the periphery and as <b>two dark triangles</b>
further in, and a tear is a bright line crossing one of them or a horn whose shape is
wrong.</p>

<p>Two things make the medial side more tractable. It is <b>tethered</b> to the deep layer
of the medial collateral ligament along its whole rim, so it moves less and deforms less
between studies. And it is more <b>C-shaped and regular</b> than the lateral, which is
nearly circular — so the bowtie-to-triangles transition happens at a more predictable
depth. The wide model reads <b>0.8705</b> here against 0.7736 on the lateral side, and the
difference is anatomy rather than labelling.</p>

<p>Its own failure mode is <b>extrusion</b>: the body squeezed out past the edge of the
tibial plateau, which is a major marker of osteoarthritis and which the sagittal plane
shows poorly. That is the coronal plane's finding, and it is the first thing this box
lacks.</p>

<p>Read on proton density, PD-FS then plain PD — a dark structure needs signal-to-noise,
not fluid conspicuity.</p>
""",
        "placement": """
<p>Centred on <b><code>med_centre</code></b> with <b>no offset</b>, the same arrangement
as the lateral box and for the same reason: the landmark is defined as midway between the
two horn apexes on the joint line, which is what the box is meant to hold.</p>

<p><b>54 × 33 mm at 0.214 mm/px</b> — slightly larger than the lateral box in both
directions, because the medial meniscus is longer front to back.</p>

<p>Where the two differ is the <b>depth window: 16 mm here against 32 on the lateral
side</b>, 12 mm inward and 4 mm outward, over <b>5 slots</b> rather than 9. This box is
the <i>unswept</i> version of the design. The lateral one went through
<code>_d24</code>, <code>_d32</code>, <code>_l6</code> and <code>_m20</code> before
settling at 32 mm; nothing equivalent was ever run for the medial side, which still
carries the original 16 mm. Given that the sweep moved the lateral column and that the
anatomy is comparable, <b>this is the most obvious untested change in the whole set</b> —
and it costs one training to find out.</p>
""",
        "study": None,
    },
    "mcl": {
        "target": "MCL",
        "run": "expert_mcl_fsonly",
        "title": "Medial collateral ligament injury",
        "subtitle": "a tall thin box on a coronal landmark, and the one case where removing a sequence paid",
        "condition": """
<p>The MCL is a flat band on the inner side of the knee, running from the medial femoral
epicondyle down to the tibia, and it resists the knee being pushed inward. It is the most
commonly injured ligament of the knee and the one that most often heals without
surgery.</p>

<p>It is a <b>coronal</b> structure in the plainest sense: the plane contains its whole
length, from one attachment to the other, in a single slice. An intact ligament is a thin
dark line hugging the bone. An injury is graded by what surrounds it — fluid and oedema
<b>along</b> it for a sprain, the band thickened and lifted off the bone for a partial
tear, frank discontinuity for a complete one.</p>

<p>So unlike the menisci, <b>this reading is about fluid</b>, and that decides the
sequences: fat-suppressed only, PD-FS first and T2-FS second. Which is where this box has
its one hard measurement.</p>
""",
        "placement": """
<p>Hung off <b><code>mcl_centre</code></b> — the ligament where it crosses the joint line,
annotated on 209 studies and predicted with a held-out p90 of 4.6 mm.</p>

<p>The box is <b>32 × 80 mm</b>, and it is the only one in the registry that is
<b>taller than it is wide</b>. That is the ligament's shape: a narrow band spanning a long
vertical distance. It is also offset <b>5 mm laterally and 12 mm inferiorly</b> of the
landmark, which carries it down the tibial attachment where a sprain's oedema collects.
On a coronal crop a positive rise goes <i>superior</i>, so the &minus;12 here goes
<b>down</b> — the sign that was inverted once and put the collateral box 52 mm above the
joint line instead of 28 below.</p>

<p>Its depth window is <b>symmetric, ±12 mm</b>, so <code>bowtie_direction</code> is never
consulted — correctly, since that rule was measured on the medial-lateral axis and a
coronal stack's through-plane direction is anterior-posterior.</p>

<p><b>And this is the box that produced our one hard result about series.</b> The run is
named <code>fsonly</code> because it reads <b>two</b> fat-suppressed sequences where the
registry's spec of the same name has since gained a third, non-fat-suppressed one.
Dropping that third sequence was worth <b>+0.0118</b> on this column. The reason is
contrast: on a non-fat-suppressed image fluid is <i>dark</i>, so that channel contradicts
exactly what the box is looking for. It is the measurement behind the rule that a
contradicting channel costs — and the measurement that must <b>not</b> be quoted against
two sequences that agree, which is a mistake this project made and corrected.</p>
""",
        "study": None,
    },
    "lateral_oa": {
        "target": "Lateral OA",
        "run": "expert_lateral_oa_coronal",
        "title": "Lateral tibiofemoral osteoarthritis",
        "subtitle": "cartilage and bone in the plane that shows the joint space",
        "condition": """
<p>Osteoarthritis is not one finding but a set that appear together: <b>cartilage
thinning</b> over the femoral condyle and tibial plateau, <b>osteophytes</b> at the
margins, <b>subchondral</b> sclerosis and cysts, and bone marrow oedema beneath the worn
surface.</p>

<p>The <b>coronal</b> plane is where the joint space itself is read — the gap between
condyle and plateau, compared side to side within the same image. A sagittal slice shows
the cartilage in profile but cannot compare the two compartments, and comparing them is
most of the judgement.</p>

<p>Three sequences serve, in order: PD-FS and T2-FS for the fluid-bright findings —
oedema, cysts, cartilage defects filled with joint fluid — and plain PD for the bone. That
third one is kept here, unlike on the MCL box, because for osteoarthritis the
<b>non-fat-suppressed image is not contradicting the finding</b>: osteophytes and
sclerosis are bone, and bone reads better where fat has not been suppressed.</p>
""",
        "placement": """
<p>Centred on <b><code>lat_centre</code></b> with <b>no offset</b> — the same point the
lateral meniscus box uses, read on a coronal series instead of a sagittal one. The
meniscus centre sits on the joint line, and the joint line is what this box is about, so
no displacement is needed.</p>

<p>Using one landmark across two planes is the economy that made this box free: no
annotation campaign, no second model. The landmark is a point in patient millimetres, and
<code>pixel_of</code> projects a point into any slice of any plane &mdash; so a sagittal
point places a coronal box with no extra machinery.</p>

<p><b>54 × 39 mm at 0.214 mm/px</b>, wide enough to hold the compartment and tall enough
to reach the subchondral bone above and below the line. The depth window is
<b>symmetric, ±14 mm</b> over <b>9 slots</b>, which covers the compartment front to back
without reaching the notch.</p>
""",
        "study": None,
    },
    "medial_oa": {
        "target": "Medial OA",
        "run": "expert_medial_oa",
        "title": "Medial tibiofemoral osteoarthritis",
        "subtitle": "the same box on the other compartment, and the side that wears first",
        "condition": """
<p>The same findings as the lateral compartment — cartilage loss, osteophytes, subchondral
change — on the side that carries more load and wears sooner. Medial compartment
osteoarthritis is the common pattern, and it is what varus alignment produces.</p>

<p>It travels with the medial meniscus: <b>extrusion</b> of that meniscus and loss of the
medial joint space are the same disease seen twice, which is why those two columns of the
submission correlate more with each other than either does with the lateral side.</p>

<p>Read coronally, on PD-FS, T2-FS and plain PD — the fluid-bright findings on the first
two, the bone on the third.</p>
""",
        "placement": """
<p><b>The mirror of the lateral box, field for field</b>: centred on
<code>med_centre</code> with no offset, 54 × 39 mm at 0.214 mm/px, symmetric ±14 mm over
9 slots, the same three coronal sequences in the same order. Only the landmark differs.</p>

<p>That symmetry was a deliberate choice and it had to be checked rather than assumed,
because the two compartments are not mirror images anatomically — the medial side is
larger and flatter. The two boxes scoring within 0.01 of each other out of fold is what
says the shared geometry is adequate for both.</p>

<p>One caution that applies to this box in particular. The specs for the two compartments
differ <i>only</i> in their landmark, so a cache keyed on the fields that decide pixels
will give them the same filename unless the landmark is one of those fields. It was not,
once: <code>medial_oa</code> read <code>lateral_oa</code>'s cache and scored the wrong
compartment. The key now includes the landmark.</p>
""",
        "study": None,
    },
    "pf_oa": {
        "target": "PF OA",
        "run": "expert_pf_oa",
        "title": "Patellofemoral osteoarthritis",
        "subtitle": "the kneecap against its groove, in the only plane that sees both facets",
        "condition": """
<p>The patella runs in the trochlear groove at the front of the femur, and the cartilage on
its back is the thickest in the body. Patellofemoral osteoarthritis is wear of that
interface: cartilage thinning on one or both <b>facets</b>, subchondral change behind
them, osteophytes at the patellar edges.</p>

<p>The <b>axial</b> plane is the only one that shows the joint as it works — patella and
trochlea in cross-section, the medial and lateral facets side by side with the groove
between them. It is also the plane that shows <b>maltracking</b>: a patella sitting off
centre or tilted in its groove, which is the mechanical cause of much of the wear.</p>

<p>A sagittal slice would show patellar height and trochlear depth, which the axial plane
cannot, and that is the first thing this box lacks. Read on PD-FS then T2-FS: the findings
are cartilage defects and subchondral oedema, both fluid-bright.</p>
""",
        "placement": """
<p>Hung off <b><code>pf_centre</code></b> — the middle of the patellofemoral joint space,
annotated on 200 studies. It is the hardest of the four points to place and its numbers
show it: a held-out p90 of 5.0 mm, the worst of the four, and the worst in-plane error
too (1.41 mm median against 0.82 for the lateral meniscus). The reason is in the
definition: "the middle of the joint space" is a <b>region</b>, not a geometric feature
like the midpoint between two horn apexes, so two sensible annotators would place it
differently.</p>

<p><b>56 × 48 mm at 0.286 mm/px</b>, and raised <b>8 mm</b> off the landmark. On an axial
crop a positive rise goes <b>anterior</b> — DICOM puts the posterior at +y, so up in the
picture is &minus;y — which carries the box forward, toward the patella itself rather than
sitting centred on the joint space. The same +8 on a coronal box would have moved it
<i>superior</i> instead; the two planes do not share a sign, and that is the trap this
field has sprung before.</p>

<p>Depth window <b>symmetric, ±16 mm</b> over <b>9 slots</b>, which spans the patella from
its upper to its lower pole on most studies.</p>
""",
        "study": None,
    },
}


def landmark_table() -> pd.DataFrame:
    """The predicted landmarks, one row per study and point."""

    return pd.read_csv(LANDMARKS, dtype={"study": str})


def point_for(points: pd.DataFrame, study: str, name: str) -> np.ndarray | None:
    row = points[(points.study == study) & (points.point == name)]
    if row.empty:
        return None
    return row.iloc[0][["x_mm", "y_mm", "z_mm"]].to_numpy(float)


def geometry_of(folder: Path, config: Config) -> tuple[list[str], list[dict]]:
    """The slice names in the order `load_series` stacks them, and their geometry.

    `order_slices` is the pipeline's own ordering, so calling it rather than sorting by
    filename is what keeps the geometry list aligned with the volume. The names are the
    SOP UIDs — the extractor copies whatever `sop` it is handed into `RoiStack.sop`, so
    putting the stem here is what lets a kept slot be matched back to its slice.
    """

    names = sorted(x.name for x in folder.glob("*.dcm"))
    names, _ = order_slices(str(folder), names, config)
    out = []
    for name in names:
        ds = pydicom.dcmread(str(folder / name), force=True, stop_before_pixels=True)
        try:
            out.append({"sop": Path(name).stem,
                        "ipp": [float(v) for v in ds.ImagePositionPatient],
                        "iop": [float(v) for v in ds.ImageOrientationPatient],
                        "ps": [float(v) for v in ds.PixelSpacing]})
        except Exception:                                      # noqa: BLE001
            out.append({})
    return [Path(n).stem for n in names], out


def pick_series(headers: pd.DataFrame, spec) -> pd.Series | None:
    """The first series of the spec's own priority list that this study carries."""

    for plane, weight, fs in spec.series:
        hit = headers[(headers.plane.astype(str) == plane)
                      & (headers.weight.astype(str) == weight)
                      & (headers.fatsat.astype(bool) == bool(fs))]
        if len(hit):
            return hit.sort_values("n_slices", ascending=False).iloc[0]
    return None


def _extract_for(study: str, spec, points: pd.DataFrame, config: Config):
    """Hand this study's chosen series to the production cropper, or `None`.

    Factored out so the one study shown in full and the examples at the foot of the page
    are the same call. Returning `None` rather than raising covers the two ordinary
    reasons a study has nothing to show — it carries none of the box's series, or the
    landmark model never placed its point — which are facts about the corpus and not
    errors.
    """

    row = pick_series(series_headers(study), spec)
    if row is None:
        return None
    point = point_for(points, study, spec.landmark)
    if point is None:
        return None
    series = load_series(row, config)
    stems, geometry = geometry_of(Path(row["dir"]), config)
    geometry, stems = geometry[:len(series.volume)], stems[:len(series.volume)]
    second = (point_for(points, study, spec.landmark2)
              if getattr(spec, "landmark2", None) else None)
    eight = window(series.volume)
    stack = extract(eight, geometry, point, spec, point2_mm=second)
    return series, stems, stack, row, eight, geometry, point, second


def _marks(spec, geometry, point, second) -> list[tuple[int, int, int, str, str]]:
    """Every point worth drawing: each landmark on its own slice, and their midpoint.

    `stack.landmark_rc` is the midpoint when a spec carries two landmarks — that is what
    the cropper hangs the box from — so drawing it alone, on the slice nearest the *first*
    point, put a correct in-plane position on the wrong slice and said nothing about the
    second. Each landmark is therefore projected on its own, with `pixel_of` over the
    geometry already read; no spec logic is involved, so there is nothing here to drift
    from `extract`.
    """

    usable = [i for i, g in enumerate(geometry) if g and g.get("ipp") is not None]
    if not usable:
        return []
    g0 = geometry[usable[0]]
    axis = normal_of(np.asarray(g0["iop"], float))
    depth = np.array([through_plane(g["ipp"], axis) if g and g.get("ipp") is not None
                      else np.inf for g in geometry])

    def place(mm, colour, label):
        home = int(np.argmin(np.abs(depth - through_plane(mm, axis))))
        r, c = pixel_of(g0["ipp"], g0["iop"], g0["ps"], mm)
        return (home, int(round(r)), int(round(c)), colour, label)

    named = [(spec.landmark, point)]
    if second is not None and getattr(spec, "landmark2", None):
        named.append((spec.landmark2, second))
    out = [place(mm, POINT_DEFS.get(name, {}).get("colour", "#ffd166"), name)
           for name, mm in named]
    if len(named) > 1:
        out.append(place((point + second) / 2.0, MID_COLOUR,
                         "their midpoint, which the box hangs from"))
    return out


def _frames(series, stems, stack, spec, geometry, point, eight, second=None):
    """The series' slices with the crop drawn on each, green where it was kept.

    `box_rc` comes off the stack: `extract` reports where it cropped, so this drawing
    cannot disagree with the crop. Each landmark goes on the one slice it sits on — the
    box is the same in plane on every slice of a parallel stack, a point is not.
    """

    kept = {s for s in stack.sop if s}
    r0, r1, c0, c1 = (int(round(v)) for v in stack.box_rc)
    marks = _marks(spec, geometry, point, second)

    frames, flagged = [], []
    for i, sop in enumerate(stems):
        rgb = cv2.cvtColor(eight[i], cv2.COLOR_GRAY2BGR)
        inside = sop in kept
        if inside:
            flagged.append(i)
        colour = IN_COLOUR if inside else OUT_COLOUR
        bgr = tuple(int(colour[j:j + 2], 16) for j in (5, 3, 1))
        thick = max(1, int(round(min(rgb.shape[:2]) / 260))) + (1 if inside else 0)
        cv2.rectangle(rgb, (c0, r0), (c1, r1), bgr, thick)
        for home, mr, mc, mcol, _ in marks:
            if i != home:
                continue
            mbgr = tuple(int(mcol[j:j + 2], 16) for j in (5, 3, 1))
            cv2.drawMarker(rgb, (mc, mr), mbgr, cv2.MARKER_CROSS,
                           max(10, int(min(rgb.shape[:2]) / 18)), thick + 1)
        uri, _ = to_jpeg(rgb, max_width=420)
        frames.append(uri)
    return frames, flagged, marks


def _legend(marks: list) -> str:
    """Which cross is which, in the colours they were drawn in."""

    return " &nbsp; ".join(
        f"<span style='color:{colour}'>&#10010; <code>{P._esc(label)}</code>"
        f"</span>" if label in POINT_DEFS else
        f"<span style='color:{colour}'>&#10010; {label}</span>"
        for _, _, _, colour, label in marks)


def more_examples(spec, points: pd.DataFrame, config: Config, n: int,
                  skip: str) -> str:
    """The same box on several more studies, scrollable, nothing else.

    Spread across the studies carrying the landmark rather than taken consecutively, so
    the series the box falls back to varies — on `baker_axial` the first choice is Axial
    PD-FS and the second Axial T2-FS, and a run of neighbours would show one of them
    only. Each viewer is a separate `extract` call: what is green is what that study's
    own crop kept, not a rule applied to a picture.
    """

    have = [s for s in points[points.point == spec.landmark].study.tolist() if s != skip]
    if not have:
        return "<p class='note'>no study carries this landmark.</p>"
    picked = [have[int(round(i))] for i in
              np.linspace(0, len(have) - 1, min(n, len(have)))]

    blocks = []
    for study in dict.fromkeys(picked):
        try:
            one = series_with_box(study, spec, points, config)
        except Exception as exc:                                   # noqa: BLE001
            blocks.append(f"<p class='note'>&hellip;{P._esc(study[-12:])}: "
                          f"{P._esc(type(exc).__name__)}</p>")
            continue
        if one is None:
            continue
        html, note = one
        blocks.append(f"<h4>&hellip;{P._esc(study[-12:])} "
                      f"<span class='meta'>{note}</span></h4>{html}")
    return "".join(blocks)


def series_with_box(study: str, spec, points: pd.DataFrame,
                    config: Config) -> tuple[str, str] | None:
    """One scrollable series with the crop drawn on it, and a line about this study."""

    built = _extract_for(study, spec, points, config)
    if built is None:
        return None
    series, stems, stack, row, eight, geometry, point, second = built
    frames, flagged, marks = _frames(series, stems, stack, spec, geometry, point, eight,
                                     second)
    live = int(stack.valid.sum())
    lr = (("posterior", "anterior") if spec.plane == "Sagittal"
          else ("medial", "lateral"))
    note = (f"{spec.plane} {row['weight']}{'-FS' if bool(row['fatsat']) else ''} "
            f"&middot; {len(stems)} slices at {stack.native_spacing_mm:.1f} mm "
            f"&middot; {live} of {spec.slots} slots kept")
    return (P.viewer(f"ex-{study[-8:]}", frames, lr[0], lr[1], series.mm_per_px,
                     flagged=flagged,
                     caption=f"<b style='color:{IN_COLOUR}'>Green</b> = this slice "
                             f"reached the branch. {_legend(marks)}"),
            note)


def panels(study: str, spec, points: pd.DataFrame, config: Config) -> str:
    """The series as it arrives, and the stack as the branch receives it.

    Both come out of one `extract` call, so neither can drift from the pipeline nor from
    the examples at the foot of the page, which share these helpers.
    """

    built = _extract_for(study, spec, points, config)
    if built is None:
        return (f"<p class='note'>&hellip;{P._esc(study[-12:])} carries none of this "
                f"box's series, or has no predicted "
                f"<code>{P._esc(spec.landmark)}</code>.</p>")
    series, stems, stack, row, eight, geometry, point, second = built
    left_frames, flagged, marks = _frames(series, stems, stack, spec, geometry, point,
                                          eight, second)

    right_frames = []
    for slot in range(stack.volume.shape[0]):
        rgb = cv2.cvtColor(stack.volume[slot], cv2.COLOR_GRAY2BGR)
        if not stack.valid[slot]:
            cv2.putText(rgb, "padding", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (90, 100, 114), 1, cv2.LINE_AA)
        uri, _ = to_jpeg(rgb, max_width=420)
        right_frames.append(uri)

    live = int(stack.valid.sum())
    lr = (("posterior", "anterior") if spec.plane == "Sagittal"
          else ("medial", "lateral"))
    r0, r1, c0, c1 = (int(round(v)) for v in stack.box_rc)
    lr_row, lr_col = (int(round(v)) for v in stack.landmark_rc)
    facts = [
        ("series taken", f"{spec.plane} {row['weight']}"
                         f"{'-FS' if bool(row['fatsat']) else ''} &mdash; "
                         f"{P._esc(row.get('SeriesDescription') or '?')}"),
        ("slices in the series", f"{len(stems)} at {stack.native_spacing_mm:.2f} mm"),
        ("kept by the window", f"{live} of {spec.slots} slots"),
        ("thinning stride", f"{stack.stride}&times;"
                            + ("" if stack.stride > 1 else " (none needed)")),
        ("centre slot", f"{stack.centre_slot}"),
        ("landmarks sit on", ", ".join(
            f"<code>{P._esc(label)}</code> on slice {home}"
            + ("" if home in flagged else " <i>(not kept)</i>")
            for home, _, _, _, label in marks if label in POINT_DEFS)
            + f", of {len(stems)}"),
        ("box in the source image",
         f"rows {r0}&ndash;{r1}, columns {c0}&ndash;{c1}, around the landmark at "
         f"({lr_row}, {lr_col})"),
        ("direction consulted", "none &mdash; the window is symmetric"
         if spec.symmetric_depth else
         f"<code>bowtie_direction</code> = {stack.toward_bowtie:+.0f}"),
        ("crop mirrored", "yes, to put lateral on the right" if stack.flipped else "no"),
    ]
    fact_rows = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in facts)

    return f"""
<div class="pair">
  <div>
    <h4>The series as it arrives <span class='meta'>{len(stems)} slices</span></h4>
    {P.viewer(f"x-{study[-8:]}-full", left_frames, lr[0], lr[1], series.mm_per_px,
              flagged=flagged,
              caption=f"The rectangle is the crop, on every slice. Each cross is on "
                      f"the one slice its point sits on. {_legend(marks)}. "
                      f"<b style='color:{IN_COLOUR}'>Green</b> = this slice reached the "
                      f"branch, grey = dropped.")}
  </div>
  <div>
    <h4>The stack the branch receives
        <span class='meta'>{spec.out_w}&times;{spec.out_h} px</span></h4>
    {P.viewer(f"x-{study[-8:]}-roi", right_frames, lr[0], lr[1],
              spec.box_w_mm / spec.out_w,
              caption=f"The actual model input, straight out of "
                      f"<code>rsna.roi.extract</code>. {live} live slots of "
                      f"{spec.slots}.")}
  </div>
</div>
<table class="facts"><tbody>{fact_rows}</tbody></table>
"""


def _window(spec) -> str:
    """How deep the box reaches, and where that number comes from.

    Three cases, and the third reads as a mistake if it is printed as a number. A spec
    with two landmarks and a zero extent does not have a window of zero millimetres: the
    zero is the *signal* that the extent comes from the two points instead, inset at each
    end. `extract` branches on exactly this, so the page has to as well.
    """

    if getattr(spec, "landmark2", None) and not (spec.lateral_mm or spec.medial_mm):
        return (f"derived from the two landmarks, not declared &mdash; the box spans the "
                f"distance between them less <b>{spec.depth_inset_mm:.0f} mm</b> at each "
                f"end, which follows the knee rather than fixing a width. Measured over "
                f"the 294 studies carrying both points, that is <b>27 mm</b> on the "
                f"narrowest and <b>48 mm</b> on the widest")
    total = spec.lateral_mm + spec.medial_mm
    if spec.symmetric_depth:
        return (f"{total:.0f} mm, &plusmn;{spec.lateral_mm:.0f} mm, symmetric &mdash; so "
                f"<code>bowtie_direction</code> is never consulted")
    return (f"{total:.0f} mm &mdash; {spec.medial_mm:.0f} mm toward the centre of the "
            f"stack and {spec.lateral_mm:.0f} mm toward its periphery")


def spec_table(spec) -> str:
    """The box, field by field, as the registry declares it."""

    window_mm = spec.lateral_mm + spec.medial_mm
    rows = [
        ("plane", spec.plane),
        ("landmark", f"<code>{spec.landmark}</code>"
                     + (f" and <code>{spec.landmark2}</code>, box centred on their "
                        f"midpoint" if getattr(spec, "landmark2", None) else "")),
        ("box in millimetres", f"{spec.box_w_mm:.0f} &times; {spec.box_h_mm:.0f} mm"),
        ("output in pixels", f"{spec.out_w} &times; {spec.out_h} px "
                             f"({spec.box_w_mm / spec.out_w:.3f} mm/px)"),
        ("offset from the landmark",
         f"{spec.box_offset_mm:+.0f} mm along columns, "
         f"{spec.box_rise_mm:+.0f} mm along rows"),
        ("depth window", _window(spec)),
        ("slots", f"{spec.slots}"),
        ("decimated to", f"{spec.decimate_to_mm:.1f} mm"),
        ("series priority",
         "<ol class='prio'>" + "".join(
             f"<li>{p} {w}{'-FS' if f else ''}</li>" for p, w, f in spec.series)
         + "</ol>"),
    ]
    body = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in rows)
    return f'<table class="facts"><tbody>{body}</tbody></table>'


def _cached_as(spec):
    """The spec under whose *name* this box was actually cut on disk.

    A cache is keyed on the fields that decide its pixels, but `cache.load` then compares
    the whole `RoiSpec`, name included. `expert_mcl_fsonly` carries a spec called `mcl`
    with two series while the registry's `mcl` has gained a third, so the file written
    for this geometry lives under `mcl_fsonly` and loading by the stored name fails.

    So: if the stored name does not match, look for the one registry entry that agrees on
    **every field but the name**, and read that. Equality on all the rest is the proof
    they are the same box; nothing is assumed from the names being similar.
    """

    from rsna.roi import SPECS

    known = SPECS.get(spec.name)
    if known is not None and known == spec:
        return spec
    fields = [f for f in spec.__dataclass_fields__ if f != "name"]
    for candidate in SPECS.values():
        if all(getattr(candidate, f) == getattr(spec, f) for f in fields):
            return candidate
    return spec


def measured(entry: dict, spec) -> str:
    """Coverage and held-out AUC, read off the run rather than quoted."""

    bits = []
    run = ROOT / "out" / entry["run"]
    hist = run / "history.json"
    if hist.is_file():
        h = json.loads(hist.read_text())
        oof, gold = h.get("oof", {}), h.get("gold", {})
        folds = len(h.get("history", {}) or {})
        # `history.json` carries whatever `result.scores` held, and for runs trained
        # between 2026-10-03 and the fix in 791b6b4 that was the **last** epoch while
        # the saved weights were the best one — 0.025 of AUC apart per fold on this run.
        # A rescore from the weights themselves is therefore preferred where it exists,
        # and which one was used is printed rather than left to be guessed.
        rescored = run / "holdout_rescored.csv"
        if rescored.is_file():
            table = pd.read_csv(rescored)
            from rsna.expert.loop import aucs
            pooled = float(np.nanmean(aucs(table[["true"]].to_numpy(np.float32),
                                           table[["pred"]].to_numpy(np.float32))))
            bits.append(f"<tr><td>held-out AUC, pooled</td><td><b>{pooled:.4f}</b> over "
                        f"{len(table)} studies, {folds} folds &mdash; rescored from the "
                        f"saved weights, because this run's <code>history.json</code> "
                        f"recorded its last epoch rather than the epoch it kept "
                        f"({oof.get('mean', float('nan')):.4f})</td></tr>")
        else:
            bits.append(f"<tr><td>held-out AUC, pooled</td><td><b>"
                        f"{oof.get('mean', float('nan')):.4f}</b> over "
                        f"{oof.get('n', 0)} studies, {folds} folds</td></tr>")
        if gold:
            bits.append(f"<tr><td>against the 58 radiologist readings</td><td>"
                        f"{gold.get('mean', float('nan')):.4f} &mdash; wide interval, "
                        f"only a collapse is readable there</td></tr>")
        bits.append(f"<tr><td>labels it was fitted on</td><td><code>"
                    f"{Path(h.get('labels', '?')).name}</code>, weighted by "
                    f"{h.get('run', {}).get('weights', '?')}</td></tr>")
    try:
        from rsna.roi import cache
        _, mask, _ = cache.load(CACHE, _cached_as(spec))
        live = mask.any(axis=(1, 2))
        bits.append(f"<tr><td>studies the box can be cut for</td><td>"
                    f"<b>{100 * live.mean():.1f} %</b> of {len(live)} "
                    f"&mdash; the rest reach the branch as padding, and the chain "
                    f"abstains on them</td></tr>")
    except Exception as exc:                                   # noqa: BLE001
        bits.append(f"<tr><td>coverage</td><td class='note'>not read: "
                    f"{P._esc(type(exc).__name__)}</td></tr>")
    return f'<table class="facts"><tbody>{"".join(bits)}</tbody></table>'


EXTRA_CSS = """
.pair{display:grid;grid-template-columns:1fr 1fr;gap:26px;align-items:start}
@media (max-width:900px){.pair{grid-template-columns:1fr}}
table.facts{border-collapse:collapse;margin:14px 0 22px;width:100%}
table.facts td{padding:6px 10px;border-top:1px solid #222a35;vertical-align:top}
table.facts td:first-child{color:#8b97a8;width:42%;text-transform:none}
ol.prio{margin:0;padding-left:18px}
.lede{color:#aab4c2;font-size:15px;line-height:1.6}
"""


def spec_of(run: str, recorded: dict) -> tuple[object, bool]:
    """The box this run actually read, and whether it had to be guessed.

    Preferred from the checkpoint, exactly as `rsna.infer.experts.load_experts` prefers
    it, because **the registry moves**. `expert_mcl_fsonly` stores a spec named `mcl`
    cut over two series, while the `mcl` in the registry has carried three since: a page
    that resolved the name would describe a box this model has never been shown, and
    would do it without any sign that something was off.

    Falls back to the registry for the runs written before specs travelled with the
    weights, and says so, since that is a guess rather than a record.
    """

    import torch

    folds = sorted((ROOT / "out" / run).glob("fold*.pt"))
    if folds:
        checkpoint = torch.load(folds[0], map_location="cpu", weights_only=False)
        if "specs" in checkpoint:
            return RoiSpec.from_dict(checkpoint["specs"][0]), False
    return ExpertConfig.from_dict(recorded["config"]).specs[0], True


def build(key: str, out_dir: Path, study: str | None,
          examples: int = 6) -> Path:
    entry = EXPERTS[key]
    config = Config()
    points = landmark_table()
    recorded = json.loads((ROOT / "out" / entry["run"] / "history.json").read_text())
    spec, from_registry = spec_of(entry["run"], recorded)

    if study is None:
        study = entry["study"]
    if study is None:
        have = points[points.point == spec.landmark].study
        study = str(have.iloc[len(have) // 2])

    body = f"""
<style>{EXTRA_CSS}</style>
<h1>{entry['title']}</h1>
<p class="lede">{entry['subtitle']}</p>

<h2>The condition</h2>
{entry['condition']}

<h2>What the expert reads</h2>
{spec_table(spec)}
{'' if not from_registry else
 '<p class="note">This run predates regions travelling with the weights, so the box above '
 'was resolved by <b>name</b> through the registry rather than read from the checkpoint. '
 'The registry moves &mdash; <code>expert_mcl_fsonly</code> names a spec that has since '
 'gained a third series &mdash; so treat this table as a reconstruction and not a record.</p>'}

<h2>How the box is placed</h2>
{entry['placement']}
<p class="note">The machinery the fields above use &mdash; which way rows and columns
run, what the depth window is measured along, when <code>bowtie_direction</code> is
consulted &mdash; is explained once in
<a href="../placement.html">how a box is placed</a>.</p>

<h2>One study, end to end</h2>
<p class="note">Study &hellip;{P._esc(study[-12:])}. Both panels are produced by handing
this series to <code>rsna.roi.extract</code> with the spec above and the predicted
<code>{P._esc(spec.landmark)}</code>; nothing here is drawn from a description of where
the box should be.</p>
{panels(study, spec, points, config)}

<h2>What is measured about it</h2>
{measured(entry, spec)}

<h2>The same box on other studies</h2>
<p class="note">Spread across the studies carrying
<code>{P._esc(spec.landmark)}</code>, so the series the box falls back to varies rather
than repeating one choice. Each is its own <code>rsna.roi.extract</code> call: what is
green is what that study's crop kept.</p>
{more_examples(spec, points, config, examples, study)}
"""
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{key}.html"
    out.write_text(P.page(f"{entry['title']} — expert", body))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("expert", choices=sorted(EXPERTS))
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "atlas" / "experts")
    ap.add_argument("--study", default=None, help="Show this study instead of a default.")
    ap.add_argument("--examples", type=int, default=6,
                    help="How many further studies to show the box on, at the foot.")
    args = ap.parse_args()
    out = build(args.expert, args.out, args.study, args.examples)
    print(f"{out}  ({out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
