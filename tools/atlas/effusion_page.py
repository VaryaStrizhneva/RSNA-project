"""The joint effusion: where a radiologist looks, and whether a box can be put there.

    python -m tools.atlas.effusion_page -o docs/atlas/effusion.html

`pipeline_v2.md` filed effusion under *whole joint, fluid-sensitive — already today's
input*, together with synovitis, the popliteal cyst, contusion and fracture. That ruling
has since been measured wrong for one of the five: a 56 mm axial box beat the wide model
on Baker's in three epochs. The reasoning that produced it confused *the wide model reads
this well* with *a box could not read it better*, so the other four are worth re-asking
rather than inheriting.

**Clinically there is no doubt where to look.** A knee effusion collects first and most in
the **suprapatellar recess**, the largest synovial space, above the patella between the
quadriceps tendon and the femur. The standard grading is the antero-posterior depth of
fluid in that recess. The posterior recesses and the gutters fill later. So the anchor is
the patella and the direction is up — which is why `pf_centre` is the landmark here and
`med_centre` is not.

**The plane follows from an asymmetry that clinical reasoning alone does not give.** For a
structure *above* the patella, what matters is whether "up" is an axis of the image or of
the stack. Measured over 250 series per plane: in **axial** up is through-plane and the
acquisition reaches a median **13 mm** above `pf_centre`, so the recess is largely not
acquired. In **sagittal** up is in-plane, limited only by the field of view, with a median
**70 mm** of headroom and +42 mm reachable on **100 %** of studies. Baker's was the mirror
image — posteromedial, so its useful axis is in-plane in axial and through-plane in
sagittal, which is why that box is axial and this one is sagittal.

**Nothing here is a radiological reading.** The effusion is located statistically, the way
the cyst was: where studies whose report carries the finding have more bright voxels than
studies that do not, on fat-suppressed fluid-sensitive images thresholded per slice at
their own 97th percentile. The labels are report-extracted.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pydicom

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from rsna.config import Config                                       # noqa: E402
from rsna.dicom.geometry import normal_of, pixel_of, through_plane   # noqa: E402
from tools.atlas import page as P                                    # noqa: E402
from tools.atlas.render import to_jpeg, window                       # noqa: E402
from tools.atlas.study import load_series, series_headers            # noqa: E402

HEADERS = Path("/data/mgr/rsna-knee/headers.parquet")
LANDMARKS = Path("/data/mgr/rsna-knee/roi-infer/landmarks.csv")
LABELS = ROOT / "data/external/stevenleehans-rsna-knee-llm-report-labels/llm_labels_v4_blend.csv"
MAP_CACHE = Path("/data/mgr/rsna-knee/effusion-map.npz")

PATCH = 14
IN_COLOUR, OUT_COLOUR = "#4fbf7a", "#6b7681"

#: Sagittal candidates: (id, AP mm, SI mm, rise, depth each way, mm/px).
#: `rise` is superior from `pf_centre`; the depth window is medial-lateral and symmetric,
#: so `symmetric_depth` stays true and `bowtie_direction` is never consulted.
SAGITTAL = [
    ("S1", 70.0, 70.0, 30.0, 18.0, 0.286),
    ("S2", 70.0, 84.0, 32.0, 18.0, 0.286),
    ("S3", 84.0, 84.0, 28.0, 24.0, 0.357),
    ("S4", 56.0, 70.0, 35.0, 15.0, 0.250),
]
#: Axial candidates: (id, ML mm, AP mm, rise anterior, depth below, depth above, mm/px).
#: Capped by the 13 mm median headroom above `pf_centre`, which is the whole point.
AXIAL = [
    ("A1", 84.0, 56.0, 0.0, 18.0, 24.0, 0.286),
    ("A2", 70.0, 56.0, 0.0, 18.0, 24.0, 0.286),
]
SHOW_SAG, SHOW_AX = "S3", "A1"

MENTION = re.compile(r"effusion|epanchement|épanchement|derrame|derrame|hydrarthros|"
                     r"liquid|fluid|versement", re.IGNORECASE)


def _maps():
    if not MAP_CACHE.is_file():
        raise SystemExit(
            f"{MAP_CACHE} missing. It is 40 minutes of DICOM reads over 800 studies; "
            f"the script that writes it is in this file's git history.")
    return np.load(MAP_CACHE)


def _out(mm: float, mm_px: float) -> int:
    return int(round(mm / mm_px / PATCH)) * PATCH


def capture_sagittal(blob, ap_w, si_h, rise, depth) -> tuple[float, float]:
    """Share of the positive excess a sagittal box holds, and its coverage there."""

    ex, cv, AP, SI, ML = (blob["ex_s"], blob["cv_s"], blob["AP"], blob["SI"],
                          blob["MLd"])
    good = cv >= 0.7
    pos = np.clip(np.where(good, np.nan_to_num(ex), 0.0), 0, None)
    inside = np.ix_(np.abs(AP) <= ap_w / 2,
                    (SI >= rise - si_h / 2) & (SI <= rise + si_h / 2),
                    np.abs(ML) <= depth)
    return float(pos[inside].sum() / pos.sum()), float(np.nanmean(cv[inside]))


def capture_axial(blob, ml_w, ap_h, rise, below, above) -> tuple[float, float]:
    ex, cv, ML, AP, SI = (blob["ex_a"], blob["cv_a"], blob["MLa"], blob["APa"],
                          blob["SId"])
    good = cv >= 0.7
    pos = np.clip(np.where(good, np.nan_to_num(ex), 0.0), 0, None)
    inside = np.ix_(np.abs(ML) <= ml_w / 2,
                    (AP >= -rise - ap_h / 2) & (AP <= -rise + ap_h / 2),
                    (SI >= -below) & (SI <= above))
    return float(pos[inside].sum() / pos.sum()), float(np.nanmean(cv[inside]))


# -- figures ---------------------------------------------------------------- #

def heat(plane_excess, coverage, rows, cols, row_label, col_label,
         slabs, slab_label) -> str:
    """One heat map per through-plane offset, with the axes named."""

    hi = float(np.nanmax(np.clip(plane_excess, 0, None)))
    cells = []
    for k, slab in enumerate(slabs):
        layer = np.nan_to_num(plane_excess[:, :, k], nan=0.0)
        layer = np.where(coverage[:, :, k] >= 0.7, layer, 0.0).T
        if not layer.any():
            continue
        scaled = np.clip(layer / max(hi, 1e-6), 0, 1)
        colour = cv2.applyColorMap((scaled * 255).astype(np.uint8),
                                   cv2.COLORMAP_INFERNO)
        big = cv2.resize(colour, (len(rows) * 4, len(cols) * 4),
                         interpolation=cv2.INTER_NEAREST)
        uri, _ = to_jpeg(big, max_width=len(rows) * 4)
        peak = np.unravel_index(np.argmax(layer), layer.shape)
        cells.append(
            f"<div><div class='sub'>{slab_label} {slab:+.0f} mm</div>"
            f"<div class='stage' style='margin-top:4px'><img src='{uri}' alt=''></div>"
            f"<div class='why'>peak {layer.max():+.3f} at {row_label}"
            f"{rows[peak[1]]:+.0f}, {col_label}{cols[peak[0]]:+.0f}</div></div>")
    return (f"<div style='display:grid;grid-template-columns:"
            f"repeat(auto-fit,minmax(150px,1fr));gap:12px'>{''.join(cells)}</div>")


def viewer(study: str, plane: str, point: np.ndarray, config: Config, box) -> str:
    headers = series_headers(study)
    rows = headers[headers.plane.astype(str).str.lower()
                   .str.startswith(plane[:3].lower())]
    if rows.empty:
        return f"<p class='note'>{study[-12:]}: no {plane.lower()} series.</p>"
    row = rows.sort_values("n_slices", ascending=False).iloc[0]
    series = load_series(row, config)

    folder = Path(row["dir"])
    first = sorted(p.name for p in folder.glob("*.dcm"))[0]
    ds = pydicom.dcmread(str(folder / first), force=True)
    spacing = [float(v) for v in ds.PixelSpacing]
    normal = normal_of(ds.ImageOrientationPatient)
    medial = -np.sign(point[0]) or 1.0

    if plane == "Sagittal":
        name, w_mm, h_mm, rise, depth, mm_px = box
        centre = point + np.array([0.0, 0.0, rise])        # superior
        band = (-depth, depth)
    else:
        name, w_mm, h_mm, rise, below, above, mm_px = box
        centre = point + np.array([0.0, -rise, 0.0])       # anterior
        band = (-below, above)

    cr, cc = pixel_of(ds.ImagePositionPatient, ds.ImageOrientationPatient,
                      spacing, centre)
    positions = np.asarray(series.positions, float)
    offset = (positions - through_plane(point, normal)) * (
        medial if plane == "Sagittal" else float(np.sign(normal[2])) or 1.0)
    inside = (offset >= band[0]) & (offset <= band[1])

    eight = window(series.volume)
    frames, scale = [], 1.0
    for i in range(len(eight)):
        rgb = cv2.cvtColor(eight[i], cv2.COLOR_GRAY2BGR)
        col = IN_COLOUR if inside[i] else OUT_COLOUR
        bgr = tuple(int(col[j:j + 2], 16) for j in (5, 3, 1))
        thick = max(1, int(round(min(rgb.shape[:2]) / 200))) + (1 if inside[i] else 0)
        hw, hh = w_mm / 2 / series.mm_per_px, h_mm / 2 / series.mm_per_px
        cv2.rectangle(rgb, (int(cc - hw), int(cr - hh)),
                      (int(cc + hw), int(cr + hh)), bgr, thick)
        if inside[i]:
            cv2.drawMarker(rgb, (int(cc), int(cr)), bgr, cv2.MARKER_CROSS,
                           max(6, int(min(rgb.shape[:2]) / 28)), thick)
        uri, scale = to_jpeg(rgb, max_width=430)
        frames.append(uri)

    left, right = (("posterior", "anterior") if plane == "Sagittal"
                   else ("medial", "lateral"))
    return (f"<h4>{plane} &mdash; {row.get('SeriesDescription') or '?'} "
            f"<span class='meta'>{len(positions)} slices, "
            f"{series.mm_per_px:.3f} mm/px</span></h4>"
            + P.viewer(f"e-{study[-8:]}-{plane[:3].lower()}", frames, left, right,
                       series.mm_per_px / max(scale, 1e-9),
                       flagged=[int(i) for i in np.flatnonzero(inside)],
                       caption=f"<b>{name}</b> &mdash; {w_mm:.0f}&times;{h_mm:.0f} mm at "
                               f"{mm_px:.3f} mm/px &rarr; {_out(w_mm, mm_px)}&times;"
                               f"{_out(h_mm, mm_px)} px. {int(inside.sum())} of "
                               f"{len(positions)} slices in the window; green = inside."))


def pick(points: pd.DataFrame, n: int) -> list[tuple[str, float, str]]:
    labels = pd.read_csv(LABELS, dtype={"StudyInstanceUID": str}).set_index(
        "StudyInstanceUID")
    reports = pd.read_csv(ROOT / "data/raw/train.csv",
                          dtype={"StudyInstanceUID": str}).set_index(
        "StudyInstanceUID")["Report"]
    headers = pd.read_parquet(HEADERS)
    have = sorted(set(points.index) & set(headers.StudyInstanceUID) & set(labels.index))
    scores = labels.loc[have, "Effusion"]

    def sentence(study: str) -> str:
        for part in re.split(r"(?<=[.;])\s+", str(reports.get(study, "") or "")):
            if MENTION.search(part):
                return MENTION.sub(lambda m: f"<mark>{m.group(0)}</mark>",
                                   P._esc(part).strip())[:400]
        return "<i>no sentence names fluid in any language this corpus uses</i>"

    rng = np.random.default_rng(6)
    out = []
    for pool in (scores[scores > 0.8].index, scores[scores < 0.15].index):
        for s in rng.choice(np.asarray(pool), max(1, n // 2), replace=False):
            out.append((str(s), float(scores[s]), sentence(str(s))))
    return out


def build(out: Path, n_studies: int) -> None:
    config = Config()
    blob = _maps()
    points = pd.read_csv(LANDMARKS)
    points = points[points.point == "pf_centre"].set_index("study")

    sag_rows = []
    for box in SAGITTAL:
        name, w, h, rise, depth, mm_px = box
        got, cov = capture_sagittal(blob, w, h, rise, depth)
        sag_rows.append(
            f"<tr{' class=good' if name == SHOW_SAG else ''}>"
            f"<td><code>{name}</code></td><td>{w:.0f}&times;{h:.0f} mm</td>"
            f"<td>{mm_px:.3f} &rarr; {_out(w, mm_px)}&times;{_out(h, mm_px)}</td>"
            f"<td>+{rise:.0f} mm</td><td>&plusmn;{depth:.0f} mm</td>"
            f"<td><b>{100*got:.1f}%</b></td><td>{100*cov:.0f}%</td></tr>")
    ax_rows = []
    for box in AXIAL:
        name, w, h, rise, below, above, mm_px = box
        got, cov = capture_axial(blob, w, h, rise, below, above)
        ax_rows.append(
            f"<tr{' class=good' if name == SHOW_AX else ''}>"
            f"<td><code>{name}</code></td><td>{w:.0f}&times;{h:.0f} mm</td>"
            f"<td>{mm_px:.3f} &rarr; {_out(w, mm_px)}&times;{_out(h, mm_px)}</td>"
            f"<td>&minus;{below:.0f} .. +{above:.0f} mm</td>"
            f"<td><b>{100*got:.1f}%</b></td><td>{100*cov:.0f}%</td></tr>")

    shown = pick(points, n_studies)
    sag_box = next(b for b in SAGITTAL if b[0] == SHOW_SAG)
    ax_box = next(b for b in AXIAL if b[0] == SHOW_AX)

    body = [f"""
<h1>The joint effusion, and the plane the anatomy chooses</h1>
<p class="lede">Where a radiologist looks for an effusion is not in doubt: the
<b>suprapatellar recess</b>, above the patella, which fills first and most and whose
antero-posterior fluid depth is the standard grading. What is less obvious is which plane
can put a box there, and the answer is the opposite of Baker's.</p>

<div class="card">
<h3 style="margin-top:0">Why the plane is sagittal, and Baker's was axial</h3>
<p>For a structure <i>above</i> the patella what matters is whether &ldquo;up&rdquo; is an
axis of the <b>image</b> or of the <b>stack</b> <b>[m, 250 series per plane]</b>:</p>
<table class="report" style="max-height:none">
<tr><th>plane</th><th>&ldquo;up&rdquo; is</th><th>median headroom above
    <code>pf_centre</code></th><th>+42 mm reachable</th></tr>
<tr><td><b>Axial</b></td><td>through-plane</td><td><b>13 mm</b></td><td>9.6%</td></tr>
<tr class="good"><td><b>Sagittal</b></td><td><b>in-plane</b></td><td><b>70 mm</b></td>
    <td><b>100%</b></td></tr>
<tr><td>Coronal</td><td>in-plane</td><td>66 mm</td><td>98.4%</td></tr>
</table>
<p>In axial, climbing costs <b>slices</b>, and knee axial series stop a median 13 mm above
the patella &mdash; the recess is largely not acquired. In sagittal, climbing is
<b>free</b>: it is an axis of the picture, bounded only by the field of view.</p>
<p>Baker's is the mirror image. The cyst is <b>posteromedial</b>, so its useful axis is
medial-lateral &mdash; in-plane on an axial image, through-plane on a sagittal one. Which
is why that box is axial and this one is sagittal, and why neither choice is a matter of
taste.</p>
</div>

<h2>Where the fluid is, measured</h2>
<div class="note">200 studies with a reported effusion against 200 without, per plane.
  Bright-voxel frequency thresholded per slice at its own 97th percentile, differenced,
  and <b>masked where fewer than 70 % of studies were acquired</b> &mdash; the guard that
  a first attempt at this map omitted, which made it read the top slice of the stack for
  every height above it and invent a peak at +42 mm on 12 % of the data.</div>

<h3>Sagittal &mdash; antero-posterior across, superior-inferior down</h3>
{heat(blob["ex_s"], blob["cv_s"], blob["AP"], blob["SI"], "AP", "SI", blob["MLd"],
      "medial-lateral")}
<p>The peak is <b>+0.339 at 42 mm above <code>pf_centre</code></b>, at the patella's own
antero-posterior position, on <b>99 %</b> coverage. That is the suprapatellar recess, and
the clinical reasoning is confirmed rather than assumed.</p>

<h3>Axial &mdash; medial-lateral across, antero-posterior down</h3>
{heat(blob["ex_a"], blob["cv_a"], blob["MLa"], blob["APa"], "ML", "AP", blob["SId"],
      "superior-inferior")}
<p>The axial peak is weaker at <b>+0.273</b>, 21 mm lateral and 24 mm up &mdash; the
lateral part of the recess, as high as the acquisition allows.</p>

<h2>Candidate boxes</h2>
<h3>Sagittal, anchored on <code>pf_centre</code> and raised</h3>
<table class="report" style="max-height:none">
<tr><th>id</th><th>box</th><th>mm/px &rarr; output</th><th>rise</th>
    <th>depth</th><th>capture</th><th>coverage</th></tr>
{''.join(sag_rows)}
</table>
<p>The depth window is symmetric, so <code>symmetric_depth</code> stays true and
<code>bowtie_direction</code> is never consulted &mdash; a rule measured on the
medial-lateral axis, which here <i>is</i> the depth axis, but whose &ldquo;nearer
end&rdquo; heuristic was calibrated for a meniscus and has no meaning for a recess.</p>

<h3>Axial, for comparison</h3>
<table class="report" style="max-height:none">
<tr><th>id</th><th>box</th><th>mm/px &rarr; output</th><th>depth window</th>
    <th>capture</th><th>coverage</th></tr>
{''.join(ax_rows)}
</table>

<h2>What the capture curve says, and it is not what Baker's said</h2>
<p>Capture climbs with <b>size</b>, steeply, and that is the finding:</p>
<table class="report" style="max-height:none">
<tr><th>box</th><th>depth</th><th>capture</th></tr>
<tr><td><code>S4</code> 56&times;70 mm</td><td>&plusmn;15</td><td>43.6%</td></tr>
<tr><td><code>S1</code> 70&times;70 mm</td><td>&plusmn;18</td><td>65.2%</td></tr>
<tr><td><code>S2</code> 70&times;84 mm</td><td>&plusmn;18</td><td>66.9%</td></tr>
<tr class="good"><td><code>S3</code> 84&times;84 mm</td><td>&plusmn;24</td><td><b>94.4%</b></td></tr>
</table>
<p>A tight box holds <b>44 %</b>. The one that holds 94 % is 84&times;84 mm at
0.357 mm/px &mdash; which is <code>knee_sagittal</code> (100&times;100 at 0.357) in all but
<b>28 mm of rise</b>. So for effusion the capture only becomes good once the box stops
being a region of interest.</p>
<p>Baker's went the other way: a <b>56&times;56 mm</b> box held 82.5 %, and tightening was
the point. A cyst sits; a recess fills. The measurement says this is a difference in the
findings, not in how hard we looked.</p>
<p><b>So the actionable conclusion is not a new tight box.</b> It is that the whole-knee
sagittal branch is <b>centred on the wrong thing for the five diffuse targets</b> — it
hangs off the joint line, and the fluid is 42 mm above it. Raising it, or adding a raised
variant, is a smaller change than a new expert and is the one this page supports.</p>

<h2>Where the boxes land, slice by slice</h2>
<div class="note">Scroll the image, drag the slider, or click it and use the arrow keys.
  Green with a centre cross means the slice is inside the depth window. Boxes shown:
  <code>{SHOW_SAG}</code> sagittal and <code>{SHOW_AX}</code> axial.</div>
"""]

    for study, score, line in shown:
        point = points.loc[study, ["x_mm", "y_mm", "z_mm"]].to_numpy(float)
        tag = ("<span class='tag pos'>effusion reported</span>" if score > 0.5
               else "<span class='tag neg'>none reported</span>")
        body.append(f"<h3>{study[-12:]} &nbsp;{tag} "
                    f"<span class='meta'>label {score:.2f}</span></h3>"
                    f"<div class='report'>{line}</div>")
        body.append(viewer(study, "Sagittal", point, config, sag_box))
        body.append(viewer(study, "Axial", point, config, ax_box))

    body.append("""
<h2>What this does not settle</h2>
<ul>
<li><b>Whether a box beats the wide model here.</b> Baker's capture was 82.5 % and its
expert beat the wide model; effusion's best sagittal box captures less, and capture is not
an AUC. The comparison is 0.8678, which is what the wide model reads on Effusion at
fold 0.</li>
<li><b>Whether the effusion is localised enough at all.</b> Unlike the cyst, the three
planes disagreed on where the excess peaks in the medial-lateral axis &mdash; +12 sagittal,
&minus;12 axial, &minus;24 coronal on a coarse first pass &mdash; which is the signature of
a <i>distributed</i> collection rather than a point. A recess that fills is not a cyst that
sits.</li>
<li><b>The series list.</b> Sagittal PD-FS, sagittal T2-FS and axial PD-FS cover
<b>93.74 %</b> of studies between them <b>[m]</b>; 276 carry none of the three. Adding
sagittal PD without fat suppression would take it near 99.8 %, at the cost of a sequence
on which fluid is far less conspicuous.</li>
<li><b>The anchor.</b> <code>pf_centre</code> is the patellofemoral joint, not the top of
the patella, and the recess is defined relative to the patella's superior pole. Every
number here is a population average around a landmark that is near but not at the right
place; a clicked point would remove that variance, and the variance is unmeasured.</li>
</ul>
""")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(P.page("The joint effusion", "".join(body)))
    print(f"wrote {out}  ({out.stat().st_size / 1024:.0f} KB, {len(shown)} studies)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--out", type=Path, default=ROOT / "docs/atlas/effusion.html")
    ap.add_argument("--studies", type=int, default=4)
    args = ap.parse_args()
    build(args.out, args.studies)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
