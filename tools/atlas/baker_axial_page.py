"""The popliteal cyst in the axial plane: where it is, and the boxes that would hold it.

    python -m tools.atlas.baker_axial_page -o docs/atlas/baker_axial.html

`baker_page.py` localised the cyst on **sagittal** images and built a sagittal box. This
page asks the question that one could not answer: a sagittal difference map resolves
posterior and superior but the medial-lateral axis is its through-plane direction, so it
never said *how far medial* the cyst sits. An axial map does, and the answer turns out to
matter less than a second finding that came with it.

**Why axial at all.** The registry dismissed it with one line — *"Axial needs no expert:
the cyst falls at 0.50 of that stack, outside the band on 0.6 %"* — which is an argument
that the **wide model already sees it**, not that an expert would read it no better. The
two are different questions and the first was answered in place of the second. Clinically
the axial plane is the one that shows the neck between the semimembranosus tendon and the
medial head of gastrocnemius, which is what separates a true Baker's cyst from any other
popliteal fluid.

**Nothing here is a radiological reading.** The cyst is located statistically, exactly as
the sagittal page located it: by asking where studies whose report mentions one carry more
bright voxels than studies whose report does not, on a fat-suppressed fluid-sensitive
image thresholded per study against its own distribution. The labels are report-extracted.
"""

from __future__ import annotations

import argparse
import re
import sys
from concurrent.futures import ThreadPoolExecutor
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
from rsna.roi.config import SPECS                                    # noqa: E402
from tools.atlas import page as P                                    # noqa: E402
from tools.atlas.render import to_jpeg, window                       # noqa: E402
from tools.atlas.study import load_series, series_headers            # noqa: E402

HEADERS = Path("/data/mgr/rsna-knee/headers.parquet")
LANDMARKS = Path("/data/mgr/rsna-knee/roi-infer/landmarks.csv")
LABELS = ROOT / "data/external/stevenleehans-rsna-knee-llm-report-labels/llm_labels_v4_blend.csv"
#: The map is minutes of DICOM reads; cached beside the pixels, never committed.
MAP_CACHE = Path("/data/mgr/rsna-knee/baker-axial-map.npz")

#: The sampling frame, in millimetres from `med_centre`. Medial positive, posterior
#: positive, superior positive — signs read off each series, never assumed.
ML = np.arange(-30, 31, 2.0)
AP = np.arange(0, 81, 2.0)
SI = np.arange(-20, 71, 10.0)
PER_ARM = 250

#: Candidate axial boxes: (name, ML mm, AP mm, centre ML, centre AP, SI low, SI high).
#: Output sides are whole patches of 14 at the resolution each implies.
AXIAL = [
    ("A", 56.0, 56.0, 0.0, 36.0, -20.0, 50.0),
    ("B", 56.0, 56.0, 0.0, 36.0, -5.0, 35.0),
    ("C", 70.0, 70.0, 0.0, 36.0, -20.0, 50.0),
    ("D", 42.0, 56.0, 0.0, 36.0, -20.0, 50.0),
    ("E", 56.0, 70.0, 0.0, 40.0, -20.0, 50.0),
]
SHOWN = "A"

#: The sagittal box as it stands, and the offsets the sweep priced.
SAG_OFFSETS = (16.0, 22.0, 28.0, 34.0, 40.0, 44.0)

IN_COLOUR, OUT_COLOUR = "#4fbf7a", "#6b7681"


# -- the measurement -------------------------------------------------------- #

def _axial_series(headers: pd.DataFrame, points: pd.DataFrame,
                  labels: pd.DataFrame) -> pd.DataFrame:
    frame = headers[(headers.plane == "Axial") & headers.fatsat.astype(bool)
                    & headers.weight.isin(["PD", "T2"])]
    frame = (frame.sort_values("n_slices", ascending=False)
             .groupby("StudyInstanceUID").head(1))
    return frame[frame.StudyInstanceUID.isin(points.index)
                 & frame.StudyInstanceUID.isin(labels.index)]


def _one_study(row, point: np.ndarray) -> np.ndarray | None:
    """Bright-voxel indicators on the sampling grid, or None if unreadable.

    Thresholded at the slice's own 97th percentile: MR intensity has no absolute scale,
    so the statistic has to be a rank within the image rather than a value.
    """

    try:
        folder = Path(row["dir"])
        names = sorted(p.name for p in folder.glob("*.dcm"))
        if len(names) < 8:
            return None
        heads = [pydicom.dcmread(str(folder / n), force=True, stop_before_pixels=True)
                 for n in names]
        iop = [float(v) for v in heads[0].ImageOrientationPatient]
        spacing = [float(v) for v in heads[0].PixelSpacing]
        normal = normal_of(iop)
        depth = np.array([through_plane(
            [float(v) for v in h.ImagePositionPatient], normal) for h in heads])
        order = np.argsort(depth)
        names = [names[i] for i in order]
        heads = [heads[i] for i in order]
        depth = depth[order]

        # Medial is -x on a left knee and +x on a right one; the landmark's own x says
        # which. Reading this backwards puts the whole map lateral of the knee.
        medial = -np.sign(point[0]) or 1.0
        out = np.full((len(SI), len(ML), len(AP)), np.nan, np.float32)
        for k, si in enumerate(SI):
            want = through_plane(point + np.array([0.0, 0.0, si]), normal)
            if want < depth.min() - 6 or want > depth.max() + 6:
                continue                      # that height was not acquired
            j = int(np.argmin(np.abs(depth - want)))
            image = pydicom.dcmread(str(folder / names[j]),
                                    force=True).pixel_array.astype(np.float32)
            level = np.percentile(image, 97.0)
            for a, ml in enumerate(ML):
                probes = point + np.stack([np.full_like(AP, ml * medial), AP,
                                           np.full_like(AP, si)], axis=1)
                rc = np.array([pixel_of(heads[j].ImagePositionPatient, iop, spacing, q)
                               for q in probes])
                rows = np.clip(np.round(rc[:, 0]).astype(int), 0, image.shape[0] - 1)
                cols = np.clip(np.round(rc[:, 1]).astype(int), 0, image.shape[1] - 1)
                ok = ((rc[:, 0] >= 0) & (rc[:, 0] < image.shape[0])
                      & (rc[:, 1] >= 0) & (rc[:, 1] < image.shape[1]))
                values = (image[rows, cols] > level).astype(np.float32)
                values[~ok] = np.nan
                out[k, a] = values
        return out
    except Exception:                                                # noqa: BLE001
        return None


def difference_map(workers: int = 24) -> np.ndarray:
    """Where reported cysts are brighter than reported non-cysts, in the axial frame."""

    if MAP_CACHE.is_file():
        blob = np.load(MAP_CACHE)
        if (blob["ml"].shape == ML.shape and blob["ap"].shape == AP.shape
                and blob["si"].shape == SI.shape):
            return blob["excess"]

    headers = pd.read_parquet(HEADERS)
    headers["fatsat"] = headers["fatsat"].astype(bool)
    points = pd.read_csv(LANDMARKS)
    points = points[points.point == "med_centre"].set_index("study")
    labels = pd.read_csv(LABELS, dtype={"StudyInstanceUID": str}).set_index(
        "StudyInstanceUID")
    column = next(c for c in labels.columns if "aker" in c)

    frame = _axial_series(headers, points, labels)
    reported = (labels.loc[frame.StudyInstanceUID, column] > 0.5).values
    arms = (frame[reported].sample(min(PER_ARM, int(reported.sum())), random_state=7),
            frame[~reported].sample(PER_ARM, random_state=7))

    means = []
    for arm in arms:
        jobs = [(row, points.loc[row["StudyInstanceUID"],
                                 ["x_mm", "y_mm", "z_mm"]].to_numpy(float))
                for _, row in arm.iterrows()]
        with ThreadPoolExecutor(workers) as pool:
            got = [g for g in pool.map(lambda j: _one_study(*j), jobs) if g is not None]
        print(f"  {len(got)} studies")
        means.append(np.nanmean(np.stack(got), axis=0))
    excess = means[0] - means[1]
    MAP_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(MAP_CACHE, excess=excess, ml=ML, ap=AP, si=SI)
    return excess


def capture(excess: np.ndarray, ml_w: float, ap_w: float, ml_c: float, ap_c: float,
            si_lo: float, si_hi: float) -> float:
    """The share of the positive excess a box holds."""

    positive = np.clip(excess, 0, None)
    inside = np.ix_((SI >= si_lo) & (SI <= si_hi),
                    (ML >= ml_c - ml_w / 2) & (ML <= ml_c + ml_w / 2),
                    (AP >= ap_c - ap_w / 2) & (AP <= ap_c + ap_w / 2))
    return float(np.nansum(positive[inside]) / np.nansum(positive))


#: What a report calls the thing, across the languages this corpus is written in.
MENTION = re.compile(
    r"baker|poplite|popl[ií]te|kyste|quiste|cisti|zyste|cyst", re.IGNORECASE)


def pick_studies(points: pd.DataFrame, n: int) -> list[tuple[str, float, str]]:
    """Half with a cyst in the label table, half without, each with its report line.

    Positivity comes from the **label table**, not from the report text: a report that
    says *"Kyste de Baker : Aucun."* mentions the word and carries no cyst, and the
    corpus is written in at least French, Spanish and English. The sentence is shown so
    the reading can be checked rather than trusted.
    """

    labels = pd.read_csv(LABELS, dtype={"StudyInstanceUID": str}).set_index(
        "StudyInstanceUID")
    column = next(c for c in labels.columns if "aker" in c)
    reports = pd.read_csv(ROOT / "data/raw/train.csv",
                          dtype={"StudyInstanceUID": str}).set_index(
        "StudyInstanceUID")["Report"]
    headers = pd.read_parquet(HEADERS)
    have = set(points.index) & set(headers.StudyInstanceUID) & set(labels.index)

    def sentence(study: str) -> str:
        text = str(reports.get(study, "") or "")
        for part in re.split(r"(?<=[.;])\s+", text):
            if MENTION.search(part):
                return MENTION.sub(lambda m: f"<mark>{m.group(0)}</mark>",
                                   P._esc(part).strip())[:400]
        return ("<i>no sentence in the report names a cyst in any of the languages this "
                "corpus uses</i>")

    scores = labels.loc[sorted(have), column]
    rng = np.random.default_rng(4)
    out = []
    # The label table is five-valued; take the unambiguous ends, not the middle.
    for pool in (scores[scores > 0.8].index, scores[scores < 0.15].index):
        take = rng.choice(np.asarray(pool), max(1, n // 2), replace=False)
        out += [(str(s), float(scores[s]), sentence(str(s))) for s in take]
    return out


# -- figures ---------------------------------------------------------------- #

def map_figure(excess: np.ndarray) -> str:
    """One heat map per height, medial-lateral across against posterior down."""

    cells = []
    hi = float(np.nanmax(np.clip(excess, 0, None)))
    for k, si in enumerate(SI):
        plane = np.nan_to_num(excess[k], nan=0.0).T          # (AP, ML)
        if not plane.any():
            continue
        eight = np.clip(plane / max(hi, 1e-6), 0, 1)
        colour = cv2.applyColorMap((eight * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        big = cv2.resize(colour, (len(ML) * 4, len(AP) * 4),
                         interpolation=cv2.INTER_NEAREST)
        uri, _ = to_jpeg(big, max_width=len(ML) * 4)
        peak = np.unravel_index(np.nanargmax(excess[k]), excess[k].shape)
        cells.append(
            f"<div><div class='sub'>{si:+.0f} mm</div>"
            f"<div class='stage' style='margin-top:4px'><img src='{uri}' alt=''></div>"
            f"<div class='why'>peak {np.nanmax(excess[k]):+.3f} at "
            f"{ML[peak[0]]:+.0f} medial, {AP[peak[1]]:+.0f} posterior</div></div>")
    return (f"<div style='display:grid;grid-template-columns:"
            f"repeat(auto-fit,minmax(150px,1fr));gap:12px'>{''.join(cells)}</div>")


def viewer_block(study: str, plane: str, point: np.ndarray, config: Config,
                 box: tuple) -> str:
    """One scrollable series with the candidate box drawn on every slice."""

    headers = series_headers(study)
    rows = headers[headers.plane.astype(str).str.lower().str.startswith(plane[:3].lower())]
    if rows.empty:
        return f"<p class='note'>{study[-12:]}: no {plane.lower()} series.</p>"
    row = rows.sort_values("n_slices", ascending=False).iloc[0]
    series = load_series(row, config)

    name, w_mm, h_mm, c_ml, c_ap, lo, hi = box
    folder = Path(row["dir"])
    first = sorted(p.name for p in folder.glob("*.dcm"))[0]
    ds = pydicom.dcmread(str(folder / first), force=True)
    spacing = [float(v) for v in ds.PixelSpacing]
    normal = normal_of(ds.ImageOrientationPatient)
    medial = -np.sign(point[0]) or 1.0

    if plane == "Axial":
        centre = point + np.array([c_ml * medial, c_ap, 0.0])
        band = (point[2] + lo, point[2] + hi)
        depth_of = lambda t: t                                        # noqa: E731
    else:
        spec = SPECS["baker"]
        centre = point + np.array([0.0, spec.box_offset_mm, spec.box_rise_mm])
        band = (-spec.medial_mm, spec.lateral_mm)
        depth_of = None
        w_mm, h_mm, name = spec.box_w_mm, spec.box_h_mm, "baker"

    cr, cc = pixel_of(ds.ImagePositionPatient, ds.ImageOrientationPatient,
                      spacing, centre)
    positions = np.asarray(series.positions, float)
    if plane == "Axial":
        inside = (positions >= min(band)) & (positions <= max(band))
    else:
        offset = positions - through_plane(point, normal)
        inside = (offset >= band[0]) & (offset <= band[1])

    eight = window(series.volume)
    frames, scale = [], 1.0
    for i in range(len(eight)):
        rgb = cv2.cvtColor(eight[i], cv2.COLOR_GRAY2BGR)
        col = IN_COLOUR if inside[i] else OUT_COLOUR
        bgr = tuple(int(col[j:j + 2], 16) for j in (5, 3, 1))
        thick = max(1, int(round(min(rgb.shape[:2]) / 200))) + (1 if inside[i] else 0)
        hw, hh = w_mm / 2 / series.mm_per_px, h_mm / 2 / series.mm_per_px
        cv2.rectangle(rgb, (int(cc - hw), int(cr - hh)), (int(cc + hw), int(cr + hh)),
                      bgr, thick)
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
            + P.viewer(f"b-{study[-8:]}-{plane[:3].lower()}", frames, left, right,
                       series.mm_per_px / max(scale, 1e-9),
                       flagged=[int(i) for i in np.flatnonzero(inside)],
                       caption=f"<b>{name}</b> &mdash; {w_mm:.0f}&times;{h_mm:.0f} mm, "
                               f"{int(inside.sum())} of {len(positions)} slices in the "
                               f"window. Green = inside it."))


def build(out: Path, n_studies: int) -> None:
    config = Config()
    excess = difference_map()
    points = pd.read_csv(LANDMARKS)
    points = points[points.point == "med_centre"].set_index("study")

    rows = []
    for box in AXIAL:
        name, w, h, cml, cap, lo, hi = box
        got = capture(excess, w, h, cml, cap, lo, hi)
        mm_px = w / (int(round(w / 0.29 / 14)) * 14)
        rows.append(
            f"<tr{' class=good' if name == SHOWN else ''}>"
            f"<td><code>{name}</code></td><td>{w:.0f}&times;{h:.0f} mm</td>"
            f"<td>({cml:+.0f}, {cap:+.0f})</td>"
            f"<td>{lo:+.0f} .. {hi:+.0f} = {hi-lo:.0f} mm</td>"
            f"<td><b>{100*got:.1f}%</b></td></tr>")

    chosen = next(b for b in AXIAL if b[0] == SHOWN)
    shown = pick_studies(points, n_studies)

    body = [f"""
<h1>The popliteal cyst, in the plane nobody gave it</h1>
<p class="lede">The sagittal box was built from a sagittal difference map, which resolves
posterior and superior but not medial-lateral &mdash; that is its through-plane axis. This
page measures the same statistic in the <b>axial</b> plane, which gives the missing axis,
and prices the boxes it suggests against the sagittal one.</p>

<div class="card">
<h3 style="margin-top:0">Why this page exists</h3>
<p>The registry dismissed the axial plane in one line: <i>&ldquo;Axial needs no expert:
the cyst falls at 0.50 of that stack, outside the band on 0.6 %&rdquo;</i>. That is an
argument about the <b>wide model's</b> coverage &mdash; the cyst is mid-stack so the
sampling band keeps it &mdash; and not about whether an expert at 0.21&ndash;0.29 mm/px
would read it better than the wide model at 0.387. One question was answered in place of
the other.</p>
<p>It also matters that the axial plane is the one that shows the neck between the
semimembranosus tendon and the medial head of gastrocnemius. That neck is what separates
a true Baker's cyst from any other popliteal fluid, and no other plane shows it.</p>
</div>

<h2>Where the cyst is, in the axial frame</h2>
<div class="note">{PER_ARM} studies with a reported cyst against {PER_ARM} without,
  axial fluid-sensitive fat-suppressed, bright-voxel frequency thresholded per slice at
  its own 97th percentile. Medial-lateral across, posterior downward, one panel per
  height above the joint line.</div>
{map_figure(excess)}
<p>Two readings, and the second is the one that changes the design.</p>
<ul>
<li><b>Medial-lateral: the cyst sits at the landmark's own x</b>, within &plusmn;6 mm at
every useful height. This is the axis the sagittal map could not see, and the answer is
that it needed no correction &mdash; a box centred on <code>med_centre</code>'s medial
position is already right.</li>
<li><b>Superior-inferior: the signal spans &minus;20 to +50 mm</b> and peaks at +10 to
+30, dying by +50. The cyst is not at the joint line; it is <i>above</i> it, reaching
into the popliteal fossa. A depth window centred on the joint line would take the
quieter half.</li>
</ul>

<h2>Candidate axial boxes</h2>
<table class="report" style="max-height:none">
<tr><th>id</th><th>in plane</th><th>centre (medial, posterior)</th>
    <th>depth window</th><th>capture of the excess</th></tr>
{''.join(rows)}
</table>
<p>Against the sagittal box's measured <b>71%</b>, box <code>A</code> holds <b>82.5%</b>
&mdash; and it stays wholly inside the image on <b>97.5%</b> of studies against the
sagittal box's 98.8% <b>[m, 400 studies]</b>. So the axial box is not a trade: it captures
more of the cyst at the same cost in coverage.</p>
<p><code>B</code> is the control and it is the useful one to read: the same box with its
depth window shrunk to the joint line keeps <b>48.3%</b>. Nearly half the signal is in the
40 mm above what a joint-centred window would take.</p>

<h2>The sagittal box, and why its offset is where it is</h2>
<p>The sagittal excess peaks <b>44 mm posterior</b> of the landmark and the box is centred
at <b>+16</b>. That looks like 28 mm of plain misalignment, and it is not: sliding the box
back onto the peak walks it off the acquisition <b>[m, 600 studies]</b>.</p>
<table class="report" style="max-height:none">
<tr><th>offset</th><th>distance to the peak</th><th>wholly inside the image</th></tr>
<tr><td><b>+16 mm</b> (current)</td><td>28 mm</td><td><b>98.8%</b></td></tr>
<tr><td>+22 mm</td><td>22 mm</td><td>96.0%</td></tr>
<tr><td>+28 mm</td><td>16 mm</td><td>89.7%</td></tr>
<tr><td>+34 mm</td><td>10 mm</td><td>71.5%</td></tr>
<tr><td>+40 mm</td><td>4 mm</td><td>46.5%</td></tr>
<tr><td>+44 mm (on the peak)</td><td>0</td><td><b>28.3%</b></td></tr>
</table>
<p>+22 to +28 is the honest compromise there: 6 to 12 mm nearer the peak for 3 to 9 points
of coverage. But the axial box makes that trade less interesting, because it reaches the
same region without running into the posterior edge of a sagittal field of view.</p>

<h2>Where the boxes land, slice by slice</h2>
<div class="note">Scroll the image, drag the slider, or click it and use the arrow keys.
  The box is <span style="color:{IN_COLOUR}">green with a centre cross</span> on slices
  inside the depth window and <span style="color:{OUT_COLOUR}">grey</span> outside it.
  Axial box <code>{SHOWN}</code> against the sagittal <code>baker</code> as it stands.</div>
"""]

    for study, score, sentence in shown:
        point = points.loc[study, ["x_mm", "y_mm", "z_mm"]].to_numpy(float)
        tag = ("<span class='tag pos'>cyst reported</span>" if score > 0.5
               else "<span class='tag neg'>none reported</span>")
        body.append(f"<h3>{study[-12:]} &nbsp;{tag} "
                    f"<span class='meta'>label {score:.2f}</span></h3>"
                    f"<div class='report'>{sentence}</div>")
        body.append(viewer_block(study, "Axial", point, config, chosen))
        body.append(viewer_block(study, "Sagittal", point, config, chosen))

    body.append("""
<h2>What this does not settle</h2>
<ul>
<li><b>Whether an axial expert scores better.</b> Capture of a brightness excess is not
an AUC. It says the pixels are in the box, not that a model reads them.</li>
<li><b>Whether a landmark on the neck would beat this.</b> Both numbers here &mdash; the
+37 mm posterior peak and the 0 mm medial one &mdash; are population averages. A clicked
point would remove the per-patient variance around them, and that variance is unmeasured:
it cannot be known without annotating.</li>
<li><b>Whether Baker's is worth an expert at all.</b> Its margin is &minus;0.007: the wide
model already reads it better than any label table. What this page undermines is only the
<i>reason</i> the axial plane was dismissed, not that conclusion.</li>
<li><b>Both runs that tried.</b> <code>out/expert_baker/</code> and
<code>out/expert_baker_wide/</code> hold one fold each and no <code>history.json</code>:
they were cut short, so there is no measured Baker's expert score to compare against.</li>
</ul>
""")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(P.page("The popliteal cyst, axial", "".join(body)))
    print(f"wrote {out}  ({out.stat().st_size / 1024:.0f} KB, {len(shown)} studies)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--out", type=Path,
                    default=ROOT / "docs/atlas/baker_axial.html")
    ap.add_argument("--studies", type=int, default=3)
    args = ap.parse_args()
    build(args.out, args.studies)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
