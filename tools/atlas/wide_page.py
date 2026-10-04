"""The whole-knee boxes: where their centre comes from, and how deep they should go.

    python -m tools.atlas.wide_page -o docs/atlas/wide.html

Pipeline v3 wants two branches nothing in the registry currently provides: a box that
sees the **whole knee**, sagittal and coronal, cut around the predicted joint centre
rather than the image centre. This page is the decision record for their geometry.

The finding that makes them necessary: **no existing box sees the whole knee in depth.**
A spec's depth window is `lateral_mm + medial_mm` around its landmark, and the widest one
on the books — `baker_wide`, 110x95 mm in plane — is a **36 mm slab**. A knee spans
99 mm along a sagittal stack (measured median). Every box we have is a third of a knee or
less, which is correct for a compartment and wrong for an effusion.

Two things are measured here and both constrain the design. **Where the centre sits inside
each stack**, which decides whether the depth window can be symmetric: sagittal is
49 mm/49 mm, coronal is 60 mm/41 mm, so the coronal one cannot be. And **how far the
acquisition reaches**, which caps what any window can ask for.

Nothing here is a radiological reading. The landmark is a model output with its own error;
the boxes are arithmetic on it.
"""

from __future__ import annotations

import argparse
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

LANDMARKS = Path("/data/mgr/rsna-knee/roi-infer/landmarks.csv")
DICOM_ROOT = Path("/data/mgr/rsna-knee/extracted")

#: The annotated header table, cached beside the pixels because walking 24371 series
#: takes minutes and three pages now want it. Built on first use, never committed.
HEADERS = Path("/data/mgr/rsna-knee/headers.parquet")

#: The joint centre is the midpoint of these two, both from the `landmark_both` run.
CENTRE_POINTS = ("lat_centre", "med_centre")

#: `patch` from the ROI registry: every output side must be a whole number of patches.
PATCH = 14

#: Candidate geometries. `depth` is (toward one end, toward the other) in mm from the
#: landmark along the stack axis; `slots` is the cap on acquired slices kept inside it.
CANDIDATES = {
    "Sagittal": [
        ("S-a", 100.0, 0.357, (45.0, 45.0), 15),
        ("S-b", 100.0, 0.357, (45.0, 45.0), 20),
        ("S-c", 100.0, 0.286, (45.0, 45.0), 20),
        ("S-d", 110.0, 0.357, (45.0, 45.0), 28),
    ],
    "Coronal": [
        ("C-a", 100.0, 0.357, (55.0, 40.0), 15),
        ("C-b", 100.0, 0.357, (55.0, 40.0), 20),
        ("C-c", 100.0, 0.286, (55.0, 40.0), 20),
        ("C-d", 110.0, 0.357, (55.0, 40.0), 28),
    ],
}

#: Drawn on the viewers: the one each plane's section illustrates.
SHOWN = {"Sagittal": "S-b", "Coronal": "C-b"}

IN_COLOUR = "#4fbf7a"      # the slice is inside the depth window
OUT_COLOUR = "#6b7681"     # it is not — the box is where it would be, dimmed


# -- geometry --------------------------------------------------------------- #

def knee_centres() -> pd.DataFrame:
    """The joint centre per study: the midpoint of the two meniscus centres.

    A **3-D point in patient millimetres**, which is the whole reason one landmark serves
    every plane: `pixel_of` projects it into any slice and `through_plane` gives its depth
    along any stack axis. Neither asks which plane the annotation was made on.
    """

    lm = pd.read_csv(LANDMARKS)
    both = lm[lm.point.isin(CENTRE_POINTS)]
    wide = both.pivot_table(index="study", columns="point",
                            values=["x_mm", "y_mm", "z_mm", "confidence", "spread_mm"])
    wide.columns = [f"{a}:{b}" for a, b in wide.columns]
    centre = both.groupby("study")[["x_mm", "y_mm", "z_mm"]].mean()
    centre.columns = ["cx", "cy", "cz"]
    return centre.join(wide)


def _reach_one(row, point: np.ndarray) -> dict | None:
    """How far one series reaches either side of the landmark, in millimetres."""

    try:
        folder = Path(row["dir"])
        positions, normal = [], None
        for name in sorted(p.name for p in folder.glob("*.dcm")):
            ds = pydicom.dcmread(str(folder / name), stop_before_pixels=True)
            if normal is None:
                normal = normal_of(ds.ImageOrientationPatient)
            positions.append(through_plane(
                np.array([float(v) for v in ds.ImagePositionPatient]), normal))
        positions = np.sort(np.asarray(positions))
        depth = through_plane(point, normal)
        return {"plane": row["plane"], "span": float(np.ptp(positions)),
                "n": len(positions),
                "step": float(np.ptp(positions)) / max(len(positions) - 1, 1),
                "low": float(depth - positions.min()),
                "high": float(positions.max() - depth)}
    except Exception:                                                # noqa: BLE001
        return None


def reach(headers: pd.DataFrame, centres: pd.DataFrame, per_plane: int = 120,
          workers: int = 32) -> pd.DataFrame:
    """The acquisition's extent, per plane, and where the landmark sits inside it."""

    have = headers[headers.StudyInstanceUID.isin(centres.index)]
    picked = []
    for _, group in have.groupby("plane"):
        picked.append(group.sample(min(len(group), per_plane), random_state=11))
    sub = pd.concat(picked)
    jobs = [(row, centres.loc[row["StudyInstanceUID"], ["cx", "cy", "cz"]]
             .to_numpy(float)) for _, row in sub.iterrows()]
    with ThreadPoolExecutor(workers) as pool:
        got = pool.map(lambda j: _reach_one(*j), jobs)
        rows = [r for r in got if r is not None]
    return pd.DataFrame(rows)


# -- drawing ---------------------------------------------------------------- #

def _out_px(extent_mm: float, mm_px: float) -> int:
    """The output side, in whole patches, for an extent at a given resolution."""

    return int(round(extent_mm / mm_px / PATCH)) * PATCH


def _draw(slice8: np.ndarray, centre_rc, box_px: float, inside: bool) -> np.ndarray:
    """The slice with the box on it, coloured by whether this slice is in the window."""

    rgb = cv2.cvtColor(slice8, cv2.COLOR_GRAY2BGR)
    colour = IN_COLOUR if inside else OUT_COLOUR
    bgr = tuple(int(colour[i:i + 2], 16) for i in (5, 3, 1))
    thick = max(1, int(round(min(rgb.shape[:2]) / 200))) + (1 if inside else 0)
    cr, cc = centre_rc
    half = box_px / 2
    r0, c0 = int(round(cr - half)), int(round(cc - half))
    cv2.rectangle(rgb, (c0, r0), (int(round(c0 + box_px)), int(round(r0 + box_px))),
                  bgr, thick)
    if inside:                           # a tick at the centre, so the point is visible
        cv2.drawMarker(rgb, (int(round(cc)), int(round(cr))), bgr,
                       cv2.MARKER_CROSS, max(6, int(min(rgb.shape[:2]) / 28)), thick)
    return rgb


def plane_block(study: str, plane: str, centre: pd.Series, config: Config,
                tag: str) -> str:
    """One scrollable series with the candidate box drawn on every slice."""

    headers = series_headers(study)
    rows = headers[headers.plane.astype(str).str.lower()
                   .str.startswith(plane[:3].lower())]
    if rows.empty:
        return f"<p class='note'>{study[-12:]}: no {plane.lower()} series.</p>"
    row = rows.sort_values("n_slices", ascending=False).iloc[0]
    series = load_series(row, config)

    name, extent_mm, mm_px, (low_mm, high_mm), slots = next(
        c for c in CANDIDATES[plane] if c[0] == tag)

    point = centre[["cx", "cy", "cz"]].to_numpy(float)
    folder = Path(row["dir"])
    first = sorted(p.name for p in folder.glob("*.dcm"))[0]
    ds = pydicom.dcmread(str(folder / first), force=True)
    spacing = [float(v) for v in ds.PixelSpacing]
    normal = normal_of(ds.ImageOrientationPatient)
    cr, cc = pixel_of(ds.ImagePositionPatient, ds.ImageOrientationPatient,
                      spacing, point)
    depth = through_plane(point, normal)

    positions = np.asarray(series.positions, float)
    offset = positions - depth
    inside = (offset >= -low_mm) & (offset <= high_mm)
    box_px = extent_mm / series.mm_per_px

    eight = window(series.volume)
    frames, scale = [], 1.0
    for i in range(len(eight)):
        uri, scale = to_jpeg(_draw(eight[i], (cr, cc), box_px, bool(inside[i])),
                             max_width=430)
        frames.append(uri)

    kept = int(inside.sum())
    left, right = ("posterior", "anterior") if plane == "Sagittal" else ("medial", "lateral")
    caption = (f"<b>{name}</b> &mdash; {extent_mm:.0f} mm box at {mm_px:.3f} mm/px "
               f"&rarr; {_out_px(extent_mm, mm_px)} px. Depth window "
               f"&minus;{low_mm:.0f}/+{high_mm:.0f} mm holds <b>{kept}</b> of "
               f"{len(positions)} acquired slices"
               f"{f', thinned to {slots}' if kept > slots else ''}. Green box = this "
               f"slice is inside it.")
    return (f"<h4>{plane} &mdash; {row.get('SeriesDescription') or '?'} "
            f"<span class='meta'>{len(positions)} slices, "
            f"{series.mm_per_px:.3f} mm/px, step {np.median(np.diff(positions)):.2f} mm"
            f"</span></h4>"
            + P.viewer(f"v-{study[-8:]}-{plane[:3].lower()}", frames, left, right,
                       series.mm_per_px / max(scale, 1e-9),
                       flagged=[int(i) for i in np.flatnonzero(inside)],
                       caption=caption))


def load_headers(split: str = "train_series") -> pd.DataFrame:
    """The annotated header table, built once and cached.

    `plane` comes from the competition's own series table rather than from the headers,
    so this path and the training path pick the same series — the same rule
    `rsna.infer.chain._headers` follows.
    """

    if HEADERS.is_file():
        return pd.read_parquet(HEADERS)
    from rsna.dicom import annotate, walk
    frame = annotate(walk(DICOM_ROOT, split))
    csv = ROOT / f"data/raw/{split.replace('_series', '')}_series.csv"
    if csv.is_file():
        planes = pd.read_csv(csv)
        frame["plane"] = frame["SeriesInstanceUID"].map(
            dict(zip(planes["SeriesInstanceUID"], planes["Anatomical_Plane"])))
    frame.to_parquet(HEADERS)
    print(f"  built {HEADERS} — {len(frame)} series")
    return frame


# -- page ------------------------------------------------------------------- #

def candidates_table(plane: str, measured: pd.DataFrame) -> str:
    ref = 336 * 336
    got = measured[measured.plane == plane]
    step = float(got.step.median()) if len(got) else 3.4
    rows = []
    for name, extent, mm_px, (low, high), slots in CANDIDATES[plane]:
        px = _out_px(extent, mm_px)
        fit = int(round((low + high) / step)) + 1
        rows.append(
            f"<tr><td><code>{name}</code></td><td>{extent:.0f} mm</td>"
            f"<td>{mm_px:.3f}</td><td>{px}&times;{px}</td>"
            f"<td>&minus;{low:.0f}/+{high:.0f} = {low+high:.0f} mm</td>"
            f"<td>{fit}</td><td><b>{slots}</b></td>"
            f"<td>{(low+high)/slots:.1f} mm</td>"
            f"<td>{slots * px * px / ref:.1f}</td></tr>")
    return (f"<table class='report' style='max-height:none'>"
            f"<tr><th>id</th><th>box</th><th>mm/px</th><th>output</th>"
            f"<th>depth window</th><th>slices that fall in</th><th>slots</th>"
            f"<th>effective step</th><th>cost in window-336&sup2;</th></tr>"
            f"{''.join(rows)}</table>")


def build(out: Path, n_studies: int) -> None:
    config = Config()
    centres = knee_centres()
    headers = load_headers()
    measured = reach(headers, centres)

    studies = sorted(set(centres.index) & set(headers.StudyInstanceUID))
    rng = np.random.default_rng(5)
    shown = [str(s) for s in rng.choice(studies, n_studies, replace=False)]

    stats = []
    for plane in ("Sagittal", "Coronal", "Axial"):
        g = measured[measured.plane == plane]
        if not len(g):
            continue
        stats.append(
            f"<tr><td><b>{plane}</b></td><td>{g.span.median():.0f} mm</td>"
            f"<td>{g.span.quantile(.1):.0f} &ndash; {g.span.quantile(.9):.0f}</td>"
            f"<td>{g.step.median():.2f} mm</td>"
            f"<td>{g.low.median():.0f} mm</td><td>{g.high.median():.0f} mm</td></tr>")

    body = [f"""
<h1>The whole-knee boxes</h1>
<p class="lede">Pipeline v3 needs two branches the registry does not have: a box that sees
the <b>whole knee</b>, sagittal and coronal, cut around the predicted joint centre instead
of the image centre. This page derives their centre, measures what the acquisition allows,
and proposes geometries you can scroll through.</p>

<div class="card">
<h3 style="margin-top:0">Why they are needed</h3>
<p>A spec's depth window is <code>lateral_mm + medial_mm</code> around its landmark. The
widest box on the books is <code>baker_wide</code> &mdash; 110&times;95 mm in plane, which
<i>looks</i> like a whole knee &mdash; and its depth window is
<b>{SPECS['baker_wide'].lateral_mm + SPECS['baker_wide'].medial_mm:.0f} mm</b>. Against a
sagittal stack spanning {measured[measured.plane=='Sagittal'].span.median():.0f} mm, every
box we own is a third of a knee or less. That is right for a compartment and wrong for an
effusion.</p>
</div>

<h2>Where the centre comes from</h2>
<p>Both boxes hang off <b>one</b> point, and it is the same point: the midpoint of
<code>lat_centre</code> and <code>med_centre</code>, the two meniscus centres predicted by
the <code>landmark_both</code> run over all 4407 studies. The two differ mainly along the
medial-lateral axis, so their midpoint is the joint centre and is stable even when one of
them is a little off.</p>
<p><b>One point serves every plane, and no new annotation is needed.</b> The landmark is a
3-D position <b>in patient millimetres</b>, not a pixel. Two functions do the rest, and
neither asks which plane the annotation was made on:</p>
<pre class="report" style="max-height:none">pixel_of(ipp, iop, spacing, point)   &rarr; (row, col) on any slice
through_plane(point, normal_of(iop)) &rarr; how far along any stack axis</pre>
<p>This is not a hope &mdash; it already runs in production. <code>lateral_oa</code> and
<code>medial_oa</code> are <b>coronal</b> boxes hanging off points annotated on
<b>sagittal</b> images, and they score 0.8121 and 0.8611 out of fold. The cross-plane
projection is measured, not assumed.</p>

<h2>What the acquisition allows</h2>
<div class="note">120 series per plane, headers only. &ldquo;Reach&rdquo; is how far the
stack extends either side of the landmark &mdash; the hard cap on any depth window.</div>
<table class="report" style="max-height:none;margin-top:10px">
<tr><th>plane</th><th>stack extent (median)</th><th>q10 &ndash; q90</th>
    <th>step</th><th>reach one way</th><th>the other</th></tr>
{''.join(stats)}
</table>
<p>Three consequences, one per plane:</p>
<ul>
<li><b>Sagittal is centred.</b> 49 mm of reach each way, so a symmetric
&plusmn;45 mm window takes almost the whole stack.</li>
<li><b>Coronal is not.</b> 60 mm one way against 41 mm the other, so a symmetric window
would be capped at &plusmn;41 and waste the 19 mm of reach on the roomy side. Its window
must be <b>asymmetric</b> &mdash; which <code>RoiSpec</code> already expresses, since
<code>lateral_mm</code> and <code>medial_mm</code> are separate fields.</li>
<li><b>Axial has 77 mm of reach above the joint centre</b> against 45 below. That is the
suprapatellar region, where an effusion actually collects &mdash; an argument for an axial
wide box later, placed above the joint line rather than on it. Not proposed here.</li>
</ul>
"""]

    for plane in ("Sagittal", "Coronal"):
        body.append(f"<h2>{plane} &mdash; candidate geometries</h2>")
        body.append(candidates_table(plane, measured))

    body.append("""
<p>Reading the table: <b>slices that fall in</b> is how many acquired slices the depth
window contains at that plane's median step, and <b>slots</b> is the cap. When the cap is
below what falls in, <code>choose_slices</code> thins by even spacing &mdash; it drops
slices, it never averages them &mdash; so the effective step is the last column. A wide
box is for diffuse findings and can afford a coarser step than a compartment box's
3.1&ndash;3.6 mm.</p>
<p><b>At these depths the window is barely a crop.</b> Measured on the studies below, a
&plusmn;45 mm sagittal window holds 22 of 24, 30 of 34, 25 of 29 acquired slices &mdash;
about 90% of the stack. So the depth window is not what selects; <b>the slot cap is</b>,
and the real decision in these tables is the last two columns. Widening the window
further buys almost nothing, while raising the cap buys resolution in depth at linear
cost.</p>
<p>Cost is in 336&sup2;-window equivalents, the unit pipeline v3 budgets in: one Raptor
arm is 94. The seven pathology boxes at one channel each come to 59 windows and 19.4 of
these units, so a pair of wide boxes at 10&ndash;15 each roughly doubles the bag's cost
and leaves it under half a Raptor arm.</p>
""")

    body.append(f"<h2>Where the box lands, slice by slice</h2>"
                f"<div class='note'>Scroll the image, drag the slider, or click it and "
                f"use the arrow keys. The box is <span style='color:{IN_COLOUR}'>green "
                f"with a centre cross</span> on slices inside the depth window and "
                f"<span style='color:{OUT_COLOUR}'>grey</span> outside it, so scrolling "
                f"shows the slab begin and end. Geometry shown: "
                f"<code>{SHOWN['Sagittal']}</code> and "
                f"<code>{SHOWN['Coronal']}</code>.</div>")

    for study in shown:
        centre = centres.loc[study]
        conf = [f"{centre.get(f'confidence:{p}', float('nan')):.3f}"
                for p in CENTRE_POINTS]
        spread = [f"{centre.get(f'spread_mm:{p}', float('nan')):.1f}"
                  for p in CENTRE_POINTS]
        body.append(
            f"<h3>{study[-12:]}</h3>"
            f"<div class='note'>joint centre "
            f"({centre.cx:+.1f}, {centre.cy:+.1f}, {centre.cz:+.1f}) mm &middot; "
            f"landmark confidence {' / '.join(conf)} &middot; fold spread "
            f"{' / '.join(spread)} mm</div>")
        for plane in ("Sagittal", "Coronal"):
            body.append(plane_block(study, plane, centre, config, SHOWN[plane]))

    body.append("""
<h2>What this page does not settle</h2>
<ul>
<li><b>Which geometry is right.</b> These are candidates. Depth coverage against
resolution is a trade only an out-of-fold run can price, and the wide branches exist for
the five targets no ROI covers &mdash; so the number that decides is their macro, not the
seven the experts already own.</li>
<li><b>The coronal window's direction.</b> The measurement says one side has 60 mm of
reach and the other 41, but which is anterior is set per series by
<code>toward_bowtie</code> from image content, not asserted here. The proposed
&minus;55/+40 assumes the roomy side is the one that matters; if that is backwards the
numbers swap and nothing else changes.</li>
<li><b>Whether a wide box beats more pathology boxes</b> for the same budget. Untested.</li>
</ul>
""")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(P.page("The whole-knee boxes", "".join(body)))
    print(f"wrote {out}  ({out.stat().st_size / 1024:.0f} KB, "
          f"{len(shown)} studies, {len(measured)} series measured)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--out", type=Path, default=ROOT / "docs/atlas/wide.html")
    ap.add_argument("--studies", type=int, default=3)
    args = ap.parse_args()
    build(args.out, args.studies)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
