"""What the wide model actually crops, and what it would crop centred on the knee.

    python -m tools.atlas.crop_page -o docs/atlas/crop.html

The wide model takes a 130 mm square and twelve slices from every series it reads. Both
are positioned **by the stack, not by the anatomy**:

    cy, cx = h // 2, w // 2                  # rsna/dicom/pixels.py
    volume = volume[:, cy - half:cy + half, cx - half:cx + half]

    lo, hi = int(band[0] * (n - 1)), int(band[1] * (n - 1))     # band = (0.2, 0.8)

So the window lands where the technician put the knee. This page measures how far that
is, shows the boxes side by side on real studies, and prices the alternative: the same
crop centred on a point we already predict for all 4407 studies — the midpoint of
`lat_centre` and `med_centre`, which needs no new annotation and no new model.

The honest summary of the measurement, so it is not buried: the knee sits a median
16 mm off the image centre, and a half-knee is about 48 mm against a 65 mm half-crop.
At the median the joint already touches the edge of the window, and half the corpus is
past it. Centring is therefore not a resolution tweak — it stops a loss that is already
happening.

The slice band, asked the same question, comes back fine: a mean 10.7 of 12 slices land
within 30 mm of the joint and never fewer than 7. Out of plane there is nothing to fix,
and this page says so rather than leaving the door open.

Nothing here is a radiological reading. The boxes are geometry, the landmark is a model
output with its own error, and the only claim is about where pixels are, not what is in
them.
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
from rsna.dicom.pixels import sample_indices                         # noqa: E402
from tools.atlas import page as P                                    # noqa: E402
from tools.atlas.render import to_jpeg, window                       # noqa: E402
from tools.atlas.study import load_series, series_headers, side_of   # noqa: E402

LANDMARKS = Path("/data/mgr/rsna-knee/roi-infer/landmarks.csv")
DROOT = Path("/data/mgr/rsna-knee/extracted/train_series")

#: The point the knee is centred on: the midpoint of the two meniscus centres. They
#: differ mainly along the medial-lateral axis, so their midpoint is the joint centre
#: and its in-plane position is stable even when one of the two is a little off.
CENTRE_POINTS = ("lat_centre", "med_centre")

#: Candidate windows. `None` means "the image centre", i.e. what runs today.
BOXES = [
    ("today", 130.0, None, "#e0564f", "130 mm, image centre — what runs today"),
    ("c130", 130.0, "knee", "#4e9ef0", "130 mm, knee centre — same window, moved"),
    ("c110", 110.0, "knee", "#4fbf7a", "110 mm, knee centre — 1.4x the pixels on the joint"),
    ("c95", 95.0, "knee", "#b183e0", "95 mm, knee centre — snug; clips a wide knee"),
]

#: How far either side of the landmark the centred slice band reaches, in millimetres.
SLAB_MM = 30.0

#: Half the width of a knee, used only to show that the margin against the half-crop is
#: thin. A round figure, not a measurement of this corpus — the conclusion holds for any
#: value between 40 and 55 mm, which is why it is a constant and not a claim.
HALF_KNEE_MM = 48.0


# -- geometry ---------------------------------------------------------------- #

def knee_centres(table: Path = LANDMARKS) -> tuple[pd.DataFrame, pd.Series]:
    """The joint centre per study, and the series each was read from."""

    lm = pd.read_csv(table)
    both = lm[lm.point.isin(CENTRE_POINTS)]
    centre = both.groupby("study")[["x_mm", "y_mm", "z_mm"]].mean()
    series = both[both.point == CENTRE_POINTS[0]].set_index("study")["series"]
    return centre, series


def _offset_one(study: str, point: np.ndarray, uid: str) -> dict | None:
    """One study's knee-centre offset from the image centre, in millimetres.

    Headers only — no pixel decoding — so this runs over the corpus in seconds and the
    distribution it produces is what decides whether the rest of the page is worth
    building.
    """

    try:
        folder = DROOT / study / uid
        files = sorted(folder.glob("*.dcm"))
        ds = pydicom.dcmread(str(files[len(files) // 2]), stop_before_pixels=True)
        sp = [float(v) for v in ds.PixelSpacing]
        row, col = pixel_of(ds.ImagePositionPatient, ds.ImageOrientationPatient, sp, point)
        d_row = (row - ds.Rows / 2) * sp[0]
        d_col = (col - ds.Columns / 2) * sp[1]
        return {"study": study, "d_row_mm": d_row, "d_col_mm": d_col,
                "dist_mm": float(np.hypot(d_row, d_col)),
                "fov_mm": float(min(ds.Rows * sp[0], ds.Columns * sp[1]))}
    except Exception:                                                # noqa: BLE001
        return None


def offsets(centre: pd.DataFrame, series: pd.Series, limit: int = 0,
            workers: int = 32) -> pd.DataFrame:
    """The offset distribution over every study that carries both meniscus points."""

    studies = sorted(set(centre.index) & set(series.index))[:limit or None]
    with ThreadPoolExecutor(workers) as pool:
        rows = pool.map(lambda s: _offset_one(s, centre.loc[s].to_numpy(float),
                                              str(series[s])), studies)
        got = [r for r in rows if r is not None]
    return pd.DataFrame(got)


# -- drawing ----------------------------------------------------------------- #

def _box_px(shape, mm_per_px: float, extent_mm: float,
            centre_rc: tuple[float, float] | None) -> tuple[int, int, int, int]:
    """One window as (r0, r1, c0, c1), clamped into the slice.

    Clamping rather than padding, because that is what the training code does: it slices
    the array and takes whatever is there. A box that runs off the edge comes back
    smaller, and the resize that follows stretches it — which is itself part of what
    this page is showing.
    """

    h, w = shape[:2]
    want = int(round(extent_mm / mm_per_px))
    half = want // 2
    cr, cc = (h / 2, w / 2) if centre_rc is None else centre_rc
    r0, c0 = int(round(cr - half)), int(round(cc - half))
    return (max(0, r0), min(h, r0 + 2 * half), max(0, c0), min(w, c0 + 2 * half))


def _overlay(slice8: np.ndarray, boxes: list[tuple]) -> np.ndarray:
    """The acquired slice with every candidate window drawn on it."""

    rgb = cv2.cvtColor(slice8, cv2.COLOR_GRAY2BGR)
    thick = max(1, int(round(min(rgb.shape[:2]) / 220)))
    for (_, _, _, colour, _), (r0, r1, c0, c1) in boxes:
        bgr = tuple(int(colour[i:i + 2], 16) for i in (5, 3, 1))
        cv2.rectangle(rgb, (c0, r0), (c1 - 1, r1 - 1), bgr, thick)
    return rgb


def _as_input(slice8: np.ndarray, rect: tuple[int, int, int, int],
              img: int) -> np.ndarray:
    """One window resized to the encoder's input, the way training resizes it."""

    r0, r1, c0, c1 = rect
    cut = slice8[r0:r1, c0:c1]
    if cut.size == 0:
        return np.zeros((img, img), np.uint8)
    return cv2.resize(cut, (img, img), interpolation=cv2.INTER_AREA)


# -- the histogram, as inline SVG -------------------------------------------- #

def histogram_svg(values: np.ndarray, width: int = 680, height: int = 190,
                  bins: int = 34, mark: float | None = None) -> str:
    """A bar chart with no library, because the atlas pages load nothing external."""

    counts, edges = np.histogram(values, bins=bins)
    top = counts.max() or 1
    pad_l, pad_b, pad_t = 38, 26, 10
    plot_w, plot_h = width - pad_l - 8, height - pad_b - pad_t
    bar_w = plot_w / bins
    bars = []
    for i, n in enumerate(counts):
        bh = plot_h * n / top
        x = pad_l + i * bar_w
        bars.append(f'<rect x="{x:.1f}" y="{pad_t + plot_h - bh:.1f}" '
                    f'width="{bar_w - 1.2:.1f}" height="{bh:.1f}" fill="#4e9ef0" '
                    f'opacity="0.85"/>')
    ticks = []
    lo, hi = float(edges[0]), float(edges[-1])
    for v in np.linspace(lo, hi, 6):
        x = pad_l + plot_w * (v - lo) / max(hi - lo, 1e-9)
        ticks.append(f'<text x="{x:.1f}" y="{height - 8}" fill="#8b949e" '
                     f'font-size="11" text-anchor="middle">{v:.0f}</text>')
    rule = ""
    if mark is not None and lo <= mark <= hi:
        x = pad_l + plot_w * (mark - lo) / max(hi - lo, 1e-9)
        rule = (f'<line x1="{x:.1f}" y1="{pad_t}" x2="{x:.1f}" '
                f'y2="{pad_t + plot_h}" stroke="#e0564f" stroke-width="1.5" '
                f'stroke-dasharray="4 3"/>'
                f'<text x="{x + 5:.1f}" y="{pad_t + 12}" fill="#e0564f" '
                f'font-size="11">{mark:.0f} mm</text>')
    return (f'<svg viewBox="0 0 {width} {height}" width="100%" '
            f'style="max-width:{width}px" role="img" '
            f'aria-label="distribution of knee-centre offset in millimetres">'
            f'<text x="4" y="{pad_t + 10}" fill="#8b949e" font-size="11">studies</text>'
            f'{"".join(bars)}{rule}{"".join(ticks)}'
            f'<text x="{width/2:.0f}" y="{height - 8}" fill="#8b949e" font-size="11" '
            f'text-anchor="end" dx="-70">mm from the image centre</text></svg>')


# -- one study's row --------------------------------------------------------- #

def study_block(study: str, centre_mm: np.ndarray, off: dict,
                config: Config) -> str:
    """The acquired slice with the windows on it, then what each window hands over."""

    headers = series_headers(study)
    sag = headers[headers.plane.str.lower().str.startswith("sag")] \
        if "plane" in headers else headers
    if sag.empty:
        sag = headers
    row = sag.iloc[len(sag) // 2]
    series = load_series(row, config)
    side, _ = side_of(headers)

    ds = pydicom.dcmread(str(next((Path(row["dir"])).glob("*.dcm"))), force=True)
    sp = [float(v) for v in ds.PixelSpacing]
    # Any slice of the stack gives the same in-plane projection: measured over six
    # series the row and column vary by 0.00 px end to end, because a parallel stack
    # advances its IPP along the normal only. So reading one header is exact here, not
    # an approximation of the slice actually displayed.
    cr, cc = pixel_of(ds.ImagePositionPatient, ds.ImageOrientationPatient, sp, centre_mm)

    eight = window(series.volume)
    mid = len(eight) // 2
    boxes = [(b, _box_px(eight[mid].shape, series.mm_per_px, b[1],
                         None if b[2] is None else (cr, cc))) for b in BOXES]

    over, _ = to_jpeg(_overlay(eight[mid], boxes), max_width=460)
    cells = []
    for (name, extent, anchor, colour, label), rect in boxes:
        uri, _ = to_jpeg(_as_input(eight[mid], rect, config.img), max_width=config.img)
        mm_px = extent / config.img
        cells.append(
            f'<div><div class="sub" style="color:{colour}">{P._esc(label)}</div>'
            f'<div class="stage" style="margin-top:6px"><img src="{uri}" alt=""></div>'
            f'<div class="why">{config.img} px over {extent:.0f} mm = '
            f'<b>{mm_px:.3f} mm/px</b></div></div>')

    return f"""
<h3>{study[-12:]} &mdash; knee {off['dist_mm']:.0f} mm off centre</h3>
<div class="note">{P._esc(str(row.get('SeriesDescription') or ''))} &middot;
  {side or '?'} knee &middot; {series.mm_per_px:.3f} mm/px acquired &middot;
  field of view {off['fov_mm']:.0f} mm &middot;
  offset {off['d_col_mm']:+.0f} mm across, {off['d_row_mm']:+.0f} mm down</div>
<div style="display:flex;gap:18px;flex-wrap:wrap;align-items:flex-start;margin-top:10px">
  <div><div class="stage"><img src="{over}" alt="windows drawn on the acquired slice"></div>
       <div class="why">the acquired slice, with every candidate window on it</div></div>
  <div style="flex:1;min-width:320px;display:grid;
              grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px">
    {''.join(cells)}
  </div>
</div>"""


def slice_coverage(centre: pd.DataFrame, series: pd.Series, config: Config,
                   limit: int = 600, workers: int = 32) -> pd.DataFrame:
    """How many of the band's slices land near the joint, over a sample of studies.

    Asked because the in-plane result invites the same question out of plane, and the
    answer turns out to be the opposite — so this exists to close that door rather than
    to open it. Headers only.
    """

    studies = sorted(set(centre.index) & set(series.index))
    rng = np.random.default_rng(7)
    if limit and limit < len(studies):
        studies = list(rng.choice(studies, limit, replace=False))

    def one(study):
        try:
            folder = DROOT / study / str(series[study])
            pos, normal = [], None
            for f in sorted(folder.glob("*.dcm")):
                ds = pydicom.dcmread(str(f), stop_before_pixels=True)
                if normal is None:
                    normal = normal_of(ds.ImageOrientationPatient)
                pos.append(through_plane(
                    np.array([float(v) for v in ds.ImagePositionPatient]), normal))
            pos = np.sort(np.asarray(pos))
            depth = through_plane(centre.loc[study].to_numpy(float), normal)
            idx = sample_indices(len(pos), config.slices, config.band)
            return {"study": study, "n": len(pos), "span": float(np.ptp(pos)),
                    "hit": int((np.abs(pos[idx] - depth) <= SLAB_MM).sum())}
        except Exception:                                            # noqa: BLE001
            return None

    with ThreadPoolExecutor(workers) as pool:
        return pd.DataFrame([r for r in pool.map(one, studies) if r is not None])


def slices_block(study: str, centre_mm: np.ndarray, config: Config,
                 cov: pd.DataFrame) -> str:
    """The band against the joint: the corpus number first, then one study to look at."""

    headers = series_headers(study)
    sag = headers[headers.plane.str.lower().str.startswith("sag")] \
        if "plane" in headers else headers
    row = (sag if not sag.empty else headers).iloc[0]
    series = load_series(row, config)

    ds = pydicom.dcmread(str(next((Path(row["dir"])).glob("*.dcm"))), force=True)
    depth = through_plane(centre_mm, normal_of(ds.ImageOrientationPatient))
    pos = np.asarray(series.positions, float)

    today = sample_indices(len(pos), config.slices, config.band)
    near = np.where(np.abs(pos - depth) <= SLAB_MM)[0]
    centred = (near[np.linspace(0, len(near) - 1, config.slices).astype(int)]
               if len(near) else today)

    eight = window(series.volume)

    def strip(idx):
        out = []
        for i in np.unique(idx):
            uri, _ = to_jpeg(cv2.resize(eight[i], (96, 96),
                                        interpolation=cv2.INTER_AREA), max_width=96)
            out.append(f'<img src="{uri}" alt="" style="border-radius:4px">')
        return "".join(out)

    worst = int(cov.hit.min())
    all_in = 100 * (cov.hit == config.slices).mean()
    return f"""
<h2>Out of plane the band is already fine &mdash; a lever that is not there</h2>
<div class="note">The in-plane result invites the same question about slices. It does not
  survive the measurement: {len(cov)} studies, headers only.</div>
<table class="report" style="max-height:none;margin-top:10px">
<tr><th>of {config.slices} band slices, mean within {SLAB_MM:.0f} mm of the joint</th>
    <td><b>{cov.hit.mean():.1f}</b></td></tr>
<tr><th>worst study in the sample</th><td>{worst} of {config.slices}</td></tr>
<tr><th>studies with fewer than 8 inside</th>
    <td>{100 * (cov.hit < 8).mean():.0f}%</td></tr>
<tr><th>studies with all {config.slices} inside</th><td>{all_in:.0f}%</td></tr>
<tr><th>median stack extent</th><td>{cov.span.median():.0f} mm</td></tr>
</table>
<p>So the through-plane band needs no landmark. The asymmetry makes sense: a sagittal
stack spans {cov.span.median():.0f} mm and is prescribed <i>on</i> the joint, because
that is what prescribing a knee series means &mdash; while the in-plane field of view is
160 mm and where the knee sits inside it is not part of the prescription. <b>Only the
in-plane crop is worth moving.</b></p>
<h3>One study, for the eye</h3>
<div class="note">{len(pos)} slices acquired, {pos.min():.0f} to {pos.max():.0f} mm along
  the stack; the landmark sits at <b>{depth:.0f} mm</b>.</div>
<table class="report" style="max-height:none;width:100%;margin-top:10px">
<tr><th style="width:150px">band (0.2, 0.8)</th><td>{strip(today)}</td></tr>
<tr><th>within {SLAB_MM:.0f} mm of the point</th><td>{strip(centred)}</td></tr>
</table>
<div class="why">The two rows are nearly the same slices, which is the point.</div>"""


# -- page -------------------------------------------------------------------- #

def build(out: Path, n_offsets: int, n_studies: int) -> None:
    config = Config()
    centre, series = knee_centres()
    off = offsets(centre, series, limit=n_offsets)
    if off.empty:
        raise SystemExit("no study yielded an offset; is the landmark table present?")

    half_crop = config.crop_mm / 2
    q = off.dist_mm.quantile([.1, .5, .75, .9, .99]).round(1)
    clipped = 100 * (off.dist_mm > half_crop - HALF_KNEE_MM).mean()

    # Studies spanning the distribution, so the panels are not all the easy case.
    picks = []
    for frac in (0.10, 0.50, 0.85, 0.98):
        target = off.dist_mm.quantile(frac)
        row = off.iloc[(off.dist_mm - target).abs().argsort().iloc[0]]
        if row.study not in [p.study for p in picks]:
            picks.append(row)

    body = [f"""
<h1>The window the wide model takes, and where the knee actually is</h1>
<p class="lede">The wide model crops 130 mm and twelve slices out of every series. Both
are placed by the stack rather than by the anatomy &mdash; the square is centred on the
image, the slices on the middle 60% of the pile. This page measures what that costs and
shows the alternative, which needs no new annotation: the midpoint of the two meniscus
centres, already predicted for all 4407 studies.</p>

<div class="card">
<h3 style="margin-top:0">What the code does today</h3>
<pre class="report" style="max-height:none">cy, cx = h // 2, w // 2
volume = volume[:, cy - half:cy + half, cx - half:cx + half]   <span
  style="color:#8b949e"># rsna/dicom/pixels.py</span>

lo, hi = int(band[0] * (n - 1)), int(band[1] * (n - 1))        <span
  style="color:#8b949e"># band = (0.2, 0.8)</span></pre>
</div>

<h2>How far off centre the knee sits</h2>
<div class="note">{len(off)} studies, headers only. The dashed line is the
  {half_crop:.0f} mm half-crop: a point beyond it is outside the window entirely.</div>
{histogram_svg(off.dist_mm.to_numpy(), mark=half_crop)}
<table class="report" style="max-height:none;margin-top:12px">
<tr><th>median</th><td>{q[0.5]:.0f} mm</td>
    <th>75th</th><td>{q[0.75]:.0f} mm</td>
    <th>90th</th><td>{q[0.9]:.0f} mm</td>
    <th>99th</th><td>{q[0.99]:.0f} mm</td>
    <th>worst</th><td>{off.dist_mm.max():.0f} mm</td></tr>
</table>
<p>The joint centre is not the edge of the knee. A half-knee is roughly {HALF_KNEE_MM:.0f} mm, so the
far side of the joint sits at <i>offset + {HALF_KNEE_MM:.0f} mm</i> while the window reaches
{half_crop:.0f} mm. At the median offset of {q[0.5]:.0f} mm that is
{q[0.5] + HALF_KNEE_MM:.0f} mm against {half_crop:.0f} &mdash; already level with the edge; and
<b>{clipped:.0f}%</b> of studies are past it, which means the crop is cutting anatomy
off, not merely wasting pixels on air.</p>
<p>So centring is not a resolution tweak. Resolution is the <i>second</i> prize: once the
window is reliably on the joint, it can also be made smaller, and the panels below show
what each size hands the encoder.</p>

<h2>The windows, on real studies</h2>
<div class="note">Four studies spanning the offset distribution &mdash; the 10th, 50th,
  85th and 98th percentiles &mdash; so the comparison is not made only where it is
  flattering. Each right-hand tile is what the encoder is handed, at its real
  {config.img} px.</div>"""]

    for row in picks:
        body.append(study_block(row.study, centre.loc[row.study].to_numpy(float),
                                row.to_dict(), config))

    cov = slice_coverage(centre, series, config)
    body.append(slices_block(picks[-1].study,
                             centre.loc[picks[-1].study].to_numpy(float), config, cov))

    body.append(f"""
<h2>What this does and does not establish</h2>
<p>It establishes where pixels are. The offsets are header geometry and the windows are
arithmetic on it; neither depends on the landmark model being <i>right</i>, only on it
being where it says it is.</p>
<p>It does not establish that centring raises the score. The landmark carries its own
error, a model trained on off-centre crops has already spent capacity learning to
tolerate them, and that capacity does not transfer for free. The measurement that
settles it is out of fold and costs no submission: refit the wide model with the window
anchored on this point and compare against {0.899:.3f}.</p>
<div class="card">
<h3 style="margin-top:0">Caveats worth keeping</h3>
<ul>
<li>The offset is measured on the sagittal series the landmark model read, not on each
of the six slots the wide model fills. Other slots are acquired in other planes and
their offsets differ.</li>
<li>&ldquo;A half-knee is {HALF_KNEE_MM:.0f} mm&rdquo; is a round figure, not a measurement of these
studies. It is used only to show the margin is thin, and the conclusion survives any
value from 40 to 55 mm.</li>
<li>The slice coverage is sampled, not exhaustive, and measured on one series per
study &mdash; the sagittal one the landmark read. A coronal or axial stack is prescribed
differently and its band may not land as well.</li>
</ul>
</div>""")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(P.page("The wide model's window", "".join(body)))
    print(f"wrote {out}  ({out.stat().st_size / 1024:.0f} KB, "
          f"{len(off)} offsets, {len(picks)} studies)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--out", type=Path, default=ROOT / "docs/atlas/crop.html")
    ap.add_argument("--offsets", type=int, default=0,
                    help="cap the offset measurement; 0 reads every study")
    ap.add_argument("--studies", type=int, default=4)
    args = ap.parse_args()
    build(args.out, args.offsets, args.studies)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
