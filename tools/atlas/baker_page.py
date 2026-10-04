"""The popliteal (Baker's) cyst: why its expert is a de-zoom, and where its box goes.

    python -m tools.atlas.baker_page -o docs/atlas/baker.html

The fifth of these pages, and the first for a target the pixel-size rule says to leave
alone. A cyst is 64 px across in the wide model's view, far past the 10 px where the
collateral expert failed, so cropping tighter cannot be the move. The measurement here
asks a different question instead — not *is it resolved* but *is it in the window* — and
answers it with the one number the wide model's sampler throws away.

Nothing here is a radiological reading. I cannot point at a cyst on these images reliably,
so the cyst is located statistically, by asking where studies whose report mentions one
are brighter than studies whose report does not. The labels are report-extracted.
"""

from __future__ import annotations

import argparse
import os
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
from tools.atlas import page as P                                    # noqa: E402
from tools.atlas.render import stack_to_jpegs, to_jpeg, window       # noqa: E402
from tools.atlas.study import (TRAIN_SERIES, load_series,            # noqa: E402
                               series_headers, side_of, stack_orientation)

LABELS = ROOT / "data/external/pilkwang-rsna-knee-llm-labels/report_labels_v2.csv"
BOTH = ROOT / "data/manual_annotations/landmarks-both.csv"
DROOT = Path("/data/mgr/rsna-knee/extracted/train_series")

#: The sampling frame of the difference map, in millimetres from `med_centre`:
#: posterior positive, superior positive, medial positive.
AP, SI, STEP = (-40, 80), (-60, 60), 1.0
DEPTH = np.arange(-8, 33, 4)
NW, NH = int((AP[1] - AP[0]) / STEP), int((SI[1] - SI[0]) / STEP)

#: The box this page proposes, and the depth window it is cut over.
BOX_W, BOX_H, OUT_W, OUT_H = 90.0, 75.0, 252, 210
OFF_POST, RISE_SUP = 16.0, 20.0
WIN = (-8.0, 20.0)
#: What the wide model's band keeps medial of the point, measured over 297 studies.
BAND_KEEP = 5.3


# ----------------------------------------------------------------- the measurement

def _series_table() -> pd.DataFrame:
    ser = pd.read_csv(ROOT / "data/raw/train_series.csv")
    return ser[(ser.Anatomical_Plane == "Sagittal") & (ser.Fluid_Sensitive == 1)
               & (ser.Fat_Suppression == 1)].groupby("StudyInstanceUID").first()


def _points() -> pd.DataFrame:
    lm = pd.read_csv(BOTH)
    return lm[(lm.point == "med_centre")
              & (~lm.skipped.astype(str).str.lower().isin(["true", "1"]))]


def _read(study: str, sag: pd.DataFrame):
    """One sagittal fat-suppressed series, ordered, with the geometry a crop needs."""

    d = DROOT / study / sag.loc[study, "SeriesInstanceUID"]
    if not d.is_dir():
        return None
    names = sorted(e.name for e in os.scandir(d) if e.name.endswith(".dcm"))
    if len(names) < 8:
        return None
    try:
        hd = {nm: pydicom.dcmread(str(d / nm), force=True) for nm in names}
        iop = [float(x) for x in hd[names[0]].ImageOrientationPatient]
        ps = [float(x) for x in hd[names[0]].PixelSpacing]
        nrm = normal_of(iop)
        t = np.array([through_plane(
            [float(x) for x in hd[nm].ImagePositionPatient], nrm) for nm in names])
        o = np.argsort(t)
        names = [names[i] for i in o]
        t = t[o]
        vol = np.stack([hd[nm].pixel_array.astype(np.float32) for nm in names])
    except Exception:
        return None
    return hd, names, iop, ps, nrm, t, vol


def _frame(iop, t, tp):
    """The three signs this page needs, read off the series rather than assumed.

    DICOM patient coordinates are LPS: +x to the patient's left, **+y posterior**,
    +z superior. Reading posterior as -y put every box of the first draft in front of the
    knee instead of behind it, so the sign is taken from the orientation and named here.
    """

    post = 1.0 if iop[1] > 0 else -1.0          # columns run posterior
    inf = 1.0 if iop[5] < 0 else -1.0           # rows run inferior
    medial = -1.0 if np.argmin(np.abs(t - tp)) < len(t) / 2 else 1.0
    return post, inf, medial


def difference_map(cache: Path, limit: int | None = None):
    """Where reported cysts are brighter than reported non-cysts, in mm from the point.

    Fluid is the brightest thing on a fat-suppressed fluid-sensitive image, so the
    statistic is the frequency of bright voxels, thresholded per study against that
    study's own distribution so scanner gain drops out.
    """

    if cache.exists():
        z = np.load(cache)
        return z["pos"], z["neg"], int(z["n_pos"]), int(z["n_neg"])

    lab = pd.read_csv(LABELS).set_index("StudyInstanceUID")["Baker's"]
    sag = _series_table()
    rows = list(_points().itertuples())[:limit]

    def one(r):
        if r.study not in sag.index or r.study not in lab.index:
            return None
        got = _read(r.study, sag)
        if got is None:
            return None
        hd, names, iop, ps, nrm, t, vol = got
        p = np.array([r.x_mm, r.y_mm, r.z_mm])
        tp = through_plane(p, nrm)
        post, inf, medial = _frame(iop, t, tp)
        step = float(np.median(np.diff(t)))
        inside = vol[vol > np.percentile(vol, 40)]
        thr = np.percentile(inside, 90) if inside.size else np.inf
        k0 = int(np.argmin(np.abs(t - tp)))
        block = np.full((len(DEPTH), NH, NW), np.nan, np.float32)
        for di, dm in enumerate(DEPTH):
            k = int(round(k0 + medial * dm / abs(step)))
            if not (0 <= k < len(t)):
                continue
            ds = hd[names[k]]
            rr, cc = pixel_of([float(x) for x in ds.ImagePositionPatient], iop, ps, p)
            img = vol[k]
            ys = rr - inf * (np.arange(NH) * STEP + SI[0])[::-1] / ps[0]
            xs = cc + post * (np.arange(NW) * STEP + AP[0]) / ps[1]
            gy, gx = np.meshgrid(ys, xs, indexing="ij")
            ok = ((gy >= 0) & (gy < img.shape[0]) & (gx >= 0) & (gx < img.shape[1]))
            samp = np.zeros((NH, NW), np.float32)
            samp[ok] = img[gy[ok].astype(int), gx[ok].astype(int)]
            m = (samp > thr).astype(np.float32)
            m[~ok] = np.nan
            block[di] = m
        return float(lab.loc[r.study]), block

    with ThreadPoolExecutor(16) as ex:
        got = [x for x in ex.map(one, rows) if x is not None]
    pos = np.nanmean(np.stack([b for y, b in got if y > 0.5]), axis=0)
    neg = np.nanmean(np.stack([b for y, b in got if y < 0.1]), axis=0)
    n_pos = sum(1 for y, _ in got if y > 0.5)
    n_neg = sum(1 for y, _ in got if y < 0.1)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, pos=pos, neg=neg, n_pos=n_pos, n_neg=n_neg)
    return pos, neg, n_pos, n_neg


def capture(excess, off, rise, w, h) -> float:
    """Share of the measured excess a box of this size and centre contains."""

    c0 = int(round(((off - w / 2) - AP[0]) / STEP))
    r0 = int(round((SI[1] - (rise + h / 2)) / STEP))
    c1, r1 = c0 + int(w / STEP), r0 + int(h / STEP)
    c0, r0, c1, r1 = max(c0, 0), max(r0, 0), min(c1, NW), min(r1, NH)
    if c1 <= c0 or r1 <= r0:
        return 0.0
    return float(excess[r0:r1, c0:c1].sum() / excess.sum())


# ----------------------------------------------------------------------- figures

def _text(img, s, x, y, colour, scale=0.5):
    cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4)
    cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 1)


def map_figure(pos, neg) -> str:
    """The difference map, with the proposed box drawn on it."""

    use = (DEPTH >= WIN[0]) & (DEPTH <= WIN[1])
    flat = np.nan_to_num(np.nanmean(pos[use] - neg[use], axis=0))
    base = np.nan_to_num(np.nanmean(neg, axis=0))
    img = np.zeros((NH, NW, 3), np.uint8)
    s = np.percentile(np.abs(flat), 99.5) or 1.0
    img[..., 2] = np.clip(flat / s, 0, 1) * 255
    img[..., 0] = np.clip(-flat / s, 0, 1) * 255
    img[..., 1] = np.clip(base / (np.percentile(base, 99) or 1), 0, 1) * 95
    big = cv2.resize(img, (NW * 4, NH * 4), interpolation=cv2.INTER_NEAREST)

    def xy(ap, si):
        return int((ap - AP[0]) / STEP) * 4, int((SI[1] - si) / STEP) * 4

    x0, y0 = xy(OFF_POST - BOX_W / 2, RISE_SUP + BOX_H / 2)
    x1, y1 = xy(OFF_POST + BOX_W / 2, RISE_SUP - BOX_H / 2)
    cv2.rectangle(big, (x0, y0), (x1, y1), (255, 255, 255), 2)
    cx, cy = xy(0, 0)
    cv2.drawMarker(big, (cx, cy), (255, 255, 0), cv2.MARKER_CROSS, 26, 2)
    _text(big, "med_centre", cx + 10, cy - 10, (255, 255, 0), 0.6)
    _text(big, f"boite {BOX_W:.0f} x {BOX_H:.0f} mm", x0 + 8, y0 + 24, (255, 255, 255), 0.6)
    _text(big, "POSTERIEUR ->", 12, big.shape[0] - 16, (210, 210, 210), 0.6)
    _text(big, "rouge = plus souvent brillant chez les positifs", 12, 26,
          (200, 200, 200), 0.52)
    return to_jpeg(big, max_width=900)[0]


def profile_figure(pos, neg) -> str:
    """How the excess fades medially, against what the wide model's band keeps."""

    w, h = 900, 300
    img = np.zeros((h, w, 3), np.uint8)
    half = slice(int((0 - AP[0]) / STEP), NW)
    vals = [float(np.nanmean(pos[i][:, half]) - np.nanmean(neg[i][:, half]))
            for i in range(len(DEPTH))]
    top = max(vals) * 1.25
    pad, base_y = 60, h - 54
    bw = (w - pad - 20) / len(DEPTH)
    for i, (dm, v) in enumerate(zip(DEPTH, vals)):
        x = int(pad + i * bw)
        bh = int(max(v, 0) / top * (base_y - 40))
        kept = dm <= BAND_KEEP
        colour = (120, 200, 120) if kept else (90, 120, 255)
        cv2.rectangle(img, (x + 3, base_y - bh), (int(x + bw) - 3, base_y), colour, -1)
        _text(img, f"{dm:+d}", x + 6, base_y + 20, (190, 190, 190), 0.45)
    cut = int(pad + (np.searchsorted(DEPTH, BAND_KEEP)) * bw)
    cv2.line(img, (cut, 24), (cut, base_y), (255, 255, 255), 1)
    _text(img, "<- ce que la bande du modele large garde", 12, 20, (120, 200, 120), 0.52)
    _text(img, "ce qu'elle jette ->", cut + 10, 20, (90, 120, 255), 0.52)
    _text(img, "millimetres en dedans de med_centre", pad, h - 14, (190, 190, 190), 0.5)
    return to_jpeg(img, max_width=900)[0]


def box_figure(study: str, point: np.ndarray) -> tuple[str, str]:
    """The proposed box on the slice through the point, on one study."""

    sag = _series_table()
    got = _read(study, sag)
    if got is None:
        raise SystemExit(f"no sagittal fat-suppressed series for {study}")
    hd, names, iop, ps, nrm, t, vol = got
    tp = through_plane(point, nrm)
    post, inf, medial = _frame(iop, t, tp)
    k = int(np.argmin(np.abs(t - tp)))
    vis = window(vol)
    ds = hd[names[k]]
    rr, cc = pixel_of([float(x) for x in ds.ImagePositionPatient], iop, ps, point)
    rgb = cv2.cvtColor(vis[k], cv2.COLOR_GRAY2BGR)
    c2 = cc + post * OFF_POST / ps[1]
    r2 = rr - inf * RISE_SUP / ps[0]
    hw, hh = BOX_W / 2 / ps[1], BOX_H / 2 / ps[0]
    cv2.rectangle(rgb, (int(c2 - hw), int(r2 - hh)), (int(c2 + hw), int(r2 + hh)),
                  (120, 255, 120), 2)
    cv2.drawMarker(rgb, (int(cc), int(rr)), (255, 255, 0), cv2.MARKER_CROSS, 26, 2)
    _text(rgb, "med_centre", int(cc) + 10, int(rr) - 10, (255, 255, 0), 0.5)
    big = cv2.resize(rgb, (620, int(620 * rgb.shape[0] / rgb.shape[1])),
                     interpolation=cv2.INTER_CUBIC)
    cap = (f"&hellip;{P._esc(study[-11:])} &middot; coupe {k}/{len(names)} &middot; "
           f"{ps[1]:.2f} mm/px &mdash; ant&eacute;rieur &agrave; gauche, "
           f"sup&eacute;rieur en haut.")
    return to_jpeg(big, max_width=620)[0], cap


def crop_figure(study: str, point: np.ndarray) -> str:
    """What the encoder would be handed, walked across the depth window."""

    sag = _series_table()
    got = _read(study, sag)
    hd, names, iop, ps, nrm, t, vol = got
    tp = through_plane(point, nrm)
    post, inf, medial = _frame(iop, t, tp)
    step = float(np.median(np.diff(t)))
    k0 = int(np.argmin(np.abs(t - tp)))
    vis = window(vol)
    tiles = []
    for dm in (-8, -4, 0, 4, 8, 12, 16, 20):
        k = int(np.clip(round(k0 + medial * dm / abs(step)), 0, len(names) - 1))
        ds = hd[names[k]]
        rr, cc = pixel_of([float(x) for x in ds.ImagePositionPatient], iop, ps, point)
        c2, r2 = cc + post * OFF_POST / ps[1], rr - inf * RISE_SUP / ps[0]
        hw, hh = BOX_W / 2 / ps[1], BOX_H / 2 / ps[0]
        r0, r1 = int(r2 - hh), int(r2 + hh)
        c0, c1 = int(c2 - hw), int(c2 + hw)
        canvas = np.zeros((r1 - r0, c1 - c0), np.uint8)
        sr0, sc0 = max(r0, 0), max(c0, 0)
        sr1, sc1 = min(r1, vis.shape[1]), min(c1, vis.shape[2])
        if sr1 > sr0 and sc1 > sc0:
            canvas[sr0 - r0:sr1 - r0, sc0 - c0:sc1 - c0] = vis[k][sr0:sr1, sc0:sc1]
        tile = cv2.cvtColor(cv2.resize(canvas, (210, 175),
                                       interpolation=cv2.INTER_CUBIC),
                            cv2.COLOR_GRAY2BGR)
        kept = dm <= BAND_KEEP
        _text(tile, f"{dm:+d} mm", 6, 20, (120, 255, 120) if kept else (90, 160, 255), 0.5)
        if not kept:
            cv2.rectangle(tile, (1, 1), (208, 173), (90, 160, 255), 2)
        tiles.append(tile)
    return to_jpeg(np.hstack(tiles), max_width=1340)[0]


def case(uid: str, vid: str, note: str) -> str:
    headers = series_headers(uid)
    sag = headers[(headers["plane"] == "Sagittal")]
    if not len(sag):
        return ""
    row = sag.sort_values(["fatsat", "n_slices"], ascending=[False, False]).iloc[0]
    series = load_series(row, Config())
    side, _ = side_of(headers)
    first, last = stack_orientation("Sagittal", side)
    frames, _ = stack_to_jpegs(series.volume)
    caption = (f"{P._esc(series.label)} &middot; {series.n} coupes &middot; "
               f"{series.mm_per_px:.2f} mm/px &middot; genou {side or '?'} &mdash; "
               f"ant&eacute;rieur &agrave; gauche. {note}")
    return (f'<h3>&hellip;{P._esc(uid[-11:])}</h3>'
            + P.viewer(vid, frames, first, last, series.mm_per_px, caption=caption))


# -------------------------------------------------------------------------- page

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--out", type=Path, default=ROOT / "docs/atlas/baker.html")
    ap.add_argument("--cases", type=int, default=2)
    args = ap.parse_args()

    cache = args.out.parent / ".baker-map.npz"
    print("measuring where the cysts are ...", flush=True)
    pos, neg, n_pos, n_neg = difference_map(cache)
    use = (DEPTH >= WIN[0]) & (DEPTH <= WIN[1])
    flat = np.nan_to_num(np.nanmean(pos[use] - neg[use], axis=0))
    excess = np.clip(flat, 0, None)
    cap = capture(excess, OFF_POST, RISE_SUP, BOX_W, BOX_H)
    ij = np.unravel_index(int(np.argmax(flat)), flat.shape)
    peak_ap, peak_si = AP[0] + ij[1] * STEP, SI[1] - ij[0] * STEP

    lab = pd.read_csv(LABELS)
    pts = _points().set_index("study")
    sag = _series_table()
    have = [u for u in pts.index if u in sag.index and (TRAIN_SERIES / u).is_dir()]
    bakers = lab.set_index("StudyInstanceUID")["Baker's"]
    pos_u = [u for u in have if bakers.get(u, 0) > 0.5]
    neg_u = [u for u in have if bakers.get(u, 1) < 0.1]

    demo = pos_u[0]
    p_demo = pts.loc[demo, ["x_mm", "y_mm", "z_mm"]].to_numpy(float)
    print("drawing the figures ...", flush=True)
    box_img, box_cap = box_figure(demo, p_demo)
    crop_img = crop_figure(demo, p_demo)
    map_img = map_figure(pos, neg)
    prof_img = profile_figure(pos, neg)

    c = Config()
    px = c.crop_mm / c.img

    body = [f"""
<header>
  <h1>Kyste poplit&eacute; &mdash; le kyste de Baker</h1>
  <p class="lede">La premi&egrave;re cible que la r&egrave;gle des pixels dit de laisser
  tranquille. Un kyste fait <b>64&nbsp;px</b> dans la vue du mod&egrave;le large, six fois
  le seuil o&ugrave; l'expert du ligament collat&eacute;ral a &eacute;chou&eacute;&nbsp;:
  recadrer plus serr&eacute; ne peut rien apporter. La question pos&eacute;e ici n'est donc
  pas <em>est-ce r&eacute;solu</em> mais <b>est-ce dans la fen&ecirc;tre</b> &mdash; et la
  r&eacute;ponse est non, dans une direction pr&eacute;cise et mesurable.</p>
</header>

<div class="card">
<h2>1 &middot; Ce qui a chang&eacute; la question</h2>
<p>Les quatre experts pr&eacute;c&eacute;dents partaient du m&ecirc;me geste&nbsp;: zoomer
sur une l&eacute;sion que {px:.3f}&nbsp;mm par pixel ne r&eacute;sout pas. Mesur&eacute;
sur <b>297 &eacute;tudes</b> annot&eacute;es, le mod&egrave;le large ne manque pas le kyste
par manque de r&eacute;solution&nbsp;; il le manque parce que son
&eacute;chantillonneur jette les coupes o&ugrave; il est.</p>
<p>Le mod&egrave;le large prend <code>band = [0,20, 0,80]</code> de chaque pile, soit les
60&nbsp;% centraux. Sur une pile sagittale, cela rogne environ 20&nbsp;mm &agrave; chaque
bout &mdash; c'est-&agrave;-dire les coupes les plus m&eacute;diales, exactement
l&agrave; o&ugrave; se loge un kyste poplit&eacute;.</p>
<table class="report">
<tr><th>tissu au-del&agrave; du condyle m&eacute;dial</th><th>acquis</th><th>gard&eacute; par la bande</th></tr>
<tr><td>m&eacute;diane</td><td>26,4 mm</td><td><b>5,3 mm</b></td></tr>
<tr><td>&eacute;tudes avec &ge; 10 mm</td><td>97,6 %</td><td><b>19,9 %</b></td></tr>
<tr><td>&eacute;tudes avec &ge; 15 mm</td><td>94,3 %</td><td><b>4,7 %</b></td></tr>
</table>
<p>Et dans <b>13,5&nbsp;%</b> des &eacute;tudes, le centre du condyle m&eacute;dial
lui-m&ecirc;me tombe hors de la bande.</p>
<h3>Une correction, parce qu'elle change la conclusion</h3>
<p class="note">La premi&egrave;re version de cette mesure lisait le post&eacute;rieur comme
&minus;y au lieu de +y &mdash; les coordonn&eacute;es DICOM sont LPS, le post&eacute;rieur
est <b>+y</b> &mdash; et prenait le haut de l'image du mauvais bout de l'axe des lignes.
Elle annon&ccedil;ait 68&nbsp;mm au-dessus du condyle, dont 50 gard&eacute;s, et j'en avais
conclu que le crop coupait la <b>poche sus-patellaire</b> et donc l'&eacute;panchement.
C'&eacute;tait la distance vers le <em>bas</em>. Recalcul&eacute;&nbsp;: <b>101,7&nbsp;mm
acquis au-dessus, 82,3 gard&eacute;s</b>, et 93&nbsp;% des &eacute;tudes conservent
&ge;&nbsp;70&nbsp;mm. <b>La poche est couverte, l'argument sur l'&eacute;panchement est
retir&eacute;.</b> Le trou m&eacute;dio-lat&eacute;ral ci-dessus, lui, ne d&eacute;pendait
pas du signe et tient.</p>
</div>

<div class="card">
<h2>2 &middot; Le kyste</h2>
<p>Un kyste poplit&eacute; n'est pas une tumeur&nbsp;: c'est du <b>liquide articulaire</b>
qui a fui par l'arri&egrave;re de la capsule et s'est accumul&eacute; dans une bourse d&eacute;j&agrave;
pr&eacute;sente, entre le tendon du <b>semi-membraneux</b> et le chef m&eacute;dial du
<b>gastrocn&eacute;mien</b>. Il communique avec l'articulation par un col &eacute;troit, ce
qui explique sa position constante&nbsp;: <b>post&eacute;rieur et m&eacute;dial</b>, jamais
ailleurs.</p>
<p>Cliniquement il est surtout un <em>t&eacute;moin</em>&nbsp;: il accompagne ce qui fait
produire du liquide &mdash; arthrose, l&eacute;sion m&eacute;niscale, synovite. C'est aussi
pour &ccedil;a qu'il est l'une des cibles les mieux pr&eacute;dites par le mod&egrave;le
large, et l'une des plus mal pr&eacute;dites par la comorbidit&eacute; seule.</p>
<table class="report">
<tr><th></th><th>positifs</th><th>comorbidit&eacute; seule</th><th>mod&egrave;le large</th><th>apport des pixels</th></tr>
<tr><td>Baker's</td><td>1098</td><td>0.6782</td><td>0.8781</td><td><b>+0.212</b></td></tr>
</table>
<p class="note">Le plus gros apport pixel des douze cibles, et la deuxi&egrave;me meilleure
AUC. Ce qui se voit se voit bien&nbsp;; la question est ce qui ne se voit pas.</p>
</div>

<div class="card">
<h2>3 &middot; O&ugrave; il est &mdash; mesur&eacute;, pas r&eacute;cit&eacute;</h2>
<p>Je ne sais pas d&eacute;signer un kyste sur ces images de fa&ccedil;on fiable, et une
bo&icirc;te plac&eacute;e d'apr&egrave;s un souvenir d'anatomie est une devinette
d&eacute;guis&eacute;e en mesure. Le kyste est donc localis&eacute; comme le reste de ce
projet localise les choses&nbsp;: en demandant <b>o&ugrave; les &eacute;tudes dont le
compte rendu mentionne un kyste sont plus brillantes</b> que celles dont il n'en parle
pas.</p>
<p>Le liquide est ce qu'il y a de plus brillant sur une image fluide-sensible satur&eacute;e
en graisse, donc la statistique est la fr&eacute;quence des voxels brillants, seuil&eacute;e
<b>par &eacute;tude contre sa propre distribution</b> pour que le gain du scanner
dispara&icirc;sse. {n_pos} positifs contre {n_neg} n&eacute;gatifs.</p>
<figure style="margin:14px 0"><img src="{map_img}" style="width:100%;border-radius:8px">
<figcaption class="note">Rouge&nbsp;: plus souvent brillant chez les positifs. Vert&nbsp;:
l'image moyenne des n&eacute;gatifs, pour le rep&egrave;re anatomique. Le pic est
&agrave; <b>{peak_ap:+.0f}&nbsp;mm en arri&egrave;re</b> et <b>{peak_si:+.0f}&nbsp;mm
au-dessus</b> de <code>med_centre</code>. La bande rouge verticale derri&egrave;re le
condyle est le kyste&nbsp;; l'arc rouge qui &eacute;pouse le contour post&eacute;rieur est
probablement le r&eacute;cessus articulaire, c'est-&agrave;-dire l'&eacute;panchement qui
l'accompagne. La bo&icirc;te blanche est celle propos&eacute;e plus bas.</figcaption></figure>
<p><b>Le point &agrave; retenir&nbsp;: le pic est au-<em>dessus</em> du rep&egrave;re.</b>
Mon premier jet descendait la bo&icirc;te de 12&nbsp;mm, par analogie avec le ligament
collat&eacute;ral. C'&eacute;tait l'inverse de ce que dit la mesure.</p>
<h3>Et &agrave; quelle profondeur</h3>
<figure style="margin:14px 0"><img src="{prof_img}" style="width:100%;border-radius:8px">
<figcaption class="note">L'exc&egrave;s de brillance dans la moiti&eacute;
post&eacute;rieure, coupe par coupe, en millim&egrave;tres en dedans du rep&egrave;re. Il
culmine <b>sur la coupe du rep&egrave;re lui-m&ecirc;me</b>, reste net jusqu'&agrave;
+20&nbsp;mm et meurt &agrave; +24. En vert ce que la bande du mod&egrave;le large
conserve, en bleu ce qu'elle jette.</figcaption></figure>
<p>La fen&ecirc;tre <b>[&minus;8, +20]&nbsp;mm</b> contient <b>88&nbsp;%</b> de
l'exc&egrave;s total&nbsp;; [&minus;8, +12] n'en contient que 72&nbsp;%. &Agrave;
3,3&nbsp;mm d'espacement m&eacute;dian, cela fait <b>9 coupes</b>.</p>
</div>

<div class="card">
<h2>4 &middot; La preuve que la bo&icirc;te contient quelque chose</h2>
<p>Avant d'entra&icirc;ner quoi que ce soit&nbsp;: un classifieur trivial, sans aucun
apprentissage &mdash; <b>la fraction de voxels brillants dans la bo&icirc;te</b>, un seul
nombre par &eacute;tude, compar&eacute; au label.</p>
<table class="report">
<tr><th>un seul nombre compt&eacute; dans&hellip;</th><th>AUC</th></tr>
<tr><td>la bo&icirc;te propos&eacute;e, fen&ecirc;tre [&minus;8, +20] mm</td><td><b>0.6847</b></td></tr>
<tr class="danger"><td>la m&ecirc;me bo&icirc;te, <b>coupes gard&eacute;es par la bande</b></td><td><b>0.5500</b></td></tr>
<tr><td>la coupe enti&egrave;re, m&ecirc;mes coupes</td><td>0.6323</td></tr>
</table>
<p>C'est le r&eacute;sultat qui justifie toute la page. Un comptage de pixels dans la bonne
r&eacute;gion vaut <b>0.68</b>&nbsp;; le m&ecirc;me comptage limit&eacute; aux coupes que le
mod&egrave;le large regarde vaut <b>0.55</b>, soit presque le hasard. <b>Le signal est dans
les coupes jet&eacute;es.</b> Et la bo&icirc;te bat la coupe enti&egrave;re (0.63), donc
elle ne se contente pas de compter du liquide n'importe o&ugrave;.</p>
<p class="note">0.68 n'est pas un r&eacute;sultat d'expert&nbsp;: c'est un plancher, obtenu
sans rien apprendre. Il dit que la r&eacute;gion porte du signal, pas combien un encodeur
en tirera.</p>
</div>

<div class="card">
<h2>5 &middot; Quelles s&eacute;ries</h2>
<p><b>Sagittale PD fat-sat en premier, et c'est un choix physique et non statistique.</b>
Un kyste est du liquide&nbsp;: sur une s&eacute;quence fluide-sensible avec saturation de
la graisse, il est blanc sur un fond devenu sombre, et c'est la seule combinaison o&ugrave;
il ne peut pas &ecirc;tre confondu avec la graisse poplit&eacute;e, qui est abondante
exactement l&agrave;. La carte du &sect;3 a &eacute;t&eacute; faite sur ces
s&eacute;ries-l&agrave; pour cette raison.</p>
<table class="report">
<tr><th>s&eacute;rie</th><th>couverture</th><th>pourquoi</th></tr>
<tr><td>Sagittale PD fat-sat</td><td>81,3 %</td><td>le liquide est blanc, la graisse noire</td></tr>
<tr><td>Sagittale PD sans fat-sat</td><td>36,3 %</td><td>rattrapage de couverture&nbsp;; le kyste y reste visible par sa forme, moins par son signal</td></tr>
<tr><td><b>au moins une des deux</b></td><td><b>99,7 %</b></td><td></td></tr>
</table>
<h3>Les deux autres plans, et pourquoi je ne les prends pas</h3>
<p>Mesur&eacute; en projetant le centre de l'exc&egrave;s dans chaque pile&nbsp;:</p>
<table class="report">
<tr><th>plan</th><th>position du kyste dans la pile</th><th>hors bande [0,20&ndash;0,80]</th></tr>
<tr><td>Sagittal</td><td>0,37</td><td>16,8 %</td></tr>
<tr class="danger"><td>Coronal</td><td>0,78</td><td><b>43,3 %</b></td></tr>
<tr><td>Axial</td><td>0,50</td><td><b>0,6 %</b></td></tr>
</table>
<p><b>L'axial est d&eacute;j&agrave; vu.</b> Le kyste y tombe au centre exact de la pile,
hors bande dans 0,6&nbsp;% des cas, et la couverture axiale fluide fat-sat est de
100&nbsp;%. C'est le plan o&ugrave; un kyste est le plus caract&eacute;ristique &mdash; une
collection ronde post&eacute;ro-m&eacute;diale &mdash; et c'est pr&eacute;cis&eacute;ment
pour &ccedil;a que le mod&egrave;le large s'en sort d&eacute;j&agrave; &agrave; 0.8781.
Un expert axial reverrait ce qui est d&eacute;j&agrave; vu.</p>
<p><b>Le coronal est coup&eacute; aussi</b>, et plus souvent que le sagittal&nbsp;: sa
profondeur est ant&eacute;ro-post&eacute;rieure et le kyste est post&eacute;rieur, donc il
tombe au bord de la bande. Il serait le bon deuxi&egrave;me spec &mdash; mais il demande
un champ que <code>RoiSpec</code> n'a pas&nbsp;: la fen&ecirc;tre de profondeur y est
centr&eacute;e sur le rep&egrave;re, alors que le kyste est 22&nbsp;mm derri&egrave;re, et
<code>bowtie_direction</code> ne peut pas servir &agrave; la d&eacute;caler &mdash; il
r&eacute;pond &agrave; une question sur l'axe m&eacute;dio-lat&eacute;ral. Il faudrait un
<code>depth_offset_mm</code>. <b>Je le laisse pour un second temps</b>, quand le sagittal
aura dit si l'id&eacute;e tient.</p>
</div>

<div class="card">
<h2>6 &middot; Le ROI propos&eacute;</h2>
<p>La taille a &eacute;t&eacute; choisie sur un crit&egrave;re et un seul&nbsp;: la
<b>part de l'exc&egrave;s mesur&eacute; qu'elle contient</b>. Chaque candidate garde des
pixels isotropes et des c&ocirc;t&eacute;s divisibles par 14, faute de quoi le patch
embedding en jette une bande sans rien dire.</p>
<table class="report">
<tr><th>bo&icirc;te</th><th>sortie</th><th>mm/px</th><th>centre optimal</th><th>exc&egrave;s captur&eacute;</th></tr>
<tr><td>64 &times; 56 mm</td><td>224 &times; 196</td><td>0,286</td><td>+36 arr, +16 ht</td><td>52,0 %</td></tr>
<tr><td>80 &times; 70 mm</td><td>224 &times; 196</td><td>0,357</td><td>+22 arr, +20 ht</td><td>64,1 %</td></tr>
<tr class="good"><td><b>90 &times; 75 mm</b></td><td><b>252 &times; 210</b></td><td><b>0,357</b></td><td><b>+16 arr, +20 ht</b></td><td><b>71,0 %</b></td></tr>
<tr><td>96 &times; 84 mm</td><td>224 &times; 196</td><td>0,429</td><td>+16 arr, +18 ht</td><td>78,5 %</td></tr>
</table>
<p>Je prends <b>90 &times; 75&nbsp;mm</b>. Plus grand capture encore davantage, mais
96&nbsp;&times;&nbsp;84 passe &agrave; 0,429&nbsp;mm/px &mdash; <em>plus grossier que le
mod&egrave;le large</em> &mdash; et sa bordure post&eacute;rieure sort de l'image sur
quelques pour cent des &eacute;tudes. &Agrave; 0,357 on reste plus fin que le large tout en
couvrant 71&nbsp;% de l'exc&egrave;s.</p>
<p>La bo&icirc;te est <b>tol&eacute;rante</b>, ce qui compte quand le rep&egrave;re viendra
d'un mod&egrave;le et non d'un clic&nbsp;: la d&eacute;placer de 6&nbsp;mm dans n'importe
quel sens co&ucirc;te moins de 2 points de capture.</p>
<figure style="margin:14px 0"><img src="{box_img}" style="width:100%;border-radius:8px">
<figcaption class="note">{box_cap}</figcaption></figure>
<figure style="margin:14px 0"><img src="{crop_img}" style="width:100%;border-radius:8px">
<figcaption class="note">Le crop lui-m&ecirc;me, travers&eacute; de &minus;8 &agrave;
+20&nbsp;mm vers le m&eacute;dial &mdash; les 9 coupes du tenseur. En vert les coupes que
le mod&egrave;le large voit d&eacute;j&agrave;, en bleu les cinq qu'il jette et qui portent
la moiti&eacute; du signal.</figcaption></figure>
<p>Le spec, dans les termes que la machinerie existante comprend d&eacute;j&agrave;
&mdash; aucun nouveau champ, aucune nouvelle annotation, le rep&egrave;re est celui du
m&eacute;nisque m&eacute;dial&nbsp;:</p>
<pre class="code">"baker": RoiSpec(
    name="baker", landmark="med_centre", plane="Sagittal",
    box_w_mm=90.0, box_h_mm=75.0, out_w=252, out_h=210,
    box_offset_mm=16.0,      # +16 mm vers l'arriere (colonnes sagittales = posterieur)
    box_rise_mm=20.0,        # +20 mm vers le haut   (lignes sagittales = inferieur)
    lateral_mm=20.0,         # vers la peripherie, que bowtie_direction trouve = medial
    medial_mm=8.0,           # vers l'echancrure
    slots=9,
    series=(("Sagittal", "PD", True), ("Sagittal", "PD", False))),</pre>
<p class="note">Les deux noms de profondeur sont h&eacute;rit&eacute;s du m&eacute;nisque
et trompeurs ici&nbsp;: <code>lateral_mm</code> est la distance vers le bout de pile le plus
proche, que <code>bowtie_direction</code> identifie seul. Depuis <code>med_centre</code>,
ce bout est le bord m&eacute;dial &mdash; donc c'est bien lui qui ouvre la fen&ecirc;tre
vers les coupes jet&eacute;es.</p>
</div>

<div class="card">
<h2>7 &middot; Ce qu'il faut en attendre</h2>
<p>Dit avant le run, pour que &ccedil;a compte. <b>Je ne m'attends pas &agrave; ce que cet
expert batte le mod&egrave;le large en solo</b>&nbsp;: 0.8781 est la deuxi&egrave;me
meilleure AUC des douze, l'axial voit d&eacute;j&agrave; le kyste, et le plancher
mesur&eacute; dans la bo&icirc;te est &agrave; 0.68.</p>
<p>Ce que j'attends, c'est un <b>compl&eacute;ment</b>. Les quatre experts d&eacute;j&agrave;
entra&icirc;n&eacute;s gagnent tous en fusion de rangs avec le large, y compris les deux que
j'avais class&eacute;s en &eacute;chec parce qu'ils perdaient en solo&nbsp;:</p>
<table class="report">
<tr><th>cible</th><th>large</th><th>expert seul</th><th>fusion 50/50</th><th>&eacute;cart</th></tr>
<tr><td>M&eacute;nisque lat&eacute;ral</td><td>0.7868</td><td>0.8476</td><td>0.8518</td><td><b>+0.065</b></td></tr>
<tr><td>Arthrose f&eacute;moro-patellaire</td><td>0.8027</td><td>0.8305</td><td>0.8493</td><td><b>+0.047</b></td></tr>
<tr><td>LCM</td><td>0.7948</td><td>0.8042</td><td>0.8306</td><td><b>+0.036</b></td></tr>
<tr><td>Arthrose lat&eacute;rale</td><td>0.8098</td><td>0.7923</td><td>0.8258</td><td><b>+0.016</b></td></tr>
</table>
<p>La corr&eacute;lation de rangs entre expert et large ne d&eacute;passe jamais 0,65, et
c'est de l&agrave; que vient le gain. Ici le d&eacute;saccord devrait &ecirc;tre encore plus
franc, puisque l'expert regarderait litt&eacute;ralement des coupes que l'autre n'a jamais
vues.</p>
<p class="note">Le risque, sym&eacute;trique&nbsp;: si le kyste est d&eacute;j&agrave;
enti&egrave;rement lu sur l'axial, les coupes m&eacute;diales sagittales ne diront rien de
neuf et la fusion n'apportera rien. C'est la pr&eacute;diction que le run
tranchera.</p>
</div>

<div class="card">
<h2>8 &middot; Faire d&eacute;filer des piles</h2>
<p class="note">&Eacute;tiquettes issues de l'extraction des comptes rendus, pas d'une
lecture de ces pixels &mdash; et pas par un radiologue. Scrollez vers le m&eacute;dial
(un seul bout de la pile) et regardez derri&egrave;re le condyle, un peu au-dessus de
l'interligne.</p>"""]

    n = 0
    for uid in pos_u[:args.cases]:
        body.append('<h3 class="case-head">le compte rendu mentionne un kyste</h3>')
        body.append(case(uid, f"bk{n}", "Le compte rendu le mentionne."))
        n += 1
        print(f"  positive {uid[-11:]}", flush=True)
    for uid in neg_u[:args.cases]:
        body.append('<h3 class="case-head">le compte rendu n\'en parle pas</h3>')
        body.append(case(uid, f"bk{n}", "Le compte rendu n'en parle pas."))
        n += 1
        print(f"  negative {uid[-11:]}", flush=True)
    body.append("</div>")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(P.page("Kyste poplité — le kyste de Baker", "".join(body)))
    print(f"\n{args.out}  ({args.out.stat().st_size/1e6:.1f} MB)")
    print(f"capture de la boite retenue : {100*cap:.1f} %")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
