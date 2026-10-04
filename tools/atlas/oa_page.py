"""Tibiofemoral osteoarthritis, both compartments, and the region it is read in.

    python -m tools.atlas.oa_page -o docs/atlas/tibiofemoral_oa.html

Written after two experts on this disease rather than before: Lateral OA was tried on
the meniscus crops and lost. The page says what the region should have been, why the
one that was reused was wrong for it, and why the honest expectation is still low.

Nothing here is a radiological reading; the examples carry report-extracted labels.
"""

from __future__ import annotations

import argparse
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
from rsna.dicom.ordering import order_slices                         # noqa: E402
from rsna.roi import SPECS                                           # noqa: E402
from tools.atlas import page as P                                    # noqa: E402
from tools.atlas.render import stack_to_jpegs, to_jpeg, window       # noqa: E402
from tools.atlas.study import (TRAIN_SERIES, load_series,            # noqa: E402
                               series_headers, side_of, stack_orientation)

LABELS = ROOT / "data/external/pilkwang-rsna-knee-llm-labels/report_labels_v2.csv"
MED = ROOT / "data/manual_annotations/landmarks-med.csv"
LAT = ROOT / "data/manual_annotations/landmarks.csv"


def _text(img, s, x, y, colour, scale=0.5):
    cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4)
    cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 1)


def coronal_at(study: str, point: np.ndarray):
    headers = series_headers(study)
    cor = headers[headers["plane"] == "Coronal"]
    if not len(cor):
        return None
    row = cor.sort_values(["fatsat", "n_slices"], ascending=[False, False]).iloc[0]
    folder = Path(row["dir"])
    names, _ = order_slices(str(folder),
                            sorted(p.name for p in folder.glob("*.dcm")), Config())
    series = load_series(row, Config())
    head = pydicom.dcmread(str(folder / names[0]), force=True, stop_before_pixels=True)
    iop = [float(x) for x in head.ImageOrientationPatient]
    ps = [float(x) for x in head.PixelSpacing]
    n = normal_of(iop)
    t = np.array([through_plane([float(x) for x in pydicom.dcmread(
        str(folder / nm), force=True, stop_before_pixels=True).ImagePositionPatient], n)
        for nm in names])
    k = int(np.argmin(np.abs(t - through_plane(point, n))))
    ds = pydicom.dcmread(str(folder / names[k]), force=True, stop_before_pixels=True)
    return (window(series.volume)[k], [float(x) for x in ds.ImagePositionPatient],
            iop, ps, k, len(names), row)


def both_boxes_figure(study: str, pm: np.ndarray, pl: np.ndarray) -> tuple[str, str]:
    """Both compartments on one coronal slice, each with the box proposed for it."""

    got = coronal_at(study, (pm + pl) / 2)
    if got is None:
        raise SystemExit("no coronal series")
    img, ipp, iop, ps, k, n, row = got
    rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    s = SPECS["medial_oa"]
    for point, colour, label in ((pm, (120, 255, 120), "med_centre"),
                                 (pl, (255, 180, 0), "lat_centre")):
        rr, cc = pixel_of(ipp, iop, ps, point)
        hw, hh = s.box_w_mm / 2 / ps[1], s.box_h_mm / 2 / ps[0]
        cv2.rectangle(rgb, (int(cc - hw), int(rr - hh)), (int(cc + hw), int(rr + hh)),
                      colour, 2)
        cv2.drawMarker(rgb, (int(cc), int(rr)), colour, cv2.MARKER_CROSS, 32, 2)
        _text(rgb, label, int(cc - 44), int(rr - hh) - 8, colour, 0.52)
    big = cv2.resize(rgb, (660, int(660 * img.shape[0] / img.shape[1])),
                     interpolation=cv2.INTER_CUBIC)
    cap = (f"&hellip;{P._esc(study[-11:])} &middot; coronale "
           f"{P._esc(str(row['weight']))}{'FS' if row['fatsat'] else ''} &middot; "
           f"coupe {k}/{n} &middot; {ps[1]:.3f} mm/px")
    return to_jpeg(big, max_width=660)[0], cap


def planes_figure(study: str, point: np.ndarray) -> str:
    """The meniscus box in its own plane, and the proposed one in the other."""

    panels = []
    for plane, spec, colour, label in (
            ("Sagittal", SPECS["medial_meniscus"], (0, 209, 255),
             "SAGITTAL - le ROI du menisque"),
            ("Coronal", SPECS["medial_oa"], (120, 255, 120),
             "CORONAL - le ROI propose")):
        headers = series_headers(study)
        g = headers[headers["plane"] == plane]
        asc = [True, False] if plane == "Sagittal" else [False, False]
        row = g.sort_values(["fatsat", "n_slices"], ascending=asc).iloc[0]
        folder = Path(row["dir"])
        names, _ = order_slices(str(folder),
                                sorted(p.name for p in folder.glob("*.dcm")), Config())
        series = load_series(row, Config())
        head = pydicom.dcmread(str(folder / names[0]), force=True,
                               stop_before_pixels=True)
        iop = [float(x) for x in head.ImageOrientationPatient]
        ps = [float(x) for x in head.PixelSpacing]
        n = normal_of(iop)
        t = np.array([through_plane([float(x) for x in pydicom.dcmread(
            str(folder / nm), force=True,
            stop_before_pixels=True).ImagePositionPatient], n) for nm in names])
        k = int(np.argmin(np.abs(t - through_plane(point, n))))
        ds = pydicom.dcmread(str(folder / names[k]), force=True,
                             stop_before_pixels=True)
        img = window(series.volume)[k]
        rr, cc = pixel_of([float(x) for x in ds.ImagePositionPatient], iop, ps, point)
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        hw, hh = spec.box_w_mm / 2 / ps[1], spec.box_h_mm / 2 / ps[0]
        cv2.rectangle(rgb, (int(cc - hw), int(rr - hh)), (int(cc + hw), int(rr + hh)),
                      colour, 2)
        side = int(1.7 * max(spec.box_w_mm / ps[1], spec.box_h_mm / ps[0]))
        y0, x0 = int(rr - side // 2), int(cc - side // 2)
        view = np.zeros((side, side, 3), np.uint8)
        sy0, sx0 = max(y0, 0), max(x0, 0)
        sy1, sx1 = min(y0 + side, rgb.shape[0]), min(x0 + side, rgb.shape[1])
        view[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = rgb[sy0:sy1, sx0:sx1]
        view = cv2.resize(view, (430, 430), interpolation=cv2.INTER_CUBIC)
        _text(view, label, 10, 26, colour, 0.58)
        _text(view, f"{spec.box_w_mm:.0f}x{spec.box_h_mm:.0f} mm", 10, 48, colour, 0.5)
        panels.append(view)
    return to_jpeg(np.hstack(panels), max_width=880)[0]


def case(uid: str, vid: str, note: str) -> str:
    headers = series_headers(uid)
    cor = headers[headers["plane"] == "Coronal"]
    if not len(cor):
        return ""
    row = cor.sort_values(["fatsat", "n_slices"], ascending=[False, False]).iloc[0]
    series = load_series(row, Config())
    side, how = side_of(headers)
    first, last = stack_orientation("Coronal", side)
    frames, _ = stack_to_jpegs(series.volume)
    med = "droite de l'image" if side == "R" else "gauche de l'image"
    caption = (f"{P._esc(series.label)} &middot; {series.n} coupes &middot; "
               f"{series.mm_per_px:.2f} mm/px &middot; genou {side or '?'} &mdash; "
               f"le <b>m&eacute;dial est la {med}</b>. {note}")
    return (f'<h3>&hellip;{P._esc(uid[-11:])}</h3>'
            + P.viewer(vid, frames, first, last, series.mm_per_px, caption=caption))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--out", type=Path,
                    default=ROOT / "docs/atlas/tibiofemoral_oa.html")
    ap.add_argument("--cases", type=int, default=2)
    args = ap.parse_args()

    med = pd.read_csv(MED).set_index("study")
    lat = pd.read_csv(LAT)
    lat = lat[lat.point.notna()].set_index("study")
    shared = [s for s in med.index if s in lat.index]
    pick = shared[3]
    pm = np.array([med.loc[pick].x_mm, med.loc[pick].y_mm, med.loc[pick].z_mm])
    pl = np.array([lat.loc[pick].x_mm, lat.loc[pick].y_mm, lat.loc[pick].z_mm])
    boxes, boxes_cap = both_boxes_figure(pick, pm, pl)
    planes = planes_figure(pick, pm)
    gap = abs(pm[0] - pl[0])

    lab = pd.read_csv(LABELS)
    pos = [u for u in lab[lab["Medial OA"] > 0.5]["StudyInstanceUID"]
           if (TRAIN_SERIES / u).is_dir()]
    neg = [u for u in lab[lab["Medial OA"] < 0.1]["StudyInstanceUID"]
           if (TRAIN_SERIES / u).is_dir()]
    s = SPECS["medial_oa"]

    body = [f"""
<header>
  <h1>Arthrose f&eacute;moro-tibiale &mdash; les deux compartiments</h1>
  <p class="lede">La seule maladie de la liste sur laquelle un expert a d&eacute;j&agrave;
  &eacute;chou&eacute;. Cette page dit ce que c'est, pourquoi la r&eacute;gion qu'on lui
  avait donn&eacute;e &eacute;tait la mauvaise, celle qu'il faudrait &mdash; et pourquoi
  il faut malgr&eacute; tout en attendre peu.</p>
</header>

<div class="card">
<h2>1 &middot; Ce qu'on cherche</h2>
<p>L'arthrose use le cartilage entre le f&eacute;mur et le tibia, dans un compartiment ou
dans l'autre. Quatre signes, et ils ne sont pas au m&ecirc;me endroit&nbsp;:</p>
<ul>
<li><b>Pincement de l'interligne</b> &mdash; l'espace entre les deux os se r&eacute;duit.
    C'est une <em>hauteur</em>, donc &ccedil;a se lit de face.</li>
<li><b>Ost&eacute;ophytes</b> &mdash; de l'os qui pousse <em>aux berges</em> de
    l'articulation, pas en son centre.</li>
<li><b>&OElig;d&egrave;me sous-chondral</b> &mdash; dans l'os, de part et d'autre de
    l'interligne, visible seulement en suppression de graisse.</li>
<li><b>Scl&eacute;rose</b> &mdash; l'os sous le cartilage qui se densifie et s'assombrit.</li>
</ul>
<p>Deux de ces quatre sont <b>dans l'os</b> et un est <b>aux marges</b>. Une bo&icirc;te
qui serre l'interligne les rate.</p>
</div>

<div class="card">
<h2>2 &middot; Pourquoi la r&eacute;gion r&eacute;utilis&eacute;e &eacute;tait la mauvaise</h2>
<p>Lateral OA a &eacute;t&eacute; entra&icirc;n&eacute;e sur les crops du
<b>m&eacute;nisque</b>, parce qu'ils existaient d&eacute;j&agrave;. Deux choses ne vont
pas, et aucune n'est une question de taille&nbsp;:</p>
<p><b>Le plan.</b> Le m&eacute;nisque se lit en sagittal, l'arthrose de face. Un pincement
est la hauteur d'un espace&nbsp;; une coupe sagittale la traverse au lieu de la montrer,
et les berges o&ugrave; poussent les ost&eacute;ophytes y sont coup&eacute;es en travers.</p>
<p><b>Le d&eacute;calage.</b> La coronale du m&eacute;nisque est pouss&eacute;e de 6&nbsp;mm
vers la p&eacute;riph&eacute;rie, parce qu'elle cherche l'<em>extrusion</em> &mdash; le
corps m&eacute;niscal d&eacute;plac&eacute; au-del&agrave; de la berge tibiale. L'arthrose
n'est pas &agrave; la berge mais en travers du compartiment.</p>
<figure style="margin:14px 0"><img src="{planes}" style="width:100%;max-width:880px;border-radius:8px">
<figcaption class="note">Le m&ecirc;me point, dans les deux plans. &Agrave; gauche la
bo&icirc;te du m&eacute;nisque, qui cadre l'interligne de profil. &Agrave; droite celle
qu'on propose, qui prend l'interligne de face avec de l'os au-dessus et en dessous.</figcaption></figure>
<p class="note">Attention cependant&nbsp;: le run qui a &eacute;chou&eacute; <b>avait
d&eacute;j&agrave; une coronale</b> &mdash; celle du m&eacute;nisque. Le plan seul
n'explique donc pas l'&eacute;chec, et il serait malhonn&ecirc;te de le pr&eacute;senter
comme la cause.</p>
</div>

<div class="card">
<h2>3 &middot; La r&eacute;gion propos&eacute;e, et elle ne co&ucirc;te aucune annotation</h2>
<p>Il n'y a pas de nouveau rep&egrave;re &agrave; collecter. Les deux points
m&eacute;niscaux, projet&eacute;s sur une coupe coronale, tombent <b>un dans chaque
compartiment</b> &mdash; {gap:.0f}&nbsp;mm l'un de l'autre sur cette &eacute;tude.</p>
<figure style="margin:14px 0"><img src="{boxes}" style="width:100%;max-width:660px;border-radius:8px">
<figcaption class="note">{boxes_cap}. Une bo&icirc;te de
{s.box_w_mm:.0f}&times;{s.box_h_mm:.0f}&nbsp;mm autour de chaque point cadre son
interligne, l'os au-dessus et en dessous, et la berge externe.</figcaption></figure>
<table class="report">
<tr><th></th><th>m&eacute;nisque coronal</th><th>arthrose (propos&eacute;)</th></tr>
<tr><td>bo&icirc;te</td><td>48&times;30 mm</td><td><b>{s.box_w_mm:.0f}&times;{s.box_h_mm:.0f} mm</b></td></tr>
<tr><td>d&eacute;calage</td><td>6 mm vers la berge</td><td><b>aucun</b></td></tr>
<tr><td>sortie</td><td>224&times;140 px</td><td>{s.out_w}&times;{s.out_h} px</td></tr>
<tr><td>r&eacute;solution</td><td>0,214 mm/px</td><td>{s.mm_per_px:.4f} mm/px</td></tr>
<tr><td>profondeur</td><td>14/14 mm, 9 slots</td><td>{s.lateral_mm:.0f}/{s.medial_mm:.0f} mm, {s.slots} slots</td></tr>
</table>
<p>Plus haute de 9&nbsp;mm et recentr&eacute;e&nbsp;: l'&oelig;d&egrave;me et la
scl&eacute;rose sont <em>dans</em> l'os des deux c&ocirc;t&eacute;s du pincement, et les
rater revient &agrave; ne voir que deux des quatre signes. Profondeur
sym&eacute;trique&nbsp;: l'arthrose s'&eacute;value en travers du compartiment, il n'y a
pas de bowtie vers lequel &ecirc;tre court.</p>
</div>

<div class="card">
<h2>4 &middot; Ce qu'il faut en attendre</h2>
<p>Peu, et la raison n'est pas la r&eacute;gion.</p>
<table class="report">
<tr><th>cible</th><th>comorbidit&eacute; seule</th><th>mod&egrave;le large</th><th>apport des pixels</th></tr>
<tr class="danger"><td>Lateral OA <span class="tag">&eacute;chou&eacute;</span></td><td>0.8575</td><td>0.8471</td><td><b>&minus;0.011</b></td></tr>
<tr><td>Medial OA</td><td>0.8613</td><td>0.8821</td><td><b>+0.021</b></td></tr>
<tr><td>PF OA <span class="tag">r&eacute;ussi</span></td><td>0.7916</td><td>0.8093</td><td>+0.018</td></tr>
</table>
<p>Lateral OA a un apport <b>n&eacute;gatif</b>&nbsp;: le mod&egrave;le large y fait moins
bien que les onze autres labels seuls. Son score <em>est</em> de la comorbidit&eacute;, et
un expert qui ne voit que des pixels ne peut pas le rattraper &mdash; il a perdu de
0,028 avec un intervalle qui exclut z&eacute;ro.</p>
<p>Medial OA est &agrave; <b>+0,021</b>, donc positif. C'est le m&ecirc;me ordre que
PF OA, qui avait le plus faible apport des douze (+0,018) et dont l'expert a quand
m&ecirc;me battu le mod&egrave;le large de +0,021 &mdash; parce qu'il extrait ce
signal-l&agrave; mieux que lui. C'est tout l'espoir qu'on peut avoir ici, et il est
mince.</p>
<p class="note">Et le vrai gain pour ces deux cibles viendra d'ailleurs&nbsp;: d'un
m&eacute;ta-mod&egrave;le qui combine les sorties des experts avec les onze autres labels.
C'est l&agrave; qu'est leur signal.</p>
</div>

<div class="card">
<h2>5 &middot; Faire d&eacute;filer des piles</h2>
<p class="note">&Eacute;tiquettes issues de l'extraction des comptes rendus, pas d'une
lecture de ces pixels &mdash; et pas par un radiologue.</p>"""]

    n = 0
    for uid in pos[:args.cases]:
        body.append('<h3 class="case-head">le compte rendu mentionne une arthrose m&eacute;diale</h3>')
        body.append(case(uid, f"oa{n}", "Le compte rendu la mentionne."))
        n += 1
        print(f"  positive {uid[-11:]}", flush=True)
    for uid in neg[:args.cases]:
        body.append('<h3 class="case-head">le compte rendu ne la mentionne pas</h3>')
        body.append(case(uid, f"oa{n}", "Le compte rendu n'en parle pas."))
        n += 1
        print(f"  negative {uid[-11:]}", flush=True)
    body.append("</div>")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(P.page("Arthrose fémoro-tibiale", "".join(body)))
    print(f"\n{args.out}  ({args.out.stat().st_size/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
