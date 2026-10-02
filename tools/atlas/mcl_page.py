"""What a medial collateral ligament injury is, where it lives, and what it would cost.

    python -m tools.atlas.mcl_page -o docs/atlas/mcl.html

Written before any annotation, like the patellofemoral page, and for the same reason: a
landmark whose definition lives in the annotator's head gets two annotators' readings
averaged. It carries one extra thing the patellofemoral page did not need -- an attempt
to place the region of interest **for free**, from landmarks this project already has,
and the measurement of why that attempt fails.

Nothing here is a radiological reading. The positive and negative examples are labelled
from the report extraction, which is a weak label, and they illustrate what the text
said rather than a diagnosis made from the pixels.
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

from rsna.config import Config                                          # noqa: E402
from rsna.dicom.geometry import _axes, normal_of, pixel_of, through_plane  # noqa: E402
from rsna.dicom.ordering import order_slices                            # noqa: E402
from tools.atlas import page as P                                       # noqa: E402
from tools.atlas.render import stack_to_jpegs, to_jpeg, window          # noqa: E402
from tools.atlas.study import (TRAIN_SERIES, load_series,               # noqa: E402
                               series_headers, side_of, stack_orientation)

LABELS = ROOT / "data/external/pilkwang-rsna-knee-llm-labels/report_labels_v2.csv"
LANDMARKS = ROOT / "out/landmarks/train_series.csv"

#: The study the figure is drawn on, a left knee with no MCL mention in its report, and
#: the medial half of its middle slice. The marker coordinates are in the zoomed crop's
#: pixels, read off a printed grid, so they mean nothing without these three numbers.
FIGURE_STUDY = "09909132231"
CROP = (0.18, 0.88, 0.0, 0.55)      # row0, row1, col0, col1 as fractions
ZOOM = 2
POINT = (234, 297)                  # against the medial cortex, at the joint line
BOX_MM = (30.0, 70.0)


def _text(img, s, x, y, colour):
    cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 5)
    cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, colour, 1)


def anatomy_figure(series, with_point: bool) -> str:
    """The medial side of the joint, named — and where the point goes."""

    img = window(series.volume)[series.volume.shape[0] // 2]
    h, w = img.shape
    r0, r1, c0, c1 = CROP
    crop = img[int(h * r0):int(h * r1), int(w * c0):int(w * c1)]
    big = cv2.cvtColor(cv2.resize(crop, (crop.shape[1] * ZOOM, crop.shape[0] * ZOOM),
                                  interpolation=cv2.INTER_CUBIC), cv2.COLOR_GRAY2BGR)
    _text(big, "superieur", 12, 26, (210, 210, 255))
    _text(big, "MEDIAL (genou gauche)", 12, big.shape[0] - 14, (150, 255, 150))
    _text(big, "femur", 300, 180, (255, 230, 120))
    _text(big, "tibia", 300, 430, (255, 230, 120))
    _text(big, "interligne", 330, 318, (120, 255, 255))
    _text(big, "peau", 120, 120, (200, 200, 200))
    if with_point:
        cx, cy = POINT
        px = lambda mm: int(round(mm / series.mm_per_px * ZOOM))      # noqa: E731
        cv2.rectangle(big, (cx - px(BOX_MM[0]) // 2, cy - px(BOX_MM[1]) // 2),
                      (cx + px(BOX_MM[0]) // 2, cy + px(BOX_MM[1]) // 2),
                      (0, 200, 255), 2)
        cv2.drawMarker(big, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 30, 2)
        _text(big, "contre l'os,", cx - 150, cy - 26, (120, 120, 255))
        _text(big, "pas dans la graisse", cx - 150, cy - 8, (120, 120, 255))
    return to_jpeg(big, max_width=470)[0]


def coronal_of(headers: pd.DataFrame):
    """The coronal series a collateral ligament is read on: fat-suppressed first.

    An MCL sprain is oedema in and around the ligament, and oedema is only visible once
    the fat around it is suppressed. PD fat-suppressed covers 84.0 % of studies and T2
    fat-suppressed 17.0 %; together 95.5 %.
    """

    cor = headers[headers["plane"] == "Coronal"]
    if not len(cor):
        return None
    for weight, fat in (("PD", True), ("T2", True), ("PD", False), ("T1", False)):
        hit = cor[(cor["weight"] == weight) & (cor["fatsat"].astype(bool) == fat)]
        if len(hit):
            return hit.sort_values("n_slices", ascending=False).iloc[0]
    return cor.sort_values("n_slices", ascending=False).iloc[0]


def geometry_at(uid: str, point: np.ndarray):
    """The coronal slice nearest a patient point, with what is needed to draw on it."""

    headers = series_headers(uid)
    row = coronal_of(headers)
    if row is None:
        return None
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
    ipp = [float(x) for x in ds.ImagePositionPatient]
    return {"series": series, "img": window(series.volume)[k], "ipp": ipp, "iop": iop,
            "ps": ps, "k": k, "side": side_of(headers)[0], "n": len(names),
            "row": row, "names": names, "folder": folder}


def medial_edge(g, lateral_sign: float) -> float:
    """Patient x of the limb's medial skin surface on this slice."""

    img = g["img"]
    u, v = _axes(g["iop"])
    rr, cc = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    X = (np.asarray(g["ipp"])[0] + cc * g["ps"][1] * u[0] + rr * g["ps"][0] * v[0])
    bg, hi = np.percentile(img, 5), np.percentile(img, 99.5)
    m = img > bg + 0.10 * (hi - bg)
    xs = X[m]
    return float(xs.min() if lateral_sign > 0 else xs.max())


def derived_panel(uid: str, point: np.ndarray) -> tuple[str, str] | None:
    """The free placement, drawn: the existing landmark, and what it implies medially."""

    g = geometry_at(uid, point)
    if g is None or g["side"] is None:
        return None
    lateral_sign = float(np.sign(point[0]))
    edge = medial_edge(g, lateral_sign)
    guess = np.array([edge + lateral_sign * 10.0, point[1], point[2]])
    rgb = cv2.cvtColor(g["img"], cv2.COLOR_GRAY2BGR)
    lr, lc = pixel_of(g["ipp"], g["iop"], g["ps"], point)
    gr, gc = pixel_of(g["ipp"], g["iop"], g["ps"], guess)
    cv2.drawMarker(rgb, (int(lc), int(lr)), (255, 180, 0), cv2.MARKER_CROSS, 26, 2)
    cv2.drawMarker(rgb, (int(gc), int(gr)), (0, 0, 255), cv2.MARKER_CROSS, 30, 2)
    cv2.line(rgb, (int(lc), int(lr)), (int(gc), int(gr)), (90, 90, 90), 1)
    big = cv2.resize(rgb, (470, int(470 * rgb.shape[0] / rgb.shape[1])),
                     interpolation=cv2.INTER_CUBIC)
    cap = (f"&hellip;{P._esc(uid[-11:])} &middot; genou {g['side']} &middot; "
           f"largeur du membre {abs(edge - point[0]) * 100 // 1 / 100:.0f} mm "
           f"du rep&egrave;re lat&eacute;ral au bord m&eacute;dial")
    return to_jpeg(big, max_width=470)[0], cap


def case(uid: str, vid: str, note: str) -> str:
    headers = series_headers(uid)
    row = coronal_of(headers)
    if row is None:
        return ""
    series = load_series(row, Config())
    side, how = side_of(headers)
    first, last = stack_orientation("Coronal", side)
    frames, _ = stack_to_jpegs(series.volume)
    med = "droite de l'image" if side == "R" else "gauche de l'image"
    caption = (f"{P._esc(series.label)} &middot; {series.n} coupes &middot; "
               f"{series.mm_per_px:.2f} mm/px &middot; genou {side or '?'} "
               f"({P._esc(how)}) &mdash; <b>le m&eacute;dial est la {med}</b>, "
               f"le sup&eacute;rieur en haut. {note}")
    return (f'<h3>&hellip;{P._esc(uid[-11:])}</h3>'
            + P.viewer(vid, frames, first, last, series.mm_per_px, caption=caption))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--out", type=Path, default=ROOT / "docs/atlas/mcl.html")
    ap.add_argument("--cases", type=int, default=2)
    args = ap.parse_args()

    fig_uid = next(d.name for d in TRAIN_SERIES.iterdir()
                   if d.name.endswith(FIGURE_STUDY))
    fig_series = load_series(coronal_of(series_headers(fig_uid)), Config())
    plain = anatomy_figure(fig_series, with_point=False)
    marked = anatomy_figure(fig_series, with_point=True)

    lab = pd.read_csv(LABELS)
    lm = pd.read_csv(LANDMARKS).set_index("study")
    pos = [u for u in lab[lab["MCL"] > 0.5]["StudyInstanceUID"]
           if (TRAIN_SERIES / u).is_dir()]
    neg = [u for u in lab[lab["MCL"] < 0.1]["StudyInstanceUID"]
           if (TRAIN_SERIES / u).is_dir()]

    body = [f"""
<header>
  <h1>MCL &mdash; le ligament collat&eacute;ral m&eacute;dial</h1>
  <p class="lede">La bande qui tient le c&ocirc;t&eacute; interne du genou. C'est la
  cible sur laquelle le mod&egrave;le large est le <b>plus faible des douze</b>, et il y
  a du signal dans l'image &agrave; prendre. Cette page dit ce que c'est, o&ugrave; &ccedil;a
  se voit, et &mdash; contrairement &agrave; la page f&eacute;moro-patellaire &mdash;
  pourquoi le rep&egrave;re ne peut pas &ecirc;tre d&eacute;duit gratuitement de ce qu'on
  a d&eacute;j&agrave;.</p>
</header>

<div class="card">
<h2>1 &middot; Pourquoi cette cible</h2>
<p>Le bon crit&egrave;re n'est pas l'AUC du mod&egrave;le large mais ce que les
<b>pixels</b> apportent par-dessus les onze autres labels. Un expert qui ne voit qu'un
crop n'a acc&egrave;s qu'&agrave; cette part-l&agrave;.</p>
<table class="report">
<tr><th>cible</th><th>positifs</th><th>comorbidit&eacute; seule</th><th>mod&egrave;le large</th><th>apport des pixels</th></tr>
<tr><td>Baker's</td><td>447</td><td>0.6776</td><td>0.8750</td><td>+0.197</td></tr>
<tr><td>Medial Meniscus</td><td>704</td><td>0.7538</td><td>0.8785</td><td>+0.125</td></tr>
<tr class="danger"><td><b>MCL</b></td><td><b>282</b></td><td><b>0.7213</b></td>
    <td><b>0.8080</b></td><td><b>+0.087</b></td></tr>
<tr><td>Lateral Meniscus <span class="tag">fait</span></td><td>254</td><td>0.7655</td><td>0.8206</td><td>+0.055</td></tr>
<tr><td>PF OA <span class="tag">fait</span></td><td>810</td><td>0.7797</td><td>0.8093</td><td>+0.030</td></tr>
<tr><td>Lateral OA <span class="tag">&eacute;chou&eacute;</span></td><td>465</td><td>0.8347</td><td>0.8471</td><td>+0.012</td></tr>
</table>
<p class="note">0.8080 est la pire AUC des douze, la comorbidit&eacute; n'en explique que
0.7213, et le ligament est une structure <b>fine</b> &mdash; les trois conditions qui ont
fait marcher le m&eacute;nisque lat&eacute;ral. La r&eacute;serve est le nombre&nbsp;:
<b>282 positifs</b>, le plus bas des candidats s&eacute;rieux.</p>
</div>

<div class="card">
<h2>2 &middot; Ce qu'est le ligament</h2>
<p>Le LCM part de l'<b>&eacute;picondyle f&eacute;moral m&eacute;dial</b>, descend le long
du c&ocirc;t&eacute; interne en croisant l'interligne articulaire, et s'attache sur le
tibia <b>cinq &agrave; sept centim&egrave;tres plus bas</b>. Il a deux couches&nbsp;: une
superficielle, longue, et une profonde, courte, soud&eacute;e au m&eacute;nisque
m&eacute;dial. C'est une structure <b>longue et mince</b>, plaqu&eacute;e contre l'os
&mdash; ce qui dicte la forme de la bo&icirc;te&nbsp;: haute et &eacute;troite, l'inverse
de celles du m&eacute;nisque et de la rotule.</p>
<p>Une l&eacute;sion se lit comme un &oelig;d&egrave;me dans et autour du ligament, donc
en <b>hypersignal sur les s&eacute;quences &agrave; suppression de graisse</b>&nbsp;;
dans les formes compl&egrave;tes, la bande elle-m&ecirc;me est interrompue.</p>
<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:16px;margin-top:14px">
  <figure style="margin:0"><img src="{plain}" style="width:100%;border-radius:8px">
    <figcaption class="note">Le c&ocirc;t&eacute; m&eacute;dial d'un genou gauche, coupe
    coronale PD fat-sat. Ce que je nomme ici est ce que je peux identifier avec
    certitude&nbsp;: la peau, le f&eacute;mur, le tibia, l'interligne.</figcaption></figure>
  <figure style="margin:0"><img src="{marked}" style="width:100%;border-radius:8px">
    <figcaption class="note">Le point et une bo&icirc;te candidate de
    {BOX_MM[0]:.0f}&times;{BOX_MM[1]:.0f}&nbsp;mm &mdash; haute et &eacute;troite, comme
    le ligament. La position est donn&eacute;e par la <em>r&egrave;gle</em> (contre l'os,
    &agrave; hauteur de l'interligne), pas par une identification radiologique de ma
    part.</figcaption></figure>
</div>
</div>

<div class="card">
<h2>3 &middot; Quelles images</h2>
<p>Le plan coronal, celui qui d&eacute;roule le ligament sur toute sa longueur.
Mesur&eacute; sur les 4407 &eacute;tudes&nbsp;:</p>
<table class="report">
<tr><th>s&eacute;rie</th><th>des &eacute;tudes</th></tr>
<tr><td>COR PD fat-sat</td><td>84.0 %</td></tr>
<tr><td>COR T2 fat-sat</td><td>17.0 %</td></tr>
<tr><td><b>au moins une des deux</b></td><td><b>95.5 %</b></td></tr>
<tr><td>&hellip; en ajoutant AX PD fat-sat</td><td>96.8 %</td></tr>
<tr><td>COR T1 (pas d'&oelig;d&egrave;me visible)</td><td>64.1 %</td></tr>
</table>
<p>La suppression de graisse est voulue ici, comme pour l'arthrose f&eacute;moro-patellaire
et &agrave; l'inverse du m&eacute;nisque&nbsp;: l'&oelig;d&egrave;me n'appara&icirc;t
qu'une fois la graisse &eacute;teinte.</p>
<p><b>Orientation, mesur&eacute;e et non suppos&eacute;e.</b> Sur 3815 s&eacute;ries
coronales PD fat-sat, les colonnes vont vers la <b>gauche du patient</b> et les lignes
vers le <b>bas</b> &mdash; 3815 sur 3815. Donc le sup&eacute;rieur est en haut, et&nbsp;:</p>
<ul>
<li>sur un genou <b>droit</b>, le m&eacute;dial est la <b>droite</b> de l'image&nbsp;;</li>
<li>sur un genou <b>gauche</b>, le m&eacute;dial est la <b>gauche</b> de l'image.</li>
</ul>
<p class="note">Contrairement au point f&eacute;moro-patellaire, qui est sur la ligne
m&eacute;diane de son articulation, celui-ci est <b>lat&eacute;ralis&eacute; par
d&eacute;finition</b>. Le c&ocirc;t&eacute; du genou change donc o&ugrave; il se trouve,
et l'outil d'annotation devra le dire &mdash; comme il le fait pour le m&eacute;nisque.</p>
</div>"""]

    # -- the free placement, and why it does not work ------------------------ #
    panels = []
    for uid in pos:
        if uid not in lm.index or len(panels) >= 3:
            continue
        p = lm.loc[uid]
        got = derived_panel(uid, np.array([p.x_mm, p.y_mm, p.z_mm]))
        if got:
            panels.append(got)
            print(f"  derived {uid[-11:]}", flush=True)

    cells = "".join(f'<figure style="margin:0"><img src="{u}" style="width:100%;'
                    f'border-radius:6px"><figcaption class="note">{c}</figcaption>'
                    f'</figure>' for u, c in panels)
    body.append(f"""
<div class="card">
<h2>4 &middot; La r&eacute;gion d'int&eacute;r&ecirc;t &mdash; et pourquoi elle n'est pas gratuite</h2>
<p>L'id&eacute;e valait d'&ecirc;tre test&eacute;e&nbsp;: on a d&eacute;j&agrave; un
rep&egrave;re sur le m&eacute;nisque lat&eacute;ral, qui donne la <b>hauteur de
l'interligne</b>, et la silhouette du membre donne le <b>bord m&eacute;dial</b> de la
peau. Un point &agrave; dix millim&egrave;tres en dedans de ce bord, &agrave; la hauteur
de l'interligne, aurait plac&eacute; la bo&icirc;te sans aucune annotation.</p>
<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:14px;margin:14px 0">{cells}</div>
<p><span style="color:#ffb400">&#10010;</span> le rep&egrave;re m&eacute;niscal existant
&middot; <span style="color:#ff4040">&#10010;</span> le point que la r&egrave;gle en
d&eacute;duit.</p>
<p><b>&Ccedil;a ne marche pas</b>, et les trois images disent pourquoi&nbsp;: le point
tombe dans la <b>graisse sous-cutan&eacute;e</b>, alors que le ligament est plaqu&eacute;
contre l'os. L'&eacute;paisseur des tissus mous entre la peau et le ligament varie d'un
patient &agrave; l'autre &mdash; les trois membres ci-dessus mesurent 133, 155 et 158 mm
de large &mdash; et aucun d&eacute;calage fixe depuis la peau ne tombe juste sur tous.
La <b>hauteur</b>, elle, se transf&egrave;re bien&nbsp;: l'interligne est au m&ecirc;me
niveau des deux c&ocirc;t&eacute;s du genou.</p>
<p>Je ne peux pas non plus calibrer ce d&eacute;calage, puisqu'il n'existe aucune
v&eacute;rit&eacute; terrain pour le LCM dans ce corpus. <b>Il faut donc annoter.</b>
C'est un r&eacute;sultat utile et pas seulement une d&eacute;ception&nbsp;: &ccedil;a
&eacute;vite de construire un ROI sur une r&egrave;gle qui aurait eu l'air plausible et
aurait rat&eacute; le ligament sur une bonne partie des &eacute;tudes.</p>
<h3>Le point qu'il faudrait collecter</h3>
<p><b>Le ligament l&agrave; o&ugrave; il croise l'interligne articulaire</b>, sur la
coupe coronale o&ugrave; il est le plus net. C'est son milieu dans le sens de la
longueur, donc une bo&icirc;te centr&eacute;e dessus tient l'origine f&eacute;morale
au-dessus et l'insertion tibiale en dessous &mdash; et c'est un point que le genou sain
poss&egrave;de autant que le genou l&eacute;s&eacute;, ce qui est la r&egrave;gle qui
compte&nbsp;: si le rep&egrave;re d&eacute;rivait vers la l&eacute;sion, le crop
encoderait le label par sa position.</p>
<p>La bo&icirc;te serait <b>haute et &eacute;troite</b> &mdash; de l'ordre de 30 mm de
large sur 70 de haut, &agrave; confirmer sur une page de tailles comme pour les deux
autres &mdash; et le crop devrait &ecirc;tre <b>miroit&eacute;</b> pour que le
m&eacute;dial tombe toujours du m&ecirc;me c&ocirc;t&eacute;, la m&eacute;canique
&eacute;tant d&eacute;j&agrave; en place.</p>
</div>

<div class="card">
<h2>5 &middot; Faire d&eacute;filer des piles</h2>
<p>Scrolle d'avant en arri&egrave;re. Le ligament longe la corticale du c&ocirc;t&eacute;
m&eacute;dial&nbsp;; il est le plus net sur les coupes moyennes, l&agrave; o&ugrave;
l'interligne est ouvert.</p>
<p class="note">Les &eacute;tiquettes viennent de l'extraction des comptes rendus, pas
d'une lecture de ces pixels, et ce sont des labels faibles. Montr&eacute;es comme ce que
le texte disait &mdash; et pas par un radiologue.</p>""")

    n = 0
    for uid in pos[:args.cases]:
        body.append('<h3 class="case-head">le compte rendu mentionne le LCM</h3>')
        body.append(case(uid, f"mcl{n}", "Le compte rendu mentionne une atteinte du LCM."))
        n += 1
        print(f"  positive {uid[-11:]}", flush=True)
    for uid in neg[:args.cases]:
        body.append('<h3 class="case-head">le compte rendu ne le mentionne pas</h3>')
        body.append(case(uid, f"mcl{n}", "Le compte rendu n'en parle pas."))
        n += 1
        print(f"  negative {uid[-11:]}", flush=True)
    body.append("</div>")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(P.page("MCL — ligament collatéral médial", "".join(body)))
    print(f"\n{args.out}  ({args.out.stat().st_size/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
