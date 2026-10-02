"""The medial meniscus: what it is, and the region of interest it inherits.

    python -m tools.atlas.medial_meniscus_page -o docs/atlas/medial_meniscus.html

Written before the annotation round, like the two before it. It is the shortest of the
three because almost nothing here is new: the structure is the mirror of one this
pipeline already reads well, and what changes is named rather than re-derived.

Nothing here is a radiological reading. The examples are labelled from the report
extraction, which is a weak label.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from rsna.config import Config                                       # noqa: E402
from rsna.roi import SPECS                                           # noqa: E402
from tools.atlas import page as P                                    # noqa: E402
from tools.atlas.render import stack_to_jpegs, to_jpeg, window       # noqa: E402
from tools.atlas.study import (TRAIN_SERIES, load_series,            # noqa: E402
                               series_headers, side_of, stack_orientation)

LABELS = ROOT / "data/external/pilkwang-rsna-knee-llm-labels/report_labels_v2.csv"
BUNDLE = Path("/data/mgr/rsna-knee/bundle-med/studies.json")

#: The study the progression figure is drawn on — a left knee, 2D PD without fat
#: suppression, whose report does not mention the medial meniscus — and the four slices
#: walked in from its medial edge. The crop is rows 42-70 % and columns 24-76 % of the
#: rendered frame; nothing here is detected, it is read off the images.
FIGURE_STUDY = "70623075553"
WALK = ((30, "bord medial"), (28, "le BOWTIE"), (26, "il s'amincit"),
        (24, "DEUX CORNES  <- ici"))
CROP = (0.42, 0.70, 0.24, 0.76)


def _text(img, s, x, y, colour, scale=0.6):
    cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 5)
    cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 1)


def progression_figure() -> str:
    """Walking in from the medial edge: nothing, then the bowtie, then the two horns."""

    manifest = json.loads(BUNDLE.read_text())["studies"]
    study = next(x for x in manifest if x["tail"] == FIGURE_STUDY)
    root = BUNDLE.parent
    tiles = []
    for i, label in WALK:
        im = cv2.imread(str(root / study["slices"][i]["file"]), cv2.IMREAD_GRAYSCALE)
        h, w = im.shape
        r0, r1, c0, c1 = CROP
        crop = im[int(h * r0):int(h * r1), int(w * c0):int(w * c1)]
        tile = cv2.cvtColor(cv2.resize(
            crop, (340, int(340 * crop.shape[0] / crop.shape[1])),
            interpolation=cv2.INTER_CUBIC), cv2.COLOR_GRAY2BGR)
        good = i == WALK[-1][0]
        _text(tile, f"coupe {i}", 8, 24, (0, 255, 255))
        _text(tile, label, 8, tile.shape[0] - 12,
              (120, 255, 120) if good else (200, 200, 200))
        if good:
            cv2.rectangle(tile, (2, 2), (tile.shape[1] - 3, tile.shape[0] - 3),
                          (120, 255, 120), 3)
        tiles.append(tile)
    return to_jpeg(np.hstack(tiles), max_width=1360)[0]


def case(uid: str, vid: str, note: str) -> str:
    headers = series_headers(uid)
    sag = headers[headers["plane"] == "Sagittal"]
    if not len(sag):
        return ""
    row = sag.sort_values(["fatsat", "n_slices"], ascending=[True, False]).iloc[0]
    series = load_series(row, Config())
    side, how = side_of(headers)
    first, last = stack_orientation("Sagittal", side)
    frames, _ = stack_to_jpegs(series.volume)
    caption = (f"{P._esc(series.label)} &middot; {series.n} coupes &middot; "
               f"{series.mm_per_px:.2f} mm/px &middot; genou {side or '?'} "
               f"({P._esc(how)}). {note}")
    return (f'<h3>&hellip;{P._esc(uid[-11:])}</h3>'
            + P.viewer(vid, frames, first, last, series.mm_per_px, caption=caption))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--out", type=Path,
                    default=ROOT / "docs/atlas/medial_meniscus.html")
    ap.add_argument("--cases", type=int, default=2)
    args = ap.parse_args()

    walk_fig = progression_figure()
    lat = SPECS["lateral_meniscus"]
    lab = pd.read_csv(LABELS)
    pos = [u for u in lab[lab["Medial Meniscus"] > 0.5]["StudyInstanceUID"]
           if (TRAIN_SERIES / u).is_dir()]
    neg = [u for u in lab[lab["Medial Meniscus"] < 0.1]["StudyInstanceUID"]
           if (TRAIN_SERIES / u).is_dir()]

    body = [f"""
<header>
  <h1>Medial Meniscus &mdash; l'autre compartiment</h1>
  <p class="lede">La cible pour laquelle on a la meilleure raison d'esp&eacute;rer&nbsp;:
  c'est la <b>m&ecirc;me structure</b> que le m&eacute;nisque lat&eacute;ral, en miroir,
  et c'est la seule o&ugrave; la recette de l'expert a d&eacute;j&agrave;
  fonctionn&eacute; sur l'anatomie identique. Avec presque trois fois plus de positifs.</p>
</header>

<div class="card">
<h2>1 &middot; Pourquoi celle-ci, et ce que trois experts ont appris</h2>
<p>Le crit&egrave;re a chang&eacute; en cours de route, et c'est le MCL qui l'a
chang&eacute;. L'apport des pixels par-dessus la comorbidit&eacute; ne suffit pas &agrave;
pr&eacute;dire qu'un expert va marcher&nbsp;: le MCL en avait plus que l'arthrose
f&eacute;moro-patellaire et a &eacute;chou&eacute;.</p>
<table class="report">
<tr><th>cible</th><th>mod&egrave;le large</th><th>expert</th><th>&eacute;cart appari&eacute;</th><th>la l&eacute;sion</th></tr>
<tr><td>Lateral Meniscus</td><td>0.8206</td><td>0.8508</td><td><b>+0.030</b></td>
    <td>fissure de ~1,5 mm</td></tr>
<tr><td>PF OA</td><td>0.8093</td><td>0.8305</td><td><b>+0.021</b></td>
    <td>cartilage aminci, ost&eacute;ophytes</td></tr>
<tr class="danger"><td>MCL</td><td>0.8080</td><td>0.8042</td><td><b>&minus;0.013</b></td>
    <td>bande de plusieurs mm, &oelig;d&egrave;me diffus</td></tr>
</table>
<p>Ce qui s&eacute;pare les trois est la <b>taille de la l&eacute;sion devant la vue
enti&egrave;re du genou</b>. Recadrer l&egrave;ve une limite de r&eacute;solution&nbsp;;
l&agrave; o&ugrave; il n'y en a pas, &ccedil;a n'apporte rien. Le bon crit&egrave;re est
donc double&nbsp;: du signal dans l'image <b>et</b> une l&eacute;sion petite.</p>
<p class="note">C'est une hypoth&egrave;se tir&eacute;e de trois cas, pas une loi. Le
m&eacute;nisque m&eacute;dial est le test le plus franc qu'on puisse lui faire subir&nbsp;:
m&ecirc;me taille de l&eacute;sion que le lat&eacute;ral, m&ecirc;me plan, m&ecirc;me
s&eacute;quence. Si l'hypoth&egrave;se vaut quelque chose, &ccedil;a doit marcher&nbsp;;
si &ccedil;a rate, elle est fausse.</p>
<table class="report">
<tr><th></th><th>positifs</th><th>comorbidit&eacute; seule</th><th>mod&egrave;le large</th><th>apport des pixels</th></tr>
<tr><td>Lateral Meniscus <span class="tag">fait</span></td><td>1121</td><td>0.7655</td><td>0.8206</td><td>+0.055</td></tr>
<tr class="danger"><td><b>Medial Meniscus</b></td><td><b>1808</b></td><td><b>0.7538</b></td><td><b>0.8785</b></td><td><b>+0.125</b></td></tr>
</table>
<p class="note">Le mod&egrave;le large est d&eacute;j&agrave; &agrave; 0.8785, nettement
mieux que sur le lat&eacute;ral &mdash; probablement parce que les l&eacute;sions
m&eacute;diales sont bien plus fr&eacute;quentes, donc plus apprises. La marge est donc
plus &eacute;troite en haut qu'elle ne l'&eacute;tait sur le lat&eacute;ral. C'est la
r&eacute;serve honn&ecirc;te &agrave; garder.</p>
</div>

<div class="card">
<h2>2 &middot; Ce qui change anatomiquement</h2>
<p>Le m&eacute;nisque m&eacute;dial est <b>plus grand et plus ouvert</b> que le
lat&eacute;ral &mdash; un C large plut&ocirc;t qu'un O presque ferm&eacute; &mdash; et sa
corne post&eacute;rieure est nettement plus large que l'ant&eacute;rieure. Il est aussi
<b>attach&eacute; &agrave; la capsule</b> sur tout son bord, l&agrave; o&ugrave; le
lat&eacute;ral est plus libre&nbsp;: il bouge moins, se d&eacute;chire plus souvent, et
c'est pourquoi les l&eacute;sions m&eacute;diales dominent le corpus.</p>
<p>Pour l'annotation, une seule chose change vraiment&nbsp;: <b>le bowtie est au bord
m&eacute;dial</b>, donc &agrave; l'autre bout de la pile. La progression est la
m&ecirc;me, prise dans l'autre sens.</p>
<figure style="margin:14px 0"><img src="{walk_fig}" style="width:100%;border-radius:8px">
<figcaption class="note">En venant du bord m&eacute;dial&nbsp;: rien d'articulaire, puis
le bowtie &mdash; le corps du m&eacute;nisque coup&eacute; dans sa longueur &mdash; puis
il s'amincit, puis <b>deux coins sombres</b>, les cornes. Le point va sur cette
derni&egrave;re configuration, pas sur le bowtie.</figcaption></figure>
</div>

<div class="card">
<h2>3 &middot; Quelles images</h2>
<p>Les m&ecirc;mes que pour le lat&eacute;ral, et pour la m&ecirc;me raison&nbsp;: une
fissure est du signal <em>&agrave; l'int&eacute;rieur</em> du fibrocartilage, que la
suppression de graisse aplatit. Donc PD <b>sans</b> fat-sat d'abord.</p>
<table class="report">
<tr><th>s&eacute;rie</th><th>des &eacute;tudes</th></tr>
<tr><td>Sagittale PD fat-sat</td><td>82.6 %</td></tr>
<tr><td>Sagittale PD sans fat-sat</td><td>37.5 %</td></tr>
<tr><td><b>au moins une des deux</b></td><td><b>99.8 %</b></td></tr>
</table>
<p class="note">99.8 % de couverture, donc le probl&egrave;me du bloc d'ex &aelig;quo
qu'on vient de corriger sur le MCL ne se pose pas ici&nbsp;: 0.2 % d'&eacute;tudes sans
pixels co&ucirc;tent de l'ordre de 0.0005 d'AUC.</p>
</div>

<div class="card">
<h2>4 &middot; La r&eacute;gion d'int&eacute;r&ecirc;t</h2>
<p>Elle s'h&eacute;rite presque enti&egrave;rement. Le spec du m&eacute;nisque
lat&eacute;ral est <b>{lat.box_w_mm:.0f}&times;{lat.box_h_mm:.0f} mm</b> &agrave;
{lat.mm_per_px:.3f} mm/px, sur une profondeur de
{lat.lateral_mm:.0f}/{lat.medial_mm:.0f} mm &eacute;tal&eacute;e sur {lat.slots} slots.
Rien dans ces nombres ne d&eacute;pend du compartiment&nbsp;: c'est la taille d'un
m&eacute;nisque et l'&eacute;paisseur de son voisinage utile, et le m&eacute;dial est du
m&ecirc;me ordre.</p>
<p><b>Une seule chose doit s'inverser, et elle est subtile.</b> La profondeur est
asym&eacute;trique &mdash; {lat.lateral_mm:.0f} mm vers le bowtie,
{lat.medial_mm:.0f} mm vers l'&eacute;chancrure &mdash; parce que le point est &agrave;
une coupe ou deux du bowtie et que l'essentiel du m&eacute;nisque est de l'autre
c&ocirc;t&eacute;. Pour le m&eacute;nisque m&eacute;dial, le bowtie est au bord
<em>m&eacute;dial</em>, donc <b>la direction courte change de sens</b>.</p>
<p>Concr&egrave;tement&nbsp;: <code>rsna.roi.extract.bowtie_direction</code> trouve la
p&eacute;riph&eacute;rie <em>lat&eacute;rale</em> &mdash; mesur&eacute; 298 fois sur 298,
le membre imag&eacute; s'arr&ecirc;te plus vite de ce c&ocirc;t&eacute;-l&agrave;. Pour le
compartiment m&eacute;dial il faudra prendre l'oppos&eacute;, donc un champ au spec qui
dise de quel c&ocirc;t&eacute; est sa p&eacute;riph&eacute;rie. Sans quoi la bo&icirc;te
serait courte du mauvais c&ocirc;t&eacute; et perdrait la corne post&eacute;rieure
&mdash; celle qui se d&eacute;chire le plus.</p>
<p class="note">La bo&icirc;te et la profondeur auront leur page de tailles une fois les
annotations faites, comme pour les trois autres. On ne reprend pas les nombres du
lat&eacute;ral sans les regarder&nbsp;; on part de l&agrave;.</p>
</div>

<div class="card">
<h2>5 &middot; Faire d&eacute;filer des piles</h2>
<p class="note">Les &eacute;tiquettes viennent de l'extraction des comptes rendus, pas
d'une lecture de ces pixels &mdash; et pas par un radiologue.</p>"""]

    n = 0
    for uid in pos[:args.cases]:
        body.append('<h3 class="case-head">le compte rendu mentionne le m&eacute;nisque m&eacute;dial</h3>')
        body.append(case(uid, f"mm{n}", "Le compte rendu le mentionne."))
        n += 1
        print(f"  positive {uid[-11:]}", flush=True)
    for uid in neg[:args.cases]:
        body.append('<h3 class="case-head">le compte rendu ne le mentionne pas</h3>')
        body.append(case(uid, f"mm{n}", "Le compte rendu n'en parle pas."))
        n += 1
        print(f"  negative {uid[-11:]}", flush=True)
    body.append("</div>")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(P.page("Medial Meniscus — l'autre compartiment", "".join(body)))
    print(f"\n{args.out}  ({args.out.stat().st_size/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
