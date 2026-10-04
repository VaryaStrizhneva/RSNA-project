"""The anterior cruciate ligament: what it is, where it is read, and a sober forecast.

    python -m tools.atlas.acl_page -o docs/atlas/acl.html

The fourth of these pages, and the first written with three expert results behind it.
It therefore spends as much space on what to expect as on the anatomy — the pixel-size
rule those three results suggest puts this target on the wrong side of the line, and
saying so before the run is the only way the prediction means anything.

Nothing here is a radiological reading. The figure shows where to scroll, not where the
ligament is; the examples carry report-extracted labels.
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
from tools.atlas import page as P                                    # noqa: E402
from tools.atlas.render import stack_to_jpegs, to_jpeg, window       # noqa: E402
from tools.atlas.study import (TRAIN_SERIES, load_series,            # noqa: E402
                               series_headers, side_of, stack_orientation)

LABELS = ROOT / "data/external/pilkwang-rsna-knee-llm-labels/report_labels_v2.csv"
BUNDLE = Path("/data/mgr/rsna-knee/bundle-acl/studies.json")

#: The study the figure walks through, and the slices. A left knee, 2D PD without fat
#: suppression. The crop is rows 32-70 % and columns 24-76 % of the rendered frame.
FIGURE_STUDY = "70623075553"
WALK = ((14, "trop lateral"), (18, "l'echancrure s'ouvre"),
        (20, "la bande oblique"), (24, "trop medial"))
GOOD = (18, 20)
CROP = (0.32, 0.70, 0.24, 0.76)


def _text(img, s, x, y, colour, scale=0.6):
    cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 5)
    cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 1)


def walk_figure() -> str:
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
            crop, (330, int(330 * crop.shape[0] / crop.shape[1])),
            interpolation=cv2.INTER_CUBIC), cv2.COLOR_GRAY2BGR)
        good = i in GOOD
        _text(tile, f"coupe {i}", 8, 24, (0, 255, 255))
        _text(tile, label, 8, tile.shape[0] - 12,
              (120, 255, 120) if good else (190, 190, 190), 0.55)
        if good:
            cv2.rectangle(tile, (2, 2), (tile.shape[1] - 3, tile.shape[0] - 3),
                          (120, 255, 120), 3)
        tiles.append(tile)
    out = np.hstack(tiles)
    _text(out, "ANTERIEUR", 8, out.shape[0] // 2, (200, 200, 255), 0.5)
    _text(out, "POSTERIEUR", out.shape[1] - 110, out.shape[0] // 2, (200, 200, 255), 0.5)
    return to_jpeg(out, max_width=1320)[0]


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
               f"{series.mm_per_px:.2f} mm/px &middot; genou {side or '?'} &mdash; "
               f"ant&eacute;rieur &agrave; gauche. {note}")
    return (f'<h3>&hellip;{P._esc(uid[-11:])}</h3>'
            + P.viewer(vid, frames, first, last, series.mm_per_px, caption=caption))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--out", type=Path, default=ROOT / "docs/atlas/acl.html")
    ap.add_argument("--cases", type=int, default=2)
    args = ap.parse_args()

    walk = walk_figure()
    lab = pd.read_csv(LABELS)
    pos = [u for u in lab[lab["ACL"] > 0.5]["StudyInstanceUID"]
           if (TRAIN_SERIES / u).is_dir()]
    neg = [u for u in lab[lab["ACL"] < 0.1]["StudyInstanceUID"]
           if (TRAIN_SERIES / u).is_dir()]
    c = Config()
    px = c.crop_mm / c.img

    body = [f"""
<header>
  <h1>ACL &mdash; le ligament crois&eacute; ant&eacute;rieur</h1>
  <p class="lede">La derni&egrave;re structure discr&egrave;te et localisable de la liste.
  Tout ce qui reste apr&egrave;s elle est diffus, trop gros pour qu'un crop serve, ou trop
  rare. Elle se lit sur les <b>m&ecirc;mes images sagittales</b> que les deux
  m&eacute;nisques, donc elle ne co&ucirc;te qu'une troisi&egrave;me passe de clics.</p>
</header>

<div class="card">
<h2>1 &middot; Ce qu'il faut en attendre &mdash; dit avant le run</h2>
<p>Trois experts ont tourn&eacute;, et ils sugg&egrave;rent une r&egrave;gle. Ce qui
s&eacute;pare une r&eacute;ussite d'un &eacute;chec n'est pas la quantit&eacute; de
signal dans l'image mais la <b>taille de la l&eacute;sion devant la vue enti&egrave;re du
genou</b>. Le mod&egrave;le large travaille &agrave; {px:.3f}&nbsp;mm par pixel
({c.crop_mm:.0f}&nbsp;mm sur {c.img}&nbsp;px), donc&nbsp;:</p>
<table class="report">
<tr><th>l&eacute;sion</th><th>taille</th><th>pixels chez le large</th><th>r&eacute;sultat de l'expert</th></tr>
<tr><td>fissure m&eacute;niscale</td><td>1,5 mm</td><td><b>3,9 px</b></td><td>+0.030 &#10003;</td></tr>
<tr><td>cartilage aminci (PF)</td><td>2 mm</td><td><b>5,2 px</b></td><td>+0.021 &#10003;</td></tr>
<tr class="danger"><td>bande du LCM</td><td>4 mm</td><td><b>10,3 px</b></td><td>&minus;0.013 &#10007;</td></tr>
<tr class="danger"><td><b>faisceau du LCA</b></td><td><b>10 mm</b></td><td><b>25,8 px</b></td><td><b>?</b></td></tr>
<tr><td>kyste de Baker</td><td>25 mm</td><td>64,6 px</td><td>non tent&eacute;</td></tr>
</table>
<p>Les deux r&eacute;ussites sont &agrave; 4-5&nbsp;px, l'&eacute;chec &agrave; 10. En
dessous du seuil, la vue enti&egrave;re ne r&eacute;sout plus la l&eacute;sion et
recadrer la fait appara&icirc;tre&nbsp;; au-dessus, elle est d&eacute;j&agrave; visible et
le crop n'ajoute rien. <b>Le LCA est &agrave; 26&nbsp;px, donc du mauvais c&ocirc;t&eacute;.</b></p>
<p>Ce qui pourrait le sauver&nbsp;: la l&eacute;sion n'est peut-&ecirc;tre pas le faisceau
entier mais la rupture de fibres et le changement de signal <em>dedans</em>, qui sont plus
fins. Je n'ai pas de quoi le mesurer, donc c'est une esp&eacute;rance et pas un argument.</p>
<p class="note">La pr&eacute;diction est &eacute;crite ici avant le run pour qu'elle
compte. Si le LCA r&eacute;ussit, la r&egrave;gle des pixels est &agrave; revoir&nbsp;;
si elle &eacute;choue, elle aura pr&eacute;dit trois cas sur trois puis un quatri&egrave;me.</p>
<table class="report">
<tr><th></th><th>positifs</th><th>comorbidit&eacute; seule</th><th>mod&egrave;le large</th><th>apport des pixels</th></tr>
<tr><td>ACL</td><td>349</td><td>0.7588</td><td>0.8241</td><td>+0.065</td></tr>
</table>
<p class="note">Quatri&egrave;me pire AUC des douze, et un apport pixel honn&ecirc;te.
C'est pour &ccedil;a qu'elle vaut l'essai malgr&eacute; la r&egrave;gle.</p>
</div>

<div class="card">
<h2>2 &middot; Le ligament</h2>
<p>Le LCA relie le f&eacute;mur au tibia <b>&agrave; l'int&eacute;rieur</b> de
l'articulation, dans l'&eacute;chancrure intercondylienne. Il part de la face interne du
condyle f&eacute;moral <em>lat&eacute;ral</em>, au fond de l'&eacute;chancrure, et descend
en avant vers le plateau tibial, devant les &eacute;pines.</p>
<p>C'est un <b>crois&eacute;</b> et non un collat&eacute;ral, et la diff&eacute;rence
commande tout le dessin de la r&eacute;gion&nbsp;: les collat&eacute;raux (LCM, LCL) sont
p&eacute;riph&eacute;riques, verticaux, contenus dans un plan&nbsp;; les crois&eacute;s
sont centraux et <b>obliques dans les trois dimensions</b>. Le LCA n'est donc
<b>jamais enti&egrave;rement dans une coupe</b>&nbsp;: il entre par l'une et sort par une
autre. C'est la cible o&ugrave; la profondeur du ROI comptera le plus, et il faudra y
penser d&egrave;s le d&eacute;but plut&ocirc;t qu'apr&egrave;s coup.</p>
<p>Une rupture se lit comme une interruption des fibres, un ligament horizontalis&eacute;
ou absent, souvent accompagn&eacute;e d'&oelig;d&egrave;me dans l'&eacute;chancrure.</p>
</div>

<div class="card">
<h2>3 &middot; Quelles images, et dans quel sens</h2>
<p>Le plan sagittal, le m&ecirc;me que les deux m&eacute;nisques &mdash; donc les
m&ecirc;mes s&eacute;ries, la m&ecirc;me couverture&nbsp;: PD fat-sat 82,6&nbsp;%, PD sans
fat-sat 37,5&nbsp;%, <b>au moins une des deux 99,8&nbsp;%</b>.</p>
<p><b>Orientation, mesur&eacute;e.</b> Sur <b>5563</b> s&eacute;ries sagittales PD, les
colonnes vont vers le <b>post&eacute;rieur</b> et les lignes vers le bas &mdash; 5563 sur
5563. Donc sur toutes les &eacute;tudes de ce corpus&nbsp;:</p>
<ul>
<li><b>l'ant&eacute;rieur est &agrave; gauche</b> de l'image, le post&eacute;rieur &agrave; droite&nbsp;;</li>
<li>le sup&eacute;rieur est en haut&nbsp;;</li>
<li>donc le LCA court du <b>haut &agrave; droite</b> (origine f&eacute;morale) vers le
    <b>bas &agrave; gauche</b> (insertion tibiale).</li>
</ul>
<figure style="margin:14px 0"><img src="{walk}" style="width:100%;border-radius:8px">
<figcaption class="note">Du lat&eacute;ral vers le m&eacute;dial, sur une &eacute;tude dont
le compte rendu ne mentionne pas le LCA. Ces vignettes montrent <b>o&ugrave; scroller</b>,
pas o&ugrave; est le ligament&nbsp;: je ne l'identifie pas de fa&ccedil;on fiable sur ces
images, et une croix invent&eacute;e serait pire qu'une absence de croix.</figcaption></figure>
</div>

<div class="card">
<h2>4 &middot; Le point</h2>
<p><b>&Agrave; mi-chemin entre l'origine f&eacute;morale et l'insertion tibiale</b>, sur
le ligament, sur la coupe qui en montre la plus grande longueur d'un seul tenant.</p>
<p>Le milieu g&eacute;om&eacute;trique et non le site de rupture, pour la raison qui vaut
pour les quatre points pr&eacute;c&eacute;dents&nbsp;: il reste d&eacute;fini quand le
ligament est d&eacute;truit. Un rep&egrave;re qui d&eacute;rive vers la l&eacute;sion sur
les positifs fait encoder le label par la position du crop, et ce signal-l&agrave; existe
dans le jeu d'entra&icirc;nement et nulle part ailleurs.</p>
<h3>Le c&ocirc;t&eacute; ne se r&eacute;cup&egrave;re pas ici</h3>
<p>Pour les deux m&eacute;nisques, le clic suffit &agrave; dire de quel genou il s'agit&nbsp;:
ils sont &agrave; 25&nbsp;mm d'un bord, donc nettement plus pr&egrave;s d'un bout de la
pile. Le LCA est <b>au milieu</b> de l'axe gauche-droite, donc sa position dans la pile ne
dit rien. Le registre le d&eacute;clare (<code>click_near: None</code>) et un test
v&eacute;rifie qu'&ecirc;tre sagittal ne suffit pas &agrave; activer la r&egrave;gle
&mdash; sans quoi un c&ocirc;t&eacute; serait d&eacute;duit d'une position qui n'en
contient aucune information, et les deux r&eacute;ponses auraient sembl&eacute; aussi
valides l'une que l'autre.</p>
</div>

<div class="card">
<h2>5 &middot; Faire d&eacute;filer des piles</h2>
<p class="note">&Eacute;tiquettes issues de l'extraction des comptes rendus, pas d'une
lecture de ces pixels &mdash; et pas par un radiologue.</p>"""]

    n = 0
    for uid in pos[:args.cases]:
        body.append('<h3 class="case-head">le compte rendu mentionne le LCA</h3>')
        body.append(case(uid, f"acl{n}", "Le compte rendu le mentionne."))
        n += 1
        print(f"  positive {uid[-11:]}", flush=True)
    for uid in neg[:args.cases]:
        body.append('<h3 class="case-head">le compte rendu ne le mentionne pas</h3>')
        body.append(case(uid, f"acl{n}", "Le compte rendu n'en parle pas."))
        n += 1
        print(f"  negative {uid[-11:]}", flush=True)
    body.append("</div>")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(P.page("ACL — le ligament croisé antérieur", "".join(body)))
    print(f"\n{args.out}  ({args.out.stat().st_size/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
