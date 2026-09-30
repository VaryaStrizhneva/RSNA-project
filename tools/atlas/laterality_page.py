"""Why a sagittal stack has to be reversed for a right knee, shown rather than argued.

    python -m tools.atlas.laterality_page -o docs/atlas/lateralite_sagittale.html

The defect this page explains is invisible in any single image, which is exactly why it
survived a port and three weeks of runs. So the page is built around one device: a left
knee and a right knee side by side in a single composite frame, scrolled by one slider.
Two stacks that cannot drift out of step are the only way to see an error that lives in
the *order* of the slices and nowhere else.

Both halves are resampled to one common millimetres-per-pixel, so the two knees are
drawn at true relative size and a single scale bar is valid across the composite. Every
frame is a real acquired slice — the resampling picks nearest neighbours along the
depth axis, it never interpolates one.

Written in French: it answers a question asked in French by the person who has to act
on it. The code and its docstrings stay in English like the rest of the repository.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from rsna.config import Config                                    # noqa: E402
from tools.atlas import page as P                                 # noqa: E402
from tools.atlas.render import to_jpeg, window                    # noqa: E402
from tools.atlas.study import (load_series, series_headers,       # noqa: E402
                               side_of, stack_orientation)

# One left knee and one right knee, both sagittal PD without fat saturation and both at
# 3.30 mm between slices, so what differs between the two halves is the knee and not the
# acquisition. Their own SeriesDescription says LEFT and RT, which agrees with the tag.
PAIR = {
    "L": ("70691305452244378255646875206864340458",
          "43077138775719195355134064277352537333"),
    "R": ("38039631093067555454700266397073080430",
          "70073707803775716546213676851185016662"),
}

GAP = 16          # black columns between the two halves
N_FRAMES = 38     # common depth resolution for the paired scroll


# --------------------------------------------------------------------------- pixels

def _load(study_tail: str, series_tail: str, config: Config):
    """The series, its side, and which anatomical end each end of the stack is."""

    from tools.atlas.study import TRAIN_SERIES
    hits = [d for d in TRAIN_SERIES.iterdir() if d.name.endswith(study_tail)]
    if len(hits) != 1:
        raise ValueError(f"{study_tail!r} matches {len(hits)} studies")
    headers = series_headers(hits[0].name)
    row = headers[headers["SeriesInstanceUID"].str.endswith(series_tail)].iloc[0]
    series = load_series(row, config)
    side, how = side_of(headers)
    first, last = stack_orientation(series.plane, side)
    return series, side, how, first, last


def _rescale(volume_u8: np.ndarray, mm_per_px: float, target_mm_px: float) -> np.ndarray:
    """Resample so one output pixel spans `target_mm_px`, for both halves alike."""

    factor = mm_per_px / target_mm_px
    out = []
    for s in volume_u8:
        im = Image.fromarray(s)
        out.append(np.asarray(im.resize((max(1, round(im.width * factor)),
                                         max(1, round(im.height * factor))),
                                        Image.LANCZOS)))
    return np.stack(out)


def _depth_resample(volume: np.ndarray, n: int) -> np.ndarray:
    """`n` real slices spanning the stack — nearest neighbour, never interpolated."""

    return volume[np.linspace(0, len(volume) - 1, n).round().astype(int)]


def _pad(slice_u8: np.ndarray, h: int, w: int) -> np.ndarray:
    out = np.zeros((h, w), np.uint8)
    y, x = (h - slice_u8.shape[0]) // 2, (w - slice_u8.shape[1]) // 2
    y0, x0 = max(y, 0), max(x, 0)
    crop = slice_u8[max(-y, 0):max(-y, 0) + min(h, slice_u8.shape[0]),
                    max(-x, 0):max(-x, 0) + min(w, slice_u8.shape[1])]
    out[y0:y0 + crop.shape[0], x0:x0 + crop.shape[1]] = crop
    return out


def composite(a: np.ndarray, b: np.ndarray) -> tuple[list[np.ndarray], float]:
    """Two equal-length stacks, laid side by side on one canvas of equal halves."""

    h = max(a.shape[1], b.shape[1])
    w = max(a.shape[2], b.shape[2])
    sep = np.zeros((h, GAP), np.uint8)
    frames = [np.concatenate([_pad(x, h, w), sep, _pad(y, h, w)], axis=1)
              for x, y in zip(a, b)]
    return frames, w / (2 * w + GAP)    # where the left half ends, as a fraction


def tissue_span(volume_u8: np.ndarray, frac: float = 0.55) -> tuple[int, int]:
    """The run of slices that actually hold a knee rather than air.

    Both stacks extend well past the joint into plain soft tissue, and they do not do
    it symmetrically, so scrolling them together frame by frame compares a far-lateral
    slice of one against mid-joint of the other. Trimming on imaged area puts the two
    halves back on comparable ground without anyone choosing the numbers by hand.
    """

    area = (volume_u8 > 40).mean(axis=(1, 2))
    idx = np.flatnonzero(area >= frac * area.max())
    return int(idx[0]), int(idx[-1])


def strip(volume_u8: np.ndarray, indices: list[int]) -> tuple[np.ndarray, int]:
    """One row of equal cells, so HTML labels underneath line up with the slices."""

    h = max(volume_u8[i].shape[0] for i in indices)
    w = max(volume_u8[i].shape[1] for i in indices)
    return np.concatenate([_pad(volume_u8[i], h, w) for i in indices], axis=1), len(indices)


def labels_row(indices: list[int], n_total: int, accent: str) -> str:
    cells = "".join(f"<div>{i}</div>" for i in indices)
    return (f'<div class="striplab" style="grid-template-columns:repeat({len(indices)},1fr);'
            f'color:{accent}">{cells}</div>')


def arrow_row(a: str, b: str, accent: str) -> str:
    return (f'<div class="arrow" style="border-color:{accent};color:{accent}">'
            f'<span>{P._esc(a).upper()}</span>'
            f'<span class="line"></span><span>{P._esc(b).upper()} &#9654;</span></div>')


# ---------------------------------------------------------------------------- html

EXTRA_CSS = """
.pairhead{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:6px}
.pairhead div{text-align:center;font-weight:700;font-size:.92rem;letter-spacing:.02em}
.pairhead .l{color:#8fd0ff}.pairhead .r{color:#ffc08a}
.pairaxis{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:6px}
.pairaxis div{display:flex;justify-content:space-between;color:var(--dim);
              font-size:.74rem;text-transform:uppercase;letter-spacing:.05em}
.ov{position:absolute;color:#fff;font-size:.74rem;font-weight:700;letter-spacing:.08em;
    text-shadow:0 0 5px #000,0 0 3px #000;pointer-events:none}
.duo{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin:16px 0}
.duo figcaption{color:var(--dim);font-size:.84rem;margin-top:6px}
.duo .stage img{width:100%;display:block}
.trio{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}
.trio .stage img{width:100%;display:block}
.trio .cap{text-align:center;color:var(--dim);font-family:ui-monospace,monospace;
           font-size:.76rem;margin-top:4px}
.striprap{overflow-x:auto;border:1px solid var(--line);border-radius:8px;background:#000}
.striprap img{display:block;width:100%;min-width:1050px;height:auto}
.striplab{display:grid;font-family:ui-monospace,monospace;font-size:.7rem;
          text-align:center;margin-top:3px;min-width:1050px}
.arrow{display:flex;align-items:center;gap:10px;margin:8px 0 2px;font-size:.76rem;
       font-weight:700;letter-spacing:.08em;border:0}
.arrow .line{flex:1;height:2px;background:currentColor;opacity:.5}
.key{background:#12161b;border-left:3px solid var(--accent);border-radius:0 8px 8px 0;
     padding:12px 16px;margin:18px 0}
.answer{border:1px solid var(--line);border-radius:8px;background:var(--panel);
        padding:0 16px;margin:16px 0}
.answer summary{cursor:pointer;padding:12px 0;font-weight:600;color:var(--accent)}
.answer[open] summary{border-bottom:1px solid var(--line);margin-bottom:12px}
.answer p:last-child{padding-bottom:14px}
"""


def pair_viewer(vid: str, frames: list[str], mm_per_px: float,
                left_ends: tuple[str, str], right_ends: tuple[str, str],
                flagged=(), flag_text: str = "", caption: str = "") -> str:
    """Two knees, one slider. Reuses `mount` from the atlas, so nothing new can drift."""

    toggle = (f'<label style="text-transform:none;letter-spacing:0">'
              f'<input type="checkbox" id="{vid}-flag"> {P._esc(flag_text)}</label>'
              if flag_text else "")
    return f"""
<div class="pairhead"><div class="l">GENOU GAUCHE</div><div class="r">GENOU DROIT</div></div>
<div class="viewer">
  <div>
    <div class="stage" id="{vid}">
      <img alt="" src="{frames[0]}">
      <div class="bar"></div><div class="barlabel"></div>
      <div class="flag">{P._esc(flag_text)}</div>
    </div>
    <div class="pairaxis">
      <div><span>&#9664; {P._esc(left_ends[0])}</span><span>{P._esc(left_ends[1])} &#9654;</span></div>
      <div><span>&#9664; {P._esc(right_ends[0])}</span><span>{P._esc(right_ends[1])} &#9654;</span></div>
    </div>
    <div class="note">{caption}</div>
  </div>
  <div class="ctl">
    <label>coupe <span class="count" id="{vid}-n"></span></label>
    <input type="range" id="{vid}-sl" min="0" max="{len(frames)-1}" value="0">
    <label>luminosité</label>
    <input type="range" id="{vid}-b" min="0.4" max="2.2" step="0.05" value="1">
    <label>contraste</label>
    <input type="range" id="{vid}-c" min="0.5" max="2.6" step="0.05" value="1">
    <div class="note">Molette sur l'image, ou clique dessus et
      <kbd>&larr;</kbd> <kbd>&rarr;</kbd>.</div>
    {toggle}
  </div>
</div>
<script>mount({vid!r}, {json.dumps(frames)}, {json.dumps(list(flagged))}, {mm_per_px});</script>
""".replace("'" + vid + "'", '"' + vid + '"')


def _fig(uri: str, overlays: str = "", caption: str = "") -> str:
    return (f'<div class="stage" style="max-width:560px">'
            f'<img alt="" src="{uri}">{overlays}</div>'
            f'<div class="note" style="max-width:560px">{caption}</div>')


def build(config: Config) -> str:
    left, side_l, _, l_first, l_last = _load(*PAIR["L"], config)
    right, side_r, _, r_first, r_last = _load(*PAIR["R"], config)
    assert (side_l, side_r) == ("L", "R"), (side_l, side_r)

    target = max(left.mm_per_px, right.mm_per_px)
    vol_l = _rescale(window(left.volume), left.mm_per_px, target)
    vol_r = _rescale(window(right.volume), right.mm_per_px, target)

    # ---- full-stack strips, the main device ------------------------------------
    def spaced(v, k=10):
        return [int(round(x)) for x in np.linspace(0, len(v) - 1, k)]

    idx_l, idx_r = spaced(vol_l), spaced(vol_r)
    small_l = _rescale(vol_l, target, 0.62)
    small_r = _rescale(vol_r, target, 0.62)
    strip_l = to_jpeg(strip(small_l, idx_l)[0], 1500)[0]
    strip_r = to_jpeg(strip(small_r, idx_r)[0], 1500)[0]
    strip_r_fixed = to_jpeg(strip(small_r, idx_r[::-1])[0], 1500)[0]

    # ---- the same bone, at opposite ends ---------------------------------------
    FIB_L, FIB_R = 5, 31
    fib_l = to_jpeg(vol_l[FIB_L], 420)[0]
    fib_r = to_jpeg(vol_r[FIB_R], 420)[0]

    # ---- the paired scroll, trimmed to where there is a knee --------------------
    al, bl = tissue_span(vol_l)
    ar, br = tissue_span(vol_r)
    dl = _depth_resample(vol_l[al:bl + 1], N_FRAMES)
    dr = _depth_resample(vol_r[ar:br + 1], N_FRAMES)
    raw = [to_jpeg(f, 980)[0] for f in composite(dl, dr)[0]]
    fix = [to_jpeg(f, 980)[0] for f in composite(dl, dr[::-1])[0]]

    # ---- single slices ----------------------------------------------------------
    mid_l, mid_r = vol_l[len(vol_l) // 2], vol_r[len(vol_r) // 2]
    uri_l, uri_r = to_jpeg(mid_l, 560)[0], to_jpeg(mid_r, 560)[0]
    uri_l_mirror = to_jpeg(mid_l[:, ::-1], 560)[0]

    cl, cr = len(vol_l) // 2, len(vol_r) // 2
    trio_l = [to_jpeg(vol_l[cl + d], 300)[0] for d in (-1, 0, 1)]
    trio_r = [to_jpeg(vol_r[cr + d], 300)[0] for d in (-1, 0, 1)]

    BLUE, ORANGE = "#8fd0ff", "#ffc08a"
    axes_overlay = (
        '<div class="ov" style="left:12px;top:50%">&#9664; ANTÉRIEUR</div>'
        '<div class="ov" style="right:12px;top:50%">POSTÉRIEUR &#9654;</div>'
        '<div class="ov" style="left:50%;top:10px;transform:translateX(-50%)">&#9650; SUPÉRIEUR</div>'
        '<div class="ov" style="left:50%;bottom:10px;transform:translateX(-50%)">&#9660; INFÉRIEUR</div>')

    geom = (f"Les deux piles sont recadrées sur les coupes qui contiennent réellement un "
            f"genou (aire imagée &ge; 55&nbsp;% du maximum : {al}&ndash;{bl} à gauche, "
            f"{ar}&ndash;{br} à droite), puis ramenées à {N_FRAMES} niveaux pour défiler "
            f"ensemble. Chaque image reste une coupe réelle, aucune n'est interpolée. "
            f"Les deux moitiés sont à {target:.3f} mm/px, donc à la même échelle.")

    return f"""
<style>{EXTRA_CSS}</style>

<header>
  <h1>La latéralité sagittale, en images</h1>
  <p class="lede">Un volume sagittal a trois axes. Deux se voient sur n'importe quelle
  image et n'ont jamais posé problème. Le troisième ne se voit sur aucune image prise
  isolément &mdash; et c'est celui qui était faux. Cette page montre lequel est lequel,
  sur deux études réelles.</p>
</header>

<h2>1. Ce que tu vois sur une coupe</h2>
<p>Une coupe sagittale est un plan vertical orienté avant-arrière. Ses deux axes sont
l'<b>antéro-postérieur</b> (horizontal) et le <b>supéro-inférieur</b> (vertical).
Aucune ambiguïté&nbsp;: la rotule est devant, le creux poplité derrière.</p>
{_fig(uri_l, axes_overlay,
      "Genou gauche, PD sagittale. Mesuré sur les 170 études du bundle d'annotation&nbsp;: "
      "la droite de l'image va vers le postérieur et le bas vers l'inférieur, "
      "<b>170 fois sur 170</b>, genou gauche comme genou droit.")}
<div class="key"><b>Premier point.</b> L'antéro-postérieur n'est jamais retourné, ni par
le scanner ni par notre code. Si on le retournait, ça se verrait tout de suite
&mdash; section 5.</div>

<h2>2. Deux genoux. Lequel est le gauche&nbsp;?</h2>
<p>Une coupe de chacun, à mi-profondeur, à la même échelle.</p>
<div class="duo">
  <figure style="margin:0"><div class="stage"><img alt="" src="{uri_l}"></div>
    <figcaption>A</figcaption></figure>
  <figure style="margin:0"><div class="stage"><img alt="" src="{uri_r}"></div>
    <figcaption>B</figcaption></figure>
</div>
<details class="answer"><summary>La réponse</summary>
<p><b>A est le gauche, B est le droit</b> &mdash; mais tu ne pouvais pas le savoir en
regardant. Les deux axes visibles sont orientés pareil dans les deux cas. Une coupe
sagittale ne porte aucune information sur le côté du corps dont elle vient&nbsp;; c'est
la <b>pile entière</b> qui la porte.</p>
<p class="meta">A = &hellip;{PAIR["L"][0][-11:]} &laquo;{P._esc(left.description)}&raquo; &middot;
B = &hellip;{PAIR["R"][0][-11:]} &laquo;{P._esc(right.description)}&raquo; &middot;
côté lu dans le tag DICOM <code>Laterality</code> pour les deux.</p>
</details>

<h2>3. Ce qui change est dans la profondeur</h2>
<p>Le troisième axe est celui dans lequel on avance en <b>changeant de coupe</b>.
Anatomiquement c'est l'axe <b>médial&ndash;latéral</b> &mdash; celui-là même contre
lequel «&nbsp;ménisque latéral&nbsp;» est défini. Voici les deux piles entières, dix
coupes réparties sur toute la pile, dans l'ordre où le code les reçoit.</p>

<h3 style="color:{BLUE}">Genou gauche &mdash; {left.n} coupes</h3>
<div class="striprap"><img alt="" src="{strip_l}">{labels_row(idx_l, left.n, BLUE)}</div>
{arrow_row(l_first, l_last, BLUE)}

<h3 style="color:{ORANGE}">Genou droit &mdash; {right.n} coupes</h3>
<div class="striprap"><img alt="" src="{strip_r}">{labels_row(idx_r, right.n, ORANGE)}</div>
{arrow_row(r_first, r_last, ORANGE)}

<p>Les deux planches se lisent de gauche à droite dans l'ordre des coupes. Elles vont
dans des directions anatomiques <b>opposées</b>. Ce n'est pas une interprétation&nbsp;:
c'est lisible dans les coordonnées patient des deux séries.</p>
<table>
<tr><th></th><th>coupe&nbsp;0</th><th>dernière coupe</th><th>sens</th></tr>
<tr><td><b style="color:{BLUE}">genou gauche</b></td><td>x&nbsp;=&nbsp;+135,6&nbsp;mm</td>
    <td>x&nbsp;=&nbsp;+13,6&nbsp;mm</td><td>latéral &rarr; médial</td></tr>
<tr><td><b style="color:{ORANGE}">genou droit</b></td><td>x&nbsp;=&nbsp;&minus;32,0&nbsp;mm</td>
    <td>x&nbsp;=&nbsp;&minus;157,7&nbsp;mm</td><td>médial &rarr; latéral</td></tr>
</table>
<p class="note">En coordonnées DICOM (LPS), <b>+x est la gauche du patient</b>. «&nbsp;Médial&nbsp;»
veut dire «&nbsp;vers le plan médian&nbsp;», donc vers x&nbsp;=&nbsp;0. Les deux piles
descendent en x &mdash; c'est vrai sur 600 séries sagittales tirées au hasard, normale
négative 600 fois sur 600 &mdash; et c'est précisément pour ça que le sens anatomique
s'inverse entre un genou gauche et un genou droit.</p>

<h3>Le même os, aux deux bouts opposés</h3>
<p>La <b>tête du péroné</b> est le repère le plus net du côté latéral&nbsp;: un second os,
détaché du tibia, qui n'existe que de ce côté. Sur le genou gauche elle apparaît
<b>avant</b> l'articulation&nbsp;; sur le droit, <b>après</b>.</p>
<div class="duo">
  <figure style="margin:0"><div class="stage"><img alt="" src="{fib_l}"></div>
    <figcaption><b style="color:{BLUE}">Genou gauche, coupe {FIB_L} sur {left.n}</b>
    &mdash; tout au début de la pile.</figcaption></figure>
  <figure style="margin:0"><div class="stage"><img alt="" src="{fib_r}"></div>
    <figcaption><b style="color:{ORANGE}">Genou droit, coupe {FIB_R} sur {right.n}</b>
    &mdash; tout à la fin.</figcaption></figure>
</div>
<div class="claude"><h4>Identification par Claude &mdash; non vérifiée, je ne suis pas radiologue</h4>
<p style="margin:0">Ce que je lis comme la tête du péroné&nbsp;: la structure osseuse
isolée, en bas et en arrière, séparée du tibia. Sur le genou gauche elle occupe les
coupes 3 à 8 et l'articulation n'apparaît qu'ensuite &mdash; ça, c'est net sur la
planche. Sur le genou droit je la situe vers 29&ndash;32&nbsp;; moins net, vérifie en
faisant défiler ci-dessous. <b>Le sens des deux piles, lui, ne dépend pas de ma
lecture</b>&nbsp;: il vient du tableau de coordonnées ci-dessus.</p></div>

<div class="danger"><h4>Le défaut, en une phrase</h4>
La coupe 0 est <b>latérale</b> sur un genou gauche et <b>médiale</b> sur un genou droit.
Sur les 170 études du bundle&nbsp;: dernière coupe = médiale sur les <b>86 gauches</b>,
latérale sur les <b>84 droits</b>. Sans correction, l'axe de profondeur pointe dans un
sens sur la moitié du corpus et dans l'autre sens sur le reste.</div>

<h3>À faire défiler toi-même</h3>
<p>Les deux piles côte à côte, un seul curseur, elles ne peuvent pas se désynchroniser.
Regarde les libellés sous chaque moitié.</p>
{pair_viewer("v_raw", raw, target, (l_first, l_last), (r_first, r_last), caption=geom)}

<h2>4. Ce que fait le correctif</h2>
<p><code>sagittal_flip</code> renverse l'ordre des coupes des genoux droits&nbsp;:
<code>image[::-1]</code>. Les pixels ne bougent pas, seul l'ordre change. Le genou gauche
n'est pas touché. Voici la planche du genou droit, renversée&nbsp;:</p>
<h3 style="color:{ORANGE}">Genou droit, après correctif</h3>
<div class="striprap"><img alt="" src="{strip_r_fixed}">{labels_row(idx_r[::-1], right.n, ORANGE)}</div>
{arrow_row(r_last, r_first, ORANGE)}
<p>Les deux planches se lisent maintenant dans le <b>même</b> sens anatomique&nbsp;:
latéral à gauche, médial à droite. C'est tout l'objet du correctif.</p>
{pair_viewer("v_fix", fix, target, (l_first, l_last), (r_last, r_first),
             caption="Genou droit renversé. " + geom)}
<div class="key"><b>Deuxième point.</b> Un <code>[::-1]</code> sur l'axe des coupes,
uniquement si le côté est connu et vaut R. Les genoux gauches ne bougent pas. Les études
dont le côté n'a pas pu être résolu ne bougent pas non plus &mdash; on ne retourne pas
ce qu'on ne sait pas.</div>

<h2>5. Le retournement qu'il ne faut PAS faire</h2>
<p>Voilà pourquoi la docstring de la baseline commençait par un avertissement. Le miroir
correct pour une coronale &mdash; retourner les pixels horizontalement,
<code>image[..., ::-1]</code> &mdash; appliqué à une sagittale donne ceci&nbsp;:</p>
<div class="duo">
  <figure style="margin:0"><div class="stage"><img alt="" src="{uri_l}"></div>
    <figcaption>Correct&nbsp;: rotule devant, creux poplité derrière.</figcaption></figure>
  <figure style="margin:0"><div class="stage"><img alt="" src="{uri_l_mirror}"></div>
    <figcaption><b style="color:var(--bad)">Faux</b>&nbsp;: la rotule est passée derrière.
    Ce n'est pas l'autre genou, c'est un genou impossible.</figcaption></figure>
</div>
<div class="key">La phrase d'origine disait&nbsp;: <i>«&nbsp;Sagittal stacks are not mirror
images of each other &mdash; so the channel order is reversed instead.&nbsp;»</i>
Premier morceau&nbsp;: ne fais pas ça. Second morceau&nbsp;: fais un renversement de
l'ordre des coupes. Le portage a gardé l'avertissement et jeté l'instruction.</div>

<h2>6. Pourquoi ça compte pour le modèle</h2>
<p>Le réseau ne voit jamais une pile entière. Il reçoit des fenêtres 2.5D&nbsp;: trois
coupes voisines empilées comme les trois canaux d'une image couleur. La même fenêtre sur
chacun des deux genoux, <b>avant</b> correctif&nbsp;:</p>
<h3 style="color:{BLUE}">Genou gauche</h3>
<div class="trio">
  <div><div class="stage"><img alt="" src="{trio_l[0]}"></div><div class="cap">canal 1 &mdash; vers {l_first}</div></div>
  <div><div class="stage"><img alt="" src="{trio_l[1]}"></div><div class="cap">canal 2</div></div>
  <div><div class="stage"><img alt="" src="{trio_l[2]}"></div><div class="cap">canal 3 &mdash; vers {l_last}</div></div>
</div>
<h3 style="color:{ORANGE}">Genou droit</h3>
<div class="trio">
  <div><div class="stage"><img alt="" src="{trio_r[0]}"></div><div class="cap">canal 1 &mdash; vers {r_first}</div></div>
  <div><div class="stage"><img alt="" src="{trio_r[1]}"></div><div class="cap">canal 2</div></div>
  <div><div class="stage"><img alt="" src="{trio_r[2]}"></div><div class="cap">canal 3 &mdash; vers {r_last}</div></div>
</div>
<p>Même position dans le tenseur, anatomie opposée. Le canal&nbsp;1 veut dire «&nbsp;plus
latéral&nbsp;» sur un genou et «&nbsp;plus médial&nbsp;» sur l'autre. L'encodeur dépense
donc de la capacité à devenir <b>invariant à un axe qui n'est pas de l'anatomie</b>, au
lieu de s'en servir. Cinq des douze cibles sont nommées par un côté.</p>

<div class="key"><b>Ce que ça ne veut pas dire.</b> Rien ne casse sans le correctif
&mdash; un modèle peut apprendre l'invariance, et c'est probablement ce que les nôtres
ont fait. La formulation exacte est&nbsp;: <b>requis nulle part, utile partout</b>. Et la
normalisation ne remplace pas l'augmentation, elle la complète&nbsp;: elle retire l'axe
sur les ~86&nbsp;% où le côté est connu, l'augmentation couvre les ~14&nbsp;% où il est
faux ou non résolu.</div>

<h2>7. Où c'est dans le code, et ce qui manque encore</h2>
<table>
<tr><th>Quoi</th><th>Où</th></tr>
<tr><td>Le booléen</td><td><code>PixelRules.sagittal_flip</code>, défaut <code>False</code>
 &mdash; le défaut, exprès&nbsp;: un manifeste écrit avant l'existence du champ se relit
 avec la valeur par défaut, et un <code>True</code> par défaut miroiterait en silence
 l'entrée des treize packages déjà entraînés.</td></tr>
<tr><td>L'opération</td><td><code>normalise_laterality(image, plane, side, flip_sagittal)</code>
 &mdash; <code>image[..., ::-1]</code> en coronale et axiale, <code>image[::-1]</code>
 en sagittale.</td></tr>
<tr><td>Le seul point d'appel</td><td><code>dicom/cache.py:208</code>, à la construction
 du cache. Activer la règle change le tag du cache, donc l'ancien fichier est refusé
 plutôt que réutilisé en silence.</td></tr>
<tr><td>Le test qui manquait</td><td>Il n'y en avait aucun sur cette fonction jusqu'au
 commit du correctif. L'invariant à écrire&nbsp;: <b>après normalisation, le latéral est
 au même bout de la pile pour toutes les études.</b> Aujourd'hui il est à un bout pour 86
 et à l'autre pour 84.</td></tr>
</table>

<footer>Construit par <code>tools/atlas/laterality_page.py</code> depuis les DICOM bruts.
Deux études réelles, dé-identifiées&nbsp;: toutes les images sont des data URI, la page
est auto-suffisante et n'est pas destinée à être publiée.</footer>
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--out", type=Path,
                    default=ROOT / "docs/atlas/lateralite_sagittale.html")
    args = ap.parse_args()

    body = build(Config())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(P.page("Latéralité sagittale", body))
    print(f"{args.out}  ({args.out.stat().st_size/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
