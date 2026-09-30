"""What the landmark model is actually shown, and what it is asked to answer.

    python -m tools.atlas.landmark_page -o docs/atlas/landmark_explanation.html

A target here is a 3D Gaussian written in millimetres, which is easy to state and hard
to picture. The page shows three things that are worth checking with eyes rather than
with a test: that the blob sits on the joint line where the annotator clicked, that it
keeps the same *physical* size across studies whose slice spacing differs by 2.5x, and
that every augmentation moves the pixels and the target together.

That last one is the reason this page exists. An augmentation applied to the image and
forgotten on the label does not crash, does not show up in a loss curve, and teaches the
model to predict a few millimetres off for ever.
"""

from __future__ import annotations

import argparse
import base64
import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import torch                                                    # noqa: E402

from rsna.landmark import LandmarkConfig, augment as A, cache    # noqa: E402
from rsna.landmark.cache import restore                          # noqa: E402
from rsna.landmark.target import encode                          # noqa: E402
from tools.atlas import page as P                                # noqa: E402

CACHE = "/data/mgr/rsna-knee/landmark-cache"


def uri(rgb: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=88, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def overlay(slice_u8: np.ndarray, heat: np.ndarray, size: int) -> np.ndarray:
    """One slice with its target painted over it in red."""

    g = np.asarray(Image.fromarray(slice_u8).convert("RGB").resize((size, size),
                                                                   Image.LANCZOS))
    h = np.asarray(Image.fromarray((np.clip(heat, 0, 1) * 255).astype(np.uint8))
                   .resize((size, size), Image.BILINEAR)).astype(np.int32)
    out = g.astype(np.int32)
    out[..., 0] = np.clip(out[..., 0] + h, 0, 255)
    out[..., 1] = out[..., 1] * (255 - h) // 255
    out[..., 2] = out[..., 2] * (255 - h) // 255
    return out.astype(np.uint8)


def strip(volume, heat, slots, size=190, paint=True) -> str:
    cells = [overlay(volume[k], heat[k], size) if paint
             else np.asarray(Image.fromarray(volume[k]).convert("RGB")
                             .resize((size, size), Image.LANCZOS))
             for k in slots]
    return uri(np.concatenate(cells, axis=1))


def profile(heat: np.ndarray, valid: np.ndarray, peak: int, span: int = 9) -> str:
    """The target's mass per slot, as a bar per slot."""

    mass = heat.sum(axis=(1, 2))
    top = max(mass.max(), 1e-9)
    lo, hi = max(0, peak - span), min(len(mass), peak + span + 1)
    bars = []
    for k in range(lo, hi):
        pct = 100 * mass[k] / top
        cls = "bar" + (" pad" if not valid[k] else "")
        bars.append(f'<div class="prow"><span class="pk">{k}</span>'
                    f'<span class="{cls}" style="width:{pct:.1f}%"></span>'
                    f'<span class="pv">{mass[k]:.2f}</span></div>')
    return f'<div class="prof">{"".join(bars)}</div>'


def block(rec, vol, config, title: str, note: str) -> str:
    s = restore(rec, vol)
    heat = encode(rec["points"], s, config)[0]
    peak = int(np.unravel_index(int(heat.argmax()), heat.shape)[0])
    slots = [k for k in range(peak - 3, peak + 3) if 0 <= k < config.slices]
    n_lit = int((heat.sum(axis=(1, 2)) > 0.01 * heat.sum(axis=(1, 2)).max()).sum())
    return f"""
<h3>{P._esc(title)}</h3>
<p class="meta">&hellip;{P._esc(rec['study'][-11:])} &middot;
  {int(s.valid.sum())} coupes réelles sur {config.slices} &middot;
  espacement <b>{rec['native_spacing_mm']:.2f} mm</b> &middot;
  {'décimée 1/' + str(rec['stride']) + ' depuis ' + str(rec['native_n']) + ' coupes'
   if rec['stride'] > 1 else 'série 2D, prise telle quelle'} &middot;
  la cible éclaire <b>{n_lit} coupes</b></p>
<p class="note">{note}</p>
<div class="striprap"><img alt="" src="{strip(s.volume, heat, slots, paint=False)}"></div>
<div class="striprap" style="margin-top:6px"><img alt="" src="{strip(s.volume, heat, slots)}"></div>
{profile(heat, s.valid, peak)}"""


def aug_block(rec, vol, config) -> str:
    """The same study under each augmentation, target painted on."""

    s = restore(rec, vol)
    heat = encode(rec["points"], s, config)[0][None]
    v = torch.as_tensor(s.volume[None]).float() / 255.0
    h = torch.as_tensor(heat[None])

    def shown(vt, ht, label, why):
        hh = ht[0, 0].numpy()
        peak = int(np.unravel_index(int(hh.argmax()), hh.shape)[0])
        slots = [k for k in range(peak - 2, peak + 3) if 0 <= k < config.slices]
        vv = (vt[0].clamp(0, 1).numpy() * 255).astype(np.uint8)
        return (f'<h4>{P._esc(label)}</h4><p class="note">{why}</p>'
                f'<div class="striprap"><img alt="" src="{strip(vv, hh, slots)}"></div>'
                f'<p class="meta">pic au slot <b>{peak}</b></p>')

    rev_v, rev_h = A.reverse(v, h)
    dec_v, dec_h = A.decimate(v, h, 2)
    rot_v, rot_h = A.in_plane(v, h, torch.tensor([0.26]), torch.tensor([[0.10, -0.06]]),
                              torch.tensor([0.92]))
    return "".join([
        shown(v, h, "Tel quel", "La référence."),
        shown(rev_v, rev_h, "Pile renversée",
              "L'ordre des coupes s'inverse, donc le latéral change de bout. Le point "
              "dans le patient, lui, n'a pas bougé : c'est une permutation de l'entrée, "
              "et un modèle qui avait appris «&nbsp;latéral = indice bas&nbsp;» est "
              "maintenant faux une fois sur deux."),
        shown(dec_v, dec_h, "Décimée 1/2",
              "Une coupe sur deux, ce qui double l'espacement. Rien n'est inventé, on "
              "jette. La masse de la cible est sommée sur les coupes fusionnées plutôt "
              "qu'échantillonnée, donc un pic tombé sur une coupe supprimée n'est pas "
              "perdu."),
        shown(rot_v, rot_h, "Rotation 15°, translation, échelle 0,92",
              "Pixels et cible subissent exactement la même transformation affine. "
              "C'est ce qu'il faut vérifier de l'œil : le rouge doit rester collé à "
              "l'anatomie, pas glisser à côté."),
    ])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--out", type=Path,
                    default=ROOT / "docs/atlas/landmark_explanation.html")
    args = ap.parse_args()

    config = LandmarkConfig()
    vol, rec = cache.load(CACHE, config)
    sp = np.array([r["native_spacing_mm"] for r in rec])
    deep = np.array([r["stride"] > 1 for r in rec])

    picks = [
        (int(np.argmin(np.where(deep, 9e9, sp))), "L'espacement le plus fin du corpus"),
        (int(np.argmin(np.abs(sp - 3.3) + deep * 9e9)), "L'espacement médian"),
        (int(np.argmax(np.where(deep, -1, sp))), "L'espacement le plus grossier"),
        (int(np.argmax(deep)), "Une série 3D, décimée par pas entier"),
    ]
    notes = ["", "", "", ""]

    body = [f"""<header><h1>La cible du modèle de landmark</h1>
<p class="lede">Un point cliqué devient une gaussienne 3D de {config.sigma_mm:.0f} mm
écrite en millimètres, sur une grille de {config.slices} slots de
{config.grid}&times;{config.grid} cellules. En haut les coupes brutes, en bas les mêmes
avec la cible peinte en rouge.</p></header>
<h2>1. Une cible, sur quatre études différentes</h2>
<p>Ce qui doit rester constant d'une étude à l'autre, c'est la <b>taille physique</b> de
la tache &mdash; pas le nombre de coupes qu'elle éclaire. Un genou échantillonné tous les
2,2 mm et un autre tous les 5 mm doivent recevoir la même gaussienne de
{config.sigma_mm:.0f} mm, donc la première allume plus de coupes que la seconde. C'est
exactement ce que le profil de droite permet de vérifier.</p>"""]
    for i, (k, title) in enumerate(picks):
        body.append(block(rec[k], vol[k], config, title, notes[i]))

    body.append("""<h2>2. L'augmentation déplace les deux ensemble</h2>
<p>La même étude sous chaque transformation. Si le rouge se décolle de l'anatomie sur
l'une d'elles, c'est un défaut qui ne se verrait nulle part ailleurs : il ne plante pas,
il n'apparaît pas dans la courbe de perte, et il apprend au modèle à viser à côté.</p>""")
    body.append(aug_block(rec[picks[1][0]], vol[picks[1][0]], config))

    extra = """
<style>
.striprap{overflow-x:auto;border:1px solid var(--line);border-radius:8px;background:#000}
.striprap img{display:block;width:100%;min-width:760px;height:auto}
.prof{margin:14px 0 4px;font-family:ui-monospace,monospace;font-size:.72rem}
.prow{display:flex;align-items:center;gap:8px;line-height:1.45}
.pk{width:22px;text-align:right;color:var(--dim)}
.pv{color:var(--dim)}
.bar{height:9px;background:var(--accent);border-radius:2px;min-width:1px}
.bar.pad{background:#3a3f46}
h4{margin:26px 0 4px;font-size:1rem;color:var(--warn)}
</style>"""
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(P.page("Cibles du landmark", extra + "".join(body)))
    print(f"{args.out}  ({args.out.stat().st_size/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
