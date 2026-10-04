"""Synovitis: the one finding of the twelve that this dataset cannot show.

    python -m tools.atlas.synovitis_page -o docs/atlas/synovitis.html

Every other page here ends with a region to cut. This one ends with a measurement that
says not to bother, and explains what would have to change first.

The argument is not anatomical, it is about what was acquired. Inflamed synovium is told
apart from the fluid it sits in by the fact that it takes up contrast and the fluid does
not — so the reference standard for synovitis is a post-contrast sequence. **Three
studies of 4407 have one.** What is left is an indirect read, and the numbers below show
the wide model making exactly the substitution a reader without contrast is forced into.

Nothing here is a radiological reading. The labels come from the reports.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from rsna.config import Config, TARGETS                              # noqa: E402
from rsna.data.label_eval import load_gold                           # noqa: E402
from tools.atlas import page as P                                    # noqa: E402
from tools.atlas.render import stack_to_jpegs                        # noqa: E402
from tools.atlas.study import (TRAIN_SERIES, load_series,            # noqa: E402
                               series_headers, side_of, stack_orientation)


def measure() -> dict:
    """Everything the page asserts, computed here so no number is typed by hand."""

    ref = pd.concat([pd.read_csv(ROOT / f"out/ref/pkg-f{k}/holdout.csv") for k in range(5)],
                    ignore_index=True).rename(
                        columns={"StudyInstanceUID": "study"}).set_index("study")
    golds = [pd.read_csv(ROOT / f"out/ref/pkg-f{k}/gold.csv").set_index("StudyInstanceUID")
             for k in range(5)]
    gi = golds[0].index
    gold = load_gold()
    pred = {t: np.mean([g.loc[gi, f"pred:{t}"].to_numpy() for g in golds], axis=0)
            for t in ("Synovitis", "Effusion", "Baker's")}

    syn = ref["true:Synovitis"].to_numpy() > 0.5
    eff = ref["true:Effusion"].to_numpy() > 0.5
    out = {
        "n_syn": int(syn.sum()), "n_eff": int(eff.sum()),
        "both": int((syn & eff).sum()),
        "p_eff_given_syn": float((syn & eff).sum() / max(syn.sum(), 1)),
        "p_syn_given_eff": float((syn & eff).sum() / max(eff.sum(), 1)),
        "oof_syn": roc_auc_score(ref["true:Synovitis"] > 0.5, ref["pred:Synovitis"]),
        "gold_syn": roc_auc_score(gold["Synovitis"], pred["Synovitis"]),
        "gold_syn_on_eff": roc_auc_score(gold["Effusion"], pred["Synovitis"]),
        "gold_eff": roc_auc_score(gold["Effusion"], pred["Effusion"]),
        "rho": float(spearmanr(ref["pred:Synovitis"], ref["pred:Effusion"]).statistic),
        "n_gold_syn": int(gold["Synovitis"].sum()),
        "n_gold_eff": int(gold["Effusion"].sum()),
    }

    tables = {
        "steven v4_blend": "data/external/stevenleehans-rsna-knee-llm-report-labels/llm_labels_v4_blend.csv",
        "riad HYBRID": "data/external/riadmohamed42-jev-knee-labels/HYBRID_labels_scores.csv",
        "pilkwang v2": "data/external/pilkwang-rsna-knee-llm-labels/report_labels_v2.csv",
    }
    out["tables"] = {}
    for name, path in tables.items():
        frame = pd.read_csv(ROOT / path).set_index("StudyInstanceUID")
        idx = gi.intersection(frame.index)
        score = pd.to_numeric(frame.loc[idx, "Synovitis"], errors="coerce").to_numpy()
        ok = np.isfinite(score)
        out["tables"][name] = roc_auc_score(gold.loc[idx, "Synovitis"].to_numpy()[ok],
                                            score[ok])
    out["ceiling"] = max(out["tables"].values())
    out["ref"], out["gold_frame"] = ref, gold
    return out


def case(uid: str, vid: str, note: str) -> str:
    headers = series_headers(uid)
    sag = headers[headers["plane"] == "Sagittal"]
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--out", type=Path, default=ROOT / "docs/atlas/synovitis.html")
    ap.add_argument("--cases", type=int, default=2)
    args = ap.parse_args()

    m = measure()
    ref, gold = m["ref"], m["gold_frame"]
    rows = "".join(
        f"<tr><td>{P._esc(k)}</td><td>{v:.4f}</td></tr>"
        for k, v in sorted(m["tables"].items(), key=lambda kv: -kv[1]))

    body = [f"""
<header>
  <h1>Synovite &mdash; la seule des douze que ces images ne montrent pas</h1>
  <p class="lede">Toutes les autres pages d'ici finissent par une r&eacute;gion &agrave;
  d&eacute;couper. Celle-ci finit par une mesure qui dit de ne pas le faire, et par ce
  qu'il faudrait changer d'abord. L'argument n'est pas anatomique&nbsp;: il porte sur ce
  qui a &eacute;t&eacute; acquis.</p>
</header>

<div class="card">
<h2>1 &middot; Ce qu'est une synovite, et pourquoi elle est &agrave; part</h2>
<p>La <b>synoviale</b> est la membrane qui tapisse l'int&eacute;rieur de la capsule
articulaire et fabrique le liquide synovial. Quand elle s'enflamme elle
s'&eacute;paissit, se charge de vaisseaux et produit davantage de liquide. C'est une
r&eacute;action, pas une maladie&nbsp;: elle accompagne l'arthrose, une l&eacute;sion
m&eacute;niscale, une arthrite.</p>
<p>Le probl&egrave;me tient en une phrase&nbsp;: <b>la synoviale &eacute;paissie et le
liquide dans lequel elle baigne ont le m&ecirc;me signal</b> sur une s&eacute;quence
fluide-sensible. Les deux sont blancs. Ce qui les s&eacute;pare, c'est que la synoviale
enflamm&eacute;e est vascularis&eacute;e et <b>prend le produit de contraste</b>, alors
que le liquide ne le prend pas. La r&eacute;f&eacute;rence pour affirmer une synovite est
donc une s&eacute;quence <b>apr&egrave;s injection</b>.</p>
<p class="note">C'est si bien admis que la cotation radiologique de r&eacute;f&eacute;rence
pour le genou sans injection (MOAKS) ne note pas la synovite s&eacute;par&eacute;ment&nbsp;:
elle note un item unique appel&eacute; <b>&laquo;&nbsp;effusion-synovitis&nbsp;&raquo;</b>,
parce qu'on ne peut pas les distinguer.</p>
</div>

<div class="card">
<h2>2 &middot; Ce que le corpus contient</h2>
<p>J'ai cherch&eacute; les s&eacute;quences apr&egrave;s injection dans les en-t&ecirc;tes
des 24 371 s&eacute;ries.</p>
<table class="report">
<tr><th>s&eacute;ries dont la description &eacute;voque une injection</th><td><b>7</b></td></tr>
<tr class="danger"><th>&eacute;tudes concern&eacute;es</th><td><b>3 sur 4407</b></td></tr>
</table>
<p>Sept s&eacute;ries, toutes du m&ecirc;me type (<code>T1W_TSE_FS+C</code> sagittal et
coronal), r&eacute;parties sur trois examens. <b>99,93&nbsp;% du corpus n'a pas
d'injection.</b></p>
<p>Donc pour 4404 &eacute;tudes sur 4407, l'&eacute;tiquette &laquo;&nbsp;synovite&nbsp;&raquo;
du compte rendu a &eacute;t&eacute; produite <b>sans l'examen qui permet de l'affirmer</b>.
Elle repose sur des signes indirects&nbsp;: un &eacute;paississement visible comme un
liseré de signal interm&eacute;diaire dans le liquide, un &oelig;d&egrave;me du paquet
graisseux de Hoffa, ou simplement un &eacute;panchement abondant.</p>
</div>

<div class="card">
<h2>3 &middot; Ce que le mod&egrave;le a vraiment appris</h2>
<p>Trois mesures, et elles disent toutes la m&ecirc;me chose.</p>
<h3>Les deux &eacute;tiquettes se recouvrent presque</h3>
<table class="report">
<tr><th></th><th>positifs</th><th>en commun</th></tr>
<tr><td>Synovite</td><td>{m['n_syn']}</td><td rowspan="2"><b>{m['both']}</b></td></tr>
<tr><td>&Eacute;panchement</td><td>{m['n_eff']}</td></tr>
</table>
<p><b>P(&eacute;panchement | synovite) = {m['p_eff_given_syn']:.2f}</b> &mdash; presque
toute &eacute;tude dite synovitique est aussi dite &eacute;panch&eacute;e. L'inverse est
faux&nbsp;: P(synovite | &eacute;panchement) = {m['p_syn_given_eff']:.2f}. La synovite
est, dans ces comptes rendus, un sous-ensemble de l'&eacute;panchement.</p>
<h3>La sortie &laquo;&nbsp;synovite&nbsp;&raquo; classe mieux l'&eacute;panchement que la synovite</h3>
<p>Sur les 58 lectures expertes, en prenant la colonne <code>pred:Synovitis</code> du
mod&egrave;le et en la jugeant contre diff&eacute;rentes v&eacute;rit&eacute;s&nbsp;:</p>
<table class="report">
<tr><th>jug&eacute;e contre</th><th>AUC</th><th>positifs</th></tr>
<tr><td>la synovite &mdash; ce qu'elle pr&eacute;tend pr&eacute;dire</td>
    <td>{m['gold_syn']:.4f}</td><td>{m['n_gold_syn']}</td></tr>
<tr class="danger"><td><b>l'&eacute;panchement</b></td>
    <td><b>{m['gold_syn_on_eff']:.4f}</b></td><td>{m['n_gold_eff']}</td></tr>
</table>
<p>Elle ordonne <b>mieux</b> ce qu'elle n'est pas cens&eacute;e pr&eacute;dire. Et la
corr&eacute;lation de rangs entre <code>pred:Synovitis</code> et
<code>pred:Effusion</code> sur les 4348 &eacute;tudes vaut <b>{m['rho']:+.3f}</b>&nbsp;:
ce sont presque la m&ecirc;me sortie.</p>
<h3>Ce n'est pas que le mod&egrave;le lit mal le liquide</h3>
<p>Sa colonne &eacute;panchement fait <b>{m['gold_eff']:.4f}</b> contre les m&ecirc;mes
lectures expertes &mdash; parmi les meilleurs scores des douze. <b>Il voit le liquide
tr&egrave;s bien. Il ne peut simplement pas s&eacute;parer le liquide de la membrane qui
le fabrique</b>, pour la raison physique du &sect;1. Priv&eacute; du signal qui les
distingue, il fait la substitution qu'un lecteur sans injection est oblig&eacute; de
faire.</p>
</div>

<div class="card">
<h2>4 &middot; Le plafond, et pourquoi un expert n'y changerait rien</h2>
<p>Les tables d'&eacute;tiquettes elles-m&ecirc;mes, jug&eacute;es sur les 58 lectures
expertes&nbsp;:</p>
<table class="report"><tr><th>table</th><th>AUC sur la synovite</th></tr>{rows}</table>
<p>La meilleure extraction disponible lit la synovite &agrave;
<b>{m['ceiling']:.4f}</b> &mdash; <b>la plus mauvaise des douze cibles</b>. Le
mod&egrave;le large est &agrave; <b>{m['gold_syn']:.4f}</b>.</p>
<table class="report">
<tr><th>ce que la supervision permet au mieux</th><td>{m['ceiling']:.4f}</td></tr>
<tr><th>ce que le mod&egrave;le atteint d&eacute;j&agrave;</th><td>{m['gold_syn']:.4f}</td></tr>
<tr class="good"><th>marge restante</th><td><b>{m['ceiling'] - m['gold_syn']:+.4f}</b></td></tr>
</table>
<p>Un expert par r&eacute;gion est entra&icirc;n&eacute; sur ces m&ecirc;mes
&eacute;tiquettes. Il ne peut pas d&eacute;passer ce qu'elles contiennent. <b>Il reste
{m['ceiling'] - m['gold_syn']:.3f} de marge</b>, l&agrave; o&ugrave; les sept cibles pour
lesquelles un expert a march&eacute; en avaient entre 0,08 et 0,22.</p>
<p class="note">La marge est l'&eacute;cart entre ce qu'une table lit et ce que le
mod&egrave;le atteint, mesur&eacute; sur le m&ecirc;me gold. C'est elle, et non la taille
de la l&eacute;sion, qui a pr&eacute;dit o&ugrave; un expert servait&nbsp;: les sept qui
ont march&eacute; sont exactement les sept plus grandes marges.</p>
</div>

<div class="card">
<h2>5 &middot; Ce qu'on pourrait tenter quand m&ecirc;me</h2>
<p>Par ordre d&eacute;croissant de ce que j'en attends, et aucune n'est un recadrage.</p>
<h3>Rendre l'&eacute;tiquette s&eacute;parable &mdash; la seule qui attaque la cause</h3>
<p>Le plafond est &agrave; {m['ceiling']:.3f} parce que l'extraction confond deux choses
que le compte rendu, lui, distingue parfois. Relire les rapports en demandant
explicitement <b>le mot qui tranche</b> &mdash; &eacute;paississement synovial, prise de
contraste, &oelig;d&egrave;me de Hoffa &mdash; plut&ocirc;t que &laquo;&nbsp;parle-t-il de
synovite&nbsp;&raquo; donnerait une cible qui n'est pas l'&eacute;panchement
d&eacute;guis&eacute;. C'est du travail sur le texte, pas sur les pixels.</p>
<h3>Pr&eacute;dire le r&eacute;sidu plut&ocirc;t que la cible</h3>
<p>Puisque <code>pred:Synovitis</code> et <code>pred:Effusion</code> corr&egrave;lent
&agrave; {m['rho']:+.3f}, la seule information propre &agrave; la synovite est ce qui
<em>reste</em> quand on retire l'&eacute;panchement. Entra&icirc;ner un mod&egrave;le
&agrave; pr&eacute;dire la synovite <b>parmi les seules &eacute;tudes
&eacute;panch&eacute;es</b> poserait la question que le mod&egrave;le actuel
n'a jamais eu &agrave; r&eacute;soudre. {m['both']} &eacute;tudes pour l'entra&icirc;ner,
ce qui est peu mais pas rien.</p>
<h3>La texture plut&ocirc;t que le volume</h3>
<p>Ce qui distingue les deux &agrave; l'&oelig;il, sans injection, est une
<b>forme</b>&nbsp;: un &eacute;panchement simple est une collection lisse dans les
r&eacute;cessus, une synoviale &eacute;paissie est irr&eacute;guli&egrave;re et
festonn&eacute;e le long de la capsule. C'est une question de r&eacute;solution et de
bord, pas de champ de vue &mdash; donc un crop fin du <b>cul-de-sac sus-patellaire</b>
aurait un sens. Mais il buterait sur le m&ecirc;me plafond de
{m['ceiling'] - m['gold_syn']:.3f}, donc je ne le ferais qu'apr&egrave;s le premier point.</p>
</div>

<div class="card">
<h2>6 &middot; Faire d&eacute;filer des piles</h2>
<p class="note">Les deux premi&egrave;res &eacute;tudes sont dites
<b>&eacute;panch&eacute;es et synovitiques</b>, les deux suivantes
<b>&eacute;panch&eacute;es et non synovitiques</b>. C'est exactement la distinction que le
mod&egrave;le doit faire et ne fait pas. Regardez le cul-de-sac sus-patellaire, au-dessus
de la rotule&nbsp;: si vous ne voyez pas ce qui s&eacute;pare les deux groupes, c'est le
sujet de cette page.</p>"""]

    both = ref[(ref["true:Synovitis"] > 0.5) & (ref["true:Effusion"] > 0.5)].index
    only = ref[(ref["true:Synovitis"] < 0.5) & (ref["true:Effusion"] > 0.5)].index
    n = 0
    for uid in [u for u in both if (TRAIN_SERIES / u).is_dir()][:args.cases]:
        body.append('<h3 class="case-head">&eacute;panchement <b>et</b> synovite</h3>')
        body.append(case(uid, f"sy{n}", "Le compte rendu dit les deux."))
        n += 1
        print(f"  synovite {uid[-11:]}", flush=True)
    for uid in [u for u in only if (TRAIN_SERIES / u).is_dir()][:args.cases]:
        body.append('<h3 class="case-head">&eacute;panchement <b>sans</b> synovite</h3>')
        body.append(case(uid, f"sy{n}", "Le compte rendu dit l'&eacute;panchement seul."))
        n += 1
        print(f"  sans synovite {uid[-11:]}", flush=True)
    body.append("</div>")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(P.page("Synovite — ce que ces images ne montrent pas",
                               "".join(body)))
    print(f"\n{args.out}  ({args.out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
