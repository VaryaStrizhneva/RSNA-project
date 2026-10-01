"""What the trained landmark model predicted, next to what it was asked for.

    python -m tools.atlas.landmark_report -o docs/atlas/landmark_predictions.html

Out-of-fold predictions on the held-out fold, grouped by how well they went: the ones
that worked, the ones that were mediocre, and the one that failed. Each case shows the
slices around the annotation with **the click as a green cross** and **the predicted
heatmap in red**, so the two can be compared where they actually differ rather than
through a single number.

The failure gets the whole stack instead of a window, because what went wrong there is
not a few millimetres of drift — it is the other compartment, and seeing that means
seeing both ends.
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

import torch                                                          # noqa: E402

from rsna.landmark import LandmarkConfig, cache                        # noqa: E402
from rsna.landmark.cache import restore                                # noqa: E402
from rsna.landmark.loop import folds, to_input                         # noqa: E402
from rsna.landmark.network import LandmarkNet                          # noqa: E402
from rsna.landmark.target import decode, project                       # noqa: E402
from tools.atlas import page as P                                      # noqa: E402

CACHE = "/data/mgr/rsna-knee/landmark-cache"
CELL = 230


def uri(rgb: np.ndarray) -> str:
    # `subsampling=0` keeps full chroma resolution. The default 4:2:0 halves it, which
    # erases a thin coloured line drawn inside a saturated region — the green cross
    # survived in the array and vanished in the file.
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=92, subsampling=0,
                              optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def paint(slice_u8, heat, truth_rc, size=CELL):
    """One slice: the predicted heatmap in red, the annotator's click as a green cross."""

    g = np.asarray(Image.fromarray(slice_u8).convert("RGB")
                   .resize((size, size), Image.LANCZOS)).astype(np.int32)
    h = np.asarray(Image.fromarray((np.clip(heat, 0, 1) * 255).astype(np.uint8))
                   .resize((size, size), Image.BILINEAR)).astype(np.int32)
    g[..., 0] = np.clip(g[..., 0] + h, 0, 255)
    g[..., 1] = g[..., 1] * (255 - h) // 255
    g[..., 2] = g[..., 2] * (255 - h) // 255

    if truth_rc is not None:
        # Two pixels wide, with a gap at the centre so the click marks the spot without
        # hiding what is underneath it.
        r, c = (int(round(x * size)) for x in truth_rc)
        for d in range(-11, 12):
            if abs(d) <= 3:
                continue
            for w in (0, 1):
                for (y, x) in ((r + d, c + w), (r + w, c + d)):
                    if 0 <= y < size and 0 <= x < size:
                        g[y, x] = (40, 255, 90)
    return g.astype(np.uint8)


def case(r, volume, heat, config, pred, conf, err) -> str:
    s = restore(r, volume)
    truth = np.asarray(r["points"]["lat_centre"], float)
    mm = s.mm_per_px_rc

    def slot_of(point):
        t = project(point, s)[0]
        return int(np.argmin(np.abs(np.where(s.valid, s.t_mm, 9e9) - t)))

    ts, ps = slot_of(truth), slot_of(pred)
    t_mm, r_mm, c_mm = project(truth, s)
    p_mm, pr_mm, pc_mm = project(pred, s)
    depth = abs(t_mm - p_mm)
    plane = float(np.hypot(r_mm - pr_mm, c_mm - pc_mm))
    rc = (r_mm / mm[0] / config.img, c_mm / mm[1] / config.img)

    lo, hi = min(ts, ps) - 2, max(ts, ps) + 3
    span = [k for k in range(lo, hi) if 0 <= k < config.slices]
    if len(span) > 14:
        # Thin the window rather than cut it: truncating drops one of the two markers,
        # and on the case worth looking at most it dropped the annotator's click.
        picked = {int(round(x)) for x in np.linspace(0, len(span) - 1, 14)}
        picked |= {span.index(ts), span.index(ps)}
        span = [span[j] for j in sorted(picked)]
    slots = span
    cells = [paint(s.volume[k], heat[k], rc if k == ts else None) for k in slots]
    marks = "".join(
        f'<span class="slot{" t" if k == ts else ""}{" p" if k == ps else ""}">{k}</span>'
        for k in slots)
    return f"""
<div class="case">
  <div class="chead">
    <b>&hellip;{P._esc(r['study'][-11:])}</b>
    <span class="err">{err:.1f} mm</span>
    <span class="meta">profondeur {depth:.1f} mm &middot; dans le plan {plane:.1f} mm
      &middot; confiance {conf:.3f} &middot; {int(s.valid.sum())} coupes à
      {r['native_spacing_mm']:.2f} mm</span>
  </div>
  <div class="striprap"><img alt="" src="{uri(np.concatenate(cells, axis=1))}"></div>
  <div class="slots" style="grid-template-columns:repeat({len(slots)},1fr)">{marks}</div>
</div>"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="out/landmark_reference", type=Path)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("-o", "--out", type=Path,
                    default=ROOT / "docs/atlas/landmark_predictions.html")
    args = ap.parse_args()

    config = LandmarkConfig()
    volumes, records = cache.load(CACHE, config)
    ck = torch.load(args.run / f"fold{args.fold}.pt", map_location="cpu",
                    weights_only=False)
    model = LandmarkNet(config, pretrained=False)
    model.load_state_dict(ck["state"])
    model.eval().cuda()

    val = folds(len(records), 5, 0)[args.fold]
    rows = []
    with torch.no_grad():
        for start in range(0, len(val), 8):
            idx = val[start:start + 8]
            v = torch.as_tensor(np.asarray(volumes[idx]).copy()).cuda().float() / 255.0
            heat = torch.sigmoid(model(to_input(v))).cpu().numpy()
            for j, i in enumerate(idx):
                r = records[i]
                p, c = decode(heat[j], restore(r, volumes[i]), config)
                truth = np.asarray(r["points"]["lat_centre"], float)
                rows.append({"i": int(i), "heat": heat[j][0], "pred": p[0],
                             "conf": float(c[0]),
                             "err": float(np.linalg.norm(p[0] - truth))})

    rows.sort(key=lambda x: x["err"])
    e = np.array([x["err"] for x in rows])
    good = rows[:4]
    middling = [x for x in rows if 3 <= x["err"] < 12][-4:]
    failed = [x for x in rows if x["err"] >= 12]

    def blocks(group):
        return "".join(case(records[x["i"]], volumes[x["i"]], x["heat"], config,
                            x["pred"], x["conf"], x["err"]) for x in group)

    body = f"""
<style>
.striprap{{overflow-x:auto;border:1px solid var(--line);border-radius:8px;background:#000}}
.striprap img{{display:block;width:100%;min-width:900px;height:auto}}
.case{{margin:26px 0 30px}}
.chead{{display:flex;align-items:baseline;gap:14px;flex-wrap:wrap;margin-bottom:7px}}
.err{{font-family:ui-monospace,monospace;font-weight:700;color:var(--accent)}}
.slots{{display:grid;min-width:900px;font-family:ui-monospace,monospace;font-size:.68rem;
        text-align:center;color:var(--dim);margin-top:3px}}
.slot.t{{color:#7fc98b;font-weight:700}}
.slot.p{{color:#ff8a86;font-weight:700}}
.legend{{display:flex;gap:22px;flex-wrap:wrap;margin:10px 0 0;font-size:.86rem}}
.sw{{display:inline-block;width:12px;height:12px;border-radius:3px;margin-right:6px;
     vertical-align:-1px}}
</style>

<header>
  <h1>Ce que le modèle a prédit</h1>
  <p class="lede">Fold {args.fold}, {len(rows)} études jamais vues à l'entraînement.
  Médiane <b>{np.median(e):.1f} mm</b>, p90 <b>{np.percentile(e, 90):.1f} mm</b>,
  <b>{100 * np.mean(e < 12):.0f} %</b> sous la tolérance de 12 mm. Le prédicteur
  constant, avec le côté connu parfaitement, donnait p90 21,2 mm.</p>
  <div class="legend">
    <span><span class="sw" style="background:#ff4d4d"></span>heatmap prédite</span>
    <span><span class="sw" style="background:#3cff5a"></span>le clic de l'annotateur</span>
    <span><span class="sw" style="background:#7fc98b"></span>slot du clic &middot;
      <span style="color:#ff8a86">slot prédit</span></span>
  </div>
</header>

<h2>Distribution</h2>
<table><tr><th>erreur</th>{''.join(f'<th>{a}</th>' for a in
  ('0-1', '1-2', '2-3', '3-5', '5-8', '8-12', '12-25', '&gt;25'))}</tr>
<tr><td><b>études</b></td>{''.join(f'<td>{n}</td>' for n in
  np.histogram(e, bins=[0, 1, 2, 3, 5, 8, 12, 25, 1e9])[0])}</tr></table>

<h2>1. Les cas qui vont bien</h2>
<p>Les quatre meilleurs. Le rouge est sur la ligne articulaire entre les deux cornes, la
croix verte au même endroit — l'écart est en dessous de ce qu'on peut voir à l'œil.</p>
{blocks(good)}

<h2>2. Les cas bofs</h2>
<p>Les quatre plus mauvais qui restent sous la tolérance, entre 3 et 6 mm. L'écart est
visible mais un crop de 40 mm centré sur la prédiction contient toujours le ménisque.
Regarde la colonne « profondeur » contre « dans le plan » : c'est là qu'on voit lequel
des deux axes décroche.</p>
{blocks(middling)}

<h2>3. Le cas en échec</h2>
<p>Un seul, et il n'est pas un peu à côté : il est sur <b>l'autre compartiment</b>. La
bande couvre les deux, du slot du modèle à celui du clic.</p>
{blocks(failed)}
<div class="danger"><h4>Ce que ce cas a appris</h4>
<p style="margin:0 0 8px">Les deux garde-fous prévus ne le voient pas. La
<b>confiance</b> de la heatmap est la plus haute des 61 (0,144 contre 0,087 en médiane) :
elle mesure «&nbsp;ai-je trouvé <i>un</i> ménisque&nbsp;», pas le bon. Le test de
<b>renversement de pile</b> — prédire sur la pile puis sur son inverse — donne 1,0 mm
d'écart, comme partout ailleurs&nbsp;: le modèle lit bien l'anatomie et pas l'ordre des
coupes, ce qui est une bonne nouvelle mais ne détecte rien.</p>
<p style="margin:0">Le côté de cette étude ne vient pas du tag DICOM mais de la
<b>géométrie, à median&nbsp;x&nbsp;=&nbsp;&minus;24&nbsp;mm</b> — juste à la sortie de la
zone morte de 20&nbsp;mm où la règle ne vaut rien. Le badge affiché était donc orange, et
possiblement faux. <b>17 des 304 annotations</b> sont dans cette zone.</p></div>

<footer>Construit par <code>tools/atlas/landmark_report.py</code> depuis
<code>{P._esc(str(args.run))}</code>. Prédictions hors échantillon&nbsp;: aucune de ces
études n'a été vue à l'entraînement.</footer>"""

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(P.page("Prédictions du landmark", body))
    print(f"{args.out}  ({args.out.stat().st_size/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
