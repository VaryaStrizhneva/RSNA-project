"""Assemble the numbers, then render them.

`build` does the arithmetic and returns a plain dict — testable, printable, and the
same object whether it ends up as HTML or as ten lines on a terminal. `render_html`
turns it into one self-contained file, and self-contained here means genuinely one
file: the charts are inline SVG and CSS rather than embedded images, so there is
nothing to fetch, nothing to lose alongside it, and nothing that goes soft when the
page is wider than the figure was drawn.
"""

from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import figures as F
from . import metrics as M


def build(records: list, seed: int = 0) -> dict:
    """Every number the report states, computed once."""

    fold_aucs = np.array([r.best_holdout_auc for r in records], float)
    y, p = M.pool(records)
    pooled = M.macro_auc(y, p) if len(y) else float("nan")
    interval = M.bootstrap_macro(y, p, seed=seed) if len(y) else (np.nan, np.nan)
    rows = M.target_table(y, p) if len(y) else []
    finite = fold_aucs[np.isfinite(fold_aucs)]

    return {
        "experiment": records[0].experiment if records else "?",
        "labels": records[0].labels if records else "",
        "split": records[0].split if records else "",
        "written": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_folds": len(records),
        "n_studies": int(len(y)),
        "epochs": max((len(r.history) for r in records), default=0),
        "fold_aucs": fold_aucs.tolist(),
        "fold_mean": float(finite.mean()) if len(finite) else float("nan"),
        "fold_std": float(finite.std()) if len(finite) > 1 else float("nan"),
        "pooled_auc": pooled,
        "interval": [float(interval[0]), float(interval[1])],
        "targets": rows,
        "flags": M.flags(records, rows, fold_aucs),
        "_records": records,
        "_pooled_arrays": (y, p),
    }


def summary_text(report: dict) -> str:
    """The ten lines worth having without opening a browser."""

    lo, hi = report["interval"]
    lines = [
        f"{report['experiment']}  —  {report['n_folds']} folds, "
        f"{report['n_studies']} out-of-fold studies, {report['epochs']} epochs",
        f"  out-of-fold macro AUC   {report['pooled_auc']:.4f}"
        f"   (95% {lo:.3f}-{hi:.3f})",
        f"  per fold                {report['fold_mean']:.4f} "
        f"+/- {report['fold_std']:.4f}   "
        + "  ".join(f"f{i}:{v:.3f}" for i, v in enumerate(report["fold_aucs"])),
    ]
    scored = [r for r in report["targets"] if np.isfinite(r["auc"])]
    if scored:
        worst = sorted(scored, key=lambda r: r["auc"])[:3]
        best = sorted(scored, key=lambda r: -r["auc"])[:3]
        lines.append("  best    " + ", ".join(
            f"{r['target']} {r['auc']:.3f}" for r in best))
        lines.append("  worst   " + ", ".join(
            f"{r['target']} {r['auc']:.3f}" for r in worst))
    errors = [f for f in report["flags"] if f["level"] == "error"]
    warns = [f for f in report["flags"] if f["level"] == "warn"]
    lines.append(f"  flags   {len(errors)} error, {len(warns)} warning")
    for flag in errors[:3]:
        lines.append(f"    ! {flag['text']}")
    return "\n".join(lines)


_CSS = """
/* Dark by design, not by preference: this page is read beside a terminal. Every
   colour a chart uses is declared here and referenced from the SVG, so the palette
   exists once rather than in two files that drift apart. */
:root{
 --bg:#12151c; --panel:#181d26; --line:#29303d; --raise:#1f2632;
 --ink:#e9ebf1; --dim:#9aa4b8; --faint:#697285;
 --accent:#6aa8ff; --danger:#ff7d74; --warn:#ffb454; --ok:#5fd6a0;
 --f0:#6aa8ff; --f1:#ffb454; --f2:#5fd6a0; --f3:#ff7d74; --f4:#c69bff;
 --mono:ui-monospace,SFMono-Regular,"SF Mono",Menlo,Consolas,monospace;
}
*{box-sizing:border-box}
html{background:var(--bg)}
body{margin:0;background:var(--bg);color:var(--ink);-webkit-font-smoothing:antialiased;
 text-rendering:optimizeLegibility;
 font:15px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
main{max-width:1000px;margin:0 auto;padding:56px 24px 96px}

h1{font-size:27px;letter-spacing:-.015em;margin:0 0 6px;font-weight:650}
h2{font-size:11.5px;text-transform:uppercase;letter-spacing:.14em;color:var(--faint);
 margin:54px 0 18px;font-weight:700}
.sub{color:var(--dim);font-size:13.5px;margin-bottom:32px}
.sub b{color:var(--ink);font-weight:500}

.head{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));
 gap:14px;margin:0 0 18px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;
 padding:18px 20px}
.card .k{font-size:10.5px;text-transform:uppercase;letter-spacing:.1em;
 color:var(--faint);font-weight:700}
.card .v{font-size:31px;font-family:var(--mono);font-variant-numeric:tabular-nums;
 margin-top:7px;letter-spacing:-.03em;line-height:1.1}
.card .n{font-size:12.5px;color:var(--dim);margin-top:5px}

.box{background:var(--panel);border:1px solid var(--line);border-radius:12px;
 padding:16px 18px}

/* ---- charts: inline SVG, crisp at any width and any pixel density ---- */
svg.chart{display:block;width:100%;height:auto;overflow:visible}
.chart text{font-family:inherit;fill:var(--dim)}
.chart .ct{fill:var(--ink);font-size:11.5px;font-weight:600}
.chart .tk{font-size:10px;font-variant-numeric:tabular-nums}
.chart .fl{font-size:10px;font-weight:600;fill:var(--dim);font-family:var(--mono)}
.chart .grid{stroke:var(--line);stroke-width:1}
.chart .axis{stroke:var(--line);stroke-width:1}
.chart .chance{stroke:var(--faint);stroke-width:1;stroke-dasharray:3 3}
.chart .pooled{stroke:var(--accent);stroke-width:2}
.chart .band{fill:var(--accent);opacity:.14}
.chart .ln{fill:none;stroke-width:1.8;stroke-linejoin:round;stroke-linecap:round}
.chart .dot{stroke:var(--bg);stroke-width:1.5}
.panels{display:grid;grid-template-columns:1fr 1fr;gap:20px}
@media(max-width:720px){.panels{grid-template-columns:1fr}}
.legend{display:flex;flex-wrap:wrap;gap:16px;margin-top:14px;font-size:12px;
 color:var(--dim)}
.key{display:inline-flex;align-items:center;gap:6px}
.key i{width:12px;height:3px;border-radius:2px;display:inline-block}
.key i.ring{width:9px;height:9px;border-radius:50%;background:var(--dim)}

/* ---- the per-target table, with the bar living inside it ---- */
.scroll{overflow-x:auto;border:1px solid var(--line);border-radius:12px;
 background:var(--panel)}
table{border-collapse:collapse;width:100%;font-size:14px;min-width:620px}
th,td{padding:9px 14px;text-align:right;border-bottom:1px solid var(--line);
 font-family:var(--mono);font-variant-numeric:tabular-nums;white-space:nowrap}
th:first-child,td:first-child{text-align:left;font-family:inherit}
th{font-size:10px;text-transform:uppercase;letter-spacing:.1em;color:var(--faint);
 font-weight:700;background:var(--raise)}
tbody tr:last-child td{border-bottom:none}
tbody tr:hover td{background:var(--raise)}
tr.thin td{color:var(--dim)}
td.neg{color:var(--danger)}
td.pos{color:var(--dim)}
td.flat{color:var(--warn)}
td.bar{width:42%;padding:9px 16px}

.track{position:relative;height:16px;background:#171c25;border-radius:4px}
.track .fill{position:absolute;left:0;top:0;height:100%;border-radius:4px;display:block}
.fill.good{background:linear-gradient(90deg,#3d76c4,var(--accent))}
.fill.thin{background:#4d586e}
.fill.flat{background:#5a5060}
.fill.bad{background:linear-gradient(90deg,#a94f48,var(--danger))}
.track .span{position:absolute;top:0;height:100%;border-radius:4px;
 background:var(--line);border:1px solid #38414f}
.track .chance-mark{position:absolute;top:-3px;width:1px;height:22px;
 background:var(--faint)}
.track .none{font-size:11px;color:var(--faint);font-family:var(--mono)}

.flag{display:flex;gap:13px;align-items:baseline;background:var(--panel);
 border:1px solid var(--line);border-left:3px solid;padding:13px 16px;margin:9px 0;
 border-radius:0 10px 10px 0;font-size:14px;line-height:1.55}
.error{border-left-color:var(--danger)}
.warn{border-left-color:var(--warn)}
.info{border-left-color:var(--accent)}
.tag{flex:0 0 auto;font-size:9.5px;text-transform:uppercase;letter-spacing:.11em;
 font-weight:700;font-family:var(--mono);padding-top:2px}
.error .tag{color:var(--danger)}
.warn .tag{color:var(--warn)}
.info .tag{color:var(--accent)}

.note{color:var(--dim);font-size:13px;margin-top:12px}
code{background:var(--raise);border:1px solid var(--line);padding:1px 6px;
 border-radius:5px;font-family:var(--mono);font-size:12.5px;color:var(--ink)}
"""


def _cards(report: dict) -> str:
    lo, hi = report["interval"]
    return "".join([
        f'<div class="card"><div class="k">out-of-fold macro AUC</div>'
        f'<div class="v">{report["pooled_auc"]:.4f}</div>'
        f'<div class="n">95% interval {lo:.3f} – {hi:.3f}</div></div>',
        f'<div class="card"><div class="k">spread across folds</div>'
        f'<div class="v">± {report["fold_std"]:.3f}</div>'
        f'<div class="n">mean {report["fold_mean"]:.3f} over '
        f'{report["n_folds"]} folds</div></div>',
        f'<div class="card"><div class="k">studies judged</div>'
        f'<div class="v">{report["n_studies"]}</div>'
        f'<div class="n">each predicted once, unseen</div></div>',
    ])


def _target_rows(report: dict) -> str:
    rows = sorted(report["targets"],
                  key=lambda r: -(r["auc"] if np.isfinite(r["auc"]) else -1))
    out = []
    for row in rows:
        if not np.isfinite(row["auc"]):
            auc, span, klass = "—", "—", ""
        else:
            auc = f"{row['auc']:.3f}"
            span = f"{row['lo']:.3f} – {row['hi']:.3f}"
            # Straddling chance is the only verdict a single row can carry on its own.
            klass = "flat" if row.get("flat") else "pos"
        classes = " ".join(c for c in ("thin" if row["thin"] else "",) if c)
        out.append(
            f'<tr class="{classes}">'
            f"<td>{html.escape(row['target'])}</td>"
            f'<td>{row["positives"]}</td>'
            f'<td class="bar">{F.target_bar(row)}</td>'
            f'<td>{auc}</td><td class="{klass}">{span}</td></tr>')
    return "".join(out)


def render_html(report: dict) -> str:
    """One self-contained page: markup, style and charts, no external anything."""

    records = report["_records"]
    y, _ = report["_pooled_arrays"]
    note = ('<p class="note">The pale span behind each bar is where the true value '
            'plausibly sits (Hanley–McNeil, 95%). A span crossing the chance mark '
            'means the run measured nothing on that target, whichever side of it the '
            'bar ends. The label table\'s own per-target AUC is not shown: it is a '
            'property of the labels rather than of a run, and it lives in '
            '<code>docs/data.md</code>.</p>')

    flags = "".join(
        f'<div class="flag {f["level"]}"><span class="tag">{f["level"]}</span>'
        f'<span>{html.escape(f["text"])}</span></div>' for f in report["flags"]
    ) or '<p class="note">Nothing tripped a check.</p>'

    return f"""<title>{html.escape(report['experiment'])} — evaluation</title>
<style>{_CSS}</style>
<main>
<h1>{html.escape(report['experiment'])}</h1>
<div class="sub"><b>{report['n_folds']} folds</b> &middot;
 <b>{report['epochs']} epochs</b> &middot;
 split <code>{html.escape(report['split'])}</code> &middot;
 labels <code>{html.escape(Path(report['labels']).name)}</code><br>
 generated {report['written']}</div>

<div class="head">{_cards(report)}</div>
<div class="box">{F.spread(np.array(report['fold_aucs'], float),
                           report['pooled_auc'], tuple(report['interval']))}</div>

<h2>Per pathology</h2>
<div class="scroll"><table>
<thead><tr><th>target</th><th>pos.</th><th></th><th>AUC</th>
<th>95% interval</th></tr></thead>
<tbody>{_target_rows(report) if len(y) else ''}</tbody>
</table></div>
{note}

<h2>Training</h2>
<div class="box">{F.curves(records) if records else ''}</div>

<h2>What does not look right</h2>
{flags}
</main>
"""


def write(report: dict, path: str | Path) -> Path:
    """Render to `path`, and drop the numbers beside it as JSON."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_html(report), encoding="utf-8")
    payload = {k: v for k, v in report.items() if not k.startswith("_")}
    path.with_suffix(".json").write_text(json.dumps(payload, indent=1) + "\n",
                                         encoding="utf-8")
    return path
