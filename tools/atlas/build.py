"""Build one pathology page.

    python -m tools.atlas.build lateral_meniscus

Reads the raw DICOM under /data/mgr/rsna-knee/extracted, never the training cache, and
writes a single self-contained HTML file to docs/atlas/.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from rsna.config import Config, TARGETS                                   # noqa: E402
from tools.atlas import page as P                                         # noqa: E402
from tools.atlas.content import PATHOLOGIES, Case, Pathology              # noqa: E402
from tools.atlas.render import resolution_panel, stack_to_jpegs, to_jpeg  # noqa: E402
from tools.atlas.study import (load_series, series_headers, side_of,      # noqa: E402
                               stack_orientation)


def gold_table() -> pd.DataFrame:
    """The 58 studies an expert panel labelled, which is the only ground truth here."""

    train = pd.read_csv(ROOT / "data/raw/train.csv")
    return train[train[TARGETS].notna().all(axis=1)]


def resolve(tail: str, gold: pd.DataFrame) -> str:
    matches = [u for u in gold["StudyInstanceUID"] if u.endswith(tail)]
    if len(matches) != 1:
        raise ValueError(f"{tail!r} matches {len(matches)} gold studies, not 1")
    return matches[0]


def pick_series(headers: pd.DataFrame, path: Pathology, plane: str):
    """The series that actually shows this finding, preferring the right contrast."""

    same_plane = headers[headers["plane"] == plane]
    if not len(same_plane):
        return None
    for weight in path.preferred_weight:
        hit = same_plane[same_plane["weight"] == weight]
        if len(hit):
            # More slices is a denser sampling of the joint, which is what we want.
            return hit.sort_values("n_slices", ascending=False).iloc[0]
    return same_plane.sort_values("n_slices", ascending=False).iloc[0]


def highlight(report: str, quote: str) -> str:
    """The report with the sentence it was chosen for marked."""

    text = P._esc(report.strip())
    key = P._esc(quote.strip().rstrip("."))
    # Match on a distinctive fragment: reports repeat sentences with small variations.
    fragment = " ".join(key.split()[:7])
    if fragment and fragment in text:
        text = text.replace(fragment, f"<mark>{fragment}", 1)
        tail = text.find("\n", text.find("<mark>"))
        cut = text.find(".", text.find("<mark>"))
        at = min(x for x in (tail, cut, len(text)) if x > 0)
        text = text[:at] + "</mark>" + text[at:]
    return text


def build_case(case: Case, path: Pathology, gold: pd.DataFrame, config: Config,
               index: int) -> tuple[str, dict | None]:
    """One case: the report, the viewers, and my own reading of it."""

    uid = resolve(case.uid_tail, gold)
    row = gold[gold["StudyInstanceUID"] == uid].iloc[0]
    headers = series_headers(uid)
    side, how = side_of(headers)

    chunks = []
    tag = "pos" if case.positive else "neg"
    verdict = "expert: PRESENT" if case.positive else "expert: ABSENT"
    chunks.append(f"""
<div class="card">
  <div class="case-head">
    <h3 style="margin:0">Case {index}</h3>
    <span class="tag {tag}">{verdict}</span>
  </div>
  <div class="meta">study &hellip;{P._esc(case.uid_tail)} &middot;
     {P._esc(side or '?')} knee ({P._esc(how)}) &middot;
     {len(headers)} series acquired</div>
  <p style="margin:12px 0 0">{case.why}</p>
  <div class="report">{highlight(row['Report'], case.quote)}</div>""")

    panel_source = None
    for plane in path.planes:
        chosen = pick_series(headers, path, plane)
        if chosen is None:
            continue
        series = load_series(chosen, config)
        frames, _ = stack_to_jpegs(series.volume)
        left, right = stack_orientation(plane, side)
        if plane == "Sagittal":
            panel_source = series

        order = ("" if series.ordered else
                 " <b>Geometry was unusable, so the file order was kept</b> &mdash; "
                 "the slices may not be in anatomical order.")
        caption = (f"{P._esc(series.label)} &middot; {series.n} slices &middot; "
                   f"{series.mm_per_px:.2f} mm/px in plane &middot; "
                   f"{series.thickness:.0f} mm slice thickness"
                   f"{' &middot; TE ' + str(int(series.te)) if series.te else ''}"
                   f"{' &middot; TR ' + str(int(series.tr)) if series.tr else ''}"
                   f"{order}")

        vid = f"v{index}{plane[:3].lower()}"
        chunks.append(f"<h4 style='margin:18px 0 0;color:#9aa3ad'>{P._esc(plane)}</h4>")
        chunks.append(P.viewer(
            vid, frames, left, right, series.mm_per_px,
            flagged=case.claude_slices if plane == "Sagittal" else (),
            claude=case.claude if plane == "Sagittal" else "",
            caption=caption))

    chunks.append("</div>")
    return "".join(chunks), panel_source


def build(key: str) -> Path:
    path = PATHOLOGIES[key]
    config = Config()
    gold = gold_table()

    body = [f"""
<header>
  <h1>{P._esc(path.name)}</h1>
  <p class="lede">{path.one_line}</p>
  <div class="meta" style="margin-top:10px">Pathology atlas &middot; built from the
    raw DICOM stacks, not the training cache &middot; {len(path.cases)} expert-labelled
    cases</div>
</header>

<div class="danger">
  <h4>Read this before you trust anything below</h4>
  <p style="margin:0">There is <b>no localisation ground truth</b> in this corpus &mdash;
  no slice labels, no boxes. The only sourced statement of <i>where</i> a finding is
  comes from the radiologist's own report, quoted with each case. Anything labelled
  <b>Claude's reading</b> is my own guess from looking at the pixels. I am not a
  radiologist, the sections below explain exactly why this particular finding is at the
  edge of what can be resolved at all, and on several of these cases I could not confirm
  the reported tear. Those notes are there so you can see where a careful non-expert
  gets stuck &mdash; not as findings.</p>
</div>

<h2>What it is</h2>{path.what_it_is}

<h2>How it is read</h2>{path.how_it_is_read}

<h3>Which sequence</h3>{path.sequences}

<h3>What to look for, in order</h3>
<ul>{''.join(f'<li>{x}</li>' for x in path.what_to_look_for)}</ul>

<h3>What will fool you</h3>
<ul>{''.join(f'<li>{x}</li>' for x in path.pitfalls)}</ul>

<h2>Cases</h2>
<p class="lede">Every slice of every series, at the resolution it was acquired &mdash;
no crop, no band, no twelve-slice sampling. Scroll the stack; a real finding persists
across two or three slices, and that persistence is most of the signal.</p>
"""]

    panel_series = None
    for i, case in enumerate(path.cases, 1):
        chunk, series = build_case(case, path, gold, config, i)
        body.append(chunk)
        if panel_series is None and series is not None:
            panel_series = series
        print(f"  case {i} ({case.uid_tail}) done", flush=True)

    if panel_series is not None:
        mid = len(panel_series.volume) * 2 // 3
        panels = resolution_panel(panel_series.volume[mid], panel_series.mm_per_px,
                                  config)
        jpegs = [to_jpeg(p["image"], 380)[0] for p in panels]
        body.append(f"""
<h2>What the model is given instead</h2>
<p class="lede">The same slice, three times: as it was acquired, as the training cache
stores it, and at the granularity the encoder can place things. Nothing about the
pathology changes between these &mdash; only how much of it survives.</p>
{P.resolution_panel_html(panels, jpegs)}
<p class="note">The third panel is not a claim that the encoder blurs within a patch
&mdash; it does not, each patch is projected whole. It is a picture of the
<b>granularity of position</b>: everything inside one block arrives at the head as a
single token at a single location. A 1.5&nbsp;mm tear line is about a quarter of one
block, and the slot feature then averages over all 576 of them.</p>""")

    rows = "".join(f"<tr><th style='width:34%'>{P._esc(k)}</th><td>{v}</td></tr>"
                   for k, v in path.numbers.items())
    body.append(f"""
<h2>What we have measured about this target</h2>
<table>{rows}</table>

<div class="danger" style="border-color:#6b5420;background:#1d1a12">
  <h4 style="color:#f0a868">The gold labels and the reports do not always agree</h4>
  <p style="margin:0">Among the 58 expert-labelled studies, several scored
  <b>negative</b> for this target carry reports describing a tear &mdash; one reads
  <i>"complex tear (main radial and longitudinal vertical component) at the posterior
  horn of the lateral meniscus"</i> and is still labelled absent. The expert panel
  re-read the images and disagreed with the original radiologist. That is worth knowing
  before treating either source as truth: for this pathology, even the ground truth is
  contested.</p>
</div>

<footer>
  Built by <code>tools/atlas/build.py</code> from
  <code>/data/mgr/rsna-knee/extracted/train_series</code>. Patient imaging &mdash; kept
  local, not published. The radiology is textbook material written down for orientation,
  not expert knowledge; check it against a real reference.
</footer>""")

    out = ROOT / "docs" / "atlas" / f"{key}.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(P.page(f"{path.name} — pathology atlas", "".join(body)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pathology", choices=sorted(PATHOLOGIES))
    args = ap.parse_args()
    out = build(args.pathology)
    print(f"\n{out}  ({out.stat().st_size/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
