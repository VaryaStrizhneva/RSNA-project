"""The way in: every page under docs/atlas, grouped and described.

    python -m tools.atlas.index_page

Writes `docs/atlas/index.html`, which `python -m http.server` then serves instead of its
directory listing. Generated rather than written, so a page built or deleted after this
is reflected by re-running it rather than by remembering to edit a list.

Anything found but not described below is still listed, under "not yet described" — a
page that exists and is not mentioned is the failure mode an index has, and silence about
it would be worse than an untidy row.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.atlas import page as P                                    # noqa: E402
from tools.atlas.expert_page import EXPERTS                          # noqa: E402
from tools.atlas.landmarking_page import MODELS                      # noqa: E402

ATLAS = ROOT / "docs" / "atlas"


def expert_rows() -> list[tuple[str, str]]:
    """One row per entry of `expert_page.EXPERTS`, in the order it declares them.

    Derived rather than listed. A hand-written list went stale the moment a second page
    was built — the two new ones fell into "not yet described" while the index still
    claimed Baker's was the only expert — and an index that has to be edited alongside
    the thing it indexes is an index that will be wrong.
    """

    return [(f"experts/{key}.html",
             f"<b>{entry['title']}</b> &mdash; {entry['subtitle']}")
            for key, entry in EXPERTS.items()]


def landmark_rows() -> list[tuple[str, str]]:
    """One row per entry of `landmarking_page.MODELS`, with the points it places."""

    out = []
    for key, (run, points) in MODELS.items():
        names = ", ".join(f"<code>{p}</code>" for p, _ in points)
        out.append((f"landmarking/{key}.html", f"{names} &mdash; <code>{run}</code>"))
    return out

#: path relative to docs/atlas -> what it is. Order within a group is the order shown.
def groups() -> list[tuple[str, str, list[tuple[str, str]]]]:
    """The sections, built at call time so the registries are read fresh."""
    return [
    ("The experts", "One page per shipped expert: the condition it reads, the box it "
                    "reads it in, and how that box is placed. Both viewers come from one "
                    "<code>rsna.roi.extract</code> call, so no page can describe a box "
                    "the model does not read. Eight are shipped; the ones not listed here "
                    "have no page yet.", expert_rows()),
    ("The landmark models", "Three models place every box. Each page carries the "
                            "point's definition, the corpus it was fitted on, the "
                            "held-out error distribution, then the six worst studies and "
                            "six median ones — your click against the prediction of the "
                            "fold that never saw it.", landmark_rows()),
    ("How the pipeline reads a study", "Not per-condition: the machinery every box and "
                                       "every branch shares.", [
        ("landmark_explanation.html", "What a landmark is for and how the point is defined"),
        ("landmark_predictions.html", "The landmark model's output over the corpus"),
        ("lateralite_sagittale.html", "Which end of a sagittal stack is lateral, and how "
                                      "that is recovered when the DICOM does not say"),
        ("crop.html", "The whole-knee boxes, sagittal and coronal, with the slices each takes"),
        ("wide.html", "The nine boxes of the twelve-target model"),
        ("preproc.html", "How every model in the public pipeline selects its series"),
        ("preprocessing.html", "The same audit, longer"),
    ]),
    ("Proposals, not yet built", "A box drawn and argued for before any training run is "
        "spent on it, on cases the report calls positive and cases it does not. The "
        "point is that the geometry can be rejected here, where it costs an afternoon, "
        "rather than after five folds. Nothing on these pages is shipped.", [
        ("mm_coronal.html", "<b>A coronal box for the medial meniscus</b> &mdash; "
                            "extrusion, the body in cross-section and the posterior "
                            "root, none of which a sagittal box frames well"),
        ("oa_sagittal.html", "<b>A sagittal box for both osteoarthritis compartments</b> "
                             "&mdash; the anterior-posterior axis a coronal slice cannot "
                             "have, and two thirds of the MOAKS subregions with it"),
        ("contusion.html", "<b>Two boxes for bone contusion</b> &mdash; the difference "
                           "map behind them, and the axis bug that map was built with"),
    ]),
    ("Development pages, kept for their measurements", "Superseded by the pages above. "
        "Each one is the study that produced a box — difference maps, configuration "
        "sweeps, capture rates — and is kept because the number it measured is not "
        "written anywhere else. Their builders are in <code>tools/atlas/</code>.", [
        ("baker_axial.html", "Where the cyst sits in the axial frame, and the candidate boxes"),
        ("baker.html", "The earlier sagittal study of the same cyst"),
        ("effusion.html", "The effusion boxes — a <b>measured failure</b>, and the rule it gave"),
        ("synovitis.html", "What synovitis looks like, and why no box was built"),
        ("lateral_meniscus.html", "The lateral meniscus, and the depth sweep behind d32"),
        ("medial_meniscus.html", "The medial meniscus"),
        ("medial_boxes.html", "Its candidate boxes"),
        ("tibiofemoral_oa.html", "Both tibiofemoral osteoarthritis compartments"),
        ("pf_oa.html", "Patellofemoral osteoarthritis"),
        ("pf_boxes.html", "Its candidate boxes"),
        ("mcl.html", "The collateral ligament"),
        ("mcl_boxes.html", "Its candidate boxes"),
        ("mcl_box_final.html", "The box that shipped"),
        ("acl.html", "The cruciate ligament"),
        ("acl_derived.html", "Its box derived between two landmarks"),
        ("acl_final.html", "The box that shipped"),
        ("roi_boxes.html", "The first box study, over one landmark"),
        ("roi_depth.html", "How deep a box should reach"),
        ("roi_coronal.html", "The coronal boxes"),
        ("roi_coronal_viewer.html", "The same, scrollable"),
    ]),
    ]

CSS = """
.grp{margin:26px 0 8px}
.grp h2{margin:0 0 4px}
.grp .why{color:#8b97a8;font-size:14px;line-height:1.6;max-width:70ch;margin:0 0 14px}
table.idx{border-collapse:collapse;width:100%}
table.idx td{padding:7px 10px;border-top:1px solid #222a35;vertical-align:top}
table.idx td.sz{text-align:right;color:#5a6472;width:7em;white-space:nowrap}
table.idx a{color:#6ea8fe;text-decoration:none;font-weight:600}
table.idx a:hover{text-decoration:underline}
.gone{color:#5a6472}
"""


def row(rel: str, what: str) -> str:
    path = ATLAS / rel
    if not path.is_file():
        return (f'<tr><td class="gone">{P._esc(rel)} &mdash; not built</td>'
                f'<td>{what}</td><td class="sz"></td></tr>')
    mb = path.stat().st_size / 1e6
    return (f'<tr><td><a href="{P._esc(rel)}">{P._esc(rel)}</a></td>'
            f'<td>{what}</td><td class="sz">{mb:.1f} MB</td></tr>')


def build() -> Path:
    sections = groups()
    described = {rel for _, _, items in sections for rel, _ in items}
    found = {str(p.relative_to(ATLAS)) for p in ATLAS.rglob("*.html")} - {"index.html"}

    blocks = []
    for title, why, items in sections:
        rows = "".join(row(rel, what) for rel, what in items)
        blocks.append(f'<div class="grp"><h2>{title}</h2><p class="why">{why}</p>'
                      f'<table class="idx"><tbody>{rows}</tbody></table></div>')

    extra = sorted(found - described)
    if extra:
        rows = "".join(row(rel, "<i>no description in "
                                "<code>tools/atlas/index_page.py</code></i>")
                       for rel in extra)
        blocks.append('<div class="grp"><h2>Not yet described</h2>'
                      '<p class="why">Found under <code>docs/atlas</code> and not '
                      'mentioned above. Either describe it or delete it; an index that '
                      'quietly omits a page is worse than one that admits to a gap.</p>'
                      f'<table class="idx"><tbody>{rows}</tbody></table></div>')

    total = sum((ATLAS / r).stat().st_size for r in found if (ATLAS / r).is_file()) / 1e6
    body = f"""
<style>{CSS}</style>
<h1>RSNA knee &mdash; atlas</h1>
<p class="note">{len(found)} pages, {total:.0f} MB. Every one is self-contained: each MRI
slice is a data URI inside the file, so a page works from <code>file://</code> with no
server. That is also why none of them is in git &mdash;
<code>docs/atlas/**/*.html</code> is ignored, because the pages carry patient slices and
this repository has a GitHub remote. The builders are in <code>tools/atlas/</code> and
<code>tools/annotate/</code>.</p>
{''.join(blocks)}
"""
    out = ATLAS / "index.html"
    out.write_text(P.page("RSNA knee — atlas", body))
    return out


if __name__ == "__main__":
    path = build()
    print(f"{path}  ({path.stat().st_size / 1e3:.0f} kB)")
