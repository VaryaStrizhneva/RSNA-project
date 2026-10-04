"""Every model's input: which series, chosen how, cropped how, how many slices.

    python -m tools.atlas.preproc_page -o docs/atlas/preprocessing.html

Two pipelines are audited side by side. The public one (`kaggle/public-d4/notebook.ipynb`)
runs 30-odd models in five legs, and **each leg preprocesses differently** — three
different ways of choosing a series, four crop sizes, slice counts from 3 to 76. Ours
runs seven experts off predicted landmarks.

**What is checked and what is quoted.** Everything about our side is read live: the
stored spec of each run, the per-slot fill rates of the caches on disk, the out-of-fold
scores. Everything about the public side is a constant in `PUBLIC` below, each carrying a
`probe` string that must still appear in the cell it names — so if the upstream notebook
moves, this page refuses to build rather than describing a pipeline that no longer
exists.

**Where the audit stops.** The four CoAtNet readers run child processes that import
`raptor_light_inference` from inside a mounted dataset. That module is not in the
notebook and not in `/kaggle/working`, so it was not read. What is stated about those
arms comes from their own runtime docstring and from the geometry contract shipped
beside their weights, both quoted — not from the decoder.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from rsna.config import Config                                       # noqa: E402
from rsna.roi.config import SPECS                                    # noqa: E402
from tools.atlas import page as P                                    # noqa: E402

NOTEBOOK = ROOT / "kaggle/public-d4/notebook.ipynb"
ROI_CACHE = Path("/data/mgr/rsna-knee/roi-cache")

#: The seven experts in the 0.923 submission, in the order their columns are written.
SEVEN = ["expert_lm_d32", "expert_medial_meniscus", "expert_pf_oa", "expert_mcl_fsonly",
         "expert_lateral_oa_coronal", "expert_medial_oa", "expert_acl"]

#: One row per leg of the public pipeline. `probe` must still be present in `cell`, so a
#: moved notebook breaks the build instead of silently making this page a work of fiction.
PUBLIC = [
    {
        "leg": "DINOv2", "members": "20 members, 5 folds",
        "cell": 15, "probe": "SHARED_DINO_PREFIX_LAYERS = 6",
        "source": "DICOM headers, parsed by <code>annotate()</code>",
        "rule": "per slot: plane + fat-sat (+ fluid), then <b>the series with the most "
                "slices</b>",
        "slots": "6 &mdash; SAG/COR/AX fluid+FS, SAG fluid noFS, COR T1, SAG T1",
        "slices": "GROUP(3) &times; N_GROUP, where <b>N_GROUP is chosen at runtime from "
                  "free RAM</b>; band (0.20, 0.80)",
        "crop": "130 mm, image centre", "px": "336",
        "lat": "right knees flipped &mdash; sagittal along the stack axis, coronal and "
               "axial left-right.",
        "note": "This leg's preprocessing is <b>identical to our own wide model</b>: the "
                "same six slots, the same rules, and a <code>read_slot</code> that "
                "matches <code>rsna/dicom/pixels.py</code> line for line. Both are ports "
                "of the same baseline.",
    },
    {
        "leg": "A5", "members": "5 folds",
        "cell": 20, "probe": "def read_crop(path)",
        "source": "the competition's own <code>test_series.csv</code> "
                  "(<code>Anatomical_Plane</code>, <code>Fat_Suppression</code>)",
        "rule": "per slot: plane + fat-sat, then <b><code>sub.iloc[0]</code> &mdash; the "
                "first row</b>, not the longest series",
        "slots": "6 &mdash; the three planes &times; fat-sat on/off",
        "slices": "16, band (0.12, 0.88); fewer when the band is short, then centred "
                  "with zero padding",
        "crop": "130 mm, image centre &mdash; whole image when "
                "<code>PixelSpacing</code> is missing",
        "px": "336",
        "lat": "non-sagittal flipped when <code>series_side &lt; 0</code>.",
        "note": "Windows intensity <b>per slice</b> (<code>INTENSITY='slice'</code>) "
                "where the DINOv2 leg windows per series. It is also the only leg that "
                "trusts the competition's plane and fat-sat labels instead of "
                "re-deriving them from headers.",
    },
    {
        "leg": "RadImageNet", "members": "3 layouts, one encoder",
        "cell": 25, "probe": "CROP_MM = 10_000.0",
        "source": "DICOM headers, under <code>RULES_LEGACY</code>",
        "rule": "the same &lsquo;most slices&rsquo; rule, but with "
                "<b><code>slot_fallback</code> on</b>: a missing non-fluid slot is "
                "filled by any series of that plane",
        "slots": "<code>rad_e10</code>: 3 (all fat-sat) &middot; "
                 "<code>rad_e13</code>: 4 &middot; <code>rad_legacy_second</code>: 4",
        "slices": "8 per slot, band (0.20, 0.80)",
        "crop": "<b><code>rad_e10</code>: none.</b> <code>CROP_MM = 10_000</code> makes "
                "<code>16 &lt; want &lt; min(h, w)</code> false, so the crop is skipped "
                "and the whole image is resized. <code>e13</code> and <code>e11</code>: "
                "130 mm, image centre",
        "px": "224",
        "lat": "<code>corner_x</code> &mdash; the side is taken from the image corner, "
               "not from the knee.",
        "note": "Not removable, despite being 4.6% of the run. Deleting it also deletes "
                "<code>globals()['V18_CALIBRATOR_APPLIED'] = True</code>, and the BTKD "
                "calibrator &mdash; a learned regression folded into 40% of several "
                "columns &mdash; then silently does not apply.",
    },
    {
        "leg": "Raptor", "members": "4 arms, w = .60 / .20 / .10 / .10",
        "cell": 27, "probe": "CROP_MM = 140.0",
        "source": "the competition's <code>test_series.csv</code> "
                  "(<code>Anatomical_Plane</code>, <code>Fluid_Sensitive</code>)",
        "rule": "per slot: plane, preferring the fluid flag, then <b>the first series "
                "not already used by an earlier slot</b> &mdash; alone among all legs, "
                "no series is read twice",
        "slots": "5, each carrying a <b>share</b>, not a count: SAG fluid 18, SAG "
                 "non-fluid 14, COR fluid 12, COR non-fluid 8, AX either 12 "
                 "(<code>_SLOTS64</code>; arm 4 uses 12/10/8/6/8). "
                 "<code>_dense_allocate</code> spends a <b>96-slice budget</b> in those "
                 "proportions, capped by what each series actually holds, and gives the "
                 "remainder to the slots that still have room &mdash; so a study with a "
                 "short sagittal has its budget migrate to the coronal",
        "slices": "<b>94 windows per study, for all four arms.</b> The 96 picked slices "
                  "are concatenated into one volume and every interior centre becomes a "
                  "sliding RGB triplet <code>[c-1, c, c+1]</code>, so the windows "
                  "overlap. Span (0.02, 0.98) &mdash; 96% of each stack. "
                  "The <code>k_eval</code> of 62/42 in the arms table is dead code: "
                  "<code>_arm['k_eval'] = 94</code> overwrites it for every arm "
                  "immediately before the run",
        "crop": "140 mm, image centre, clamped to the short side", "px": "336 or 384",
        "lat": "one arm (<code>maxspan-v5-reverse</code>) reads the stack reversed "
               "&mdash; the same weights as arm 1, used as a second view.",
        "note": "An unknown protocol flag is treated as unknown rather than false and "
                "falls through to the plane-level choice, which is a deliberate guard "
                "against <code>int(NaN)</code>. <b>Not</b> deliberate: the triplets "
                "slide across the whole concatenation, so the two windows either side "
                "of each slot boundary carry channels from two different series &mdash; "
                "a sagittal slice in red beside a coronal one in green. With five slots "
                "that is 8 of the 94 windows. Intensity is windowed per slot "
                "(percentiles 2/98 over that slot's own slices), so the chimeras are "
                "not even on one scale.",
    },
    {
        "leg": "CoAtNet readers", "members": "4 readers &times; 3 checkpoints",
        "cell": 27, "probe": "import input_resgated_runtime as rt",
        "source": "<b>not read</b> &mdash; a child process imports "
                  "<code>raptor_light_inference</code> from inside a mounted dataset",
        "rule": "unknown from here. The arms share the Raptor family's slot vocabulary: "
                "6 anatomical slots, ids 1..6, sagittal = {1, 2}",
        "slots": "6, with per-slot window budgets <code>(20, 12, 12, 8, 16, 8)</code>",
        "slices": "<b>up to 76 windows per study.</b> Over the 58 gold studies in their "
                  "own geometry contract: min 48, max 76, mean 65.8",
        "crop": "&ldquo;the saved native <b>384 px / 130 mm</b> Gold grid&rdquo; "
                "&mdash; their runtime docstring",
        "px": "384",
        "lat": "<code>canonical_sagittal: true</code> in the shipped contract.",
        "note": "This is why the readers are 56.7% of the run: 76 windows at 384 px "
                "through a CoAtNet, three checkpoints each, four arms, and serialised "
                "rather than paired above 48 studies.",
    },
]


def check_probes(notebook: Path = NOTEBOOK) -> None:
    """Refuse to describe a notebook that has moved."""

    cells = json.load(open(notebook))["cells"]
    for row in PUBLIC:
        i = row["cell"]
        if i >= len(cells):
            raise SystemExit(f"{notebook} has {len(cells)} cells; {row['leg']} names {i}")
        if row["probe"] not in "".join(cells[i]["source"]):
            raise SystemExit(
                f"cell {i} no longer contains {row['probe']!r}, which is what this page "
                f"claims about the {row['leg']} leg. Read the notebook before "
                f"publishing anything that describes it.")
    print(f"  probes: {len(PUBLIC)}/{len(PUBLIC)} still present in {notebook.name}")


# -- our side, read live ----------------------------------------------------- #

def _cache_for(spec: dict) -> Path | None:
    """The cache a spec was cut into, matched on **geometry and series count**.

    Not on the roi name, and not on `cache.cache_tag` either. The tag is
    `name_WxHmm_WxHpx_lat-med_Nsl` — it does **not** encode the series list, so the
    two-series and three-series MCL specs share a tag, and resolving by it hands back
    whichever cache was written first. Matching the geometry in the filename and then
    checking the record width is unambiguous.
    """

    # Geometry alone does not discriminate: `lateral_oa` and `medial_oa` share a box
    # (54x39 mm at 252x182 px) and a series count, differing only in the landmark the
    # box hangs off — which is the field that decides where the pixels come from. And
    # `lateral_meniscus` and `lateral_meniscus_d32` share a box too, differing in the
    # slot count and the lateral/medial reach. So match every field that moves a pixel.
    decisive = ("box_w_mm", "box_h_mm", "out_w", "out_h",
                "lateral_mm", "medial_mm", "slots", "landmark")
    want = {f: spec.get(f) for f in decisive}
    want_series = [tuple(s) for s in (spec.get("series") or [])]
    for path in sorted(ROI_CACHE.glob("*.json")):
        blob = json.load(open(path))
        stored = blob.get("spec") if isinstance(blob, dict) else None
        if not stored:
            continue
        if any(stored.get(f) != want[f] for f in decisive):
            continue
        if [tuple(s) for s in (stored.get("series") or [])] != want_series:
            continue
        return path
    return None


def _full_spec(stored: dict) -> dict:
    """A stored spec filled out with the dataclass defaults it omitted.

    `history.json` serialises only the fields that differ from `RoiSpec`'s defaults, so
    a sagittal spec has no `plane` key at all. Reading it therefore needs the defaults —
    which also means **a changed default silently re-interprets every old run**. The
    defaults are taken from a live `RoiSpec`, so at least they are the ones this working
    tree would use.
    """

    import dataclasses

    base = SPECS[next(iter(SPECS))]
    out = {f.name: getattr(base, f.name) for f in dataclasses.fields(base)}
    out.update({"box_offset_mm": 0.0, "box_rise_mm": 0.0, "lateral_mm": 0.0,
                "medial_mm": 0.0, "depth_inset_mm": 0.0, "landmark2": None})
    out.update(stored)
    return out


def _records(spec: dict) -> list | None:
    """The cache's per-study record list, if one matching this spec is on disk."""

    path = _cache_for(spec)
    if path is None:
        return None
    recs = json.load(open(path))
    return recs if isinstance(recs, list) else recs.get("records", [])


def expert_rows() -> list[dict]:
    """One row per shipped expert, from its stored spec and its cache on disk.

    The *stored* spec, not the registry: `expert_mcl_fsonly` names the roi `mcl`, whose
    registry entry has since grown a third series. Resolving by name would describe a
    two-series run as a three-series one — and at inference would feed a two-series
    model a three-series tensor.
    """

    out = []
    for run in SEVEN:
        h = json.load(open(ROOT / f"out/{run}/history.json"))
        cfg = h["config"]
        stored = h.get("specs", h.get("spec"))
        spec = _full_spec((stored if isinstance(stored, list) else [stored])[0])
        roi = (cfg.get("rois") or [cfg.get("roi")])[0]
        oof = h["oof"]
        target = list(oof["per_target"])[0]

        recs = _records(spec)
        fill = ([100 * sum(r["series"][j] is not None for r in recs) / len(recs)
                 for j in range(len(spec["series"]))] if recs else [])
        out.append({"run": run, "target": target, "roi": roi, "spec": spec,
                    "auc": oof["per_target"][target], "n": oof["n"],
                    "fill": fill, "recs": recs,
                    "epochs": cfg.get("epochs"), "batch": cfg.get("batch")})
    return out


def empty_tensor_rows(rows: list[dict]) -> list[dict]:
    """Studies whose every series slot is empty, and what the expert says about them."""

    from sklearn.metrics import roc_auc_score

    out = []
    for row in rows:
        recs = row["recs"]
        if recs is None:
            continue
        empty = {r["study"] for r in recs
                 if not any(s is not None for s in r["series"])}
        if not empty:
            out.append({**row, "n_empty": 0})
            continue
        d = pd.read_csv(ROOT / f"out/{row['run']}/holdout.csv")
        t, p = f"true:{row['target']}", f"pred:{row['target']}"
        m = d.study.isin(empty)
        y = (d[t] > 0.5).astype(int)
        auc = (roc_auc_score(y[m], d[p][m])
               if m.sum() > 30 and y[m].nunique() == 2 else float("nan"))
        out.append({**row, "n_empty": int(m.sum()), "empty_auc": auc,
                    "empty_lo": float(d[p][m].min()), "empty_hi": float(d[p][m].max()),
                    "empty_sd": float(d[p][m].std())})
    return out


# -- rendering --------------------------------------------------------------- #

def public_section() -> str:
    head = ("<tr><th>leg</th><th>where series metadata comes from</th>"
            "<th>which series, among the candidates</th><th>slots</th>"
            "<th>slices / windows</th><th>crop</th><th>px</th></tr>")
    body = "".join(
        f"<tr><td><b>{r['leg']}</b><div class='meta'>{r['members']}</div></td>"
        f"<td>{r['source']}</td><td>{r['rule']}</td><td>{r['slots']}</td>"
        f"<td>{r['slices']}</td><td>{r['crop']}</td><td>{r['px']}</td></tr>"
        for r in PUBLIC)
    detail = "".join(
        f"<div class='card'><h3 style='margin-top:0'>{r['leg']} "
        f"<span class='meta'>&mdash; cell {r['cell']}</span></h3>"
        f"<p><b>Laterality.</b> {r['lat']}</p><p>{r['note']}</p></div>"
        for r in PUBLIC)
    return f"""
<h2>The public pipeline, leg by leg</h2>
<div class="note">Five legs, about thirty models. Read from
  <code>kaggle/public-d4/notebook.ipynb</code>; every row is probe-checked against the
  cell it names at build time, so this table cannot outlive the notebook it
  describes.</div>
<div style="overflow-x:auto"><table>{head}{body}</table></div>
<p>There is no single preprocessing here. <b>Three different ways of choosing a
series</b> (longest, first-in-the-CSV, first-unused), <b>two sources of plane and
fat-sat metadata</b> (DICOM headers vs the competition's CSV), <b>four crop
regimes</b> (130 mm, 140 mm, none, and 130 mm at 384 px), and slice counts from 3 to 76.
Two legs re-derive the plane from headers and two trust the provided labels; nothing
reconciles them.</p>
{detail}"""


def expert_section(rows: list[dict]) -> str:
    cfg = Config()
    head = ("<tr><th>expert</th><th>target</th><th>plane</th><th>landmark</th>"
            "<th>box</th><th>output px</th><th>mm/px</th><th>slices</th>"
            "<th>series asked for &mdash; and how often each is found</th>"
            "<th>OOF</th></tr>")
    body = []
    for r in rows:
        s = r["spec"]
        ser = []
        for j, (plane, weight, fs) in enumerate(s["series"]):
            pct = f"{r['fill'][j]:.0f}%" if j < len(r["fill"]) else "?"
            thin = r["fill"] and j < len(r["fill"]) and r["fill"][j] < 40
            ser.append(f"<div{' style=color:#e0564f' if thin else ''}>"
                       f"{plane} {weight}{' FS' if fs else ' noFS'} "
                       f"&mdash; <b>{pct}</b></div>")
        two = (f"<br><code>+{s['landmark2']}</code>" if s.get("landmark2") else "")
        body.append(
            f"<tr><td><code>{r['run'].replace('expert_', '')}</code></td>"
            f"<td><b>{r['target']}</b></td><td>{s['plane']}</td>"
            f"<td><code>{s['landmark']}</code>{two}</td>"
            f"<td>{s['box_w_mm']:.0f} &times; {s['box_h_mm']:.0f} mm</td>"
            f"<td>{s['out_w']} &times; {s['out_h']}</td>"
            f"<td>{s['box_w_mm'] / s['out_w']:.3f}</td><td>{s['slots']}</td>"
            f"<td>{''.join(ser)}</td><td>{r['auc']:.4f}</td></tr>")
    return f"""
<h2>Our experts</h2>
<div class="note">Read live from each run's <b>stored</b> spec and from the ROI caches on
  disk &mdash; not from the registry, which has moved since some of these were fitted.
  A fill rate under 40% is marked red.</div>
<div style="overflow-x:auto"><table>{head}{''.join(body)}</table></div>
<p>Three things differ structurally from every public leg. We select a series by the
exact triple <b>(plane, weight, fat-sat)</b> where they test a single fluid boolean that
lumps PD and T2 together &mdash; more specific, and so more often unmatched. We cut the
box around a <b>predicted landmark</b> instead of the image centre, at
<b>0.21&ndash;0.29 mm/px</b> against their 0.387 for a 130 mm window at 336 px. And we
read 5&ndash;11 slices from a thin slab where Raptor reads 94 overlapping windows over
96 slices and the CoAtNet readers up to 76 &mdash; our resolution is bought with
coverage.</p>
<p class="note">For reference the wide model reads the same six slots as their DINOv2
leg, at {cfg.crop_mm:.0f} mm and {cfg.img} px, band {cfg.band}.</p>"""


def findings_section(empties: list[dict]) -> str:
    bad = [e for e in empties if e.get("n_empty")]
    worst = max(bad, key=lambda e: e["n_empty"]) if bad else None

    def auc_cell(e):
        if not e["n_empty"]:
            return "&mdash;"
        a = e["empty_auc"]
        return f"{a:.4f}" if a == a else "too few to score"

    def range_cell(e):
        if not e["n_empty"]:
            return "&mdash;"
        return f"{e['empty_lo']:.4f} &ndash; {e['empty_hi']:.4f}"

    rows = "".join(
        f"<tr{' class=danger' if e['n_empty'] > 50 else ''}>"
        f"<td><code>{e['run'].replace('expert_', '')}</code></td>"
        f"<td>{e['n_empty']}</td><td>{auc_cell(e)}</td><td>{range_cell(e)}</td></tr>"
        for e in empties)
    total = sum(e["n_empty"] for e in empties)

    lead = "No expert scores a study with every slot empty."
    if worst is not None:
        lead = (f"<b>{worst['n_empty']} studies</b> reach "
                f"<code>{worst['run'].replace('expert_', '')}</code> with every series "
                f"slot empty, and are scored anyway. Its output on them spans "
                f"{worst['empty_lo']:.4f}&ndash;{worst['empty_hi']:.4f} "
                f"(sd {worst['empty_sd']:.5f}) &mdash; a constant, which is what an "
                f"all-zero input should give.")

    return f"""
<h2>What the audit found</h2>

<h3>1. Studies with no pixels are scored anyway, and overwrite the host</h3>
<p>{lead} A constant carries no ordering, so those studies get an arbitrary tie in place
of the ordering the host had for them. Their AUC inside the expert's own holdout reads
below chance, which is not the expert being wrong &mdash; it is tied values being broken
arbitrarily.</p>
<table><tr><th>expert</th><th>studies with 0 slots filled</th><th>AUC there</th>
<th>prediction range</th></tr>{rows}</table>
<p>It is not only the MCL expert: <b>{total} study-expert pairs</b> across the seven are
scored on nothing. Most groups are too small to measure an AUC on, which is exactly why
this was never noticed &mdash; only MCL's 194 are numerous enough to move its column
visibly. The prediction ranges are the tell: every one of them spans under 0.003, so
every one is a constant.</p>
<p><b>Fixed.</b> <code>rsna/infer/chain.py</code> already withheld the expert from a
study the cache could not cut at all, by taking its study list from the records it has.
It did not withhold from a study that <i>is</i> a record with every slot masked &mdash; a
missing row and a present-but-empty row were not the same thing to it.
<code>has_pixels()</code> now treats them the same, judging emptiness across all of an
expert's regions rather than each: a study the sagittal spec could not cut but the
coronal one could has pixels, and the model saw them.</p>
<p>Relaxing the series rule would <b>not</b> have fixed this, which is worth stating
because it is the obvious first guess. Of the 197 coronal cases, <b>none</b> is
recoverable by accepting T2 where the spec asks for PD &mdash; they carry no
fat-suppressed coronal at all. The slot that would cover 190 of them is the
non-fat-suppressed one, which is the very slot whose removal is worth +0.0117 on MCL.
The measured gain from dropping it was therefore already <i>net</i> of the hole it left,
and abstention beats either choice.</p>

<h3>2. The ablation that is already done, and never applied twice</h3>
<p>Two MCL runs differ in exactly one thing &mdash; whether the coronal <b>PD
non-fat-sat</b> slot is read:</p>
<table>
<tr><th>run</th><th>series</th><th>OOF</th></tr>
<tr><td><code>mcl</code></td><td>COR PD FS &middot; COR T2 FS &middot; COR PD noFS</td>
    <td>0.7924</td></tr>
<tr><td><code>mcl_fsonly</code></td><td>COR PD FS &middot; COR T2 FS</td>
    <td><b>0.8042</b></td></tr>
</table>
<p>Dropping that slot was worth <b>+0.0118</b>. Both coronal osteoarthritis experts
&mdash; <code>lateral_oa_coronal</code> and <code>medial_oa</code> &mdash; <b>still
carry it</b>, and the ablation has never been run for them. Two trainings, no
submission; it is the cheapest untried thing on our side.</p>

<h3>3. Thin slot coverage is the price of the specific triple</h3>
<p>The second and third series a spec asks for are found for 13&ndash;37% of studies, so
most studies are scored on one slot of two or three. That is the mask working as
designed, but it means a second slot is a bonus rather than a channel the model can lean
on &mdash; and finding 2 says a bonus can cost.</p>

<h3>4. The reading that did not survive its confound</h3>
<p>Studies with more slots filled score <i>worse</i> for four of seven experts, which
looks like evidence against the extra series. It is not. Those studies are sicker:
prevalence 0.220 against 0.403 for Lateral OA, 0.345 against 0.504 for Medial OA, 0.166
against 0.326 for MCL. A knee that got extra sequences is a knee somebody was worried
about, and the two subgroups are not comparable. Only an ablation on the same studies
settles it &mdash; which is why finding 2, and not this, is the actionable item.</p>

<h3>5. Every leg of both pipelines crops at the image centre</h3>
<p><code>read_slot</code> in the public notebook, A5's <code>read_crop</code>, Raptor's
<code>mm_crop_resize</code>, and our own <code>rsna/dicom/pixels.py</code> all compute
the window from <code>h // 2, w // 2</code>. The knee sits a median 17 mm off that
point, and half the corpus has joint anatomy outside the 130 mm window &mdash; measured
in <a href="crop.html">crop.html</a>. <b>Our experts are the only models in either
pipeline cut around anatomy rather than around the image.</b></p>

<h3>6. The cache tag does not encode which series were read</h3>
<p><code>rsna/roi/cache.py</code> names a cache
<code>{{name}}_{{W}}x{{H}}mm_{{w}}x{{h}}px_{{lat}}-{{med}}_{{N}}sl</code>. The <b>series list is not in
it</b>. So two specs that differ only in which sequences they read &mdash; exactly the
MCL pair in finding 2 &mdash; produce the same tag, and
<code>chain.py</code> skips the build when the tag already exists:</p>
<pre class="report" style="max-height:none">if not (room / f"{{tag}}.json").exists():
    cache.build(table, headers, room, spec, Config(), ...)</pre>
<p>Edit a spec's <code>series</code> without renaming it and the old cache is reused
silently, handing a two-series model a three-series tensor or the reverse. Nothing
currently triggers it, because the two MCL caches were written under different names.
It is one field in the tag away from being safe, and this page had to work around it:
it matches a cache by geometry and record width rather than by tag.</p>

<h3>7. A run's ROI cannot be resolved by name either</h3>
<p><code>expert_mcl_fsonly</code> names the roi <code>mcl</code>, whose registry entry
has since grown a third series. Reading the registry would describe a two-series run as
three-series, and at inference would hand a two-series model a three-series tensor. The
weights carry their own spec, and that is what this page and
<code>rsna/infer/experts.py</code> both read.</p>"""


def build(out: Path) -> None:
    check_probes()
    rows = expert_rows()
    empties = empty_tensor_rows(rows)

    body = f"""
<h1>Preprocessing audit: what every model is actually shown</h1>
<p class="lede">Thirty-odd public models in five legs, and our seven experts. For each:
where the series metadata comes from, which series is chosen among the candidates, how
the box is cut, and how many slices are read. The public side is quoted from the
notebook and probe-checked at build time; our side is read live from the stored specs,
the caches on disk, and the out-of-fold scores.</p>

<div class="card">
<h3 style="margin-top:0">Scope, and where it stops</h3>
<p>This is what the code <b>gives</b> each model, not what each model was
<b>trained</b> on. Training recipes live in weights we cannot inspect, except where a
shipped contract states them.</p>
<p>The four CoAtNet readers decode inside <code>raptor_light_inference</code>, a module
that is neither in the notebook nor in <code>/kaggle/working</code>. Their row is
sourced from their own runtime docstring and from the geometry contract shipped beside
their weights, and is marked as such rather than guessed.</p>
</div>
{public_section()}
{expert_section(rows)}
{findings_section(empties)}

<h2>What to run next, in the order the measurements support</h2>
<ol>
<li><b>Withhold the expert from empty studies.</b> One condition in
<code>chain.py</code>: drop a record whose mask is entirely empty so the host keeps that
study. No training, no submission.</li>
<li><b>Re-run the two coronal OA experts without the PD-noFS slot.</b> The same change
was worth +0.0118 on MCL.</li>
<li><b>Centre the wide model's window.</b> Priced in <a href="crop.html">crop.html</a>;
measurable out of fold, no submission.</li>
<li><b>Measure the offset for the coronal and axial slots.</b> crop.html covers the
sagittal series only, and three of the six wide-model slots are not sagittal.</li>
</ol>"""

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(P.page("Preprocessing audit", body))
    print(f"wrote {out}  ({out.stat().st_size / 1024:.0f} KB, "
          f"{len(PUBLIC)} public legs, {len(rows)} experts)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--out", type=Path,
                    default=ROOT / "docs/atlas/preprocessing.html")
    args = ap.parse_args()
    build(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
