"""One page per landmark model: what its point means, and where it fails.

    python -m tools.atlas.landmarking_page both
    python -m tools.atlas.landmarking_page mcl --worst 6 --typical 6

Three models place every box the experts read, and until now the only thing anyone had
looked at was `spread_mm` — the disagreement between a model's own folds, which says how
*sure* it is and never whether it is *right*.

This reads the other number. Each run's `history.json` carries an `errors` map, study by
study, in millimetres: the distance between the annotated point and the prediction of the
fold that did **not** train on that study. That is the honest figure, and it is what the
studies are ranked by here — **worst first, then a band around the median**, because a
page that only showed failures would misrepresent the model as badly as one that only
showed successes.

A second distance appears beside it and the two are not the same. The pictures draw the
point from `roi-infer/landmarks.csv`, which is what the pipeline actually uses: an average
over all five folds, and therefore partly in-sample on these annotated studies. So the
drawn gap flatters the model where the held-out error does not. Both are printed.

Images come from whichever annotation bundle recorded the click — matched on study *and*
series, then checked against the click's own recorded pixels, since the points were
collected across several bundles over three weeks and a stale pairing would draw a correct
prediction in the wrong frame.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from rsna.landmark.series import LANDMARKS                           # noqa: E402
from tools.annotate.review_page import (BUNDLES, check_transform,    # noqa: E402
                                        model_point, predicted)
from tools.atlas import page as P                                    # noqa: E402

FIGURES = ROOT / "tools" / "annotate" / "figures"

#: model run -> the points it predicts, in the order its `errors` lists them, and the
#: annotation file each was clicked into.
MODELS = {
    "both": ("landmark_both", [("lat_centre", "landmarks-both.csv"),
                               ("med_centre", "landmarks-both.csv")]),
    "mcl": ("landmark_mcl", [("mcl_centre", "landmarks-mcl.csv")]),
    "pf": ("landmark_pf", [("pf_centre", "landmarks-pf.csv")]),
}

#: What each point is for, beyond its definition — which boxes break if it moves.
SERVES = {
    "lat_centre": "the lateral meniscus box, the ACL box (with <code>med_centre</code>), "
                  "and the whole-knee boxes",
    "med_centre": "the medial meniscus box, the ACL box, <b>Baker's</b> (offset "
                  "36&nbsp;mm posterior of it), and the whole-knee boxes",
    "mcl_centre": "the MCL box and both tibiofemoral osteoarthritis boxes",
    "pf_centre": "the patellofemoral osteoarthritis box",
}


def bundle_index() -> dict:
    """(study, series) -> the bundles holding that exact rendering, newest last.

    Keyed on the series too: a study can appear in several bundles cut over different
    series, and the click's pixels belong to one of them only.
    """

    index: dict = {}
    for b in sorted(BUNDLES.glob("bundle-*")):
        manifest = json.loads((b / "studies.json").read_text())
        for entry in manifest["studies"]:
            index.setdefault((entry["study"], str(entry["series"])), []).append((b, entry))
    return index


def held_out(run: str) -> dict:
    """study -> the error of the fold that did not train on it, per point, in mm."""

    return json.loads((ROOT / "out" / run / "history.json").read_text())["errors"]


def corpus(run: str, points: list, errors: dict) -> str:
    """How many studies it saw, and why that is fewer than were clicked.

    Every study appears in `errors` exactly once — held out by one fold — so the size of
    that map *is* the training corpus. It is smaller than the number of clicks whenever a
    run needs more than one point: the landmark cache keeps only the studies carrying
    every point it was asked for, because a heatmap head cannot be trained against a
    target that is absent for half the batch.
    """

    n = len(errors)
    folds = 5
    per = n - n // folds
    clicked = []
    for point, ann_file in points:
        d = pd.read_csv(ROOT / "data/manual_annotations" / ann_file, dtype={"study": str})
        d = d[(d.point == point) & (~d.skipped.astype(bool))]
        clicked.append((point, d.study.nunique()))

    detail = ", ".join(f"<code>{P._esc(k)}</code> on {v}" for k, v in clicked)
    note = ""
    if len(clicked) > 1:
        note = (f" The two were clicked on different studies &mdash; {detail} &mdash; and "
                f"a run that predicts both keeps only the <b>{n}</b> carrying each, since "
                f"a heatmap head has nothing to learn from an absent target.")
    else:
        note = f" All {clicked[0][1]} clicks on <code>{P._esc(clicked[0][0])}</code> were usable."

    return (f"<p class=\"lede\"><b>Trained on {n} annotated studies</b>, five-fold: each "
            f"fold fits on about {per} of them and is scored on the {n - per} it never "
            f"saw, so every study is held out exactly once and the errors below cover all "
            f"{n}.{note}</p>")


def figure(point: str) -> str:
    path = FIGURES / f"{point}.jpg"
    if not path.is_file():
        return ""
    uri = "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode()
    return (f'<img src="{uri}" alt="where {P._esc(point)} goes" '
            f'style="width:100%;max-width:760px;border-radius:6px;'
            f'border:1px solid #222a35;margin:10px 0">')


def bars(values: np.ndarray) -> str:
    """The error distribution, as plain bars. No image: it has to stay readable."""

    edges = [0, 1, 2, 3, 4, 5, 7.5, 10, 20, 1e9]
    names = ["&lt;1", "1&ndash;2", "2&ndash;3", "3&ndash;4", "4&ndash;5", "5&ndash;7.5",
             "7.5&ndash;10", "10&ndash;20", "&gt;20 mm"]
    counts = [int(((values >= a) & (values < b)).sum())
              for a, b in zip(edges, edges[1:])]
    top = max(counts) or 1
    rows = []
    for name, c in zip(names, counts):
        pct = 100 * c / max(len(values), 1)
        colour = "#39d98a" if name not in ("10&ndash;20", "&gt;20 mm") else "#ff6b6b"
        rows.append(
            f'<tr><td style="width:7em;color:#8b97a8">{name}</td>'
            f'<td><div style="background:{colour};height:11px;border-radius:3px;'
            f'width:{100 * c / top:.1f}%"></div></td>'
            f'<td style="width:7em;text-align:right">{c} &middot; {pct:.0f}%</td></tr>')
    return f'<table style="width:100%;border-collapse:collapse">{"".join(rows)}</table>'


def stats_table(values: np.ndarray) -> str:
    cells = [("studies held out", f"{len(values)}"),
             ("median", f"<b>{np.median(values):.2f} mm</b>"),
             ("90th percentile", f"{np.percentile(values, 90):.2f} mm"),
             ("99th percentile", f"{np.percentile(values, 99):.2f} mm"),
             ("worst", f"<b>{values.max():.1f} mm</b>"),
             ("beyond 10 mm", f"{int((values > 10).sum())} "
                              f"({100 * (values > 10).mean():.1f} %)")]
    body = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in cells)
    return f'<table class="facts"><tbody>{body}</tbody></table>'


def draw_study(bundle: Path, entry: dict, marks: list) -> str:
    """One scrollable series carrying every point of this model, truth and prediction.

    `marks` is one tuple per thing to draw: (slice index, row, col, colour, shape, label).
    A study is shown once with all of them rather than once per point, because the two
    sagittal points are clicked on the *same* series and seeing them apart hides the only
    question worth asking of a two-headed model — whether it confused one for the other.
    """

    import cv2

    slices = sorted(entry["slices"], key=lambda s: s["i"])
    frames = []
    for s in slices:
        img = cv2.imread(str(bundle / s["file"]))
        if img is None:
            continue
        size = max(10, int(min(img.shape[:2]) / 24))
        for i, row, col, colour, shape, _ in marks:
            if i != s["i"]:
                continue
            bgr = tuple(int(colour[j:j + 2], 16) for j in (5, 3, 1))
            cv2.drawMarker(img, (int(round(col)), int(round(row))), bgr,
                           cv2.MARKER_TILTED_CROSS if shape == "gt" else cv2.MARKER_CROSS,
                           size, 2)
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 72])
        if ok:
            frames.append("data:image/jpeg;base64,"
                          + base64.b64encode(buf.tobytes()).decode())
    if not frames:
        return ""

    legend = " &nbsp; ".join(
        f"<span style='color:{c}'>{'&#10006;' if sh == 'gt' else '&#10010;'} {label}</span>"
        for _, _, _, c, sh, label in marks)
    return P.viewer(f"lm-{entry['tail']}", frames, entry["first_end"], entry["last_end"],
                    entry["mm_per_px_shown"],
                    caption=legend + " &mdash; &#10006; annotated, &#10010; predicted, "
                                     "each on the slice it belongs to.")


def holdout_points(run: str, points: list) -> dict:
    """point -> {study: the mm coordinates of the fold that held that study out}.

    Written by `scripts.predict_landmark_holdout`, which checks itself against the
    distances `history.json` recorded before writing anything.
    """

    path = ROOT / "out" / run / "holdout_points.csv"
    if not path.is_file():
        raise SystemExit(f"{path} is missing — run scripts.predict_landmark_holdout first")
    table = pd.read_csv(path, dtype={"study": str})
    return {point: {r.study: np.array([r.x_mm, r.y_mm, r.z_mm], float)
                    for r in table[table.point == point].itertuples()}
            for point, _ in points}


def collect(points: list, errors: dict, index: dict, preds: dict, holdout: dict) -> list:
    """One record per study: its worst held-out error, and everything to draw on it."""

    per_point = {}
    for column, (point, ann_file) in enumerate(points):
        d = pd.read_csv(ROOT / "data/manual_annotations" / ann_file, dtype={"study": str})
        d = d[(d.point == point) & (~d.skipped.astype(bool))]
        per_point[point] = {r.study: r for _, r in d.iterrows()}

    out = []
    for study, err in errors.items():
        values = err if isinstance(err, list) else [err]
        found, marks, labels, worst = None, [], [], 0.0
        for column, (point, _) in enumerate(points):
            click = per_point[point].get(study)
            if click is None:
                continue
            if found is None:
                for bundle, entry in index.get((study, str(click.series)), []):
                    if check_transform(entry, click) < 3.0:
                        found = (bundle, entry)
                        break
            if found is None:
                continue
            colour = LANDMARKS[point]["colour"]
            marks.append((int(click["slice"]), float(click["jpeg_row"]),
                          float(click["jpeg_col"]), colour, "gt", point))
            # The point drawn is the **held-out** one: predicted by the single fold that
            # never saw this study. Drawing `roi-infer` instead — the mean of five, four
            # of which trained on it — showed a model that looked right where its honest
            # answer was 53 mm out, which is not a validation view at all. The mean is
            # still reported beside it, because it is what the boxes are actually cut
            # from, and the gap between the two columns is the in-sample flattery.
            held = float(values[column])
            q = holdout[point].get(study)
            if q is not None:
                guess = model_point(found[1], q)
                if guess:
                    marks.append((guess[0], guess[1], guess[2], colour, "model", point))
            shipped = spread = None
            pred = preds[point]
            if study in pred.index:
                row = pred.loc[study]
                shipped = float(np.linalg.norm(
                    row[["x_mm", "y_mm", "z_mm"]].to_numpy(float)
                    - click[["x_mm", "y_mm", "z_mm"]].to_numpy(float)))
                spread = float(row["spread_mm"])
            worst = max(worst, held)
            labels.append(
                f"<span style='color:{colour}'><b>{P._esc(point)}</b> "
                f"held out <b>{held:.1f} mm</b>"
                + (f", shipped mean {shipped:.1f} mm" if shipped is not None else "")
                + (f", folds spread {spread:.1f} mm" if spread is not None else "") + "</span>")
        if found and marks:
            out.append((worst, found, marks, labels))
    return out


def gallery(records: list, worst: int, typical: int) -> str:
    """The failures, then a band around the median."""

    records.sort(key=lambda r: -r[0])
    blocks = []
    for title, chosen in (("Where it fails", records[:worst]),
                          ("And where it does not", _around_median(records, typical))):
        blocks.append(f"<h3>{title}</h3>")
        if title.startswith("And"):
            blocks.append('<p class="note">A band around the median, so the eye has '
                          'something to calibrate against. These are the common case.</p>')
        for _, (bundle, entry), marks, labels in chosen:
            html = draw_study(bundle, entry, marks)
            if html:
                blocks.append(f"<h4>&hellip;{P._esc(entry['tail'])} "
                              f"<span class='meta'>{entry['side']} knee &middot; "
                              f"{' &middot; '.join(labels)}</span></h4>" + html)
    return "".join(blocks)


def _around_median(rows: list, n: int) -> list:
    if not rows or n <= 0:
        return []
    ordered = sorted(rows, key=lambda r: r[0])
    mid = len(ordered) // 2
    lo = max(0, mid - n // 2)
    return ordered[lo:lo + n]


def build(key: str, out: Path, worst: int, typical: int) -> Path:
    run, points = MODELS[key]
    errors = held_out(run)
    index = bundle_index()
    # One frame per point, indexed by study. Reading the whole table and testing
    # `study in pred.index` against a RangeIndex is always false, which silently drew no
    # prediction at all rather than failing — the shape of bug this page exists to catch.
    raw = predicted()
    preds = {point: raw[raw.point == point].set_index("study")
             for point, _ in points}
    config = json.loads((ROOT / "out" / run / "history.json").read_text()).get("run", {})
    holdout = holdout_points(run, points)
    records = collect(points, errors, index, preds, holdout)

    sections = []
    for column, (point, ann_file) in enumerate(points):
        meta = LANDMARKS[point]
        values = np.array([(e[column] if isinstance(e, list) else e)
                           for e in errors.values()], float)
        sections.append(f"""
<h2><code>{P._esc(point)}</code> &mdash; {P._esc(meta['what'])}</h2>
<p class="lede">Clicked on a <b>{P._esc(meta['plane'])}</b> slice.
{'Annotated toward the <b>' + P._esc(meta['click_near']) + '</b> end of the stack. '
 if meta.get('click_near') else ''}
It places {SERVES.get(point, 'boxes in the registry')}.</p>
{figure(point)}

<h3>How far off it is, held out</h3>
{stats_table(values)}
{bars(values)}
""")

    body = f"""
<style>
table.facts{{border-collapse:collapse;margin:12px 0 18px;width:100%}}
table.facts td{{padding:6px 10px;border-top:1px solid #222a35}}
table.facts td:first-child{{color:#8b97a8;width:42%}}
.lede{{color:#aab4c2;font-size:15px;line-height:1.6}}
</style>
<h1><code>{P._esc(run)}</code></h1>
<p class="lede">{len(points)} point{'s' if len(points) > 1 else ''},
5 folds, {config.get('epochs', '?')} epochs, batch {config.get('batch', '?')},
lr {config.get('lr', '?')}, heatmap weight {config.get('weight', '?')}.</p>
{corpus(run, points, errors)}

<p class="note"><b>Which number is which.</b> The millimetres in each heading are the
<b>held-out</b> error: the annotated point against the prediction of the fold that did not
train on that study. The gap printed inside each viewer is a different thing &mdash; the
annotated point against <code>roi-infer/landmarks.csv</code>, which the pipeline uses and
which averages all five folds, so it is partly in-sample here and reads better than it
should. Rank on the first; look at the second only to see which way the model drifted.</p>

<p class="note">The annotation is one person's. A large distance is a disagreement, and
the picture is there so you can judge which side is wrong.</p>
{''.join(sections)}

<h2>Study by study</h2>
<p class="note"><b>The cross drawn is the held-out one</b> &mdash; placed by the single
fold that never saw this study &mdash; and that is also what the ranking uses, taking the
worse of this model's points. The other two numbers describe the same study and are printed
so they cannot be mistaken for it: <b>shipped mean</b> is how far
<code>roi-infer/landmarks.csv</code> lands, which is what the boxes are actually cut from
but which averages five folds and is therefore mostly in-sample here; <b>folds spread</b>
is how far those five disagree. A held-out error far above the shipped mean is the
in-sample flattery made visible, and the spread tells you it was not a quiet one.
{len(records)} studies carry both a usable click and an image.</p>
{gallery(records, worst, typical)}
"""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(P.page(f"{run} — what it places and where it fails", body))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model", choices=sorted(MODELS))
    ap.add_argument("-o", "--out", type=Path, default=None)
    ap.add_argument("--worst", type=int, default=6)
    ap.add_argument("--typical", type=int, default=6)
    args = ap.parse_args()
    out = args.out or ROOT / "docs" / "atlas" / "landmarking" / f"{args.model}.html"
    path = build(args.model, out, args.worst, args.typical)
    print(f"{path}  ({path.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
