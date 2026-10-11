"""Your clicks and the model's predictions, on the same images.

    python -m tools.annotate.review_page med -o docs/atlas/landmarks/med.html

The annotation tool collects points and forgets them: it exports a JSON, which `to_mm`
turns into `data/manual_annotations/*.csv`, and nothing afterwards puts them back on the
pictures. So the only view anyone has had of the landmark model is `spread_mm`, the
disagreement between its own folds — which says how sure it is, never whether it is right.

This page answers the other question. Both points are drawn in the bundle's own pixel
frame: the click as it was recorded, and the prediction projected through the slice
geometry the bundle stored beside each image. Studies are ordered **worst first**, so
what opens is the model's failures rather than a gallery of agreements.

Two cautions. The annotation is one person's, so a large distance is a disagreement and
not proof of an error. And the model was fitted on these studies, so its agreement here
flatters it — the honest reading is the shape of the tail, not the median.
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

from rsna.dicom.geometry import normal_of, pixel_of, through_plane   # noqa: E402
from tools.atlas import page as P                                    # noqa: E402

BUNDLES = Path("/data/mgr/rsna-knee")
PREDICTED = BUNDLES / "roi-infer" / "landmarks.csv"

#: bundle directory, the annotation file it was clicked into, and the point it carries.
SETS = {
    "med": ("bundle-med", "landmarks-med.csv", "med_centre"),
    "lat": ("bundle-acl", "landmarks-both.csv", "lat_centre"),
    "mcl": ("bundle-mcl", "landmarks-mcl.csv", "mcl_centre"),
    "pf": ("bundle-pf", "landmarks-pf.csv", "pf_centre"),
}

CLICK, MODEL = "#ffd166", "#39d98a"


def manifest(bundle: Path) -> dict:
    return json.loads((bundle / "studies.json").read_text())


def predicted() -> pd.DataFrame:
    return pd.read_csv(PREDICTED, dtype={"study": str})


def to_shown(entry: dict, row_native: float, col_native: float) -> tuple[float, float]:
    """Native pixels to the pixels of the JPEG the tool displayed.

    `transform` is what `bundle.render` recorded in order to **undo** itself —
    `scale = want / OUT_PX`, the factor the renderer divided by — so the mapping forward
    subtracts the origin and divides. Getting that backwards put the marker at (1070,
    667) in a 576-pixel image, off-canvas and therefore invisible rather than visibly
    wrong, which is why `check_transform` below now refuses to let it pass unnoticed.
    """

    t = entry["transform"]
    return ((row_native - t["row0"]) / t["scale"],
            (col_native - t["col0"]) / t["scale"])


def check_transform(entry: dict, click: pd.Series) -> float:
    """How far this mapping lands from where the click was actually recorded.

    The annotation carries both its millimetres and its pixels, so the forward mapping
    can be validated per study against a point whose answer is known. Anything beyond a
    pixel or two means the geometry is being read wrongly, and a page that silently drew
    the model somewhere plausible would be worse than one that did not draw it at all.
    """

    r, c = to_shown(entry, float(click["native_row"]), float(click["native_col"]))
    return float(np.hypot(r - float(click["jpeg_row"]), c - float(click["jpeg_col"])))


def model_point(entry: dict, point: np.ndarray) -> tuple[int, float, float] | None:
    """Which displayed slice the prediction sits on, and where on it.

    The bundle kept `ipp`, `iop` and `ps` per slice, so the projection uses the same
    geometry the pipeline uses and needs no DICOM read.
    """

    slices = [s for s in entry["slices"] if s.get("ipp") and s.get("iop")]
    if not slices:
        return None
    axis = normal_of(np.asarray(slices[0]["iop"], float))
    depth = np.array([through_plane(np.asarray(s["ipp"], float), axis) for s in slices])
    k = int(np.argmin(np.abs(depth - through_plane(point, axis))))
    s = slices[k]
    r, c = pixel_of(s["ipp"], s["iop"], s["ps"], point)
    rr, cc = to_shown(entry, r, c)
    return s["i"], rr, cc


def one_study(bundle: Path, entry: dict, click: pd.Series,
              point: np.ndarray | None) -> tuple[str, float | None]:
    """One scrollable series with the click on its slice and the model's on its own."""

    slices = sorted(entry["slices"], key=lambda s: s["i"])
    guess = model_point(entry, point) if point is not None else None
    gap = None
    if point is not None:
        gap = float(np.linalg.norm(point - click[["x_mm", "y_mm", "z_mm"]].to_numpy(float)))

    drift = check_transform(entry, click)
    if drift > 3.0:
        guess = None

    frames = []
    for s in slices:
        img = cv2.imread(str(bundle / s["file"]))
        if img is None:
            continue
        size = max(10, int(min(img.shape[:2]) / 26))
        if s["i"] == int(click["slice"]):
            bgr = tuple(int(CLICK[j:j + 2], 16) for j in (5, 3, 1))
            cv2.drawMarker(img, (int(round(click["jpeg_col"])),
                                 int(round(click["jpeg_row"]))),
                           bgr, cv2.MARKER_TILTED_CROSS, size, 2)
        if guess and s["i"] == guess[0]:
            bgr = tuple(int(MODEL[j:j + 2], 16) for j in (5, 3, 1))
            cv2.drawMarker(img, (int(round(guess[2])), int(round(guess[1]))),
                           bgr, cv2.MARKER_CROSS, size, 2)
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 72])
        if ok:
            import base64
            frames.append("data:image/jpeg;base64,"
                          + base64.b64encode(buf.tobytes()).decode())

    if not frames:
        return "", gap
    tail = entry["tail"]
    where = (f"click on slice {int(click['slice'])}"
             + (f", model on slice {guess[0]}" if guess else
                (f", model withheld &mdash; this bundle's transform reproduces the "
                 f"recorded click {drift:.0f} px off" if drift > 3.0
                 else ", model not predicted")))
    dist = "&mdash;" if gap is None else f"{gap:.1f} mm apart"
    head = (f"<h3>&hellip;{P._esc(tail)} "
            f"<span class='meta'>{entry['side']} knee &middot; {where} &middot; "
            f"<b>{dist}</b></span></h3>")
    return head + P.viewer(f"lm-{tail}", frames, entry["first_end"], entry["last_end"],
                           entry["mm_per_px_shown"],
                           caption=f"<span style='color:{CLICK}'>&#10006; your click</span>"
                                   f" &nbsp; <span style='color:{MODEL}'>&#10010; the "
                                   f"model</span> &mdash; each on the slice it belongs to."), gap


def build(key: str, out: Path, n: int, order: str) -> Path:
    bundle_name, ann_name, point_name = SETS[key]
    bundle = BUNDLES / bundle_name
    man = manifest(bundle)
    by_study = {e["study"]: e for e in man["studies"]}

    clicks = pd.read_csv(ROOT / "data/manual_annotations" / ann_name,
                         dtype={"study": str})
    clicks = clicks[(clicks.point == point_name) & (~clicks.skipped.astype(bool))]
    clicks = clicks[clicks.study.isin(by_study)]

    pred = predicted()
    pred = pred[pred.point == point_name].set_index("study")

    rows = []
    for _, click in clicks.iterrows():
        p = None
        if click.study in pred.index:
            p = pred.loc[click.study, ["x_mm", "y_mm", "z_mm"]].to_numpy(float)
        gap = (None if p is None else
               float(np.linalg.norm(p - click[["x_mm", "y_mm", "z_mm"]].to_numpy(float))))
        rows.append((click, p, gap))

    gaps = np.array([g for _, _, g in rows if g is not None])
    if order == "worst":
        rows.sort(key=lambda r: -(r[2] if r[2] is not None else -1))
    shown = rows[:n]

    blocks = []
    for click, p, _ in shown:
        html, _ = one_study(bundle, by_study[click.study], click, p)
        if html:
            blocks.append(html)

    q = (f"median <b>{np.median(gaps):.1f} mm</b>, "
         f"90th percentile {np.percentile(gaps, 90):.1f} mm, "
         f"worst {gaps.max():.1f} mm" if len(gaps) else "no prediction to compare")
    body = f"""
<h1>{P._esc(point_name)} &mdash; your clicks against the model</h1>
<p class="note">{len(clicks)} studies you annotated, all of them present in
<code>{P._esc(bundle_name)}</code>. Over those, the model lands {q}.
Shown below: the {len(shown)} {'furthest apart' if order == 'worst' else 'first'}.</p>

<p class="note"><b>Two cautions.</b> The annotation is one person's, so a large distance
is a disagreement, not a proven error &mdash; the picture is there for you to judge which.
And the landmark model was fitted on these studies, so its agreement here flatters it;
what is worth reading is the shape of the tail, not the median.</p>

{''.join(blocks)}
"""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(P.page(f"{point_name} — clicks against the model", body))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("point", choices=sorted(SETS))
    ap.add_argument("-o", "--out", type=Path, default=None)
    ap.add_argument("-n", type=int, default=20, help="How many studies to show.")
    ap.add_argument("--order", choices=("worst", "file"), default="worst")
    args = ap.parse_args()
    out = args.out or ROOT / "docs" / "atlas" / "landmarks" / f"{args.point}.html"
    path = build(args.point, out, args.n, args.order)
    print(f"{path}  ({path.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
