"""Build a self-contained annotation bundle from the raw DICOM.

    python -m tools.annotate.bundle --n 30 --out /data/mgr/rsna-knee/bundle-v1

Annotating does not need 533 GB of DICOM. It needs a few hundred studies rendered at
the resolution a human can actually click on, which is a couple of hundred megabytes —
so the bundle is built once, on the machine that holds the data, and the annotation
session afterwards depends on nothing.

Three properties the bundle has to have, because each of them protects an annotation
that would otherwise be quietly wrong:

* **A constant physical field of view.** Every slice is rendered to the same
  millimetres-across, whatever its native spacing. A knee that looks three times larger
  than the last one gets its landmark placed differently, and that variance goes
  straight into the model's error floor.

* **Enough resolution to be worth clicking.** 180 mm over 576 px is 0.31 mm/px, against
  a native median of 0.33 — so a click is as precise as the acquisition allows, and
  re-annotating at higher resolution later would gain nothing.

* **Everything needed to recover patient millimetres**, per slice: the SOP Instance UID,
  `ImagePositionPatient`, `ImageOrientationPatient`, the pixel spacing, and the crop and
  scale this renderer applied. A click is stored as a pixel on a named slice; the
  millimetres are *derived*, and can be re-derived if the derivation is ever found wrong.

What is deliberately **not** done here: no laterality mirroring and no reordering beyond
the geometric sort. The bundle shows the acquisition as it is. Which end is lateral is
*reported* so the annotator can see it, not *imposed* by flipping the pixels — a stored
patient coordinate is independent of any convention we might later change our mind about.
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

from rsna.config import Config, TARGETS                                  # noqa: E402
from tools.atlas.render import window                                    # noqa: E402
from tools.atlas.study import (TRAIN_SERIES, load_series, series_headers,  # noqa: E402
                               side_of, stack_orientation)

#: The field of view every slice is rendered to, in millimetres. Above the 95th
#: percentile of the corpus (200 mm) it would be mostly padding; below the 5th (139 mm)
#: it would crop anatomy. 180 keeps the joint on every study we measured.
FOV_MM = 180.0

#: 180 mm / 576 px = 0.3125 mm/px, against a native median of 0.33. Clicking is then
#: limited by the acquisition, not by the bundle.
OUT_PX = 576

JPEG_QUALITY = 85


def patient_mm(ipp, iop, spacing, row: float, col: float) -> list[float]:
    """Pixel (row, col) of a slice -> position in the patient, in millimetres.

    The DICOM formula. `ImageOrientationPatient` is the direction cosines of the first
    row and the first column: the first triplet points along increasing *column* index,
    the second along increasing *row* index. `PixelSpacing` is (between rows, between
    columns), so it pairs with them the other way round — which is the detail this
    function exists to get right once instead of at every call site.
    """

    ipp = np.asarray(ipp, float)
    iop = np.asarray(iop, float)
    return list(ipp + col * float(spacing[1]) * iop[0:3]
                    + row * float(spacing[0]) * iop[3:6])


def render(slice_: np.ndarray, mm_per_px: float) -> tuple[np.ndarray, dict]:
    """One slice at a constant physical field of view, plus the transform to undo it."""

    want = int(round(FOV_MM / mm_per_px))
    h, w = slice_.shape
    cy, cx = h // 2, w // 2
    r0, c0 = cy - want // 2, cx - want // 2

    # Pad rather than shrink the field: a smaller acquisition keeps its scale and gains
    # a black margin, so the anatomy is the same size on screen in every study.
    canvas = np.zeros((want, want), slice_.dtype)
    sr0, sc0 = max(r0, 0), max(c0, 0)
    sr1, sc1 = min(r0 + want, h), min(c0 + want, w)
    canvas[sr0 - r0:sr1 - r0, sc0 - c0:sc1 - c0] = slice_[sr0:sr1, sc0:sc1]

    out = cv2.resize(canvas, (OUT_PX, OUT_PX), interpolation=cv2.INTER_AREA)
    return out, {"row0": r0, "col0": c0, "scale": want / OUT_PX}


def pick_sagittal(headers: pd.DataFrame):
    """The sagittal series a meniscus is actually read on: PD first, then T2, then T1."""

    sag = headers[headers["plane"] == "Sagittal"]
    if not len(sag):
        return None
    for weight in ("PD", "T2", "T1"):
        hit = sag[sag["weight"] == weight]
        if len(hit):
            # Prefer no fat suppression: short-TE without FS shows signal *inside* the
            # fibrocartilage best, which is the whole basis of meniscal grading.
            hit = hit.sort_values(["fatsat", "n_slices"], ascending=[True, False])
            return hit.iloc[0]
    return sag.sort_values("n_slices", ascending=False).iloc[0]


def build_study(uid: str, out: Path, config: Config, headers=None) -> dict | None:
    headers = series_headers(uid) if headers is None else headers
    chosen = pick_sagittal(headers)
    if chosen is None:
        return None

    side, how = side_of(headers)
    series = load_series(chosen, config)
    first_end, last_end = stack_orientation("Sagittal", side)

    tail = uid[-11:]
    folder = out / "img" / tail
    folder.mkdir(parents=True, exist_ok=True)

    eight = window(series.volume)
    directory = Path(chosen["dir"])
    names = _ordered_names(directory, config)

    slices, transform = [], None
    for i in range(series.n):
        image, transform = render(eight[i], series.mm_per_px)
        cv2.imwrite(str(folder / f"{i:03d}.jpg"), image,
                    [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        meta = _slice_meta(directory, names[i]) if i < len(names) else {}
        slices.append({"i": i, "file": f"img/{tail}/{i:03d}.jpg", **meta})

    # The field is rounded to a whole number of native pixels, so the rendered scale is
    # very slightly off the nominal FOV_MM/OUT_PX. Publish the value that is true for
    # this study rather than the one that is true for the design.
    shown = transform["scale"] * series.mm_per_px if transform else FOV_MM / OUT_PX

    # SliceThickness is the slab, not the gap between slice centres — they differ
    # whenever the acquisition has a skip. Physical resampling needs the measured step.
    steps = np.diff([p for p in series.positions if np.isfinite(p)])
    spacing = float(np.median(np.abs(steps))) if len(steps) else series.thickness

    return {
        "study": uid, "tail": tail, "series": str(chosen["SeriesInstanceUID"]),
        "sequence": series.label, "description": series.description,
        "side": side, "side_from": how,
        "first_end": first_end, "last_end": last_end,
        "n": series.n, "ordered": bool(series.ordered),
        "mm_per_px_native": round(series.mm_per_px, 5),
        "mm_per_px_shown": round(shown, 6),
        "slice_spacing_mm": round(spacing, 3),
        "thickness": series.thickness, "fov_mm": FOV_MM, "out_px": OUT_PX,
        "transform": transform, "slices": slices,
    }


def _ordered_names(directory: Path, config: Config) -> list[str]:
    from rsna.dicom.ordering import order_slices

    names = sorted(p.name for p in directory.glob("*.dcm"))
    names, _ = order_slices(str(directory), names, config)
    return names


def _slice_meta(directory: Path, name: str) -> dict:
    """What a click on this slice needs to become a position in the patient."""

    import pydicom

    try:
        ds = pydicom.dcmread(str(directory / name), force=True, stop_before_pixels=True)
        return {"sop": str(ds.SOPInstanceUID),
                "ipp": [float(v) for v in ds.ImagePositionPatient],
                "iop": [float(v) for v in ds.ImageOrientationPatient],
                "ps": [float(v) for v in ds.PixelSpacing]}
    except Exception:
        # A slice with no geometry cannot be converted to millimetres. Say so rather
        # than invent one: the annotator can still click it, and the click is stored.
        return {"sop": name.replace(".dcm", ""), "ipp": None, "iop": None, "ps": None}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=30, help="how many studies")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--gold", action="store_true",
                    help="draw from the 58 expert-labelled studies instead of the corpus")
    ap.add_argument("--tagged-only", action="store_true",
                    help="keep only studies whose side comes from the DICOM tag. The "
                         "geometric fallback disagrees with the tag on about 5%% of the "
                         "studies where both exist, and a wrong side means annotating "
                         "the opposite meniscus — 40 mm away, with confidence. Worth "
                         "paying for a first training set; the model can resolve the "
                         "untagged half afterwards.")
    args = ap.parse_args()

    train = pd.read_csv(ROOT / "data/raw/train.csv")
    if args.gold:
        train = train[train[TARGETS].notna().all(axis=1)]
    pool = [u for u in train["StudyInstanceUID"] if (TRAIN_SERIES / u).is_dir()]
    rng = np.random.default_rng(args.seed)
    chosen = list(rng.permutation(pool)[:args.n])

    args.out.mkdir(parents=True, exist_ok=True)
    config = Config()
    studies, failed = [], []
    for k, uid in enumerate(chosen, 1):
        try:
            # The side comes from five header reads; rendering a study costs thirty
            # full decodes and a hundred JPEG writes. So the filter runs first —
            # otherwise half the build time is spent on studies that get deleted.
            headers = series_headers(uid)
            side, how = side_of(headers)
            if args.tagged_only and "tag" not in (how or ""):
                failed.append((uid, f"side not from the tag ({how})"))
                continue

            record = build_study(uid, args.out, config, headers=headers)
            if record is None:
                failed.append((uid, "no sagittal series"))
            else:
                studies.append(record)
        except Exception as exc:  # noqa: BLE001
            failed.append((uid, f"{type(exc).__name__}: {exc}"))
        print(f"  {k}/{len(chosen)}  {uid[-11:]}", flush=True)

    # No global mm/px: it is a property of each study's rounding, not of the bundle.
    manifest = {"fov_mm": FOV_MM, "out_px": OUT_PX,
                "seed": args.seed, "pool": "gold" if args.gold else "train",
                "studies": studies}
    (args.out / "studies.json").write_text(json.dumps(manifest))

    # The bundle carries its own annotator, so what ships is one self-sufficient
    # directory: serve it with `python -m http.server` and there is nothing else to install.
    import shutil
    shutil.copy(Path(__file__).with_name("annotate.html"), args.out / "index.html")

    size = sum(f.stat().st_size for f in args.out.rglob("*") if f.is_file())
    print(f"\n{len(studies)} studies, {sum(s['n'] for s in studies)} slices, "
          f"{size/1e6:.0f} MB -> {args.out}")
    unresolved = [s["tail"] for s in studies if s["side"] is None]
    if unresolved:
        print(f"side UNRESOLVED on {len(unresolved)}: {unresolved}")
    for uid, why in failed:
        print(f"  FAILED {uid[-11:]}: {why}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
