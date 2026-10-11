"""Where bone contusions are, measured rather than assumed.

    python -m tools.atlas.contusion_map --per-arm 250

Writes `/data/mgr/rsna-knee/contusion-map.npz` and prints what it found. No model, no
annotation of a lesion: studies whose report mentions a contusion are compared with
studies whose report does not, by how often each cell of a millimetre grid holds one of
the brightest voxels of its own slice.

**Why this target suits the method.** A bone contusion *is* marrow oedema, and oedema is
bright on a fat-suppressed sequence. So the statistic the method computes — "positives
carry more bright voxels here" — is the finding itself rather than a proxy for it. On
fracture the same map would find the oedema *around* a line it cannot see, which is
usable but one step removed; this is the cleaner of the two and so it goes first.

**What it decides.** Three boxes hung off landmarks cost one training run of forty
minutes and keep the 0.21 mm/px that is our whole advantage over the public pipeline's
0.42. A whole-knee box at that resolution is 28 times the pixel budget: batch 8 instead
of 32, and about fifteen hours for five folds. Which of the two is right depends on
whether the excess concentrates or is flat, and nothing short of measuring says which.

**The grid is anchored between the two menisci**, at the midpoint of `lat_centre` and
`med_centre` — the intercondylar notch, the most central point the landmark models give
us, and the one the ACL box already hangs from. A contusion can sit anywhere in the
distal femur, the proximal tibia or the patella, so unlike Baker's there is no landmark
near the finding and the origin only has to be *stable*.

**The out-of-stack guard is the whole difficulty at this size.** Baker's grid reached
20 mm above and below its landmark; this one reaches 70, and most acquisitions are not
140 mm tall. A cell nobody acquired must stay NaN and be dropped, never silently read
from the nearest end slice — doing exactly that once invented a +42 mm effusion peak out
of 87.5 % of studies answering with their top slice. So every cell carries a coverage
count beside it, and anything measured on fewer than `MIN_COVER` of each arm is masked
out of the result rather than reported thinly.
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from rsna.dicom.geometry import normal_of, pixel_of, through_plane    # noqa: E402

HEADERS = Path("/data/mgr/rsna-knee/headers.parquet")
LANDMARKS = Path("/data/mgr/rsna-knee/roi-infer/landmarks.csv")
LABELS = ROOT / "data/external/stevenleehans-rsna-knee-llm-report-labels/llm_labels_v4_blend.csv"
MAP_CACHE = Path("/data/mgr/rsna-knee/contusion-map.npz")

#: Millimetres from the notch centre. Coarser than Baker's 2 mm because the volume is
#: ten times larger and the question is "which region", not "which centimetre".
ML = np.arange(-60, 61, 4.0)      #: medial(-) to lateral(+), signed by the knee's side
AP = np.arange(-60, 61, 4.0)      #: posterior(-) to anterior(+)
SI = np.arange(-70, 71, 5.0)      #: inferior(-) to superior(+)

#: A cell must be acquired on at least this share of an arm to be reported at all.
MIN_COVER = 0.60

def log(message: str) -> None:
    print(message, flush=True)


def fs_series(headers: pd.DataFrame, plane: str) -> pd.DataFrame:
    """One fat-suppressed series per study in the wanted plane, the longest of them.

    Fat-suppressed because marrow oedema is only conspicuous where the marrow's own fat
    has been suppressed; on a T1 the same lesion is a *darkening* and would need the
    opposite threshold.
    """

    hit = headers[(headers.plane.astype(str) == plane) & headers.fatsat.astype(bool)]
    return hit.sort_values("n_slices", ascending=False).drop_duplicates("StudyInstanceUID")


def _one_study(row, point: np.ndarray) -> np.ndarray | None:
    """Bright-voxel indicators on the grid, NaN wherever nothing was acquired.

    Thresholded at each slice's own 97th percentile: MR intensity has no absolute scale,
    so the statistic has to be a rank within the image rather than a value, and a rank
    taken per slice rather than per study survives the intensity drift along a stack.
    """

    try:
        folder = Path(row["dir"])
        names = sorted(p.name for p in folder.glob("*.dcm"))
        if len(names) < 8:
            return None
        heads = [pydicom.dcmread(str(folder / n), force=True, stop_before_pixels=True)
                 for n in names]
        iop = [float(v) for v in heads[0].ImageOrientationPatient]
        spacing = [float(v) for v in heads[0].PixelSpacing]
        normal = normal_of(iop)
        depth = np.array([through_plane(
            [float(v) for v in h.ImagePositionPatient], normal) for h in heads])
        order = np.argsort(depth)
        names = [names[i] for i in order]
        heads = [heads[i] for i in order]
        depth = depth[order]
        step = float(np.median(np.abs(np.diff(depth)))) if len(depth) > 1 else 4.0

        # Lateral is +x on a left knee and -x on a right one; the landmark's own x says
        # which. Reading this backwards mirrors the whole map.
        lateral = np.sign(point[0]) or 1.0

        out = np.full((len(SI), len(ML), len(AP)), np.nan, np.float32)
        cached: dict[int, tuple[np.ndarray, float]] = {}
        for k, si in enumerate(SI):
            want = through_plane(point + np.array([0.0, 0.0, si]), normal)
            # The guard. A height outside the acquisition is not the nearest slice: it is
            # nothing, and must stay NaN. Half a slice of tolerance, never more.
            if want < depth.min() - step / 2 or want > depth.max() + step / 2:
                continue
            j = int(np.argmin(np.abs(depth - want)))
            if j not in cached:
                image = pydicom.dcmread(str(folder / names[j]),
                                        force=True).pixel_array.astype(np.float32)
                cached[j] = (image, float(np.percentile(image, 97.0)))
            image, level = cached[j]
            for a, ml in enumerate(ML):
                probes = point + np.stack([np.full_like(AP, ml * lateral), AP,
                                           np.full_like(AP, si)], axis=1)
                rc = np.array([pixel_of(heads[j].ImagePositionPatient, iop, spacing, q)
                               for q in probes])
                rows = np.clip(np.round(rc[:, 0]).astype(int), 0, image.shape[0] - 1)
                cols = np.clip(np.round(rc[:, 1]).astype(int), 0, image.shape[1] - 1)
                inside = ((rc[:, 0] >= 0) & (rc[:, 0] < image.shape[0])
                          & (rc[:, 1] >= 0) & (rc[:, 1] < image.shape[1]))
                values = (image[rows, cols] > level).astype(np.float32)
                values[~inside] = np.nan
                out[k, a] = values
        return out
    except Exception:                                                # noqa: BLE001
        return None


def arm_mean(frame: pd.DataFrame, points: pd.DataFrame,
             workers: int) -> tuple[np.ndarray, np.ndarray]:
    """The arm's mean indicator per cell, and the share of studies that reached it."""

    jobs = []
    for _, row in frame.iterrows():
        uid = row["StudyInstanceUID"]
        lat = points.loc[(uid, "lat_centre")][["x_mm", "y_mm", "z_mm"]].to_numpy(float)
        med = points.loc[(uid, "med_centre")][["x_mm", "y_mm", "z_mm"]].to_numpy(float)
        jobs.append((row, (lat + med) / 2.0))
    with ThreadPoolExecutor(workers) as pool:
        got = [g for g in pool.map(lambda j: _one_study(*j), jobs) if g is not None]
    log(f"    {len(got)} of {len(jobs)} studies read")
    stack = np.stack(got)
    cover = np.mean(~np.isnan(stack), axis=0)
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(stack, axis=0)
    return mean, cover


def build(plane: str, per_arm: int, workers: int) -> dict:
    headers = pd.read_parquet(HEADERS)
    headers["fatsat"] = headers["fatsat"].astype(bool)
    points = pd.read_csv(LANDMARKS, dtype={"study": str})
    points = points[points.point.isin(("lat_centre", "med_centre"))]
    # Which studies carry *both*: the origin is their midpoint, so one of the two is of
    # no use on its own. Counted before indexing, since `to_frame()` on a MultiIndex
    # leaves `study` as both a level and a column and `groupby` refuses that.
    counts = points.groupby("study").point.nunique()
    both = set(counts[counts == 2].index)
    points = points.set_index(["study", "point"])

    labels = pd.read_csv(LABELS, dtype={"StudyInstanceUID": str}).set_index(
        "StudyInstanceUID")
    column = next(c for c in labels.columns if "ontusion" in c)

    frame = fs_series(headers, plane)
    frame = frame[frame.StudyInstanceUID.isin(both)
                  & frame.StudyInstanceUID.isin(labels.index)]
    reported = (labels.loc[frame.StudyInstanceUID, column] > 0.5).values
    log(f"  {plane} fat-suppressed: {len(frame)} studies with both landmarks, "
        f"{int(reported.sum())} reported")

    out = {}
    for name, subset in (("positive", frame[reported]), ("negative", frame[~reported])):
        take = subset.sample(min(per_arm, len(subset)), random_state=7)
        log(f"  {name} arm, {len(take)} studies")
        out[f"{name}_mean"], out[f"{name}_cover"] = arm_mean(take, points, workers)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plane", default="Coronal",
                    help="Coronal first: it spans both condyles and both plateaus, which "
                         "is where contusions sit, in one image.")
    ap.add_argument("--per-arm", type=int, default=250)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--out", type=Path, default=MAP_CACHE)
    args = ap.parse_args()

    log(f"contusion map, {args.plane}, {args.per_arm} per arm")
    got = build(args.plane, args.per_arm, args.workers)
    excess = got["positive_mean"] - got["negative_mean"]
    thin = (got["positive_cover"] < MIN_COVER) | (got["negative_cover"] < MIN_COVER)
    excess[thin] = np.nan
    log(f"\n  cells: {excess.size}, masked for thin coverage: {int(thin.sum())} "
        f"({100 * thin.mean():.0f} %)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, excess=excess, ml=ML, ap=AP, si=SI, plane=args.plane,
             positive_cover=got["positive_cover"], negative_cover=got["negative_cover"])

    finite = excess[np.isfinite(excess)]
    if not finite.size:
        log("  nothing survived the coverage rule")
        return 1
    log(f"  excess: median {np.median(finite):+.4f}, p99 {np.percentile(finite, 99):+.4f}, "
        f"max {finite.max():+.4f}")
    k, a, b = np.unravel_index(np.nanargmax(np.where(np.isfinite(excess), excess, -np.inf)),
                               excess.shape)
    log(f"  peak at SI {SI[k]:+.0f} mm, ML {ML[a]:+.0f} mm, AP {AP[b]:+.0f} mm")

    # Flat or concentrated: the share of the positive excess inside the best 10 % of
    # cells. Near 0.10 means uniform, near 1.0 means a point source.
    positive = np.clip(np.where(np.isfinite(excess), excess, 0.0), 0, None)
    order = np.sort(positive.ravel())[::-1]
    top = order[:max(1, order.size // 10)].sum() / max(order.sum(), 1e-9)
    log(f"  concentration: the top 10 % of cells hold {100 * top:.0f} % of the excess")
    log(f"\n{args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
