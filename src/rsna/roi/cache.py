"""Cut every study's ROI once, and keep the bytes.

4407 studies at two series of five slots of 126 x 224 is **1.2 GB** — two orders of
magnitude under the classifier's 36 GB cache, because that is what cropping buys. It
fits in memory, so a training step costs no DICOM.

The tag names everything that decides a voxel. A cache written under one `RoiSpec` is
never read under another: the crop is small enough that a two-millimetre change in the
box moves the anatomy across a tenth of it, and nothing downstream would notice.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Config
from ..dicom.ordering import order_slices
from ..dicom.pixels import _read_plane
from .config import RoiSpec
from .extract import extract


def cache_tag(spec: RoiSpec) -> str:
    return (f"{spec.name}_{int(spec.box_w_mm)}x{int(spec.box_h_mm)}mm_"
            f"{spec.out_w}x{spec.out_h}px_{spec.lateral_mm:g}-{spec.medial_mm:g}_"
            f"{spec.slots}sl")


def _geometry(directory: Path, name: str) -> dict | None:
    import pydicom

    try:
        ds = pydicom.dcmread(str(directory / name), force=True, stop_before_pixels=True)
        return {"sop": str(ds.SOPInstanceUID),
                "ipp": [float(v) for v in ds.ImagePositionPatient],
                "iop": [float(v) for v in ds.ImageOrientationPatient],
                "ps": [float(v) for v in ds.PixelSpacing]}
    except Exception:  # noqa: BLE001
        return None


def read_series(directory: Path, config: Config) -> tuple[np.ndarray, list]:
    """A whole series as uint8 on one window, with its per-slice geometry.

    Windowed over the series rather than per slice: a crop whose brightness jumps
    between neighbouring slices teaches the encoder a gradient that is not anatomy.
    """

    names, _ = order_slices(str(directory), sorted(p.name for p in directory.glob("*.dcm")),
                            config)
    planes = [_read_plane(str(directory / n)) for n in names]
    keep = [i for i, p in enumerate(planes) if p is not None]
    if not keep:
        raise ValueError(f"{directory} decoded no slice")
    shape = planes[keep[0]].shape
    keep = [i for i in keep if planes[i].shape == shape]

    stack = np.stack([planes[i] for i in keep])
    lo, hi = np.percentile(stack, (1.0, 99.5))
    volume = np.clip((stack - lo) / max(hi - lo, 1e-6), 0, 1).__mul__(255).astype(np.uint8)
    return volume, [_geometry(directory, names[i]) for i in keep]


def build(landmarks: str | Path, series: pd.DataFrame, out: str | Path, spec: RoiSpec,
          config: Config | None = None, workers: int = 8, log=print) -> Path:
    """Crop every study named in the landmark table, for every series the spec wants.

    `series` is the annotated header table, which must carry `plane`, `weight`,
    `fatsat`, `dir` and `StudyInstanceUID`.
    """

    config = config or Config()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    points = pd.read_csv(landmarks)
    have = sorted(points["point"].dropna().unique())
    # A region defined between two landmarks needs both, so a study carrying only one is
    # not a study this spec can cut.
    second = None
    if spec.landmark2:
        second = points[points["point"] == spec.landmark2].set_index("study")
    points = points[points["point"] == spec.landmark].set_index("study")
    if second is not None:
        points = points[points.index.isin(second.index)]
    studies = list(points.index)
    if not studies:
        # It used to write the empty cache and report `nan %` coverage, which the next
        # stage would then happily train on. A spec asking for a point the table does
        # not carry is the likely cause and is invisible otherwise: both names are
        # valid, they just come from different annotation passes.
        wanted = spec.landmark + (f" and {spec.landmark2}" if spec.landmark2 else "")
        raise ValueError(
            f"{Path(landmarks).name} holds no study with {wanted}, which "
            f"{spec.name} hangs off — it carries {have}. Point the spec at the right "
            f"landmark, or the build at the right table.")
    by_study = {s: g for s, g in series.groupby("StudyInstanceUID")}
    log(f"{len(studies)} studies, {len(spec.series)} series slots -> {out}")

    volumes = np.lib.format.open_memmap(
        out / f"{cache_tag(spec)}.npy", mode="w+", dtype=np.uint8,
        shape=(len(studies), len(spec.series), spec.slots, spec.out_h, spec.out_w))
    mask = np.zeros((len(studies), len(spec.series), spec.slots), bool)
    records: list = []

    def one(index_study):
        i, study = index_study
        point = points.loc[study, ["x_mm", "y_mm", "z_mm"]].to_numpy(float)
        group = by_study.get(study)
        point2 = None
        if second is not None:
            r2 = second.loc[study]
            point2 = (float(r2.x_mm), float(r2.y_mm), float(r2.z_mm))
        found = []
        for k, (plane, weight, fatsat) in enumerate(spec.series):
            if group is None:
                continue
            hit = group[(group["plane"] == plane) & (group["weight"] == weight)
                        & (group["fatsat"].astype(bool) == fatsat)]
            if not len(hit):
                continue
            row = hit.sort_values("n_slices", ascending=False).iloc[0]
            try:
                volume, geom = read_series(Path(row["dir"]), config)
                stack = extract(volume, geom, point, spec, point2)
            except Exception as exc:  # noqa: BLE001
                found.append((k, None, f"{type(exc).__name__}: {exc}"))
                continue
            found.append((k, stack, str(row["SeriesInstanceUID"])))
        return i, study, found

    failed = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for n, (i, study, found) in enumerate(pool.map(one, enumerate(studies)), 1):
            rec = {"study": study, "series": [None] * len(spec.series),
                   "toward_bowtie": [None] * len(spec.series),
                   "spacing_mm": [None] * len(spec.series)}
            for k, stack, note in found:
                if stack is None:
                    failed += 1
                    continue
                volumes[i, k] = stack.volume
                mask[i, k] = stack.valid
                rec["series"][k] = note
                rec["toward_bowtie"][k] = stack.toward_bowtie
                rec["spacing_mm"][k] = stack.native_spacing_mm * stack.stride
            records.append(rec)
            if n % 250 == 0:
                log(f"  {n}/{len(studies)}")

    volumes.flush()
    np.save(out / f"{cache_tag(spec)}_mask.npy", mask)
    (out / f"{cache_tag(spec)}.json").write_text(json.dumps(
        {"spec": spec.to_dict(), "records": records}))
    filled = mask.any(axis=2).mean(axis=0)
    log(f"{len(records)} studies, {failed} series failed")
    for (plane, weight, fatsat), share in zip(spec.series, filled):
        log(f"  {plane} {weight}{' FS' if fatsat else ''}: {100 * share:.1f} % of studies")
    log(f"  at least one: {100 * mask.any(axis=(1, 2)).mean():.1f} %")
    return out


class OneChannel:
    """A cache read as **one series per study**, chosen by a priority list.

    The caches carry two or three series on their own axis, and most studies fill only
    the first: measured over the seven shipped boxes, the leading series is present for
    83-84% of studies and the others for 13-37%. Feeding the rest as extra channels was
    tested once and cost: dropping the coronal non-fat-suppressed series from the MCL
    expert gained **+0.0118**. Using it only where nothing better exists is a different
    thing entirely, and it takes coverage from 84% to 99.8%.

    So this collapses the series axis to one, per study, by taking the first entry of
    `order` whose slot mask is not empty. Lazily: `fit` indexes a batch at a time, the
    underlying array is a memmap of up to 19.3 GB, and materialising the gather would
    copy 22 GB across nine groups for no gain.
    """

    def __init__(self, volumes, mask: np.ndarray, order):
        n, series = mask.shape[:2]
        self._v = volumes
        self.order = tuple(int(i) for i in order)
        if sorted(self.order) != list(range(series)):
            raise ValueError(
                f"priority {self.order} is not a permutation of the {series} series this "
                f"cache carries; a missing index would make that series unreachable")
        filled = mask.reshape(n, series, -1).any(axis=2)
        # -1 where no series has a slot, which `has_pixels` and the slot mask both read
        # as a study with nothing; the row is still returned, as zeros.
        chosen = np.full(n, -1, np.int64)
        for index in reversed(self.order):
            chosen = np.where(filled[:, index], index, chosen)
        self.chosen = chosen
        self.shape = (n, 1) + tuple(mask.shape[2:]) + tuple(volumes.shape[3:])
        self.dtype = volumes.dtype
        self.nbytes = int(np.prod(self.shape)) * volumes.dtype.itemsize

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, idx):
        rows = np.atleast_1d(np.asarray(idx))
        pick = self.chosen[rows]
        out = np.zeros((len(rows),) + self.shape[1:], self.dtype)
        for k, (row, series) in enumerate(zip(rows, pick)):
            if series >= 0:
                out[k, 0] = self._v[row, series]
        return out


def collapse(volumes, mask: np.ndarray, order=None):
    """`(volumes, mask)` read as one series per study. Returns them and the choice.

    `order` is best-first indices into the cache's series axis; `None` keeps the order
    the spec declared. It is a **read-time** decision and deliberately not a `RoiSpec`
    field: the cache cuts every series it was asked for, and which one a model reads is
    the model's business. Putting it in the spec would also make `cache.load` refuse the
    cache, since it compares the whole spec for equality.
    """

    series = mask.shape[1]
    view = OneChannel(volumes, mask, range(series) if order is None else order)
    rows = np.arange(len(view))
    one = np.zeros((len(view), 1) + mask.shape[2:], bool)
    got = view.chosen >= 0
    one[got, 0] = mask[rows[got], view.chosen[got]]
    return view, one, view.chosen


def load(out: str | Path, spec: RoiSpec):
    """The crops, their slot masks and their records, refusing a different spec."""

    out = Path(out)
    meta = json.loads((out / f"{cache_tag(spec)}.json").read_text())
    if RoiSpec.from_dict(meta["spec"]) != spec:
        raise ValueError("this cache was cut under a different RoiSpec; rebuild it "
                         "rather than read it under the wrong one")
    return (np.load(out / f"{cache_tag(spec)}.npy", mmap_mode="r"),
            np.load(out / f"{cache_tag(spec)}_mask.npy"), meta["records"])
