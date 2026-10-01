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
    points = points[points["point"] == spec.landmark].set_index("study")
    studies = list(points.index)
    if not studies:
        # It used to write the empty cache and report `nan %` coverage, which the next
        # stage would then happily train on. A spec asking for a point the table does
        # not carry is the likely cause and is invisible otherwise: both names are
        # valid, they just come from different annotation passes.
        raise ValueError(
            f"{Path(landmarks).name} holds no rows for {spec.landmark!r}, which "
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
                stack = extract(volume, geom, point, spec)
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


def load(out: str | Path, spec: RoiSpec):
    """The crops, their slot masks and their records, refusing a different spec."""

    out = Path(out)
    meta = json.loads((out / f"{cache_tag(spec)}.json").read_text())
    if RoiSpec.from_dict(meta["spec"]) != spec:
        raise ValueError("this cache was cut under a different RoiSpec; rebuild it "
                         "rather than read it under the wrong one")
    return (np.load(out / f"{cache_tag(spec)}.npy", mmap_mode="r"),
            np.load(out / f"{cache_tag(spec)}_mask.npy"), meta["records"])
