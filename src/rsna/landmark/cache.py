"""Sample every annotated study once, and keep the bytes.

304 studies at 48 x 256 x 256 uint8 is under a gigabyte, so the whole training set fits
in memory and a step costs no DICOM. The geometry travels beside the pixels in a JSON
sidecar, because a volume without the millimetres of its slices is not a landmark
example — it is a stack of pictures with no way back to the patient.

Rebuilt rather than patched when the sampling changes: the tag names the settings that
decide what a voxel is, so a cache written under one `LandmarkConfig` is never silently
read under another.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Config
from ..dicom.ordering import order_slices
from .config import LandmarkConfig
from .sample import Sampled, sample_series


def cache_tag(config: LandmarkConfig) -> str:
    """The name a sampled set is stored under."""

    return (f"{config.img}px_{int(config.fov_mm)}mm_{config.slices}sl_"
            f"{config.decimate_to_mm:.1f}dec")


def _record(s: Sampled, study: str, series: str, points: dict) -> dict:
    return {"study": study, "series": series,
            "valid": s.valid.tolist(), "t_mm": s.t_mm.tolist(),
            "ipp": s.ipp.tolist(), "iop": np.asarray(s.iop).tolist(),
            "normal": np.asarray(s.normal).tolist(),
            "spacing": np.asarray(s.spacing).tolist(), "transform": s.transform,
            "native_spacing_mm": s.native_spacing_mm, "native_n": s.native_n,
            "stride": s.stride, "sop": s.sop,
            "points": {k: list(map(float, v)) for k, v in points.items()}}


def restore(record: dict, volume: np.ndarray) -> Sampled:
    """A record and its slab of pixels, back into the shape the model reads."""

    return Sampled(volume=volume,
                   valid=np.asarray(record["valid"], bool),
                   t_mm=np.asarray(record["t_mm"], float),
                   ipp=np.asarray(record["ipp"], float),
                   iop=np.asarray(record["iop"], float),
                   normal=np.asarray(record["normal"], float),
                   transform=record["transform"],
                   spacing=np.asarray(record["spacing"], float),
                   native_spacing_mm=record["native_spacing_mm"],
                   native_n=record["native_n"], stride=record["stride"],
                   sop=record["sop"])


def build(annotations: str | Path, dicom_root: str | Path, out: str | Path,
          config: LandmarkConfig, log=print) -> Path:
    """Sample every annotated study into one array, with its geometry beside it."""

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    base = Config()

    table = pd.read_csv(annotations)
    table = table[table["point"].notna()]
    grouped = {s: g for s, g in table.groupby("study")}
    log(f"{len(grouped)} annotated studies -> {out}")

    volumes = np.lib.format.open_memmap(
        out / f"volumes_{cache_tag(config)}.npy", mode="w+", dtype=np.uint8,
        shape=(len(grouped), config.slices, config.img, config.img))

    records, failed = [], []
    for i, (study, group) in enumerate(grouped.items()):
        series = str(group["series"].iloc[0])
        folder = Path(dicom_root) / study / series
        try:
            names, ordered = order_slices(
                str(folder), sorted(p.name for p in folder.glob("*.dcm")), base)
            s = sample_series(folder, names, config)
        except Exception as exc:  # noqa: BLE001
            failed.append((study, f"{type(exc).__name__}: {exc}"))
            continue
        points = {r.point: (r.x_mm, r.y_mm, r.z_mm) for r in group.itertuples()}
        # Every configured point, or the study does not go in. `encode` leaves a point
        # it was not given at zero, and a plain mean-squared error reads that as "the
        # landmark is nowhere in this knee" rather than "nobody said" — so a two-point
        # model would learn that these studies have no second meniscus. Dropping them is
        # the cheap half of that trade: the two meniscus passes overlap on 302 of 320
        # studies, so it costs 18.
        absent = [n for n in config.points if points.get(n) is None]
        if absent:
            failed.append((study, f"no {', '.join(absent)}"))
            continue
        volumes[len(records)] = s.volume
        rec = _record(s, study, series, points)
        rec["ordered"] = bool(ordered)
        records.append(rec)
        if (i + 1) % 50 == 0:
            log(f"  {i + 1}/{len(grouped)}")

    volumes.flush()
    if len(records) < len(grouped):
        # Shorten rather than leave zeroed rows that would train as black studies.
        trimmed = np.lib.format.open_memmap(
            out / f"volumes_{cache_tag(config)}.npy", mode="r+", dtype=np.uint8,
            shape=(len(grouped), config.slices, config.img, config.img))[:len(records)]
        np.save(out / f"volumes_{cache_tag(config)}.npy", np.asarray(trimmed))

    (out / f"geometry_{cache_tag(config)}.json").write_text(json.dumps(
        {"config": config.to_dict(), "records": records}))
    log(f"{len(records)} sampled, {len(failed)} failed or incomplete")
    for study, why in failed:
        log(f"  FAILED {study[-11:]}: {why}")
    return out


def load(out: str | Path, config: LandmarkConfig) -> tuple[np.ndarray, list[dict]]:
    """The sampled volumes and their records, refusing a cache built differently."""

    out = Path(out)
    meta = json.loads((out / f"geometry_{cache_tag(config)}.json").read_text())
    stored = LandmarkConfig.from_dict(meta["config"])
    if stored != config:
        raise ValueError("this cache was built under a different LandmarkConfig; "
                         "rebuild rather than read it under the wrong one")
    volumes = np.load(out / f"volumes_{cache_tag(config)}.npy", mmap_mode="r")
    return volumes, meta["records"]
