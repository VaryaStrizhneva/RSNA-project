"""Decode every (study, slot) once, and keep the bytes.

Fine-tuning revisits the same pixels every epoch. Reading them from the mount each
time would make the epoch count a function of I/O rather than of learning, so they are
decoded once and held as bytes: intensity is already normalised into [0, 1] by
`read_slot`, and eight bits cost nothing a bilinear resize has not already cost.

**One difference from the notebook this is ported from.** There, the cache lives in
RAM for the duration of a single Kaggle session, so its size is a function of how much
memory the machine will lend. We do not train inside one session, so a cache can be
written to disk once and reused for days across many runs — which is the single largest
gain in iteration speed available here. `build_cache(path=...)` does that; without a
path it behaves as the original does.

A cache on disk is only useful if a stale one cannot be mistaken for a fresh one, so
it is written with a sidecar recording the config and the study order, and
`load_cache` refuses anything that does not match.
"""

from __future__ import annotations

import gc
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import numpy as np

from ..config import Config
from .laterality import normalise_laterality
from .ordering import order_slices
from .pixels import read_slot

#: Share of free memory an in-memory cache may take. The rest is not slack: the
#: encoder, its activations and the batches come out of the same pool, and the cache is
#: the one allocation big enough that overshooting it kills the run outright.
CACHE_FRACTION = 0.45
#: Hard ceiling regardless of what the machine reports.
CACHE_BUDGET_MAX_GB = 24.0
#: Fallback for a machine that will not say.
CACHE_BUDGET_GB = 12.0
#: Floor on the test corpus relative to the training one. The visible test split is a
#: stub and the scored one is not, so sizing against training alone passes every run
#: that can be watched and overruns the one that counts.
TEST_SHARE = 0.30

ORDER_THREADS = 32
PIXEL_THREADS = 12

_Log = Callable[[str], None]


def _noop(_: str) -> None:
    pass


def available_gb() -> float:
    """Memory this machine will actually lend, read rather than assumed.

    A hardcoded ceiling is a guess about a machine the author is not sitting at: too
    low costs coverage silently, too high ends the run. The machine will say, so it is
    asked.
    """

    try:
        with open("/proc/meminfo") as fh:
            info = {k.strip(): v for k, v in (l.split(":", 1) for l in fh if ":" in l)}
        return int(info["MemAvailable"].split()[0]) / 1024 ** 2
    except Exception:
        return CACHE_BUDGET_GB / CACHE_FRACTION


def plan_cache(n_study: int, config: Config, n_test: int = 0, n_group_max: int = 1,
               log: _Log = _noop) -> int:
    """How many groups of slices per slot the available memory allows.

    The cache is `n_study x n_slot x slices x img^2` bytes. Coverage is the cheap axis
    — linear — and resolution the expensive one, so when the budget binds it is the
    slice count that gives way, never the pixel grid. Decided once, from the training
    corpus, so train and test caches share a group layout.
    """

    available = available_gb()
    budget = min(available * CACHE_FRACTION, CACHE_BUDGET_MAX_GB)
    total = n_study + max(n_test, int(TEST_SHARE * n_study))
    per_slice = total * config.n_slot * config.img * config.img
    afford = int(budget * 1024 ** 3 // max(per_slice, 1))
    groups = max(1, min(n_group_max, afford // config.group))

    log(f"memory: {available:.1f} GB available, {budget:.1f} GB to the cache; "
        f"sizing for {n_study} train + {total - n_study} test studies -> "
        f"{groups} group(s) of {config.group} = {groups * config.group} slices per slot"
        + (f" (wanted {n_group_max})" if groups < n_group_max else ""))
    return groups


def order_pass(jobs: list, config: Config, order_cache: str | Path | None = None,
               log: _Log = _noop, budget_s: float = 5400.0) -> int:
    """Resolve the slice order of every chosen series, in its own pass.

    It reads one header per slice — far more file opens than the decode that follows —
    and on a network mount that is latency, not work, so it gets its own wider pool.

    A remembered order is validated by the file count, so a tree that changed under it
    is recomputed rather than trusted: order is derived data, and a stale entry would
    be invisible in exactly the way that matters most.
    """

    start = time.time()
    seen: dict = {}
    resolved = 0

    if order_cache and Path(order_cache).is_file():
        try:
            seen = json.loads(Path(order_cache).read_text())
        except (OSError, ValueError):
            seen = {}
        hits = 0
        for record in jobs:
            entry = seen.get(record["SeriesInstanceUID"])
            if entry and len(entry["files"]) == len(record["files"]):
                record["ordered"] = entry["files"]
                resolved += int(entry["good"])
                hits += 1
        jobs = [r for r in jobs if "ordered" not in r]
        log(f"{hits} slot-series ordered from {order_cache}, {len(jobs)} to read")

    done = 0
    with ThreadPoolExecutor(max_workers=ORDER_THREADS) as pool:
        for block_start in range(0, len(jobs), 1024):
            block = jobs[block_start:block_start + 1024]
            results = pool.map(
                lambda r: order_slices(r["dir"], r["files"], config), block)
            for record, (files, good) in zip(block, results):
                record["ordered"] = files
                resolved += int(good)
                done += 1
                if order_cache:
                    seen[record["SeriesInstanceUID"]] = {"files": files, "good": bool(good)}
            if time.time() - start > budget_s:
                log(f"ordering budget spent at {done}/{len(jobs)}; the rest keep file order")
                break

    if order_cache and done:
        tmp = Path(order_cache).with_suffix(".tmp")
        tmp.write_text(json.dumps(seen))
        tmp.replace(Path(order_cache))

    return resolved


def _sidecar(path: Path) -> Path:
    return path.with_suffix(".json")


def build_cache(slot_map: dict, plane_map: dict, lat_map: dict, config: Config,
                cache_slices: int, path: str | Path | None = None,
                order_cache: str | Path | None = None, log: _Log = _noop,
                decode_failures: list | None = None) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Decode every (study, slot) into `[study, slot, slice, img, img]` bytes.

    With `path`, the array is a memmap on disk and survives the process; a sidecar
    records the config and study order so `load_cache` can refuse a stale one.
    """

    studies = sorted(slot_map)
    index = {s: i for i, s in enumerate(studies)}
    shape = (len(studies), config.n_slot, cache_slices, config.img, config.img)

    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        cache = np.lib.format.open_memmap(path, mode="w+", dtype=np.uint8, shape=shape)
    else:
        cache = np.zeros(shape, np.uint8)
    mask = np.zeros((len(studies), config.n_slot), np.float32)
    log(f"cache {shape} = {np.prod(shape) / 1024 ** 3:.1f} GB"
        + (f" -> {path}" if path is not None else " in memory"))

    jobs = [(study, k, plane, slot_map[study][name])
            for study in studies
            for k, (name, plane, _, _) in enumerate(config.slots)
            if name in slot_map[study]]

    records = [job[3] for job in jobs]
    n_headers = sum(len(r["files"]) for r in records)
    log(f"ordering {len(records)} slot-series ({n_headers} slice headers)")
    resolved = order_pass(records, config, order_cache, log)
    log(f"ordered {resolved}/{len(records)} by geometry "
        f"({len(records) - resolved} kept arbitrary)")

    log(f"decoding {len(jobs)} slot-series")
    done = 0
    with ThreadPoolExecutor(max_workers=PIXEL_THREADS) as pool:
        for block_start in range(0, len(jobs), 512):
            block = jobs[block_start:block_start + 512]
            images = pool.map(
                lambda j: read_slot(j[3], config, cache_slices, config.img,
                                    decode_failures), block)
            for (study, k, plane, _), image in zip(block, images):
                done += 1
                if image is None:
                    continue
                cache[index[study], k] = normalise_laterality(
                    image.numpy(), plane, lat_map.get(study))
                mask[index[study], k] = 1.0

    log(f"{int(mask.sum())}/{len(jobs)} slots filled")

    if path is not None:
        cache.flush()
        _sidecar(path).write_text(json.dumps({
            "config": config.to_dict(),
            "cache_slices": cache_slices,
            "cache_tag": config.cache_tag(),
            "studies": studies,
            "mask": mask.tolist(),
        }))
    gc.collect()
    return studies, cache, mask


class CacheMismatch(RuntimeError):
    """Raised when a cache on disk was not decoded under the config being asked for."""


def load_cache(path: str | Path, config: Config,
               cache_slices: int) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Reopen a cache, refusing one decoded under a different reading.

    The check is on the tag rather than the shape: two configurations that disagree on
    how a slice is chosen produce different arrays of identical shape, and that is
    precisely the mismatch nothing downstream can detect.
    """

    path = Path(path)
    meta = json.loads(_sidecar(path).read_text())

    if meta["cache_tag"] != config.cache_tag():
        raise CacheMismatch(
            f"{path} was decoded under {meta['cache_tag']!r}, this run asks for "
            f"{config.cache_tag()!r}: the arrays would have the same shape and "
            f"different pixels")
    if meta["cache_slices"] != cache_slices:
        raise CacheMismatch(
            f"{path} holds {meta['cache_slices']} slices per slot, not {cache_slices}")
    if meta["config"]["slots"] != [list(s) for s in config.slots]:
        raise CacheMismatch(f"{path} was decoded for different slots")

    cache = np.load(path, mmap_mode="r")
    mask = np.asarray(meta["mask"], np.float32)
    return meta["studies"], cache, mask
