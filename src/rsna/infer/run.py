"""The whole submission, from a directory of DICOMs to `submission.csv`.

This lives in the library rather than in `scripts/predict.py` for one reason: the
scored Kaggle notebook runs it too. A notebook that reimplemented these twenty lines
would diverge from the command line the first time either changed, and the divergence
would show up as a leaderboard score nobody could reproduce locally. Both callers now
enter here, so there is one implementation and one thing to test.

Nothing is decided here. Resolution, slice count, slots, encoder and stem all come from
the package's own manifest, so the same call runs any model that was ever fitted.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Config
from ..dicom import annotate, build_cache, laterality_of, pick_slots, walk
from ..model import load_member, read_manifest
from .predict import blend_members, predict_member
from .submission import benchmark_submission, write_submission


def run_submission(package, data_root, split: str = "test_series",
                   encoder=None, out="submission.csv", cache=None,
                   max_windows: int | None = None, device: str = "cpu",
                   log=print) -> Path:
    """Read a package over a split and write a submission. Returns the path written."""

    package, data_root, out = Path(package), Path(data_root), Path(out)
    manifest = read_manifest(package)
    entries = manifest["members"]
    log(f"package: {len(entries)} member(s), written {manifest.get('written')}")

    # Every member must agree about the pixels, because one cache is decoded for all.
    # Compared as serialised text: a config holds lists, which are not hashable.
    import json
    configs = {json.dumps(e["config"], sort_keys=True) for e in entries}
    if len(configs) > 1:
        raise ValueError("members disagree about the config; decode one cache per group")
    config = Config.from_dict(entries[0]["config"])
    log(f"config: {config.img}px, {config.slices} slices, group {config.group}, "
        f"stem {config.stem}")

    headers = annotate(walk(data_root, split))
    if headers.empty:
        raise FileNotFoundError(f"no series under {data_root / split}")
    csv = data_root / f"{split.replace('_series', '')}_series.csv"
    plane_map = dict(zip(*pd.read_csv(csv)[["SeriesInstanceUID",
                                            "Anatomical_Plane"]].values.T))
    headers["plane"] = headers["SeriesInstanceUID"].map(plane_map)

    sides, _ = laterality_of(headers, config)
    slots = pick_slots(headers, plane_map, config)
    studies = sorted(slots)

    # The fallback goes down before anything expensive runs: a crash after the decode
    # pass has spent the costly half and should still score 0.500 rather than nothing.
    benchmark_submission(studies, out)
    log(f"benchmark submission written for {len(studies)} studies")

    _, pixels, mask = build_cache(slots, plane_map, sides, config,
                                  cache_slices=config.slices, path=cache, log=log)

    starts = config.windows(overlap=True, limit=max_windows)
    log(f"{len(starts)} window(s) per member: {starts}")

    predictions = []
    for entry in entries:
        model, member_config, distance = load_member(package, entry, device,
                                                     source=encoder)
        log(f"{entry['id']}: fingerprint matches within {distance:.2g}")
        predictions.append(
            predict_member(model, pixels, mask, member_config, device, starts))
        del model

    blended = blend_members(predictions) if len(predictions) > 1 else predictions[0]
    write_submission(blended, studies, out)

    values = pd.read_csv(out).iloc[:, 1:].to_numpy()
    log(f"submission: {values.shape}, {np.unique(values).size} distinct values")
    if np.unique(values).size <= 1:
        log("WARNING: every prediction is identical — this would score 0.500")
    return out
