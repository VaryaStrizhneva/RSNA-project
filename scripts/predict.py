"""Read a weights package over a decoded cache and write a submission.

This is what the scored Kaggle notebook will call. It does no training and holds no
model definition of its own — everything comes from `src/rsna`, and the config comes
from the package rather than from here, so a member is always read the way it was
fitted.

    python -m scripts.predict --package out/package --split test_series
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.rsna.config import Config
from src.rsna.dicom import annotate, build_cache, laterality_of, pick_slots, walk
from src.rsna.infer import (benchmark_submission, blend_members, predict_member,
                            write_submission)
from src.rsna.model import load_member, read_manifest

T0 = time.time()


def log(message: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {message}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--data-root", default="data/raw", type=Path)
    parser.add_argument("--split", default="test_series")
    parser.add_argument("--encoder", default="models/dinov2-small")
    parser.add_argument("--out", default="submission.csv", type=Path)
    parser.add_argument("--cache", default=None, type=Path)
    parser.add_argument("--max-windows", type=int, default=None,
                        help="Read each member over at most this many windows.")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    manifest = read_manifest(args.package)
    entries = manifest["members"]
    log(f"package: {len(entries)} member(s), written {manifest.get('written')}")

    # Every member must agree about the pixels, because one cache is decoded for all.
    # Compared as serialised text: a config holds lists, which are not hashable.
    configs = {json.dumps(e["config"], sort_keys=True) for e in entries}
    if len(configs) > 1:
        raise SystemExit("members disagree about the config; decode one cache per group")
    config = Config.from_dict(entries[0]["config"])
    log(f"config: {config.img}px, {config.slices} slices, group {config.group}")

    headers = annotate(walk(args.data_root, args.split))
    if headers.empty:
        raise SystemExit(f"no series under {args.data_root / args.split}")
    csv = args.data_root / f"{args.split.replace('_series', '')}_series.csv"
    plane_map = dict(zip(*pd.read_csv(csv)[["SeriesInstanceUID", "Anatomical_Plane"]].values.T))
    headers["plane"] = headers["SeriesInstanceUID"].map(plane_map)

    sides, _ = laterality_of(headers, config)
    slots = pick_slots(headers, plane_map, config)
    studies = sorted(slots)

    # The fallback goes down before anything expensive runs: a crash after the decode
    # pass has spent the costly half and should still score 0.500 rather than nothing.
    benchmark_submission(studies, args.out)
    log(f"benchmark submission written for {len(studies)} studies")

    _, cache, mask = build_cache(slots, plane_map, sides, config,
                                 cache_slices=config.slices, path=args.cache, log=log)

    starts = config.windows(overlap=True, limit=args.max_windows)
    log(f"{len(starts)} window(s) per member: {starts}")

    predictions = []
    for entry in entries:
        model, member_config, distance = load_member(args.package, entry, device,
                                                     source=args.encoder)
        log(f"{entry['id']}: fingerprint matches within {distance:.2g}")
        predictions.append(predict_member(model, cache, mask, member_config, device, starts))
        del model

    blended = blend_members(predictions) if len(predictions) > 1 else predictions[0]
    write_submission(blended, studies, args.out)
    values = pd.read_csv(args.out).iloc[:, 1:].to_numpy()
    log(f"submission: {values.shape}, {np.unique(values).size} distinct values")
    if np.unique(values).size <= 1:
        log("WARNING: every prediction is identical — this would score 0.500")


if __name__ == "__main__":
    main()
