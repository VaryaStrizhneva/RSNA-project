"""Gather the folds of a sweep into one package, ready to be a Kaggle dataset.

Training writes one package per fold, because each fold is a separate run that can
fail on its own. Inference wants the opposite: one directory holding every member, so
that `predict.py` averages them and the notebook mounts a single dataset.

    python -m scripts.bundle out/sweep-window-20 --out out/submit-window

Members must agree about the config, because one pixel cache is decoded for all of
them. Two experiments cannot be bundled together; that is a refusal, not a warning,
since the alternative is a submission produced from pixels half the members never saw.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.rsna.config import Config
from src.rsna.model import Member, read_manifest, write_package


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path,
                        help="A sweep directory holding one package per fold.")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--note", default="")
    args = parser.parse_args()

    sources = sorted(p for p in args.run.iterdir()
                     if p.is_dir() and (p / "manifest.json").is_file())
    if not sources:
        raise SystemExit(f"no package under {args.run}")

    members, configs, notes = [], set(), []
    for source in sources:
        manifest = read_manifest(source)
        notes.append(manifest.get("note", ""))
        for entry in manifest["members"]:
            payload = torch.load(source / entry["file"], map_location="cpu",
                                 weights_only=False)
            configs.add(json.dumps(payload["config"], sort_keys=True))
            members.append(Member(
                name=entry["id"],
                config=Config.from_dict(payload["config"]),
                state_dict=payload["state_dict"],
                fingerprint=np.asarray(payload["fingerprint"], np.float32),
                fold=entry.get("fold"),
                holdout_auc=entry.get("holdout"),
                epoch=entry.get("epoch"),
            ))
            score = entry.get("holdout")
            print(f"  {entry['id']}  fold {entry.get('fold')}  "
                  f"holdout {score:.4f}" if score is not None else f"  {entry['id']}")

    if len(configs) > 1:
        raise SystemExit(
            f"{len(configs)} different configs among {len(members)} members. One cache "
            f"is decoded for the whole package, so they cannot travel together.")

    scores = [m.holdout_auc for m in members if m.holdout_auc is not None]
    note = args.note or (f"{len(members)} folds of {notes[0]}" if notes else "")
    path = write_package(args.out, members, note=note)
    print(f"\n{len(members)} member(s) -> {path}")
    if scores:
        print(f"holdout mean {np.mean(scores):.4f} +/- {np.std(scores):.4f}")
    print(f"{sum(f.stat().st_size for f in path.iterdir()) / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
