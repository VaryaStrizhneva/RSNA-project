"""Measure a finished run the way inference will measure it.

Training selects an epoch on a cheap proxy — disjoint windows, four encoder passes —
and, since the run that introduced this, records a second score over the windows
`rsna.infer` actually slides. This script produces that second score for runs written
before it existed, so that an old package can be compared with a new one.

It predicts on the holdout and the expert studies **only**: the record already names
them and carries their targets, and every other study in the split was trained on.

    python -m scripts.rescore --package out/ref/pkg-f0 --cache /path/cache.npy

`--write` folds the result back into `history.json` and the prediction CSVs, so the
run ends up carrying exactly the fields a run trained today would carry.

Unlike `scripts.evaluate_train`, this never builds a cache: it opens one read-only
through `load_cache`, which refuses a cache decoded under a different configuration
rather than silently overwriting it.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from src.rsna.dicom import CacheMismatch, load_cache
from src.rsna.eval import read_run_record
from src.rsna.eval.record import GOLD, HISTORY, HOLDOUT, _predictions
from src.rsna.model import load_member, read_manifest
from src.rsna.train import macro_auc, predict

T0 = time.time()


def log(message: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {message}", flush=True)


def rows_for(uids, position, what):
    """Cache rows for `uids`, refusing to guess when one is missing."""

    missing = [u for u in uids if u not in position]
    if missing:
        raise SystemExit(
            f"{len(missing)} {what} study/studies are not in the cache "
            f"(first: {missing[0]}). The cache and the record describe different runs.")
    return np.array([position[u] for u in uids], dtype=int)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--encoder", default="models/dinov2-small")
    parser.add_argument("--device", default=None)
    parser.add_argument("--write", action="store_true",
                        help="Fold the result back into the run record.")
    args = parser.parse_args()

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    record = read_run_record(args.package)
    log(f"run: {record.experiment} fold {record.fold}, best epoch {record.best_epoch + 1}, "
        f"{record.n} holdout + {record.n_gold} expert studies")
    if record.inference_windows is not None:
        log(f"already scored over {record.inference_windows} window(s); rescoring anyway")

    manifest = read_manifest(args.package)
    entries = manifest["members"]
    if len(entries) != 1:
        raise SystemExit(f"{args.package} holds {len(entries)} members; this reads one "
                         f"fold's own holdout, so it wants exactly one")

    model, config, distance = load_member(args.package, entries[0], device=device,
                                          source=args.encoder)
    log(f"{entries[0]['id']}: fingerprint matches within {distance:.3g}")

    try:
        studies, cache, mask = load_cache(args.cache, config, config.slices)
    except CacheMismatch as exc:
        raise SystemExit(f"cache refused: {exc}")
    position = {uid: i for i, uid in enumerate(studies)}
    log(f"cache: {cache.shape}, opened read-only")

    n_select, n_final = len(config.windows(overlap=False)), len(config.windows(overlap=True))
    log(f"selection read {n_select} disjoint window(s); scoring over {n_final} sliding")

    idx = rows_for(record.uids, position, "holdout")
    holdout_p = predict(model, cache, mask, idx, config, device, overlap=True)
    holdout_auc = macro_auc((record.y > 0.5).astype(int), holdout_p)
    log(f"holdout: {record.best_holdout_auc:.4f} at {n_select} windows "
        f"-> {holdout_auc:.4f} at {n_final}  ({holdout_auc - record.best_holdout_auc:+.4f})")

    gold_p = gold_auc = None
    if record.n_gold:
        gidx = rows_for(record.gold_uids, position, "expert")
        gold_p = predict(model, cache, mask, gidx, config, device, overlap=True)
        gold_auc = macro_auc((record.gold_y > 0.5).astype(int), gold_p)
        was = macro_auc((record.gold_y > 0.5).astype(int), record.gold_p)
        log(f"gold:    {was:.4f} at {n_select} windows "
            f"-> {gold_auc:.4f} at {n_final}  ({gold_auc - was:+.4f})")

    if not args.write:
        log("nothing written (pass --write to fold this into the record)")
        return

    meta = json.loads((args.package / HISTORY).read_text(encoding="utf-8"))
    meta["inference_windows"] = n_final
    meta["selection_windows"] = n_select
    meta["final_holdout_auc"] = float(holdout_auc) if np.isfinite(holdout_auc) else None
    meta["final_gold_auc"] = (float(gold_auc) if gold_auc is not None
                              and np.isfinite(gold_auc) else None)
    (args.package / HISTORY).write_text(json.dumps(meta, indent=1) + "\n",
                                        encoding="utf-8")
    _predictions(record.uids, record.y, holdout_p).to_csv(
        args.package / HOLDOUT, index=False)
    if gold_p is not None:
        _predictions(record.gold_uids, record.gold_y, gold_p).to_csv(
            args.package / GOLD, index=False)
    log(f"record updated in place: {args.package}")


if __name__ == "__main__":
    main()
