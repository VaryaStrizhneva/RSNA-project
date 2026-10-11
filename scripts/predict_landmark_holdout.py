"""Where each landmark model put the point on the studies it did **not** train on.

    python -m scripts.predict_landmark_holdout --experiment landmark_both \\
        --cache /data/mgr/rsna-knee/landmark-cache-both

Writes `out/<run>/holdout_points.csv`: one row per study and point, in millimetres,
predicted by the single fold that held that study out.

`history.json` already records the held-out *distance*, but it throws the coordinates
away — `_errors` decodes the point, measures it and keeps only the number. That is enough
to report a median and useless for looking at anything: a page cannot draw a distance.

And the alternative it was drawing instead is worse than nothing for validation.
`roi-infer/landmarks.csv` averages all five folds, four of which trained on any given
annotated study, so the point it carries is mostly in-sample there. Measured on
`landmark_both`, two studies sit 53.5 and 40.5 mm from the click when scored by the fold
that never saw them, and 10.4 and 7.1 mm when scored by the mean of five — a page showing
the second while reporting the first makes the model look like it failed somewhere the
picture says it did not.

This closes that gap: the fold assignment is recomputed from the same `folds(n, k, seed)`
over the same cache order, so the study held out here is the study held out then.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from rsna.landmark import LandmarkConfig, cache                      # noqa: E402
from rsna.landmark.cache import restore                              # noqa: E402
from rsna.landmark.loop import folds, to_input                       # noqa: E402
from rsna.landmark.network import LandmarkNet                        # noqa: E402
from rsna.landmark.target import decode                              # noqa: E402


def log(message: str) -> None:
    print(message, flush=True)


@torch.no_grad()
def predict_fold(model, volumes, records, index: np.ndarray,
                 config: LandmarkConfig, device: str, batch: int) -> list[dict]:
    """Decode this fold's answer for the studies it was scored on.

    The same `decode(heat, restore(record, volume), config)` the training loop calls, so
    the coordinates here are the ones whose distance `history.json` reports. Anything
    else — re-running the sampler, or decoding at a different resolution — would produce
    a point that agrees with no recorded number.
    """

    rows = []
    for start in range(0, len(index), batch):
        take = index[start:start + batch]
        # `.float() / 255` as the training loop's own `load` does — the cache is uint8
        # and a model fed raw bytes answers a question about a different image.
        volume = torch.as_tensor(np.stack([volumes[i] for i in take]),
                                 device=device).float() / 255.0
        heat = torch.sigmoid(model(to_input(volume))).float().cpu().numpy()
        for j, i in enumerate(take):
            record = records[i]
            got, _ = decode(heat[j], restore(record, volumes[i]), config)
            for k, name in enumerate(config.points):
                truth = record["points"].get(name)
                rows.append({
                    "study": record["study"], "point": name,
                    "x_mm": float(got[k][0]), "y_mm": float(got[k][1]),
                    "z_mm": float(got[k][2]),
                    "error_mm": (float(np.linalg.norm(got[k] - np.asarray(truth, float)))
                                 if truth is not None else float("nan")),
                })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experiment", default="landmark_both")
    ap.add_argument("--cache", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    run = ROOT / "out" / args.experiment
    recorded = json.loads((run / "history.json").read_text())
    config = LandmarkConfig.from_dict(recorded["config"])
    settings = recorded.get("run", {})

    cache_dir = args.cache or Path("/data/mgr/rsna-knee/landmark-cache")
    volumes, records = cache.load(cache_dir, config)
    log(f"{args.experiment}: {len(records)} studies from {cache_dir}, "
        f"points {', '.join(config.points)}")

    splits = folds(len(records), settings.get("folds", 5), settings.get("seed", 0))
    rows: list[dict] = []
    for k, val in enumerate(splits):
        path = run / f"fold{k}.pt"
        if not path.is_file():
            log(f"  fold {k}: no weights, skipped")
            continue
        state = torch.load(path, map_location="cpu", weights_only=False)
        model = LandmarkNet(LandmarkConfig.from_dict(state["config"]))
        model.load_state_dict(state["state"])
        model = model.eval().to(args.device)
        got = predict_fold(model, volumes, records, val, config, args.device, args.batch)
        for row in got:
            row["fold"] = k
        rows += got
        err = np.array([r["error_mm"] for r in got], float)
        log(f"  fold {k}: {len(val)} held out, median "
            f"{np.nanmedian(err):.2f} mm, worst {np.nanmax(err):.1f} mm")

    table = pd.DataFrame(rows)
    out = args.out or run / "holdout_points.csv"
    table.to_csv(out, index=False)

    # The sum has to match what the run recorded, or the fold assignment drifted and
    # every point here belongs to the wrong study.
    recorded_errors = recorded.get("errors", {})
    checked = mismatch = 0
    for study, values in recorded_errors.items():
        values = values if isinstance(values, list) else [values]
        for k, name in enumerate(config.points):
            hit = table[(table.study == study) & (table.point == name)]
            if hit.empty:
                continue
            checked += 1
            if abs(float(hit.iloc[0]["error_mm"]) - float(values[k])) > 0.05:
                mismatch += 1
    log(f"\n{out}  ({len(table)} rows)")
    log(f"  reproduces history.json on {checked - mismatch} of {checked} points"
        + ("" if not mismatch else f" — {mismatch} DISAGREE, the split has moved"))
    return 1 if mismatch else 0


if __name__ == "__main__":
    raise SystemExit(main())
