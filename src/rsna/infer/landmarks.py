"""Running the trained landmark models over a split, and what they found.

Lifted out of `scripts/predict_landmark.py` for the reason the submission chain already
lives in the library: the scored Kaggle notebook has to run this too, and a notebook
that reimplemented it would drift from the command line the first time either changed.
The script is now a thin wrapper over these two functions.

Streams rather than caching. Training reads the same 301 studies sixty times, so it pays
for a cache; inference reads every study once, and a 14 GB file written and read back in
the same minute is work for nothing.

The output is one row per study per point — the landmark in patient millimetres, the
confidence, and the spread between folds. That table is the only thing the region side
needs, which is what keeps the two models independent of each other.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..config import Config
from ..dicom.ordering import order_slices
from ..landmark.config import LandmarkConfig
from ..landmark.loop import to_input
from ..landmark.network import LandmarkNet
from ..landmark.sample import sample_series
from ..landmark.series import LANDMARKS, PICKERS
from ..landmark.target import decode


def load_landmarks(run: str | Path, device: str = "cpu") -> tuple[list, LandmarkConfig]:
    """Every fold of a run, and **the config it was fitted under**.

    Read from the weights rather than rebuilt from the module default. Taking the
    default meant the caller could only ever predict the landmark that happened to be
    `LandmarkConfig`'s default — it refused a patellofemoral run outright, and had the
    two configs differed in a field that does not change the state dict it would have
    loaded the weights and sampled the input the wrong way instead, which is the version
    of this bug that says nothing.

    The folds still have to agree with each other: a run assembled from two trainings is
    not one model.
    """

    run = Path(run)
    paths = sorted(run.glob("fold*.pt"))
    if not paths:
        raise ValueError(f"no fold weights in {run}")
    config, models = None, []
    for path in paths:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        stored = LandmarkConfig.from_dict(ck["config"])
        if config is None:
            config = stored
        elif stored != config:
            raise ValueError(f"{path} was fitted under a different LandmarkConfig than "
                             f"{paths[0]}; this run is two models, not one")
        model = LandmarkNet(config, pretrained=False)
        model.load_state_dict(ck["state"])
        models.append(model.eval().to(device))
    return models, config


def _run_block(block, models, config, device, chosen) -> list:
    """Predict one batch with every fold, and keep their spread as a second signal.

    The folds are an ensemble by accident — they exist for validation — but they answer
    independently, so how far apart they land is a confidence the heatmap cannot give:
    the peak sharpness says "I found *a* meniscus", the fold spread says "and we agree
    which one".
    """

    volumes = torch.as_tensor(
        np.stack([s.volume for _, s, _, _ in block])).to(device).float() / 255.0
    x = to_input(volumes)
    heats = []
    with torch.no_grad():
        for model in models:
            heats.append(torch.sigmoid(model(x)).cpu().numpy())

    rows = []
    for j, (study, sampled, ordered, _) in enumerate(block):
        fold_points = [decode(h[j], sampled, config) for h in heats]
        pts = np.stack([p for p, _ in fold_points])              # (fold, point, 3)
        conf = np.mean([c for _, c in fold_points], axis=0)
        for k, name in enumerate(config.points):
            mean = np.nanmean(pts[:, k], axis=0)
            spread = float(np.nanmax(np.linalg.norm(pts[:, k] - mean, axis=1))) \
                if len(pts) > 1 else 0.0
            rows.append({
                "study": study, "point": name,
                "x_mm": round(float(mean[0]), 3), "y_mm": round(float(mean[1]), 3),
                "z_mm": round(float(mean[2]), 3),
                "confidence": round(float(conf[k]), 4),
                "spread_mm": round(spread, 3),
                "series": str(chosen[study]["SeriesInstanceUID"]),
                "n_slices": int(sampled.valid.sum()),
                "spacing_mm": round(float(sampled.native_spacing_mm), 3),
                "stride": int(sampled.stride), "ordered": bool(ordered),
            })
    return rows


def predict_landmarks(run: str | Path, series: pd.DataFrame, device: str = "cpu",
                      batch: int = 8, workers: int = 8, limit: int = 0,
                      log=print) -> tuple[pd.DataFrame, list]:
    """Every point of one landmark run over every study that carries its plane.

    `series` is the annotated header table, which must already carry `plane` — the
    competition's own label rather than something recovered from the header, so that
    this path and the training path pick the same series.

    Returns the table and the studies that failed, rather than raising on them: at
    inference a study that cannot be read must still reach the submission, scored by
    whatever the region machinery does with no pixels.
    """

    models, config = load_landmarks(run, device)
    log(f"{len(models)} folds from {run}, predicting {', '.join(config.points)}")

    # Which series to read is a property of the point, not of the caller: the picker and
    # its prefer_deep must be the ones the annotation bundle used, or the model is shown
    # a sequence it never trained on.
    landmark = LANDMARKS[config.points[0]]
    plane, deep = landmark["plane"], landmark["prefer_deep"]
    chosen = {}
    for study, group in series.groupby("StudyInstanceUID"):
        row = PICKERS[plane](group, prefer_deep=deep)
        if row is not None:
            chosen[study] = row
    log(f"  {len(chosen)} studies have a {plane.lower()} series")
    studies = list(chosen)[:limit or None]

    base = Config()

    def sample(study):
        try:
            folder = Path(chosen[study]["dir"])
            names, ordered = order_slices(
                str(folder), sorted(p.name for p in folder.glob("*.dcm")), base)
            return study, sample_series(folder, names, config), ordered, None
        except Exception as exc:  # noqa: BLE001
            return study, None, False, f"{type(exc).__name__}: {exc}"

    # Progress, because this stage is minutes long and otherwise silent: a scored
    # notebook that prints nothing for ten minutes is indistinguishable from one that
    # has hung, and the only way to tell is to wait for the nine-hour limit.
    rows, failed, block = [], [], []
    started = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for got in pool.map(sample, studies):
            if got[1] is None:
                failed.append((got[0], got[3]))
                continue
            block.append(got)
            if len(block) >= batch:
                rows += _run_block(block, models, config, device, chosen)
                done += len(block)
                block = []
                if done % 500 < batch:
                    rate = done / max(time.time() - started, 1e-6)
                    left = (len(studies) - done) / max(rate, 1e-6) / 60
                    log(f"  {done}/{len(studies)}  ({rate:.1f} studies/s, "
                        f"{left:.0f} min left)")
        if block:
            rows += _run_block(block, models, config, device, chosen)
    return pd.DataFrame(rows), failed
