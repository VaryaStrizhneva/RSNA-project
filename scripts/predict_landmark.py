"""Run the trained landmark models over a whole split and write what they found.

    python -m scripts.predict_landmark --run out/landmark_reference --split train_series

Streams rather than caching. Training reads the same 301 studies sixty times, so it pays
for a cache; inference reads 4407 studies once, and a 14 GB file to be written and read
back in the same minute is work for nothing.

The output is one row per study — the landmark in patient millimetres, the confidence,
and the agreement between folds. That table is the only thing the ROI side needs, which
is what keeps the two models independent of each other.
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rsna.config import Config                                       # noqa: E402
from rsna.dicom.headers import annotate, walk                        # noqa: E402
from rsna.dicom.ordering import order_slices                         # noqa: E402
from rsna.landmark import LandmarkConfig, sample_series            # noqa: E402
from rsna.landmark.series import LANDMARKS, PICKERS                  # noqa: E402
from rsna.landmark.loop import to_input                              # noqa: E402
from rsna.landmark.network import LandmarkNet                        # noqa: E402
from rsna.landmark.target import decode                              # noqa: E402


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def load_models(run: Path, device: str) -> tuple[list, LandmarkConfig]:
    """Every fold of a run, and **the config it was fitted under**.

    Read from the weights rather than rebuilt from the module default. Taking the
    default here meant this script could only ever predict the landmark that happened
    to be `LandmarkConfig`'s default — it refused a patellofemoral run outright, and had
    the two configs differed in a field that does not change the state dict it would
    have loaded the weights and sampled the input the wrong way instead, which is the
    version of this bug that says nothing.

    The folds still have to agree with each other: a run assembled from two trainings is
    not one model.
    """

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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="out/landmark_reference", type=Path)
    ap.add_argument("--data-root", default="data/raw", type=Path)
    ap.add_argument("--split", default="train_series")
    ap.add_argument("--dicom-root", default=None, type=Path,
                    help="where the pixels are, if not under --data-root")
    ap.add_argument("--out", default=None, type=Path)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    models, config = load_models(args.run, args.device)
    log(f"{len(models)} folds from {args.run}, predicting {', '.join(config.points)}")

    root = args.dicom_root or args.data_root
    series = annotate(walk(Path(root), args.split))
    # The plane is the competition's own label, not something recovered from the
    # header — `scripts/train.py` reads it the same way, so both paths agree.
    planes = pd.read_csv(args.data_root / f"{args.split.replace('_series', '')}_series.csv")
    series["plane"] = series["SeriesInstanceUID"].map(
        dict(zip(planes["SeriesInstanceUID"], planes["Anatomical_Plane"])))
    log(f"{len(series)} series over {series.StudyInstanceUID.nunique()} studies, "
        f"{int(series.plane.isna().sum())} with no plane")

    # Which series to read is a property of the point, not of this script: the picker
    # and its prefer_deep must be the ones the annotation bundle used, or the model is
    # shown a sequence it never trained on.
    landmark = LANDMARKS[config.points[0]]
    plane, deep = landmark["plane"], landmark["prefer_deep"]
    log(f"{config.points[0]}: reading the {plane.lower()} series"
        f"{', preferring a 3D acquisition' if deep else ''}")
    chosen = {}
    for study, group in series.groupby("StudyInstanceUID"):
        row = PICKERS[plane](group, prefer_deep=deep)
        if row is not None:
            chosen[study] = row
    log(f"{len(chosen)} studies have a {plane.lower()} series")
    studies = list(chosen)[:args.limit or None]

    base = Config()

    def sample(study):
        try:
            folder = Path(chosen[study]["dir"])
            names, ordered = order_slices(
                str(folder), sorted(p.name for p in folder.glob("*.dcm")), base)
            return study, sample_series(folder, names, config), ordered, None
        except Exception as exc:  # noqa: BLE001
            return study, None, False, f"{type(exc).__name__}: {exc}"

    rows, failed = [], []
    started = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        block: list = []
        for got in pool.map(sample, studies):
            if got[1] is None:
                failed.append((got[0], got[3]))
                continue
            block.append(got)
            if len(block) < args.batch:
                continue
            rows += run_block(block, models, config, args.device, chosen)
            block = []
            if len(rows) % 400 < args.batch:
                rate = len(rows) / max(time.time() - started, 1e-6)
                log(f"  {len(rows)}/{len(studies)}  ({rate:.1f} studies/s, "
                    f"{(len(studies) - len(rows)) / max(rate, 1e-6) / 60:.0f} min left)")
        if block:
            rows += run_block(block, models, config, args.device, chosen)

    out = args.out or Path("out/landmarks") / f"{args.split}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(out, index=False)
    log(f"\n{len(table)} studies, {len(failed)} failed -> {out}")
    for point in config.points:
        sel = table[table["point"] == point]
        log(f"  {point}: confidence median {sel.confidence.median():.3f}, "
            f"fold spread median {sel.spread_mm.median():.1f} mm, "
            f"p90 {sel.spread_mm.quantile(0.9):.1f} mm")
    for study, why in failed[:10]:
        log(f"  FAILED {study[-11:]}: {why}")
    return 0


def run_block(block, models, config, device, chosen) -> list:
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
        for fold_points in ([decode(h[j], sampled, config) for h in heats],):
            pts = np.stack([p for p, _ in fold_points])          # (fold, point, 3)
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


if __name__ == "__main__":
    raise SystemExit(main())
