"""Evaluate a trained package on labelled training studies.

This is a post-training audit, not a submission step. It predicts on a train split,
compares the image model to the chosen weak report-label table, and compares to the
58 expert labels where they are present.

    python -m scripts.evaluate_train --package out/package --data-root data/raw \
        --dicom-root /path/to/kaggle_root --labels data/external/.../labels.csv \
        --out out/eval/package
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.rsna.config import TARGETS, Config
from src.rsna.data.labels import (
    load_confidence_table,
    load_label_table,
    load_verdict_table,
)
from src.rsna.dicom import annotate, build_cache, laterality_of, pick_slots, walk
from src.rsna.infer import blend_members, predict_member
from src.rsna.model import load_member, read_manifest
from src.rsna.train.loop import macro_auc

T0 = time.time()


def log(message: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {message}", flush=True)


def plane_map_for(root: Path, split: str) -> dict:
    csv = root / f"{split.replace('_series', '')}_series.csv"
    table = pd.read_csv(csv, dtype={"SeriesInstanceUID": str})
    return dict(zip(table["SeriesInstanceUID"], table["Anatomical_Plane"]))


def gold_labels(train: pd.DataFrame, studies: list[str]) -> pd.DataFrame:
    gold = train.set_index("StudyInstanceUID")[TARGETS]
    gold = gold[gold.notna().all(axis=1)].astype(int)
    return gold.reindex(studies)


def auc_against_binary(label_frame: pd.DataFrame, predictions: pd.DataFrame) -> float:
    y = label_frame.reindex(predictions.index)[TARGETS]
    keep = y.notna().all(axis=1)
    if not keep.any():
        return float("nan")
    return macro_auc((y.loc[keep].to_numpy() > 0.5).astype(int),
                     predictions.loc[keep, TARGETS].to_numpy())


def write_audit(
    path: Path,
    studies: list[str],
    predictions: pd.DataFrame,
    weak: pd.DataFrame,
    gold: pd.DataFrame,
    confidence: pd.DataFrame | None,
    verdict: pd.DataFrame | None,
    reports: pd.Series,
) -> None:
    rows = []
    weak = weak.reindex(studies)
    gold = gold.reindex(studies)
    confidence = confidence.reindex(studies) if confidence is not None else None
    verdict = verdict.reindex(studies) if verdict is not None else None

    for study in studies:
        for target in TARGETS:
            pred = float(predictions.at[study, target])
            weak_value = weak.at[study, target]
            gold_value = gold.at[study, target]
            row = {
                "StudyInstanceUID": study,
                "target": target,
                "model_prediction": pred,
                "weak_label": weak_value,
                "weak_binary": (
                    int(float(weak_value) > 0.5) if pd.notna(weak_value) else pd.NA
                ),
                "weak_abs_diff": (
                    abs(pred - float(weak_value)) if pd.notna(weak_value) else pd.NA
                ),
                "gold_label": gold_value,
                "gold_abs_diff": (
                    abs(pred - float(gold_value)) if pd.notna(gold_value) else pd.NA
                ),
                "confidence": (
                    confidence.at[study, target]
                    if confidence is not None and study in confidence.index else pd.NA
                ),
                "verdict": (
                    verdict.at[study, target]
                    if verdict is not None and study in verdict.index else pd.NA
                ),
                "report": reports.get(study, ""),
            }
            rows.append(row)

    audit = pd.DataFrame(rows)
    audit = audit.sort_values(
        ["gold_abs_diff", "weak_abs_diff", "StudyInstanceUID", "target"],
        ascending=[False, False, True, True],
        na_position="last",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--data-root", default="data/raw", type=Path)
    parser.add_argument("--dicom-root", default=None, type=Path,
                        help="Folder holding DICOMs. Defaults to --data-root.")
    parser.add_argument("--split", default="train_series")
    parser.add_argument("--labels",
                        default="data/external/pilkwang-rsna-knee-llm-labels/"
                                "report_labels_v2.csv")
    parser.add_argument("--encoder", default="models/dinov2-small")
    parser.add_argument("--out", default="out/eval_train", type=Path)
    parser.add_argument("--cache", default=None, type=Path)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-windows", type=int, default=None,
                        help="Read each member over at most this many windows.")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    dicom_root = args.dicom_root or args.data_root
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    args.out.mkdir(parents=True, exist_ok=True)
    log(f"device: {device}")

    manifest = read_manifest(args.package)
    entries = manifest["members"]
    configs = {json.dumps(e["config"], sort_keys=True) for e in entries}
    if len(configs) > 1:
        raise SystemExit("members disagree about the config; decode one cache per group")
    config = Config.from_dict(entries[0]["config"])
    log(f"package: {len(entries)} member(s), config {config.img}px/{config.slices} slices")

    headers = annotate(walk(dicom_root, args.split))
    if headers.empty:
        raise SystemExit(f"no series under {dicom_root / args.split}")
    if args.limit:
        keep = sorted(headers["StudyInstanceUID"].unique())[: args.limit]
        headers = headers[headers["StudyInstanceUID"].isin(keep)]
        log(f"limited to {len(keep)} studies")

    plane_map = plane_map_for(args.data_root, args.split)
    headers["plane"] = headers["SeriesInstanceUID"].map(plane_map)
    log(f"{len(headers)} series across {headers['StudyInstanceUID'].nunique()} studies")

    sides, stats = laterality_of(headers, config)
    log(f"laterality: {stats['from_tag']} tagged, {stats['from_geometry']} from geometry, "
        f"{stats['unresolved']} unresolved, {stats['disagree']} disagree")
    slots = pick_slots(headers, plane_map, config)
    studies = sorted(slots)
    log(f"slots: {sum(len(s) for s in slots.values())}/{len(studies) * config.n_slot} filled")

    failures: list = []
    _, cache, mask = build_cache(slots, plane_map, sides, config,
                                 cache_slices=config.slices, path=args.cache,
                                 log=log, decode_failures=failures)
    if failures:
        log(f"{len(failures)} series had a slice that would not decode")

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
    pred = pd.DataFrame(blended, columns=TARGETS, index=studies)
    pred.index.name = "StudyInstanceUID"
    pred.to_csv(args.out / "predictions.csv")

    weak = load_label_table(args.labels)
    confidence = load_confidence_table(args.labels)
    verdict = load_verdict_table(args.labels)
    train = pd.read_csv(args.data_root / "train.csv", dtype={"StudyInstanceUID": str})
    gold = gold_labels(train, studies)
    reports = train.set_index("StudyInstanceUID")["Report"]

    weak_auc = auc_against_binary(weak, pred)
    gold_auc = auc_against_binary(gold, pred)
    weak_coverage = int(weak.reindex(studies)[TARGETS].notna().all(axis=1).sum())
    gold_coverage = int(gold[TARGETS].notna().all(axis=1).sum())

    write_audit(args.out / "audit_long.csv", studies, pred, weak, gold,
                confidence, verdict, reports)

    summary = {
        "package": str(args.package),
        "label_source": str(args.labels),
        "split": args.split,
        "studies": len(studies),
        "series": int(len(headers)),
        "slots_filled": int(mask.sum()),
        "slots_total": int(len(studies) * config.n_slot),
        "weak_labelled_studies": weak_coverage,
        "gold_labelled_studies": gold_coverage,
        "weak_auc": None if not np.isfinite(weak_auc) else float(weak_auc),
        "gold_auc": None if not np.isfinite(gold_auc) else float(gold_auc),
        "max_windows": args.max_windows,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    log(f"weak AUC: {weak_auc:.4f}" if np.isfinite(weak_auc) else "weak AUC: NaN")
    log(f"gold AUC: {gold_auc:.4f}" if np.isfinite(gold_auc) else "gold AUC: NaN")
    log(f"wrote {args.out / 'predictions.csv'}")
    log(f"wrote {args.out / 'audit_long.csv'}")
    log(f"wrote {args.out / 'summary.json'}")


if __name__ == "__main__":
    main()
