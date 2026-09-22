"""Build a reproducible blend of report-derived label tables.

Original external tables stay untouched. This writes a new CSV under
`data/processed/labels/` by default, plus a sidecar JSON describing exactly which
tables and weights produced it.

Examples:

    python -m tools.build_label_blend \
        --table data/external/pilkwang-rsna-knee-llm-labels/report_labels_v2.csv \
        --table data/external/stevenleehans-rsna-knee-llm-report-labels/llm_labels_v4_blend.csv \
        --weight 0.5 --weight 0.5 \
        --method rank \
        --out data/processed/labels/pilkwang_steven_rank_50_50.csv

    python -m tools.build_label_blend \
        --table data/external/pilkwang-rsna-knee-llm-labels/report_labels_v2.csv \
        --table data/external/stevenleehans-rsna-knee-llm-report-labels/llm_labels_v4_blend.csv \
        --weight 0.25 --weight 0.75 \
        --method rank \
        --out data/processed/labels/pilkwang_steven_rank_25_75.csv

    python -m tools.build_label_blend \
        --recipe configs/label_blends/pilkwang_steven_conservative.json \
        --out data/processed/labels/pilkwang_steven_target_conservative.csv
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.rsna.data.labels import load_label_table
from src.rsna.data.metadata import TARGET_COLUMNS


DEFAULT_METHOD = "rank"


def normalise_weights(n: int, weights: list[float] | None) -> np.ndarray:
    """Return one non-negative weight per source, normalised to sum to one."""

    if weights is None or not weights:
        weights = [1.0] * n
    if len(weights) != n:
        raise SystemExit(f"got {len(weights)} weights for {n} tables")

    out = np.asarray(weights, dtype=float)
    if not np.isfinite(out).all() or (out < 0).any():
        raise SystemExit("weights must be finite and non-negative")
    if out.sum() <= 0:
        raise SystemExit("at least one weight must be positive")
    return out / out.sum()


def normalise_named_weights(weights: dict[str, float], source_names: set[str]) -> dict[str, float]:
    """Validate and normalise one target's source weights."""

    unknown = sorted(set(weights) - source_names)
    if unknown:
        raise SystemExit(f"unknown source name(s) in target weights: {unknown}")

    values = {name: float(weight) for name, weight in weights.items()}
    if any(not np.isfinite(weight) or weight < 0.0 for weight in values.values()):
        raise SystemExit("target weights must be finite and non-negative")

    total = sum(values.values())
    if total <= 0.0:
        raise SystemExit("each target needs at least one positive source weight")
    return {name: weight / total for name, weight in values.items() if weight > 0.0}


def prepare_table(table: pd.DataFrame, method: str) -> pd.DataFrame:
    """Put one table onto the scale used for blending."""

    if method == "rank":
        # Percentile ranks put different score scales onto the same [0, 1] ordering.
        return table.rank(method="average", pct=True)
    if method == "raw":
        return table.copy()
    raise ValueError(f"unknown blend method {method!r}")


def weighted_available_mean(values: list[pd.DataFrame], weights: np.ndarray) -> pd.DataFrame:
    """Weighted mean, renormalising where a source is missing a study/target."""

    index = values[0].index
    numerator = pd.DataFrame(0.0, index=index, columns=TARGET_COLUMNS)
    denominator = pd.DataFrame(0.0, index=index, columns=TARGET_COLUMNS)

    for value, weight in zip(values, weights):
        value = value.reindex(index)
        present = value.notna()
        numerator = numerator + value.fillna(0.0) * weight
        denominator = denominator + present.astype(float) * weight

    out = numerator / denominator.replace(0.0, np.nan)
    return out


def weighted_target_mean(
    values_by_source: dict[str, pd.DataFrame],
    weights_by_target: dict[str, dict[str, float]],
) -> pd.DataFrame:
    """Weighted mean with different source weights for each target."""

    index = next(iter(values_by_source.values())).index
    out = pd.DataFrame(index=index, columns=TARGET_COLUMNS, dtype=float)

    for target in TARGET_COLUMNS:
        numerator = pd.Series(0.0, index=index)
        denominator = pd.Series(0.0, index=index)

        for source_name, weight in weights_by_target[target].items():
            value = values_by_source[source_name][target].reindex(index)
            present = value.notna()
            numerator = numerator + value.fillna(0.0) * weight
            denominator = denominator + present.astype(float) * weight

        out[target] = numerator / denominator.replace(0.0, np.nan)

    return out


def build_blend(paths: list[Path], weights: np.ndarray, method: str) -> pd.DataFrame:
    raw = [load_label_table(path) for path in paths]
    union = sorted(set().union(*(set(table.index) for table in raw)))
    prepared = [prepare_table(table, method).reindex(union) for table in raw]
    blended = weighted_available_mean(prepared, weights)
    blended.index.name = "StudyInstanceUID"
    return blended.reset_index()


def load_recipe(path: Path) -> dict[str, Any]:
    """Load a target-specific blend recipe JSON."""

    try:
        recipe = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"invalid recipe JSON {path}: {exc}") from exc

    if not isinstance(recipe, dict):
        raise SystemExit("recipe must be a JSON object")
    if "sources" not in recipe:
        raise SystemExit("recipe is missing 'sources'")
    if "targets" not in recipe:
        raise SystemExit("recipe is missing 'targets'")
    if not isinstance(recipe["sources"], dict) or not recipe["sources"]:
        raise SystemExit("recipe 'sources' must be a non-empty object")
    if len(recipe["sources"]) < 2:
        raise SystemExit("blend at least two recipe sources")
    if not isinstance(recipe["targets"], dict):
        raise SystemExit("recipe 'targets' must be an object")

    method = recipe.get("method", DEFAULT_METHOD)
    if method not in {"rank", "raw"}:
        raise SystemExit("recipe method must be 'rank' or 'raw'")

    return recipe


def recipe_paths(recipe: dict[str, Any]) -> dict[str, Path]:
    """Return source-name to path mapping from a recipe."""

    return {str(name): Path(path) for name, path in recipe["sources"].items()}


def recipe_target_weights(recipe: dict[str, Any], source_names: set[str]) -> dict[str, dict[str, float]]:
    """Expand recipe target weights to all competition targets."""

    target_block = recipe["targets"]
    unknown_targets = sorted(set(target_block) - set(TARGET_COLUMNS) - {"default"})
    if unknown_targets:
        raise SystemExit(f"unknown target name(s) in recipe: {unknown_targets}")

    default_weights = target_block.get("default")
    if default_weights is None:
        missing = [target for target in TARGET_COLUMNS if target not in target_block]
        if missing:
            raise SystemExit(f"recipe has no default weights and misses targets: {missing}")
    elif not isinstance(default_weights, dict):
        raise SystemExit("recipe target 'default' must be an object")

    out: dict[str, dict[str, float]] = {}
    for target in TARGET_COLUMNS:
        weights = target_block.get(target, default_weights)
        if not isinstance(weights, dict):
            raise SystemExit(f"recipe target {target!r} must be an object")
        out[target] = normalise_named_weights(weights, source_names)
    return out


def build_recipe_blend(recipe: dict[str, Any], method: str) -> tuple[pd.DataFrame, dict[str, Path], dict[str, dict[str, float]]]:
    paths_by_source = recipe_paths(recipe)
    weights_by_target = recipe_target_weights(recipe, set(paths_by_source))

    raw = {name: load_label_table(path) for name, path in paths_by_source.items()}
    union = sorted(set().union(*(set(table.index) for table in raw.values())))
    prepared = {name: prepare_table(table, method).reindex(union) for name, table in raw.items()}

    blended = weighted_target_mean(prepared, weights_by_target)
    blended.index.name = "StudyInstanceUID"
    return blended.reset_index(), paths_by_source, weights_by_target


def write_metadata(path: Path, args, method: str, frame: pd.DataFrame, blend_config: dict[str, Any]) -> None:
    meta_path = args.meta_out or path.with_suffix(".meta.json")
    payload = {
        "written": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "method": method,
        "output": str(path),
        "n_studies": int(len(frame)),
        "targets": TARGET_COLUMNS,
        "missing_cells": int(frame[TARGET_COLUMNS].isna().sum().sum()),
        **blend_config,
    }
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"metadata written to {meta_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", action="append", type=Path,
                        help="Label table to blend. Repeat at least twice.")
    parser.add_argument("--weight", action="append", type=float, default=None,
                        help="Weight for the matching --table. Defaults to equal weights.")
    parser.add_argument("--recipe", type=Path, default=None,
                        help="JSON recipe with source paths and per-target weights.")
    parser.add_argument("--method", choices=["rank", "raw"], default=None,
                        help="Blend percentile ranks or raw scores.")
    parser.add_argument("--out", required=True, type=Path,
                        help="CSV path to write, usually under data/processed/labels.")
    parser.add_argument("--meta-out", type=Path, default=None,
                        help="Sidecar JSON path. Defaults to OUT with .meta.json suffix.")
    args = parser.parse_args()

    if args.recipe is not None and args.table:
        raise SystemExit("use either --recipe or --table inputs, not both")

    if args.recipe is not None:
        recipe = load_recipe(args.recipe)
        method = args.method or recipe.get("method", DEFAULT_METHOD)
        frame, paths_by_source, weights_by_target = build_recipe_blend(recipe, method)
        blend_config = {
            "recipe": str(args.recipe),
            "sources": {name: str(path) for name, path in paths_by_source.items()},
            "target_weights": weights_by_target,
        }
        weight_summary = []
        for target in TARGET_COLUMNS:
            weights = ", ".join(f"{name}={weight:.3f}" for name, weight in weights_by_target[target].items())
            weight_summary.append(f"{target}: {weights}")
    else:
        if not args.table or len(args.table) < 2:
            raise SystemExit("blend at least two --table inputs")

        method = args.method or DEFAULT_METHOD
        weights = normalise_weights(len(args.table), args.weight)
        frame = build_blend(args.table, weights, method)
        blend_config = {
            "tables": [str(Path(table)) for table in args.table],
            "weights": weights.tolist(),
        }
        weight_summary = [
            ", ".join(f"{path.name}={weight:.3f}" for path, weight in zip(args.table, weights))
        ]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(f"blend written to {args.out}")
    print(f"studies: {len(frame)}")
    print(f"missing target cells: {int(frame[TARGET_COLUMNS].isna().sum().sum())}")
    print("weights:")
    for line in weight_summary:
        print(f"  {line}")
    write_metadata(args.out, args, method, frame, blend_config)


if __name__ == "__main__":
    main()
