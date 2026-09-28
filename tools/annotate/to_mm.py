"""Turn an exported annotation file into the durable dataset, in millimetres.

    python -m tools.annotate.to_mm landmarks-2026-09-28.json -o data/annotations/landmarks.csv
    python -m tools.annotate.to_mm mine.json hers.json --agreement

The browser stores a click as a pixel on a named slice, because that is the only thing
it actually observed. Millimetres in the patient are *derived* here, from the geometry
the bundle carried — so if this derivation is ever found wrong, it is re-run, and nobody
clicks anything again. The raw columns travel with the derived ones for exactly that
reason.

`--agreement` compares two exports of the same studies. That number is the floor: no
landmark model can be more accurate than the two people who defined the landmark
disagree with each other, and it is worth knowing before annotating three hundred more.
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.annotate.bundle import patient_mm   # noqa: E402


def rows(export: dict, annotator: str) -> list[dict]:
    out = []
    for a in export["annotations"]:
        if a.get("skipped"):
            out.append({"study": a["study"], "series": a["series"], "point": None,
                        "skipped": True, "annotator": annotator})
            continue
        t = a["transform"]
        for name, p in a["points"].items():
            # jpeg pixel -> native pixel -> patient millimetres
            native_row = t["row0"] + p["row"] * t["scale"]
            native_col = t["col0"] + p["col"] * t["scale"]
            mm = (patient_mm(p["ipp"], p["iop"], p["ps"], native_row, native_col)
                  if p.get("ipp") else [np.nan] * 3)
            out.append({
                "study": a["study"], "series": a["series"], "point": name,
                "sop": p["sop"], "slice": p["slice"],
                "jpeg_row": p["row"], "jpeg_col": p["col"],
                "native_row": round(native_row, 3), "native_col": round(native_col, 3),
                "x_mm": round(mm[0], 3), "y_mm": round(mm[1], 3), "z_mm": round(mm[2], 3),
                "side": a.get("side"), "side_from": a.get("side_from"),
                "skipped": False, "annotator": annotator,
            })
    return out


def agreement(tables: list[pd.DataFrame], names: list[str]) -> None:
    """How far apart two people put the same landmark, in millimetres."""

    print("\ninter-annotator agreement — the floor on any model's accuracy\n")
    for (i, a), (j, b) in combinations(enumerate(tables), 2):
        merged = a.merge(b, on=["study", "point"], suffixes=("_a", "_b"))
        merged = merged[merged["point"].notna()]
        if not len(merged):
            print(f"  {names[i]} vs {names[j]}: no study annotated by both")
            continue
        d = np.linalg.norm(
            merged[["x_mm_a", "y_mm_a", "z_mm_a"]].to_numpy(float)
            - merged[["x_mm_b", "y_mm_b", "z_mm_b"]].to_numpy(float), axis=1)
        print(f"  {names[i]} vs {names[j]} — {len(merged)} points on "
              f"{merged['study'].nunique()} studies")
        print(f"    median {np.median(d):5.1f} mm   p90 {np.percentile(d, 90):5.1f} mm   "
              f"max {d.max():5.1f} mm")
        for point, group in merged.assign(d=d).groupby("point"):
            print(f"      {point:10s} median {group['d'].median():5.1f} mm  "
                  f"(n={len(group)})")
        print(f"\n    A ROI of 40 mm tolerates about 12 mm. "
              f"{'Within it.' if np.percentile(d, 90) < 12 else 'NOT within it — the landmark definition needs work before annotating more.'}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("exports", nargs="+", type=Path)
    ap.add_argument("-o", "--out", type=Path)
    ap.add_argument("--agreement", action="store_true")
    args = ap.parse_args()

    tables, names = [], []
    for path in args.exports:
        export = json.loads(path.read_text())
        name = path.stem
        table = pd.DataFrame(rows(export, name))
        tables.append(table)
        names.append(name)
        points = table[table["point"].notna()]
        print(f"{path.name}: {points['study'].nunique()} studies, {len(points)} points, "
              f"{int(table['skipped'].sum())} skipped")

    if args.agreement:
        if len(tables) < 2:
            print("\n--agreement needs two exports")
        else:
            agreement(tables, names)

    if args.out:
        combined = pd.concat(tables, ignore_index=True)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(args.out, index=False)
        print(f"\nwrote {args.out}  ({len(combined)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
