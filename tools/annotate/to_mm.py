"""Turn an exported annotation file into the durable dataset, in millimetres.

    python -m tools.annotate.to_mm landmarks-2026-09-28.json -o data/manual_annotations/landmarks.csv
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


def lateral_end_from_click(a: dict, point: dict) -> str | None:
    """Which end of the stack is lateral, read off the click itself.

    The lateral meniscus sits about 25 mm from the lateral edge of a knee roughly 80 mm
    wide, so a point on it lands distinctly nearer one end of the stack than the other —
    and that end is the lateral one. Measured on the first 161 annotations: it agrees with
    the DICOM `Laterality` tag **161 times out of 161**, and the most central click was
    still 31 % of the half-stack away from the middle.

    So the annotator is never asked which side they are on. Clicking the lateral meniscus
    *is* saying it, and a second field would only restate what the first already contains.
    """

    n = a.get("n")
    if not n or n < 2 or point is None:
        return None
    return "first" if point["slice"] < (n - 1) / 2 else "last"


def side_at(a: dict, lat_end: str | None) -> str | None:
    """L or R, from which end of the stack sits further toward the patient's left.

    DICOM patient coordinates put the patient's left at positive x, so on a left knee the
    lateral end is the one with the **larger** x and on a right knee the smaller. Comparing
    the two ends rather than reading the sign of one is what makes this reliable: x is
    measured from the scanner isocentre, which is set on the knee and not on the body
    midline, so the sign alone is only *mostly* the side — **159 of 161** against the tag,
    where the comparison is **161 of 161**.

    Nothing downstream needs L or R; a landmark model maps pixels to a point. This exists
    so the dataset can be grouped and cross-checked, and the tag wins wherever it exists.
    """

    x_first, x_last = a.get("x_first"), a.get("x_last")
    if lat_end is None or x_first is None or x_last is None:
        return None
    x_lat, x_med = (x_first, x_last) if lat_end == "first" else (x_last, x_first)
    return "L" if x_lat > x_med else "R"


def rows(export: dict, annotator: str) -> list[dict]:
    out = []
    for a in export["annotations"]:
        if a.get("skipped"):
            out.append({"study": a["study"], "series": a["series"], "point": None,
                        "skipped": True, "annotator": annotator,
                        "side": a.get("side"), "side_from": a.get("side_from")})
            continue
        t = a["transform"]
        for name, p in a["points"].items():
            lat_end = lateral_end_from_click(a, p)
            from_click = side_at(a, lat_end)
            side = a.get("side") or from_click
            side_from = (a.get("side_from") if a.get("side")
                         else ("the click's own position in the stack" if side
                               else "unresolved"))
            # Where the tag exists the click restates it, so the two can disagree — and a
            # disagreement is a point placed on the wrong compartment, which is the one
            # error this dataset cannot survive and nothing else would report.
            agrees = (None if not (a.get("side") and from_click)
                      else a["side"] == from_click)

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
                "side": side, "side_from": side_from,
                "side_tagged": a.get("side"), "side_from_click": from_click,
                "lat_end": lat_end, "side_agrees": agrees,
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
    ap.add_argument("--excluded", type=Path,
                    default=Path("data/manual_annotations/excluded.csv"),
                    help="studies to drop, with a reason, read by default so a re-run "
                         "cannot quietly bring back an annotation that was withdrawn. "
                         "What it dropped is always printed.")
    ap.add_argument("--no-exclude", action="store_true",
                    help="keep everything, including withdrawn annotations")
    ap.add_argument("--bundle", type=Path,
                    help="a bundle's studies.json, to backfill fields an older export "
                         "predates — the stack length and the patient x at each end. "
                         "Without them the side cannot be recovered from the click, and "
                         "the cross-check below stays silent.")
    args = ap.parse_args()

    tables, names = [], []
    dropped, seen = {}, set()
    if args.excluded.exists() and not args.no_exclude:
        table = pd.read_csv(args.excluded)
        dropped = dict(zip(table["study"], table["reason"]))

    backfill = {}
    if args.bundle:
        backfill = {b["study"]: b for b in json.loads(args.bundle.read_text())["studies"]}

    for path in args.exports:
        export = json.loads(path.read_text())
        kept = [a for a in export["annotations"] if a["study"] not in dropped]
        for a in export["annotations"]:
            if a["study"] in dropped:
                seen.add(a["study"])
                print(f"  DROPPED …{a['study'][-11:]}: {dropped[a['study']]}")
        export["annotations"] = kept
        for a in export["annotations"]:
            b = backfill.get(a["study"])
            if b:
                for k in ("n", "x_first", "x_last"):
                    a.setdefault(k, b.get(k))
        name = path.stem
        table = pd.DataFrame(rows(export, name))
        tables.append(table)
        names.append(name)
        points = table[table["point"].notna()]
        print(f"{path.name}: {points['study'].nunique()} studies, {len(points)} points, "
              f"{int(table['skipped'].sum())} skipped")
        if "side_agrees" in points and points["side_agrees"].notna().any():
            checked = points[points["side_agrees"].notna()]
            # A click that contradicts the DICOM tag and one that contradicts the
            # geometric guess are opposite events. The tag is the scanner's own record,
            # so a disagreement there points at the click. The guess is right about 91 %
            # of the time, so a disagreement there is the click *correcting* it — which
            # is the whole reason the bundle stopped excluding untagged studies.
            tagged = checked[checked["side_from"].str.contains("tag", na=False)]
            guessed = checked[~checked["side_from"].str.contains("tag", na=False)]
            for label, group, verdict in (
                    ("DICOM tag", tagged, "CHECK THIS ONE — the tag is the scanner's own "
                                          "record, so the click is the likelier error"),
                    ("geometric guess", guessed, "the click wins: the guess is only ~91 % "
                                                 "reliable and this is what it is for")):
                if not len(group):
                    continue
                bad = group[~group["side_agrees"].astype(bool)]
                print(f"  vs the {label}: {len(group) - len(bad)}/{len(group)} clicks agree")
                for r in bad.itertuples():
                    print(f"    …{r.study[-11:]}  slice {r.slice}  "
                          f"says {r.side_tagged}, click says {r.side_from_click}")
                if len(bad):
                    print(f"      -> {verdict}")

    # A study id is forty digits and nobody types one correctly. An entry that matched
    # nothing is a typo that silently keeps the annotation it was meant to withdraw —
    # which is what happened the first time this file was written.
    missing = sorted(set(dropped) - seen)
    if missing:
        raise SystemExit(
            f"{args.excluded} lists {len(missing)} studies that appear in none of the "
            f"exports: {[m[-11:] for m in missing]}. Fix the ids rather than run on, or "
            f"the annotations they withdraw stay in the output.")

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
