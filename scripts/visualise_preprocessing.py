"""The preprocessing figures, without Jupyter.

Same functions as notebooks/preprocessing.ipynb, written to PNG instead of shown
inline. For regenerating the images, or for anyone without a notebook environment.

    python -m scripts.visualise_preprocessing --out out/steps
    python -m scripts.visualise_preprocessing --study 7456991776 --slot SAG_FLUID_FS
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")

from src.rsna import viz
from src.rsna.config import Config
from src.rsna.dicom import annotate, laterality_of, pick_slots, walk


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/raw")
    parser.add_argument("--split", default="test_series")
    parser.add_argument("--study", default=None, help="Suffix of the StudyInstanceUID.")
    parser.add_argument("--slot", default=None, help="e.g. SAG_FLUID_FS")
    parser.add_argument("--out", default="out/steps")
    args = parser.parse_args()

    root = Path(args.data_root)
    config = Config()
    headers = annotate(walk(root, args.split))
    if headers.empty:
        raise SystemExit(f"no series under {root / args.split}")

    csv = root / f"{args.split.replace('_series', '')}_series.csv"
    table = pd.read_csv(csv)
    plane_map = dict(zip(table.SeriesInstanceUID, table.Anatomical_Plane))
    headers["plane"] = headers["SeriesInstanceUID"].map(plane_map)

    slots = pick_slots(headers, plane_map, config)
    sides, _ = laterality_of(headers, config)

    candidates = [(s, n) for s, filled in slots.items() for n in filled
                  if (not args.study or s.endswith(args.study))
                  and (not args.slot or n == args.slot)]
    if not candidates:
        raise SystemExit("no (study, slot) matches those filters")

    study, slot = candidates[0]
    record = slots[study][slot]
    record["plane"] = plane_map.get(record["SeriesInstanceUID"], "")
    side = sides.get(study)

    # Refuse to draw a preprocessing that is not the one that runs.
    viz.verify_against_read_slot(record, config)

    print(f"{slot}  {record['SeriesDescription']!r}  {record['n_slices']} slices  "
          f"{record['px']:.3f} mm/px  side {side or 'unresolved'}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{study[-10:]}_{slot}"
    for name, figure in (("stack", viz.figure_stack(record, config)),
                         ("steps", viz.figure_steps(record, config, side)),
                         ("channels", viz.figure_channels(record, config, side))):
        path = out / f"{stem}_{name}.png"
        figure.savefig(path, dpi=110, bbox_inches="tight")
        print(f"  wrote {path}")


if __name__ == "__main__":
    main()
