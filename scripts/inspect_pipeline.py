"""Watch each stage of the DICOM pipeline decide, on real studies.

The test suite proves the mechanics on synthetic files. This proves nothing — it
*shows*, on actual competition data, what every module in `rsna.dicom` concludes and
why. Run it to read the pipeline by watching it work rather than top-to-bottom in an
editor: each section below is one module.

    python -m scripts.inspect_pipeline
    python -m scripts.inspect_pipeline --split test_series --png out/slots

Needs DICOMs on disk under `data/raw/<split>/<study>/<series>/*.dcm`. They are not
versioned; see docs/data.md for how to fetch a few.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.rsna.config import Config
from src.rsna.dicom import (annotate, laterality_of, normalise_laterality,
                            order_slices, pick_slots, read_slot, sample_indices, walk)


def rule(title: str) -> None:
    print(f"\n\033[1m{title}\033[0m\n" + "-" * len(title))


def show_headers(headers: pd.DataFrame) -> pd.DataFrame:
    rule("1. walk + annotate — what one header per series says")
    print(f"{len(headers)} series across {headers['StudyInstanceUID'].nunique()} studies")

    annotated = annotate(headers)
    cols = ["SeriesDescription", "ScanOptions", "RepetitionTime", "EchoTime",
            "weight", "fluid", "fatsat", "px", "n_slices"]
    view = annotated[cols].copy()
    view["px"] = view["px"].round(4)
    print(view.to_string(index=False))

    print("\nRecovered against what the competition ships:")
    print("  the delivered Fluid_Sensitive and Fat_Suppression are equal on every")
    print("  training row, so they carry one axis. These are two.")
    print(f"  weight  : {dict(annotated['weight'].value_counts())}")
    print(f"  fatsat  : {dict(annotated['fatsat'].value_counts())}")
    return annotated


def show_laterality(headers: pd.DataFrame, config: Config) -> dict:
    rule("2. laterality — which knee, and from what evidence")
    sides, stats = laterality_of(headers, config)
    print(f"from the tag: {stats['from_tag']}   from geometry: {stats['from_geometry']}   "
          f"unresolved: {stats['unresolved']}   they disagree on: {stats['disagree']}")
    for study, side in sides.items():
        tagged = [str(x).strip().upper()[:1] for x in
                  headers[headers.StudyInstanceUID == study]["Laterality"].dropna()]
        print(f"  ...{study[-14:]}  ->  {side or 'UNRESOLVED'}   (tag: {set(tagged) or 'absent'})")
    print("\n  Five of the twelve targets are named for a side. An unresolved study is")
    print("  left unmirrored, which is a decision, not a default.")
    return sides


def show_slots(annotated: pd.DataFrame, config: Config) -> dict:
    rule("3. pick_slots — mapping a variable acquisition onto fixed positions")
    plane_map = _plane_map(annotated)
    slots = pick_slots(annotated, plane_map, config)

    for study, chosen in slots.items():
        print(f"\n  study ...{study[-14:]}")
        for name, plane, fluid, fatsat in config.slots:
            want = f"{plane}, fluid={fluid}, fatsat={fatsat}"
            if name in chosen:
                row = chosen[name]
                print(f"    {name:16s} <- {row['SeriesDescription']!r:26s} "
                      f"{row['n_slices']:3d} slices   [{want}]")
            else:
                print(f"    {name:16s} <- \033[2m(absent)\033[0m{' ' * 24}[{want}]")

    filled = sum(len(c) for c in slots.values())
    print(f"\n  {filled} of {len(slots) * config.n_slot} slots filled. An empty slot stays")
    print("  empty: the presence mask exists to say a sequence was not acquired.")
    return slots


def _plane_map(annotated: pd.DataFrame) -> dict:
    """Plane per series, from the competition CSV when present, else the description."""

    for candidate in (Path("data/raw/test_series.csv"), Path("data/raw/train_series.csv")):
        if candidate.is_file():
            table = pd.read_csv(candidate)
            mapping = dict(zip(table["SeriesInstanceUID"], table["Anatomical_Plane"]))
            if set(annotated["SeriesInstanceUID"]) & set(mapping):
                return mapping
    guess = {}
    for row in annotated.itertuples():
        text = str(row.SeriesDescription).lower()
        guess[row.SeriesInstanceUID] = ("Axial" if "tra" in text or "ax" in text else
                                        "Coronal" if "cor" in text else "Sagittal")
    return guess


def _selection(slots: dict, limit: int, only_study: str | None,
               only_slot: str | None) -> list[tuple[str, str, dict]]:
    """(study, slot, record) triples to walk through, honouring the filters."""

    chosen = []
    for study, filled in slots.items():
        if only_study and not study.endswith(only_study):
            continue
        for name, record in filled.items():
            if only_slot and name != only_slot:
                continue
            chosen.append((study, name, record))
    return chosen if limit <= 0 else chosen[:limit]


def show_ordering(slots: dict, config: Config, selection: list) -> None:
    rule("4. order_slices — file order is not physical order")
    from scipy import stats as sp

    for study, name, record in selection:
        ordered, resolved = order_slices(record["dir"], record["files"], config)
        rank = {f: i for i, f in enumerate(ordered)}
        by_file = [rank[f] for f in record["files"]]
        rho = sp.spearmanr(range(len(by_file)), by_file).statistic
        record["ordered"] = ordered
        print(f"  ...{study[-10:]} / {name:16s} {len(ordered):3d} slices  "
              f"geometry: {'yes' if resolved else 'NO — arbitrary order kept'}  "
              f"Spearman(file order, physical order) = {rho:+.3f}")
    print("\n  A file name here is a SOP Instance UID: unique by construction, ordered by")
    print("  nothing. Sorting by it fails silently — no exception, just noise.")


def show_pixels(selection: list, sides: dict, config: Config,
                png_dir: Path | None) -> None:
    rule("5. read_slot — constant physical scale, then resize")
    for study, name, record in selection:
        if True:
            if "ordered" not in record:
                record["ordered"], _ = order_slices(record["dir"], record["files"], config)
            n = len(record["ordered"])
            idx = sample_indices(n, config.group, config.band)
            spacing = record.get("px")
            crop = int(round(config.crop_mm / spacing)) if spacing else None

            image = read_slot(record, config)
            if image is None:
                print(f"  ...{study[-10:]} / {name}: nothing decoded")
                continue
            mirrored = normalise_laterality(image.numpy(), record.get("plane", ""),
                                            sides.get(study))
            print(f"  ...{study[-10:]} / {name:16s} {n:3d} slices, sampled {list(idx)}  "
                  f"{spacing:.3f} mm/px -> crop {crop}px -> {config.img}px  "
                  f"side {sides.get(study) or '?'}"
                  f"{'  (mirrored)' if not np.array_equal(image.numpy(), mirrored) else ''}")

            if png_dir is not None:
                _write_png(png_dir / f"{study[-10:]}_{name}.png", mirrored)

    print(f"\n  {config.crop_mm} mm is below the field of view of 99.6% of series and still")
    print("  contains the joint. PixelSpacing varies 3.4x across the corpus, so a")
    print("  fixed-pixel resize would change the physical scale by that factor.")


def _write_png(path: Path, volume: np.ndarray) -> None:
    """The slices side by side, exactly as the encoder receives them."""

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    n = volume.shape[0]
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3))
    for i, ax in enumerate(np.atleast_1d(axes)):
        ax.imshow(volume[i], cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"slice {i}", fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/raw")
    parser.add_argument("--split", default="test_series")
    parser.add_argument("--png", default=None, metavar="DIR",
                        help="Also write the slot images the model would receive.")
    parser.add_argument("--limit", type=int, default=4, metavar="N",
                        help="How many (study, slot) pairs to walk through in sections "
                             "4 and 5. 0 for all of them.")
    parser.add_argument("--study", default=None, metavar="SUFFIX",
                        help="Only this study — the last characters of its UID are enough.")
    parser.add_argument("--slot", default=None, metavar="NAME",
                        help="Only this slot, e.g. SAG_FLUID_FS.")
    args = parser.parse_args()

    config = Config()
    headers = walk(Path(args.data_root), args.split)
    if headers.empty or "SeriesDescription" not in headers:
        raise SystemExit(
            f"no series under {args.data_root}/{args.split}. See docs/data.md.")

    annotated = show_headers(headers)
    sides = show_laterality(annotated, config)
    plane_map = _plane_map(annotated)
    annotated["plane"] = annotated["SeriesInstanceUID"].map(plane_map)
    slots = show_slots(annotated, config)
    for chosen in slots.values():
        for name, record in chosen.items():
            record["plane"] = plane_map.get(record["SeriesInstanceUID"], "")
    selection = _selection(slots, args.limit, args.study, args.slot)
    if not selection:
        raise SystemExit("no (study, slot) pair matches those filters")
    show_ordering(slots, config, selection)
    show_pixels(selection, sides, config, Path(args.png) if args.png else None)


if __name__ == "__main__":
    main()
