"""What each stage *decided*, as text.

The figures show pixels; these show choices — which sequence filled which slot, where
the side came from, how many millimetres a crop covered. Both live in the package
rather than in a notebook, so the notebook holds no logic and cannot drift from what
runs.
"""

from __future__ import annotations


from ..config import Config


def plane_map_from(root, split: str, annotated=None) -> dict:
    """Series -> anatomical plane, from the competition CSV when it is there.

    Falls back to guessing from the description, which is only good enough for
    looking at a study that has no CSV beside it.
    """

    from pathlib import Path

    import pandas as pd

    csv = Path(root) / f"{split.replace('_series', '')}_series.csv"
    if csv.is_file():
        table = pd.read_csv(csv)
        return dict(zip(table["SeriesInstanceUID"], table["Anatomical_Plane"]))

    guess = {}
    for row in annotated.itertuples():
        text = str(row.SeriesDescription).lower()
        guess[row.SeriesInstanceUID] = ("Axial" if "tra" in text or "ax" in text else
                                        "Coronal" if "cor" in text else "Sagittal")
    return guess


def report_headers(annotated) -> None:
    """What one header per series says, and the two axes recovered from it."""

    columns = ["SeriesDescription", "ScanOptions", "RepetitionTime", "EchoTime",
               "weight", "fluid", "fatsat", "px", "n_slices"]
    view = annotated[columns].copy()
    view["px"] = view["px"].round(4)
    print(f"{len(annotated)} series across "
          f"{annotated['StudyInstanceUID'].nunique()} studies\n")
    print(view.to_string(index=False))
    print("\nThe competition ships Fluid_Sensitive and Fat_Suppression, equal on every")
    print("training row — one axis. These are two, recovered from the header:")
    print(f"  weight : {dict(annotated['weight'].value_counts())}")
    print(f"  fatsat : {dict(annotated['fatsat'].value_counts())}")


def report_laterality(annotated, config: Config) -> dict:
    """Which knee each study is, and on what evidence."""

    from ..dicom import laterality_of

    sides, stats = laterality_of(annotated, config)
    print(f"from the tag: {stats['from_tag']}   from geometry: {stats['from_geometry']}"
          f"   unresolved: {stats['unresolved']}   disagree: {stats['disagree']}\n")
    for study, side in sides.items():
        tagged = [str(x).strip().upper()[:1] for x in
                  annotated[annotated.StudyInstanceUID == study]["Laterality"].dropna()]
        print(f"  ...{study[-14:]}  ->  {side or 'UNRESOLVED'}   "
              f"(tag: {set(tagged) or 'absent'})")
    print("\nFive of the twelve targets are named for a side. An unresolved study is")
    print("left unmirrored, which is a decision rather than a default.")
    return sides


def report_slots(annotated, plane_map: dict, config: Config) -> dict:
    """Which series filled which slot, and which slots stayed empty."""

    from ..dicom import pick_slots

    slots = pick_slots(annotated, plane_map, config)
    for study, chosen in slots.items():
        acquired = annotated[annotated.StudyInstanceUID == study]
        print(f"\n  study ...{study[-14:]}  ({len(acquired)} series acquired)")
        for name, plane, fluid, fatsat in config.slots:
            want = f"{plane}, fluid={fluid}, fatsat={fatsat}"
            if name in chosen:
                row = chosen[name]
                print(f"    {name:16s} <- {row['SeriesDescription']!r:28s} "
                      f"{row['n_slices']:3d} slices   [{want}]")
            else:
                print(f"    {name:16s} <- (empty){' ' * 24}[{want}]")
        unused = set(acquired.SeriesDescription) - {
            chosen[k]["SeriesDescription"] for k in chosen}
        if unused:
            print(f"    discarded: {sorted(unused)}")

    filled = sum(len(c) for c in slots.values())
    print(f"\n  {filled} of {len(slots) * config.n_slot} positions filled "
          f"({len(slots)} studies x {config.n_slot} slots). An empty slot stays empty:")
    print("  the presence mask exists to say a sequence was not acquired.")
    return slots


def report_ordering(selection: list, config: Config) -> None:
    """How far the file order is from the physical order, per series."""

    from scipy import stats as sp

    from ..dicom import order_slices

    for study, name, record in selection:
        ordered, resolved = order_slices(record["dir"], record["files"], config)
        rank = {f: i for i, f in enumerate(ordered)}
        rho = sp.spearmanr(range(len(record["files"])),
                           [rank[f] for f in record["files"]]).statistic
        record["ordered"] = ordered
        print(f"  ...{study[-10:]} / {name:16s} {len(ordered):3d} slices  "
              f"geometry: {'yes' if resolved else 'NO - arbitrary order kept'}  "
              f"Spearman(file order, physical order) = {rho:+.3f}")
    print("\nA filename here is a SOP Instance UID: unique by construction, ordered by")
    print("nothing. Sorting by it fails silently - no exception, just noise.")


def report_pixels(selection: list, sides: dict, config: Config) -> None:
    """The physical scaling each series went through."""

    from ..dicom import sample_indices

    for study, name, record in selection:
        if "ordered" not in record:
            from ..dicom import order_slices
            record["ordered"], _ = order_slices(record["dir"], record["files"], config)
        n = len(record["ordered"])
        spacing = record.get("px")
        crop = int(round(config.crop_mm / spacing)) if spacing else None
        idx = [int(i) for i in sample_indices(n, config.slices, config.band)]
        print(f"  ...{study[-10:]} / {name:16s} {n:3d} slices, sampled {idx[:4]}...  "
              f"{spacing:.3f} mm/px -> crop {crop}px -> {config.img}px  "
              f"side {sides.get(study) or '?'}")
    print(f"\n{config.crop_mm:.0f} mm is below the field of view of 99.6% of series and")
    print("still contains the joint. PixelSpacing varies 3.4x across the corpus, so a")
    print("fixed-pixel resize would change the physical scale by that factor.")


def selection_of(slots: dict, limit: int = 4, study: str | None = None,
                 slot: str | None = None) -> list:
    """(study, slot, record) triples to walk through, honouring the filters."""

    chosen = []
    for uid, filled in slots.items():
        if study and not uid.endswith(study):
            continue
        for name, record in filled.items():
            if slot and name != slot:
                continue
            chosen.append((uid, name, record))
    return chosen if limit <= 0 else chosen[:limit]
