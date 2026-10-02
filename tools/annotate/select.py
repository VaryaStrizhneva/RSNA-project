"""Choose the studies for a bundle that closes a measured gap, instead of drawing blind.

    python -m tools.annotate.select --scan /data/mgr/rsna-knee/study_scan.csv \
        --exclude /data/mgr/rsna-knee/bundle-v2/studies.json --total 300 -o studies-v3.txt

The first bundle was drawn at random under `--tagged-only`, and that filter turned out to
select a vendor: the `Laterality` tag is written by whole manufacturers, so 78.8 % of it is
Siemens against 45 % of the corpus, and GE — 16.5 % of the corpus — came to three studies.
It also caught no 3D series at all, against 12.8 % of the corpus.

Drawing another random sample would reproduce both gaps. So this picks deliberately, to a
target the union of the two bundles is measured against: **the corpus vendor mix**. What is
already annotated counts toward the target, which is why `--exclude` takes the old bundle
rather than ignoring it.

Two things are deliberately *not* filtered here:

* **Tagged or not.** The point of the exercise is to stop excluding untagged studies, not
  to prefer them. A study that comes with the tag keeps its badge and costs the annotator
  nothing; one that does not gets the declaration panel. Both are wanted.
* **The pathology.** A landmark model learns where the meniscus is, not whether it is torn.

A study counts as 3D only if the series the bundle would *actually render* is the deep one.
That needs `bundle.py --prefer-3d`: the default preference is PD without fat suppression,
and a 3D sagittal PD is almost always fat suppressed, so a study holding both a 320-slice
PD FS and a 30-slice PD returns the 30. Build the bundle with that flag or the 3D quota
here is met on paper and empty in fact.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from rsna.dicom.headers import annotate as annotate_headers  # noqa: E402
from rsna.dicom.headers import walk                          # noqa: E402
from rsna.dicom.laterality import centre_x                   # noqa: E402
from rsna.landmark.series import LANDMARKS, pick_sagittal    # noqa: E402
from tools.atlas.study import series_headers             # noqa: E402

#: The corpus mix, measured over a 200-study draw and recorded in `docs/pipeline_v2.md`.
CORPUS = {"SIEMENS": .450, "PHILIPS": .320, "GE": .165, "TOSHIBA": .045, "CANON": .015}

#: Share of corpus studies holding a sagittal series over 100 slices **[m]**.
THREE_D_SHARE = 0.128


def renders_3d(uid: str) -> bool:
    """Would the bundle actually show the deep series for this study?"""

    try:
        chosen = pick_sagittal(series_headers(uid), prefer_deep=True)
        return chosen is not None and int(chosen["n_slices"]) > 100
    except Exception:  # noqa: BLE001
        return False


def quotas(have: pd.Series, total: int) -> dict[str, int]:
    """How many of each vendor the new bundle needs so the union matches the corpus."""

    return {v: max(0, int(round(share * total)) - int(have.get(v, 0)))
            for v, share in CORPUS.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scan", required=True, type=Path)
    ap.add_argument("--dicom-root", default="/data/mgr/rsna-knee/extracted", type=Path,
                    help="where the headers are, for --min-margin")
    ap.add_argument("--exclude", type=Path, action="append", default=[],
                    help="studies.json of a bundle already built; repeatable")
    ap.add_argument("--total", type=int, default=300,
                    help="size of the union the vendor mix is measured against")
    ap.add_argument("--three-d", type=int, default=None,
                    help="how many 3D studies to force in (default: the corpus share of "
                         "--total, minus what the excluded bundles already hold)")
    ap.add_argument("--n", type=int, default=None,
                    help="cap the new bundle at this many studies. The vendor quotas are "
                         "then scaled down proportionally to what is still missing, so a "
                         "smaller bundle closes the same gaps in the same order rather "
                         "than a different set of them.")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("-o", "--out", required=True, type=Path)
    ap.add_argument("--min-margin", type=float, default=0.0,
                    help="how far from the midline an untagged study's centre must sit "
                         "before its geometric side is trusted, in millimetres; 0 keeps "
                         "everything. Measured on 150 tagged studies, the rule agrees "
                         "with the DICOM tag 140 times and eight of the ten "
                         "disagreements are within 31 mm of zero, so 40 buys most of "
                         "the safety. Use it for a point whose side the click cannot "
                         "recover — anything read on a coronal or axial picture.\n\n"
                         "Do NOT reach for `--tagged-only`, or for preferring tagged "
                         "studies inside the quota, to get the same safety. The tag is "
                         "written by whole manufacturers and, within one, by particular "
                         "protocols: 84.3 %% of Siemens studies carry it against 4.1 %% "
                         "of GE, and **every one of those 37 tagged GE studies lacks a "
                         "fat-suppressed coronal**, against 95.6 %% of GE overall. "
                         "Preferring the tag inside GE's quota was tried and produced a "
                         "draw where 26 %% of studies had the sequence the pathology is "
                         "read on. The margin filter has no such effect: 94.4 %% of "
                         "studies beyond 40 mm carry a fat-suppressed coronal against "
                         "95.5 %% of all of them, and that holds vendor by vendor.")
    ap.add_argument("--landmark", default="lat_centre",
                    help="which point the bundle is for. Only changes what is reported "
                         "here — the laterality note is meaningless for a point that "
                         "does not depend on which knee it is.")
    args = ap.parse_args()

    scan = pd.read_csv(args.scan)
    scan = scan[(scan.vendor != "ERR") & (scan.n_sag > 0)]

    already, have = set(), []
    for path in args.exclude:
        studies = json.loads(path.read_text())["studies"]
        already |= {s["study"] for s in studies}
        have += [s["study"] for s in studies]
    prior = scan.set_index("study").reindex(have)["vendor"].value_counts()
    print(f"already annotated: {len(already)} studies")
    print(prior.to_string(), "\n")

    pool = scan[~scan.study.isin(already)].copy()
    pool = pool.sample(frac=1.0, random_state=args.seed)
    if args.min_margin > 0:
        # Measured here rather than read off the scan's `median_x`, which is the x of the
        # image **corner**: that sits half a field of view from the centre — about 90 mm
        # in this corpus, and 68 mm from the centre in the median study — so thresholding
        # it is not a weaker version of this filter, it is a different quantity. It is
        # also the rule that put a left knee on the right and cost an annotation.
        # Over every series, because that is what `side_from_geometry` thresholds to
        # produce the side the annotator will be shown. Restricting it to the point's
        # own plane was tried: it is a cleaner number and it answers a question nobody
        # asked, since the badge it would qualify is decided on all of them.
        print("measuring how far each study sits from the midline — the centre, not the "
              "corner, over the same series the side rule reads")
        headers = annotate_headers(walk(Path(args.dicom_root), "train_series"))
        margin = {s: abs(x) for s, x in centre_x(headers).items()}
        pool["margin_mm"] = pool.study.map(margin)
        far = pool.margin_mm.fillna(0) >= args.min_margin
        dropped = int((~pool.tagged & ~far).sum())
        pool = pool[pool.tagged | far]
        print(f"{dropped} untagged studies sit within {args.min_margin:.0f} mm of the "
              f"midline, where the geometric rule is near a coin flip; left out rather "
              f"than guessed at\n")

    need = quotas(prior, args.total)
    if args.n:
        total_need = sum(need.values())
        if total_need > args.n:
            # Largest remainder, so the scaled quotas still sum to exactly --n.
            exact = {v: c * args.n / total_need for v, c in need.items()}
            need = {v: int(x) for v, x in exact.items()}
            for v, _ in sorted(exact.items(), key=lambda kv: kv[1] - int(kv[1]),
                               reverse=True)[:args.n - sum(need.values())]:
                need[v] += 1
    want_3d = args.three_d
    if want_3d is None:
        want_3d = max(0, int(round(THREE_D_SHARE * args.total)))
    print("vendor quota for the new bundle:",
          " ".join(f"{k} {v}" for k, v in need.items()), f"· 3D {want_3d}\n")

    chosen, by_vendor, n_3d = [], {v: 0 for v in CORPUS}, 0

    # 3D first: it is the scarcer constraint, and every 3D pick also spends a vendor slot.
    print("checking which 3D candidates the bundle would actually render as 3D…")
    cand = pool[pool.has_3d_pd & pool.vendor.isin(CORPUS)]
    for uid, vend in zip(cand.study, cand.vendor):
        if n_3d >= want_3d:
            break
        if by_vendor[vend] >= need[vend] or not renders_3d(uid):
            continue
        chosen.append(uid); by_vendor[vend] += 1; n_3d += 1
    print(f"  {n_3d} kept\n")

    for uid, vend in zip(pool.study, pool.vendor):
        if uid in chosen or vend not in CORPUS:
            continue
        if by_vendor[vend] < need[vend]:
            chosen.append(uid); by_vendor[vend] += 1

    picked = scan.set_index("study").loc[chosen]
    print(f"selected {len(chosen)} studies")
    print(pd.DataFrame({
        "new": picked.vendor.value_counts(),
        "quota": pd.Series(need),
        "already": prior,
    }).fillna(0).astype(int).to_string(), "\n")
    print(f"3D (as rendered): {n_3d}")
    print(f"side from the scanner's own record: {int(picked.tagged.sum())}/{len(picked)}"
          f" ({100 * picked.tagged.mean():.0f} %)")
    loose = picked[~picked.tagged]
    if len(loose) and "margin_mm" in pool.columns:
        m = pool.set_index("study").margin_mm.reindex(loose.index)
        print(f"the other {len(loose)} rest on the geometric rule, all at "
              f"{m.min():.0f} mm or more from the midline")
    elif len(loose):
        print(f"the other {len(loose)} rest on the geometric rule, at whatever distance "
              f"from the midline they happen to sit — pass --min-margin to bound it")
    cue = LANDMARKS.get(args.landmark, {}).get("side_cue", "stack-end")
    if cue == "stack-end":
        untagged = len(picked) - int(picked.tagged.sum())
        print(f"side from the DICOM tag: {int(picked.tagged.sum())}/{len(picked)} — the "
              f"other {untagged} show no badge, so the annotator orients on the fibular "
              f"head. Nothing extra to enter: the click's own position in the stack "
              f"recovers which end was lateral (161/161 on the first bundle).")
    elif cue == "image-side":
        untagged = len(picked) - int(picked.tagged.sum())
        print(f"side from the DICOM tag: {int(picked.tagged.sum())}/{len(picked)}. For "
              f"{args.landmark} the side decides which *edge of the picture* is medial, "
              f"and the click cannot recover it — so the {untagged} untagged ones lean "
              f"on the geometric rule, and the annotator checks against the fibula, "
              f"which only exists laterally.")
    else:
        print(f"laterality is not asked about for {args.landmark}: its stack runs the "
              f"same way on both knees and the point is on the midline of its own "
              f"joint, so tagged and untagged studies cost the annotator the same.")

    args.out.write_text("\n".join(chosen) + "\n")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
