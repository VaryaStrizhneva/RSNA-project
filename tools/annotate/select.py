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

from tools.annotate.bundle import pick_sagittal          # noqa: E402
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
    rng = np.random.default_rng(args.seed)
    pool = pool.sample(frac=1.0, random_state=args.seed)

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
    untagged = len(picked) - int(picked.tagged.sum())
    print(f"side from the DICOM tag: {int(picked.tagged.sum())}/{len(picked)} — the other "
          f"{untagged} show no badge, so the annotator orients on the fibular head. "
          f"Nothing extra to enter: the click's own position in the stack recovers which "
          f"end was lateral (161/161 on the first bundle).")

    args.out.write_text("\n".join(chosen) + "\n")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
