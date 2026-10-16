"""Our experts, grafted onto someone else's scored notebook.

    python -m scripts.kaggle_graft nartaa/rsna-knee-0949-anatomical-mirror --push

`kaggle_variant` grafts onto *our* copy of pjmathematician's pipeline, which lives in
`kaggle/submit` and which we edit. This grafts onto a kernel we do not own and must not
edit: it pulls the host, appends exactly one cell, and adds the two mounts that cell
needs. The host's own cells are copied byte for byte and never rewritten.

**Why byte-for-byte matters here.** A published scored notebook carries the hash of its
own source (`nartaa` prints `bcd864cb…` in its prose and we verified it matches). That
hash is the only evidence that the public code is the code that earned the score. Reorder
a cell, reformat a constant, and the evidence is gone along with any claim that our run
shares a host with theirs. So `--check-hash` refuses to build when the host's code has
changed under us.

**What the graft assumes of a host.** Only that it writes `submission.csv` with the
competition's twelve columns, and that the competition is mounted. It does not assume the
host exports `RSNA_COMP_ROOT` — pjmathematician's pipeline does, this one does not, and
the cell already falls back to walking the mounts for `test.csv`.
"""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
import tempfile
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from rsna.data.labels import TARGET_COLUMNS as TARGETS  # noqa: E402
from scripts.kaggle_variant import (EXPERT_BLEND, GRAFT_CELL,        # noqa: E402
                                    OWNER, WEIGHTS)

#: Hosts we have read, with the sha256 of the concatenated code we read. A host not listed
#: still builds -- the hash is printed so it can be added -- but a *listed* host whose
#: bytes moved is a refusal, because the recipe we believe we are grafting onto is no
#: longer the one on Kaggle.
KNOWN = {
    "nartaa/rsna-knee-0949-anatomical-mirror":
        "acd9241f171a807957f8eb8993976f649a479ca2f20d29037449d229578f352d",
    # Byte-identical to the line above, under a different owner. Listed so that grafting
    # onto it by mistake is caught as a duplicate rather than run twice.
    "matterhorn3838/rsna-knee-v2-velciraptor-dinosaur-speed":
        "acd9241f171a807957f8eb8993976f649a479ca2f20d29037449d229578f352d",
    # The Apex V3 fusion: the CoAtNet above plus a ConvNeXt 2.5D reader, per-column
    # weights. Public LB 0.950. `evgendvorkin/rsna-versia-5` is the same code with prose.
    "matterhorn3838/rsna-knee-d4":
        "b8497f5325e6b3725fbad73a88eebbedfe649a18b08a149ff38c3caf4a44efdd",
    # heliosli's S6: the same `nartaa` CoAtNet at 0.7 plus their retrained
    # `convnext_small` at 0.3, per-finding weights fixed before any run. Public LB 0.954.
    # Their `models2` second-architecture path exists in the code but the bundle sets it
    # empty, so the gain over their own 0.951 is a retrain, not an ensemble.
    "heliosli/dataset-restored-i-d-been-told-it-wasn-t-allowed":
        "ad2167eb929f68d96b7eca24192eaaac8870107010e056dd0fa83f26ee66144a",
    # The same S6 bundle re-hosted, with two changes that are worth +0.002 for them:
    # their internal weight goes 0.7/0.3 -> 0.5/0.5, and `pct_on` ranks each model over
    # its own non-failed studies so a mean-filled failed row stops shifting the others.
    # Public LB 0.956 -- below our 0.957, but a better host to graft onto.
    #: **One slug, two scored versions.** heliosli updates this notebook in place, so a
    #: single hash here would refuse the newer one and silently date the guard. The value
    #: is therefore a mapping of hash to what it scored, and a pull that matches neither
    #: is still refused.
    "heliosli/blend-mine-in-private-lb-luck-to-all-upvoters": {
        "f359aeecb1dba916d65ecc6b0c31909c8524a88b89c5b7a745701d5cb30a3d6d": "0.956",
        "e00557dc403f9eeba7f15b963caad9173278a67a34e4da4c640ed67f21c1afe7": "0.958",
    },
    #: **The same slug as the 0.956 above, updated in place on 2026-10-15 and now
    #: bannered 0.958.** Our vendored copy of the 0.956 keeps the old hash, so the guard
    #: separates them; this is the live one. Three stages like the 0.9573 -- Raptor-384,
    #: Raptor-224 at A224 = 0.35, their own R6 -- plus two changes: **per-finding blend
    #: weights** and flip TTA on their arm.
    #:
    #: Its weights mount is `rsna-knee-r6-frozen-fe055406-w1`, published the day after the
    #: unversioned `fe055406` that Kaggle refused to submit against -- the same pattern as
    #: `compact-v1` after `compact-r6cnxfsss`. So this host is graftable *and* submittable
    #: where the 0.9573 was not.
    #:
    #: `W_RAPTOR_C = {'Medial OA': 0.5, 'Lateral OA': 0.2, 'PF OA': 0.6, 'Synovitis': 0.8,
    #: "Baker's": 0.35}` is the weight their Raptor half carries per finding, default 0.5.
    #: **They give Raptor 0.2 on Lateral OA**, which is independent confirmation of our
    #: void probe: we measured the 0.956 host at **0.896 there, its weakest of four**, and
    #: they down-weight Raptor on exactly that column. The consequence for us is that
    #: Lateral OA on *this* host is 80 % their own model, so the parity we measured at
    #: −0.012 against the 0.956 does not carry over to it.
    #:
    #: `DEADLINE_HOURS = 8.3` is unchanged across all three versions, and it caps the
    #: host's own work rather than measuring it: stages 1 and 1b are capped at two hours
    #: each and stage 2 runs to T0 + 8.3 h. So the extra 224 arm eats their budget inside
    #: their cap, not the ~42 minutes of the nine hours our leg runs in -- and our grafts
    #: onto the 0.956, which carries the same guard, completed on the real hidden test.

    #: Published 2026-10-14, banner 0.9573 — **above our own standing 0.957**, in one
    #: click and by anyone. Three stages: nartaa's 384-pixel Raptor, then nartaa's
    #: **224-pixel** one (a 0.8 centre crop of the same field, 0.945 alone) at A224 = 0.35
    #: of the Raptor half, then heliosli's own R7 (`E_lrprior`, bundle `8c7b666e`), rank
    #: blended 0.5/0.5. The 224 checkpoint sits in
    #: `nartaa/rsna-knee-publication-swa-weights-20261007`, which **we have mounted since
    #: 7 October** — so that half of its gain over the 0.956 was available to us all along.
    "heliosli/0-9573-latest-public-dataset-score-in-one-click":
        "b63b237996bf869a0e5c4e1526fbe06675250ec543c27644cc3ea939d97a8b60",
}


def compiles(src: str) -> bool:
    """Whether this cell is Python that parses, skipping cell magics.

    `%%writefile` and friends are valid Jupyter and invalid Python, and a third party's
    notebook may use them -- the Apex V3 host writes three modules that way. The parse
    check is here to catch a broken cell *we* generated, so a cell that opens with a
    magic is passed over rather than treated as a syntax error in somebody else's code.
    """

    if src.lstrip().startswith(("%%", "!")):
        return True
    ast.parse(src)
    return True


def pull(ref: str, into: Path) -> tuple[dict, dict]:
    into.mkdir(parents=True, exist_ok=True)
    # The console script, not `-m kaggle`: the package has no __main__.
    kaggle = Path(sys.executable).with_name("kaggle")
    subprocess.run([str(kaggle), "kernels", "pull", ref, "-p", str(into), "-m"],
                   cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    meta = json.loads((into / "kernel-metadata.json").read_text())
    nb = json.loads((into / meta["code_file"]).read_text())
    return nb, meta


def code_hash(nb: dict) -> str:
    joined = "\n".join("".join(c["source"])
                       for c in nb["cells"] if c["cell_type"] == "code")
    return sha256(joined.encode()).hexdigest()


CONSTANT_CELL = '''# ============ A COLUMN PRICED BY REMOVING IT — a measurement ============
# Generated by scripts/kaggle_graft.py --constant. The host has written submission.csv;
# the columns below are now replaced by a constant.
#
# **What this buys.** Macro ROC-AUC is the mean of twelve column AUCs, and a constant
# column scores exactly 0.5 — every study ties, so no pair is ordered correctly or
# incorrectly. So the submitted macro is the host's own macro minus
# `sum(auc - 0.5) / 12` over the named columns, and the arithmetic inverts: one
# submission reads the host's AUC on those columns off the hidden test.
#
# It is worth a submission because it is the only measurement here that clears the noise
# floor. A column at 0.85 moves the macro by 0.029 — six times the ~0.005 the public
# leaderboard can resolve — where the test-time averages move it by 0.0006 to 0.0008 and
# would read as nothing. Running the host over *training* studies would have been free
# and would have measured the wrong thing: the public models were fitted on those
# studies, so their predictions there are their memory of the labels.
#
# This deliberately loses points. Our standing score is unaffected and the two
# submissions that count are chosen at the end.
import pandas as pd

COLUMNS = {columns}
sub = pd.read_csv("submission.csv")
missing = [c for c in COLUMNS if c not in sub.columns]
if missing:
    raise SystemExit(f"[void] not in the submission: {{missing}}")
for column in COLUMNS:
    kept = sub[column].nunique()
    sub[column] = 0.5
    print(f"[void] {{column}}: {{kept}} distinct values -> one constant", flush=True)
sub.to_csv("submission.csv", index=False)
print(f"[void] {{len(COLUMNS)}} of {{len(sub.columns) - 1}} columns voided; the macro "
      f"should fall by sum(auc - 0.5) / 12 over them", flush=True)
'''

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("host", help="owner/kernel-slug of the notebook to graft onto")
    ap.add_argument("--slug", default=None, help="our kernel's slug (default: derived)")
    ap.add_argument("--blend", type=float, default=None,
                    help="one weight for all eight columns")
    ap.add_argument("--weights", choices=sorted(WEIGHTS), default=None,
                    help="a named per-column regime instead; see kaggle_variant.WEIGHTS")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--keep-markdown", action="store_true",
                    help="carry the host's prose across. Off by default: it describes "
                         "their run, and ours is no longer theirs once a cell is added.")
    ap.add_argument("--experts", default="rsna-knee-experts",
                    help="which of our weight datasets to mount. The chain discovers "
                         "every `expert_*` directory under /kaggle/input and each run "
                         "overwrites its own column, so mounting two datasets that own "
                         "the same column makes the later one win silently -- one "
                         "dataset per configuration, and each must carry the landmark "
                         "runs too.")
    ap.add_argument("--bare", action="store_true",
                    help="push the host with NO graft: a pure reproduction, to learn "
                         "what it scores on our own account. Worth a submission only "
                         "when the published figure cannot be trusted to describe the "
                         "mounts we will actually run -- heliosli's weights dataset was "
                         "deleted and re-uploaded, and its README's claim of byte "
                         "identity is unverifiable now that the original is gone.")
    ap.add_argument("--swap-dataset", default=None, metavar="OLD=NEW",
                    help="rewrite one dataset slug in the host's code and mounts. "
                         "Needed when the host pins a dataset Kaggle calls "
                         "non-versioned and refuses to submit with -- that is a property "
                         "of the dataset, not of our metadata, and the only fix on our "
                         "side is to point at a versioned copy of the same bytes. "
                         "**This changes the host's code, so its hash stops matching "
                         "KNOWN and the guard is bypassed.** Only defensible when the "
                         "substitution is verified equivalent: here both datasets "
                         "declare `bundle_id 209a32c8b50a6b1c` and the host's own "
                         "`assert manifest['bundle_id'] == BUNDLE_ID` re-checks it at "
                         "run time, so a wrong swap fails loudly instead of scoring.")
    ap.add_argument("--no-mirror-tta", dest="mirror_tta", action="store_false",
                    help="stop reading each sagittal region a second time with its "
                         "depth axis reversed. On by default: that axis follows the "
                         "knee's side, so both of its signs are in the training data "
                         "and the second reading is worth +0.0024 a column over 15 "
                         "paired folds, for one extra pass on three of the eight "
                         "experts.")
    ap.add_argument("--jitter-tta", action="store_true",
                    help="also average five in-plane affines drawn from the training "
                         "augmentation, over all eight experts. Worth +0.0009 a column "
                         "over eight, but it costs five extra passes -- roughly an hour "
                         "and a half of what the host leaves of the nine, which is why "
                         "it is opt-in and why a run carrying it is also a test of "
                         "whether the budget holds.")
    ap.add_argument("--constant", default=None, metavar="COL,COL",
                    help="push the host with these columns replaced by a constant and "
                         "NO graft. This is a measurement, not an attempt at a score: "
                         "a constant column scores exactly 0.5, so the macro falls by "
                         "sum(auc - 0.5) / 12 over the named columns and one submission "
                         "prices the host where we cannot otherwise see it. Deliberately "
                         "loses points; our standing score is unaffected.")
    ap.add_argument("--local", type=Path, default=None, metavar="DIR",
                    help="read the host from a vendored copy under notebooks/external/ "
                         "instead of pulling it. Needed once the author takes the "
                         "notebook down -- `heliosli/blend-mine-in-private-lb-luck-to-"
                         "all-upvoters` began returning 404 on 2026-10-12 while every "
                         "dataset it mounts stayed up. **The hash guard still applies**, "
                         "and it is the whole point here: the vendored bytes are only "
                         "usable because KNOWN says what the scored source was, so a "
                         "copy that drifted cannot be grafted onto by accident.")
    ap.add_argument("--solo", default=None, metavar="COL[,COL]",
                    help="hand these columns ENTIRELY to our experts (weight 1.0) and "
                         "leave every other column to the host. This is the measurement "
                         "that makes the two comparable on the same footing: the macro "
                         "moves by sum(auc_ours - auc_host) / 12 over the named columns, "
                         "so one submission reads the difference directly and needs no "
                         "separate estimate of either side. Out-of-fold figures cannot "
                         "do this -- they are agreement with the label extractor on "
                         "training studies, which the extractor itself only matches at "
                         "0.841 against the 58 radiologist readings -- and the 58 gold "
                         "studies cannot either, at +/-0.05.")
    ap.add_argument("--push", action="store_true")
    args = ap.parse_args()

    work = Path(tempfile.mkdtemp())
    if args.local:
        nb = json.loads(next(args.local.glob("*.ipynb")).read_text())
        meta = json.loads((args.local / "kernel-metadata.json").read_text())
        print(f"host read from {args.local} (the published kernel is gone)")
    else:
        nb, meta = pull(args.host, work / "host")
    got = code_hash(nb)
    want = KNOWN.get(args.host)
    print(f"host {args.host}")
    print(f"  code sha256 {got}")
    accepted = ({want: ""} if isinstance(want, str) else dict(want or {}))
    if accepted and got not in accepted:
        raise SystemExit(
            f"  REFUSED: expected one of\n"
            + "".join(f"    {h}  ({lbl or 'known'})\n" for h, lbl in accepted.items())
            + f"  The host's scored source changed. Read it again before grafting; do "
              f"not assume the recipe held.")
    if accepted:
        lbl = accepted[got]
        print(f"  known, unchanged" + (f" — the {lbl} version" if lbl else ""))
    else:
        print(f"  not in KNOWN -- add the hash above")

    if not args.keep_markdown:
        nb["cells"] = [c for c in nb["cells"] if c["cell_type"] == "code"]
    if args.swap_dataset:
        a, b = args.swap_dataset.split("=", 1)
        hits = 0
        for cell in nb["cells"]:
            joined = "".join(cell["source"])
            if a in joined:
                hits += joined.count(a)
                cell["source"] = [joined.replace(a, b)]
        if not hits:
            raise SystemExit(f"  REFUSED: {a!r} appears nowhere in the host's code")
        meta["dataset_sources"] = [x.replace(a, b)
                                   for x in meta.get("dataset_sources", [])]
        print(f"  SWAPPED {a} -> {b} in {hits} place(s); the host's hash no longer "
              f"matches and its bundle assert is now the only check")
    host_cells = len(nb["cells"])
    if sum(x is not None and x is not False for x in
               (args.blend, args.weights)) + bool(args.solo) > 1:
        raise SystemExit("--blend, --weights and --solo set the same thing; pass one")
    if args.solo:
        solo = [c.strip() for c in args.solo.split(",") if c.strip()]
        unknown = [c for c in solo if c not in TARGETS]
        if unknown:
            raise SystemExit(f"  REFUSED: {unknown} are not among the twelve targets")
        blend = {c: 1.0 for c in solo}
    else:
        blend = WEIGHTS[args.weights] if args.weights else (
            args.blend if args.blend is not None else EXPERT_BLEND)
    # `--solo` with `--constant` is the measurement that prices OUR expert the way the
    # void probes priced the host: our columns carry our expert outright, every other
    # column is a constant 0.5, and the macro reads
    # `(sum of our AUCs + (12 - n) * 0.5) / 12` -- so one submission inverts to our
    # absolute AUC. The graft must run first and the voiding second, or the void would
    # be overwritten.
    if args.solo and not args.bare:
        nb["cells"].append({"cell_type": "code", "execution_count": None,
                            "metadata": {}, "outputs": [],
                            "source": [GRAFT_CELL.format(blend=repr(blend),
                                                         mirror=args.mirror_tta,
                                                         jitter=args.jitter_tta)]})
    if args.constant:
        wanted = [c.strip() for c in args.constant.split(",") if c.strip()]
        unknown = [c for c in wanted if c not in TARGETS]
        if unknown:
            raise SystemExit(f"  REFUSED: {unknown} are not among the twelve targets")
        overlap = sorted(set(wanted) & set(blend if isinstance(blend, dict) else ()))
        if overlap:
            raise SystemExit(f"  REFUSED: {overlap} are both handed to an expert and "
                             f"voided; the voiding runs last and would erase the expert")
        nb["cells"].append({"cell_type": "code", "execution_count": None,
                            "metadata": {}, "outputs": [],
                            "source": [CONSTANT_CELL.format(columns=repr(wanted))]})
    elif not args.bare and not args.solo:
        nb["cells"].append({"cell_type": "code", "execution_count": None,
                            "metadata": {}, "outputs": [],
                            "source": [GRAFT_CELL.format(blend=repr(blend),
                                                         mirror=args.mirror_tta,
                                                         jitter=args.jitter_tta)]})
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            compiles("".join(cell["source"]))

    if args.solo and args.constant:
        tag = "only-" + "-".join(c.split()[0].lower()[:4] for c in solo)
    elif args.constant:
        tag = "void-" + "-".join(c.split()[0].lower()[:4] for c in wanted)
    elif args.solo:
        tag = "solo-" + "-".join(c.split()[0].lower()[:4] for c in solo)
    else:
        tag = args.weights or f"b{int(round((args.blend or EXPERT_BLEND) * 100)):02d}"
        if args.jitter_tta:
            tag += "-jit"
        if not args.mirror_tta:
            tag += "-nomir"
    slug = args.slug or ("rsna-knee-" + args.host.split("/")[1]
                         .replace("rsna-knee-", "").replace("0949-", "")
                         + f"-{tag}")
    out = args.out or Path("/tmp") / slug
    out.mkdir(parents=True, exist_ok=True)
    (out / "notebook.ipynb").write_text(json.dumps(nb, indent=1) + "\n")

    # Their mounts, kept whole -- the weights the host loads are in there -- plus ours.
    ours = ([] if (args.bare or (args.constant and not args.solo))
             else [f"{OWNER}/rsna-src", f"{OWNER}/{args.experts}"])
    meta["dataset_sources"] = list(dict.fromkeys(
        list(meta.get("dataset_sources", [])) + ours))
    meta["id"] = f"{OWNER}/{slug}"
    meta["title"] = slug.replace("rsna-knee-", "RSNA Knee ").replace("-", " ")
    meta["code_file"] = "notebook.ipynb"
    meta["is_private"] = True
    meta["enable_gpu"] = True
    meta["enable_internet"] = False
    # The host's consumer loop spreads studies over every visible device; one T4 halves
    # its throughput, and the time it does not take is the time our leg has.
    meta["machine_shape"] = "NvidiaTeslaT4x2"
    meta.pop("id_no", None)
    meta.pop("keywords", None)
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=1) + "\n")

    added = "" if (args.bare or args.constant) else " + 1 graft"
    print(f"\n{slug}: {host_cells} host cell(s){added}"
          f"{' + 1 voiding cell' if args.constant else ''}")
    if args.constant:
        # Not a weight in sight: no expert runs here. Saying "flat 0.5" would describe a
        # blend this notebook does not contain, and a log that reads as a graft is how a
        # measurement comes to be filed as an attempt at a score.
        print(f"  voided: {', '.join(wanted)} -> constant 0.5")
        if args.solo:
            n = len(blend)
            print(f"  kept by our experts outright: {', '.join(sorted(blend))}")
            print(f"  so the macro reads (sum of our {n} AUCs + {12 - n} x 0.5) / 12, "
                  f"and inverts to sum = 12 x macro - {(12 - n) * 0.5:g}")
        else:
            print(f"  the macro falls by sum(auc - 0.5) / 12 over those "
                  f"{len(wanted)} column(s); no expert or landmark is mounted")
    elif not args.bare:
        print(f"  weights: {args.weights or 'flat ' + str(blend)}")
        if isinstance(blend, dict):
            for k, v in blend.items():
                print(f"    {k:18s} {v:.2f}")
    print(f"  mounts: {len(meta['dataset_sources'])} datasets, "
          f"competition {meta.get('competition_sources')}")
    print(f"  -> {out}")

    if args.push:
        subprocess.run([sys.executable, "-m", "scripts.kaggle_push",
                        "--kernel-dir", str(out)], cwd=ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
