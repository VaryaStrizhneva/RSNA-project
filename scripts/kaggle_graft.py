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
    "heliosli/blend-mine-in-private-lb-luck-to-all-upvoters":
        "f359aeecb1dba916d65ecc6b0c31909c8524a88b89c5b7a745701d5cb30a3d6d",
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
    ap.add_argument("--push", action="store_true")
    args = ap.parse_args()

    work = Path(tempfile.mkdtemp())
    nb, meta = pull(args.host, work / "host")
    got = code_hash(nb)
    want = KNOWN.get(args.host)
    print(f"host {args.host}")
    print(f"  code sha256 {got}")
    if want and got != want:
        raise SystemExit(f"  REFUSED: expected {want}\n"
                         f"  The host's scored source changed. Read it again before "
                         f"grafting; do not assume the recipe held.")
    print(f"  {'known, unchanged' if want else 'not in KNOWN -- add the hash above'}")

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
    if args.blend is not None and args.weights:
        raise SystemExit("--blend and --weights set the same thing; pass one")
    blend = WEIGHTS[args.weights] if args.weights else (
        args.blend if args.blend is not None else EXPERT_BLEND)
    if not args.bare:
        nb["cells"].append({"cell_type": "code", "execution_count": None,
                            "metadata": {}, "outputs": [],
                            "source": [GRAFT_CELL.format(blend=repr(blend))]})
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            compiles("".join(cell["source"]))

    tag = args.weights or f"b{int(round((args.blend or EXPERT_BLEND) * 100)):02d}"
    slug = args.slug or ("rsna-knee-" + args.host.split("/")[1]
                         .replace("rsna-knee-", "").replace("0949-", "")
                         + f"-{tag}")
    out = args.out or Path("/tmp") / slug
    out.mkdir(parents=True, exist_ok=True)
    (out / "notebook.ipynb").write_text(json.dumps(nb, indent=1) + "\n")

    # Their mounts, kept whole -- the weights the host loads are in there -- plus ours.
    ours = [] if args.bare else [f"{OWNER}/rsna-src", f"{OWNER}/{args.experts}"]
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

    print(f"\n{slug}: {host_cells} host cell(s)"
          f"{'' if args.bare else ' + 1 graft'}")
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
