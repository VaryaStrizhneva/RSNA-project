"""Two hosts and our experts, stacked in rank space.

    python -m scripts.kaggle_stack --w-host 0.70 --blend 0.50 --push

`kaggle_variant` patches our copy of the public pipeline. `kaggle_graft` appends our
experts to somebody else's scored notebook. This does both at once and adds the thing
neither needs: a **blend between two independent hosts**.

    d4 cells, `nocoat` applied        -> submission.csv, stashed as leg A
    the third party's cells, verbatim -> submission.csv, stashed as leg B
    one blend cell                    -> (1-w)*rank(A) + w*rank(B) per column
    our expert cell                   -> the eight columns we own, re-ordered

**Why stack instead of substituting weights.** The obvious move is to drop the four CoAt
readers out of d4 and put the better CoAtNet in their place. It cannot be done: that
model eats a different input entirely -- 96 slices in a 26/22/18/12/18 slot layout at
`IMG=384`, `CROP_MM=140`, 94 windows of three consecutive slices -- while d4's readers
consume pre-cached volumes from `raptor-knee-native384`, `maxspan` and `native384dense`.
There is no slot to drop the weights into. But there is no need: the third party's cell
preps its own studies end to end and writes its own submission, so the two legs compose
by appending rather than by surgery, and neither one's bytes are touched.

**`nocoat` is not optional here.** The four readers it removes are 56.7 % of the
pipeline's compute for a measured +0.003 (row #19), and the whole point of the stack is
that their slot is better filled from outside. Keeping them would cost 4.5 hours to
duplicate, badly, what leg B does well.

**AUC is rank-only, so the blend writes ranks and does not restore a marginal.** Both
legs are rank-percentiled per column before mixing; the output's scale is arbitrary
because every monotone transform of a column leaves its AUC identical. The expert cell
downstream still maps itself onto whatever marginal it finds, which is unchanged
behaviour.
"""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.kaggle_graft import KNOWN, code_hash                   # noqa: E402
from scripts.kaggle_variant import (EXPERT_BLEND, GRAFT_CELL,       # noqa: E402
                                    OWNER, PATCHES, SOURCE)

EXTERNAL = ROOT / "notebooks" / "external"

#: Stash leg A and hand the GPU back. d4's stages free their own models as they finish,
#: but torch keeps what it reserved, and leg B is about to load a 73 M-parameter CoAtNet
#: onto each device.
STASH_A = '''# ===================== leg A parked =====================
import shutil
import torch

shutil.copy2("/kaggle/working/submission.csv", "/kaggle/working/_leg_a.csv")
if torch.cuda.is_available():
    for _d in range(torch.cuda.device_count()):
        with torch.cuda.device(_d):
            torch.cuda.empty_cache()
print("[stack] leg A parked at _leg_a.csv; GPU released", flush=True)
'''

#: The blend. Merged on StudyInstanceUID, never on row order: the two legs discover the
#: test set independently and nothing promises they enumerate it the same way.
BLEND_CELL = '''# ===================== the two hosts, blended =====================
import numpy as np
import pandas as pd

W_HOST = {w_host}   # leg B's share. 1.0 would be leg B alone, which is a cheaper kernel.

a = pd.read_csv("/kaggle/working/_leg_a.csv", dtype={{"StudyInstanceUID": str}})
b = pd.read_csv("/kaggle/working/submission.csv", dtype={{"StudyInstanceUID": str}})
cols = [c for c in a.columns if c != "StudyInstanceUID"]
assert list(b.columns) == list(a.columns), f"column drift: {{list(b.columns)}}"
a = a.set_index("StudyInstanceUID")
b = b.set_index("StudyInstanceUID")
assert set(a.index) == set(b.index), "the two legs disagree on the test set"
b = b.reindex(a.index)

# A leg that silently did nothing would otherwise be invisible: the blend of a column
# with itself is itself, and the kernel would report success for a stack of one.
same = [c for c in cols if np.allclose(a[c].to_numpy(), b[c].to_numpy())]
assert len(same) < len(cols), "the two legs are identical on every column -- one did not run"
if same:
    print(f"[stack] identical on {{len(same)}} column(s): {{same}}", flush=True)


def rankpct(v):
    r = pd.Series(v).rank(method="average").to_numpy()
    return (r - 1.0) / max(len(r) - 1, 1)


out = pd.DataFrame({{c: (1.0 - W_HOST) * rankpct(a[c].to_numpy())
                       + W_HOST * rankpct(b[c].to_numpy()) for c in cols}}, index=a.index)
out = out.reset_index()[["StudyInstanceUID"] + cols]
assert np.isfinite(out[cols].to_numpy()).all()
out.to_csv("/kaggle/working/submission.csv", index=False)
print(f"[stack] blended {{len(out)}} studies x {{len(cols)}} columns, "
      f"leg A {{1 - W_HOST:.2f}} / leg B {{W_HOST:.2f}}", flush=True)
'''



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


def host_cells(slug: str) -> tuple[list[dict], dict]:
    """The third party's code cells and metadata, from the read-only copy in git.

    Read from `notebooks/external/` rather than pulled live: a scored stack should be
    built from bytes that are versioned alongside the thing that blends them. The hash is
    still checked, so a copy that drifted from what `KNOWN` records is a refusal.
    """

    folder = EXTERNAL / slug
    meta = json.loads((folder / "kernel-metadata.json").read_text())
    nb = json.loads((folder / meta["code_file"]).read_text())
    got = code_hash(nb)
    want = KNOWN.get(meta["id"])
    print(f"leg B {meta['id']}\n  code sha256 {got}")
    if want and got != want:
        raise SystemExit(f"  REFUSED: expected {want}")
    print(f"  {'known, unchanged' if want else 'not in KNOWN'}")
    return [c for c in nb["cells"] if c["cell_type"] == "code"], meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="rsna-knee-0949-anatomical-mirror",
                    help="folder under notebooks/external/ holding leg B")
    ap.add_argument("--w-host", type=float, required=True,
                    help="leg B's share of the host blend, in [0,1]")
    ap.add_argument("--blend", type=float, default=EXPERT_BLEND,
                    help="our experts' share of the columns they own")
    ap.add_argument("--slug", default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--push", action="store_true")
    args = ap.parse_args()
    if not 0.0 <= args.w_host <= 1.0 or not 0.0 <= args.blend <= 1.0:
        raise SystemExit("weights are shares, so both live in [0,1]")

    nb = json.loads((SOURCE / "notebook.ipynb").read_text())
    meta = json.loads((SOURCE / "kernel-metadata.json").read_text())
    PATCHES["nocoat"](nb)
    print(f"leg A {meta['id']} with `nocoat`: {len(nb['cells'])} cells")

    def cell(src: str) -> dict:
        return {"cell_type": "code", "execution_count": None, "metadata": {},
                "outputs": [], "source": [src]}

    b_cells, b_meta = host_cells(args.host)
    nb["cells"] = ([c for c in nb["cells"] if c["cell_type"] == "code"]
                   + [cell(STASH_A)] + b_cells
                   + [cell(BLEND_CELL.format(w_host=args.w_host)),
                      cell(GRAFT_CELL.format(blend=args.blend, mirror=True, jitter=False))])
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            compiles("".join(cell["source"]))

    tag = f"{int(round(args.w_host * 100)):02d}-{int(round(args.blend * 100)):02d}"
    slug = args.slug or f"rsna-knee-stack-{tag}"
    out = args.out or Path("/tmp") / slug
    out.mkdir(parents=True, exist_ok=True)
    (out / "notebook.ipynb").write_text(json.dumps(nb, indent=1) + "\n")

    # Every mount of both kernels is kept, the four CoAt weight datasets included: d4's
    # `rsna_asset_preflight` raises on a missing checkpoint whether or not a reader will
    # read it, and trimming mounts is a separate change with its own way of failing.
    meta["dataset_sources"] = list(dict.fromkeys(
        list(meta.get("dataset_sources", [])) + list(b_meta.get("dataset_sources", []))
        + [f"{OWNER}/rsna-src", f"{OWNER}/rsna-knee-experts"]))
    meta["id"] = f"{OWNER}/{slug}"
    # Derived from the slug, not from the weights: Kaggle resolves a title to its own
    # slug and refuses the push with a 409 when that disagrees with the id. Building the
    # title from `tag` sent "RSNA Knee stack 70-50" at a kernel called
    # `rsna-knee-stack-apex-70` and the conflict was with yesterday's kernel of that name.
    meta["title"] = slug.replace("rsna-knee-", "RSNA Knee ").replace("-", " ")
    meta["machine_shape"] = "NvidiaTeslaT4x2"
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=1) + "\n")

    print(f"\n{slug}: {len(nb['cells'])} cells, "
          f"leg A {1 - args.w_host:.2f} / leg B {args.w_host:.2f}, "
          f"experts {args.blend:.2f}")
    print(f"  {len(meta['dataset_sources'])} datasets -> {out}")
    if args.push:
        subprocess.run([sys.executable, "-m", "scripts.kaggle_push",
                        "--kernel-dir", str(out)], cwd=ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
