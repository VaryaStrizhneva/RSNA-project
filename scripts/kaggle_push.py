"""Push a submission notebook to Kaggle, stamped with the commit that produced it.

Kaggle records a leaderboard score against a *kernel version*. Git records what the
code was. Neither knows about the other, so this script prepends a cell that prints
the commit into the run's own log — provenance without anyone having to remember.

Notebooks know nothing about this. The stamp cell is added to a staged copy, so the
same mechanism works for our own notebook and for a vendored third-party one that
must stay byte-identical to what its author published.

    python -m scripts.kaggle_push
    python -m scripts.kaggle_push --kernel-dir kaggle/baseline
    python -m scripts.kaggle_push --kernel-dir kaggle/baseline --no-weights
    python -m scripts.kaggle_push --labels --bootstrap-labels
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

DATASET_DIR = Path("kaggle/datasets/labels")
LABEL_FILE = DATASET_DIR / "report_labels_blend.csv"

#: A kernel directory may name its notebook here instead of holding a copy of it,
#: so that a vendored third-party notebook stays in one place and unedited.
SOURCE_POINTER = "notebook-source.txt"

#: Datasets stripped by --no-weights. Removing the weights package is what makes the
#: baseline notebook take its training path instead of its inference path.
WEIGHT_DATASETS = ("pilkwang/rsna-knee-weights",)


def run(cmd: list[str], dry_run: bool = False) -> None:
    print("$ " + " ".join(str(c) for c in cmd), flush=True)
    if dry_run:
        return
    if subprocess.run(cmd).returncode != 0:
        raise SystemExit("command failed")


def git_stamp() -> str:
    """Short commit, marked dirty when the tree does not match it.

    A stamp claiming a clean commit while uncommitted edits were pushed would be
    worse than no stamp at all.
    """

    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "no-git"
    dirty = subprocess.run(["git", "diff", "--quiet"]).returncode != 0
    return f"{sha}-dirty" if dirty else sha


def stamp_cell(stamp: str) -> dict:
    source = (
        "# Prepended by scripts/kaggle_push.py. Ties this run to the commit that\n"
        "# produced it: Kaggle knows kernel versions, git knows code, this joins them.\n"
        f'print("run stamp: {stamp}")\n'
    )
    return {
        "cell_type": "code",
        "metadata": {},
        "outputs": [],
        "execution_count": None,
        "source": source.splitlines(keepends=True),
    }


def resolve_notebook(kernel_dir: Path, code_file: str) -> Path:
    """Where the notebook actually lives.

    Either beside the metadata, or wherever `notebook-source.txt` points — which is
    how a vendored notebook is pushed without being copied into the kernel folder.
    """

    pointer = kernel_dir / SOURCE_POINTER
    if pointer.is_file():
        target = Path(pointer.read_text(encoding="utf-8").strip())
        if not target.is_file():
            raise SystemExit(f"{pointer} points at {target}, which does not exist")
        return target

    local = kernel_dir / code_file
    if not local.is_file():
        raise SystemExit(f"{kernel_dir} holds neither {code_file} nor {SOURCE_POINTER}")
    return local


def stage(kernel_dir: Path, stamp: str, no_weights: bool) -> Path:
    """Build the directory that is actually pushed, leaving the repo untouched."""

    meta = json.loads((kernel_dir / "kernel-metadata.json").read_text(encoding="utf-8"))
    notebook_path = resolve_notebook(kernel_dir, meta["code_file"])

    if no_weights:
        kept = [d for d in meta.get("dataset_sources", []) if d not in WEIGHT_DATASETS]
        dropped = set(meta.get("dataset_sources", [])) - set(kept)
        if not dropped:
            print("note: --no-weights removed nothing; no weights dataset was attached")
        else:
            print(f"--no-weights: dropped {', '.join(sorted(dropped))}")
        meta["dataset_sources"] = kept

    staging = Path(tempfile.mkdtemp(prefix="rsna-kernel-"))
    (staging / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")

    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    notebook["cells"].insert(0, stamp_cell(stamp))
    (staging / meta["code_file"]).write_text(
        json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return staging


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel-dir", default="kaggle/submit", type=Path)
    parser.add_argument("--no-weights", action="store_true",
                        help="Detach the weights package, so the baseline trains "
                             "instead of running inference from published weights.")
    parser.add_argument("--labels", action="store_true",
                        help="Regenerate and publish the label dataset too.")
    parser.add_argument("--blend", default="default")
    parser.add_argument("--bootstrap-labels", action="store_true",
                        help="Create the label dataset instead of versioning it. Once only.")
    parser.add_argument("--message", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    stamp = git_stamp()
    print(f"commit stamp: {stamp}\n")
    if stamp.endswith("-dirty"):
        print("WARNING: the working tree is dirty, so this run cannot be reproduced\n"
              "         from a commit alone. Commit first if the result matters.\n")

    if args.labels:
        message = args.message or f"blend={args.blend} @ {stamp}"
        run([sys.executable, "-m", "scripts.blend_labels",
             "--write", args.blend, "--out", str(LABEL_FILE)], args.dry_run)
        verb = "create" if args.bootstrap_labels else "version"
        cmd = ["kaggle", "datasets", verb, "-p", str(DATASET_DIR), "--dir-mode", "zip"]
        if not args.bootstrap_labels:
            cmd += ["-m", message]
        run(cmd, args.dry_run)

    staging = stage(args.kernel_dir, stamp, args.no_weights)
    try:
        run(["kaggle", "kernels", "push", "-p", str(staging)], args.dry_run)
    finally:
        # Always: a dry run that left directories behind would accumulate them in
        # /tmp for no benefit, since it prints everything it would have pushed.
        shutil.rmtree(staging, ignore_errors=True)

    kernel_id = json.loads(
        (args.kernel_dir / "kernel-metadata.json").read_text(encoding="utf-8")
    )["id"]
    print(f"\nPushed {stamp}. Run: https://www.kaggle.com/code/{kernel_id}")
    print("Submit from that page once it completes, then record the score in "
          "docs/experiments.md.")


if __name__ == "__main__":
    main()
