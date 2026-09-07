"""Push the submission notebook to Kaggle, stamped with the commit that produced it.

Kaggle records a leaderboard score against a *kernel version*. Git records what the
code was. Neither knows about the other, so this script writes the commit into the
notebook itself before pushing: the run then prints its own provenance into its log,
and a score can be traced back to a commit without anyone having to remember.

The label table is a separate Kaggle dataset and is left alone unless asked for
(`--labels`), so a notebook-only change cannot silently republish data.

    python -m scripts.kaggle_push                    # notebook only
    python -m scripts.kaggle_push --labels           # also version the label dataset
    python -m scripts.kaggle_push --labels --bootstrap-labels   # first time: create it
    python -m scripts.kaggle_push --dry-run
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

DATASET_DIR = Path("kaggle/datasets/labels")
KERNEL_DIR = Path("kaggle/submit")
LABEL_FILE = DATASET_DIR / "report_labels_blend.csv"


def run(cmd: list[str], dry_run: bool = False) -> None:
    print("$ " + " ".join(str(c) for c in cmd), flush=True)
    if dry_run:
        return
    if subprocess.run(cmd).returncode != 0:
        raise SystemExit("command failed")


def git_stamp() -> str:
    """Short commit, marked dirty when the tree does not match it.

    A stamp that claimed a clean commit while uncommitted edits were being pushed
    would be worse than no stamp at all.
    """

    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "no-git"
    dirty = subprocess.run(["git", "diff", "--quiet"]).returncode != 0
    return f"{sha}-dirty" if dirty else sha


def staged_kernel(stamp: str) -> Path:
    """Copy the kernel next to a stamped notebook, so the repo copy stays untouched."""

    staging = Path(tempfile.mkdtemp(prefix="rsna-kernel-"))
    for name in ("kernel-metadata.json", "notebook.ipynb"):
        shutil.copy(KERNEL_DIR / name, staging / name)

    path = staging / "notebook.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    replaced = 0
    for cell in notebook["cells"]:
        source = "".join(cell["source"])
        if "RUN_STAMP" in source:
            cell["source"] = re.sub(
                r'RUN_STAMP = "[^"]*"', f'RUN_STAMP = "{stamp}"', source
            ).splitlines(keepends=True)
            replaced += 1
    if replaced != 1:
        raise SystemExit(f"expected exactly one RUN_STAMP cell, found {replaced}")

    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return staging


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", action="store_true",
                        help="Regenerate and publish the label dataset too.")
    parser.add_argument("--blend", default="default",
                        help="Which blend to publish; see scripts/blend_labels.py")
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

    staging = staged_kernel(stamp) if not args.dry_run else KERNEL_DIR
    run(["kaggle", "kernels", "push", "-p", str(staging)], args.dry_run)

    kernel_id = json.loads((KERNEL_DIR / "kernel-metadata.json").read_text())["id"]
    print(f"\nPushed {stamp}. Run: https://www.kaggle.com/code/{kernel_id}")
    print("Submit from that page once it completes, then record the score in "
          "docs/experiments.md.")


if __name__ == "__main__":
    main()
