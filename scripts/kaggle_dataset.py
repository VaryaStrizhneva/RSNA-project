"""Publish the library, or a weights package, as a Kaggle dataset.

The scored notebook runs with internet off. It cannot install, clone or download
anything, so a mounted dataset is the only way code and weights reach it. This stages
the files in a temp directory with the metadata Kaggle wants and pushes them.

    python -m scripts.kaggle_dataset --source                     # src/rsna
    python -m scripts.kaggle_dataset --package out/submit-window  # the weights
    python -m scripts.kaggle_dataset --source --dry-run

The first push of a slug creates the dataset; every later one adds a version, so the
notebook's `dataset_sources` never has to change. Both are private: a weights package
is ours, and a dataset is the one place where competition data could leak by accident
— nothing here reads `data/`, and it should stay that way.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

#: The CLI installed beside this interpreter rather than whatever is on PATH, so the
#: script works without the virtualenv activated. Same reasoning as `kaggle_push`.
KAGGLE = str(Path(sys.executable).with_name("kaggle"))

OWNER = "mathysgouverneur"
SOURCE_SLUG = "rsna-src"
WEIGHTS_SLUG = "rsna-knee-weights"


def owns(slug: str) -> bool:
    """Whether the account already has this dataset.

    Asked by listing our own datasets rather than by querying the slug: `datasets
    status` and `datasets files` both answer with an HTTP error printed to stdout and
    an exit code of 0, so neither can be tested. This one either lists the ref or does
    not — and `create` on an existing slug fails, as does `version` on a missing one,
    so the answer has to be right.
    """

    listing = subprocess.run([KAGGLE, "datasets", "list", "-m", "--page-size", "200"],
                             capture_output=True, text=True)
    return any(line.split()[:1] == [f"{OWNER}/{slug}"]
               for line in listing.stdout.splitlines() if line.strip())


def run(cmd: list[str], dry_run: bool = False) -> None:
    print("$ " + " ".join(str(c) for c in cmd), flush=True)
    if dry_run:
        return
    if subprocess.run(cmd).returncode != 0:
        raise SystemExit("command failed")


def stage_source(into: Path) -> str:
    """`src/rsna` at the dataset root, so the notebook can `import rsna`."""

    shutil.copytree(Path("src/rsna"), into / "rsna",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    files = list((into / "rsna").rglob("*.py"))
    print(f"{len(files)} module(s) staged")
    return SOURCE_SLUG


def stage_package(into: Path, package: Path) -> str:
    """A weights package, manifest included, at the dataset root."""

    manifest = json.loads((package / "manifest.json").read_text())
    for member in manifest["members"]:
        shutil.copy2(package / member["file"], into / member["file"])
    shutil.copy2(package / "manifest.json", into / "manifest.json")
    size = sum(f.stat().st_size for f in into.iterdir()) / 1e6
    print(f"{len(manifest['members'])} member(s), {size:.0f} MB staged")
    return WEIGHTS_SLUG


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="store_true",
                        help="Publish src/rsna as the code dataset.")
    parser.add_argument("--package", type=Path,
                        help="Publish this weights package.")
    parser.add_argument("--slug", default=None, help="Override the dataset slug.")
    parser.add_argument("--message", default="update", help="Version message.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if bool(args.source) == bool(args.package):
        raise SystemExit("pass exactly one of --source or --package")

    with tempfile.TemporaryDirectory() as tmp:
        staged = Path(tmp)
        slug = (stage_source(staged) if args.source
                else stage_package(staged, args.package))
        slug = args.slug or slug
        title = ("RSNA Knee source" if args.source else "RSNA Knee weights")

        (staged / "dataset-metadata.json").write_text(json.dumps({
            "title": title,
            "id": f"{OWNER}/{slug}",
            "licenses": [{"name": "unknown"}],
        }, indent=1) + "\n", encoding="utf-8")

        # -r zip: directories are *skipped* by default, and the library is seven
        #         subpackages deep, so without this the source dataset uploads empty.
        # -t:     without it Kaggle rewrites anything it reads as tabular into CSV,
        #         which would quietly reshape manifest.json.
        common = ["-p", str(staged), "-r", "zip", "-t"]
        if owns(slug):
            run([KAGGLE, "datasets", "version", *common,
                 "-m", args.message, "-d"], args.dry_run)
        else:
            run([KAGGLE, "datasets", "create", *common], args.dry_run)

    print(f"\nmounted at /kaggle/input/{slug} once Kaggle finishes processing it")


if __name__ == "__main__":
    main()
