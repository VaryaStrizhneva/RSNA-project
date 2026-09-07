"""Strip notebook outputs on the way into git. Reads stdin, writes stdout.

Notebook outputs here contain rendered MRI slices, and a committed notebook would
carry them as base64 inside the .ipynb — which is competition data made available to
whoever can read the repository. Rule 2.4.b forbids that, and "remember to clear
outputs before committing" is not a mechanism.

Configured as a git clean filter, so it runs on every commit whether or not anyone
remembers. The working copy keeps its outputs; only the committed blob is stripped.

One-time setup per clone (see notebooks/README.md):

    git config filter.nbstrip.clean "python scripts/nbstrip.py"
"""

from __future__ import annotations

import json
import sys


def main() -> int:
    try:
        notebook = json.load(sys.stdin)
    except (ValueError, UnicodeDecodeError):
        # Not JSON: pass it through untouched rather than corrupting a file git
        # happened to route here.
        sys.stdin.buffer.seek(0)
        sys.stdout.write(sys.stdin.read())
        return 0

    for cell in notebook.get("cells", []):
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
        cell.get("metadata", {}).pop("execution", None)

    notebook.get("metadata", {}).pop("widgets", None)
    json.dump(notebook, sys.stdout, indent=1, ensure_ascii=False)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
