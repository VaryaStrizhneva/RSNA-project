# Notebooks

Exploration only. **No logic lives here** — the figures come from `rsna.viz`, the
pipeline from `rsna.dicom`. A notebook that grows its own implementation drifts from
what actually runs, and nobody notices until a score does not reproduce.

| Notebook | What it is for |
|---|---|
| `preprocessing.ipynb` | **The preprocessing, end to end**: why each step exists, what each stage decides on real studies, and the pictures. Narrative, tables and figures in one place. |
| `inspect_metadata.ipynb` | The original metadata exploration |
| `external/` | Read-only copies of public notebooks. Never edited. |

## One-time setup per clone

Notebook outputs here render MRI slices from the competition. Committing them would
put competition data in the repository, which rule 2.4.b forbids — so a git filter
strips outputs on the way in:

```bash
git config filter.nbstrip.clean "$(pwd)/.venv/bin/python scripts/nbstrip.py"
```

Without this, `git add` of a notebook stores its images. Your working copy keeps them
either way; only the committed blob is stripped.

## No logic here

Every table and figure comes from `rsna.viz`, and `viz.verify_against_read_slot`
asserts the illustrated chain ends exactly where `read_slot` ends — so the notebook
cannot drift into describing a preprocessing nobody runs.
