# Notebooks

Exploration only. **No logic lives here** — the figures come from `rsna.viz`, the
pipeline from `rsna.dicom`. A notebook that grows its own implementation drifts from
what actually runs, and nobody notices until a score does not reproduce.

| Notebook | What it is for |
|---|---|
| `preprocessing.ipynb` | Watch a raw DICOM slice become the array the encoder receives |
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

## Same figures without Jupyter

```bash
python -m scripts.visualise_preprocessing --slot SAG_FLUID_FS --out out/steps
```
