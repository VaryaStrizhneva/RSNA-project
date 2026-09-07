# Submission pipeline

Everything Kaggle runs is generated from this repo and pushed with the CLI.
**Nothing is edited in the Kaggle UI** — the next push overwrites it, and a browser
edit is invisible to the other person.

```
kaggle/
  submit/
    kernel-metadata.json   what to mount, GPU, internet off
    notebook.ipynb         the submission notebook
  datasets/labels/
    dataset-metadata.json  our label table, for when a run needs it
    report_labels_blend.csv    generated, gitignored
```

## Pushing

```bash
python -m scripts.kaggle_push            # notebook only
python -m scripts.kaggle_push --dry-run  # print the commands, run nothing
```

Then submit from <https://www.kaggle.com/code/mathysgouverneur/rsna-knee-submit>.
The CLI cannot submit a code-competition notebook.

Record the result in [`../docs/experiments.md`](../docs/experiments.md) immediately.

## The commit stamp

Before pushing, the script copies the kernel to a temp directory and rewrites
`RUN_STAMP` in the notebook with the current commit (plus `-dirty` if the tree does
not match it). The repo copy is never modified.

The run therefore prints its own provenance into its Kaggle log. Kaggle records a
score against a *kernel version*; git records the code; this is the only thing
joining them.

## The label dataset

Left alone unless asked for, so a notebook change cannot silently republish data:

```bash
python -m scripts.kaggle_push --labels --bootstrap-labels   # first time: creates it
python -m scripts.kaggle_push --labels                      # thereafter
python -m scripts.kaggle_push --labels --blend steven_only
```

It exists because the scored notebook runs with **internet disabled**: mounted
Kaggle datasets are the only way our own files reach it. Model weights will arrive
the same way.

Its contract matches the public baseline's `find_label_table()` — any
`report_labels*.csv` under `/kaggle/input` carrying `StudyInstanceUID` and the twelve
targets. So the same dataset works with their notebook and with ours.

## What the notebook does today

Nothing but plumbing: 0.5 for every study, plus schema assertions. Nothing is
attached — no labels, no weights — so a failure can only be about the submission
mechanics themselves.

**Expected score: 0.500.** Once that lands, the pipeline is trusted and real work
can be added to the same notebook.
