# The Kaggle side

What Kaggle is told to run, kept in git instead of in a browser.

```
kaggle/
  submit/    the scored notebook — what Kaggle runs to produce a submission
  extract/   copies N studies into one archive, so a laptop can have real data
```

Each holds a `kernel-metadata.json` (the slug, what to mount, GPU, internet off) and
the notebook that runs.

`kaggle kernels push -p DIR` requires a directory holding both files, which is why this
exists as a folder at all. Pushing to the same `id` creates a **new version** of the
same kernel, never a new kernel.

## Pushing

```bash
python -m scripts.kaggle_push                              # kaggle/submit
python -m scripts.kaggle_push --kernel-dir kaggle/extract
python -m scripts.kaggle_push --dry-run
```

The script never modifies the repo: it copies the kernel to a temp directory, prepends a
cell printing the current commit, and pushes that. The run then states its own
provenance in its Kaggle log — the only thing joining a leaderboard score to a commit.

Then submit from the kernel's page (the CLI cannot), and record the result in
[`../docs/experiments.md`](../docs/experiments.md).

## What `notebook.ipynb` is

A **launcher**, not a pipeline. It mounts three things and calls one function:

| mount | what it is |
|---|---|
| `rsna-src` | our library, published as a dataset by `scripts/kaggle_dataset.py --source` |
| `rsna-knee-weights` | a package: `manifest.json` plus one `.pt` per fold |
| `metaresearch/dinov2` | the encoder, hosted by Kaggle — we publish no copy of our own |

Internet is off in a scored run, so a mounted dataset is the only channel. The notebook
locates each mount **by content** rather than by path, because Kaggle names a mount
after its dataset and a rename would otherwise break it in silence.

Every decision — resolution, slice count, slots, stem, encoder variant — comes from the
package's own manifest. Nothing about the model is written in the notebook, so it runs
*any* model we train without being edited. The chain itself is
[`rsna.infer.run_submission`](../src/rsna/infer/run.py), which `scripts/predict.py`
calls too: one implementation, so a leaderboard score is reproducible locally.

It carries **no `try/except`**. A run that swallows its failure writes the 0.5 fallback
and reports COMPLETE, which looks exactly like a model that learnt nothing — see
[`../docs/pipeline_pitfalls.md`](../docs/pipeline_pitfalls.md) §10, which is where that
lesson came from.

The metadata pins `"machine_shape": "NvidiaTeslaT4"`. Without it Kaggle may hand out a
P100, whose compute capability the installed PyTorch no longer supports.

## Rehearsing before pushing

The mounts can be faked locally — a directory per dataset under one root — and the
notebook's cells run against them. That catches a missing mount, a bad import or a
fingerprint mismatch in seconds instead of in a queued Kaggle run.
