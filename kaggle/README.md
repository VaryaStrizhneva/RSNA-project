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

## What `notebook.ipynb` is today, and what it must become

Today it is a **smoke test**: it writes 0.5 for every study and asserts the submission
schema. It scored 0.500, which proved the mechanics — mounts, filename, internet off.

It has to become a **generic inference notebook**: twenty lines that mount a weights
package and run it, with no model definition of its own.

```python
sys.path.insert(0, "/kaggle/input/rsna-src")     # internet is off; a mounted
from rsna.infer import predict_member, write_submission   # dataset is the only channel
from rsna.package import find_package, load_member
```

Every decision then comes from the package's own `manifest.json` — encoder, resolution,
slices, slots — so the same notebook runs *any* model we train without being edited.
That is the point: the notebook is a launcher, the library is the pipeline.

Two things it will need that it does not have yet: `dataset_sources` naming the weights
and the source package, and `"machine_shape": "NvidiaTeslaT4"` — without the pin, Kaggle
may hand out a P100, whose compute capability the installed PyTorch no longer supports.
See [`../docs/pipeline_pitfalls.md`](../docs/pipeline_pitfalls.md) §10.
