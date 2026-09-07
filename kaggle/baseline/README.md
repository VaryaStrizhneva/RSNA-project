# Baseline reproduction

Runs pilkwang's public notebook **unmodified**, from our pipeline, to establish a
reference we can measure changes against.

There is no notebook here. [`notebook-source.txt`](notebook-source.txt) points at the
vendored copy in
[`../../notebooks/external/rsna-knee-baseline-v1/`](../../notebooks/external/rsna-knee-baseline-v1/),
which stays byte-identical to what its author published. The push script reads it
from there and stages a copy. **The day we change anything, we put our own
`notebook.ipynb` here and delete the pointer** — owning a copy starts when editing
starts, not before.

## Two experiments, one source

The notebook has both paths and picks between them at run time with
`find_weights()`: with a weights package mounted it runs inference and returns;
without one it trains for up to 8 hours, then infers.

```bash
# 2b — inference from the published weights. Expect 0.891.
python -m scripts.kaggle_push --kernel-dir kaggle/baseline

# 2c — same code, same labels, weights detached: it trains. Hours.
python -m scripts.kaggle_push --kernel-dir kaggle/baseline --no-weights
```

`--no-weights` drops `pilkwang/rsna-knee-weights` from `dataset_sources` while
staging. One variable differs between the two runs, and it is visible in the command
that produced them.

## What each result would mean

| Run | Expected | If it differs |
|---|---|---|
| 2b | 0.891 | The published weights or their inference path do not behave as we think — everything downstream is built on sand |
| 2c | unknown | This is the number that matters: what *their training recipe*, reproduced by us, is worth. Every later change is measured against it, not against 2b |

2c is the real baseline. 2b only proves we can run their inference.

## Attribution

The code is pilkwang's, run in a private kernel. Nothing is republished. If any of it
survives into a submission we keep, it must be credited — see
[`../../docs/references.md`](../../docs/references.md).

## Why `machine_shape` is pinned

```json
"machine_shape": "NvidiaTeslaT4"
```

Without it Kaggle may assign a **P100**, whose compute capability (sm_60) the
installed PyTorch no longer ships kernels for. The run then dies at the first CUDA
allocation:

```
Tesla P100-PCIE-16GB with CUDA capability sm_60 is not compatible
torch.AcceleratorError: CUDA error: no kernel image is available
```

It failed this way on the first attempt. The notebook's `try/except` caught it and
left its 0.5 fallback behind, so the kernel reported **COMPLETE** and produced a
valid-looking `submission.csv` full of 0.5. Submitting it would have scored 0.500.

**`COMPLETE` does not mean the run did what it was meant to.** Read the log, or check
that the predictions are not all identical, before submitting anything.
