# Pipeline pitfalls

> For *why* the pipeline is shaped the way it is, rather than what breaks it, read
> [`notebooks/preprocessing.ipynb`](../notebooks/preprocessing.ipynb).
>
> Traps that produce **wrong results without raising an error**. Each one was
> learned from a public notebook — see [`references.md`](references.md) for
> attribution — and each is worth re-verifying on our own data before we trust it.

---

## 1. Slice order is not file order

**The trap.** A series is a directory of `.dcm` files. The obvious move is to sort
by filename. The filename is the **SOP Instance UID**, assigned to be *unique*, not
*ordered* — so sorting by it yields a slice sequence uncorrelated with anatomy.

**Why it's dangerous.** Nothing raises. The model trains on shuffled anatomy and
merely underperforms, which is indistinguishable from "this problem is hard".

**The fix.** Sort geometrically: project `ImagePositionPatient` (0020,0032) onto the
slice normal derived from `ImageOrientationPatient` (0020,0037). Fall back to
`InstanceNumber` (0020,0013) when geometry is absent.

---

## 2. Left and right knees are not interchangeable

**The trap.** Five of the twelve targets are named for a side — `Medial Meniscus`,
`Lateral Meniscus`, `Medial OA`, `Lateral OA`, `MCL`. Medial and lateral are defined
relative to the body midline, so which side of the *image* they fall on depends on
which knee was scanned.

**Why it's dangerous.** Without normalisation, those five targets see medial and
lateral structures in the same image position roughly half the time each. They learn
noise, while the seven side-agnostic targets look fine — so the pipeline seems to
work.

**The fix.** Mirror every study onto a single convention (left knee). Coronal and
axial views mirror under a horizontal flip; sagittal stacks do not mirror the same
way. Read `Laterality` from the DICOM header.

---

## 3. Resize by millimetres, not by pixels

**The trap.** An `N × N` slice with spacing `s` mm/pixel covers `N·s` mm of anatomy.
Both `N` and `s` vary widely across this corpus, so resizing everything to a fixed
pixel size hands the encoder images whose **physical scale differs by a factor of
several**.

**The fix.** Resample to a target mm-per-pixel. And note the hard limit underneath:
a feature narrower than two pixels cannot survive sampling, which bounds how coarse
the resolution may go regardless of the encoder.

---

## 4. Identical reports leak across folds

**The trap.** Some reports are **byte-identical** across different studies —
templates read out for an unremarkable knee. Since our targets are derived from the
report, every study in such a group gets the same target vector.

**Why it's dangerous.** Split that group across a fold boundary and the model is
scored on a target it already saw. Validation goes up, the leaderboard does not.

**The fix.** Group-split on report text (hash the report), not on study ID.

---

## 5. The metric reads order only

**The trap.** Macro-AUC is invariant under any strictly increasing transform of the
scores for a given label. Calibration, thresholds and sigmoid temperature are worth
**exactly nothing**.

**The consequence.** Public notebooks submit **per-column percentile ranks** rather
than probabilities. It also means a systematic bias in our derived labels (uniformly
too generous, say) costs far less than it looks — non-systematic noise is what hurts.

---

## 6. Train offline, infer inside the 9 hours

**The trap.** Trying to train inside the scored notebook caps the model at whatever
one accelerator fits in the time limit.

**The fix.** Nothing requires the scored run to be the run that learned the weights.
Train elsewhere, publish the weights as a Kaggle dataset, attach it, and keep the
scored notebook **inference-only**. The genuinely un-precomputable part is reading
studies nobody has seen.

**Budget.** 32,400 s ÷ ~1,300 test studies ≈ **25 s per study**, everything included
(model load, DICOM decode, preprocessing, forward pass).

---

## 7. Decode once, not once per epoch

A study is roughly 150 files across its series. That is affordable once; it is not
affordable every epoch. Decode slot images a single time into a cache, then train
many times off the cache.

---

## 8. `Fluid_Sensitive` and `Fat_Suppression` carry one axis, not two

They agree on **every row** of `train_series.csv` (our own measurement, and
independently confirmed by pilkwang's notebook). But the official data description
warns they are *not necessarily equivalent for every case* — so do not write code
that assumes one stands in for the other at test time. See [`data.md`](data.md).

---

## 9. Joining the public label tables drops a study

`pilkwang/rsna-knee-llm-labels` has **4,406 rows for 4,407 training studies**. A
naive `merge` silently loses one. Check row counts after every join.

---

## 10. A Kaggle run can report COMPLETE while having failed

Two mechanisms combine to hide a failed run.

**Kaggle may assign a P100.** Its compute capability is sm_60, and the installed
PyTorch ships kernels only from sm_70 up, so the run dies at the first CUDA
allocation with `no kernel image is available`. Pin the accelerator in
`kernel-metadata.json`:

```json
"machine_shape": "NvidiaTeslaT4"
```

**The notebook writes a fallback submission first.** The public baseline calls
`write_benchmark_submission()` before anything else, so a crash still leaves a valid
`submission.csv` — full of 0.5. That is good engineering: a run that dies after the
expensive decode pass still scores something instead of nothing.

Together they produce a kernel marked **COMPLETE**, holding a well-formed submission
that would score exactly 0.500.

**Before submitting, check the predictions are not all identical.** Status is not
evidence, and neither is a file existing.

```bash
kaggle kernels output <owner>/<slug> -p /tmp/out
python -c "import pandas as pd,sys; d=pd.read_csv('/tmp/out/submission.csv'); \
  print('distinct values:', d.iloc[:,1:].stack().nunique())"
```

