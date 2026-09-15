# What we ran, and what it told us

One entry per experiment that was **trained**, submitted or not.
[`../docs/experiments.md`](../docs/experiments.md) is the other half of the record: one
row per **submission**, with the leaderboard score. A run lands here the day it finishes
and there the day it is submitted.

**No number on this page is a leaderboard number.** Everything below is measured against
one of two things, and they are not the same:

- **OOF AUC** — every study predicted once by the fold that did not train on it, scored
  against the pilkwang label table. Four thousand studies, but the table itself reaches
  only **0.867** against expert truth, so this metric has a ceiling and says nothing
  about the last third of the gap.
- **Expert AUC** — the 58 expert-labelled studies, held out of every fold, predicted by
  all five models and averaged. Real radiologist reads, so no ceiling — but 58 studies,
  some targets with nine positives. It ranks; it does not separate.

## The runs

| Experiment | Folds | Epochs | OOF AUC | Expert AUC | Output | Date |
|---|---|---|---|---|---|---|
| `window_reference` | 5 | 10 | **0.8330** | **0.8258** | `out/ref/` | 2026-09-15 |
| `window_60epochs` | fold 0 only | 60 | 0.8427 * | 0.8454 * | `out/window_60epochs/` | 2026-09-15 |
| `depth_compress` | never run | | | | | |

**\* Read the single-fold row carefully — it is not comparable to the row above it.**
With one fold, "out-of-fold" is that fold's own 870 held-out studies rather than the
whole corpus, and the expert score is **one model** where the reference's is the average
of five. A single 60-epoch model scoring 0.8454 against a five-model 10-epoch ensemble
at 0.8258 is a real signal, but the honest comparison is fold 0 to fold 0: **0.8454
against 0.8270**, both single models on the same 58 studies.

---

### `window_reference` — the yardstick

The published baseline, to the letter: pilkwang's table, `weights: confidence`, 10
epochs, batch 8. That configuration scored **0.891** on the leaderboard. It is not meant
to be improved; it is what the ideas are measured against.

```
out-of-fold macro AUC   0.8330   (95% 0.828-0.838)
per fold                0.8333 +/- 0.0121   f0:0.830 f1:0.815 f2:0.853 f3:0.831 f4:0.837
expert-label macro AUC  0.8258   (95% 0.787-0.861), 58 studies
```

Per target, the two metrics disagree in a way worth remembering. Against the label
table the best are Fracture 0.875, Synovitis 0.874, Baker's 0.873; against the experts
they are Effusion 0.973, Baker's 0.953, Medial OA 0.910. **Synovitis is second-best on
the table and third-worst against the experts (0.779)** — the model reproduces what the
LLM read in the report, and that matches poorly what a radiologist sees. `Lateral
Meniscus` is bad on both (0.788 / 0.639), which is a model problem, not a label problem.

The report raised **3 warnings and 5 infos, all the same thing**: every fold was still
improving when its tenth epoch ended, and the training loss was still falling. The
run diagnosed itself as undertrained, which is what `window_60epochs` went to check.

Cost: 1 h 25 for five folds, 68 s per epoch, 5.13 GiB of VRAM on a 44 GiB L40 — about
10% of the card.

### `window_60epochs`

`window_reference` with `epochs: 60` and nothing else. Fold 0 only, 69 minutes.

```
best epoch 25 (selected on holdout)   holdout 0.8427   gold 0.8454
reference, same fold, best epoch 10   holdout 0.8300   gold 0.8270
                                             +1.3 pt         +1.8 pt
```

Averaged five epochs at a time, to see past the noise:

| epochs | loss | holdout | gold |
|---|---|---|---|
| 1-5 | 0.4520 | 0.7402 | 0.7297 |
| 11-15 | 0.4113 | 0.8267 | 0.8239 |
| 21-25 | 0.3877 | 0.8360 | 0.8417 |
| **26-30** | 0.3756 | **0.8379** | **0.8492** |
| 31-40 | 0.3583 | 0.8360 | 0.8487 |
| 51-60 | 0.3347 | 0.8344 | 0.8471 |

Both metrics peak in the same window and decline slowly after, while the training loss
keeps falling from 0.376 to 0.335. That is overfitting with nothing bought.

Its own report (`out/window_60epochs/report.html`) raises **0 warnings**, where the
reference raised three. The undertrained diagnosis is gone, which is the point of the
run. Per target it is strongest on Medial OA 0.873, Synovitis 0.872 and Baker's 0.870,
weakest on MCL 0.807, PF OA 0.807 and Contusion 0.826 — a flatter spread than the
reference, whose worst target sat at 0.788.

The package was written while the experiment was still called `long`; the id, the
filename, the manifest note and `history.json` have since been renamed by hand, and the
fingerprint was re-verified after the edit. Only `train-f0.log` still says `long`, and
it should: it is the record of what actually ran.

### `depth_compress` — never run

All twelve cached slices at once, mixed to three channels by a learned projection.

**Buys:** one encoder pass per slot instead of ten — most of the inference budget back —
and training sees exactly what inference sees.

**Costs:** the stack augmentation and the test-time averaging, both free in the window
scheme. If it underperforms, look there first; the answer is probably band jitter rather
than abandoning the stem.

Ported from `bend-the-knee-to-the-dinosaurs`, where it sits behind a switch whose
default is off — one competitor's idea, tried, not settled practice. See
[`../docs/references.md`](../docs/references.md).

---

## What we think we know

1. **Ten epochs constrains the model.** The reference is undertrained, on its own
   report's diagnosis and on the 60-epoch run's evidence. Any experiment that means to
   measure something else should not be run at 10 epochs, or the epoch budget is the
   thing being measured.

2. **The plateau is around 26-30 epochs**, on both metrics at once. Sixty is too many,
   but not disastrously — the decline from peak to epochs 51-60 is only ~0.3 point.

3. **`epochs` is not a duration.** It sets `total_steps` on the OneCycle schedule, so
   sixty epochs is not ten epochs continued: the peak moves from epoch 1.5 to epoch 9
   and the whole cycle restretches. Comparing two epoch counts compares two schedules.

4. **"Best epoch is the last epoch" means almost nothing.** OneCycle drives the learning
   rate to 0.0004% of its peak by the end, so the model always finishes at its most
   settled point. The reference flagged this on three folds; it is the schedule, not the
   model.

5. **The last epoch and a half of any OneCycle run barely learn.** At 10 epochs the
   final epoch starts at 3.4% of peak learning rate. That is ~10% of the budget spent
   moving almost nothing — the cheapest thing to fix is the floor, not the curve shape.

6. **The expert score is too noisy to select on.** It moved 0.8422 → 0.8581 → 0.8470
   over four consecutive epochs. A peak on one epoch is not a result. Selection reads
   the holdout alone, deliberately, and that is right even though the holdout is capped.

7. **Do not run experiments side by side on one GPU — run them one after another.**
   Three runs launched together took **238 s an epoch each**, against **49 s** for the
   same run alone. In aggregate that is 78 s per epoch delivered against 49 s: a 60%
   loss. Pure queueing would have cost 3 × 113 ms a step; the observed step was 548 ms,
   so 209 ms of every step went to nothing but switching between CUDA contexts. Without
   MPS a GPU runs one context at a time, and taking turns is not free.

8. **Free VRAM is not spare capacity.** Those three runs used 12.9 of 44.3 GiB and the
   card was still the bottleneck. Memory says how many models you can *hold*; it says
   nothing about how many you can *run*. The measurement to quote about headroom is
   seconds per epoch, never gigabytes.

9. **The training loop ships four times the pixels it uses.** `loop.py` copies the whole
   cached stack to the device and only then slices the window: 62 MiB crossed per step
   where 15 would do, unpinned and synchronous, with the GPU idle throughout. Slicing
   before the transfer would fix it — conditionally, since `stem: compress` really does
   consume all twelve slices. Not measured, not done.

## What we do not know

- **How any of this maps to the leaderboard.** Nothing trained on the full corpus has
  been submitted. The only submission so far came from an 80-study run and scored 0.677.
  Until a reference submission exists, the scale between our internal numbers and the
  public LB is guesswork.
- **Whether any weighting beats `uniform`** on the full corpus. `weights: confidence` is
  used everywhere because the published baseline uses it, not because we measured it.
- **What `depth_compress` does.** Never run.

## The obvious next runs

| | Why |
|---|---|
| `window_30epochs`, 5 folds | lands on the measured plateau, and gives the OneCycle a sane shape — peak at epoch 4.5 instead of 1.5 or 9 |
| submit `window_reference` | the only way to calibrate everything above against a leaderboard number |
| `dinov2-base` | 86M parameters against 22M, and the same pixels, so the cache is reused. Whether the L40 has the room is a question about seconds per epoch, not about the 5 GiB the small encoder occupies |
| layer-wise LR decay | the encoder has one learning rate for six unfrozen blocks; the standard recipe gives each block its own |
