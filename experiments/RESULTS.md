# What we ran, and what it told us

One entry per experiment that was **trained**, submitted or not.
[`../docs/experiments.md`](../docs/experiments.md) is the other half of the record: one
row per **submission**, with the leaderboard score. A run lands here the day it finishes
and there the day it is submitted.

**Only one number on this page is a leaderboard number**, and it is the last column of
the table below. Everything else is measured against one of two things, and neither is
on the leaderboard's scale:

- **OOF AUC** — every study predicted once by the fold that did not train on it, scored
  against the pilkwang label table. Four thousand studies, but the table itself reaches
  only **0.867** against expert truth, so this metric has a ceiling and says nothing
  about the last third of the gap.
- **Expert AUC** — the 58 expert-labelled studies, predicted by a model that did not
  train on them, scored against the radiologist reading. Real truth, so no ceiling — but
  58 studies, some targets with nine positives. It ranks; it does not separate.

Both are measured over the windows inference slides, since the run that introduced the
final aligned pass; earlier records were brought up to date with `scripts.rescore`
rather than retrained.

**The scale between them and the leaderboard is now known, and it is large.** The run
that scored **0.891** publicly reads **0.8443** out of fold and **0.8405** against the
experts. Roughly five points separate our best internal measurement from the public
score, in our favour. Every conclusion drawn here before 2026-09-16 — including a whole
day spent wondering what was missing to reach 0.891 — was drawn without knowing that.

## The runs

| Experiment | Folds | Epochs | Gold | OOF AUC | Expert AUC | Public LB |
|---|---|---|---|---|---|---|
| `window_reference` | 5 | 10 | held out | 0.8351 | 0.8265 | — |
| `window_60epochs` | fold 0 | 60 | held out | 0.8459 * | 0.8519 * | — |
| `window_30epochs` | fold 0 | 30 | held out | 0.8464 * | 0.8606 * | — |
| `window_30epochs_uniform` | fold 0 | 30 | held out | 0.8465 * | 0.8564 * | — |
| `window_30epochs_v4blend` | fold 0 | 30 | held out | **0.8560** * | 0.8591 * | — |
| **`window_30epochs_goldin`** | **5** | **30** | **in training** | **0.8443** | **0.8405** † | **0.891** |
| `depth_compress` | never run | | | | | |

Outputs are `out/<experiment>/`, except `window_reference`, which is `out/ref/`. All run
on 2026-09-15, except `window_30epochs_goldin`, which ran overnight into 2026-09-16.

**\* A single-fold row is not comparable to a five-fold row.** With one fold,
"out-of-fold" is that fold's own ~870 held-out studies rather than the whole corpus, and
the expert score comes from one model where a five-fold row averages five. Compare
single folds with single folds.

**† A different measurement again, and the one place the two designs cannot meet.**
Where the gold studies are *held out*, all five models predict them and the score is the
ensemble a submission carries. Where they are *in training*, only the fold that held a
given study out can score it honestly — so 0.8405 is one model per study, and the
ensemble that was actually submitted cannot be measured at all. It is a floor, not an
estimate: the submitted ensemble scored 0.891.

---

### `window_reference` — the yardstick, with one unintended departure

pilkwang's table, `weights: confidence`, 10 epochs, batch 8 — the published baseline's
configuration, which scored **0.891** on the leaderboard.

**But `out/ref` was trained with `--holdout-gold`, and the baseline is not.** Its §7 is
explicit: *"The annotated studies stay in training, at elevated weight, because they are
the only labels read from the images rather than from text."* This run therefore trained
on 59 studies fewer than the recipe it claims to reproduce, and those 59 carry
`gold_weight` 3.0 — the heaviest and only image-read labels in the corpus. Read it as a
yardstick for our own experiments, which share the same departure, and not as a
reproduction of 0.891. `window_30epochs_goldin` is the run that does not depart.

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

### `window_30epochs`, `_uniform`, `_v4blend` — three folds-of-one, run together

Same fold, same 30 epochs, one field apart each. Run simultaneously, which turned out to
cost more than running them in sequence — see point 7 below.

| | labels | weights | OOF * | expert * |
|---|---|---|---|---|
| `window_30epochs` | pilkwang | confidence | 0.8464 | 0.8606 |
| `..._uniform` | pilkwang | uniform | 0.8465 | 0.8564 |
| `..._v4blend` | v4_blend | assertedness | **0.8560** | 0.8591 |

**Weighting does nothing.** `confidence` and `uniform` differ by 0.0001 out of fold and
0.004 against the experts, despite a quarter of all label cells weighing under 0.5 in
one and exactly 1.0 in the other, and despite mean weights of 0.776 against 1.026 — a
32% difference in gradient magnitude. The README calls this *"the question worth
answering first"*. On this fold the answer is no.

**`v4blend` leads, and we cannot say why.** It changes the table *and* the formula
together, because `llm_labels_v4_blend` publishes no confidence column and
`weights: confidence` is a hard error on it. A bridge run — pilkwang with
`assertedness` — would have isolated the table; it was written and then dropped.

### `window_30epochs_goldin` — the first run to match the baseline

`window_30epochs` with the expert studies left in training, over all five folds. The
first run of ours trained on the whole corpus, and the first to be submitted.

```
out-of-fold macro AUC   0.8443   (95% 0.840-0.849), 4407 studies
per fold                0.8423 +/- 0.0090   f0:0.836 f1:0.834 f2:0.859 f3:0.840 f4:0.841
expert, out of fold     0.8405   (95% 0.797-0.879), 58 studies, one model each
public leaderboard      0.891
```

**0.891 is exactly the published baseline's score**, and exactly our own baseline
reproduction from 2026-09-07. Two caveats on reading that as a tie: the public
leaderboard is a *sample* of the test set, rounded to three decimals, so two nearby
models can land on the same value; and this run is not the baseline recipe but a longer
one — 30 epochs against 10, five folds against one fixed split.

What it settles is the thing a whole day of comparison could not: **the chain works at
full scale, and our internal numbers read about five points below the public score.**

Keeping the expert studies in training costs the ensemble measurement — see † above —
and buys 59 studies, of which 48 landed in this fold's training set.

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

0. **Our internal numbers read about five points below the public leaderboard.** The
   submitted run measures 0.8443 out of fold and 0.8405 against the experts, and scored
   0.891. Neither internal metric is on the leaderboard's scale — one is capped by a
   label table at 0.867, the other rests on 58 studies — so the gap is not an error to
   be closed but an offset to be remembered. It is listed first because a day was spent
   without it.

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

- **Whether we can beat the baseline, as opposed to matching it.** 0.891 equals the
  published score. Nothing here has yet produced a submission above it.
- **Whether the table or the formula is what puts `v4blend` ahead.** The two moved
  together; the bridge run that would separate them was dropped.
- **What our TTA pooling costs.** The baseline averages *probabilities*
  (`TTA_POOL = "prob"`); `loop.py` and `infer/predict.py` both average *logits* and take
  the sigmoid afterwards. Their notebook says the choice *"orders studies differently"*
  and that they settled it by measuring — and AUC reads nothing but order. The port kept
  their `TTA_OVERLAP` and not their `TTA_POOL`. `scripts.rescore` can answer this on the
  packages we already have, without retraining.
- **What `depth_compress` does.** Never run.

## The obvious next runs

| | Why |
|---|---|
| probability pooling at inference | one line, no retraining, and the only known divergence from the recipe that scored 0.891 |
| `window_30epochs_v4blend`, 5 folds | the best single fold we have, and the table question deserves five folds rather than one |
| `dinov2-base` | 86M parameters against 22M, and the same pixels, so the cache is reused. Whether the L40 has the room is a question about seconds per epoch, not about the 5 GiB the small encoder occupies |
| layer-wise LR decay | the encoder has one learning rate for six unfrozen blocks; the standard recipe gives each block its own |
| four windows at inference | the four disjoint windows already tile the stack — the ten sliding ones re-see the same slices for 2.5x the compute and about +0.003. There is an efficiency prize |
