# Experiments

One JSON per experiment, defining the whole run. `--experiment <name>` and nothing else
is enough to reproduce it.

```bash
python -m scripts.train --experiment depth_compress
```

```json
{
  "config": { "stem": "compress", "stem_depth": 1 },
  "run":    { "split": "train_series", "fold": 0 }
}
```

Two sections. **`config`** becomes a `Config` and is written verbatim into the manifest
of every weights package the run produces — it says what the model *is*. **`run`** is
orchestration and stays out of the manifest: two models trained on different label
tables are the same architecture, and a package should not claim otherwise.

Name only what differs; everything else comes from the defaults below. A field that no
longer exists makes the load refuse loudly, which is what should happen to a stale
experiment.

Only machine paths stay on the command line — `--data-root`, `--cache`, `--out`. Every
other flag (`--img`, `--epochs`, `--fold`, `--split`, …) overrides the file, so a quick
check can shrink a real run without editing its definition.

---

## `run`

| Key | Default | |
|---|---|---|
| `split` | `train_series` | which directory of DICOMs to fit on |
| `labels` | `data/external/pilkwang-…/report_labels_v2.csv` | the report-derived label table |
| `encoder` | `models/dinov2-small` | where the pretrained encoder lives |
| `fold` | `0` | which fold is held out |

## `config` — pixels

| Key | Default | |
|---|---|---|
| `img` | `336` | side of the square handed to the encoder |
| `crop_mm` | `130.0` | millimetres of anatomy kept. Below the field of view of 99.6% of series |
| `slices` | `12` | slices decoded per slot and kept in the cache |
| `group` | `3` | slices the encoder takes at once |
| `band` | `[0.2, 0.8]` | portion of the stack sampled. The ends are soft tissue outside the joint |
| `slots` | 6 slots | the fixed positions, as `[name, plane, fluid, fatsat]` |

Pixel spacing is not a field: it is `crop_mm / img`, so 0.387 mm by default. Change
either to move it.

## `config` — pixel rules

Under `"rules"`. Each one changes *which pixels* a slot holds without changing any
shape, which is why they are named: a package can state the reading it needs, and an
unrecognised value is refused rather than defaulted.

| Key | Default | Alternative |
|---|---|---|
| `order` | `normal` | `dominant_axis` — sorts on the raw patient coordinate instead of the slice normal; on sagittal series the two come out reversed |
| `laterality` | `centre` | `corner_x` — thresholds the image corner instead of its centre, with a 5 mm dead zone instead of 20 |
| `slot_fallback` | `false` | `true` — fills an empty structural slot by relaxing the weighting. Rejected by default: it puts one series in two slots for 2,383 of 4,407 studies |
| `decode_fill` | `nearest` | `zero` — blacks out an unreadable slice instead of copying the closest one that decoded |

## `config` — encoder and head

| Key | Default | |
|---|---|---|
| `encoder` | `dinov2` | matched against the mounted directory name |
| `encoder_variant` | `small` | 12 blocks, hidden 384, patch 14 |
| `pool` | `cls_mean` | or `cls_mean_focal`, which adds the top eighth of each channel |
| `prior` | `false` | the fixed anatomical tilt on the attention logits |
| `unfreeze_last` | `6` | trainable encoder blocks, counted from the end |
| `stem` | `window` | or `compress` — see the two experiments below |
| `stem_depth` | `1` | gated residual blocks before the projection, for `compress` |
| `weights` | `confidence` | where a report-derived label's loss weight comes from. `confidence` reads the table's own `__conf` columns, as the published baseline does — a table without them is a hard error, never a silent fallback. `uniform` weighs every study 1.0, the comparison the others must be measured against. `assertedness` derives the weight from the scores, so a table with no confidence column can still be weighted |
| `silence` | `null` | the score a table gives a finding its report never mentions — needed by `assertedness`, and **not** the middle of the range: `llm_labels_v4_blend` puts it at 0.25, `llm_labels_v2` at 0.50, pilkwang at 0.28. Leave it null and `scripts.train` looks it up in `rsna.data.labels.LABEL_SOURCES` and stamps the resolved value into the manifest, so an experiment never repeats a property of a file. With `silence: 0.5`, `assertedness` is exactly prvsiyan's `certainty` |
| `weight_floor` | `0.25` | what a study weighs when its source expresses no confidence. Every published formula is `floor + (1 - floor) * signal`, and the teams disagree: pilkwang's baseline uses `0.25`, prvsiyan's V52 uses `0.15`. It sets how hard a report that never mentions a finding pulls — a quarter of every table, and 84% of the `Synovitis` column. Nobody has tuned it |

## `config` — fitting

| Key | Default | |
|---|---|---|
| `epochs` | `10` | |
| `batch_studies` | `8` | a study is a bag of up to 6 slot images |
| `lr_head` | `1e-3` | |
| `lr_backbone` | `8e-6` | the encoder is adapted, not retrained |
| `warmup_frac` | `0.15` | fraction of the schedule warming the learning rate up |
| `weight_decay` | `0.02` | |
| `eval_batch` | `8` | |
| `seed` | `2026` | |
| `gold_weight` | `3.0` | loss weight of the 58 expert-labelled studies |
| `n_folds` | `5` | |

## `config` — augmentation

Applied per slot image, so each acquisition gets its own jitter.

| Key | Default | |
|---|---|---|
| `aug_rot_deg` | `8.0` | rotation, plus or minus |
| `aug_scale` | `0.08` | zoom **in only** |
| `aug_shift` | `0.05` | translation, as a fraction |
| `aug_intensity` | `0.10` | intensity scale, plus or minus |

Neither flip is available, deliberately. A horizontal one would reintroduce the axis
laterality normalisation removes; a vertical one moves the input off the distribution,
since a knee is acquired in a canonical orientation and where a finding sits in the
frame is information — a Baker's cyst is identified by lying in the popliteal fossa.

## `config` — laterality

| Key | Default | |
|---|---|---|
| `lat_min_offset_mm` | `20.0` | inside this distance from the midline the side is not readable from geometry, and the study is left unresolved rather than guessed |
| `lat_legacy_offset_mm` | `5.0` | the dead zone the `corner_x` rule was fitted with |

---

## Not configurable yet

Things a JSON cannot currently express. The first three are hyperparameters that happen
to live in code, which is an accident; the rest are structural.

| | Where | Worth exposing? |
|---|---|---|
| **Percentile bounds `[1, 99]`** | `dicom/pixels.py` | **Yes.** A real hyperparameter: `[2, 98]` clips more, `[0.5, 99.5]` less. One line |
| **Slot dropout** | not implemented | **Yes.** Randomly masking a *present* slot during training is the most natural augmentation here — missing slots are the norm at test time (12 of 18 on our studies), and the presence mask already exists |
| **Band jitter** | not implemented | **Yes, and it matters for `depth_compress`.** That stem loses the free stack augmentation the window scheme gets; shifting the band a little each epoch is the obvious replacement |
| Crop centre | `dicom/pixels.py` | Maybe. It is centred on the *image*, not the joint — which is where a posterior Baker's cyst gets cut off |
| Zoom out | `train/augment.py` | Maybe. `border` padding repeats the edge row, and that edge is the popliteal fossa, so zooming out fabricates tissue exactly where a cyst is looked for |
| Optimiser, scheduler, loss | `train/loop.py` | Not yet. AdamW, OneCycle and BCE are hardcoded. Making them configurable means strings mapped to classes — machinery for a need that does not exist |
| Interpolation, sampling strategy | `dicom/pixels.py` | No. Changing those is code, not a parameter |

The rule: **if a run needs something `Config` cannot express, the field belongs in
`Config`** — not the experiment in code.

---

## The experiments

### `window_reference`

The published baseline, to the letter: pilkwang's table, `weights: confidence`, 10
epochs, batch 8. Three contiguous slices reach the encoder; training draws one window at
random per step, inference slides the window and averages — **ten encoder passes per
slot**.

Nothing here is our idea. It names the four fields it does because a yardstick should
state its own terms, not inherit them silently from a default that may move. That
configuration scored **0.891** on the leaderboard; it is not meant to be improved, it is
what the ideas are measured against.

### `window_60epochs`

`window_reference` with `epochs: 60` and nothing else. It exists because the reference
run flagged all five folds as undertrained — the training loss was still falling when
the tenth epoch ended.

It answers a narrower question than it looks. `epochs` also sets `total_steps` on the
OneCycle schedule, so sixty epochs is not ten epochs continued: the whole cycle is
restretched, the peak moves from epoch 1.5 to epoch 9, and the model trains at a high
learning rate for far longer. It tells you whether a longer cycle wins, not where the
model stops learning. The second question needs a schedule that does not depend on the
epoch count, which `Config` cannot express today.

### `depth_compress`

All twelve cached slices at once, mixed to three channels by a learned projection.

**Buys:** one encoder pass per slot instead of ten — most of the inference budget back —
and training sees exactly what inference sees.

**Costs:** the stack augmentation and the test-time averaging, both free in the window
scheme. If it underperforms, look there first; the answer is probably band jitter rather
than abandoning the stem.

Ported from `bend-the-knee-to-the-dinosaurs`, where it sits behind a switch whose
default is off — one competitor's idea, tried, not settled practice. See
[`../docs/references.md`](../docs/references.md).

## Comparing two of these

Change **one field** between two experiments, or the result answers no question. The
weighting fields make that easy to get wrong, because three of them interact:

| to isolate | hold fixed | vary |
|---|---|---|
| the weighting | `labels` | `weights`: `confidence` against `uniform` |
| the table | `weights` | `labels` |
| the formula | `labels` | `weights`: `confidence` against `assertedness` |

Note the last one is only available on a table that reports confidence — pilkwang does,
`llm_labels_v4_blend` does not. On a table without it, `assertedness` is the only
weighting there is, and the comparison is against `uniform`.

Nothing here is measured yet. Whether *any* weighting beats `uniform` on the full corpus
is open, and it is the question worth answering first: if it does not, the rest of this
section is moot.

---

Record every run in [`../docs/experiments.md`](../docs/experiments.md).
