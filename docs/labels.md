# The label problem

## Why this file exists

We are asked to predict twelve findings per study. **Only 58 of the 4,407 training
studies carry those findings as labels** — 1.3%, and it is all-or-nothing: those 58
have all twelve, the other 4,349 have none.

What every study does carry is the **radiology report** written when it was read, in
one of roughly nine languages. The data description invites deriving labels from it,
and the schemas make the intent explicit: `train.csv` has a `Report` column and
`test.csv` does not. Text is available when fitting and absent when predicting.

So this is a **weak supervision problem wearing a computer-vision costume**. The
image model can only ever be as good as the targets it is trained on, which makes
label quality the ceiling on everything downstream.

The 58 annotated studies are therefore **not a training set**. They are the only
ground truth we have, and their entire job is to tell us whether a set of derived
labels is any good.

## What is already public

Deriving labels from the reports has been done and published, several times over.
Rather than starting from scratch we vendored the tables into `data/external/`, one
directory per dataset, each with a `PROVENANCE.md` giving the source, licence,
checksums and measured quality.

| Directory in `data/external/` | Files | Licence |
|---|---|---|
| [`pilkwang-rsna-knee-llm-labels/`](../data/external/pilkwang-rsna-knee-llm-labels/) | `report_labels_v2.csv` + the extraction script | TO CHECK |
| [`stevenleehans-rsna-knee-llm-report-labels/`](../data/external/stevenleehans-rsna-knee-llm-report-labels/) | three variants of one reading | CC0-1.0 |
| [`yunusgmsoy-rsna-knee-llm-labels-4-source-merged/`](../data/external/yunusgmsoy-rsna-knee-llm-labels-4-source-merged/) | `report_labels_v5.csv` | CC0-1.0 |

At least six more exist on Kaggle; [`references.md`](references.md) lists them.

The sources are registered in [`src/data/labels.py`](../src/data/labels.py), which
loads them, converts them to ranks, and blends them.

## How they compare

```bash
python -m scripts.eval_label_sources
```

Macro AUC on the 58 expert-labelled studies — the competition metric, computed on
the only ground truth available:

| Source | Macro AUC | 95% CI | Coverage |
|---|---|---|---|
| `steven_v4_blend` | **0.8927** | 0.857–0.923 | 58/58 |
| `steven_v2` | 0.8873 | 0.851–0.919 | 58/58 |
| `steven_full` | 0.8780 | 0.842–0.909 | 58/58 |
| `pilkwang` | 0.8672 | 0.829–0.900 | 57/58 |
| `yunus_merged` | ~~1.0000~~ | — | ⚠️ contaminated |

Per target, no source wins everywhere:

| Target | pilkwang | steven_v4_blend |
|---|---|---|
| ACL | 0.987 | 0.987 |
| MCL | **0.976** | 0.968 |
| Medial Meniscus | 0.944 | 0.948 |
| Lateral Meniscus | 0.842 | **0.879** |
| Medial OA | 0.909 | **0.932** |
| Lateral OA | 0.789 | **0.833** |
| PF OA | 0.886 | 0.902 |
| Effusion | 0.832 | **0.877** |
| Synovitis | 0.687 | **0.790** |
| Baker's | 0.932 | 0.944 |
| Contusion | 0.753 | **0.860** |
| Fracture | **0.870** | 0.793 |

`Synovitis` is the weak target everywhere. In pilkwang's table 3,712 of 4,406
studies come back `UNK` — the reports simply do not mention synovitis. That is a
source problem, not an extraction problem, and the only recourse is the image.

## Three findings that shape how these are used

### 1. Four published tables reproduce the official labels

On the 58 annotated studies their twelve columns equal the gold exactly (0/1), while
being continuous everywhere else. Training on them is entirely legitimate — using
real labels where they exist is the right thing to do. **Scoring against those same
studies is circular**, and returns a meaningless 1.000.

Such sources carry `contaminated=True` in `src/data/labels.py` and are excluded from
every comparison. If you train on one, drop the 58 annotated rows before measuring
anything.

### 2. Three of the four sources are the same reading

Mean Spearman correlation across targets, on all 4,407 studies:

| | pilkwang | steven_v4_blend | steven_v2 | steven_full |
|---|---|---|---|---|
| **pilkwang** | 1.000 | 0.836 | 0.834 | 0.860 |
| **steven_v4_blend** | 0.836 | 1.000 | **0.995** | **0.966** |
| **steven_v2** | 0.834 | 0.995 | 1.000 | **0.970** |
| **steven_full** | 0.860 | 0.966 | 0.970 | 1.000 |

The three Steven files are one extractor with three settings. Blending them together
mostly averages a reading with itself. There are **two distinct opinions available**,
not four.

### 3. Picking the best source per target does not survive validation

Tempting, given the per-target table above. But:

```
in sample        0.9010   <- optimistic, this is a fit
out of sample    0.8890   <- what the pick is worth
optimism         0.0119
```

Out of sample means: half the studies choose the source per target, the other half
score that choice, 300 repeats. The result lands **below simply taking the best
single source**. With 58 studies and nine positives for MCL, the per-target ranking
is mostly noise. **Do not cherry-pick per target.**

## What we use

```bash
python -m scripts.blend_labels --selection
python -m scripts.blend_labels --write default --out data/processed/report_labels_blend.csv
```

Candidates, all scored the same way:

| Blend | Sources | Macro AUC | 95% CI |
|---|---|---|---|
| `default` | pilkwang + steven_v4_blend | **0.8930** | 0.854–0.925 |
| `steven_only` | steven_v4_blend | 0.8927 | 0.857–0.923 |
| `three` | + steven_v2 | 0.8914 | 0.853–0.923 |
| `all_four` | + steven_full | 0.8899 | 0.851–0.921 |
| `steven_family` | the three Steven files | 0.8887 | 0.852–0.920 |
| `pilkwang_only` | pilkwang | 0.8669 | 0.828–0.900 |

The default is an equal-weight **rank** blend of the two genuinely distinct sources.

**It is not measurably better than `steven_only`** — 0.0003 apart, inside an interval
seven points wide. It is preferred on robustness grounds: it does not stake every
target on one extractor, at no measured cost. `steven_only` is a perfectly
defensible alternative and is simpler.

### Why ranks

Each column is converted to a percentile rank before averaging. Two reasons:

- **The scales are incomparable.** pilkwang's `ACL` column holds 5 distinct values;
  steven's holds 107. Averaging raw scores would let the more spread-out source
  dominate for no good reason.
- **The metric reads order only.** Macro AUC is invariant under any strictly
  increasing transform, so ranking discards nothing that could ever score a point.
  See [`pipeline_pitfalls.md`](pipeline_pitfalls.md) §5.

A worked example, one study, target `ACL`:

| | value |
|---|---|
| pilkwang raw | 0.28 |
| steven_v4_blend raw | 0.25 |
| pilkwang rank | 0.6816 |
| steven rank | 0.7276 |
| **mean → written to the CSV** | **0.7046** |

A study missing from a source is given that source's mean rank, so it neither helps
nor hurts. This is what handles pilkwang's missing study.

## The honest summary

Every comparison on this page sits inside the noise of a 58-study measurement, whose
confidence interval is roughly seven points wide. We have not found the best table.
We have **excluded two mistakes** — training on contaminated tables and believing
per-target cherry-picking — and chosen a defensible default.

The real arbiter will be the public leaderboard on ~1,300 studies, which is a far
finer instrument than anything we can compute locally.

## Open leads

- Six or more published tables remain unmeasured, including multi-source merges.
- `Synovitis` at 0.79 is the ceiling-setter. Improving it means reading the images,
  not the reports.
- `barun2104`'s pseudo-labels were produced with **Qwen2.5-7B-Instruct running
  locally** — proof that this work can be done entirely offline with open weights,
  should the rules turn out to forbid a hosted API.
