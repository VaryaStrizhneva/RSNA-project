# The competition data

What we measured ourselves on the five CSVs Kaggle provides, in `data/raw/`.

Read [`competition_description.md`](competition_description.md) first for what the
files are *supposed* to contain, and
[`notebooks/preprocessing.ipynb`](../notebooks/preprocessing.ipynb) for how the
heterogeneity measured below is turned into a fixed-shape tensor — this file is about what they actually contain.
Everything below is recomputable:

```bash
python -m tools.metadata_report --data-root data/raw
```

The short version: **the images are the easy part**. `train_series.csv` is clean and
complete, but only 58 of 4,407 studies carry the labels we are asked to predict, so
the targets have to come from the free-text reports. That problem has its own file,
the section at the bottom of this file.

Last measured: 7 September 2026.

---

## `train.csv` — 4,407 studies

Columns: `StudyInstanceUID`, `Report`, plus the 12 targets.

### The central fact: almost no labels

**58 studies out of 4,407 (1.3%) carry expert labels.** And it is all-or-nothing:
those same 58 studies have all 12 labels filled in, the other 4,349 have none.
There is no partial annotation to salvage.

Direct consequence for strategy: **labels must be extracted from the reports**
(weak supervision), and the 58 annotated studies are not a training set — they are
our **only validation set** for the label extractor. Treat them as such: don't
tune anything on them we couldn't justify, the sample is tiny.

### Positive rates on the 58 annotated studies

| Target | Labeled | Positive | Rate |
|---|---|---|---|
| `Effusion` | 58 | 35 | 60% |
| `Synovitis` | 58 | 27 | 47% |
| `Medial Meniscus` | 58 | 26 | 45% |
| `ACL` | 58 | 24 | 41% |
| `Lateral Meniscus` | 58 | 23 | 40% |
| `PF OA` | 58 | 21 | 36% |
| `Contusion` | 58 | 19 | 33% |
| `Fracture` | 58 | 18 | 31% |
| `Medial OA` | 58 | 15 | 26% |
| `Baker's` | 58 | 12 | 21% |
| `Lateral OA` | 58 | 11 | 19% |
| `MCL` | 58 | 9 | 16% |

⚠️ Out of 58 studies, a 16% rate means **9 positive cases**. These percentages are
indicative, not reliable prevalences — and Kaggle explicitly warns that prevalence
is not guaranteed to be the same across train, public leaderboard and final
evaluation sets.

### Reports

- **No empty report** among the 4,407.
- Length: min 52, p25 587, **median 977**, p75 1,460, max 4,743 characters.
- Languages (keyword heuristic, purely indicative): ~1,400 English, ~900 Spanish,
  ~250 Dutch, ~120 French, a few German, and ~1,700 unclassified by the heuristic.
  The official list of 22 contributing sites (Thailand, Taiwan, Morocco, Türkiye,
  Malta, Croatia, Greece, Bulgaria, Mongolia, …) suggests far more languages than
  that.
- **No actual mojibake**: 0 occurrences of `Ã` across the 4,407 reports. The broken
  encoding seen initially ("TÃ©cnica") was a local display artifact, not a data
  problem. The `ftfy` dependency used in `clean_report_text` is likely unnecessary
  — harmless, but not to be treated as a required fix.

---

## `train_series.csv` — 24,371 series

4,407 studies represented: **no study in `train.csv` is without series**.

- **4 to 14 series per study**, median 5. Distribution:
  3→1, 4→675, 5→2,299, 6→698, 7→310, 8→145, 9→176, 10→74, 11→21, 12→5, 13→2, 14→1.
- Planes: Sagittal 9,864, Coronal 8,609, Axial 5,898.
- `Fluid_Sensitive`: 14,010 set to 1, 10,361 set to 0.

### `Fluid_Sensitive` and `Fat_Suppression` are identical here

The two columns hold **exactly the same value on all 24,371 rows**.

But the Kaggle docs state that while often correlated, they are **not necessarily
equivalent for every case**. In other words, this perfect identity is a property of
*our* sample, not a guarantee. Code must not assume one column can stand in for the
other — though there is no point feeding them as two distinct signals at training
time, since here they carry only one.

### Slot coverage (plane × fluid)

This is the series-selection logic implemented in
[`build_series_slots`](../src/data/metadata.py). Coverage is very uneven:

| Slot | Studies covered | Rate |
|---|---|---|
| Axial / fluid | 4,407 / 4,407 | **100.0%** |
| Sagittal / non-fluid | 4,266 / 4,407 | 96.8% |
| Coronal / fluid | 4,248 / 4,407 | 96.4% |
| Sagittal / fluid | 4,150 / 4,407 | 94.2% |
| Coronal / non-fluid | 3,406 / 4,407 | 77.3% |
| **Axial / non-fluid** | 857 / 4,407 | **19.4%** |

Any multi-view model must therefore **tolerate missing inputs**. The Axial /
non-fluid slot is close to unusable as-is; the top four slots are available for
≥ 94% of studies.

---

## `test.csv` — 3 studies in the repo

This is only a sample file. The real test set holds **~1,300 studies**, substituted
at scoring time.

⚠️ **The `Report` field does not exist at test time.** The text serves only to
build training labels; at inference the model sees images only. Any architecture
that would consume the report as an input is disqualified outright.

---

## DICOM images

**Not in the repo** (see `.gitignore`: `rsna_knee_dicom_mini_part001.zip`), to be
downloaded separately. Per the official docs: 20–45 slices per series (median 30,
long tail out to a few hundred), varying intensities / orientations / resolutions,
a mix of transfer syntaxes, 86 retained DICOM tags.

**Nobody has opened a single DICOM yet** — still to do.

---

## Where the labels come from

The 1.3% label coverage above is the defining constraint of this competition: the
targets have to be derived from the free-text reports, and several people have
published tables that do exactly that. Three are vendored under `data/external/`, each
with an `INFOS.md` recording its source, licence and checksums; the registry is
[`src/rsna/data/labels.py`](../src/rsna/data/labels.py).

Measured against the 58 expert-labelled studies with
`python -m tools.eval_label_sources`:

| Source | Macro AUC |
|---|---|
| `stevenleehans/llm_labels_v4_blend.csv` | **0.893** |
| `pilkwang/report_labels_v2.csv` | 0.867 |
| `yunusgmsoy/report_labels_v5.csv` | ⚠️ reproduces the official labels — trainable, not scorable |

⚠️ A 58-study macro AUC carries a confidence interval about seven points wide. These
numbers order the tables; they do not separate them.

The first training run uses **pilkwang's table**, for one reason only: it is what the
published checkpoints were fitted with, so a run against it is comparable to theirs.
It is *not* the best table, and an earlier version of this file gave a second reason
that turned out to be wrong — see below.

### What a "score" in these tables actually is

Not a binary label. The reports are not binary either: a finding can be asserted,
denied, or simply **never mentioned**, and an assertion can be graded. Every table
encodes those three states plus a gradation on one axis in [0, 1].

pilkwang's is the clearest, because its extractor
([`api_labeler.py`](../data/external/pilkwang-rsna-knee-llm-labels/api_labeler.py))
asks `claude-opus-5` for exactly two fields — `verdict` in `{YES, NO, UNK}` and
`severity` in `{0,1,2,3}` — and a fixed lookup projects the five outcomes onto a score.
The whole 4,406 x 12 table holds **five distinct values**:

| state | score |
|---|---|
| YES, severity 3 | 0.94 |
| YES, severity 2 | 0.82 |
| YES, severity 1 | 0.68 |
| **UNK — not mentioned** | **0.28** |
| NO | 0.08 |

`UNK` sitting above `NO` is the important part: silence is not denial, and it has to
rank between the two.

**Where each table puts silence is a free parameter nobody has tuned**, and it covers
a quarter of every table — 84% of the cells on `Synovitis`:

| Table | Distinct score values | Silence at |
|---|---|---|
| `pilkwang/report_labels_v2.csv` | 5 | 0.28 |
| `stevenleehans/llm_labels_v2.csv` | 81 | 0.50 |
| `stevenleehans/llm_labels_v4_blend.csv` | 211 | 0.25 |
| `yunusgmsoy/report_labels_v5.csv` | 1,987 | varies (a 4-source mean) |

### Confidence: what the `__conf` columns are worth

Two claims that used to be in this file are false.

**"pilkwang is the only table with `__conf` columns."** yunusgmsoy has them too, and
they are better: 2,291 distinct values in [0.079, 1.0], correlating +0.64 with the
certainty of its own target and +0.28 with agreement between the other two tables.
They are also **contaminated** — exactly 1.000 on the 58 expert-labelled studies,
where the table copies the official labels.

**"A `__conf` column is what the loss weighting needs."** pilkwang's carries no
information at all: across 4,406 x 12 cells only three `(verdict, conf)` pairs exist
— `YES -> 0.95`, `NO -> 0.85`, `UNK -> 0.05`. It is a hand-written lookup on the
verdict, which the score already encodes. Applied through `0.25 + 0.75 * conf` it
reduces to three constants: `YES 0.9625`, `NO 0.8875`, `UNK 0.2875` — decaying
silence, not weighting by confidence.

A per-study confidence can be derived from **any** table with continuous scores, no
extra column required:

```python
certainty = np.clip(2.0 * np.abs(y - 0.5), 0, 1)
```

Measured on the 58 expert studies (684 cells), splitting cells at the median of each
criterion and reading the AUC of the target against the truth — AUC is used because it
ignores the offset that the silence convention introduces, which an error metric does
not:

| Criterion | AUC, low half | AUC, high half | Gap |
|---|---|---|---|
| `certainty` from the score | 0.791 | **0.929** | **+0.138** |
| agreement, pilkwang vs stevenleehans v4 | 0.834 | 0.908 | +0.074 |

Both carry signal; certainty carries twice as much and is free. And agreement has a
narrow reach whatever its quality: the median disagreement between the two tables is
**0.055**, with only **3.8%** of cells apart by more than 0.3.

⚠️ 57 studies, and the twelve cells of one study are correlated. This orders the two
criteria; it does not measure either. And it evaluates the *label table*, not the
model — that a cell is more reliable does not yet show that weighting it helps
training. Only paired runs can say.

### One idea nobody has tested

Combining several tables into one — averaging their ranks, or weighting each study by
how much the sources *agree* — is an obvious next step and completely unvalidated. We
tried it once and threw it away: on 58 studies, no combination separated itself from
the best single source, and per-target cherry-picking scored *worse* out of sample than
in it. Note that what was thrown away was the blend as a **target**; the blend as a
**weight** has never been measured.

prvsiyan's notebook does something worth copying, in `_v60_targets` (cell 66) — it is
the target contract of every branch, not of one EfficientNet-B3 variant as this file
used to say. It takes three tables, uses their mean as the target, and derives the
loss weight from their disagreement:

```python
agreement = np.clip(1.0 - 2.0 * np.abs(cube - target).mean(0), 0, 1)
certainty = np.clip(2.0 * np.abs(target - 0.5), 0, 1)
weight    = 0.15 + 0.85 * (0.65 * agreement + 0.35 * certainty)
```

It needs **two** sources, not three — we have two clean ones, pilkwang and
stevenleehans, so no further download is required. One caveat when comparing tables
this way: they must place silence at the same value, or a quarter of the cells shows
a disagreement that is pure convention. pilkwang (0.28) and `llm_labels_v4_blend`
(0.25) are compatible; `llm_labels_v2` (0.50) is not.

Its known failure mode is that agreement is not correctness: extractors sharing a
blind spot agree perfectly while being uniformly wrong. The constants are one team's,
unvalidated like the rest.
