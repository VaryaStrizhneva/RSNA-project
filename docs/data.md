# The competition data

What we measured ourselves on the five CSVs Kaggle provides, in `data/raw/`.

Read [`competition_description.md`](competition_description.md) first for what the
files are *supposed* to contain — this file is about what they actually contain.
Everything below is recomputable:

```bash
python -m scripts.metadata_report --data-root data/raw
```

The short version: **the images are the easy part**. `train_series.csv` is clean and
complete, but only 58 of 4,407 studies carry the labels we are asked to predict, so
the targets have to come from the free-text reports. That problem has its own file,
[`labels.md`](labels.md).

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

The 1.3% label coverage above is the defining constraint of this competition, and it
is treated separately: see [`labels.md`](labels.md) for the published label sources,
how they compare, and which one we use.
