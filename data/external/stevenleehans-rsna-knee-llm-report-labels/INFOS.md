# stevenleehans/rsna-knee-llm-report-labels

Vendored copy, unmodified.

| | |
|---|---|
| Source | <https://www.kaggle.com/datasets/stevenleehans/rsna-knee-llm-report-labels> |
| Author | `stevenleehans` |
| License | **CC0-1.0** |
| Updated on Kaggle | 2026-08-09 |
| Downloaded | 2026-09-07 |

```bash
kaggle datasets download stevenleehans/rsna-knee-llm-report-labels --unzip \
  -p data/external/stevenleehans-rsna-knee-llm-report-labels
```

## Contents

Three variants of the same reading, each 4,407 studies x 12 targets with
continuous scores and no auxiliary columns.

| File | What it is |
|---|---|
| `llm_labels_v4_blend.csv` | The strongest variant — **the one we use** |
| `llm_labels_v2.csv` | Earlier revision |
| `llm_labels_full.csv` | Earlier revision |

## Measured on the 58 expert-labelled studies

Produced by `python -m scripts.compare_label_tables`. Macro AUC is the unweighted
mean over the twelve targets — the competition metric, computed on the only ground
truth we have.

| | Macro AUC | Coverage |
|---|---|---|
| `llm_labels_v4_blend.csv` | **0.8927** | 58/58 |
| `llm_labels_v2.csv` | 0.8873 | 58/58 |
| `llm_labels_full.csv` | 0.8780 | 58/58 |

`llm_labels_v4_blend` per target: ACL 0.99 · MCL 0.97 · Medial Meniscus 0.95 ·
Lateral Meniscus **0.88** · Medial OA 0.93 · Lateral OA **0.83** · PF OA 0.90 ·
Effusion **0.88** · Synovitis **0.79** · Baker's 0.94 · Contusion **0.86** ·
Fracture 0.79

**Best single source overall**, and the best available on eight of the twelve
targets — including `Synovitis`, the weakest target across every table (0.79
against pilkwang's 0.69).

## Caveats

- The three files are **near-duplicates of one another** (mean Spearman 0.966-0.995
  across targets). Treat them as one source with three settings, not three
  opinions — blending them together mostly averages one extractor with itself.
- Not contaminated: values on the 58 annotated studies are continuous.

## Checksums (as vendored)

```
4086a0f9dab5a0f911fdd38b2d09e1d0  llm_labels_full.csv
dbe1ab6b23079a468af922b856f71ca8  llm_labels_v2.csv
44874d8ded7699d356b935ca8d7e8df7  llm_labels_v4_blend.csv
```
