# flight0234/rsna-knee-hybrid-report-labels

Vendored copy, unmodified.

| | |
|---|---|
| Source | <https://www.kaggle.com/datasets/flight0234/rsna-knee-hybrid-report-labels> |
| Author | `flight0234` |
| License | **CC0-1.0**, as reported by the Kaggle CLI at download |
| Updated on Kaggle | 2026-08-11 |
| Downloaded | 2026-09-23 |

```bash
kaggle datasets download flight0234/rsna-knee-hybrid-report-labels --unzip \
  -p data/external/flight0234-rsna-knee-hybrid-report-labels
```

## Contents

| File | What it is |
|---|---|
| `report_labels_v4hybrid.csv` | 4,407 studies x 12 findings: a score and a `__conf` column per target, 25 columns in all. |

## It is `llm_labels_v4_blend` with one column replaced

Measured, not assumed. Column by column against steven's table on the 4,407 shared
studies:

| | |
|---|---|
| Identical to `steven_v4_blend` | **eleven of twelve targets**, to floating-point equality |
| Different | `Fracture` alone — and not a copy of pilkwang's column either |

That one column is worth the whole difference between the two tables. `Fracture` is where
steven is weakest (0.793 on the expert studies) and pilkwang strongest (0.870); this table
reads 0.870 there.

**It replaces a column, it does not average one.** Our own `labels_blend_*` runs averaged
percentile ranks across both sources and landed *below* either of them — 0.8401 and 0.8399
out of fold against 0.8574 for steven alone. Swapping a whole column keeps each source's
internal ordering intact; blending destroys it. The two are not the same operation and
this table is the evidence.

## Measured on the 58 expert-labelled studies

| | Macro AUC | Coverage |
|---|---|---|
| `report_labels_v4hybrid.csv` | **0.8991** | 58/58 |

Per target: ACL 0.99 · MCL 0.97 · Medial Meniscus 0.95 · Lateral Meniscus 0.88 ·
Medial OA 0.93 · Lateral OA 0.83 · PF OA 0.90 · Effusion 0.88 · Synovitis 0.79 ·
Baker's 0.94 · Contusion 0.86 · Fracture 0.87

## Caveats

- **Not contaminated**: continuous values, macro AUC 0.8991.
- Eleven of its twelve columns are already in the repo under
  `stevenleehans-rsna-knee-llm-report-labels`. Training on both tables would be training
  twice on the same labels but for one finding.
- It does carry `__conf` columns, so `weights: confidence` is available — though the
  seven-way screen found weighting makes no measurable difference either way.

## Checksums (as vendored)

```
cafa25739b1428b1e67d89f37ee72e00  report_labels_v4hybrid.csv
```
