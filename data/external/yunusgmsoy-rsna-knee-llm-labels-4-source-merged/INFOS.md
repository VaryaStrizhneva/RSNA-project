# yunusgmsoy/rsna-knee-llm-labels-4-source-merged

Vendored copy, unmodified.

| | |
|---|---|
| Source | <https://www.kaggle.com/datasets/yunusgmsoy/rsna-knee-llm-labels-4-source-merged> |
| Author | `yunusgmsoy` |
| License | **CC0-1.0** |
| Updated on Kaggle | 2026-08-23 |
| Downloaded | 2026-09-07 |

```bash
kaggle datasets download yunusgmsoy/rsna-knee-llm-labels-4-source-merged --unzip \
  -p data/external/yunusgmsoy-rsna-knee-llm-labels-4-source-merged
```

## Contents

| File | What it is |
|---|---|
| `report_labels_v5.csv` | 4,407 studies x 12 targets, merging four published label sources. Plausibly the strongest **training** target available. |

## Measured on the 58 expert-labelled studies

Produced by `python -m scripts.compare_label_tables`. Macro AUC is the unweighted
mean over the twelve targets — the competition metric, computed on the only ground
truth we have.

| | Macro AUC | Coverage |
|---|---|---|
| `report_labels_v5.csv` | ~~1.0000~~ | 58/58 |

**The 1.000 measures nothing.** See the caveat below — this table cannot be scored
against the expert labels, so we have no estimate of its quality.

## Caveats

- ⚠️ **Contaminated — training only, never validation.** On the 58 expert-labelled
  studies its twelve columns are *exactly* the official labels (0/1), while
  continuous everywhere else. Using official labels where they exist is entirely
  legitimate for training; scoring against those same studies is circular.
- Flagged `contaminated=True` in [`src/data/labels.py`](../../../src/data/labels.py)
  and excluded from every comparison.
- If you train on it, **drop the 58 annotated rows before measuring anything**.

## Checksums (as vendored)

```
b6ae42279c75b3ae8970704e43832a44  report_labels_v5.csv
```
