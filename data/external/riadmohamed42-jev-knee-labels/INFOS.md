# riadmohamed42/jev-knee-labels

Vendored copy, unmodified — one file of several.

| | |
|---|---|
| Source | <https://www.kaggle.com/datasets/riadmohamed42/jev-knee-labels> |
| Author | `riadmohamed42` |
| License | **CC0-1.0**, as reported by the Kaggle CLI at download |
| Updated on Kaggle | 2026-09-21 |
| Downloaded | 2026-09-23 |

```bash
kaggle datasets download riadmohamed42/jev-knee-labels --unzip \
  -p data/external/riadmohamed42-jev-knee-labels
```

## Contents

| File | What it is |
|---|---|
| `HYBRID_labels_scores.csv` | 4,407 studies x 12 findings, one continuous score per target. No confidence column, so `weights: assertedness` or `uniform` — never `confidence`. |

The dataset ships five more tables (`FINAL2_*`, `HYBRID_labels_binary`, a 74-study
`gold74_eval_set`). Only the one above is vendored; `FINAL2_labels_scores` measured
0.8745, below steven's, and the binary variant throws away the ordering the metric reads.

## Measured on the 58 expert-labelled studies

`python -m tools.eval_label_sources --table <path>`

| | Macro AUC | Coverage |
|---|---|---|
| `HYBRID_labels_scores.csv` | **0.9025** | 58/58 |

Per target: ACL 0.99 · MCL **0.98** · Medial Meniscus 0.95 · Lateral Meniscus 0.87 ·
Medial OA 0.93 · Lateral OA **0.86** · PF OA 0.90 · Effusion 0.88 · Synovitis 0.79 ·
Baker's 0.92 · Contusion **0.87** · Fracture **0.90**

The best source we have measured, ahead of `steven_v4_blend` (0.8927) — but by less than
the width of the interval, which is about seven points on 58 studies. Best of all sources
on `MCL`, `Lateral OA`, `Contusion` and `Fracture`; weaker than steven on `Baker's`.

## Caveats

- **Not contaminated**: continuous values on the annotated studies, macro AUC well below
  1.0. The check matters — `tasmeemreza/rsna-knee-refined-llm-labels`, downloaded the
  same day, scores exactly 1.0000 and reproduces the official labels.
- Called an *ensemble* by its author. Its per-target profile differs from every source we
  hold, so it is not a copy of one of them, but how it was built is not documented.
- No `__conf` columns, so the weighting question does not arise: use `uniform`, which the
  seven-way screen picked anyway.

## Checksums (as vendored)

```
0bc3bc343998eb34ffc168e00b48c589  HYBRID_labels_scores.csv
```
