# pilkwang/rsna-knee-llm-labels

Vendored copy, unmodified.

| | |
|---|---|
| Source | <https://www.kaggle.com/datasets/pilkwang/rsna-knee-llm-labels> |
| Author | `pilkwang` |
| License | **TO CHECK on the dataset page** |
| Updated on Kaggle | 2026-08-06 |
| Downloaded | 2026-09-07 |

```bash
kaggle datasets download pilkwang/rsna-knee-llm-labels --unzip \
  -p data/external/pilkwang-rsna-knee-llm-labels
```

## Contents

| File | What it is |
|---|---|
| `report_labels_v2.csv` | 4,406 studies x 12 findings. Three columns per target: a continuous score, `__conf`, and `__verdict` (`YES`/`NO`/`UNK`). `UNK` means the report does not say — not the same as negative. |
| `api_labeler.py` | The extraction script, prompt included. `claude-opus-5` via the Anthropic API, schema-enforced structured output, bulk submission with prefix caching. Runs offline, outside Kaggle. |

## Measured on the 58 expert-labelled studies

Produced by `python -m scripts.compare_label_tables`. Macro AUC is the unweighted
mean over the twelve targets — the competition metric, computed on the only ground
truth we have.

| | Macro AUC | Coverage |
|---|---|---|
| `report_labels_v2.csv` | **0.8672** | 57/58 |

Per target: ACL 0.99 · MCL **0.98** · Medial Meniscus 0.94 · Lateral Meniscus 0.84 ·
Medial OA 0.91 · Lateral OA 0.79 · PF OA 0.89 · Effusion 0.83 · Synovitis **0.69** ·
Baker's 0.93 · Contusion 0.75 · Fracture **0.87**

Best of all sources on `MCL` and `Fracture`; weakest on `Synovitis`, where 3,712 of
4,406 studies come back `UNK` — the reports simply do not mention it.

## Caveats

- **4,406 rows for 4,407 training studies.** One study is absent; a naive merge
  drops it silently.
- `api_labeler.py` is **not runnable as shipped**: it imports `llm_labeler`
  (`FINDING_SPEC`, `TARGETS`, `to_scores`), a module present in neither the dataset
  nor any of the vendored notebooks. The prompt, output schema and verdict logic
  are all readable; the finding definitions are not.
- Not contaminated: its values on the 58 annotated studies are continuous.

## Checksums (as vendored)

```
2a8f25249b5280a40266cc93fb1e1ba8  api_labeler.py
fa9f175fa43577278b7c33798c1fda37  report_labels_v2.csv
```
