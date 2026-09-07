# pilkwang/rsna-knee-llm-labels

Vendored copy, unmodified.

- Source: <https://www.kaggle.com/datasets/pilkwang/rsna-knee-llm-labels>
- Title: "RSNA Knee LLM-read report labels"
- Author: pilkwang
- Dataset last updated on Kaggle: 2026-08-06
- Downloaded: 2026-09-07
- License: **TO CHECK on the dataset page**

```bash
kaggle datasets download pilkwang/rsna-knee-llm-labels --unzip \
  -p data/external/pilkwang-rsna-knee-llm-labels
```

## Checksums (as vendored)

```
2a8f25249b5280a40266cc93fb1e1ba8  api_labeler.py
fa9f175fa43577278b7c33798c1fda37  report_labels_v2.csv
```

## Contents

| File | What it is |
|---|---|
| `api_labeler.py` | The extraction script, prompt included. `claude-opus-5` via the Anthropic API, schema-enforced structured output, bulk submission with prefix caching. Runs offline, outside Kaggle. |
| `report_labels_v2.csv` | 4,406 studies x 12 findings. Three columns per target: continuous score, `__conf`, and `__verdict` (`YES`/`NO`/`UNK`). `UNK` means the report does not say — not the same as negative. |

## Why it is committed rather than downloaded on demand

1 MB is cheap, and public Kaggle datasets get revised in place. Pinning the exact
bytes we measured against keeps our results reproducible.

⚠️ **4,406 rows for 4,407 training studies** — one study is absent. A naive merge
drops it silently.

Measured macro-AUC against our 58 expert-labelled studies (57 matched): **0.870**.
Per-target breakdown in [`docs/references.md`](../../../docs/references.md).
