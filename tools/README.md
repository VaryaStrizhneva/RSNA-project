# Exploration

Scripts that answer a question once. Their conclusions are already written down in
[`../docs/data.md`](../docs/data.md) — these exist so the numbers can be recomputed when
the data or the sources change, not because anything depends on them.

Nothing here is on the path to a submission. Deleting a file in this folder would break
no run.

| | |
|---|---|
| **`metadata_report.py`** | Everything the five competition CSVs say: 58 labelled studies of 4,407, positive rates, series per study, slot coverage, report lengths. |
| **`eval_label_sources.py`** | Scores each vendored label table against the 58 expert-labelled studies, with a bootstrap interval. Writes nothing. Use it before adopting a new table. |

Getting DICOMs onto a machine is **not** here: see
[`../kaggle/extract/`](../kaggle/extract/). Fetching a subset file by file costs one API
request per slice — we tried it, exhausted the account's download quota, and blocked
every download for hours. A Kaggle notebook archives the subset instead, and the archive
comes back in one request.

```bash
python -m tools.metadata_report
python -m tools.eval_label_sources --bootstrap 400
```

⚠️ A 58-study macro AUC carries a confidence interval about seven points wide. The
second command orders the tables; it does not separate them.
