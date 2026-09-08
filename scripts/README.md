# Entry points

Four commands, all on the critical path from data to a leaderboard score. They contain
no logic worth testing — argument parsing, wiring, logging. Everything they call lives
in [`../src/rsna/`](../src/rsna/), which is what the Kaggle notebook imports too, so a
notebook and a command line cannot drift apart.

Exploration that produced a documented conclusion lives in [`../tools/`](../tools/)
instead.

| | |
|---|---|
| **`train.py`** | Fit one model end to end and write a weights package. The equivalent of the public notebook's `main()`, except it stops after training: the scored run does inference only. Takes `--limit`, `--img`, `--slices`, `--epochs`, `--batch`, `--fold`, `--cache`, so it can run tiny for a plumbing check or full for a real run. |
| **`predict.py`** | Read a weights package over a decoded cache and write `submission.csv`. Holds no model definition: the config comes from the package, so a member is always read the way it was fitted. **This is what the Kaggle notebook will call.** |
| **`kaggle_push.py`** | Stamp the current commit into a copy of the kernel and push it. Never modifies the repo. The stamp is the only thing joining a leaderboard score to a commit. |
| **`nbstrip.py`** | A git clean filter, not something you run. Strips notebook outputs on the way into git, because rendered slices are competition data. See [`../notebooks/README.md`](../notebooks/README.md). |

## The usual sequence

```bash
python -m scripts.train --split train_series --cache /data/cache.npy --out out/package
python -m scripts.predict --package out/package --split test_series --out submission.csv
python -m scripts.kaggle_push
```

The first runs where there is a GPU, the second is a local check that the package reads
back, the third pushes what Kaggle will re-run on the ~1,300 hidden studies.
