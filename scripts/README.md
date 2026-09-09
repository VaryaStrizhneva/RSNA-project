# Entry points

Five commands, all on the critical path from data to a leaderboard score. They contain
no logic worth testing — argument parsing, wiring, logging. Everything they call lives
in [`../src/rsna/`](../src/rsna/), which is what the Kaggle notebook imports too, so a
notebook and a command line cannot drift apart.

Exploration that produced a documented conclusion lives in [`../tools/`](../tools/)
instead.

| | |
|---|---|
| **`train.py`** | Fit one model end to end and write a weights package. The equivalent of the public notebook's `main()`, except it stops after training: the scored run does inference only. Takes `--limit`, `--img`, `--slices`, `--epochs`, `--batch`, `--fold`, `--cache`, so it can run tiny for a plumbing check or full for a real run. |
| **`predict.py`** | Read a weights package over a decoded cache and write `submission.csv`. Holds no model definition: the config comes from the package, so a member is always read the way it was fitted. **This is what the Kaggle notebook will call.** |
| **`report.py`** | Turn a finished sweep into one self-contained HTML page and ten lines of text: out-of-fold AUC with an interval, a row per pathology against the label table's own ceiling, the training curves, and a list of what does not look right. Reads only the `history.json` and `holdout.csv` that `train.py` writes beside the weights, so it needs no GPU and says the same thing every time. |
| **`kaggle_push.py`** | Stamp the current commit into a copy of the kernel and push it. Never modifies the repo. The stamp is the only thing joining a leaderboard score to a commit. |
| **`nbstrip.py`** | A git clean filter, not something you run. Strips notebook outputs on the way into git, because rendered slices are competition data. See [`../notebooks/README.md`](../notebooks/README.md). |

## The usual sequence

```bash
python -m scripts.train --split train_series --cache /data/cache.npy --out out/package
python -m scripts.report out/package
python -m scripts.predict --package out/package --split test_series --out submission.csv
python -m scripts.kaggle_push
```

The first runs where there is a GPU, the second reads back what it measured, the third
is a local check that the package loads, the fourth pushes what Kaggle will re-run on
the ~1,300 hidden studies.

Several folds at once, then one report over all of them:

```bash
for f in 0 1 2 3 4; do
  python -m scripts.train --experiment window_baseline --fold $f \
    --cache out/cache-80.npy --out out/sweep-window-20/pkg-f$f
done
python -m scripts.report out/sweep-window-20
```
