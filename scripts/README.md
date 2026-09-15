# Entry points

Seven commands, all on the critical path from data to a leaderboard score. They contain
no logic worth testing — argument parsing, wiring, logging. Everything they call lives
in [`../src/rsna/`](../src/rsna/), which is what the Kaggle notebook imports too, so a
notebook and a command line cannot drift apart.

Exploration that produced a documented conclusion lives in [`../tools/`](../tools/)
instead.

| | |
|---|---|
| **`train.py`** | Fit one model end to end and write a weights package. The equivalent of the public notebook's `main()`, except it stops after training: the scored run does inference only. Takes `--limit`, `--img`, `--slices`, `--epochs`, `--batch`, `--fold`, `--cache`, so it can run tiny for a plumbing check or full for a real run. |
| **`evaluate_train.py`** | Read a trained package back over `train_series`, then write predictions, weak-label disagreements, and gold-label disagreements where expert labels exist. This is for model audit after training, not for submission. |
| **`predict.py`** | Read a weights package over a decoded cache and write `submission.csv`. Holds no model definition: the config comes from the package, so a member is always read the way it was fitted. **This is what the Kaggle notebook will call.** |
| **`report.py`** | Turn a finished sweep into one self-contained HTML page and ten lines of text: out-of-fold AUC with a confidence interval, a row per pathology with its own interval, the training curves, what the run scored against the 58 expert readings when `--holdout-gold` set them aside, and a list of what does not look right. Reads only the `history.json`, `holdout.csv` and `gold.csv` that `train.py` writes beside the weights, so it needs no GPU and says the same thing every time. |
| **`bundle.py`** | Gather the folds of a sweep into one package. Training writes one package per fold so a fold can fail alone; inference wants one directory holding every member, which `predict.py` averages and the notebook mounts as a single dataset. Refuses to bundle members that disagree about the config. |
| **`kaggle_dataset.py`** | Publish `src/rsna`, or a weights package, as a private Kaggle dataset. The scored notebook has internet off, so a mount is the only way code and weights reach it. First push creates, later ones add a version — the notebook's `dataset_sources` never changes. |
| **`kaggle_push.py`** | Stamp the current commit into a copy of the kernel and push it. Never modifies the repo. The stamp is the only thing joining a leaderboard score to a commit. |
| **`nbstrip.py`** | A git clean filter, not something you run. Strips notebook outputs on the way into git, because rendered slices are competition data. See [`../notebooks/README.md`](../notebooks/README.md). |

## The usual sequence

```bash
python -m scripts.train --split train_series --cache /data/cache.npy --out out/package
python -m scripts.report out/package
python -m scripts.predict --package out/package --split test_series --out submission.csv
python -m scripts.kaggle_push
```

## From five folds to a leaderboard score

```bash
python -m scripts.bundle out/sweep-window-20 --out out/submit-window
python -m scripts.predict --package out/submit-window --split test_series   # rehearsal
python -m scripts.kaggle_dataset --source
python -m scripts.kaggle_dataset --package out/submit-window
python -m scripts.kaggle_push
```

Then submit from the kernel page. Only the first two need to be repeated when the model
changes — `--source` only when the library does.

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

## Local mini-DICOM check

The first runs where there is a GPU, the second is a local check that the package reads
back, the third pushes what Kaggle will re-run on the ~1,300 hidden studies.

For a local mini-DICOM check, keep the CSV metadata in `data/raw` and point
`--dicom-root` at a short local folder that mirrors Kaggle's DICOM tree:

```bash
python -m scripts.train --data-root data/raw --dicom-root /path/to/kaggle_style_mini \
  --split train_series --labels data/external/stevenleehans-rsna-knee-llm-report-labels/llm_labels_v4_blend.csv \
  --limit 20 --img 112 --slices 6 --epochs 2 --batch 2 --out out/package-steven-small
```

The DICOM root must contain `train_series/<study>/<series>/*.dcm`, the same shape as
Kaggle. Mini DICOM folders stay ignored by git.

After training, audit model predictions against weak labels and the 58 expert labels:

```bash
python -m scripts.evaluate_train --package out/package-steven-small \
  --data-root data/raw --dicom-root /path/to/kaggle_style_mini \
  --labels data/external/stevenleehans-rsna-knee-llm-report-labels/llm_labels_v4_blend.csv \
  --out out/eval-steven-small
```
