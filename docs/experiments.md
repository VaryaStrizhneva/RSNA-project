# Experiment log

One row per submission. Kaggle knows the score, git knows the code, and **nothing
connects the two but this file** — so it is filled in by hand, every time, right
after submitting.

Each pushed run prints its own `run stamp` in the Kaggle log
(`scripts/kaggle_push.py` writes the commit into the notebook before pushing), so a
row can always be recovered from a run that was not logged at the time.

| # | Date | Kernel version | Commit | What changed | Public LB | Notes |
|---|---|---|---|---|---|---|
| 1 | | | | Dummy 0.5 benchmark — pipeline smoke test | *(expect 0.500)* | |

## Reference points, not our runs

Scores from public notebooks, for calibration. Not produced by our pipeline.

| Source | Public LB |
|---|---|
| `sample_submission.csv` benchmark | 0.500 |
| `rsna-knee-baseline-v1` (pilkwang) | 0.891 |
| `rsna-knee-read-the-report-then-the-knee` (prvsiyan) | 0.906 |
| `bend-the-knee-to-the-dinosaurs` (mattiaangeli) | 0.939 |

## Rules that constrain how many experiments we get

- **5 submissions per day.** Not a lot when a run takes hours; queue deliberately.
- **2 final submissions** selected at the end. Everything else is thrown away, so
  the last week is about choosing, not exploring.
- The public leaderboard is a *sample* of the test data, and the prevalence of
  findings is **not guaranteed identical** between public and private splits. A
  gain under ~0.005 on the public LB is not evidence of anything.

## How to fill a row

1. `python -m scripts.kaggle_push` — note the printed stamp
2. Submit from the kernel page
3. `kaggle competitions submissions rsna-knee-abnormality-detection` — read the score
4. Add the row. **"What changed" should name one variable.** If two things changed,
   the row cannot explain the delta and the experiment was wasted.
