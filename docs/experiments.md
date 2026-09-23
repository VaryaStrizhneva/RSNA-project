# Experiment log

One row per submission. Kaggle knows the score, git knows the code, and **nothing
connects the two but this file** — so it is filled in by hand, every time, right
after submitting.

Each pushed run prints its own `run stamp` in the Kaggle log
(`scripts/kaggle_push.py` writes the commit into the notebook before pushing), so a
row can always be recovered from a run that was not logged at the time.

| # | Date | Kernel version | Commit | What changed | Public LB | Notes |
|---|---|---|---|---|---|---|
| 1 | 2026-09-07 | — | — | Dummy 0.5 benchmark — pipeline smoke test | 0.500 | as expected |
| 2 | 2026-09-07 | baseline repro v2 | — | pilkwang's notebook, re-run on our account | 0.891 | ours, though it reproduces someone else's recipe |
| 3 | 2026-09-10 | 2 | — | trained on 80 studies — plumbing, not a model | 0.677 | |
| 4 | 2026-09-16 | 4 | `7188488` | the whole corpus: 4407 studies, 5 folds, 30 epochs, expert labels left in training | 0.891 | matches the published baseline exactly |
| 5 | 2026-09-16 | 5 | `8eafeb3` | the label table: steven `llm_labels_v4_blend` + `assertedness`, everything else as #4 | **0.899** | first submission above the baseline |
| — | 2026-09-17 | 6 | `f214adc` | DINOv2-base, 2 folds | *(none)* | kernel ERROR: the weights dataset was still processing when the kernel ran |
| — | 2026-09-22 | 7 | `1cd86a1` | screening run, first attempt | *(none)* | same cause as #6 — pushed before the dataset was `ready` |
| 6 | 2026-09-22 | 8 | `1cd86a1` | **screening family**: 1 fold, expert labels held out, steven v4 + `assertedness` | 0.884 | not comparable to #4/#5 — one model instead of five, 59 fewer training studies |
| 7 | 2026-09-22 | 9 | `8f1aaaa` | same screen, pilkwang + `confidence` | 0.884 | **identical to #6**, though they differ by 0.0096 out of fold |
| 8 | 2026-09-23 | 10 | `8f1aaaa`-dirty | same screen, pilkwang + `uniform` | 0.881 | dirty tree was documentation only; the code is `8f1aaaa` |
| 9 | 2026-09-23 | 11 | `8f1aaaa`-dirty | same screen, **steven v4 + `uniform`** | **0.886** | best of the four screens here and out of fold; same note on the stamp |
| 10 | 2026-09-23 | 12 | `8f1aaaa`-dirty | same screen, per-target rank blend of both tables | 0.864 | 0.0175 below #9 out of fold, 0.022 below here — the holdout predicted this one |

**More than one variable moved in row #5**, and a single-fold screen was run to find out
which. Both metrics agree: the table, not the weighting. Rows #6-#9 rank the same way
here as out of fold (rank correlation +0.63), with the leaderboard compressing the gap
by about four. Read them as groups — the two steven rows against the two pilkwang rows —
and not pair by pair, since #7 and #8 differ by 0.0001 out of fold. See
[`../experiments/RESULTS.md`](../experiments/RESULTS.md).

### Two scales, not one

Internal numbers sit below the leaderboard, but not by a constant — how far depends on
how the submission was built:

| Family | Internal OOF | Public LB | Gap |
|---|---|---|---|
| 5 folds, experts in training (#4, #5) | 0.8557 | 0.899 | +4.3 |
| 1 fold, experts held out (#6) | 0.8560 | 0.884 | +2.8 |

The out-of-fold score always measures **one model per study**. A five-fold submission
averages five, so its OOF understates it by more than a single-fold one. Converting an
internal number to an expected leaderboard score needs the right rule of the two.

### Pushing a kernel: wait for `ready`

Rows without a score above failed the same way twice. `scripts.kaggle_dataset` returns
while Kaggle is still processing the upload — it prints *"Dataset version is being
created"* — and a kernel pushed straight after mounts nothing, then exits on
`no weights package is mounted`. Check `kaggle datasets status <slug>` reads `ready`
before `scripts.kaggle_push`.

That the run *failed* rather than scoring 0.5 is by design: the scored notebook has no
`try/except` around `run_submission`, precisely so a broken mount cannot be mistaken for
a model that learnt nothing.

## Reference points, not our runs

Scores from public notebooks, for calibration. Not produced by our pipeline — except
pilkwang's, which row #2 above reproduces on our own account.

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
