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
| 11 | 2026-09-23 | 13 | `8f1aaaa`-dirty | same screen, **riad HYBRID** table (ceiling 0.9025) | 0.882 | the host rows #12/#13 graft onto — logged late, which is why it is here out of order |
| 12 | 2026-10-02 | `rsna-knee-submit` | `041d7b6` | **seven experts added** over the #11 host, each taking its column outright; gold held out of expert training | **0.913** | +0.031 over the same host alone — the largest single gain of the project |
| 13 | 2026-10-02 | `rsna-knee-submit` | `a47d559` | regions cut on disk instead of `/tmp` | 0.913 | **identical to #12**: the fix moved where bytes land, not what was scored |
| 14 | 2026-10-03 | `rsna-knee-submit` | `43f1df8` | host swapped to the #5 wide model (0.899), experts still outright | 0.920 | +0.021 over that host — 0.010 less than #12 bought. See *The host sets the price* |
| 15 | 2026-10-03 | `rsna-knee-submit` | `1e3d00c` | each expert **averaged** with the host in rank space, 50/50 | **0.923** | +0.003 over #14; out of fold the same change is worth +0.0101 |
| 16 | 2026-10-03 | `rsna-knee-d4-public` | `310db66` | **not our model**: pjmathematician's public pipeline, re-run on our account | **0.943** | reproduces the published score exactly, in ~8 h of the 9 h limit. Our frontier reference, and the host #19/#20 target |
| 17 | 2026-10-03 | `rsna-knee-v4g-experts-alone` | `3fac31e` | experts refitted on **v4_blend + gold in training + last epoch**, outright | 0.920 | identical to #14 |
| 18 | 2026-10-03 | `rsna-knee-v4g-experts-blended` | `5728f2f` | the same refitted experts, blended 50/50 | 0.922 | −0.001 against #15, and −0.002 out of fold. **Adding gold to expert training did not pay** |
| 19 | 2026-10-04 | `rsna-knee-nocoat` | `80763e8`-dirty | **the four CoAtNet readers removed** from #16, nothing else | 0.940 | the amputation costs **−0.003**, not the +0.001 the notebook credits CoAt with. Dirty tree was documentation only |
| 20 | 2026-10-04 | `rsna-knee-nocoat-experts` | `80763e8`-dirty | our seven experts over #19, blend 0.5 | **0.944** | **+0.004** over #19 — our best score, and the first time our own work beats the public pipeline (0.943). Rank 337 → 278 |

Rows #1-#11 name a *version* of one kernel; from #12 on there is a kernel per experiment,
so the column holds its slug. Commits from #14 on name **the commit that introduced the
code that ran**, not the one pushed before it — the kernels were pushed while the feature
was still uncommitted, and the user's tree is committed on request rather than per change.

**More than one variable moved in row #5**, and a single-fold screen was run to find out
which. Both metrics agree: the table, not the weighting. Rows #6-#9 rank the same way
here as out of fold (rank correlation +0.63), with the leaderboard compressing the gap
by about four. Read them as groups — the two steven rows against the two pilkwang rows —
and not pair by pair, since #7 and #8 differ by 0.0001 out of fold. See
[`../experiments/RESULTS.md`](../experiments/RESULTS.md).

### The host sets the price

The same seven experts, grafted onto three different hosts, are worth less each time the
host gets better:

| Host | Host alone | With experts, outright | Experts worth |
|---|---|---|---|
| riad HYBRID (#11) | 0.882 | 0.913 (#12) | **+0.031** |
| #5 wide model (#14) | 0.899 | 0.920 (#14) | **+0.021** |
| `nocoat` (#19) | 0.940 | 0.944 (#20), blended | **+0.004** |

**The line through the first two rows was wrong in sign.** Fitted on them it gave about
−0.6 of gain per unit of host and predicted **−0.003** at a 0.940 host; the measurement
is **+0.004**. The decay is real — the blend is worth +0.024 over a 0.899 host and +0.004
over a 0.940 one — but the extrapolation past the data was an artefact of the straight
line, and it nearly argued us out of submitting #20. Keep the rest of this section as the
reasoning that should have been used instead:

Two points determine a line and nothing else does, and there are two specific reasons the
line misled here:

- It is fitted on **outright** grafts, and #20 is a **blend**. For the blend path we have
  exactly one measurement (#15), so there is no trend to extrapolate at all.
- The gain is **bounded**: AUC stops at 1, so it must approach zero as the host
  approaches its ceiling. A straight line crosses zero and keeps going negative, which is
  a property of the line, not a prediction about the pipeline.

What the mechanism says instead, and it points somewhere else entirely: a rank blend at a
fixed 0.5 fails **per column**, not in macro. Averaging ranks half-and-half with a ranker
much weaker than the host costs that column. So the quantity that decides #20 is not
0.943 against 0.899 in macro-12 — it is, on each of the seven targets we own, whether our
expert is comparable to theirs. A host can be excellent on Effusion and Fracture, which
we never touch, and mediocre on MCL. **Macro averages away exactly the information that
decides.**

Two things do argue the blend holds up, and both are about disagreement rather than about
the curve:

- The public pipeline contains **no landmark model and no per-pathology model** —
  `landmark` and `roi` appear zero times in its 4744 lines.
- Our columns correlate in rank **0.57-0.75** with it, where its own legs correlate
  0.75-0.89 with each other. Its own code blends in proportion to disagreement, and we
  disagree with it more than it does with itself.

### The measurement that would replace the guess, for free

`RSNA_COMP_ROOT` already redirects the pipeline at training studies — the bench proved
it. So `nocoat` over ~800 of them in a **commit run** costs no submission and no quota,
and scoring it **per target** against our reference labels says, column by column,
whether our expert beats theirs. That is the number the blend is actually betting on.

One confound, named before the fact: the public weights were fitted on this training
corpus, so their per-target AUC measured there is **overstated** — partly memorisation.
The bias runs in the useful direction, though. An expert that wins on a target *even
though the host trained on it* is safe to blend into that column; a target where we lose
tells us less, because some of their margin is recall.

### Where the public pipeline's 8 hours go

Measured, not estimated: `rsna-knee-bench-full` runs the whole pipeline over a
**120-study** cohort of training data with `RSNA_COMP_ROOT` redirected, so a free draft
commit prices every stage. 2×T4, 38.9 min total.

| Stage | Seconds | Share |
|---|---|---|
| `input_metadata` | 1.9 | 0.1 % |
| `dinov2` | 308.9 | 13.2 % |
| `a5` | 88.3 | 3.8 % |
| `radimagenet` | 107.3 | 4.6 % |
| `public_raptor` | 503.8 | 21.6 % |
| `residual_coat` | 398.3 | 17.1 % |
| `global96_coat` | 398.6 | 17.1 % |
| `d4` | 229.7 | 9.8 % |
| `repairv1_coat` | 295.6 | 12.7 % |

The four CoAtNet arms — `residual`, `global96`, `d4`, `repairv1`, the "4-way consensus"
the notebook's own title advertises — are **56.7 %** of the run between them.

**The cohort size had to be above 48, and 120 is.** The readers are subprocesses that
each reload their own weights, and the notebook gates their pairing on the test-set size:

```python
coat_readers_parallel_pairs = (os.environ.get('RSNA_PARALLEL_COAT_READERS',
                                 '1' if len(_RSNA_TEST_IDS) <= 48 else '0') == '1')
```

So at 48 studies or fewer they run two at a time, and above that strictly one after
another — the regime the scored run's 1322 studies are in. A bench on 48 or fewer would
have measured the pairs and understated CoAt's share; 120 measures the serial path the
real run takes. Dropping all
four leaves 43.3 %, which is what the `nocoat` ablation does and what makes room for our
leg at all. Raptor, the obvious-looking target at 21.6 %, is not where the time is.

**RadImageNet is not removable**, at 4.6 %. Deleting it also deletes
`globals()['V18_CALIBRATOR_APPLIED'] = True`, and the BTKD calibrator — a learned linear
regression folded into 40 % of several columns — then silently does not apply. The
coupling is a string key, so dependency analysis of the AST does not see it.

### Queued, not yet submitted

Built, verified on the 3-study draft, waiting on the daily quota:

| # | Kernel | Measures |
|---|---|---|
| 19 | `rsna-knee-nocoat` | what the amputation costs in score (~0.942 expected) |
| 20 | `rsna-knee-nocoat-experts` | what our seven columns add to the remaining 28 models, blend 0.5 |

Submit in that order: #20 without #19 cannot separate the amputation's cost from the
experts' gain. **Both are needed, and #19 is the one that is easy to skip** — it measures
nothing new by itself and is therefore the one worth submitting first.

The amputation is verified effective in the draft log, not assumed: `nocoat` prints
`[trim] CoAt readers skipped; public Raptor stands`, and the string `coat` falls from 11
occurrences in #16's log to 5. Draft wall-clock is not evidence of anything — three
studies are dominated by environment setup — so what #19 costs in time is itself one of
the two things it is being submitted to find out.

### Two scales, not one

Internal numbers sit below the leaderboard, but not by a constant — how far depends on
how the submission was built:

| Family | Internal OOF | Public LB | Gap |
|---|---|---|---|
| 5 folds, experts in training (#4, #5) | 0.8557 | 0.899 | +4.3 |
| 1 fold, experts held out (#6) | 0.8560 | 0.884 | +2.8 |
| wide + 7 experts, blend 0.5 (#15) | 0.8608 | 0.923 | +6.2 |

The last row's internal number has **no committed script behind it** — it was computed in
a session, from the stored holdout predictions, and `scripts/evaluate_train.py` does not
know how to join an expert's columns onto a wide model's. Until it does, treat 0.8608 as a
note rather than a reproducible measurement. The wide host it was measured against scored
0.8349 on the same macro, so the blend is worth **+0.0259** internally against **+0.003**
here — the leaderboard compresses this family by about nine, not four.

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
| `rsna-knee-d4-blend` (pjmathematician) | 0.943 |

## Rules that constrain how many experiments we get

### Where a score actually places, 2026-10-03

5042 teams. Read off the full leaderboard, not the top 20 the CLI prints by default
(`kaggle competitions leaderboard … -d`, then rank the `Score` column):

| | Rank | Score |
|---|---|---|
| gold | top 19 | 0.9570 |
| **silver** | **top 252** | **0.9450** |
| bronze | top 504 | 0.9430 |

So #16's 0.943 sits at rank ~337 — inside bronze, **+0.002 short of silver**. But the
pack there is pathological: **1387 teams live between 0.940 and 0.943**, because most of
them are running the same public notebook with different weights. At that density the
public leaderboard no longer separates models, and medals are decided on the **private**
split, where this pile will reshuffle. The practical consequence is that a gain which is
*orthogonal* is worth far more than a gain which is *tuned* — and ours is the former,
which is the whole argument for rows #19/#20.


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
