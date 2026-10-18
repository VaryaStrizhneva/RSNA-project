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
| 21 | 2026-10-04 | `rsna-knee-submit` v24 | `116cc0f`-dirty | #15 exactly, plus **`has_pixels`**: an expert abstains on a study cut over no pixels | 0.923 | **identical to #15.** +0.0013 was the out-of-fold prediction and three decimals cannot hold it. Not a loss — the correction ships either way |
| 22 | 2026-10-05 | `rsna-knee-submit` v25 | `116cc0f`-dirty | #21 plus **`expert_baker_axial`** as an eighth expert, blend 0.5 | **0.925** | **+0.002, the predicted figure.** Out of fold the column goes 0.8960 → 0.9161, worth +0.00167 of macro. See *When the host is the comparator* |
| 23 | 2026-10-05 | `rsna-knee-submit` v26 | `116cc0f`-dirty | #22 with the seven refitted on **steven v4_blend**, one variable: the label table | 0.924 | **−0.001.** The gold predicted −0.0011 over the same seven columns, in sign and size. The table was not the lever |
| 24 | 2026-10-05 | `rsna-knee-nocoat-experts` v5 | `df4cb77`-dirty | **#20 plus Baker, and the seven columns fused**: pilkwang and v4_blend as ten folds each, averaged | **0.947** | **+0.003 — silver.** Rank 311 → **201 of 5280**, where the cutoff is 264. Two changes at once; row #25 separates them |
| 25 | 2026-10-05 | `rsna-knee-submit` v27 | `df4cb77`-dirty | #22 with the same seven columns fused — the fusion alone, Baker on both sides | 0.925 | **0.000.** So the fusion is worth nothing measurable, and **Baker carried all of #24**. See *A fusion that is unbiased and imprecise* |
| 26 | 2026-10-07 | `rsna-knee-full-experts` | `134d052`-dirty | the **whole** public pipeline, four CoAtNet readers included, plus our eight experts at 0.5 | 0.948 | the *worst* of the five sent this night. Keeping the readers costs **−0.004** against #28; `nocoat` is confirmed as the default |
| 27 | 2026-10-07 | `rsna-knee-anatomical-mirror-experts` | `134d052`-dirty | **host replaced**: `nartaa`'s single CoAtNet (0.949 published) byte for byte, plus our eight experts at 0.5 | **0.951** | our experts are worth **+0.002** on a 0.949 host — half the +0.004 they bought over `nocoat`, exactly as their own gains on our columns predicted |
| 28 | 2026-10-07 | `rsna-knee-stack-70-50` | `134d052`-dirty | **two hosts stacked**: `nocoat`-d4 at 0.30 and `nartaa` at 0.70 in rank space, experts 0.5 | **0.952** | **our best. Rank 201 → ~96 of 5459**, where silver is top 273 and gold top 21. See *What the three stack regimes did and did not separate* |
| 29 | 2026-10-07 | `rsna-knee-stack-85-50` | `134d052`-dirty | #28 with d4 cut to 0.15, one variable: the host weight | 0.949 | −0.003, but #27 put the same weight at **zero** and scored 0.951. Non-monotone, so this is noise, not a curve |
| 30 | 2026-10-07 | `rsna-knee-stack-70-65` | `134d052`-dirty | #28 with the experts raised to 0.65, one variable: our own weight | 0.950 | **−0.002 for more expert weight.** The one clean single-knob contrast of the night, and it confirms what `EXPERT_BLEND`'s own comment predicted |
| 31 | 2026-10-10 | `rsna-knee-s6-bare` | `134d052`-dirty | **not our model**: `heliosli`'s S6 run verbatim, no graft — `nartaa`'s CoAtNet at 0.7 plus their retrained `convnext_small` at 0.3 | **0.954** | reproduces their published figure exactly. Worth a submission only because their weights dataset had been deleted and re-uploaded, so the README's claim of byte identity was unverifiable |
| 32 | 2026-10-10 | `rsna-knee-s6-r18` | `134d052`-dirty | #31 plus our **resnet18** experts, blend 0.5 | 0.953 | **−0.001. Our experts went negative.** They were worth +0.002 on the 0.949 host and +0.002 on the 0.950 one. Three points, monotone: *the host sets the price*, and this host finally knows our columns better than our resnet18 experts do |
| 33 | 2026-10-10 | `rsna-knee-s6-cnx` | `134d052`-dirty | #31 plus our **convnext_nano** experts, blend 0.5 — one variable against #32 | **0.957** | **+0.003 over the host, +0.004 over #32. Our best. Rank 173 → 61 of 5754.** Out of fold the same change predicted +0.0025 of macro, so it transferred at better than 100 % |
| 34 | 2026-10-10 | `rsna-knee-s6-cnx-b60` | `134d052`-dirty | #33 at blend 0.6, one variable | 0.956 | **−0.001.** The fourth weight variant to lose to a flat 0.5, after `shrunk`, `corrected` and `divided` |

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

### Submitted 2026-10-12, read 2026-10-13

Host is heliosli's 0.956 throughout, `--weights flat40`, experts `rsna-knee-experts-cnx`.
**The published host returned 404 that evening** — the author took the notebook down —
so all five are grafted onto the vendored copy under
`notebooks/external/rsna-knee-helios-s6-5050-0956`, whose hash still matches `KNOWN`
(`f359aee…`). Every dataset it mounts is still up; only the notebook is gone.

| # | Kernel | One variable | What it reads |
|---|---|---|---|
| 35 | `rsna-knee-s7-mirror` | the depth-reversed sagittal reading | the mirror's worth against the standing 0.957 (+0.0006 projected) |
| 36 | `rsna-knee-s7-mirror-jitter` | five jittered readings on top of #35 | the jitter's worth (+0.0008 projected) **and whether 7 passes fit in the time the host leaves** |
| 37 | `rsna-knee-s7-void-orphans` | Effusion, Synovitis, Contusion, Fracture voided | the host's AUC where we have no expert |
| 38 | `rsna-knee-s7-void-ours` | our eight columns voided | the host's AUC where we do, so our true margin |
| 39 | `rsna-knee-s7-void-fracture` | Fracture alone voided | that column exactly, and with #37 the other three as a sum |

**#35 and #36 are below the noise floor and known to be.** Their projections are 0.0006
and 0.0008 against a leaderboard that resolves ~0.005, so neither is expected to read as
anything; they are pushed because the out-of-fold measurement behind them is solid
(t = 3.39 and 7 of 8 columns) and because the code has to ship to be in the final
selection at all. **#37–#39 are the only rows here that clear the floor** — a column at
0.85 moves the macro by 0.029, six times the resolution — which is why three of five
submissions went to measurement rather than to score.

**#38 is also a check on the method.** `#37 + #38 − 0.5` must reconstruct the host's own
0.956. If it does not, the voiding arithmetic is wrong and nothing built on #37 or #39
should be believed.

Two GPU sessions run at once and each host run is ~8 h, so #37–#39 are queued by
`/tmp/push_queue.sh` and go out as slots free, not tonight.

### What the five said

| # | Kernel | Score | Reading |
|---|---|---|---|
| 35 | `rsna-knee-s7-mirror` | **0.957** | unchanged, as written down beforehand |
| 36 | `rsna-knee-s7-mirror-jitter` | **0.957** | unchanged, likewise |
| 37 | `rsna-knee-s7-void-orphans` | **0.807** | our eight columns sum to 7.684, **mean 0.9605** |
| 38 | `rsna-knee-s7-void-ours` | **0.647** | the four orphans sum to 3.764, **mean 0.9410** |
| 39 | `rsna-knee-s7-void-fracture` | **0.916** | Fracture alone at **0.956 to 0.980** |

**#35 and #36 behaved exactly as predicted, and that is the point of having predicted
it.** +0.0006 and +0.0008 against a leaderboard resolving ~0.005 read as nothing. The
out-of-fold measurement behind them stands (t = 3.39 over 15 paired folds, and +0.0009 a
column over eight for the jitter); what this says is that the leaderboard cannot see
them, not that they are absent. They ship because they cost nothing and the code has to
be in the final selection to be in it at all.

**The probes are the result, and they close a direction.** The host reads **0.9605 in the
mean on the eight columns we own** and **0.9410 on the four we do not** — Fracture itself
is the strongest column in the whole submission at 0.956 or better. There is **no soft
column among the twelve**: the weakest of the orphans averages 0.928 to 0.936. So no
orphan expert will pay, and Fracture, nominated twice on intuition, is refuted on a
number.

**The consistency check does not quite close, and the gap is unexplained.**
`#37 + #38 − 0.5` reconstructs the host at **0.9540** against the **0.956** it scores
bare — −0.002, where three-decimal rounding on each probe allows only ±0.001. Candidates,
none verified: the host's own run-to-run variation under its `DEADLINE_HOURS` guard (the
same host has read 0.956 twice and 0.957 once with our leg), or a column the metric
treats differently from the arithmetic above. **Every per-column figure here should carry
that ±0.002, and the ranking between them is what survives it, not the third decimal.**

A caution on reading these against our own numbers: the host's column AUCs are against a
radiologist's labels on the hidden test, and our experts' out-of-fold figures are against
`v4_blend` on training studies. Weak labels attenuate an AUC, so **0.9605 against our
0.8958 on Lateral Meniscus is not a comparison** and must not be read as one.

### 0.959, and the contribution is additive rather than capped, 2026-10-16

Six grafts onto the 0.956 host all read **0.957**, and the graft onto the 0.9573 also read
0.957. Two readings fit that: our leg adds a flat **+0.001** to whatever host it sits on,
or it saturates at 0.957 and the host underneath stops mattering. The two are the same
number on every submission made until now and they imply opposite strategies — under
saturation, grafting onto a better host is wasted work.

`s13-h958` discriminates them, and it reads **0.959**. So

| host, bare | with our leg | delta |
|---|---|---|
| 0.956 | 0.957 | +0.001 |
| 0.9573 | 0.957 | ~0.000 |
| **0.958** | **0.959** | **+0.001** |

**Additive.** The 0.9573 row stays the odd one and three-decimal rounding does not quite
excuse it (0.9573 + 0.001 = 0.9583, which displays as 0.958), so that submission is the
one to distrust, not this one. **Grafting onto each new public host is worth doing**, and
the +0.001 has now been reproduced across a 0.002 change of host.

**Both regimes earned it, and that is the second result.** They went out a minute apart
and read identically:

| regime | allocation | score |
|---|---|---|
| `flat40` | all eight columns at 0.40, **MCL included** | **0.959** |
| `parity` | seven at 0.50, **MCL handed back to the host** | **0.959** |

`parity` predicted it would beat `flat40` and did not. But it did not lose either, so this
is **not the fifth failed reweighting — it is a tie**, and the difference between the two
*is* what our MCL column is worth, which is therefore under 0.001.

**That closes the weight dimension.** Seven allocations have now been submitted — `flat`,
`shrunk`, `corrected`, `divided`, flat 0.6, `flat40`, `parity` — and the whole spread
between the best and worst of them is 0.002, with every pair inside 0.001 of its
neighbour. **No further submission should be spent on a reweighting.** The out-of-fold
optimum per column ranges 0.45 to 0.77 and its mean is 0.51, which is the arithmetic
reason a flat 0.5 keeps matching everything: the macro averages twelve columns, so a
per-column misallocation of 0.1 of weight costs a fraction of a thousandth.

**Which to select, since the leaderboard cannot choose.** `parity`, on the per-column
measurements rather than on the score. MCL is the host's **best** probed column (0.992)
and the one where our expert is furthest behind, and `flat40` blends ours into it at 0.40
while `parity` does not. On the seven that remain, the solo probes put us within **0.012
to 0.024** of him — equal quality, where 0.5 is the decorrelation optimum rather than a
default. Every per-column number points the same way, and none of them is the out-of-fold
comparator that caused the four earlier failures.

### The runs we had already thrown away, averaged, 2026-10-18

Four days from the deadline the proposed lever was to retrain with the 58 gold studies
folded back in. **It is worth +0.00018 of macro and was dropped on arithmetic alone**:
4348 to 4407 studies is 0.019 of a doubling, the learning curve is +0.016 of AUC per
doubling ([[data-is-the-binding-constraint]]), so +0.0003 a column — three times under
the leaderboard's rounding and thirty times under the noise of our own paired comparison.
Nor does it pay through label quality: 59 radiologist readings among 4348 extractor labels
do not move a fit.

Looking for a replacement turned up the largest lever of the programme, and it needs **no
training at all**. Every column has between 4 and 16 runs on disk from earlier
experiments, each with a `holdout.csv` over the same 4348 studies and the same
report-hash folds. Averaging their **ranks** — not their probabilities, which are not
calibrated against each other — prices out as:

| k runs a column | gain a column | signs | macro over 8 |
|---|---|---|---|
| 2 | **+0.0083** | 39/40 | +0.0056 |
| **3** | **+0.0117** | 39/40 | **+0.0078** |
| 4 | +0.0122 | 38/40 | +0.0081 |
| 5 | +0.0134 | 34/35 | +0.0090 |

The rule is fixed in advance — the shipped run plus the next best by out-of-fold, no
search over pairs — and every figure is the paired per-fold delta, which is the only
predictor that has ever had the right sign ([[gold-58-is-not-a-predictor]]). The third
run adds +0.0034 and the fourth +0.0005, so **k = 3 is where it stops**.

| column | shipped run | partner, by the fixed rule | one | two | signs | ROI cuts |
|---|---|---|---|---|---|---|
| **ACL** | `acl_cnxnano` | `acl_v4` | 0.8781 | **0.8978** | 5/5 | **1, shared** |
| **MCL** | `mcl_cnxnano` | `mcl_cnx_lr1e-4` | 0.8200 | **0.8314** | 5/5 | **1, shared** |
| **PF OA** | `pf_oa_wide_lr1e-4` | `pf_oa_cnxnano` | 0.8488 | 0.8597 | 5/5 | 2 |
| Lateral Meniscus | `lm_wide_s11_lr1e-4` | `lm_d32_cnx_jev` | 0.9031 | 0.9126 | 5/5 | 2 |
| Medial OA | `medial_oa_wide_lr1e-4` | `medial_oa_cnxnano_b16` | 0.8887 | 0.8949 | 5/5 | 2 |
| Medial Meniscus | `mm_wide_s11_lr1e-4` | `mm_d32_s11_cnxnano` | 0.9213 | 0.9262 | 5/5 | 2 |
| Lateral OA | `lateral_oa_wide_lr1e-4` | `lateral_oa_cnxnano_b16` | 0.8500 | 0.8544 | 5/5 | 2 |
| Baker's | `baker_cnxnano_b16` | `baker_axial_pilkwang` | 0.9158 | 0.9191 | 4/5 | **1, shared** |
| **mean** | | | | | **+0.0088** | **13 cuts** |

**Read the shipped column off the published dataset, not off the inventory.** The first
version of this table had MCL's base as `expert_mcl_cnx_lr1e-4`; `bundle5` actually ships
`expert_mcl_cnxnano`, and the partner dataset was built with a second copy of that same
run. MCL with the right base is **+0.0114 over 5/5 folds and shares its cut** — the third
largest gain of the eight and free in cutting time, not the +0.0078 first recorded. The
duplicate would not have raised: the graft cell keys runs by directory name, so the copy
collapsed into one entry and MCL would silently have carried a single run.

**The mechanism is fully accounted for by one line of arithmetic.** An AUC counts
(positive, negative) pairs. On ACL the two runs agree on 87.6 % of them — both right on
81.0 %, both wrong on 6.5 % — and averaging can do nothing there. They disagree on
**12.4 %**, where the shipped run alone is right **47.8 %** of the time and their average
is right **62.4 %**. So `0.124 x (0.624 - 0.478) = +0.018` against **+0.0195** measured.
Averaging keeps the *magnitude* of each disagreement, so the confident reading takes the
pair, and an error specific to one crop or one encoder is diluted while the signal both
runs share is not.

**The partner is the weaker model on seven columns of eight** — `lateral_oa_coronal` is
0.028 below the run it joins — and the pair still wins every time. A run is worth keeping
because it errs *elsewhere*, not because it scores well. That is why an inventory we had
been treating as superseded drafts turned out to be the asset.

**What decorrelates them, measured as Kendall's tau between 0.45 and 0.70**: a different
ROI box (a different crop is different pixels and so different errors), a different
encoder, or a different label table. Medial Meniscus is decorrelated by **cropping
alone** — three boxes, one encoder — and still takes +0.0061.

**Choosing the partner per column is worth less than not choosing it.** The obvious
improvement on the fixed rule is to search, per column, for the partner that ensembles
best — the first exploratory sweep found pairs worth +0.0027 (PF OA), +0.0026 (Lateral
OA) and +0.0036 (Baker's) more than the ones shipped. Priced honestly, by selecting on
four folds and judging on the fifth the search never saw, rotated:

| | selection | fixed rule | |
|---|---|---|---|
| over 35 held-out folds | | | **−0.0006, 9/35 signs, p = 0.999** |
| both against the single run | +0.0078 | **+0.0084** | |

**The search is significantly worse**, and those three "missed" gains were its luck. ACL
is the clearest case: the search picks `expert_acl` — the *weakest* of seven ACL runs at
0.8626 — and picks it **5 times of 5**, so the choice is stable rather than erratic. It
wins on the four folds that chose it and loses 0.0018 on the fifth. Selection catches the
largest decorrelation visible in those studies, and that decorrelation does not
reproduce.

The same table carries the unbiased confirmation of the lever itself: **the fixed rule
beats the single run by +0.0084 on folds that chose nothing**, against the +0.0083 the
in-sample figure reported. The ensemble is real; only the choice of partner is not. So
k = 3 ships the three best by out-of-fold, with no search.

**And it refutes a row of the zero table.** "Averaging resnet18 with convnext_nano:
+0.00018" is contradicted by +0.0118 on PF OA, +0.0069 on Baker's and +0.0065 on MCL,
all same-box cross-encoder pairs. How the old figure was taken is not recorded; this one
is pre-specified, paired per fold, and 39/40.

**The sober number.** These are out-of-fold AUCs, i.e. agreement with the extractor, and
our ACL measures **0.980** on the hidden test — +0.0197 cannot be added to that.
Transferred as a relative error reduction on the three columns whose true AUC the solo
probes gave:

| column | out of fold | error cut | true AUC | expected |
|---|---|---|---|---|
| ACL | 0.8781 → 0.8987 | 16.9 % | 0.980 | 0.9834 (**+0.0034**) |
| PF OA | 0.8488 → 0.8671 | 12.1 % | 0.908 | 0.9192 (**+0.0112**) |
| Lateral OA | 0.8500 → 0.8605 | 7.0 % | 0.884 | 0.8921 (**+0.0081**) |

**Macro +0.0050 at k = 3**, and less again after the 0.5 blend with the host. So expect
**+0.002 to +0.004**, not +0.0078. At 0.959 with gold at 0.963 and the silver cut 0.001
below us, that is the range that decides something.

**What it cost in code.** `chain.py` raised when two runs claimed a column; it now holds
each run's **percentile rank over the studies that run could read** and writes one
average per column at the end, so a study only one run could cut keeps that run's reading
without a special case. Verified against the old path on synthetic data: with one run per
column the written column is **identical to the bit** at blend None, 0.5 and 0.4. A cut is
now also kept until its last reader has used it — the three shared-cut columns were
cutting the same region twice. k = 2 needs **13 cuts and 16 models** against 8 and 8.

### What 0.959 is worth in rank, 2026-10-16

Full public leaderboard, 6092 teams, snapshot 14:41 UTC:

| | rank | score |
|---|---|---|
| gold | 22 | **0.9630** |
| **us, 0.959** | **78 to 118** | the 0.959 band spans those ranks |
| silver | 305 | **0.9580** |
| bronze | 609 | 0.9570 |

**The density has climbed with the public notebook and that is the whole value of the
+0.001.** 499 teams sit at 0.958 or better and 1483 at 0.956 or better, because the 0.958
host is public and several hundred teams run exactly it. Our one thousandth is not worth
"a point", it is worth **~380 ranks**, because it lifts us off a plateau rather than
moving us along a gradient. The same structure was recorded on 2026-10-03 at a lower
altitude — 1387 teams between 0.940 and 0.943 — and the conclusion drawn there holds
twice now: **an orthogonal gain is worth far more than a tuned one**, and on the private
split the plateau reshuffles while our leg does not.

Gold is **+0.004** away. The only unspent lever of that size is the ACL weight, which is
0 today and which `h958-hostonly-acl` is being submitted to price: our expert measures
0.980, so handing it the column pays `(0.980 - auc_host) / 12`, between +0.0015 and
+0.0040 depending on where his ACL lands.

### A study is worth more than any lever we have found

Two runs of five folds, each fitted on a stratified subsample of its fold's training
studies with the **held-out portion untouched and its original labels kept**, so the
three runs are scored on the same thing.

| training studies | mean over 5 folds | paired against 100 % | t | folds worse |
|---|---|---|---|---|
| 3478 (100 %) | 0.8979 | — | — | — |
| 2435 (70 %) | 0.8896 | **−0.0082 ± 0.0028** | −2.97 | 4/5 |
| 1739 (50 %) | 0.8805 | **−0.0174 ± 0.0076** | −2.28 | 5/5 |

Monotone, and the two pairs agree on the slope independently: **+0.0160** of AUC per
doubling of data from the 70 % pair, **+0.0174** from the 50 % pair. That agreement is
the internal check — two different subsample sizes had no reason to imply the same slope
unless the curve is real.

**What it settles.** Dropping 30 % of the training studies costs **0.0082** of AUC; the
label filter that would drop them recovers under **0.7 points of a 2.8 % noise pool**
(two independent extractors agree on 97.2 % of cells). So the dropping form of that
filter is refuted by measurement rather than by assertion, and only the **downweighting**
form — which loses no studies — remains defensible.

**What it opens.** At +0.016 a column per doubling, and eight columns over twelve
targets, a doubling of the corpus projects to **+0.011 of macro** — three times what the
encoder change was worth, and more than every other lever in this file combined. MRNet's
1370 knees are 0.31 of a doubling, so **~+0.003 of macro**, and that is before any
discount for its different scanners, protocols and label set, and for the concavity that
makes the slope above 100 % smaller than the slope below it.

**The correction that goes with it.** Fold 0 alone read −0.0122 and was quoted here as
"five times the mirror" before the other four folds landed. The paired five-fold figure
is **−0.0082**, about 3.4 times. Same direction, same conclusion, smaller number — and
the single fold should not have been quoted.

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

### When the host is the comparator

Row #22 is the first time an out-of-fold figure arrived on the leaderboard at face value,
and the reason is worth keeping: **its host was the model the prediction was measured
against.** The kernel log shows
`window_30epochs_v4blend_goldin_fold0_336px: fingerprint matches within 0.00023`, and the
decomposition was computed from that run's own five holdouts. Predicted +0.00167 of macro,
read +0.002.

Every earlier expert row discounted by roughly two thirds — #14 → #15 moved +0.0101 out of
fold and +0.003 here — because there the Kaggle host was a *different* model, better on
the seven columns than ours is. The discount is not noise in the metric; it is the host.
**So an out-of-fold gain transfers in proportion to how closely the local comparator
matches the host it will actually blend against**, which is the argument for benching the
public pipeline over labelled studies before building anything else for it.

The decomposition itself is arithmetic, not an experiment: the experts own **disjoint**
columns and an AUC depends only on its own column, so contributions to the macro add with
no cross terms. Measured against our wide model **[m, 4348 studies]**, blended at 0.5:

| expert | column | wide | blend | outright | Δ macro |
|---|---|---|---|---|---|
| `expert_lm_d32` | Lateral Meniscus | 0.7736 | 0.8498 | 0.8476 | **+0.00635** |
| `expert_acl` | ACL | 0.8065 | 0.8649 | 0.8626 | +0.00486 |
| `expert_pf_oa` | PF OA | 0.8083 | 0.8527 | 0.8305 | +0.00370 |
| `expert_medial_meniscus` | Medial Meniscus | 0.8612 | 0.8964 | 0.8791 | +0.00294 |
| `expert_lateral_oa_coronal` | Lateral OA | 0.8180 | 0.8446 | *0.8121* | +0.00221 |
| `expert_mcl_fsonly` | MCL | 0.8135 | 0.8382 | *0.8042* | +0.00206 |
| `expert_baker_axial` | Baker's | 0.8960 | 0.9161 | 0.9063 | +0.00167 |
| `expert_medial_oa` | Medial OA | 0.8652 | 0.8848 | *0.8611* | +0.00163 |

Sum: **+0.0254** blended, +0.0134 outright. Three things it settles.

**Blending beats handing the column over, on all eight** — and on the three in italics
*outright scores below the wide model alone*, so replacing would destroy information. An
aggregate total hides that completely.

**The gain tracks the host's weakness, not the expert's quality.** Lateral Meniscus pays
most because the wide model is worst there (0.7736). Baker's pays least although it is our
best expert, because the wide model already reads 0.8960 on that column.

**Per-expert submissions are therefore not worth running.** Eight runs would reproduce this
table, and half of its rows move the macro by less than the leaderboard can resolve.

### A fusion that is unbiased and imprecise

Rows #24 and #25 contradict the transfer rule of the section above, and the error is
worth keeping because it was mine twice over.

Two experts on the same boxes, the same folds and the same architecture, differing only
in which label table they were fitted on, correlate in rank **0.755** (0.647-0.870)
**[m, 4348 studies]** — as decorrelated as two *different architectures* in the public
stack, whose legs sit at 0.75-0.89. Both score the same on the leaderboard (0.925 and
0.924). That is the textbook condition for averaging to pay, and `ρ = 0.755` over two
members predicts an error-variance factor of 0.878, a 6% cut in error SD.

Averaged on the 58 gold studies, the fusion beat pilkwang on **5 of 6** columns and
v4_blend on 6 of 6, summing **+0.0396** — which was read here as +0.0033 of macro and
predicted to show. It showed **0.000** (#25).

**Why the reading was wrong.** Averaging carries no selection bias, which is a real
advantage over picking the better of two per column — that scheme would have harvested
+0.034 where pure noise alone produces +0.039, so it measures nothing. But *unbiased is
not precise*: the SE of that +0.0396 over six columns is 0.03-0.05, so it sat within one
standard error of zero. The whole day's warning about 58 studies was applied to the
selection case and forgotten for the averaging case.

**And the transfer went the other way than predicted.** The fusion gained +0.003 on the
strong host (#24, which also added Baker) and nothing on the weak host whose model *was*
the comparator (#25). Taking #25 at face value — it is the clean one-variable test, with
Baker on both sides — **Baker alone carried #24's +0.003**, against a host better than
ours on every column it does not own. The rule from the previous section survives only as
a statement about #22; it does not generalise.

What is kept: ρ = 0.755 is measured and labels-free, the fusion does not *hurt*, and ten
folds cost only forward passes. What is dropped: any expectation of gain from it.

### What the three stack regimes did and did not separate

Five submissions on 2026-10-07, rows #26-#30. The design held one knob fixed per pair so
that no contrast would confound two changes the way #17/#18 did. Three of the four
readings are at or under the leaderboard's resolution, and saying which is which is the
whole value of the night.

**Read as a curve, the host weight is flat.** d4's share against `nartaa`'s CoAtNet:

| d4's weight | row | score |
|---|---|---|
| 0.00 | #27 | 0.951 |
| 0.15 | #29 | 0.949 |
| 0.30 | #28 | **0.952** |

A real effect that is monotone in a weight does not dip in the middle. It dips here, so
the 0.003 spread is most consistent with leaderboard noise of about ±0.0015, and the
honest conclusion is that **d4's weight changes nothing measurable over [0, 0.30]** — not
that 0.30 is the optimum. The practical consequence is the opposite of flattering: if the
28-model public leg is worth nothing on top of leg B, it is 2 h 45 of compute we can stop
paying, and 0.951 from `nartaa` plus our experts alone is within noise of our best score.

**Two readings are above the floor, and they point the same way.**

Keeping d4's four CoAtNet readers costs **−0.004** (#26's 0.948 against #28's 0.952).
That is the largest difference of the night and it is the third independent measurement
agreeing: row #19 priced the amputation at −0.003 in the other direction, and the readers
are 56.7 % of the pipeline's compute. `nocoat` is settled.

Our eight experts are worth **+0.002** on a 0.949 host (#27 against `nartaa`'s published
figure). Against `nocoat` they bought +0.004. The halving was predicted before the
submission, from their own gate-58 gains landing on four of the columns our experts own —
Lateral Meniscus +0.071, Lateral OA +0.048, Medial Meniscus +0.035, ACL +0.028. An expert
is worth what the host does not already know, and this host learnt some of it.

**The one clean single-knob contrast went against the request that produced it.** #30
raised the expert weight from 0.50 to 0.65 with the host mix untouched, and lost 0.002.
`EXPERT_BLEND`'s own comment had already said why: the 0.5 was fitted out of fold against
a host scoring 0.835 internally, and "a stronger host deserves less weight, so 0.5 is
probably too generous here". The host went from 0.943 to 0.949 and the measurement
followed the comment. 0.002 is still near the floor, but unlike the host-weight series
there was a prior, the contrast moves one knob, and the sign matches. **The next thing to
test on this axis is 0.35, not 0.8.**

### The encoder was the only lever that paid

Three days and forty-five runs. One change moved the leaderboard; everything else is
written below as a measured zero, which is the more useful half of the record.

**`convnext_nano` against `resnet18`**, same boxes, same label table, same batch, same
learning rate — one key in the config. Paired per fold, because the fold-to-fold spread
(0.018 to 0.057 depending on the column) swamps the effect when averaged:

| column | resnet18 | convnext_nano | Δ | signs |
|---|---|---|---|---|
| Lateral Meniscus | 0.8856 | 0.8978 | +0.0123 | `+++++` |
| MCL | 0.8097 | 0.8200 | +0.0103 | `-++++` |
| Medial OA | 0.8745 | 0.8844 | +0.0099 | `-++++` |
| Lateral OA | 0.8331 | 0.8390 | +0.0059 | `-++++` |
| Medial Meniscus | 0.9058 | 0.9104 | +0.0046 | `+-+++` |
| PF OA | 0.8402 | 0.8445 | +0.0043 | `++++-` |
| Baker's | 0.9134 | 0.9158 | +0.0024 | `++--+` |
| ACL | 0.8832 | **0.8781** | **−0.0051** | `-+---` |

**+0.0056 ± 0.0016 over 40 pairs, t = 3.56, seven columns of eight.** The three columns
whose boxes did not fit at batch 32 had their resnet18 baseline retrained at batch 16 as
well, so every pair differs by the encoder alone.

**It transferred at better than 100 %**, which nothing else this project measured has
done. The slope fitted out of fold — 0.0555 of macro per point of column AUC — predicted
+0.0025 of macro for eight columns; the leaderboard gave **+0.004** (#33 against #32).
The reading is that a stronger host devalues the *decorrelation* term of a blend while
leaving the *quality* term intact, so against a 0.954 host the only thing that pays is
being better, not being different.

**Why resnet18 had to be replaced, not supplemented.** Averaging the two encoders was
measured on the three columns where both sides have a trustworthy holdout: it lifts the
column (+0.0087, +0.0101, +0.0029 over the better of the two) and then buys **+0.00018**
of macro once blended against the host. The blend against the host is already a 50/50
rank average, and the same noise cannot be cancelled twice.

**What ACL costs us, and what it is not.** Its two arms both converged cleanly — zero
epochs under 0.60 for convnext against one per fold for resnet18 — so the −0.0051 is not
a training accident. At `lr = 1e-4` it recovers to −0.0007, so part of it was the learning
rate. No mechanism is claimed beyond that.

### A nuisance axis we had been paying for, 2026-10-12

`roi.extract.choose_slices` returns the kept slices **in stack order**, and a sagittal
stack's order is anatomical only up to the knee's side: increasing through-plane position
runs medial-to-lateral on a left knee and lateral-to-medial on a right one.
`bowtie_direction` fixes the depth window's *extent* — 8 mm toward the bowtie against 24
toward the notch on `lateral_meniscus_d32` — and **nothing fixes its direction**.

Confirmed on the cached crops before any GPU was spent, by the per-slot lit-area profile:

| region | plane | corr(L, R) | corr(L, reversed R) |
|---|---|---|---|
| `lateral_meniscus_d32` | Sagittal | +0.223 | **+0.773** |
| `baker` | Sagittal | −0.068 | **+0.887** |
| `medial_meniscus` | Sagittal | −0.503 | **+0.872** |
| `acl` | Sagittal, symmetric window | +0.899 | +0.722 |
| `mcl` | Coronal | **+0.973** | −0.902 |
| `medial_oa` | Coronal | **+0.864** | −0.280 |

The two coronal rows are the control, and they come out the way the code says they must:
there the depth axis is anterior-posterior, laterality is in-plane, and `crop_plane`
already mirrors it so lateral is on the right. `acl`'s window is centred between the two
meniscus points and so is palindromic — the profile cannot decide it, though the code can.

**This is not `PixelRules.sagittal_flip`.** That flag (also False, the public baseline's
dropped `torch.flip`) governs the whole-image cache; the ROI depth ordering is re-derived
from geometry and carries the nuisance independently of it.

So both signs are in the training data in roughly equal number, reversing the axis is an
input the folds have **already seen**, and averaging the two readings costs one forward
pass and no training:

| column | plain | mirror alone | averaged | paired Δ | SE | folds won |
|---|---|---|---|---|---|---|
| Lateral Meniscus | 0.8958 | 0.8955 | **0.8989** | +0.0028 | 0.0015 | 4/5 |
| Medial Meniscus | 0.8994 | 0.8974 | **0.9027** | +0.0027 | 0.0008 | 5/5 |
| ACL | 0.8693 | 0.8627 | **0.8701** | +0.0017 | 0.0015 | 3/5 |

**Pooled: +0.0024 ± 0.0007 over 15 paired folds, t = 3.39** — as significant as the
encoder change (t = 3.56) and free. The identity reading reproduces each run's own
`holdout.csv` to four decimals, which is what makes the rest of the table readable.

**Do not canonicalise instead.** The reversed reading *alone* scores 0.8955 against
0.8958, so the encoder is already invariant and there is no capacity left to recover;
what the averaging buys is an ensemble of two views agreeing only at Spearman 0.92.
Ordering the slots by signed anatomical distance would remove the nuisance **and** the
free ensemble.

**What it is worth, stated small.** Only **3 of 12 columns are sagittal** —
`lateral_meniscus_d32`, `medial_meniscus`, `acl`; the shipped Baker's is `baker_axial`,
which the guard in `mirror_groups` refused rather than reverse. Sum of per-column gains
0.0072, so the projected macro is **+0.0006**, under what three decimals shows.

One docstring is wrong because of this and should be corrected, not trusted:
`src/rsna/expert/augment.py` refuses depth-reversal augmentation because "the crop is
already oriented — the ROI extractor put the bowtie on a known side". It put the bowtie
on a known *extent*.

### The depth sweep the medial compartment never had, 2026-10-13

The shipped pair was lopsided by accident of order, not by a finding:
`lateral_meniscus_d32` carries 8 + 24 mm over 9 slots because that sweep ran and won,
while `medial_meniscus` was still on the original 4 + 12 over 5. `medial_meniscus_d32`
copies the lateral window onto the medial box and **changes nothing else** — verified
field by field, `box_w_mm`, `box_h_mm`, `out_w`, `out_h`, `landmark`, `plane`, `series`,
`box_rise_mm`, `box_offset_mm` and `decimate_to_mm` all identical, so the resolution
stays exactly 0.2143 mm/px. Memory was measured rather than assumed (21.2 GiB of peak at
batch 32 against 9.1 for the 5-slot box), so the batch did not have to move either.

| fold | `medial_meniscus` | `medial_meniscus_d32` | paired Δ | kept epoch |
|---|---|---|---|---|
| 0 | 0.9166 | **0.9211** | +0.0045 | 13 → 19 |
| 1 | 0.9041 | **0.9073** | +0.0032 | 13 → 12 |
| 2 | 0.9093 | **0.9226** | +0.0133 | 16 → 22 |
| 3 | 0.9020 | **0.9071** | +0.0051 | 5 → 12 |
| 4 | 0.9202 | **0.9222** | +0.0020 | 23 → 22 |

**Paired: +0.0056 ± 0.0020, t = 2.82, 5 of 5 folds** — the same size as the encoder
change (+0.0056, t = 3.56) which was worth +0.004 of leaderboard over eight columns. So
the projection for one column is **+0.0005** of macro. Gold went 0.9447 → 0.9615 and is
not counted: 58 studies read ±0.05.

**The out-of-fold figure moved much further and most of it is not quality.**

| | mean of the folds | OOF | pooling cost |
|---|---|---|---|
| `medial_meniscus` | 0.9104 | 0.8994 | **−0.0111** |
| `medial_meniscus_d32` | 0.9161 | 0.9146 | **−0.0015** |

0.8994 → 0.9146 is +0.0152, of which +0.0056 is per-fold quality and ~+0.0096 is
**cross-fold calibration recovered**: the longer window pools 7 depth windows instead of
3, and the five models end up agreeing about the scale of their outputs. That is a real
property — the submission averages the folds, so better-calibrated folds average better
— but only the +0.0056 is calibrated against a known leaderboard outcome, so only that
is projected.

**Why the acquisition allowed it, where the patellofemoral box would not.** A longer
window at the same 4.0 mm pitch fills *better*: the 5-slot medial window finds 4.40
slices of 5 (88 %) and the 9-slot one 8.41 of 9 (**93.4 %**), because more acquired
slices fall inside and `choose_slices` thins a surplus down to `slots` where a narrow
window cannot be topped up. `pf_oa`'s axial stacks find only 7.80 of 9, so lengthening
*that* window buys padding — which is why the same proposal was refused there on
2026-10-13 before any GPU was spent.

**Two things the kept epochs say.** They slid later — 13→19, 16→22, 5→12 — which is what
seven windows to pool rather than three should do, and it means the 30-epoch budget is
being used rather than wasted.

### A larger frame around the same window, stopped at four folds

`medial_meniscus_d32_wide` holds `d32`'s depth, slots, landmark, plane, series and
millimetres per pixel and moves only the in-plane extent: 54 x 33 to **63 x 39 mm** on
294 x 182 px, +16.7 % and +18.2 % of side and +37.9 % of area. The sizes are forced —
holding 3/14 mm per pixel with both output sides divisible by 14 makes 63 and 39 the
nearest expressible box to a flat +15 %.

Two things were checked before the GPU was spent, and both said go. Sagittal
acquisitions are 0.332 mm/px against this crop's 0.214, so it **oversamples by 1.55x**
and a larger box at a fixed pixel count would still oversample by 1.35x — nothing
acquired is lost, unlike on `pf_oa` where the margin is 1.09x. And measured on the crop,
every one of the four edges carries **more** tissue than its centre (0.81 to 0.91 against
0.77) with **no study** having the box run off the acquired image, so the added area
would be filled with real pixels rather than padding.

| fold | `d32` | `d32_wide` | paired Δ |
|---|---|---|---|
| 0 | 0.9211 | 0.9092 | **−0.0119** |
| 1 | 0.9073 | 0.9088 | +0.0015 |
| 2 | 0.9226 | 0.9254 | +0.0028 |
| 3 | 0.9071 | 0.9071 | ±0.0000 |
| 4 | 0.9222 | — | diverged at epoch 7, loss 0.4545 → 0.6665, AUC 0.9127 → 0.5935 |

**Mean of the four clean folds: −0.0019.** Against `d32`, which was positive on five of
five. Stopped there rather than finished: four folds at zero do not become a gain on the
fifth, and the fifth had derailed anyway. Folds 0–3 are kept as weights.

**So the mechanism question is answered, and it was the first reading.** `d32` paid
because it reached more *meniscus* — the posterior horn runs along the depth axis — and
not because context or pooled windows reduce variance. In plane, 54 mm already covers a
35–40 mm meniscus with margin, and the 4.5 mm added front and back is fat pad and
popliteal tissue while the 3 mm above and below is condyle and plateau. Tissue, measured;
not signal. **Enlarging a box toward the host's framing does not pay** — which is what
[[experts-what-pays]] has said since Baker's and Effusion, now with a third case.

### Three changes at once, on both meniscus columns, 2026-10-13/14

Not an experiment and recorded as such: box, slot pitch and learning rate all move, so a
row that shifts cannot say which moved it. It was asked for as a bid for the best model
after the three components had been measured apart, and the components said it should
lose.

| | fold 0 | 1 | 2 | 3 | 4 | paired Δ | signs |
|---|---|---|---|---|---|---|---|
| medial, `d32` → `wide_s11_lr1e-4` | −0.0052 | +0.0113 | +0.0061 | +0.0049 | +0.0092 | **+0.0053 ± 0.0028** | 4/5 |
| lateral, `d32` → `wide_s11_lr1e-4` | +0.0034 | +0.0150 | +0.0037 | +0.0027 | +0.0012 | **+0.0052 ± 0.0025** | **5/5** |
| **pooled** | | | | | | **+0.0052** | **9/10, sign test p = 0.011** |

**Per column that is the size of the encoder change (+0.0056), and it is significant.**

**Adding the measured components up predicted the wrong sign on the lateral column.**
There, `lr = 1e-4` had been run on `lateral_meniscus_d32` itself, one variable, and cost
**−0.0104**; the frame measured −0.0019 on the medial side and the pitch closed as null
at 5 of 10 signs. The sum predicted about **−0.012** and the measurement returned
**+0.0052** — a 0.017 miss *and* the opposite direction. On the medial side the same
arithmetic summed to +0.0138 against a measured +0.0053, over by 2.6x. So component sums
are not a usable predictor of a bundle here, in either magnitude or sign, and three
bundles asked for against that reasoning have now won.

**What did not improve: the out-of-fold figure on the lateral column.** 0.8958 → 0.8944,
because the cross-fold pooling cost went from −0.0021 to −0.0086 — the five models agree
less about the scale of their outputs, the opposite of what `medial_meniscus_d32` did
(−0.0111 → −0.0015). Per-fold quality and cross-fold calibration moved in opposite
directions, and only the first is what the paired comparison measures. The submission
averages the folds rather than concatenating them, so the effect there is indirect; it is
not nothing.

**Projected macro: +0.0009** — two columns of twelve at +0.0052, by the calibration that
reproduced the encoder change to 7 %. The family every lever has landed in since.

Also noted: the kept epochs came early on the lateral run — 15, 12, 4, 8, 9 — which is
not what a lower learning rate usually does, and is worth a look if this configuration is
taken further.

### The same bundle on MCL: nothing, 2026-10-14

`mcl_tall_d36` + `lr = 1e-4` against the shipped `mcl_fsonly`, two keys apart.

| fold | `mcl_fsonly` | `mcl_tall_d36` + lr | Δ | kept epoch |
|---|---|---|---|---|
| 0 | 0.8271 | 0.8309 | +0.0038 | 2 → 9 |
| 1 | 0.7877 | 0.7962 | +0.0085 | 14 → 5 |
| 2 | 0.8404 | 0.8324 | **−0.0080** | 14 → 10 |
| 3 | 0.8287 | 0.8288 | +0.0001 | 14 → 7 |
| 4 | 0.8160 | 0.8018 | **−0.0142** | 24 → 5 |

**−0.0020 ± 0.0041, t = −0.48, 3 of 5, sign test p = 0.50.** Nothing. And the out-of-fold
figure fell much further — 0.8160 → **0.8016** — because the cross-fold pooling cost blew
out from −0.0040 to **−0.0164**. Gold was flat (0.9229 → 0.9206).

Pooled over the three bundles the effect survives but is diluted: **12 of 15 folds
positive, p = 0.018**, against 9 of 10 at p = 0.011 before this run.

**One hypothesis is refuted and one stands.** It is *not* that a bigger box needs more
positives than this column has: Lateral Meniscus carries **668** positives and gained,
MCL carries **660** and lost. What distinguishes MCL is that it is the only one of the
three where the box was **moved** and not only enlarged — `box_rise_mm` from −12 to −4,
8 mm upward — where both meniscus bundles grew symmetrically about an unchanged centre.
Repositioning is the category this project has repeatedly measured at zero or worse; the
Effusion expert lost 0.025 for bringing only better centring.

The kept epochs point the same way: 2/14/14/14/24 became 9/5/10/7/5, so the new
configuration peaks **earlier** on four folds of five. With a *lower* learning rate that
is backwards, and looks like faster overfitting — which a 40 % taller box feeding twice
the windows would explain.

**Testable at one variable:** hold the 112 mm height and put `box_rise_mm` back to −12,
so the box grows +16 mm up and +16 mm down about the centre it already had. If that comes
back positive, the cost was the move rather than the size.

### The bundle on a fourth column, and the four of them together

`pf_oa_wide_d40` + `lr = 1e-4`, two keys from the shipped run. Batch stayed at 32 on both
sides (32.4 GiB of peak against 44.3), so no fourth variable.

| fold | `pf_oa` | bundle | Δ |
|---|---|---|---|
| 0 | 0.8659 | 0.8582 | **−0.0077** |
| 1 | 0.8309 | 0.8309 | ±0.0000 |
| 2 | 0.8491 | 0.8565 | +0.0074 |
| 3 | 0.8432 | 0.8487 | +0.0055 |
| 4 | 0.8336 | 0.8495 | **+0.0159** |

**+0.0042 ± 0.0039, t = 1.08, 3 of 5 positive, sign test p = 0.50.** Positive in the mean
and the size of the meniscus results, too scattered to carry on its own. Out of fold
0.8435 → 0.8448; gold 0.9009 → 0.9176 and not counted.

| column | paired Δ | signs |
|---|---|---|
| Medial Meniscus | **+0.0053** | 4/5 |
| Lateral Meniscus | **+0.0052** | 5/5 |
| MCL | −0.0020 | 3/5 |
| PF OA | **+0.0042** | 3/5 |
| **pooled** | **+0.0032** | **15/20, p = 0.021** |

**The bundle holds across four columns and is uneven inside them.** The pooled sign test
is significant, the mean falls from the +0.0052 of the two meniscus columns to +0.0032
over four, and MCL contributes nothing. **Projected macro +0.0011.**

**On what to ship.** Choosing the bundle column by column on these same paired deltas is
selection on noise, which this file has been burned by before. It barely matters here:
shipping all four projects +0.00106 and keeping only the positive three +0.00123. So the
defensible rule is the simple one — take the bundle where the delta is positive, keep the
shipped expert on MCL where it is indistinguishable from zero — and the choice is worth
0.0002 either way.

### The bundle on the two coronal OA columns, and the six of them pooled

On these two the bundle is **the box alone**: their shipped runs already use
`lr = 1e-4`, so only the region moved — 54 x 39 to 60 x 42 mm and 28 mm over 9 slots to
35 over 11, the 3.50 mm pitch left at the coronal acquisition's own rate. Batch stayed at
16 on both sides. One key apart.

| | fold 0 | 1 | 2 | 3 | 4 | paired Δ | t | signs |
|---|---|---|---|---|---|---|---|---|
| Medial OA | +0.0125 | −0.0025 | +0.0027 | +0.0015 | +0.0075 | +0.0043 ± 0.0026 | 1.68 | 4/5 |
| **Lateral OA** | +0.0085 | +0.0105 | +0.0130 | +0.0040 | +0.0189 | **+0.0110 ± 0.0025** | **4.45** | **5/5** |

**Lateral OA is the largest single-column gain of the whole programme** — double the
encoder change — and its out-of-fold figure rose with it, 0.8308 → 0.8404. Medial OA is
positive and inconclusive alone, with a flat out-of-fold (−0.0005) because its pooling
cost widened from −0.0084 to −0.0132.

**A prediction made before the run came true, for the first time this session.** Lateral
OA is the column the void probes had just priced as the **host's weakest (0.896)**, with
the **smallest gap to our expert (+0.065)** of the four probed. [[experts-what-pays]]
records that the gain tracks the host's weakness rather than the expert's quality, and
that is exactly where the bundle paid most.

| column | paired Δ | signs |
|---|---|---|
| **Lateral OA** | **+0.0110** | 5/5 |
| Medial Meniscus | +0.0053 | 4/5 |
| Lateral Meniscus | +0.0052 | 5/5 |
| Medial OA | +0.0043 | 4/5 |
| PF OA | +0.0042 | 3/5 |
| MCL | −0.0020 | 3/5 |
| **pooled** | **+0.0047** | **24/30, p = 0.00072** |

**Projected macro +0.0023** over six columns, **+0.0025** shipping only the five positive
ones — double the estimate from four columns and the largest measured lever since the
encoder.

**And it corrects a conclusion drawn three days earlier.** `medial_meniscus_d32_wide`
measured −0.0019 and was written down as "enlarging a box toward the host's framing does
not pay". On these two coronal columns the enlargement *is* the whole change, with no
learning rate to carry it, and it took +0.0043 and +0.0110. **That conclusion does not
generalise**: the box pays on the coronal OA columns and not on the sagittal meniscus, and
nothing measured here says why.

### What the per-fold AUC is and what the out-of-fold figure is

They are not the same measurement and the gap is **not** epoch-selection bias. Taken from
one run's `holdout.csv`, with nothing retrained and nothing reselected:

```
fold 0  0.9166   fold 3  0.9020      mean of the folds  0.9104
fold 1  0.9041   fold 4  0.9202      AUC of the whole   0.8994   <- the OOF
fold 2  0.9093                       gap               -0.0111
```

Both numbers come from the same predictions, so any selection bias inflates them
equally and cannot be the difference. The cause is **pooling**. An AUC counts correctly
ordered positive/negative *pairs*, and of this run's 4,547,347 pairs only **20 % are
within a fold** — the other **80 % oppose a study scored by fold 1's model to one scored
by fold 3's**, two models whose outputs are not calibrated to each other.

Demonstrated: re-ranking each fold to within-fold percentiles before pooling, which
cancels any scale difference between folds without touching the order inside one,
recovers **+0.0108 of the 0.0111**. The gap was calibration, 97 % of it. The project had
already met this cost at 0.0050 on Baker's and 0.0158 on `expert_mm_sagcor`.

**So: compare two runs fold-paired** — same held-out studies, the pooling penalty absent
from both sides. The OOF stays the run's headline figure, but it *understates* what
ships, because the submission averages the five folds per study and so has no cross-fold
pairs at all.

### The host priced column by column, 2026-10-13/14

Five submissions, each the unmodified host with one column replaced by a constant. A
constant column ties every study, and an AUC counts tied pairs as half, so such a column
scores **exactly 0.5** — which inverts to

```
auc_column = 0.5 + 12 x (macro_host - macro_voided)
```

**`macro_host` is 0.954, not the 0.956 the host scores bare**, and that is now fixed by
two independent routes. The two block probes give it directly: voiding our eight columns
read 0.647 and voiding the four orphans read 0.807, and their sum is `macro_host + 0.5`,
so `0.807 + 0.647 - 0.5 = 0.954`. And requiring every per-column AUC to land inside
[0.5, 1.0] forces `macro_host` into **[0.9210, 0.9547]** — which excludes 0.956 and
admits 0.954. With 0.956 the MCL column computes to **1.016**. The 0.002 discrepancy
against what the host reports is unexplained; candidates are its own run-to-run variation
under `DEADLINE_HOURS`, or something about how the metric treats a degenerate column.

| column | host AUC | our out-of-fold | submission |
|---|---|---|---|
| **MCL** | **0.992** | 0.8160 | `void-mcl` 0.913 |
| Lateral Meniscus | 0.968 | 0.8958 | `void-latmen` 0.915 |
| Fracture *(orphan)* | 0.956 | — | `void-fracture` 0.916 |
| PF OA | 0.932 | 0.8435 | `void-pfoa` 0.918 |
| **Lateral OA** | **0.896** | 0.8308 | `void-latoa` 0.921 |

Block figures from the same arithmetic: our eight columns average **0.9605**, the four
orphans **0.9410**, and the four of ours not probed individually — ACL, Medial Meniscus,
Medial OA, Baker's — average **0.9740**.

**Read the ranking, not the third decimal.** An absolute AUC carries **±0.030**:
12 x (0.002 of uncertainty on `macro_host` + 0.0005 of rounding on the submitted score).
A *gap between two columns* carries only **±0.012**, because `macro_host` cancels —
`auc_a - auc_b = 12 x (score_b - score_a)`. So MCL sits 0.096 ± 0.012 above Lateral OA,
eight times the uncertainty, while MCL over Lateral Meniscus is +0.024 and the only gap
inside two sigma. MCL at 0.992 ± 0.030 may be saturated or may be 0.96.

**The "our out-of-fold" column is the weakest part of the table and is not a comparison.**
Ours are agreement with the label extractor on training studies; the host's are against a
radiologist on the hidden test, and weak labels attenuate an AUC. Measuring the two on the
same footing needs a different probe. **It has since been run** — see *Our experts priced
absolutely* below, where the solo probes put us at 0.908 on PF OA against his 0.932 and
0.884 on Lateral OA against his 0.896. **The gaps in the table above are overstated by a
factor of four or more** and none of the "+0.065" style figures in this section should be
used.

**Two decisions follow, and one explains a result from the same day.**

**MCL is the host's best column of the five and our expert is far below it.** That is why
the MCL bundle measured −0.0020 and why no box would have helped: there was nothing to win
on a column already solved. The morning's hypothesis — that the box had been *moved*
rather than only enlarged — is not needed to explain it. **Dropping our MCL column to
weight 0** is the action, and the chain already supports it: a column absent from the
weight mapping is left to the host untouched.

**Lateral OA is the host's weakest and the gap to us is the smallest of the four**
(+0.065). [[experts-what-pays]] records that the gain tracks the host's weakness rather
than the expert's quality, so that is where to invest — and it is what was training when
this was computed.

**Method note for the next batch.** Four of these five scores were nearly lost: the
competition API does not link a submission to the kernel that produced it and the
description field was left empty, so the mapping had to come from the user's own browser.
**Type the column name into the submission description.**

### The bundle on a seventh column, ACL, 2026-10-15/16

`acl_inset8` against the shipped `expert_acl_cnxnano`: the box inset 8% on each axis and
`lr = 1e-4`. The one column of the programme where the bundle *shrank* the region.

| | fold 0 | 1 | 2 | 3 | 4 | paired Δ | t | signs |
|---|---|---|---|---|---|---|---|---|
| old | 0.8780 | 0.8558 | 0.8828 | 0.8797 | 0.8940 | | | |
| new | 0.8734 | 0.8445 | 0.8958 | 0.8677 | 0.9007 | | | |
| **ACL** | −0.0046 | −0.0113 | **+0.0130** | −0.0120 | **+0.0067** | **−0.0016 ± 0.0050** | −0.33 | **2/5** |

Nothing, and the fold spread (0.025 peak to trough) is five times the effect. The
out-of-fold is flat to four decimals, 0.8693 → 0.8698.

| column | paired Δ | signs |
|---|---|---|
| **Lateral OA** | **+0.0110** | 5/5 |
| Medial Meniscus | +0.0053 | 4/5 |
| Lateral Meniscus | +0.0052 | 5/5 |
| Medial OA | +0.0043 | 4/5 |
| PF OA | +0.0042 | 3/5 |
| MCL | −0.0020 | 3/5 |
| ACL | −0.0016 | 2/5 |
| **pooled** | **+0.0038** | **26/35, p = 0.0030** |

**Projected macro +0.0022** over seven columns, **+0.0025** shipping only the five
positives — the seventh column dilutes the six-column figure without weakening the
pooled sign test, which is still 0.003.

### Our experts priced absolutely, and the 58 gold studies refuted

The symmetric of the void probes. There the host kept eleven columns and lost one; here
**our expert owns one column and the other eleven are constant**, so

```
auc_ours = 12 x macro - 5.5
```

`scripts/kaggle_probe.py` is a standalone notebook with **no host at all** — it writes its
own twelve-column floor, runs one expert, and finishes in minutes. Two details are load-
bearing. The floor column must be a **ramp**, not a constant: `_overwrite` remaps our
ordering onto the values the column already holds, and a constant column has one distinct
value, so the remap would flatten our predictions and the probe would report 0.5 for every
expert (verified synthetically: constant → 1 distinct value, spearman nan; ramp → 1000
values, spearman +1.0000). And the chain now **drops every expert the blend does not
name** before loading anything, which took a one-expert probe from 4 min and 8 ROI caches
to one.

**This is the only measurement of ours comparable to the host's**, because it is the same
labels on the same hidden test. It carries **±0.006** — the three-decimal rounding alone,
with no `macro_host` to inherit — against ±0.030 for the host's absolute figures.

| column | ours | host | gap |
|---|---|---|---|
| **ACL** | **0.980** | not probed alone | — |
| PF OA | 0.908 | 0.932 | −0.024 |
| Lateral OA | 0.884 | 0.896 | −0.012 |

**We are level with the host, not below it.** The gaps are −0.024 and −0.012 where the
out-of-fold comparison had claimed −0.089 and −0.065. That earlier claim was the error the
user named: our out-of-fold is agreement with the label extractor on training studies, the
host's is a radiologist on the hidden test, and the two cannot be subtracted. **What our
experts contribute is decorrelation at equal quality, not superiority** — which is also
why no per-column weight regime tuned against the out-of-fold comparator ever paid.

**ACL at 0.980 is our best column and may be the first we hold over the host.** His ACL
was never probed alone; ACL + Medial OA sum to 1.924 from the block arithmetic, so his ACL
exceeds ours only if his Medial OA falls to 0.944 — below any column he is known to hold.
At 0.962 each we would be ahead by +0.018. A `void-acl` submission would settle it, and
our weights currently leave that column to him.

**The 58 gold studies are not a representative predictor, and that is now measured.** Two
paired probes, the only variable the ACL expert:

| predictor | said | sign | error |
|---|---|---|---|
| **gold, 58 studies** | **+0.0110** | **wrong** | 0.023 |
| out-of-fold, absolute | +0.0005 | wrong | 0.013 |
| **out-of-fold, paired per fold** | **−0.0016** | **right** | 0.010 |
| **leaderboard (`only-acl-old` 0.540, `only-acl-new` 0.539)** | **−0.012 ± 0.008** | | |

The gold did not merely miss the magnitude, it **inverted the sign** — 0.9828 → 0.9939
read as the clearest gain of the session. Its own uncertainty is ±0.05 on 58 studies,
wider than any effect we look for, so the 0.023 miss is inside its error bars and the
reading was never admissible. This is its third failure, after row #25's +0.0396 that
scored 0.000 on the board.

**The paired per-fold comparison was the only one of the three with the right sign.** It
understates the magnitude by 7x, which is expected — it removes the fold-to-fold variance
rather than estimating an absolute AUC. **It is the arbiter from here, and the gold is
retired from decisions of this kind.** The old expert ships, and `bundle5` already carries
it.

### Measuring the host per column without a contaminated comparator

Every per-column decision this project has made was taken against **our** wide model at
0.87 while the host reads ~0.95, and that is what the four failed weight regimes have in
common. The fix is not to run the host on training studies: the public models were
**trained on those studies**, so that measures their memorisation.

The clean probe costs submissions and no compute. Replace **one** column of the
unmodified host with a constant and the macro falls by exactly `(auc - 0.5) / 12`, on the
real hidden test. Twelve submissions would price every column; the four worth pricing
first are the **orphans** — Effusion, Contusion, Synovitis, Fracture — where a new expert
would *add* a column instead of competing for one the host already reads well. The note
in memory that they sit in the host's stronger half was measured against d4 at 0.940, not
against today's 0.956.

### Fifteen levers that are measured zero

Kept because the cost of re-trying one of these is a day, and the cost of writing it down
was a line.

| lever | verdict |
|---|---|
| label tables, the pilkwang + v4_blend fusion | **0.000** (#25). And `steven_v4`, which we already use, is the best of the six untainted public sources on macro |
| `riad` / JEV as a third table | **gold-contaminated** — its per-finding weights were fitted on the 58 gold studies we hold out |
| per-column expert weights — `shrunk`, `corrected`, `divided` | **−0.001, −0.001, −0.002** |
| a flat 0.6 | **−0.001** (#34) |
| `nocoat`-d4 and its 28 models | **0.000**, measured twice — the whole public stack buys nothing over `nartaa` + our experts |
| d4's four CoAtNet readers | **−0.004** (#26 against #28) |
| adding a plane to an existing expert — Effusion, medial meniscus coronal | **0.000**, three times. The one plane that ever paid was Baker's axial, and that was a *new* expert |
| Contusion on the OA boxes | **−0.00084** of macro in blend |
| `weight_decay = 0.05` against our 1e-4 | **+0.0002** on both encoders — the best-epoch rule already absorbs the post-peak drift it would have fixed |
| ~~averaging resnet18 with convnext_nano~~ | ~~**+0.00018**~~ — **refuted 2026-10-18**: same-box cross-encoder pairs take +0.0118 (PF OA), +0.0069 (Baker's), +0.0065 (MCL), paired per fold |
| more epochs | no run of fifty keeps its last epoch; the median kept epoch is 10 of 30 for resnet18 and 15 for convnext |
| `lr = 1e-4` across the board | +0.0045 on ACL and +0.0036 on MCL but **−0.0051 on Lateral Meniscus**, our largest contributor. Mean +0.0010 over three columns against +0.0055 predicted from one. Not generalised |

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
