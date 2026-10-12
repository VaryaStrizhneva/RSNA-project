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
| averaging resnet18 with convnext_nano | **+0.00018** |
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
