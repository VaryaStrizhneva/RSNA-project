# `matterhorn3838/rsna-knee-d4` — the "Apex V3" fusion, public LB 0.950

Pulled 2026-10-08. Read-only, like everything under `external/`. `sha256` of the
concatenated code: `b8497f5325e6b3725fbad73a88eebbedfe649a18b08a149ff38c3caf4a44efdd`.

**Three slugs, one notebook.** `evgendvorkin/rsna-versia-5` has the same code with
markdown wrapped around it — its fusion cell differs from this one by **zero lines**.
`haideptry/rsna-knee-v2-vs-apex-comparative-study` is the same two legs with a longer
fusion cell (11,580 chars against 6,080) that adds a gold-58 grid search and calls itself
V5; its own table records **V3 and V4 as the 0.950 that was verified**, and V5's target as
an aspiration. So this is the version that actually holds the score.

And `matterhorn3838/rsna-knee-v2-velciraptor-dinosaur-speed` is **not** a fourth variant:
it is byte-identical to [`../rsna-knee-0949-anatomical-mirror/`](../rsna-knee-0949-anatomical-mirror/),
`sha256 acd9241f…`. Same code, same single weights mount, so it cannot score above 0.949.

## What it is

Two legs, blended in rank space, per pathology.

| Leg | What | Mount |
|---|---|---|
| 1 | `nartaa`'s CoAtNet, **unchanged** but for two lines that save a backup copy | `nartaa/rsna-knee-publication-swa-weights-20261007` |
| 2 | a **ConvNeXt 2.5D MIL reader**, 3 folds of `convnext_tiny`, `MM_PER_PX = 0.4`, `SIZE = 384` (153.6 mm field), 64 slices, 6 slots (3 planes × fat-sat or not) with a learned slot embedding | `goodpjw2008/rsna-knee-2-5d-convnext-reader` |

The weights are hand-set per column, and **that is the whole of the +0.001**:

| ConvNeXt's share | columns |
|---|---|
| 0.35 | Baker's |
| 0.30 | Contusion |
| 0.25 | Medial OA, ACL |
| 0.20 | Medial Meniscus, MCL |
| 0.08 | Lateral Meniscus, Fracture, Lateral OA, PF OA |
| 0.04 / 0.03 | Effusion / Synovitis |

Mean 0.162. The comments justify each one with a gold-58 reading ("ConvNeXt is dominant,
0.976 vs 0.940"), which is 58 studies with a handful of positives per column — and
`haideptry`'s own V5 notebook condemns exactly that: *"adjusting weights intuitively
without ground-truth empirical feedback"*. So the shape of the allocation is probably
right and the exact values are not to be trusted.

## Why it matters to us: it is complementary, not redundant

Crossing their weights against our experts' measured contributions, the ConvNeXt is
**strong precisely where we are weak**:

| Column | ConvNeXt's weight | our expert |
|---|---|---|
| Baker's | **0.35** | among our weakest contributors |
| Medial OA | **0.25** | weak |
| MCL | **0.20** | weak |
| Contusion | **0.30** | **none** — the column we failed to build |
| Lateral Meniscus | 0.08 | **our strongest**, +0.081 over the host solo |
| PF OA | 0.08 | strong |

So the two legs divide the twelve columns rather than fight over them, and Contusion —
where our own expert measured **−0.00084** of macro and was abandoned — is covered for
free.

## Running it

Both mounts are public. The reader ships its own wheels (`pylibjpeg`, and a `timm`
wheel if present) and installs them with `--no-index` into `/kaggle/working/_own_env`,
so internet stays off. It also carries an `oof.py`, which is worth a look if we ever want
its out-of-fold predictions rather than its leaderboard reputation.

`scripts/kaggle_stack.py` can take this as a leg; `scripts/kaggle_graft.py` can graft our
experts onto it unchanged.
