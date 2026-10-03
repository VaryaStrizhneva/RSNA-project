# `pjmathematician/rsna-knee-d4-blend`

Pulled 2026-10-03 with `kaggle kernels pull pjmathematician/rsna-knee-d4-blend -m`.
Read-only, like everything under `external/`. Last run by its author 2026-09-28,
148 votes. The banner claims **public LB 0.943** and sub-30-minute inference.

## What it is

Five stages, every one of them combined in rank space:

| Stage | What runs |
|---|---|
| 1 | DINO ensemble — 20 members over a shared frozen prefix |
| 2 | A5 attention pooling — 5 folds, variable-length slice pooling |
| 3 | RadImageNet heads — one encoder, three head layouts (alpha 0.55, second pass 0.20) |
| 4-5 | Raptor (4 views) + 4 CoAtNet readers, averaged 25 % each, blended 40 % into Raptor |

Then a sixth section the author calls **"OUR LEG"**, which stashes the public result,
runs their own models from two private datasets, and blends them in at **45 %**.

## Can we run it as-is

The public path: yes. The whole 0.943: no.

**What blocks the 0.943.** `dataset_sources` begins with three blank entries — private
datasets the API hides from anyone but their owner. `OUR LEG — 2/3` looks for
`rsna-knee-eff6-assets` and `rsna-knee-ens14-assets`, finds nothing, and takes its
SOFT FAIL branch. What is left is what the author's own comment calls
**"the public 0.941"**. The missing 0.002 is their private leg, not a trick we can copy.

**What it demands to start at all.** `_d4_check_runtime` raises, not warns, on each of:

- `torch.cuda.device_count() != 2` — **exactly two T4s**, so the kernel must be set to
  `T4 x2`; our own submit kernel is a single T4 and would die on the first stage;
- `cv2.__version__ != '4.12.0'` and `timm.__version__ != '1.0.22'`, pinned through
  wheels inside the mounted datasets;
- a dozen SHA-256 identities over manifests, adapters and checkpoints;
- `rsna_asset_preflight` raises on any missing Raptor or A5 weight.

All of those assets are in **public** datasets, so they mount for us. The strictness is
a feature: nothing silently degrades into a different model.

## What is worth taking, and what is not

**The shared frozen prefix** (`_shared_dino_prefix` / `_shared_dino_tail`,
`SHARED_DINO_PREFIX_LAYERS = 6`). DINOv2-small has twelve layers; their members freeze
the first six, so those six are identical across members and are computed **once** per
image rather than once per member. Our wide model satisfies their own eligibility test
exactly — `unfreeze_last=6`, `variant=small`, `pool=cls_mean`.

It would nevertheless buy us close to nothing. Measured over 4407 studies, our DINO
forward is **203 s of a 2241 s stage**; the rest is DICOM decoding. Five members would
go from 60 layer-passes to 36, saving about 85 s in two and a half hours. Their gain is
real because they run twenty members, not five.

The same goes for the two GPUs: our bottleneck is the CPU, and `ORDER_THREADS = 32` /
`PIXEL_THREADS = 12` are already what we use, as is `specific_tags` on the header reads.

**What is worth taking is the argument in their blending cell**, not their code: a blend
pays in proportion to how much the two sides disagree, and they measure rank
correlations of **0.75-0.89** between the legs they combine. Our experts sit at
**0.57-0.75** against the wide model — more decorrelated than anything in their
ensemble. Their own criterion says our leg should be worth more in a blend than the
arms they are paying for.

## Licence

Kaggle notebooks are shared under the licence their author selects, Apache 2.0 by
default. Reusing a publicly shared notebook is permitted by the competition rules;
attribution belongs in `docs/experiments.md` on any submission derived from it.
