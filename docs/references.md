# Public notebooks and external sources

> Copies live under [`notebooks/external/`](../notebooks/external/), one directory
> per kernel, **never edited**. Pull them with:
>
> ```bash
> kaggle kernels pull <owner>/<slug> -p notebooks/external/<slug> -m
> ```
>
> `-m` also fetches `kernel-metadata.json`, which records the attached datasets,
> models and hardware.
>
> ⚠️ **Licenses are not in `kernel-metadata.json`** — they are shown on the Kaggle
> page. Fill the License field in by hand before reusing any code: winners must
> open-source their solution, and attribution has to be right by then.

---

## pilkwang/rsna-knee-baseline-v1 — "RSNA Knee baseline v1"

- <https://www.kaggle.com/code/pilkwang/rsna-knee-baseline-v1>
- Public LB: **0.809** | License: *TO CHECK* | Pulled: 2026-09-07
- 35 cells (21 code), T4 GPU, internet off
- Encoder: DINOv2-small. Attached: `pilkwang/rsna-knee-llm-labels`,
  `pilkwang/rsna-knee-weights`

**Read this one first.** It is the pedagogical reference of the three: every design
choice is argued in prose before the code, and the reasoning is unusually careful.

What's useful:

- §1 — why macro-AUC means *only order matters*, so calibration and thresholds are
  worthless. Submissions are written as percentile ranks.
- §2 — a rule-based, nine-language report extractor with explicit negation and
  polarity handling, as a fallback when the LLM label table is not mounted.
- §3–§7 — the pipeline traps, now written up separately in
  [`pipeline_pitfalls.md`](pipeline_pitfalls.md): slice ordering, laterality,
  mm-per-pixel resampling, decode-once caching, offline training, and the
  identical-report validation leak.
- §6 — `SlotHead`: per-diagnosis attention over slot embeddings with a presence
  mask, rather than mean-pooling six slots.

Used in: *(nothing yet)*

---

## prvsiyan/rsna-knee-read-the-report-then-the-knee

- <https://www.kaggle.com/code/prvsiyan/rsna-knee-read-the-report-then-the-knee>
- Public LB: **0.906** claimed (Version 52) | License: *TO CHECK* | Pulled: 2026-09-07
- 74 cells (47 code), ~896k chars of source, TPU v5e-8, internet off
- Encoders: BiomedCLIP, DINOv2-small/base, EfficientNet-B3, RadImageNet ResNet-50

**Its §4–§7 are the same cells as pilkwang's baseline**, byte-for-byte in several
places. What is genuinely its own is §2 and the accretion above §8.

What's useful:

- §2 — a **much larger report extractor** (~19k chars of pathology lexicon) across
  nine languages, with compartment scoping and severity grading.
- §2.2 — **how to validate an extractor**: the dangerous failure is silent, so it
  measures a *silence rate* (how often a rule never fires) alongside AUC on the 58
  labelled studies, and warns that three of five ablation deltas are smaller than
  the sampling error of a 58-study AUC.
- §2.3 — silence broken down **by language**, which turns the gauge into a concrete
  to-do list of missing vocabulary.

What we're not taking: everything from §8 on is version-by-version accretion —
specialist heads, base64-embedded experiment blobs, guarded retry branches for
incompatible GPUs. Impressive, unreadable, and irrelevant to a first submission.

Used in: *(nothing yet)*

---

## mattiaangeli/bend-the-knee-to-the-dinosaurs

- <https://www.kaggle.com/code/mattiaangeli/bend-the-knee-to-the-dinosaurs>
- Public LB: best of the three, exact score *TO CHECK* | License: *TO CHECK* | Pulled: 2026-09-07
- 27 cells (26 code), T4 GPU, internet off
- 12 attached datasets, 2 parent kernels, DINOv2-small + CoAtNet

The same pipeline again, with **every explanatory comment stripped** (`pick_slots`
is 739 chars here against 2,386 in pilkwang's), plus heavy ensemble stacking.

Highest score, lowest teaching value. Useful only as evidence of what the top of
the leaderboard is made of: many checkpoints, blended.

Used in: *(nothing yet)*

---

## Public label datasets

All three notebooks mount a **pre-computed table of LLM-derived labels** rather
than deriving labels at run time:

| Dataset | Used by |
|---|---|
| `pilkwang/rsna-knee-llm-labels` | all three |
| `stevenleehans/rsna-knee-llm-report-labels` | prvsiyan |
| `lixin73/rsna-knee-llm-report-labels-sol56` | prvsiyan |

At least nine competing tables exist publicly, including multi-source merges
(`yunusgmsoy/rsna-knee-llm-labels-4-source-merged`) and one shipping stratified
folds (`barun2104/rsna-knee-stratified-folds-and-llm-soft-labels`, highest
usability rating).

### `pilkwang/rsna-knee-llm-labels`, inspected

Ships **both** the extractor and its output:

- `api_labeler.py` (13 KB) — the full script, prompt included. `claude-opus-5` via
  the Anthropic API, structured output (schema-enforced, not regex-parsed), bulk
  submission with prefix caching. Runs offline, so the notebook internet ban does
  not apply — but whether the rules permit sending report text to a hosted API is
  still an open question for the *Rules* tab.
- `report_labels_v2.csv` (1 MB) — 4,406 studies x 12 findings, three columns each:
  a continuous score, a confidence, and a `YES`/`NO`/`UNK` verdict. `UNK` means the
  report does not say, which is not the same as negative.

**Measured against our 58 expert-labelled studies (57 matched): macro-AUC 0.870.**

| Target | AUC | | Target | AUC |
|---|---|---|---|---|
| ACL | 0.997 | | Fracture | 0.871 |
| MCL | 0.976 | | Lateral Meniscus | 0.841 |
| Medial Meniscus | 0.943 | | Effusion | 0.830 |
| Baker's | 0.932 | | Lateral OA | 0.789 |
| Medial OA | 0.908 | | Contusion | 0.768 |
| PF OA | 0.891 | | **Synovitis** | **0.694** |

`Synovitis` is the hole: 3,712 of 4,406 studies come back `UNK`. Reports simply do
not mention it — a source problem, not an extraction problem, and the only recourse
is the image. Caveat: 57 studies is a thin measurement (9 positives for MCL), so
read these as orders of magnitude.

## Pre-trained encoders seen in use

`metaresearch/dinov2` (small, base) · `marwanmath/resnet-50-radimagenet-marwan` ·
`mrquadian/biomedclip` · `timm/tf-efficientnet-b3` ·
`royceplayz/rsna-medicalnet-resnet50-weights` · `raidium/curia-2`

## Official tooling

- [RSNA Knee Abnormalities — Efficiency LB](https://www.kaggle.com/code/ryanholbrook/rsna-knee-abnormalities-efficiency-lb)
  — the official Efficiency Leaderboard notebook, not a source to copy.
