# Pipeline v3 — one wide model, framed on anatomy

**Status: design.** Nothing here is implemented. This document is the decision record for
the model we want to replace the current wide one with, and it supersedes the
*architecture* section of [`pipeline_v2.md`](pipeline_v2.md) — not its ROI work, which
shipped and is measured.

Every number is marked **[m]** when it was measured, with the sample size. Anything
unmarked is a judgement. **There are no estimates left in this document** — the training
cost was the last one, and §7 replaced it with a measurement.

---

## 0. What changed since v2

`pipeline_v2.md` was written before two things happened. First, the seven per-pathology
experts were built and scored. Second, we read all five legs of the public 0.943
pipeline, line by line, and found that **not one of them places its window on the knee**.

Three consequences for the design:

- v2 proposed fusing **one vector per branch**. Raptor's head shows a better shape:
  attention over **windows**, per target. We adopt it.
- v2 did not know the trunk was constrained. It is, and the constraint settles the
  choice — see §6.
- v2 justified itself on comorbidity reasoning being most of the wide model's score.
  That framing was loose: the 0.774 figure is measured with the *true* other eleven
  labels, which inference does not have. The real argument is narrower and still holds —
  a twelve-target head has label structure to learn, and single-target experts have
  none of it.

---

## 1. What the measurements say now

**The experts work, and they work where v2 predicted.** Out of fold, against the
reference label table **[m, 4348 studies]**: Lateral Meniscus 0.8476, Medial Meniscus
0.8791, PF OA 0.8305, MCL 0.8042, Lateral OA 0.8121, Medial OA 0.8611, ACL 0.8626. These
are exactly the focal and structural targets v2 singled out as the ones whose input was
wrong.

**The gain survives onto a strong host, and shrinks.** Grafted as a rank blend at 0.5
**[m, public leaderboard]**:

| host | host alone | with the seven experts | gain |
|---|---|---|---|
| riad HYBRID | 0.882 | 0.913 | **+0.031** |
| our #5 wide model | 0.899 | 0.920 | **+0.021** |
| `nocoat` (public minus CoAt) | 0.940 | **0.944** | **+0.004** |

Do not fit a line to those three; the first two are outright grafts and the third is a
blend. What they establish is the direction and the fact that the gain is still positive
against a 0.940 host.

**Why it survives: we disagree with them more than they disagree with each other.** Rank
correlation of our columns against the host is **0.57–0.75** **[m]**, where their own legs
correlate **0.75–0.89** with each other **[m]**. Their own code blends in proportion to
disagreement.

---

## 2. Architecture

```
7 ROI branches, native shapes (224x126, 112x280, 252x182, ...)  ┐
2 wide slots, 336x336                                           ├─> ONE shared trunk
                                                                ┘   (resnet18)
                                        │
                              features, one vector per window
                                        │
                     per-target attention over all ~168 windows
                     + a positional feature per window
                                        │
                                   12 logits
```

Three properties, in order of how much they matter.

**It is a strict generalisation.** If a branch adds nothing the attention weights it to
zero and we recover a wide model. This was v2's argument and it is still the reason to
build this shape rather than a new model from scratch: it cannot be worse than the thing
it replaces, only equal or better.

**One trunk, not one per branch.** Every window of every branch passes through the same
weights. Measured: a model with one group and a model with eight have **15.13 M
parameters each**; one trunk per group would be **119.62 M** **[m]**, and each copy would
see an eighth of the supervision. The `for imgs, mask in groups` loop in
`ExpertNet.forward` exists **only** because a 224×126 tensor and a 112×280 tensor cannot
be concatenated into one batch — same weights on every call. The bag costs **compute**,
not parameters.

**Attention over windows, per target** — Raptor's head, six lines:

```python
a = torch.softmax(self.att(h), dim=1)          # (B, windows, 12)
pooled = torch.einsum('bkn,bkf->bnf', a, h)    # (B, 12, dim)
logits = (pooled * self.clsW).sum(-1) + self.clsb
```

Each pathology gets its own distribution over the windows. It is the learned analogue of
what the experts do by construction, and because it is a softmax-weighted **sum** it is
indifferent to the bag's size — which is what §7 depends on.

**Never stack different planes as channels.** Pixel (i,j) of a sagittal slice and of a
coronal slice are different anatomy; a convolution assumes its channels are co-located.
This was v2's rule and the public pipeline breaks it — see §5.

---

## 3. Resolution is the headline

A meniscal tear is about 1.5 mm. How many pixels that is, per leg **[m, arithmetic on each
leg's crop and input size; the 160 mm field of view is the measured median over 400
series]**:

| leg | mm/px | the tear |
|---|---|---|
| RadImageNet `rad_e10` — **no crop at all** | 0.714 | **2.1 px** |
| RadImageNet `e13` / `e11` | 0.580 | 2.6 px |
| Raptor `maxspan-v5` | 0.417 | 3.6 px |
| DINOv2, A5 | 0.387 | 3.9 px |
| Raptor `native384` | 0.365 | 4.1 px |
| CoAt resgated | 0.339 | 4.4 px |
| **our ROI crops** | **0.214–0.286** | **5.2–7.0 px** |

Their best leg sees the lesion across 4.4 pixels, their worst across 2.1. We see it
across 7. For reference a DINOv2 patch is 14 px, so on their input the finding is a
fraction of the encoder's smallest unit of position.

---

## 4. What we take from the public pipeline

Four of their decisions are better than ours and should be copied. This is not a
rejection of their work.

**Per-target MIL attention over windows** (Raptor). Already argued in §2.

**Geometric ordering.** Slices sorted by `IPP · normal` — the true through-plane
coordinate — rather than by `InstanceNumber`, which is bookkeeping and can run backwards.
Our landmark path does this; the rest of our pipeline should.

**The span at 96% of the stack.** They read `[0.02n, 0.98n]`. We read thin slabs. Measured
**[m, 600 studies]**, our band is not the problem — 10.7 of 12 slices land within 30 mm of
the joint and never fewer than 7 — but that is an argument about *our* slabs, not about
reading more.

**Capacity-aware budget redistribution** (`_dense_allocate`). A fixed slice budget spread
across slots in fixed proportions, **capped by what each series actually holds**, with the
remainder given to slots that still have room. This is the direct fix for our worst
coverage defect: 197 studies reach the MCL expert with every slot empty **[m]**, where
this mechanism would have moved the budget to a slot that had pixels.

**Train on a small bag, infer on a large one** — as an augmentation, not a necessity.
Their CoAt reader declares `train_windows_max: 32` against `eval_slot_budgets` summing to
76 **[m, from the shipped runtime]**. The MIL head is permutation-invariant and
size-agnostic, so this is sound. We do not *need* it — §7 measures the full bag training
in 3.4 h per fold — but sampling a different subset each epoch is instance-level dropout,
and it would teach the attention not to lean on any one window being present. That is
worth having on its own terms, given the coverage holes in §5.

---

## 5. What we fix

**Window placement.** All five of their legs centre the crop on the image:

```python
cy,cx = h//2, w//2                        # DINOv2, RadImageNet
cy,cx = arr.shape[0]//2, arr.shape[1]//2  # A5
y0 = (h - cpx) // 2                       # Raptor
```

Measured over all 4407 studies **[m]**, the knee centre sits a median **17 mm** from the
image centre, 29 mm at the 90th percentile, 68 mm at worst. Against a 65 mm half-crop and
a half-knee of roughly 48 mm, **half the corpus has joint anatomy outside the window**.
Centring is not a resolution tweak; it stops a loss that is already happening. Pictures
and distribution: [`atlas/crop.html`](atlas/crop.html).

The landmark costs nothing new: the midpoint of `lat_centre` and `med_centre`, already
predicted for all 4407 studies.

**Window shape.** Their windows are square because their trunk requires it (§6). Ours are
the shape of the anatomy — the MCL box is 32×80 mm because the collateral ligament is a
vertical band, and squaring it would throw that away.

**Series selection.** They pick the first row of a CSV, or the longest series. On the
worked example in this corpus, a perfectly good sagittal series is discarded because it
appears *after* another one in a CSV. And their metadata cannot express what we need:
`Fluid_Sensitive` and `Fat_Suppression` are **identical on all 24371 rows** **[m]** — one
bit under two names. Our header-derived `fatsat` agrees with that bit 98.8% of the time,
but our `fluid` (PD or T2) agrees only **80.5%** **[m]**. So **PD versus T2 is invisible to
anyone using the organisers' CSV**, and it matters: the MCL expert scores **+0.031 better**
on studies where its PD fat-sat series is missing and it falls back to T2 fat-sat
**[m, 500 studies]**.

**Cross-series contamination.** Raptor concatenates its five slots into one volume and
slides RGB triplets over the whole thing, so windows straddling a slot boundary carry
channels from two different series — a sagittal slice in red beside a coronal one in
green. **8 of 94 windows [m]**. Cutting windows per slot instead is **strictly free**: the
86 clean windows are the same 86, and 8 trunk passes are saved. Clamping the outer
channels at the acquisition edge — which `ExpertNet.windows()` already does — gives 96
clean windows for the same slice budget.

**Position and slot identity.** `RaptorClassifier` has no slot embedding and no positional
encoding, and its pooling is a sum over the bag, so the window order is discarded too. It
sees an unordered bag and does not know, for any window, which plane it came from or where
in the knee it sits. We predict landmarks in millimetres, so every window has a plane and
a signed offset from the joint centre. A `Linear` on that, added to the attention's input,
is a few hundred parameters.

**Studies with no pixels.** Already fixed, in `rsna/infer/chain.py`: 299 of 4407 studies
**[m]** reach an expert as an all-zero tensor, the model answers with a constant (the MCL
expert returns 0.2499–0.2516, sd 0.0006 **[m]**), and writing that in trades the host's
ordering for a tie. `has_pixels()` now withholds those rows, judging emptiness across all
of an expert's regions rather than each.

---

## 6. The trunk — settled by a constraint

CoAtNet is the appealing choice: convolution at the bottom where resolution is high, which
is what a 7-pixel lesion needs, and attention on top. But **every CoAtNet variant accepts
only its training resolution, square** — the relative-position table is fixed
**[m, timm 1.0.x, forward pass attempted at each size]**:

| trunk | params | dim | 224×126 | 112×280 | 336² | 384² |
|---|---|---|---|---|---|---|
| **`resnet18`** | **11.2M** | **512** | ✅ | ✅ | ✅ | ✅ |
| `resnet50` | 23.5M | 2048 | ✅ | ✅ | ✅ | ✅ |
| `convnext_nano` | 15.0M | 640 | ✅ | ✅ | ✅ | ✅ |
| `convnext_tiny` | 27.8M | 768 | ✅ | ✅ | ✅ | ✅ |
| `coatnet_pico_rw_224` | 10.3M | 512 | ❌ | ❌ | ❌ | ❌ |
| `coatnet_0_rw_224` | 26.7M | 768 | ❌ | ❌ | ❌ | ❌ |
| `coatnet_rmlp_2_rw_384` | 72.9M | 1024 | ❌ | ❌ | ❌ | 384² only |

No CoAtNet accepts a single one of our ROI shapes.

The resolution is clean rather than a compromise: **CoAtNet's appeal was its internal
attention, and the attention we need is in the head, not the trunk.** The MIL head gives
it to us, per pathology, over windows. The trunk then has one job — good local features at
the right resolution — which is what a pure convolutional net is for, and what v2 already
argued against the ViT.

**Choice: `resnet18` — the one the experts already use.** There is no measured reason to
upgrade, and the throughput measurement argues against it **[m, L40, full bag, bf16]**:

| trunk | peak | studies/s | h/fold (30 ep) |
|---|---|---|---|
| **`resnet18`** | **4.4 G** | **19.5** | **1.9** |
| `convnext_nano` | 12.9 G | 10.5 | 3.4 |

`resnet18` is 1.9× faster on a third of the memory, and it is the known quantity: the
seven measured experts run on it, so a difference in the result is attributable to the
preprocessing rather than to the encoder. Changing the trunk at the same time as the input
would confound exactly what this model exists to measure.

**The trunk is a separate experiment, afterwards.** And when it is run, the candidate is
`resnet50` + RadImageNet rather than `convnext_nano`: pre-training on 1.35 M radiological
images is plausibly worth more than an architecture refinement on ImageNet, and
`rsna/expert/pretrained.py` already loads it with zero missing and zero unexpected keys
**[m]**.

Raptor needs CoAtNet because its windows are all 336 or 384 squares, which follows from
cropping a square 140 mm box at the image centre. We do not inherit that constraint
because our boxes have the shape of the anatomy.

---

## 7. Budget

**Inference — measured, and generous.** From the bench **[m, 120 studies, 2×T4]**, the
Raptor stage runs 503.8 s for four arms, which is ≈**0.25 min per window-336² for the
whole 1322-study test set**. One arm, 94 windows, is 23 min.

A ROI crop is 0.25–0.41× the pixels of a 336² window, so the bag is cheap:

| | windows | cost in window-336² | vs one Raptor arm |
|---|---|---|---|
| Raptor today | 94 | 94 | 100% |
| 7 ROIs, one series each | 59 | 20 | 21% |
| 7 ROIs, every series | 136 | 46 | 49% |
| **2 wide slots ×16 + 7 ROIs, allocated** | **168** | **78** | **83%** |
| **the same bag, *expected*** | **99** | **54** | **58%** |
| 3 wide slots ×16 + 7 ROIs, allocated | 184 | 94 | 100% |

**The allocated bag is not the cost.** It counts every (ROI × series) pair, and measured
fill is 83% for a first series and 13–37% for the others **[m]**, so about half of those
pairs do not exist for a given study. `ExpertNet.encode` already runs the trunk only on
filled windows — `feat[keep] = self.trunk(flat[keep])` — so the real bag averages
**99 windows, 54 window-336²**. Every figure below that uses 168 is therefore a ceiling,
and the training measurement in particular is a worst case by about a third.

For the price of one Raptor arm we buy **184 windows instead of 94**, 136 of them framed
on anatomy at twice the resolution.

Reserving 75 min for the landmark models and cutting the seven ROI caches:

| scenario | left for the model | window-336² per study, all folds | per fold (×5) | headroom |
|---|---|---|---|---|
| our model alone | 445 min | ~1800 | 360 | **3.8×** |
| grafted onto `nocoat` | 237 min | ~960 | 192 | **2×** |
| grafted onto the full 0.943 | negative | — | — | does not fit |

**Training — measured, and the full bag fits.** The question was whether the bag has to be
subsampled during training. It does not. Measured on the L40 (44.3 GiB) with the real
group shapes and window counts — 168 windows per study, forward and backward, random
tensors **[m]**:

| trunk | batch | bf16 | peak | studies/s | min/epoch | h/fold (30 ep) | 5 folds |
|---|---|---|---|---|---|---|---|
| `convnext_nano` | 2 | no | 22.3 G | 4.7 | 15.5 | 7.8 | 39 h |
| `convnext_nano` | 2 | yes | 12.9 G | 10.5 | 6.9 | 3.4 | 17 h |
| `convnext_nano` | 6 | yes + ckpt | 11.0 G | 7.4 | 9.8 | 4.9 | 25 h |
| **`resnet18`** | **2** | **yes** | **4.4 G** | **19.5** | **3.7** | **1.9** | **9.5 h** |

So the whole bag, 30 epochs, five folds, is **9.5 h on `resnet18`** — about what the
seven experts' last batch took, and less once the expected-bag correction above is
applied (these rows assume all 168 windows filled; the average is 99). Three notes:

- **`bf16` autocast is what unlocks it**, not gradient checkpointing: 2.2× the throughput
  and half the memory. Checkpointing allows a larger batch but is slower, and at batch 2
  there is nothing to solve.
- **This is a worst case.** It assumes every slot filled, where measured fill is 83% for
  the first series and 13–37% for the others **[m]**, so the average bag is smaller.
- **Data loading is not in this measurement.** The tensors were random. The ROI caches are
  memory-mapped `uint8`, about 5.9 MB per study for 168 windows, so 10.5 studies/s is
  ≈62 MB/s of reads — not a plausible bottleneck, but unmeasured.

Subsampling the bag during training therefore stays an *option* — it is still a useful
instance-level augmentation, and §4 notes that their CoAt reader does it — but it is no
longer a constraint we have to design around.

## 8. Decisions taken

**The bag.** Nine boxes: the seven pathology boxes already cached, plus two whole-knee
branches. **99 windows, ≈40 window-336²** — 43% of one Raptor arm.

**One channel per box, filled by a priority list.** Not several series as separate
channels. The saving in windows is small (67 → 58 expected) and that is not the argument.
The argument is that it **reconciles two measurements that looked contradictory**: the MCL
ablation found that removing the coronal PD-non-fat-sat series gained **+0.0118**, yet that
same series is what would cover 190 of the 197 studies with nothing. A *channel* is fed to
every study that has it, alongside their primary — and that hurt. A *fallback* is used only
where nothing better exists. A priority list takes the fallback and drops the channel.

Coverage, cumulative over a three-deep list **[m, 4407 studies]**:

| plane | first | +second | +third | left with nothing |
|---|---|---|---|---|
| Sagittal | 82.6% | — | **99.8%** | 0 |
| Coronal | 84.0% | 95.5% | **99.8%** | 2 |
| Axial | 72.2% | **98.6%** | 100.0% | 1 |

A deeper list costs **no extra windows** — it is one channel either way — so the lists are
as deep as there are plausible sequences. The residual 0–2 studies per plane are what
`has_pixels()` abstains on, instead of 197.

**No cache rebuild for the seven.** The existing caches already hold the series a good
list needs: the three coronal boxes are complete, and the sagittal ones hold PD-FS +
PD-noFS, which *is* the 99.8% list. Only `pf_oa` lacks a third (98.6% without it). One
change: use `mcl` (three series) rather than `mcl_fsonly` (two).

**The priority order is measured, not declared.** The trained experts carry a per-target
attention over series, so their weights say which sequence each model actually leans on.
Read out over studies carrying at least two series **[m]**:

| box | PD-FS | T2-FS | PD-noFS | n |
|---|---|---|---|---|
| `lateral_oa` | **0.93** | 0.07 | 0.00 | 632 |
| `lateral_meniscus_d32` | **0.90** | — | 0.10 | 889 |
| `mcl` | **0.90** | 0.10 | — | 239 |
| `acl` | **0.80** | — | 0.20 | 886 |
| `medial_oa` | **0.69** | 0.16 | 0.15 | 632 |
| `pf_oa` | 0.58 | 0.42 | — | *79* |
| **`medial_meniscus`** | 0.44 | — | **0.56** | 889 |

PD-fat-sat first is validated for five of seven. **`medial_meniscus` wants PD-noFS first**,
which is what `Attention`'s own docstring predicts — a meniscal tear is signal inside the
fibrocartilage, which non-fat-suppressed proton density shows. Curiously `lateral_meniscus`
contradicts it. And `lateral_oa` puts **0.00** on PD-noFS, independently agreeing with the
MCL ablation that this is the least useful coronal series.

Two caveats. `pf_oa` is read on 79 studies and is not usable. And for **MCL the two
measurements disagree**: the attention says 0.90 on PD-FS, while the fallback AUC says
0.8360 against 0.8049 when PD-FS is *absent* and T2-FS is read instead. The attention may
be self-fulfilling, since PD-FS is present 84% of the time and so dominates training.
Neither is decisive; the order is one config field, so both get run.

The resulting lists:

```
lateral_meniscus_d32   PD-FS  → PD-noFS
medial_meniscus        PD-noFS → PD-FS            ← reversed, on the readout above
acl                    PD-FS  → PD-noFS
pf_oa                  PD-FS  → T2-FS             (readout not usable, n=79)
mcl                    PD-FS  → T2-FS → PD-noFS   (and the reverse, as a variant)
lateral_oa, medial_oa  PD-FS  → T2-FS → PD-noFS
knee_sagittal/coronal  PD-FS  → T2-FS → PD-noFS
```

**The whole-knee boxes.** Added to the registry as `knee_sagittal` and `knee_coronal`:
100×100 mm on 280×280 px = **0.357 mm/px**, depth ±45 mm (sagittal) and ±40 mm (coronal),
20 slots at 4.5 mm, three series each. Their geometry is derived and shown in
[`atlas/wide.html`](atlas/wide.html). Verified on real studies before the 19 GB build:
20 of 20 slots filled spanning 86–88 mm sagittal, 72–76 mm coronal, ~100% non-zero pixels.

Three things the build surfaced, each a correction to the first proposal:

- **No existing box sees the whole knee.** A spec's depth window is
  `lateral_mm + medial_mm`, and the widest on the books — `baker_wide`, 110×95 mm in
  plane — is a **36 mm slab** against a 99 mm sagittal stack **[m]**. The wide branches
  are genuinely new, not a relabelling.
- **The coronal window is symmetric, and that is deliberate.** The joint centre sits
  60 mm from one end of a coronal stack and 41 from the other **[m, 120 series]**, which
  tempts an asymmetric window. Declined: asymmetry makes `symmetric_depth` false, which
  sends `extract` to `bowtie_direction` — a rule measured on the medial-lateral axis,
  applied to an anterior-posterior one. ±40 mm is what both ends support, and since the
  window holds ~90% of the stack either way, the slot cap decides, not the window.
- **Two landmarks now set the centre without setting the extent.** `extract` coupled
  them: `landmark2` meant "window between the two points, inset", which is 27–48 mm —
  a compartment, not a knee. The signal to tell them apart already existed, since `acl`
  is the only two-landmark spec and carries `lateral_mm = medial_mm = 0`. So a zero extent
  derives the window from the points and a non-zero one keeps the spec's. `acl` is
  bit-identical; six tests pin the behaviour.

**Everything else**, for the record: the wide model's label table (v4_blend +
assertedness, the one the 0.899 used, or the comparison is void); the existing fold
assignment; gold held out; `resnet18`; the full bag in training.

---

## 9. What this does not attempt

**It is not a replacement for their pipeline. It is a better member for a stack.** Their
0.943 against our best wide model's 0.899 means the ensemble bought them 0.044, and one
better-framed model will not beat thirty stacked ones however well it is framed.

Also out of reach: the BTKD calibrator, fitted on data we do not have; and their trained
weights, which are weeks of GPU we will not repeat.

What makes a single member worth building anyway is orthogonality, and that is measured
(§1): we disagree with their host more than their legs disagree with each other, and a
blend pays in proportion to disagreement.

---

## 10. Order of work

1. **The window cutter, per group.** Cuts per slot with edge clamping, no cross-series
   triplets. Reuses `ExpertNet.windows()`.
2. **The head.** Per-target attention over windows, plus the positional feature. Raptor's
   six lines with a `Linear` on (plane, signed mm) added.
3. **One fold, 10 epochs, the full bag, `bf16`.** ≈1.1 h by §7. Validates the plumbing
   and the data loader — the only thing §7 did not measure — before five folds are
   committed.
4. **Compare out of fold against 0.899**, on our five folds. No submission. Two runs are
   needed — this preprocessing and theirs — or architecture and preprocessing are
   confounded.
5. **Capacity-aware redistribution**, once the shape works. It is the fix for the 197-study
   hole and it is independent of everything above.

---

## See also

- [`pipeline_v2.md`](pipeline_v2.md) — the ROI specifications and the annotation protocol,
  both of which shipped
- [`atlas/preprocessing.html`](atlas/preprocessing.html) — every leg's input, audited
  against the notebook at build time
- [`atlas/crop.html`](atlas/crop.html) — where the knee actually is, and the boxes
- [`experiments.md`](experiments.md) — the submissions behind §1's leaderboard numbers
