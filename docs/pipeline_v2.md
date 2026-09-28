# Pipeline v2 — one input specification per pathology

**Status: design in progress.** Nothing here is implemented yet except the annotation
tooling. This document is the decision record: what we are building, why, and which
claims are measured rather than assumed.

Every number below is marked **[m]** when it was measured on this corpus, with the
sample size. Anything unmarked is a proposal or a judgement.

---

## 1. Why

The current model gives all twelve targets **the same input**: six slots, the whole
knee, 336 px over a 130 mm crop. Measured against what each target is actually worth,
that input is right for some and wrong for others.

**How much each target gains from the image**, over predicting it from the other eleven
labels alone **[m, 4406 studies]**:

```
Baker's           +0.197   ┐
Medial Meniscus   +0.117   │  diffuse or large findings
ACL               +0.094   │  the current input is adequate
MCL               +0.090   │
Effusion          +0.087   │
Contusion         +0.085   │
Synovitis         +0.063   │
Fracture          +0.049   ┘
------------------------------------------------
Lateral Meniscus  +0.035   ┐
PF OA             +0.022   │  focal or structural findings
Medial OA         +0.019   │  the current input is wrong
Lateral OA        −0.017   ┘  (the model does worse than comorbidity)
```

The mechanism, in one line **[m]**: a meniscal tear is ~1.5 mm; one DINOv2 patch token
covers **5.4 mm** at our resolution; the slot feature then averages over 576 of them.
The finding is a quarter of the encoder's smallest unit of position.

| configuration | mm/px | mm/token | tear in tokens |
|---|---|---|---|
| **today**: 130 mm crop, 336 px, patch 14 | 0.387 | 5.42 | **0.28** |
| ROI: 40 mm crop, 224 px, patch 14 | 0.179 | 2.50 | 0.60 |
| ROI: 40 mm crop, 224 px, **ViT patch 8** | 0.179 | 1.43 | **1.05** |
| ROI: 40 mm crop, 224 px, **CNN stride 4** | 0.179 | 0.71 | 2.10 |

---

## 2. Architecture

**Add ROI branches beside the current wide view — do not replace it.**

```
wide view (today's 6 slots)  ->  encoder  ->  vector  ┐
ROI branch A                 ->  encoder  ->  vector  ├─> attention -> 12 logits
ROI branch B                 ->  encoder  ->  vector  │
...                                                   ┘   + presence mask
```

Three reasons for the shape:

- **It is a strict generalisation.** If a ROI branch adds nothing, the attention can
  weight it to zero and we recover today's model. It cannot be worse, only equal or
  better.
- **The wide view carries the comorbidity reasoning**, which is most of today's
  performance **[m: 0.774 of 0.844 is obtainable from the other labels alone]**. A
  specialist that only sees a 40 mm crop loses it.
- **Several modalities at once, not one.** A branch is (plane, weighting, ROI). A
  pathology gets as many as it needs, each with its own presence mask.

**Never stack different planes as channels.** Pixel (i,j) of a sagittal slice and of a
coronal slice are different anatomy; a convolution assumes its channels are co-located.
Separate branches, fused after encoding.

---

## 3. The ROI groups

The twelve targets do not need twelve ROIs. They need about seven, and two of them are
what we already have.

| ROI | targets | status |
|---|---|---|
| **lateral compartment** | Lateral Meniscus, Lateral OA | **§5, in progress** |
| **medial compartment** | Medial Meniscus, Medial OA | after the lateral |
| patellofemoral | PF OA | not specified |
| intercondylar notch | ACL | not specified |
| medial collateral | MCL | not specified |
| whole joint, fluid-sensitive | Effusion, Synovitis, Baker's | **= today's input** |
| bone marrow, fat-suppressed | Contusion, Fracture | **= today's input** |

The last two rows are the finding: today's single input is already correct for five of
twelve targets, and those are exactly the five the model sees best.

---

## 4. Finding the ROI

**The key decision: landmarks are stored and predicted in patient millimetres, not
pixels.** All series of a study share the patient frame — verified **[m: the x-range a
sagittal stack sweeps is contained in the x-range a coronal image covers, on 119 of 120
studies]** — so a landmark found once indexes into every series of that study, whatever
its plane.

That is why one annotation pass, on one sequence, serves every branch.

### Three ways to get the landmark, by cost

1. **Geometry + classical image processing.** Free, no annotation. Rejected as the sole
   method: a centred crop does **not** contain the joint reliably — the joint line
   shifts by several centimetres and tilts by up to ~40° between studies (see
   `figures/centred_crop.png` in the annotation bundle).
2. **A landmark model trained on manual annotations.** ← **what we are doing.** See §6.
3. **Learned attention (MIL), no ROI at all.** Fixes the supervision problem but not the
   resolution one: the whole 130 mm image still goes in.

### Architecture of the landmark model (proposed, not built)

One encoder, one head per landmark — **not** one model per landmark. Measured: 7 heads
on one encoder is **14.9 M** parameters against **38.9 M** for 7 separate models
**[m]**, and one forward pass instead of seven.

- **Output a heatmap, not coordinates.** A coordinate is a global, non-spatial output
  and convolutions are bad at it. *(The RSNA-2024 lumbar winner regressed coordinates
  directly and won — but they had thousands of supplied coordinate labels. We will have
  ~170. In a low-data regime the structured output wins.)*
- **Encoder ~4–15 M parameters, pretrained.** ConvNeXt-base (88 M) would memorise 170
  annotations.
- **Resample to fixed mm/px, not to a fixed array size.** Convolutions are translation
  invariant, not scale invariant, and our spacing ranges 0.19–0.56 mm/px **[m]**.
- **Balance left and right knees, and augment by reversing the stack** with the labels
  swapped — otherwise the model learns "lateral = high slice index" instead of the
  anatomy, and that shortcut works perfectly until it meets the other knee.

---

## 5. Lateral Meniscus — the first specification

### What it is

Fibrocartilage, nearly a closed **O** seen from above. A sagittal plane crosses it
twice, so on a sagittal slice it appears as **two dark triangles** on the tibial plateau
— the **anterior horn** in front, the **posterior horn** behind — apexes pointing toward
the centre of the joint.

Normal fibrocartilage has almost no free water, so it is **uniformly black on every
sequence**. That is the whole basis of the diagnosis.

### How it is read

> **Bright signal inside the black meniscus that reaches an articular surface.**

| grade | what is seen | meaning |
|---|---|---|
| 1 | globular bright spot inside, not touching a surface | degeneration, not a tear |
| 2 | linear bright signal inside, not touching a surface | degeneration, not a tear |
| 3 | bright signal **reaching a surface** | **tear** |

Grades 2 and 3 differ only by whether a thin line touches an edge — a sub-millimetre
judgement, which is why the resolution argument in §1 matters for this target above all.

*(Textbook radiology, written down for orientation. Not expert knowledge — check it
against a real reference before relying on it.)*

### Modalities — several at once

A branch is a **(plane, weighting, fat suppression)** triple with its own presence mask.
Nothing is ever *chosen between* — a study that has two useful series contributes both.

| branch | coverage **[m, 300 studies]** | what it contributes |
|---|---|---|
| **`SAG_PD_NOFS`** | **36.3 %** | short TE without fat suppression shows signal *inside* the fibrocartilage best, which is exactly what the grade 1/2/3 scale is defined on |
| **`SAG_PD_FS`** | **81.3 %** | the same anatomy with fat suppressed: better for the oedema and fluid around the tear, worse for the substance |
| *sagittal — at least one* | **99.7 %** | |
| *sagittal — both* | **18.0 %** | |
| **`COR_PD_FS`** | **85.7 %** | extrusion, root tears, meniscocapsular separation — **17 % of meniscus-positive reports mention a finding judged chiefly on coronal** **[m, 2645]**. Also the only plane our pipeline already laterality-normalises, so medial and lateral sit at fixed image positions. |
| **`COR_PD_NOFS`** | **15.0 %** | the coronal counterpart of `SAG_PD_NOFS`: unsuppressed short TE, intrasubstance signal |
| *coronal — at least one* | **90.3 %** | |
| *coronal — both* | **10.3 %** | |

Four branches, all of them proton density. `SAG_T2` (~50 %) and `COR_T2_FS` (16.3 %)
remain plausible and are deliberately held back to step 3 of §9 — adding them at the same
time as the ROI would make the result uninterpretable.

### The rule for admitting a branch

> **A branch is admitted if it shows the finding. Coverage decides how often it
> contributes, never whether it belongs.**

The case that makes it unmistakable is **`COR_T1`**: available in **60.3 %** of studies
**[m]** — more than any branch above except the two fat-suppressed ones — and **excluded**,
because intrasubstance meniscal signal is not conspicuous at T1. Meanwhile `COR_PD_NOFS`
is present in only 15.0 % and is admitted. Availability is not an argument: admitting a
sequence because it happens to be there spends the model's capacity on discovering that
it is useless.

The two objections to a sparse branch both fail on measurement:

- **Cost.** A branch costs one encoder pass per study whether present or absent —
  `Model.forward` folds every slot through the encoder and the mask only acts later, in
  the head's softmax. That is **+17 % of encoder passes, which is +0.03 s against a
  24.9 s per-study budget [m]** — 0.1 %.
- **Undertraining.** A slot present in 15 % of studies still sees **661 studies**, which
  is ample for one slot embedding.

What the three coronal findings need is not one contrast but three: extrusion is a
judgement of *position* (the dark meniscus overhanging the tibial edge), a root tear
needs meniscal *substance*, and a meniscocapsular separation needs *fluid*. Both PD
branches contribute; only T1 contributes to none of them.

**Why two PD branches rather than one.** Fat suppression cannot be standardised: 63 % of
studies have only the suppressed variant, 18 % only the unsuppressed, 18 % both **[m]**.
Forcing a choice would discard a useful series in 18 % of studies *and* leave the single
branch's contrast varying without being declared. Two branches with presence masks make
the variation explicit instead — which is what the mask is for.

This costs **one extra encoder pass per study and zero extra parameters**: slots already
share one encoder (`Model.forward` folds `(batch, slot, …)` into `(batch·slot, …)` for a
single pass).

The existing pipeline already splits on fat suppression — `SAG_FLUID_FS` and
`SAG_FLUID_NOFS` are two slots. Its defect is the *other* axis: "fluid" lumps PD together
with T2 (see §8.2). These branches name the weighting explicitly.

**No fallback to T2 when PD is absent.** ~0.2 % of studies **[m]**. A branch trained on
PD and fed T2 at inference sees a distribution it never met and will be confidently
wrong; an *absent* branch is a state the presence mask already expresses and the model
is trained on. Degrade cleanly, not silently. This is the same argument
`rsna.dicom.slots.pick_slots` already makes in its docstring.

### The ROI

- **Extent**: ~40 mm around the meniscus centre. Tolerance is generous — a 40 mm crop
  still contains a 15 mm meniscus if the landmark is within **±12 mm**.
- **Depth**: ~15 mm, i.e. 4–5 slices at the median 3.4 mm spacing **[m]**.
- **How found**: the landmark model of §4, trained on the manual annotations of §6.

### Measured facts specific to this target

| | |
|---|---|
| our model, holdout AUC | **0.792** — worst of twelve |
| gain over comorbidity | **+0.035** — it barely sees it |
| label extraction quality | **0.841** against the 58 expert studies, **10 % UNK** — the reports *do* record it, so this is not an information failure |
| **realistic ceiling** | **~+0.05 on this target ≈ +0.004 macro.** A model cannot much exceed its labels. The value of the experiment is **diagnostic**, not competitive: if a ROI branch moves this target, the same fix applies to the other focal ones. |
| where reports localise it **[m, 1121 positives]** | posterior horn 61 %, body 50 %, anterior horn 40 % (of the 442 that say). Free weak supervision for an auxiliary task. |

---

## 6. The annotation protocol (running now)

**Tooling**: `tools/annotate/` — see its README. Bundle at
`/data/mgr/rsna-knee/bundle-v2`, served locally, never published.

| | |
|---|---|
| **what** | one point: `lat_centre`, midway between the two horn tips, at the joint line |
| **on what** | the sagittal PD, one series per study |
| **how many** | 170 studies |
| **selection** | `--tagged-only`: side from the DICOM tag only |
| **stored as** | `SOPInstanceUID` + raw pixel + derived patient millimetres |

**Why `--tagged-only`.** The geometric fallback for the side disagrees with the DICOM tag
on **9.4 %** of the studies where both exist **[m, 607 studies]** — and a wrong side means
annotating the **opposite meniscus**, 40 mm away, with confidence. The tagged half is
51 % of the corpus **[m]**; the model learns the anatomy from clean data and resolves the
untagged half itself.

Bundle composition **[m]**: 86 left knees, 84 right — near-perfect balance, which is what
kills the "lateral = high slice index" shortcut. 100 % side-from-tag, 170/170 with usable
slice geometry.

### What the annotation set does not cover

Two gaps, measured on the bundle against the corpus. Both are recorded rather than fixed,
because fixing either changes what the bundle is.

**1. Vendor skew — a side effect of `--tagged-only`.**

```
                corpus (200)   bundle (170)
  SIEMENS            45.0 %        78.8 %
  PHILIPS            32.0 %        18.2 %
  GE                 16.5 %         1.8 %     <- 3 studies
  TOSHIBA             4.5 %         0.0 %
  CANON               1.5 %         0.0 %
```

The cause is mechanical: the `Laterality` tag is written **by vendor**, not at random
**[m, 300 studies]** — Siemens 86 %, Philips 36 %, **GE 2 %**, Toshiba 0 %, Canon 0 %. So
selecting tagged studies selects Siemens. `docs/pipeline_pitfalls.md` already said the tag
was missing "by whole vendors rather than scattered series"; we did not connect it to the
filter.

**Consequence**: a landmark model trained on this bundle has seen almost no GE, Toshiba or
Canon — together **22 % of the corpus**. It may fail on them, and nothing in training or
validation would reveal it.

**The recovery, if we want it later**: admit untagged studies whose median |x| ≥ 80 mm,
where geometry agrees with the tag **97.7 %** of the time **[m, 309 studies]** against
90.6 % overall. On a 300-study draw that adds **+33 GE, +34 Philips, +4 Toshiba, +4
Canon**. The trade is 2.3 % side error against covering four vendors — and a side error is
detectable afterwards (the model's own prediction can be checked against geometry) where an
absent vendor is not.

**2. No 3D series — but the deployment will have them.**

The bundle holds **0 studies** with a sagittal series over 100 slices; the corpus has
12.8 % **[m]**. More precisely, for the branches of §5:

- **The two coronal branches can never receive one**: there are **zero coronal 3D series**
  in the corpus **[m, 709 series >100 slices: 373 axial, 336 sagittal, 0 coronal]**.
- **`SAG_PD_FS` will**, in **3.4 % of studies (148)** **[m]**. Of the 336 sagittal 3D
  series, 45 % classify as PD fat-suppressed; the rest are T1 or GRE and fill nothing.
- **This is not a selection choice**: in **147 of those 148 studies the 3D series is the
  only sagittal PD fat-suppressed series there is**. Preferring 2D would change one study.

Those series carry **320 slices at 0.6 mm spacing** against ~30 at 3.4 mm. Physical
resampling absorbs the difference — a fixed-millimetre window gives the same five planes
either way — but the landmark model will meet an appearance it never trained on.

---

**Before scaling up**: both annotators do the same 20 studies blind and run
`to_mm.py --agreement`. That number is the floor on any model's accuracy. A 40 mm ROI
tolerates ~12 mm; if the p90 exceeds that, the landmark definition needs work, and it is
much cheaper to learn that at forty clicks than at six hundred.

---

## 7. Geometry facts this pipeline depends on

Measured once, recorded here so they are not re-derived or mis-remembered.

| fact | measured on | value |
|---|---|---|
| sagittal stacks run in **decreasing patient x** | 600 series | **600 / 600**, median n_x −0.991 |
| therefore **slice 0 is medial on a right knee and lateral on a left one** | — | and nothing in the pipeline corrects it — see §8 |
| coronal stacks run anterior → posterior; axial inferior → superior | 3 studies × 2 planes | consistent, and **not** reversed between knees |
| side from the DICOM `Laterality` tag | 200 studies | **51 %** |
| side from geometry | 200 studies | 38.5 %, and it disagrees with the tag **9.4 %** of the time (607 studies) |
| side unresolvable | 200 studies | **10.5 %** — all within 20 mm of the midline |
| patient coordinates comparable across series | 120 studies | **99.2 %** (the `FrameOfReferenceUID` differs on 100 % of studies — an anonymisation artefact, not a real one) |
| slices per series | all 24,371 | median **30**, p25–p75 25–34, range 11–320 |
| slice spacing | 58 studies | median **3.4 mm** — and `SliceThickness` is **not** the spacing (one study: 4 mm thick, 5 mm apart) |
| in-plane spacing | 336 series | median 0.33 mm, p5–p95 0.19–0.56 |
| 3D isotropic series | all | 2.9 % of series, **12.8 % of studies**, 0.6 mm spacing, 1.4:1 anisotropy. Target the **cartilage**, not the meniscus. Their studies are a different population **[m: Effusion +9.4 pts, Lateral Meniscus −7.9 pts]**, so they are **not** a clean subgroup for evaluation. |

---

## 8. Known defects, found while designing this

Recorded so they are fixed deliberately rather than rediscovered.

1. **The sagittal stack is not reversed for right knees — the port dropped the line.**
   The public baseline does it:

   ```python
   def normalise_laterality(img, plane, lat):          # baseline notebook
       if lat != "R":                    return img
       if plane in ("Coronal", "Axial"): return torch.flip(img, dims=[-1])
       return torch.flip(img, dims=[0])                # <- the sagittal reversal
   ```

   Ours keeps the first two lines and returns `image` for sagittal. It has never held the
   reversal (`git log -S "::-1"`, one commit, `9c7c2c7`), and the docstring was reworded
   from *"the channel order is reversed instead"* into *"is handled by the slice order, not
   here"*, which reads as a deferral rather than a description.

   **What it costs.** The 2.5D triplet `[c-1, c, c+1]` becomes `[c+1, c, c-1]` on half the
   corpus — a real transformation the encoder must spend capacity being invariant to, on a
   nuisance axis we could simply remove. Five of twelve targets are side-defined.
   Normalising takes the corpus from ~50/50 mixed to **~86 % consistent** (not 100 %:
   geometry errs 9 % of the time and 10.5 % stay unresolved, hence unflipped).

   **What it does not cost.** Nothing on the ROI path of §4: the landmark is learned from
   appearance and returned in patient millimetres, and the stack-reversal augmentation makes
   the model side-agnostic by construction. This is a defect of the *current* pipeline, not
   a blocker for the next one.

   **Status: the code is fixed, the data is not.** `PixelRules.sagittal_flip` now exists
   and `normalise_laterality` honours it. It defaults to **False — the defect — on
   purpose**: a manifest written before the field existed replays through
   `Config.from_dict` as the default, so defaulting to True would silently mirror the
   input of all thirteen existing packages at inference, and the fingerprint cannot catch
   it (it is computed on synthetic pixels that never pass through this function). A
   forgotten `True` costs an improvement; a wrong `True` costs correctness.

   Turning it on changes the cache tag — `336px_12sl_130mm_0.20-0.80` becomes
   `…_44e6dd` — so the old cache is refused rather than silently reused. Verified: with
   the rule off, the 36 GB cache still loads and all 31 members reproduce their
   fingerprint distance exactly.

   **When to turn it on.** With the cache rebuild the ROI branches require anyway. Set
   `rules: {sagittal_flip: true}` in the first experiment that rebuilds.

   **Consequence for the record**: no run in `out/` reproduces the published baseline.
   `experiments/RESULTS.md` has been corrected.
2. **Slot weighting is chosen by slice count, not by sequence.** `pick_slots` sorts on
   `n_slices`, so when a study has both a sagittal PD and a sagittal T2 for the same
   slot it is a coin toss — **PD 9, T2 8** in the non-fat-sat sagittal slot **[m]**.
   `SAG_FLUID_NOFS` therefore holds T2 **43 %** of the time, and the slot does not mean
   the same thing across studies.
3. **`UNK` maps to the constant 0.28 for all twelve targets**, whose positive rates
   range from 7 % to 61 %. On Synovitis, 84 % of cells are UNK and **62 % of that
   target's training weight** teaches the model to emit a constant **[m]**.
4. **The `__conf` column carries no information** — it is a lookup on the verdict
   (YES 0.95, NO 0.85, UNK 0.05). Which explains why `weights="confidence"` and
   `weights="uniform"` scored 0.0001 apart.
5. **Severity is conflated with probability** — a mild tear is scored 0.68, but a mild
   tear is certainly a tear. Severity belongs in the loss *weight*, not the target
   *value*.

Items 3–5 need **no re-extraction**: they are a different reading of the CSV we already
have.

---

## 9. Experiment order

Each step isolates one variable. The baseline already includes coronal and axial views,
so what is added at each step is a **cropped, high-resolution** view, not a plane.

| | what changes | question answered |
|---|---|---|
| 0 | — | baseline, measured: **0.792** on Lateral Meniscus |
| 1 | fix defect §8.1 | prerequisite; nothing depth-indexed is trustworthy without it |
| 2 | + **sagittal PD ROI branch** | **was resolution the bottleneck?** |
| 3 | + coronal ROI branch | does a tight coronal crop add anything beyond the wide coronal we already have? |
| 4 | + medial compartment (same branches, other side) | does the ROI approach generalise, and does it lift `Medial OA` / `Lateral OA` too? |

**Prediction, recorded before measuring**: step 2 should move, step 3 probably not —
extrusion and root tears are displacements of several millimetres, which survive 5.4 mm
tokens far better than a 1.5 mm tear does. If the opposite happens, the resolution
hypothesis needs revisiting.
