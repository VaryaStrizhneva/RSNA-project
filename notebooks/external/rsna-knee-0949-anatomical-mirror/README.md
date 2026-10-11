# `nartaa/rsna-knee-0949-anatomical-mirror`

Pulled 2026-10-07 with `kaggle kernels pull nartaa/rsna-knee-0949-anatomical-mirror -m`.
Read-only, like everything under `external/`. The banner claims **public LB 0.949**,
which is above our own 0.947.

**`matterhorn3838/rsna-knee-v2-velciraptor-dinosaur-speed` is the same notebook.** Not a
variant, not a fork with changes: the concatenated code is byte-identical, `sha256
acd9241f171a807957f8eb8993976f649a479ca2f20d29037449d229578f352d`, same single weights
mount, same verification cell. Two titles, two accounts, one recipe. Only this copy is
kept.

The author publishes `sha256 bcd864cb…a79c5` for the large cell as the hash of the
*scored* source, and the pulled bytes match it. That is the only evidence available that
the public code is the code that earned the score, so the hash is pinned in
`scripts/kaggle_graft.py`'s `KNOWN` and the graft refuses to build if it moves.

## What it is

**One model.** No ensemble, no multi-arm blend.

```python
ARMS = [{"file": "raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt",
         "arch": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", "res": 384, "w": 1.0}]
```

One seed (42), one SWA checkpoint averaging epochs [14, 11, 12]. The author's own comment
says why the ensemble went away: *"on the live leaderboard CoAtNet alone scored 0.914
while every blend scored 0.914-0.915, so ensembling is worth ~+0.001 there — the ~+0.010
it showed on the old 45-study gold set was gold-set noise."* That is our own
"unbiased is not precise" lesson, reached independently.

The input is the Raptor layout at 96 slices, `CROP_MM = 140`, native 384 px, centre-cropped
to 320 at inference, 94 windows of three consecutive slices:

| Slot | Plane | Fat-sat | Slices |
|---|---|---|---|
| 0 | Sagittal | yes | 26 |
| 1 | Sagittal | no | 22 |
| 2 | Coronal | yes | 18 |
| 3 | Coronal | no | 12 |
| 4 | Axial | either | 18 |

## Where the 0.949 comes from

Three levers, and not one of them is an architecture:

| Lever | Measured effect |
|---|---|
| Corpus 3,155 → 4,349 studies | **+0.0131** on the 58-study gate, better in 92.7 % of 2,000 bootstraps |
| OAI (Osteoarthritis Initiative, via NDA) as external training data | folded into the above; not an inference input |
| Anatomical mirror TTA, two views averaged 50/50 | not isolated by the author |

Their per-finding gains on the gate land almost entirely on columns **our experts own**:
Lateral Meniscus +0.071, Fracture +0.057, Lateral OA +0.048, Medial Meniscus +0.035,
ACL +0.028. So this host has moved onto our ground, and the `+0.004` our leg measured
over `nocoat` should not be expected to transfer intact.

## The mirror is worth reading twice

```python
# sagittal (slots 0,1) -> reverse the 3 slice-channels;
# coronal/axial (2,3,4) -> flip the width.
return torch.where(sag, tri.flip(1), tri.flip(-1))
```

Mirroring a knee left-to-right is **not one operation**. On a sagittal stack the
medial-lateral axis is the through-plane normal, so the mirror is a *depth* reversal; a
naive in-plane `flip(-1)` there would reverse anterior-posterior instead and put the
patella behind the joint. Only coronal and axial take the width flip. It is applied to
the uint8 volume *before* the per-channel LUT — "mirror the knee, then normalise".

We hold the laterality already (`rsna.dicom` sagittal side recovery, the `lateral` sign
in `roi/extract.py`), so this is applicable to our own experts without retraining.

## What we still bring over it

Resolution and centring, narrowed but not gone. Its 320-crop of a 140 mm field is
**0.365 mm/px over 117 mm centred on the image**; our boxes are **0.214–0.286 mm/px over
32–60 mm centred on a predicted landmark**. Against the older public stack's 0.42 the
factor was 2; here it is 1.3 to 1.7.

It reads all three planes, with 18 axial slices rather than the 12 of the older Raptor —
which is the plane Baker's axial expert exploited.

## Can we run it as-is

Yes, and we do. The single weights dataset
`nartaa/rsna-knee-publication-swa-weights-20261007` is public and mounts for us; the
notebook asserts the checkpoint's sha256 before loading and substitutes nothing. Internet
is off, no pip installs, two T4s expected.

`scripts/kaggle_graft.py` pushes these two cells **unmodified** plus one appended cell —
our eight experts, blend 0.5 in rank space — as
`mathysgouverneur/rsna-knee-anatomical-mirror-experts`. The host writes
`/kaggle/working/submission.csv` as **rank percentiles**, not probabilities, which our
`np.sort(host)[rank]` blend handles unchanged since it preserves whatever marginal the
host emits.
