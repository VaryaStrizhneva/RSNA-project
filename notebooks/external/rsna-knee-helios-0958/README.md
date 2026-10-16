# `heliosli/blend-mine-in-private-lb-luck-to-all-upvoters`, the 0.958 version

Pulled 2026-10-16. Read-only, like everything under `external/`. `sha256` of the
concatenated code: `e00557dc403f9eeba7f15b963caad9173278a67a34e4da4c640ed67f21c1afe7`.

**The same slug as [the 0.956](../rsna-knee-helios-s6-5050-0956/), updated in place.**
`KNOWN` therefore maps this slug to *two* hashes with what each scored, because a single
one would have refused the newer version and silently dated the guard.

| | 0.956 | 0.958 |
|---|---|---|
| code cells | 6 | 7 |
| characters | 499,032 | 596,548 |
| weights mount | `rsna-knee-compact-r6cnxfsss` | `rsna-knee-r6-frozen-fe055406-w1` |
| `BUNDLE_ID` | `209a32c8b50a6b1c` | `fe055406fad6a12f` |

## What changed

Three stages instead of two: nartaa's 384-pixel Raptor, then **nartaa's 224-pixel
CoAtNet** (a 0.8 centre crop of the same field, 0.945 alone) at `A224 = 0.35` of the
Raptor half, then their own R6 with flip TTA. Stage 3 adds **per-finding blend weights**.

**`-w1` is the versioned mirror.** The unversioned `fe055406` was published on the 14th
and Kaggle refuses to submit against it; `-w1` followed on the 15th at 12:02. Same pattern
as `compact-v1` after `compact-r6cnxfsss`. So this host is graftable *and* submittable
where [the 0.9573](../rsna-knee-helios-0957/) was blocked.

## The free information in stage 3

```python
W_RAPTOR_C = {'Medial OA': 0.5, 'Lateral OA': 0.2, 'PF OA': 0.6,
              'Synovitis': 0.8, "Baker's": 0.35}
```

That is the weight their **Raptor half** carries per finding, default 0.5. They give Raptor
**0.2 on Lateral OA** — independent confirmation of our void probe, which measured the
0.956 host at **0.896 there, its weakest of the four probed**. Two unrelated methods, same
column.

The consequence for us: Lateral OA on *this* host is 80 % their own model, so the parity we
measured against the 0.956 (−0.012) does not carry over to it — and that is the column
where our bundle gained most (+0.0110).

## Timing

`DEADLINE_HOURS = 8.3`, unchanged across all three versions, and it **caps** their own work
rather than measuring it: stages 1 and 1b are capped at two hours each and stage 2 runs to
`T0 + 8.3 h`. The extra 224 arm therefore eats their budget inside their cap, not the ~42
minutes of the nine hours our leg runs in. Our grafts onto the 0.956, which carries the
same guard, completed on the real hidden test — that is the only evidence we have that our
leg fits, and it is evidence rather than a measurement.

**Not read in full**: 596,548 characters across seven cells, most of it two vendored
`%%writefile` modules. What is recorded here comes from the stage comments and the blend
cell.
