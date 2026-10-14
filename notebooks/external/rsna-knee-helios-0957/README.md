# `heliosli/0-9573-latest-public-dataset-score-in-one-click`

Pulled 2026-10-14 with `kaggle kernels pull`. Read-only, like everything under
`external/`. `sha256` of the concatenated code:
`a00d033f1de72d294924bc389ce108f73561c51bd1b83d02effff72692500731`, pinned in
`scripts/kaggle_graft.KNOWN`.

**The banner claims 0.9573, which is above our own standing 0.957** — so the bare public
notebook now beats our best submission, in one click and for anyone.

## What it is, against the 0.956 we had been grafting onto

| Stage | What runs | Weight |
|---|---|---|
| 1 | nartaa's 384-pixel Raptor CoAtNet | 0.65 of the Raptor half |
| 1b | **nartaa's 224-pixel CoAtNet** — new | 0.35 of the Raptor half |
| 2 | heliosli's own R7, `E_lrprior`, bundle `8c7b666e`, 4 folds | — |
| 3 | rank blend, Raptor half 0.5 / theirs 0.5, fixed before any run | — |

Two changes from `blend-mine-in-private-lb-luck-to-all-upvoters`: the 224 arm, and a new
bundle for their own leg (`rsna-knee-compact-r6cnxfsss` → `rsna-knee-r6-frozen-fe055406`,
published the same day at 18:12).

**The 224 checkpoint is one we already carry.**
`nartaa/rsna-knee-publication-swa-weights-20261007` holds two files —
`raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt` and
`..._c08_r224_swa.pt` — and we have mounted that dataset since 7 October while using only
the 384. So half of this notebook's gain over the 0.956 needed no new download at all.

Each stage runs as its own subprocess so its GPU memory is released before the next, and a
failed stage stops the notebook rather than letting it submit a single model silently.
`DEADLINE_HOURS = 8.3`.

**Not read in full**: 596,327 characters of code across seven cells, most of it two
vendored `%%writefile` modules. What is recorded above comes from the stage comments and
the blend cell, which are the parts that decide the arithmetic.
