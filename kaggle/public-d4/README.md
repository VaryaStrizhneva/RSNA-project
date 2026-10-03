# The public d4 pipeline, run on our account

A private copy of `pjmathematician/rsna-knee-d4-blend` (archived read-only under
`notebooks/external/rsna-knee-d4-blend/`), pushed as
`mathysgouverneur/rsna-knee-d4-public` to establish what the public frontier scores for
us before anything of ours is blended into it.

**Two deliberate differences from the original.**

The three blank entries in its `dataset_sources` are datasets private to its author;
Kaggle hides their slugs and a push rejects an empty source, so they are dropped. That
is what makes this the public path: `OUR LEG 2/3` then finds nothing and the notebook
keeps what its own comment calls "the public 0.941".

`machine_shape` is set to `NvidiaTeslaT4x2`, which the original leaves to a server-side
setting. It is not optional: `_d4_check_runtime` raises on
`torch.cuda.device_count() != 2`.

**One patch, two lines.** The author's SOFT FAIL only *prints* when the private assets
are missing; the next line still evaluates `_EFF6 / 'arms'` with `_EFF6` at `None`, and
the guard above it is conditioned on `_HIDDEN = n > 10` so it is not even reached on the
three-study public run. Version 1 died there. The two lengths are now computed only when
the paths exist, which lets `OUR LEG 3/3` reach its documented public-only fallback.

**Where our leg would attach.** `OUR LEG 3/3` reads `/kaggle/working/ours_fleet/submission.csv`,
ranks it against the public output and blends at `RSNA_OURS_W` (0.45 by default), after
checking coverage and the tie fraction. Our `run_expert_submission` writes exactly that
shape, so branching in is a matter of pointing its `out` at that path — not of editing
their notebook.
