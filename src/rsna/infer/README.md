# Inference

The scored notebook runs with **internet off, on Kaggle hardware, inside 9 hours**.
That budget is the same for everyone, so nothing our own machine can do makes it
looser — it is the one constraint we share with every competitor.

For scale: the public baseline's 20 members × 10 windows × 6 slots is **1,200 encoder
passes per study**. At the 0.067 s per pass we measured, ~13 s per study, ~4.8 h for
~1,300 studies. Our own submission of it took about that.

Below is what the public notebooks do about it. Provenance is named because most of it
comes from one notebook, and one notebook is not a consensus.

---

## 1. Two GPUs, one worker each

*Source: `bend-the-knee-to-the-dinosaurs`, cells 4 and 14.*

Kaggle offers a **T4 ×2** accelerator. The baseline ignores it (`dev = torch.device("cuda")`,
one device); dinosaurs runs one worker thread per device, pulling members off a shared
queue:

```python
DEVS = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())
        if _cuda_execution_probe(i)]
threads = [threading.Thread(target=worker, args=(d,)) for d in DEVS]
```

**A factor of two, for no modelling decision at all.** Set `machine_shape` accordingly
in the kernel metadata; the queue makes the split self-balancing, so a slow member does
not idle the other card.

## 2. Probe each device before trusting it

Same source. `_cuda_execution_probe(i)` actually *runs* something on each GPU and drops
the ones that fail.

This is the defence against being assigned a **P100**: compute capability sm_60, for
which the installed PyTorch ships no kernels, so the run dies at the first allocation.
It killed our first baseline run. Pinning `"machine_shape": "NvidiaTeslaT4"` avoids it;
probing survives it even when the pin does not hold.

## 3. Reduced precision

Same source, cell 23. `AMP_PREF = "bf16"` with an automatic fall back to `float16`
below sm_80 — so fp16 on a T4. Roughly 1.5–2× on encoder passes.

## 4. A scheduler that surrenders work instead of failing

Same source, cell 14. The most valuable idea here, and the least obvious.

It **measures** the cost of a member as the run proceeds — a fixed part and a per-window
part — then computes what the remaining time affords and degrades in two stages:

```python
left = TIME_BUDGET - (time.time() - T0)
room = afford / max(slots_left, 1)
n_win = int((room - est["fixed"]) / per_win)
if n_win < len(starts_full):
    mid = (len(starts_full) - n_win) // 2
    starts = starts_full[mid:mid + n_win]   # keep the CENTRAL windows
if est["fixed"] + est["win"] > room:
    pending.clear()                          # not one more member fits
```

- first it **drops windows**, keeping the central ones — the ends of a stack are mostly
  soft tissue outside the joint
- then it **abandons the remaining members** entirely

Members are queued sorted by holdout score descending, so what gets sacrificed is the
weakest. The run always returns something rather than timing out.

Note `predict_member(..., starts=...)` in the baseline takes the window list as a
parameter for exactly this reason — pilkwang's docstring says so — but the baseline
never reduces it.

## 5. Write a fallback submission first

*Source: `rsna-knee-baseline-v1`, `main()`.* Before anything expensive:

```python
write_benchmark_submission()      # 0.5 everywhere
```

A crash after the decode pass then still scores 0.500 instead of nothing. It is also
how a failed run can look successful — see `docs/pipeline_pitfalls.md` §10.

## 6. Per-target window pooling

Same source, cell 14. Not a speed-up, but it lives in the same code and is worth
recording:

```python
mode in ("max", "mean", "logit_mean", "top2", "top3")
```

The 10 windows are not averaged uniformly. A focal lesion appears in **one** window, so
a mean over ten dilutes it by an order of magnitude — `max` or `top3` suits it, while a
diffuse sign suits `mean`. The mapping is per finding.

---

## What we should carry over

| | Cost to us | Confidence |
|---|---|---|
| T4 ×2, one worker per device | one metadata field + a queue | high — pure engineering |
| Probe devices before use | ~15 lines | high — it has already bitten us |
| fp16 autocast | one context manager | high |
| Budget-aware degradation | a day's work | high — the alternative is a timeout on 21 October |
| Fallback submission first | trivial | high |
| Per-target window pooling | needs validation we cannot do on 58 studies | low — a modelling choice, not plumbing |

The first five are plumbing: they change how fast, never what is predicted, so they can
be adopted without any experiment. The sixth is a modelling decision and has to wait for
a reference to measure against.
