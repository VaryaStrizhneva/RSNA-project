"""Fitting one expert, and the numbers used to judge it.

The metric is the AUC of a single target, which needs saying because it cannot be read
against the wide model's 0.792 on the same target. That number includes the comorbidity
reasoning — **0.774 of the twelve-target macro is obtainable from the other eleven
labels alone, with no pixels** — and an expert reading a 48 mm crop has none of it. The
honest comparison is against a single-target model on the wide view, which is a separate
run and does not exist yet.

What this run *can* answer on its own: whether a tight crop at 0.21 mm/px carries enough
signal for a model to beat chance on a target the wide pipeline barely sees at all
(+0.035 over comorbidity). If it does not, resolution was not the bottleneck.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from ..eval.metrics import per_target_auc
from . import augment as A
from .config import ExpertConfig


@dataclass
class EpochResult:
    epoch: int
    loss: float
    #: The mean over the group's targets — what the epoch is selected on. A group is
    #: trained together because it shares a region of interest, so it is judged
    #: together; picking the epoch that suits one target would hand the other a model
    #: chosen against it.
    auc: float
    per_target: list = field(default_factory=list)
    positives: int = 0
    #: The same model against the 58 expert-labelled studies, when they were held out.
    #: Never used to choose the epoch — 23 positives against 35 negatives give an
    #: interval 0.22 wide, so selecting on it would be selecting on noise.
    gold_auc: float = float("nan")
    gold_per_target: list = field(default_factory=list)
    #: The same two numbers for the moving average of the weights, when one is kept.
    #: Recorded, never selected on — see `ExpertConfig.ema_decay`.
    ema_auc: float = float("nan")
    ema_gold_auc: float = float("nan")


@dataclass
class FitResult:
    history: list[EpochResult] = field(default_factory=list)
    best: EpochResult | None = None
    state: dict | None = None
    #: The moving average's weights and its held-out scores, when one was kept. Saved so
    #: that preferring it later costs a reload rather than a retraining.
    ema_state: dict | None = None
    ema_scores: np.ndarray | None = None
    scores: np.ndarray | None = None
    truth: np.ndarray | None = None
    studies: list[str] = field(default_factory=list)
    gold_scores: np.ndarray | None = None


def aucs(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """One AUC per column, NaN where a column holds a single class."""

    y = (np.asarray(y) > 0.5).astype(int)
    p = np.asarray(p)
    if y.ndim == 1:
        y, p = y[:, None], p[:, None]
    return per_target_auc(y, p)


def auc(y: np.ndarray, p: np.ndarray) -> float:
    """The mean AUC over the columns given, or one column's if there is one.

    The truth is thresholded at 0.5 because the label table gives probabilities, not
    bits — a report that never mentions a finding scores 0.28 for it, not 0. The wide
    pipeline scores itself the same way (`rsna.train.loop`, line 151), so the two
    numbers are at least measured on the same convention even when they are not
    comparable for other reasons.
    """

    return float(np.nanmean(aucs(y, p)))


def fit(model, groups, y: np.ndarray, w: np.ndarray,
        studies: list[str], train_index: np.ndarray, val_index: np.ndarray,
        config: ExpertConfig, device="cuda", seed: int = 0,
        gold_index: np.ndarray | None = None, gold_y: np.ndarray | None = None,
        log=print) -> FitResult:
    """Fit one fold. Keeps the epoch with the **best held-out AUC**, and its weights.

    This is what `rsna/train/loop.py` has always done, and what this loop did until
    2026-10-03, when it was switched to the last epoch and switched back a day later.
    The reasoning is at the selection site below; the short version is that the switch
    made every expert-against-wide comparison a last epoch against a best one.

    `gold_index` names studies a radiologist read, kept out of training by the caller.
    They are scored every epoch and reported, never selected on: the weak-label AUC says
    whether the model agrees with the label extractor, and the gold one says whether the
    extractor was worth agreeing with. A model at 0.86 on the first and 0.55 on the
    second has learned the extractor and nothing else.
    """

    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=config.lr,
                            weight_decay=config.weight_decay)
    steps = max(1, config.epochs * int(np.ceil(len(train_index) / config.batch)))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=config.lr, total_steps=steps)
    rng = torch.Generator(device=device).manual_seed(seed)
    order_rng = np.random.default_rng(seed)
    result = FitResult(studies=[studies[i] for i in val_index],
                       truth=y[val_index].copy())
    counts = " ".join(str(int(c)) for c in (y[val_index] > 0.5).sum(axis=0))
    log(f"  {len(val_index)} held out, positives per target: {counts}")

    # The moving average rides alongside and decides nothing. Integer buffers —
    # `num_batches_tracked` is one — are copied rather than averaged: a fractional count
    # of batches is not a number, and `lerp_` on an integer tensor raises.
    ema = None
    if config.ema_decay > 0:
        ema = copy.deepcopy(model).eval()
        for parameter in ema.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update_ema():
        live = model.state_dict()
        for key, value in ema.state_dict().items():
            other = live[key]
            if value.dtype.is_floating_point:
                value.mul_(config.ema_decay).add_(other.detach(),
                                                  alpha=1.0 - config.ema_decay)
            else:
                value.copy_(other)

    def load(idx):
        """One batch of every ROI group, in the order the model expects them.

        The slot-level mask, not the series-level one: the network needs to know which
        windows hold an acquired slice so the padding never reaches the trunk.
        """

        return [(torch.as_tensor(np.asarray(v[idx]).copy(), device=device).float() / 255.0,
                 torch.as_tensor(m[idx].astype(bool), device=device))
                for v, m in groups]

    for epoch in range(1, config.epochs + 1):
        model.train()
        total, seen = 0.0, 0
        shuffled = train_index[order_rng.permutation(len(train_index))]
        for start in range(0, len(shuffled), config.batch):
            idx = np.sort(shuffled[start:start + config.batch])
            batch = [(A.augment_batch(v, rng), m) for v, m in load(idx)]
            logits = model(batch)
            target = torch.as_tensor(y[idx], device=device).float()
            weight = torch.as_tensor(w[idx], device=device).float()
            if target.ndim == 1:
                target, weight = target[:, None], weight[:, None]
            loss = (nn.functional.binary_cross_entropy_with_logits(
                logits, target, reduction="none") * weight).sum() / weight.sum().clamp(min=1e-6)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            if ema is not None:
                update_ema()
            total += float(loss.detach()) * len(idx)
            seen += len(idx)

        model.eval()

        def score(index, module=None):
            module = model if module is None else module
            out = []
            with torch.no_grad():
                for start in range(0, len(index), config.batch):
                    out.append(torch.sigmoid(
                        module(load(index[start:start + config.batch]))).cpu().numpy())
            return np.concatenate(out) if out else np.array([])

        p = score(val_index)
        gold_p = score(gold_index) if gold_index is not None and len(gold_index) else None
        per = aucs(y[val_index], p)
        gper = aucs(gold_y, gold_p) if gold_p is not None else np.array([np.nan])
        ema_auc = ema_gold = float("nan")
        if ema is not None:
            ema_p = score(val_index, ema)
            ema_auc = float(np.nanmean(aucs(y[val_index], ema_p)))
            if gold_index is not None and len(gold_index):
                ema_gold = float(np.nanmean(aucs(gold_y, score(gold_index, ema))))
            result.ema_scores = ema_p
            result.ema_state = {k: v.detach().cpu().clone()
                                for k, v in ema.state_dict().items()}
        r = EpochResult(epoch, total / max(seen, 1), float(np.nanmean(per)),
                        [float(x) for x in per], int((y[val_index] > 0.5).sum()),
                        float(np.nanmean(gper)), [float(x) for x in gper],
                        ema_auc, ema_gold)
        result.history.append(r)
        # **The best held-out epoch, which is what the wide model has always kept.**
        #
        # This was the last epoch for a while, on a measurement that replayed the argmax
        # over 35 fold-runs and found it bought +0.018 of held-out AUC while losing 0.003
        # on the 58 gold studies: selecting noise in the weak labels rather than a better
        # model. Two findings reversed it.
        #
        # First, the asymmetry. `rsna/train/loop.py` keeps `best_state` on
        # `holdout_auc > best_auc`, and always has. So every expert-against-wide
        # comparison was a last epoch against a best one, biased against the expert. On
        # Baker's that was the whole conclusion: 0.8851 last against the wide model's
        # 0.8837 reads as a loss, and 0.9128 best against the same 0.8837 reads as
        # **+0.029**.
        #
        # Second, the aggregate hid a split. Over 85 curves the argmax gained −0.0008 on
        # the gold — a coin flip — but those curves are mostly gentle plateaus where the
        # argmax is noise. `baker_axial` falls 0.046 from its peak, and there the argmax
        # gains **+0.031 on the gold, on three folds out of three**. The right rule
        # depends on how steeply a run overfits, which averaging erased.
        #
        # What this costs is honesty about the number: selecting on the held-out fold and
        # then reporting that fold is optimistic by construction, and the bias is roughly
        # the 0.020 that peak-minus-last measures. It is tolerable while the numbers are
        # used to compare models against each other — all of them now carry it equally —
        # and it is not tolerable in anything quoted as a final score. The schedule is
        # the real fix: a run that peaks at epoch 8 of 30 should be a run of 10 epochs,
        # whose last epoch is annealed and needs no selection at all.
        if result.best is None or r.auc > result.best.auc:
            result.best = r
            result.state = {k: v.detach().cpu().clone()
                            for k, v in model.state_dict().items()}
            # With the weights, not beside them. These are what `holdout.csv` and the
            # pooled `oof` are computed from, so leaving them outside the branch -- as
            # the last epoch's -- described a model that is not the one saved, and by a
            # measured 0.025 of AUC per fold on `baker_axial`. The two must move together.
            result.scores = p
            result.gold_scores = gold_p
        detail = " ".join(f"{x:.3f}" for x in r.per_target)
        log(f"  epoch {epoch:3d}  loss {r.loss:.4f}  auc {r.auc:.4f} [{detail}]"
            + (f"  gold {r.gold_auc:.4f}" if np.isfinite(r.gold_auc) else "")
            + (f"  ema {r.ema_auc:.4f}" if np.isfinite(r.ema_auc) else "")
            + (f"/{r.ema_gold_auc:.4f}" if np.isfinite(r.ema_gold_auc) else ""))
    return result
