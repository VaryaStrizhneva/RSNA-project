"""Fitting one model, and the two numbers used to judge it.

The public baseline runs this inline inside `main()`, mixed with orchestration. Here
the loop takes arrays and a config and returns a result, so it can be called from a
notebook, a script, or a test with sixteen synthetic studies.

Two measurements are reported per epoch and they are not interchangeable:

* **holdout** — macro AUC against the report-derived targets of the held-out fold.
  This is what selects the epoch, because it is the only measurement with enough
  studies behind it to separate one from another.
* **annotation** — macro AUC against the expert labels of the studies that fell in
  the holdout. It measures something better (agreement with a reading of the *images*
  rather than of the reports) on far too few studies: with 58 expert labels spread
  over five folds, a single fold holds between 6 and 20. Reported, never used to
  choose.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from ..config import TARGETS, Config
from .augment import augment, take_group


def macro_auc(y: np.ndarray, p: np.ndarray) -> float:
    """The competition metric. A target with one class present contributes nothing."""

    from sklearn.metrics import roc_auc_score

    scores = [roc_auc_score(y[:, j], p[:, j]) if len(set(y[:, j])) > 1 else np.nan
              for j in range(y.shape[1])]
    if all(np.isnan(s) for s in scores):
        return float("nan")
    return float(np.nanmean(scores))


@torch.no_grad()
def predict(model, cache, mask, index, config: Config, device,
            img_size: int | None = None) -> np.ndarray:
    """Average the logits over the groups of each slot.

    Training sees one group at a time, which acts as augmentation along the stack;
    inference averages over all of them, so a prediction does not depend on which group
    a single draw happened to pick. Where the cache holds one group per slot the two
    coincide.
    """

    model.eval()
    out = []
    for start in range(0, len(index), config.eval_batch):
        sel = index[start:start + config.eval_batch]
        m = torch.as_tensor(mask[sel]).to(device)
        acc = None
        for g in range(config.n_group):
            # Gathered a group at a time rather than whole and then sliced. The two are
            # the same pixels, but taking a whole study out of the cache allocates every
            # slice it holds — most of which this pass will not look at until a later
            # iteration, by which time they have been fetched again.
            lo = g * config.group
            rows = torch.as_tensor(
                np.ascontiguousarray(cache[sel, :, lo:lo + config.group])).to(device)
            with torch.autocast("cuda", enabled=str(device).startswith("cuda")):
                z = model(rows, m, img_size).float()
            acc = z if acc is None else acc + z
        out.append(torch.sigmoid(acc / config.n_group).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, len(TARGETS)), np.float32)


@dataclass
class EpochResult:
    epoch: int
    loss: float
    holdout_auc: float
    annotation_auc: float


@dataclass
class FitResult:
    """What a run produced, and enough to say what it was."""

    best_epoch: int
    best_holdout_auc: float
    state_dict: dict
    history: list[EpochResult] = field(default_factory=list)
    config: Config | None = None


def fit(model, cache, mask, y, w, train_index, holdout_index, config: Config, device,
        img_size: int | None = None, gold_index=None, gold_y=None,
        on_epoch=None) -> FitResult:
    """Fine-tune one model and keep the state that scored best on the holdout.

    `w` carries the per-target loss weight built by `folds.build_targets`: expert
    labels at `config.gold_weight`, report-derived ones scaled by their source's
    confidence. Weighting inside the loss rather than resampling keeps every study in
    every epoch, which matters when the good labels number 58.
    """

    img_size = config.img if img_size is None else img_size
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)

    optimiser = torch.optim.AdamW([
        {"params": [p for p in model.backbone.parameters() if p.requires_grad],
         "lr": config.lr_backbone},
        {"params": model.head.parameters(), "lr": config.lr_head},
    ], weight_decay=config.weight_decay)

    steps = max(config.epochs * (len(train_index) // config.batch_studies), 1)
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimiser, max_lr=[config.lr_backbone, config.lr_head],
        total_steps=steps, pct_start=0.15)
    scaler = torch.amp.GradScaler("cuda", enabled=str(device).startswith("cuda"))

    holdout_y = (y[holdout_index] > 0.5).astype(int)
    best_auc, best_state, best_epoch = -1.0, None, -1
    history: list[EpochResult] = []

    for epoch in range(config.epochs):
        model.train()
        order = rng.permutation(train_index)
        total, n_steps = 0.0, 0

        for start in range(0, len(order) - config.batch_studies + 1, config.batch_studies):
            sel = order[start:start + config.batch_studies]
            rows = torch.as_tensor(np.ascontiguousarray(cache[sel])).to(device)
            # One group per step: which slices a study is seen through varies between
            # steps, which is augmentation along the stack rather than within a slice.
            g = int(torch.randint(config.n_group, (1,)).item())
            imgs = augment(take_group(rows, g, config), config)
            m = torch.as_tensor(mask[sel]).to(device)
            target = torch.as_tensor(y[sel]).to(device)
            weight = torch.as_tensor(w[sel]).to(device)

            with torch.autocast("cuda", enabled=str(device).startswith("cuda")):
                logits = model(imgs, m, img_size)
                loss = (F.binary_cross_entropy_with_logits(
                    logits, target, reduction="none") * weight).mean()

            optimiser.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()
            schedule.step()
            total += loss.item()
            n_steps += 1

        holdout_p = predict(model, cache, mask, holdout_index, config, device, img_size)
        holdout_auc = macro_auc(holdout_y, holdout_p)

        annotation_auc = float("nan")
        if gold_index is not None and len(gold_index):
            gold_p = predict(model, cache, mask, gold_index, config, device, img_size)
            annotation_auc = macro_auc(gold_y, gold_p)

        result = EpochResult(epoch, total / max(n_steps, 1), holdout_auc, annotation_auc)
        history.append(result)
        if on_epoch is not None:
            on_epoch(result)

        # Selection reads the holdout alone; see the module docstring.
        if holdout_auc > best_auc:
            best_auc, best_epoch = holdout_auc, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    return FitResult(best_epoch, best_auc, best_state, history, config)
