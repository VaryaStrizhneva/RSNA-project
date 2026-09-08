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
from .augment import augment, take_window


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
            img_size: int | None = None, overlap: bool = False) -> np.ndarray:
    """Average the logits over the windows of each slot.

    `overlap=False` by default, deliberately: this runs once per epoch to choose one,
    and the disjoint windows cost four encoder passes where the sliding ones cost ten.
    Final inference uses `overlap=True` — see `rsna.infer.predict_member`. The two
    therefore measure slightly different things, which is fine for ranking epochs
    against each other and would not be fine for comparing to a leaderboard score.
    """

    windows = config.windows(overlap=overlap)
    model.eval()
    out = []
    for start in range(0, len(index), config.eval_batch):
        sel = index[start:start + config.eval_batch]
        m = torch.as_tensor(mask[sel]).to(device)
        acc = None
        for begin in windows:
            # Gathered a window at a time rather than whole and then sliced. The two are
            # the same pixels, but taking a whole study out of the cache allocates every
            # slice it holds — most of which this pass will not look at until a later
            # window, by which time they have been fetched again.
            rows = torch.as_tensor(
                np.ascontiguousarray(cache[sel, :, begin:begin + config.group])).to(device)
            with torch.autocast("cuda", enabled=str(device).startswith("cuda")):
                z = model(rows, m, img_size).float()
            acc = z if acc is None else acc + z
        out.append(torch.sigmoid(acc / len(windows)).cpu().numpy())
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

    train_windows = config.windows(overlap=False)
    holdout_y = (y[holdout_index] > 0.5).astype(int)
    best_auc, best_state, best_epoch = -1.0, None, -1
    last_state = None
    history: list[EpochResult] = []

    for epoch in range(config.epochs):
        model.train()
        order = rng.permutation(train_index)
        total, n_steps = 0.0, 0

        for start in range(0, len(order) - config.batch_studies + 1, config.batch_studies):
            sel = order[start:start + config.batch_studies]
            rows = torch.as_tensor(np.ascontiguousarray(cache[sel])).to(device)
            # One window per step: which slices a study is seen through varies between
            # steps, which is augmentation along the stack rather than within a slice.
            start = int(rng.choice(train_windows))
            imgs = augment(take_window(rows, start, config), config)
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

        # Kept every epoch, because selection can fail to pick any: a holdout where a
        # target has one class present scores NaN, every comparison against NaN is
        # False, and the run would otherwise finish having saved nothing at all.
        last_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        # Selection reads the holdout alone; see the module docstring.
        if np.isfinite(holdout_auc) and holdout_auc > best_auc:
            best_auc, best_epoch = holdout_auc, epoch
            best_state = last_state

    if best_state is None:
        # Nothing was selectable. Return the final epoch and say so with a NaN score,
        # rather than a state that a caller would read as "the best one".
        return FitResult(len(history) - 1, float("nan"), last_state, history, config)

    return FitResult(best_epoch, best_auc, best_state, history, config)
