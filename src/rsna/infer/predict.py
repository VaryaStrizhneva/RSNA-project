"""Reading a package over a decoded cache."""

from __future__ import annotations

import numpy as np
import torch

from ..config import TARGETS, Config


@torch.no_grad()
def predict_member(model, cache, mask, config: Config, device="cpu",
                   starts: list[int] | None = None, batch: int | None = None,
                   img_size: int | None = None) -> np.ndarray:
    """One member's probabilities, averaged over its windows.

    `starts` is a parameter rather than always derived here so a caller that has
    measured what the remaining time affords can read a member over fewer windows —
    the reduction then lives in one place instead of being re-derived.
    """

    img_size = config.img if img_size is None else img_size
    batch = config.eval_batch if batch is None else batch
    starts = config.windows(overlap=True) if starts is None else list(starts)
    if not starts:
        raise ValueError("predict_member was given no windows to average over")

    model.eval()
    out = []
    for begin in range(0, len(cache), batch):
        sel = slice(begin, begin + batch)
        m = torch.as_tensor(np.ascontiguousarray(mask[sel])).to(device)
        total = None
        for start in starts:
            # Gathered one window at a time rather than whole and then sliced: taking a
            # whole study out of the cache allocates every slice it holds, most of which
            # this pass will not look at until a later window.
            rows = torch.as_tensor(
                np.ascontiguousarray(
                    cache[sel, :, start:start + config.window_size])).to(device)
            with torch.autocast("cuda", enabled=str(device).startswith("cuda")):
                logits = model(rows, m, img_size).float()
            total = logits if total is None else total + logits
        out.append(torch.sigmoid(total / len(starts)).cpu().numpy())

    return np.concatenate(out) if out else np.zeros((0, len(TARGETS)), np.float32)


def blend_members(predictions: list[np.ndarray], weights: list[float] | None = None) -> np.ndarray:
    """Combine members in rank space.

    Ranking each member first is what makes a weight mean the same thing for every
    member, whatever scale it emits — and the metric reads order only anyway.
    """

    import pandas as pd

    if not predictions:
        raise ValueError("nothing to blend")
    weights = [1.0] * len(predictions) if weights is None else list(weights)
    if len(weights) != len(predictions):
        raise ValueError(f"{len(predictions)} members but {len(weights)} weights")

    total = sum(weights)
    ranked = [pd.DataFrame(p).rank(method="average", pct=True).to_numpy()
              for p in predictions]
    return sum(w / total * r for w, r in zip(weights, ranked))
