"""Fitting the landmark model, and the number used to judge it.

The number is a **distance in millimetres**, not a loss. A heatmap loss falls smoothly
while the prediction sits in the wrong compartment, and the thing we actually need to
know is whether a 40 mm crop taken around the prediction still contains the meniscus —
which it does when the landmark is within about 12 mm. So every epoch decodes its
validation heatmaps back to patient coordinates and reports the median and the p90 of
the error, and the p90 is what decides.

Two baselines are worth carrying in the head while reading those numbers, both measured
on these 304 annotations. A constant in-plane position gives p90 16.6 mm. A constant
offset from the lateral end of the stack, with the side known perfectly, gives p90
12.6 mm in depth and 21.2 mm in three dimensions. A model that does not beat 21.2 has
learned nothing a median could not have told us.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from . import augment as A
from .cache import restore
from .config import LandmarkConfig
from .target import decode, encode


@dataclass
class EpochResult:
    epoch: int
    loss: float
    median_mm: float
    p90_mm: float
    within_12: float
    #: Per point, in the order of `config.points`. The scalars above are the mean over
    #: them, which for a one-point run is the point itself — so a single-point run reads
    #: exactly as it always did.
    per_point: tuple[tuple[float, float], ...] = ()


@dataclass
class FitResult:
    history: list[EpochResult] = field(default_factory=list)
    best: EpochResult | None = None
    state: dict | None = None
    errors: np.ndarray | None = None       #: per validation study, in millimetres
    studies: list[str] = field(default_factory=list)


def build_targets(records: list[dict], volumes, config: LandmarkConfig) -> np.ndarray:
    """One Gaussian heatmap per study, computed once.

    Augmentation moves the target with the volume rather than re-encoding it, so the
    expensive part happens here and never again — and a transform can never be applied
    to the pixels and forgotten on the label.
    """

    out = np.zeros((len(records), len(config.points), config.slices,
                    config.grid, config.grid), np.float32)
    for i, r in enumerate(records):
        out[i] = encode(r["points"], restore(r, volumes[i]), config)
    return out


def to_input(volume: torch.Tensor) -> torch.Tensor:
    """(batch, slice, h, w) -> (batch, slice, 3, h, w), each slice with its neighbours.

    The three channels are the slice and the two beside it, which is both the shape a
    pretrained RGB encoder expects and a free window of depth context. At the ends of
    the stack the neighbour is padding, which is what a slice at the edge of the
    acquisition actually has.
    """

    prev = torch.roll(volume, 1, dims=1)
    nxt = torch.roll(volume, -1, dims=1)
    prev[:, 0] = volume[:, 0]
    nxt[:, -1] = volume[:, -1]
    return torch.stack([prev, volume, nxt], dim=2)


def augment_batch(volume: torch.Tensor, heat: torch.Tensor, rng: torch.Generator):
    """Everything at once, on the batch, on the device it already lives on."""

    b, dev = volume.shape[0], volume.device
    u = lambda lo, hi: torch.rand(b, device=dev, generator=rng) * (hi - lo) + lo  # noqa: E731

    flip = torch.rand(b, device=dev, generator=rng) < 0.5
    if flip.any():
        v, h = A.reverse(volume[flip], heat[flip])
        volume, heat = volume.clone(), heat.clone()
        volume[flip], heat[flip] = v, h

    # One thinning for the whole batch: it changes the depth axis, and mixing two
    # spacings inside a batch buys nothing a second batch does not.
    if torch.rand(1, device=dev, generator=rng).item() < 0.25:
        volume, heat = A.decimate(volume, heat, int(torch.randint(
            2, 4, (1,), device=dev, generator=rng).item()))

    shift = int(torch.randint(-6, 7, (1,), device=dev, generator=rng).item())
    volume, heat = A.shift_depth(volume, heat, shift)

    volume, heat = A.in_plane(
        volume, heat,
        angle=u(-0.26, 0.26),                                   # +-15 degrees
        shift=torch.stack([u(-0.12, 0.12), u(-0.12, 0.12)], 1),  # +-~11 mm
        scale=u(0.9, 1.1))
    volume = A.intensity(volume, brightness=u(-0.12, 0.12), contrast=u(0.85, 1.15),
                         gamma=u(0.8, 1.25), noise=0.02, generator=rng)
    return volume, heat


def _errors(logits: torch.Tensor, records: list[dict], volumes,
            index: np.ndarray, config: LandmarkConfig) -> np.ndarray:
    """Decode a batch of predictions and measure how far off they are, in millimetres."""

    heat = torch.sigmoid(logits).float().cpu().numpy()
    out = np.full((len(index), len(config.points)), np.nan)
    for j, i in enumerate(index):
        r = records[i]
        got, _ = decode(heat[j], restore(r, volumes[i]), config)
        for k, name in enumerate(config.points):
            truth = r["points"].get(name)
            if truth is not None:
                out[j, k] = float(np.linalg.norm(got[k] - np.asarray(truth, float)))
    return out


def heatmap_loss(pred: torch.Tensor, target: torch.Tensor, weight: float) -> torch.Tensor:
    """Mean squared error, with the target itself as the weight.

    A 3D heatmap is far sparser than the 2D ones heatmap regression is usually written
    for: a 4 mm Gaussian covers about 180 cells of 196 608, which is 0.09 %. Plain MSE
    is then minimised almost perfectly by predicting background everywhere, and that is
    exactly what it does — two epochs of it leave the error at 30 mm, which is where a
    prediction sits when it means nothing.

    Weighting by `1 + weight * target` costs one multiply and makes the peak worth as
    much as the background it sits in. It changes what the loss rewards, not what the
    target says.
    """

    return (((pred - target) ** 2) * (1.0 + weight * target)).mean()


def fit(model, volumes, records: list[dict], targets: np.ndarray,
        train_index: np.ndarray, val_index: np.ndarray, config: LandmarkConfig,
        device="cuda", epochs: int = 60, batch: int = 8, lr: float = 3e-4,
        weight: float = 200.0, seed: int = 0, log=print) -> FitResult:
    """Fit one fold. Returns the best epoch by p90, and that epoch's weights."""

    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=max(1, epochs * int(np.ceil(len(train_index) / batch))))
    rng = torch.Generator(device=device).manual_seed(seed)
    order_rng = np.random.default_rng(seed)
    result = FitResult(studies=[records[i]["study"] for i in val_index])

    def load(idx):
        v = torch.as_tensor(np.asarray(volumes[idx]), device=device).float() / 255.0
        h = torch.as_tensor(targets[idx], device=device)
        return v, h

    for epoch in range(1, epochs + 1):
        model.train()
        total, seen = 0.0, 0
        shuffled = train_index[order_rng.permutation(len(train_index))]
        for start in range(0, len(shuffled), batch):
            # Sorted within the batch only: the memmap is read in order, while which
            # studies land together still changes every epoch.
            idx = np.sort(shuffled[start:start + batch])
            v, h = load(idx)
            v, h = augment_batch(v, h, rng)
            pred = torch.sigmoid(model(to_input(v)))
            loss = heatmap_loss(pred, h, weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            total += float(loss.detach()) * len(idx)
            seen += len(idx)

        model.eval()
        errs = []
        with torch.no_grad():
            for start in range(0, len(val_index), batch):
                idx = val_index[start:start + batch]
                v, _ = load(idx)
                errs.append(_errors(model(to_input(v)), records, volumes, idx, config))
        # Every point, not just the first. Taking column 0 reported a two-point model's
        # accuracy as its first point's and chose the epoch on that alone, leaving the
        # second one unmeasured and unoptimised — and the number looked perfectly
        # healthy either way.
        all_e = np.concatenate(errs)
        per = tuple((float(np.nanmedian(all_e[:, k])),
                     float(np.nanpercentile(all_e[:, k], 90)))
                    for k in range(all_e.shape[1]))
        e = all_e
        r = EpochResult(epoch, total / max(seen, 1),
                        float(np.mean([m for m, _ in per])),
                        float(np.mean([q for _, q in per])),
                        float(np.nanmean(e < 12)), per)
        result.history.append(r)
        if result.best is None or r.p90_mm < result.best.p90_mm:
            result.best = r
            result.state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            result.errors = e
        detail = ("  [" + " ".join(f"{q:.1f}" for _, q in r.per_point) + "]"
                  if len(r.per_point) > 1 else "")
        log(f"  epoch {epoch:3d}  loss {r.loss:.5f}  median {r.median_mm:5.1f} mm  "
            f"p90 {r.p90_mm:5.1f} mm{detail}  within 12 mm {100 * r.within_12:3.0f} %")
    return result


def folds(n: int, k: int = 5, seed: int = 0) -> list[np.ndarray]:
    """A plain K-fold over studies: each appears once, so nothing subtler is needed."""

    order = np.random.default_rng(seed).permutation(n)
    return [order[i::k] for i in range(k)]
