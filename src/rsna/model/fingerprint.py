"""A model's answer to a question with no data in it.

Weights that load but are read through the wrong preprocessing produce predictions,
not errors. The submission is well formed, the log says nothing, and the difference is
a number no output of the run reveals. Scaling that never happens, or happens twice, is
enough on its own and changes no shape anywhere.

So a set of weights carries the output it gave on a synthetic bag generated from a
seed — identical on any machine — pushed through the whole forward path: byte scaling,
ImageNet normalisation, resize, encoder, slot attention. Any of those differing moves
the output by order one.

This checks the model computes what it computed when fitted. It cannot check that the
pixels reaching it are the right pixels; the header pass and the pixel reader answer
to their own tests.
"""

from __future__ import annotations

import numpy as np
import torch

from ..config import FINGERPRINT_TOL, Config


class WeightsError(RuntimeError):
    """Raised when a weights package is attached but cannot be trusted."""


def fingerprint(model, config: Config, device: torch.device | str = "cpu",
                img_size: int | None = None) -> np.ndarray:
    """The model's output on a fixed synthetic bag."""

    img_size = config.img if img_size is None else img_size
    generator = torch.Generator().manual_seed(config.seed)
    imgs = torch.randint(0, 256,
                         (2, config.n_slot, config.window_size, img_size, img_size),
                         generator=generator, dtype=torch.uint8).to(device)
    mask = torch.ones(2, config.n_slot, device=device)
    mask[1, -1] = 0.0  # exercise the masked branch of the softmax

    was_training = model.training
    model.eval()
    with torch.no_grad():
        # float32 throughout: autocast would make the value depend on which device
        # happened to run it, and the point of the number is that it does not.
        out = model(imgs, mask, img_size).float().cpu().numpy()
    if was_training:
        model.train()
    return out


def check_fingerprint(model, config: Config, expected, device="cpu",
                      img_size: int | None = None, tol: float = FINGERPRINT_TOL,
                      tag: str = "") -> float:
    """Compare against a stored fingerprint; raise when the model is not the same map."""

    got = fingerprint(model, config, device, img_size)
    exp = np.asarray(expected, np.float32)

    if got.shape != exp.shape:
        raise WeightsError(
            f"{tag}fingerprint shape {got.shape} != stored {exp.shape}: the "
            f"architecture is not the one these weights were fitted to")

    distance = float(np.abs(got - exp).max())
    if distance > tol:
        raise WeightsError(
            f"{tag}fingerprint differs by {distance:.4g} (tolerance {tol:g}). The "
            f"weights load but do not compute what they computed when fitted — "
            f"preprocessing, resolution or architecture has moved between the two runs.")
    return distance
