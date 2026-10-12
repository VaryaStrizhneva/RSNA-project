"""Reading a trained expert back, over crops it never saw.

The scored Kaggle notebook enters here, which is why this is in the library and not in a
script. Two things are deliberate.

**The config and the regions come off the weights.** A checkpoint carries its
`ExpertConfig` and the `RoiSpec` of every region it was cut over, so nothing here has to
be told what it is reading. Rebuilding either from a module default is the bug that
already happened once on the landmark side: a run was refused outright because the
default named a different point, and had the two configs differed in a field that does
not change the state dict it would have loaded the weights and cut the input the wrong
way instead — the version that says nothing and scores badly.

**The trunk is built with `pretrained=False`.** `ExpertNet` defaults to True, which asks
timm for ImageNet weights over the network. The scored notebook has no network, so the
default turns into a crash at model construction — before any pixel is read, and after
the mount has already cost minutes. Every weight the model needs is in the checkpoint;
the download would be overwritten by `load_state_dict` on the very next line.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..expert.config import ExpertConfig
from ..expert.network import ExpertNet
from ..roi.config import RoiSpec


def load_experts(run: str | Path, device: str = "cpu",
                 log=None) -> tuple[list, ExpertConfig, list[RoiSpec]]:
    """Every fold of one expert run, the config it was fitted under, and its regions.

    The folds must agree with each other. A run assembled from two trainings is not one
    model, and averaging its folds would average two different questions.

    A checkpoint written before regions were stored alongside the weights has only their
    *names*, and they are then looked up in the registry — which is a guess, not a
    record, and it is announced as one. The registry moves: `expert_mcl_fsonly` names
    `mcl` and was cut over two series, while the `mcl` in the registry has carried three
    since. Resolving that run by name would hand a two-series model a three-series
    tensor, load cleanly, and score nonsense. The lookup is therefore the fallback and
    never the path.
    """

    run = Path(run)
    paths = sorted(run.glob("fold*.pt"))
    if not paths:
        raise ValueError(f"no fold weights in {run}")

    config: ExpertConfig | None = None
    specs: list[RoiSpec] | None = None
    models = []
    for path in paths:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        stored = ExpertConfig.from_dict(ck["config"])
        if "specs" in ck:
            cut = [RoiSpec.from_dict(s) for s in ck["specs"]]
        else:
            cut = list(stored.specs)
            if log is not None:
                log(f"  {path.name} predates regions being stored with the weights; "
                    f"resolving {list(stored.rois)} through the registry. Verify the "
                    f"run still reproduces its own holdout before trusting it.")
        if config is None:
            config, specs = stored, cut
        elif stored != config or cut != specs:
            raise ValueError(
                f"{path.name} was fitted under a different config or region than "
                f"{paths[0].name}; this run is two models, not one")
        model = ExpertNet(config, pretrained=False)
        model.load_state_dict(ck["state"])
        models.append(model.eval().to(device))
    return models, config, specs


def mirror_groups(groups: list, specs: list) -> list:
    """The same groups with every sagittal region's depth axis reversed.

    **Why this is a mirror and not a corruption.** `roi.extract.choose_slices` returns
    the kept slices *in stack order*, and a sagittal stack's order is anatomical only up
    to the knee's side: increasing through-plane position runs medial-to-lateral on a
    left knee and lateral-to-medial on a right one. `bowtie_direction` fixes the depth
    window's *extent* -- 8 mm toward the bowtie against 24 toward the notch on
    `lateral_meniscus_d32` -- and nothing fixes its *direction*. Measured on the cached
    crops, the per-slot lit-area profile of the left knees correlates **+0.773 with the
    right knees reversed** and only +0.223 with them as they are. So the axis is a
    nuisance axis, both of its signs are in the training data in roughly equal number,
    and reversing it is an input the encoder has already seen.

    **Why only sagittal.** On a coronal or axial region the depth axis is
    anterior-posterior or cranio-caudal, neither of which is a laterality axis, and
    laterality is in-plane and already canonicalised -- `crop_plane` mirrors the pixels
    so that lateral is on the right. The same profile test gives +0.97 as-is against
    -0.90 reversed on `mcl` and `medial_oa`, which is the control. Reversing those would
    be an input no fold was ever shown.

    **Volume and mask together.** A mask left alone would declare the padding acquired
    and the acquired slices padding, which is a different input rather than a mirrored
    one.

    `extract` centres the acquired run in the slot grid, so where the padding is odd the
    reversal shifts the run by one slot. `ExpertNet.windows` clamps a window's outer
    channels to the ends of the run and pools over windows, so that shift changes
    neither the set of windows nor their pooled order -- measured, it changes the live
    window count on 0 of 4407 studies for `lateral_meniscus_d32`.
    """

    out = []
    for (volumes, mask), spec in zip(groups, specs):
        if spec.plane == "Sagittal":
            out.append((np.ascontiguousarray(np.asarray(volumes)[:, :, ::-1]),
                        np.ascontiguousarray(np.asarray(mask)[:, :, ::-1])))
        else:
            out.append((volumes, mask))
    return out


def inplane_jitter(angle: float, scale: float, dx: float, dy: float):
    """A fixed in-plane affine, as a transform `score_experts` can be handed.

    The same formulation as `rsna.expert.augment.augment_batch`, which is the point: the
    transforms the folds were trained under are the only ones a test-time average may
    draw from, and re-deriving the matrix here would risk averaging over a family the
    model never saw. The photometric half of that augmentation is deliberately not
    reproduced -- brightness, gamma and noise carry no information about the anatomy, so
    averaging over them adds variance to the estimate and nothing to it.

    One affine for the whole batch rather than one per study: a test-time average has to
    be reproducible, and a per-study draw makes the submission depend on batch
    boundaries.
    """

    import torch
    from torch import nn

    def apply(batched: list) -> list:
        out = []
        for x, mask in batched:
            shape = x.shape
            flat = x.reshape(shape[0], -1, *shape[-2:])
            cos = torch.cos(torch.tensor(angle, device=x.device)) / scale
            sin = torch.sin(torch.tensor(angle, device=x.device)) / scale
            theta = torch.zeros(flat.shape[0], 2, 3, device=x.device, dtype=x.dtype)
            theta[:, 0, 0], theta[:, 0, 1], theta[:, 0, 2] = cos, -sin, dx
            theta[:, 1, 0], theta[:, 1, 1], theta[:, 1, 2] = sin, cos, dy
            grid = nn.functional.affine_grid(theta, flat.shape, align_corners=False)
            moved = nn.functional.grid_sample(flat, grid, mode="bilinear",
                                              align_corners=False)
            out.append((moved.reshape(shape), mask))
        return out

    return apply


def score_experts(models: list, groups: list, batch: int = 32,
                  device: str = "cpu", transform=None) -> np.ndarray:
    """One probability per study per target, averaged over the folds.

    `groups` is one `(volumes, mask)` pair per region, in the order the run's specs are
    listed — the same structure the training loop is handed, and assembled the same way,
    because a difference here is a difference the leaderboard would show and no local
    metric would.

    The average is over probabilities rather than logits, matching how the gold scores
    were pooled when the run was fitted. The two differ, and the one that was measured is
    the one to reproduce.

    `transform` rewrites each assembled batch before the model reads it, which is where a
    test-time augmentation has to act: the crops arrive as uint8 arrays and an in-plane
    affine over the whole corpus would neither fit in memory nor run at this speed.
    `None` leaves the input exactly as the training loop handed it over, so the default
    path is unchanged and a run still reproduces its own holdout.
    """

    if not models:
        raise ValueError("no models to score with")
    n = len(groups[0][0])
    totals = None
    for model in models:
        parts = []
        with torch.no_grad():
            for start in range(0, n, batch):
                idx = np.arange(start, min(start + batch, n))
                batched = [
                    (torch.as_tensor(np.asarray(v[idx]).copy(),
                                     device=device).float() / 255.0,
                     torch.as_tensor(m[idx].astype(bool), device=device))
                    for v, m in groups]
                if transform is not None:
                    batched = transform(batched)
                parts.append(torch.sigmoid(model(batched)).float().cpu().numpy())
        p = np.concatenate(parts) if parts else np.zeros((0, 1), np.float32)
        totals = p if totals is None else totals + p
    return totals / len(models)
