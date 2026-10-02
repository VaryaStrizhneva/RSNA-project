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


def score_experts(models: list, groups: list, batch: int = 32,
                  device: str = "cpu") -> np.ndarray:
    """One probability per study per target, averaged over the folds.

    `groups` is one `(volumes, mask)` pair per region, in the order the run's specs are
    listed — the same structure the training loop is handed, and assembled the same way,
    because a difference here is a difference the leaderboard would show and no local
    metric would.

    The average is over probabilities rather than logits, matching how the gold scores
    were pooled when the run was fitted. The two differ, and the one that was measured is
    the one to reproduce.
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
                parts.append(torch.sigmoid(model(batched)).float().cpu().numpy())
        p = np.concatenate(parts) if parts else np.zeros((0, 1), np.float32)
        totals = p if totals is None else totals + p
    return totals / len(models)
