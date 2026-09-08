"""A trained model, packaged so that inference can refuse to guess.

Training happens on our own machine; scoring happens in a Kaggle notebook with no
internet. The only thing that crosses is a directory of files, so that directory has to
carry everything needed to read the weights *the way they were fitted* — not just the
weights.

Three things travel with each member:

* the **state dict**, obviously;
* the **`Config`** it was fitted under, because two configs that agree on array shapes
  can disagree on which pixels those arrays hold, and nothing downstream can tell;
* a **fingerprint**, the model's output on a fixed synthetic bag, which catches the
  case where the weights load and are then read through the wrong preprocessing —
  that produces predictions, not errors.

`load_package` refuses rather than defaults. A package that cannot be reproduced
exactly should stop the run: a wrong-but-plausible submission is worse than none,
because it scores.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from ..config import FINGERPRINT_TOL, Config
from .fingerprint import WeightsError, fingerprint
from .network import build_model

MANIFEST = "manifest.json"


def _finite(value):
    """None for a score that could not be measured, rather than a NaN."""

    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None
FORMAT_VERSION = 1


@dataclass
class Member:
    """One trained model inside a package."""

    name: str
    config: Config
    state_dict: dict
    fingerprint: np.ndarray
    fold: int | None = None
    holdout_auc: float | None = None
    epoch: int | None = None

    @property
    def file(self) -> str:
        return f"{self.name}.pt"


def write_package(path: str | Path, members: list[Member], note: str = "") -> Path:
    """Write a package: one .pt per member, plus a manifest describing all of them."""

    if not members:
        raise ValueError("a package with no members is not a package")

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    entries = []
    for member in members:
        torch.save({"state_dict": member.state_dict,
                    "fingerprint": np.asarray(member.fingerprint, np.float32),
                    "config": member.config.to_dict()},
                   path / member.file)
        entries.append({
            "id": member.name,
            "file": member.file,
            "config": member.config.to_dict(),
            "fold": member.fold,
            # NaN is not JSON: json.dumps emits a bare `NaN` that no other parser will
            # read back. An unmeasurable score is null, which is what it means.
            "holdout": _finite(member.holdout_auc),
            "epoch": member.epoch,
        })

    (path / MANIFEST).write_text(json.dumps({
        "format": FORMAT_VERSION,
        "written": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "note": note,
        "members": entries,
    }, indent=1) + "\n", encoding="utf-8")
    return path


def find_package(root: str | Path = "/kaggle/input", name: str = MANIFEST) -> Path | None:
    """Locate a mounted package by its manifest, or return None.

    Searched by content rather than by an expected path, so the notebook keeps working
    whatever the mount is called. A manifest whose files are missing raises instead of
    being skipped: an incomplete package that is silently ignored looks exactly like no
    package at all, and the run would fall back to something else without saying so.
    """

    root = Path(root)
    if not root.is_dir():
        return None

    import os

    for current, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in ("train_series", "test_series")]
        if name not in files:
            continue
        try:
            manifest = json.loads((Path(current) / name).read_text())
        except (OSError, ValueError):
            continue
        members = manifest.get("members")
        if not isinstance(members, list) or not members:
            continue
        missing = [m["file"] for m in members if not (Path(current) / m["file"]).is_file()]
        if missing:
            raise WeightsError(
                f"{current} holds a manifest listing {len(members)} members but "
                f"{len(missing)} of their files are absent (first {missing[0]!r})")
        return Path(current)
    return None


def read_manifest(path: str | Path) -> dict:
    return json.loads((Path(path) / MANIFEST).read_text(encoding="utf-8"))


def load_member(path: str | Path, entry: dict, device="cpu", source=None,
                tol: float = FINGERPRINT_TOL):
    """Rebuild one member and verify it computes what it computed when fitted."""

    path = Path(path)
    payload = torch.load(path / entry["file"], map_location="cpu", weights_only=False)
    config = Config.from_dict(payload["config"])

    if entry.get("config") and entry["config"] != payload["config"]:
        raise WeightsError(
            f"{entry['id']}: the manifest and the checkpoint disagree about the config")

    model = build_model(config, source=source).to(device)
    missing, unexpected = model.load_state_dict(payload["state_dict"], strict=False)
    if missing or unexpected:
        raise WeightsError(
            f"{entry['id']}: state dict does not match the architecture "
            f"({len(missing)} missing, {len(unexpected)} unexpected; "
            f"first missing {missing[:1]}, first unexpected {unexpected[:1]})")

    got = fingerprint(model, config, device)
    expected = np.asarray(payload["fingerprint"], np.float32)
    if got.shape != expected.shape:
        raise WeightsError(
            f"{entry['id']}: fingerprint shape {got.shape} != stored {expected.shape}")
    distance = float(np.abs(got - expected).max())
    if distance > tol:
        raise WeightsError(
            f"{entry['id']}: fingerprint differs by {distance:.4g} (tolerance {tol:g}). "
            f"The weights load but do not compute what they computed when fitted.")

    return model, config, distance
