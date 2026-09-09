"""One file per experiment, defining the whole thing.

An experiment is data, not a code path. Everything that decides what a run *is* lives
in its JSON — the model, the preprocessing, the labels it trains on, which fold is held
out — so `--experiment <name>` and nothing else is enough to reproduce it.

Two sections, and the split matters:

``config``
    Becomes a `rsna.config.Config`, and is written verbatim into the manifest of every
    weights package the run produces. It says what the model *is*. Because a package
    records it, `rsna.model.load_member` rebuilds the right architecture without being
    told, and comparing a package to an experiment is a dict comparison.

``run``
    Orchestration: which split, which label table, which fold, which encoder
    directory. Not part of what the model is, so it stays out of the manifest.

Only machine-specific paths are left on the command line — `--data-root`, `--cache`,
`--out` — because they differ between a laptop and the training box and have no
business being versioned.

Each section holds only the **delta** from the defaults; the rest is filled in. A field
that no longer exists makes `Config.from_dict` refuse loudly, which is what should
happen to a stale experiment.

    python -m scripts.train --experiment depth_compress

The reasoning behind each one is in README.md, next to them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from src.rsna.config import Config

HERE = Path(__file__).parent

#: Defaults for the `run` section, so an experiment need only name what it changes.
RUN_DEFAULTS = {
    "split": "train_series",
    "labels": "data/external/pilkwang-rsna-knee-llm-labels/report_labels_v2.csv",
    "encoder": "models/dinov2-small",
    "fold": 0,
}


@dataclass(frozen=True)
class Experiment:
    """A named run: what the model is, and what it is fitted on."""

    name: str
    config: Config
    run: dict = field(default_factory=dict)

    def __getattr__(self, key: str):
        # `experiment.split` rather than `experiment.run["split"]`; the run section is
        # a small fixed set of keys, and reading it should not look like a dict lookup.
        try:
            return self.run[key]
        except KeyError as exc:
            raise AttributeError(key) from exc


def available() -> list[str]:
    """Every experiment defined in this directory."""

    return sorted(p.stem for p in HERE.glob("*.json"))


def load(name: str) -> Experiment:
    """Read one experiment, filling in every default it does not name."""

    path = HERE / f"{name}.json"
    if not path.is_file():
        raise SystemExit(
            f"unknown experiment {name!r}; choose from {', '.join(available())}")

    raw = json.loads(path.read_text(encoding="utf-8"))
    unknown = set(raw) - {"config", "run"}
    if unknown:
        raise SystemExit(f"{path.name} has unexpected sections: {sorted(unknown)}")

    return Experiment(name=name,
                      config=Config(**raw.get("config", {})),
                      run={**RUN_DEFAULTS, **raw.get("run", {})})
