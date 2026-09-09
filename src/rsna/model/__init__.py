"""Encoder plus per-diagnosis slot attention.

Ported from pilkwang's public baseline; both top-scoring public notebooks carry these
five definitions byte-identical, which is as close to a consensus architecture as this
competition has. See docs/references.md.

The port changes one thing: the original reads `N_SLOT`, `SLOTS`, `TARGETS` and
`POOL_PARTS` from module globals, so a model silently depends on whatever the notebook
last set. Here they come from a `Config`, which is what lets a weights package state
the architecture it needs.
"""

from .fingerprint import WeightsError, check_fingerprint, fingerprint
from .heads import SlotHead
from .network import Model, build_model, find_encoder
from .stems import DepthCompress, build_stem
from .package import (Member, find_package, load_member, read_manifest,
                      write_package)

__all__ = [
    "Model", "SlotHead", "build_model", "find_encoder", "DepthCompress", "build_stem",
    "fingerprint", "check_fingerprint", "WeightsError",
    "Member", "write_package", "find_package", "read_manifest", "load_member",
]
