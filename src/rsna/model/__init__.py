"""Encoder plus per-diagnosis slot attention.

Ported from pilkwang's public baseline; both top-scoring public notebooks carry these
five definitions byte-identical, which is as close to a consensus architecture as this
competition has. See docs/references.md.

The port changes one thing: the original reads `N_SLOT`, `SLOTS`, `TARGETS` and
`POOL_PARTS` from module globals, so a model silently depends on whatever the notebook
last set. Here they come from a `Config`, which is what lets a weights package state
the architecture it needs.
"""

from .fingerprint import check_fingerprint, fingerprint
from .heads import SlotHead
from .network import Model, build_model, find_encoder

__all__ = ["Model", "SlotHead", "build_model", "find_encoder", "fingerprint", "check_fingerprint"]
