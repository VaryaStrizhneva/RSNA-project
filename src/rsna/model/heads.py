"""Per-diagnosis attention over the slot embeddings of one study."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..config import SLOT_PRIOR_STRENGTH, SLOT_PRIOR_TABLE, TARGETS, Config


class SlotHead(nn.Module):
    """One attention distribution over slots, per diagnosis.

    Each finding is read on particular sequences — cruciates sagittally, collateral
    ligaments and the meniscal body coronally, patellar cartilage axially — so pooling
    the slots identically would dilute the one carrying the evidence with the rest.

    The aggregation is deliberately this simple. With a study-level label there is no
    signal saying *which part* of a study matters, so attention parameters below the
    slot level would have nothing to learn from and would spend their capacity fitting
    noise.
    """

    def __init__(self, dim: int, n_slot: int, n_out: int, hidden: int = 256,
                 dropout: float = 0.2, prior: bool = False,
                 config: Config | None = None):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU())
        self.slot_emb = nn.Parameter(torch.randn(n_slot, hidden) * 0.02)
        self.query = nn.Parameter(torch.randn(n_out, hidden) * 0.02)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden, n_out)
        self.hidden = hidden
        self.prior = prior

        # A fixed per-(diagnosis, slot) tilt, set from anatomy rather than learned. It
        # is a buffer, so it travels in the state dict and must exist for a member
        # fitted with it to load at all.
        if prior:
            table = torch.zeros(n_out, n_slot)
            slot_names = config.slot_names if config is not None else None
            if slot_names is not None and n_slot == len(slot_names) and n_out == len(TARGETS):
                for target, slots in SLOT_PRIOR_TABLE.items():
                    if target in TARGETS:
                        table[TARGETS.index(target), list(slots)] = SLOT_PRIOR_STRENGTH
            self.register_buffer("slot_prior", table)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """x: (batch, slot, dim). mask: (batch, slot), 1 where the slot is present."""

        h = self.proj(x) + self.slot_emb
        att = torch.einsum("bsh,oh->bos", h, self.query) / self.hidden ** 0.5
        if self.prior:
            att = att + self.slot_prior.unsqueeze(0)
        # An absent slot is excluded from the softmax rather than fed zeros: zeros are
        # a picture of something, absence is not.
        att = att.masked_fill(mask.unsqueeze(1) < 0.5, -1e4).softmax(-1)
        ctx = self.drop(torch.einsum("bos,bsh->boh", att, h))
        return (ctx * self.out.weight.unsqueeze(0)).sum(-1) + self.out.bias
