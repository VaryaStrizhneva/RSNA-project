"""A model for one pathology, reading one region of interest.

Separate from the twelve-target classifier on purpose: it answers one question — does a
tight, high-resolution crop see the meniscus better than the wide view — and answering it
needs the comorbidity reasoning out of the way, not folded in.
"""

from .config import ExpertConfig
from .network import Attention, ExpertNet

__all__ = ["ExpertConfig", "ExpertNet", "Attention"]
