"""DICOM headers and geometry: the path from files on disk to comparable images.

Imported by both `train` and `infer`. That is the point — the original notebook keeps
training and inference in one file precisely so the pixel path cannot diverge between
them, and a package only keeps that guarantee if both sides import the same module.

Ported from pilkwang's public baseline (see docs/references.md). The reasoning in the
docstrings is theirs and is kept, because most of it records measurements over this
corpus that are not recoverable by reading the code.
"""

from .cache import CacheMismatch, available_gb, build_cache, load_cache, plan_cache
from .headers import HDR_TAGS, annotate, hdr_vec, probe, walk
from .laterality import laterality_of, normalise_laterality, side_from_corner_x, side_from_geometry
from .ordering import order_slices
from .pixels import read_slot, sample_indices
from .slots import pick_slots

__all__ = [
    "CacheMismatch", "available_gb", "build_cache", "load_cache", "plan_cache",
    "HDR_TAGS", "annotate", "hdr_vec", "probe", "walk",
    "laterality_of", "normalise_laterality", "side_from_corner_x", "side_from_geometry",
    "order_slices", "pick_slots", "read_slot", "sample_indices",
]
