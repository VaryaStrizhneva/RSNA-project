"""Cropping a branch's input around a landmark the model found.

The crop is defined in **patient millimetres**, like the landmark it hangs off, so one
predicted point cuts every series of a study whatever its plane or its slice spacing.
That is the whole reason the landmark is stored the way it is.
"""

from . import cache
from .config import SPECS, RoiSpec
from .extract import RoiStack, bowtie_direction, choose_slices, crop_plane, extract

__all__ = ["cache", "SPECS", "RoiSpec", "RoiStack", "bowtie_direction", "choose_slices",
           "crop_plane", "extract"]
