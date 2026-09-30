"""Finding the region of interest, so a branch can be cropped around it.

A landmark is stored and predicted in **patient millimetres**, never in pixels. Every
series of a study shares the patient frame, so one point indexes into all of them
whatever their plane — which is why one annotation pass, on one sequence, serves every
branch of the classifier. It is also what makes an annotation outlive the conventions
that produced it: the series shown can change, the laterality rule can change, and the
point still means the same place in the knee.
"""

from . import cache
from .config import LandmarkConfig
from .series import pick_sagittal
from .sample import Sampled, choose_depth, place, resample_plane, sample_series
from .target import decode, encode, project, unproject

__all__ = ["cache", "LandmarkConfig", "Sampled", "choose_depth", "place", "resample_plane",
           "sample_series", "pick_sagittal", "decode", "encode", "project", "unproject"]
