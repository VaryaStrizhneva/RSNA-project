"""Turning a trained package into a submission.

The scored run is inference only: weights are fitted elsewhere and mounted here. See
README.md in this directory for what the public notebooks do to fit inside nine hours,
and where each trick comes from.

Which slices reach the encoder is decided by `Config.windows`, not here — it is a
property of the configuration a member was fitted under, so training and inference
cannot disagree about it.
"""

from .predict import blend_members, predict_member
from .submission import benchmark_submission, write_submission

__all__ = ["predict_member", "blend_members", "benchmark_submission", "write_submission"]
