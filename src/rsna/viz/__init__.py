"""Looking at the pipeline: what it decided, and what the pixels became.

Nothing here decides anything. It is imported by `notebooks/preprocessing.ipynb`, which
holds no logic of its own precisely so that the notebook cannot describe a preprocessing
nobody runs.

    steps    the pixel path replayed, with every intermediate kept
    figures  those intermediates drawn
    reports  the same stages as tables
"""

from .figures import (figure_cache, figure_channels, figure_stack, figure_steps,
                      figure_windows)
from .reports import (plane_map_from, report_headers, report_laterality,
                      report_ordering, report_pixels, report_slots, selection_of)
from .steps import load_stack, preprocessing_steps, verify_against_read_slot

__all__ = [
    "load_stack", "preprocessing_steps", "verify_against_read_slot",
    "figure_stack", "figure_steps", "figure_channels", "figure_cache", "figure_windows",
    "plane_map_from", "report_headers", "report_laterality", "report_slots",
    "report_ordering", "report_pixels", "selection_of",
]
