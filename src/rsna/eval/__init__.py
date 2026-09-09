"""Reading a finished run.

The third step of the cycle, beside `rsna.train` and `rsna.infer`, and deliberately
separate from both: training writes a record, evaluation reads it. Nothing here loads a
checkpoint or touches the pixel cache, so a report can be regenerated on any machine
that has the few kilobytes of text a run leaves behind.
"""

from .metrics import (auc_interval, bootstrap_macro, flags, macro_auc,
                      per_target_auc, pool, rank_normalise, target_table)
from .record import RunRecord, read_run_record, read_sweep, write_run_record
from .report import build, render_html, summary_text, write

__all__ = [
    "RunRecord", "write_run_record", "read_run_record", "read_sweep",
    "per_target_auc", "macro_auc", "rank_normalise", "pool", "bootstrap_macro",
    "target_table", "auc_interval", "flags",
    "build", "summary_text", "render_html", "write",
]
