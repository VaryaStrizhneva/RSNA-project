"""One header read per series, and what can be recovered from it.

The competition ships `Fluid_Sensitive` and `Fat_Suppression` per series, but they are
equal on all 24,371 training rows, so as delivered they carry one axis rather than
two. `annotate` recovers both from the header — series description, scan options, TR
and TE — which is what makes the six-slot scheme in `config.SLOTS_RECOVERED` possible.
"""

from __future__ import annotations

import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom

#: Read once per series, from a single slice. Position and orientation cost nothing
#: extra here and are what recovers the side when `Laterality` is absent — which it is
#: for half the studies in this corpus.
HDR_TAGS = [
    "SeriesDescription", "SequenceName", "ScanOptions", "ScanningSequence",
    "RepetitionTime", "EchoTime", "Laterality", "PixelSpacing", "Rows",
    "Columns", "RescaleSlope", "RescaleIntercept",
    "ImagePositionPatient", "ImageOrientationPatient",
]

#: GE writes SAT_GEMS for spatial saturation, so scan options are matched as exact
#: tokens; a substring test on "SAT" fires on series that are not fat-suppressed.
FATSAT_OPTS = {"FS", "FATSAT", "FAT_SAT", "FSAT"}

_SEP = re.compile(r"[_\-.]")
_FATSAT_RX = re.compile(r"\bfs\b|fatsat|fat sat|\bstir\b|\bspair\b|\bspir\b|\bwe\b|"
                        r"water excit|\btirm\b|\bsting\b|\bfatsup\b")
_T1_RX = re.compile(r"\bt1\b|\bt1w\b")
_T2_RX = re.compile(r"\bt2\b|\bt2w\b")
_PD_RX = re.compile(r"\bpd\b|\bpdw\b|proton|\bdp\b|dens")

_EMPTY_COLUMNS = ["split", "StudyInstanceUID", "SeriesInstanceUID", "dir", "files",
                  "n_slices"] + HDR_TAGS


def hdr_vec(value: object, n: int) -> np.ndarray | None:
    """Parse a DICOM multi-value string as `probe` stores it: floats joined by `|`."""

    if not isinstance(value, str):
        return None
    try:
        parts = [float(x) for x in value.split("|")]
    except ValueError:
        return None
    return np.array(parts) if len(parts) >= n else None


def probe(item: tuple[str, str, str, str]) -> dict:
    """Read one series: its file list and the header of its middle slice."""

    split, study, series, path = item
    row = {"split": split, "StudyInstanceUID": study, "SeriesInstanceUID": series,
           "dir": path}
    try:
        files = sorted(e.name for e in os.scandir(path) if e.name.endswith(".dcm"))
        row["files"] = files
        row["n_slices"] = len(files)
        if not files:
            return row
        ds = pydicom.dcmread(os.path.join(path, files[len(files) // 2]),
                             stop_before_pixels=True, force=True)
        for tag in HDR_TAGS:
            value = getattr(ds, tag, None)
            if value is None:
                row[tag] = None
            elif isinstance(value, (list, tuple)) or type(value).__name__ == "MultiValue":
                row[tag] = "|".join(str(x) for x in value)
            else:
                row[tag] = str(value)
    except Exception as exc:  # a series that will not open is reported, not dropped
        row["err"] = str(exc)[:120]
    return row


def walk(root: Path, split: str, threads: int = 16) -> pd.DataFrame:
    """Every series directory of a split, with one header read per series.

    An absent split returns an empty frame **with the columns `annotate` expects**.
    Returning a bare DataFrame looks like the same thing and is not: the next call
    indexes `SeriesDescription` and raises `KeyError`, so the branch that exists to
    survive a missing split is what turns it into a crash.
    """

    base = Path(root) / split
    if not base.is_dir():
        return pd.DataFrame(columns=_EMPTY_COLUMNS)

    items = []
    for study in os.scandir(base):
        if study.is_dir():
            for series in os.scandir(study.path):
                if series.is_dir():
                    items.append((split, study.name, series.name, series.path))

    with ThreadPoolExecutor(max_workers=threads) as pool:
        rows = list(pool.map(probe, items))
    return pd.DataFrame(rows)


def annotate(df: pd.DataFrame) -> pd.DataFrame:
    """Recover fat suppression and pulse-sequence weighting from the header.

    Adds `fatsat` (bool), `weight` (T1/T2/PD/GRE/UNK), `fluid` (bool) and `px`
    (in-plane spacing, mm). Text evidence wins where it exists; TR and TE decide the
    rest.
    """

    df = df.copy()
    desc = (df["SeriesDescription"].fillna("") + " " + df["SequenceName"].fillna(""))
    desc = desc.str.lower().str.replace(_SEP, " ", regex=True)

    options = df["ScanOptions"].fillna("").str.upper().str.split("|")
    options_fs = options.apply(lambda ts: any(t.strip() in FATSAT_OPTS for t in ts))
    df["fatsat"] = desc.str.contains(_FATSAT_RX) | options_fs

    tr = pd.to_numeric(df["RepetitionTime"], errors="coerce")
    te = pd.to_numeric(df["EchoTime"], errors="coerce")
    gre = df["ScanningSequence"].fillna("").str.upper().str.contains("GR")
    t1 = desc.str.contains(_T1_RX)
    t2 = desc.str.contains(_T2_RX)
    pdw = desc.str.contains(_PD_RX)

    df["weight"] = np.where(
        t1 & ~t2 & ~pdw, "T1",
        np.where(t2 & ~pdw, "T2",
        np.where(pdw, "PD",
        np.where(gre, "GRE",
        np.where(tr < 800, "T1",
        np.where(te > 60, "T2",
        np.where(tr >= 800, "PD", "UNK")))))))

    df["fluid"] = np.isin(df["weight"], ["PD", "T2"])
    df["px"] = pd.to_numeric(
        df["PixelSpacing"].fillna("").str.split("|").str[0].replace("", np.nan),
        errors="coerce")
    return df
