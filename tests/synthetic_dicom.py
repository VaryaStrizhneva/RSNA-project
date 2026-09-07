"""Write real .dcm files with known geometry.

The pixel path cannot be exercised on arrays alone: it opens files, reads rescale
tags, and reconstructs slice order from geometry. These are genuine DICOM files —
written and read back by pydicom — with the two properties the tests need: the file
*names* are deliberately shuffled relative to physical position, and the intensity of
each slice encodes its true index, so a wrong order is visible in the pixels.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian


def write_series(directory: Path, n_slices: int = 12, size: int = 64,
                 spacing: float = 0.5, orientation=(1, 0, 0, 0, 1, 0),
                 first_x: float = 40.0, step: float = 3.0,
                 corrupt: set[int] | None = None) -> Path:
    """One series whose k-th slice sits at a known position and is filled with k.

    Slice k is written at position `first_x + k * step` along the normal, and its
    pixels hold `k * 100` plus an in-plane gradient. Read the stack back in the right
    order and the per-slice means increase; read it in file order and they do not.

    The gradient matters: `read_slot` normalises by the 1st-99th percentile of the
    whole stack, so slices of a single constant value would legitimately normalise to
    zero and hide whether anything was read at all.
    """
    ramp = np.linspace(0, 60, size, dtype=np.float32)
    gradient = np.tile(ramp, (size, 1))

    directory.mkdir(parents=True, exist_ok=True)
    corrupt = corrupt or set()

    for k in range(n_slices):
        ds = Dataset()
        ds.file_meta = FileMetaDataset()
        ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds.file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.4"
        ds.file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()

        ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.4"
        ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID
        ds.Modality = "MR"
        ds.SeriesDescription = "SAG PD FS"
        ds.InstanceNumber = k + 1
        ds.ImageOrientationPatient = list(orientation)
        # The normal of (1,0,0)x(0,1,0) is +z, so stepping z steps along the stack.
        ds.ImagePositionPatient = [first_x, 0.0, float(k) * step]
        ds.PixelSpacing = [spacing, spacing]
        ds.Rows, ds.Columns = size, size
        ds.BitsAllocated, ds.BitsStored, ds.HighBit = 16, 16, 15
        ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
        ds.PixelRepresentation = 0
        ds.RescaleSlope, ds.RescaleIntercept = 1, 0
        ds.PixelData = (gradient + k * 100).astype(np.uint16).tobytes()

        # A SOP Instance UID for a name: unique, and uncorrelated with position.
        path = directory / f"{uuid.uuid4().hex}.dcm"
        if k in corrupt:
            path.write_bytes(b"not a dicom file")
        else:
            ds.save_as(path, enforce_file_format=True)

    return directory


def series_record(directory: Path, spacing: float = 0.5) -> dict:
    """The shape `read_slot` expects, with files in arbitrary (file-name) order."""

    files = sorted(p.name for p in Path(directory).iterdir() if p.suffix == ".dcm")
    return {"dir": str(directory), "files": files, "px": spacing,
            "SeriesInstanceUID": Path(directory).name}
