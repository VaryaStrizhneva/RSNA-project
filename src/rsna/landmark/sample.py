"""One study's sagittal stack, as the landmark model receives it.

Two axes, two different rules, and the asymmetry is deliberate.

**In plane, resample.** The corpus runs 0.17 to 0.94 mm/px and a convolution is
invariant to translation, not to scale — an unresampled knee would be 300 pixels wide on
one study and 900 on another, and with ~300 annotations the model would have to learn
each scale separately. The field of view is the annotation bundle's 180 mm, so what the
model sees is what the annotator looked at. 306 of 320 studies are *reduced*, by 2.2x in
the median and up to 5.1x, so the filter is an area average: bilinear reads four source
pixels where the output pixel covers up to twenty-six, and what it misses folds back as
aliasing that varies with the source resolution — the exact nuisance variable that
resampling exists to remove.

**Through plane, do not.** Every slot holds an acquired slice or nothing. A 2D stack is
placed as it was acquired and the remaining slots are padding; a 3D stack is thinned by
an integer stride, which discards slices without inventing any. Nothing is interpolated
between slices, and no slice is ever duplicated.

The price is that the physical spacing varies between studies — 2.20 to 5.50 mm, a
factor of 2.5 — so a depth kernel of three slices spans 6.6 mm on one study and 16.5 mm
on another. That is why `t_mm` travels with the volume: the model's depth axis is an
index, but the target and the decoding are in millimetres, and the two are joined per
study rather than assumed equal.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import LandmarkConfig


@dataclass(frozen=True)
class Sampled:
    """A study's stack in the model's frame, with everything needed to leave it."""

    volume: np.ndarray        #: (slices, img, img) uint8 — padding is zero
    valid: np.ndarray         #: (slices,) bool — which slots hold an acquired slice
    t_mm: np.ndarray          #: (slices,) through-plane coordinate, NaN where invalid
    ipp: np.ndarray           #: (slices, 3) patient origin per slot, NaN where invalid
    iop: np.ndarray           #: (6,) direction cosines, taken as constant over the stack
    normal: np.ndarray        #: (3,) unit vector the stack advances along
    transform: dict           #: native pixel <-> resampled pixel, per `resample_plane`
    spacing: np.ndarray       #: (2,) native PixelSpacing — (between rows, between cols)
    native_spacing_mm: float
    native_n: int
    stride: int               #: 1 unless the series was decimated
    sop: list                 #: SOPInstanceUID per slot, None where padded

    @property
    def mm_per_px_rc(self) -> np.ndarray:
        """Millimetres per resampled pixel, along rows and along columns.

        Not `config.mm_per_px`: the field of view is rounded to a whole number of native
        pixels before the resize, so the realised scale is very slightly off the nominal
        one. Using the nominal value would put a small, study-dependent error into every
        coordinate — small enough never to be noticed, which is the kind that survives.
        """

        return np.asarray(self.spacing, float) * self.transform["scale"]


def choose_depth(n: int, spacing_mm: float,
                 config: LandmarkConfig) -> tuple[np.ndarray, int]:
    """Which acquired slices to keep, and the integer stride used to get there.

    A series is thinned only when its spacing is finer than **half** the target, which
    is to say only when it is a genuine 3D acquisition. Deciding by "does it fit" would
    halve a 50-slice 2D series that overflows 48 slots by two, turning 3.3 mm into
    6.6 mm to save two slices at the periphery; cropping the ends is the cheaper answer
    there, and the ends are outside the joint.
    """

    stride = 1
    if spacing_mm > 0 and spacing_mm < config.decimate_to_mm / 2:
        stride = max(1, int(round(config.decimate_to_mm / spacing_mm)))

    keep = np.arange(0, n, stride)
    if len(keep) > config.slices:
        # Too long even after thinning: keep the middle. The joint is centred in the
        # acquisition and the slices lost are the empty periphery.
        start = (len(keep) - config.slices) // 2
        keep = keep[start:start + config.slices]
    return keep, stride


def place(count: int, config: LandmarkConfig, offset: int | None = None) -> int:
    """Where the acquired slices start among the slots.

    Centred by default. Training should pass a random offset instead: with the stack
    always centred, "the joint is in the middle" is true of the padding pattern as well
    as of the anatomy, and the model can read the depth off the padding without ever
    looking at a knee. Same reasoning as reversing the stack for laterality.
    """

    room = max(0, config.slices - count)
    if offset is None:
        return room // 2
    return int(np.clip(offset, 0, room))


def resample_plane(slice_: np.ndarray, mm_per_px: float,
                   config: LandmarkConfig) -> tuple[np.ndarray, dict]:
    """One slice at the fixed physical field of view, plus the transform to undo it."""

    import cv2

    want = int(round(config.fov_mm / mm_per_px))
    h, w = slice_.shape
    r0, c0 = h // 2 - want // 2, w // 2 - want // 2

    # Pad rather than shrink the field: a smaller acquisition keeps its scale and gains
    # a black margin, so a millimetre is the same number of pixels in every study.
    canvas = np.zeros((want, want), slice_.dtype)
    sr0, sc0 = max(r0, 0), max(c0, 0)
    sr1, sc1 = min(r0 + want, h), min(c0 + want, w)
    canvas[sr0 - r0:sr1 - r0, sc0 - c0:sc1 - c0] = slice_[sr0:sr1, sc0:sc1]

    interp = cv2.INTER_AREA if want >= config.img else cv2.INTER_LINEAR
    out = cv2.resize(canvas, (config.img, config.img), interpolation=interp)
    return out, {"row0": float(r0), "col0": float(c0), "scale": want / config.img}


def to_native(transform: dict, row: float, col: float) -> tuple[float, float]:
    """Resampled pixel -> the native pixel it came from."""

    return (transform["row0"] + row * transform["scale"],
            transform["col0"] + col * transform["scale"])


def to_resampled(transform: dict, row: float, col: float) -> tuple[float, float]:
    """Native pixel -> where it lands in the resampled slice."""

    return ((row - transform["row0"]) / transform["scale"],
            (col - transform["col0"]) / transform["scale"])


def _slice_geometry(path) -> dict | None:
    """What a slice needs to be placed in the patient, or None if it cannot be."""

    import pydicom

    try:
        ds = pydicom.dcmread(str(path), force=True, stop_before_pixels=True)
        return {"sop": str(ds.SOPInstanceUID),
                "ipp": np.array([float(v) for v in ds.ImagePositionPatient]),
                "iop": np.array([float(v) for v in ds.ImageOrientationPatient]),
                "ps": np.array([float(v) for v in ds.PixelSpacing])}
    except Exception:  # noqa: BLE001
        return None


def sample_series(directory, names: list[str], config: LandmarkConfig,
                  offset: int | None = None) -> Sampled:
    """Read one already-ordered sagittal series into the model's frame.

    `names` must be in through-plane order — `rsna.dicom.ordering.order_slices` produces
    it. Passing file order instead silently produces a stack whose depth axis means
    nothing, which is the failure `ordering.py` was written to document.
    """

    import pydicom

    from ..dicom.geometry import normal_of, through_plane

    directory = Path(directory)
    geom = [_slice_geometry(directory / n) for n in names]
    usable = [i for i, g in enumerate(geom) if g is not None]
    if not usable:
        raise ValueError(f"{directory} has no slice with usable geometry")

    iop = geom[usable[0]]["iop"]
    spacing = geom[usable[0]]["ps"]
    normal = normal_of(iop)
    t_all = np.array([through_plane(g["ipp"], normal) if g else np.nan for g in geom])

    steps = np.abs(np.diff([x for x in t_all if np.isfinite(x)]))
    native_spacing = float(np.median(steps)) if len(steps) else float("nan")

    keep, stride = choose_depth(len(names), native_spacing, config)
    start = place(len(keep), config, offset)

    volume = np.zeros((config.slices, config.img, config.img), np.uint8)
    valid = np.zeros(config.slices, bool)
    t_mm = np.full(config.slices, np.nan)
    ipp = np.full((config.slices, 3), np.nan)
    sop: list = [None] * config.slices
    transform = None

    # One window over the whole series, so brightness does not jump between slices the
    # way it would if each were scaled to its own extremes.
    planes, lo, hi = {}, None, None
    raw = {}
    for i in keep:
        try:
            ds = pydicom.dcmread(str(directory / names[i]), force=True)
            a = ds.pixel_array.astype(np.float32)
            raw[i] = a * float(getattr(ds, "RescaleSlope", 1) or 1) \
                + float(getattr(ds, "RescaleIntercept", 0) or 0)
        except Exception:  # noqa: BLE001
            continue
    if raw:
        stacked = np.concatenate([v.ravel() for v in raw.values()])
        lo, hi = np.percentile(stacked, (1.0, 99.5))
    for i, a in raw.items():
        planes[i] = np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1).__mul__(255).astype(np.uint8)

    for j, i in enumerate(keep):
        if i not in planes or geom[i] is None:
            continue
        out, transform = resample_plane(planes[i], float(geom[i]["ps"][1]), config)
        slot = start + j
        volume[slot] = out
        valid[slot] = True
        t_mm[slot] = t_all[i]
        ipp[slot] = geom[i]["ipp"]
        sop[slot] = geom[i]["sop"]

    if transform is None:
        raise ValueError(f"{directory} decoded no slice")

    return Sampled(volume=volume, valid=valid, t_mm=t_mm, ipp=ipp, iop=iop,
                   normal=normal, transform=transform, spacing=spacing,
                   native_spacing_mm=native_spacing, native_n=len(names),
                   stride=stride, sop=sop)
