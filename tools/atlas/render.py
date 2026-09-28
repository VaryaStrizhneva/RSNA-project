"""Pixels to something a browser can show, without lying about scale.

Two rules here. Windowing is computed once **per series**, not per slice, so that
scrolling the stack shows the anatomy changing rather than the contrast changing. And
every image knows its millimetres per pixel, so a caption can say how big a thing is
instead of how many pixels wide it is — which is the whole argument about why a
meniscal tear is hard.
"""

from __future__ import annotations

import base64

import cv2
import numpy as np

JPEG_QUALITY = 90
MAX_WIDTH = 512


def window(volume: np.ndarray, lo_pct: float = 1.0, hi_pct: float = 99.5) -> np.ndarray:
    """Whole volume to uint8 on one scale.

    MR intensity has no absolute meaning, so something has to set the display range.
    Per *series* rather than per slice: a slice-wise window would renormalise every
    frame, and scrolling would show contrast breathing rather than anatomy moving.
    """

    finite = volume[np.isfinite(volume)]
    lo, hi = np.percentile(finite, [lo_pct, hi_pct]) if finite.size else (0.0, 1.0)
    if hi <= lo:
        hi = lo + 1.0
    out = np.clip((volume - lo) / (hi - lo), 0, 1)
    return (out * 255).astype(np.uint8)


def to_jpeg(image: np.ndarray, max_width: int = MAX_WIDTH) -> tuple[str, float]:
    """One slice as a data URI, plus the scale factor it was resized by."""

    scale = 1.0
    if max_width and image.shape[1] > max_width:
        scale = max_width / image.shape[1]
        image = cv2.resize(image, (max_width, int(round(image.shape[0] * scale))),
                           interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    if not ok:
        raise RuntimeError("cv2 refused to encode a slice")
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode(), scale


def stack_to_jpegs(volume: np.ndarray, max_width: int = MAX_WIDTH):
    """Every slice of a series, on one window, as data URIs."""

    eight = window(volume)
    frames, scale = [], 1.0
    for i in range(len(eight)):
        uri, scale = to_jpeg(eight[i], max_width)
        frames.append(uri)
    return frames, scale


# -- what the model is given, beside what was acquired ----------------------- #

def crop_mm(image: np.ndarray, mm_per_px: float, extent_mm: float) -> np.ndarray:
    """The centred square crop the training pipeline takes, in millimetres."""

    want = int(round(extent_mm / mm_per_px))
    h, w = image.shape[:2]
    if want >= min(h, w):
        return image
    cy, cx, half = h // 2, w // 2, want // 2
    return image[max(0, cy - half):cy + half, max(0, cx - half):cx + half]


def patchify(image: np.ndarray, grid: int) -> np.ndarray:
    """Block-average to `grid` x `grid`, then blow back up.

    Not a claim that the encoder discards within-patch detail — it does not, a patch is
    projected whole. It is a picture of the *granularity of position*: everything inside
    one of these blocks arrives at the head as a single token, at one location.
    """

    small = cv2.resize(image, (grid, grid), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, image.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)


def resolution_panel(slice_: np.ndarray, mm_per_px: float, config) -> list[dict]:
    """The same slice as acquired, as cached, and as the token grid resolves it."""

    eight = window(slice_[None])[0]
    native_mm = mm_per_px

    cropped = crop_mm(eight, mm_per_px, config.crop_mm)
    crop_px = cropped.shape[0]
    cached = cv2.resize(cropped, (config.img, config.img), interpolation=cv2.INTER_AREA)
    cache_mm = config.crop_mm / config.img

    patch = 14                       # DINOv2-small
    grid = config.img // patch
    tokens = patchify(cached, grid)
    token_mm = cache_mm * patch

    return [
        {"title": "as acquired",
         "sub": f"{eight.shape[1]}×{eight.shape[0]} px · {native_mm:.2f} mm/px",
         "note": "the full field of view, native resolution — what a radiologist reads",
         "image": eight, "mm_per_px": native_mm},
        {"title": "as cached for training",
         "sub": f"{config.img}×{config.img} px · {cache_mm:.2f} mm/px",
         "note": f"cropped to {config.crop_mm:.0f} mm ({crop_px} px of the original) "
                 f"then resampled",
         "image": cached, "mm_per_px": cache_mm},
        {"title": "as the token grid locates it",
         "sub": f"{grid}×{grid} tokens · {token_mm:.1f} mm per token",
         "note": "one block is one patch token: everything inside it reaches the head "
                 "at a single position",
         "image": tokens, "mm_per_px": cache_mm},
    ]


def scale_bar_px(mm_per_px: float, displayed_scale: float, mm: float = 10.0) -> float:
    """How many displayed pixels span `mm` millimetres."""

    return (mm / mm_per_px) * displayed_scale
