"""What the pixels look like at each stage.

Matplotlib is imported inside each function rather than at module scope: this package
is imported by `rsna.viz`, which the training script pulls in, and a headless run has
no reason to pay for a plotting backend it will never use.
"""

from __future__ import annotations

import numpy as np

from ..config import Config
from .steps import load_stack, preprocessing_steps


def figure_stack(record: dict, config: Config, max_shown: int = 24):
    """Every slice in physical order; the sampled ones outlined."""

    import matplotlib.pyplot as plt

    stack, sampled = load_stack(record, config)
    n = len(stack)
    shown = list(range(n)) if n <= max_shown else sorted(
        set(np.linspace(0, n - 1, max_shown).astype(int)) | set(sampled))

    cols = min(8, len(shown))
    rows = int(np.ceil(len(shown) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(1.7 * cols, 1.8 * rows))
    for ax, i in zip(np.atleast_1d(axes).ravel(), shown):
        lo, hi = np.percentile(stack, [1, 99])
        ax.imshow(np.clip((stack[i] - lo) / max(hi - lo, 1e-6), 0, 1), cmap="gray")
        ax.set_xticks([]); ax.set_yticks([])
        picked = i in sampled
        ax.set_title(f"{i}{'  ←' if picked else ''}", fontsize=8,
                     color="tab:red" if picked else "black")
        for spine in ax.spines.values():
            spine.set_edgecolor("tab:red" if picked else "0.85")
            spine.set_linewidth(2.0 if picked else 0.5)
    for ax in np.atleast_1d(axes).ravel()[len(shown):]:
        ax.axis("off")
    fig.suptitle(f"{n} slices in physical order — sampled: {sampled}", fontsize=10)
    fig.tight_layout()
    return fig


def figure_steps(record: dict, config: Config, side: str | None = None, slice_index: int = 1):
    """One panel per transformation, on a single slice."""

    import matplotlib.pyplot as plt

    steps = preprocessing_steps(record, config, side)
    fig, axes = plt.subplots(1, len(steps), figsize=(2.6 * len(steps), 3.2))
    for ax, (title, volume) in zip(np.atleast_1d(axes), steps):
        image = volume[min(slice_index, len(volume) - 1)]
        ax.imshow(image, cmap="gray")
        ax.set_title(title, fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    return fig


def figure_channels(record: dict, config: Config, side: str | None = None,
                    start: int = 0):
    """One window as the encoder's channels, ordered geometrically and by file name.

    The cache holds `config.slices`; the encoder takes `config.group` of them, so this
    shows the window beginning at `start` rather than the whole cache.

    The composite is the point: neighbouring slices agree, so the RGB stays close to
    grey with thin fringes at edges. Arbitrary slices do not agree, and the colour
    fringing across the whole image is the third dimension carrying noise — with no
    error raised anywhere.
    """

    import matplotlib.pyplot as plt

    group = config.group

    def window(rec: dict) -> np.ndarray:
        volume = preprocessing_steps(rec, config, side)[-1][1]
        begin = min(start, max(len(volume) - group, 0))
        return volume[begin:begin + group]

    correct = window(record)

    shuffled = dict(record)
    shuffled["ordered"] = list(record["files"])  # file-name order: a SOP UID sort
    wrong = window(shuffled)

    fig, axes = plt.subplots(2, group + 1, figsize=(2.7 * (group + 1), 6))
    axes = np.atleast_2d(axes)
    for row, (volume, label) in enumerate(((correct, "geometric order"),
                                           (wrong, "file-name order"))):
        for c in range(group):
            axes[row, c].imshow(volume[c], cmap="gray", vmin=0, vmax=255)
            axes[row, c].set_title(f"{label} — channel {c}", fontsize=8)
        axes[row, -1].imshow(np.transpose(volume, (1, 2, 0)))
        axes[row, -1].set_title(f"{label} — as RGB", fontsize=8)
        for ax in axes[row]:
            ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    return fig


def figure_cache(record: dict, config: Config, side: str | None = None):
    """The slices the cache actually holds, in order, as the model will see them."""

    import matplotlib.pyplot as plt

    volume = preprocessing_steps(record, config, side)[-1][1]
    n = volume.shape[0]
    fig, axes = plt.subplots(1, n, figsize=(1.5 * n, 2.1))
    for i, ax in enumerate(np.atleast_1d(axes)):
        ax.imshow(volume[i], cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"cache {i}", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"{n} slices cached per slot — the encoder takes {config.group} at a time",
                 fontsize=10)
    fig.tight_layout()
    return fig


def figure_windows(record: dict, config: Config, side: str | None = None,
                   overlap: bool = True, max_rows: int = 10):
    """One row per window: the three channels, and what they make as an RGB image.

    `overlap=False` shows the windows a training step can draw from — disjoint, one
    per step, chosen at random. `overlap=True` shows the windows inference slides
    across and averages. They are not the same set, which is deliberate: training is
    stochastic, inference averages the randomness away.
    """

    import matplotlib.pyplot as plt

    volume = preprocessing_steps(record, config, side)[-1][1]
    starts = config.windows(overlap=overlap)[:max_rows]

    fig, axes = plt.subplots(len(starts), config.group + 1,
                             figsize=(2.1 * (config.group + 1), 2.0 * len(starts)))
    axes = np.atleast_2d(axes)
    for row, start in enumerate(starts):
        window = volume[start:start + config.group]
        for c in range(config.group):
            axes[row, c].imshow(window[c], cmap="gray", vmin=0, vmax=255)
            axes[row, c].set_title(f"slice {start + c}", fontsize=7)
        axes[row, -1].imshow(np.transpose(window, (1, 2, 0)))
        axes[row, -1].set_title(f"window {start} as RGB", fontsize=7)
        for ax in axes[row]:
            ax.set_xticks([]); ax.set_yticks([])
    kind = ("inference: every window, averaged" if overlap
            else "training: one of these, drawn at random per step")
    fig.suptitle(f"{len(starts)} windows — {kind}", fontsize=10)
    fig.tight_layout()
    return fig
