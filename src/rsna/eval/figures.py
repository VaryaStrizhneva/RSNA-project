"""The charts, as SVG and HTML rather than as images.

Three of them, deliberately, and none of them a PNG. A raster figure is baked at one
size and one background: stretched into the page it goes soft, it carries its own copy
of the palette, its labels are not text, and it costs a hundred kilobytes each. These
draw with the page instead of beside it — the strokes take their colour from the same
CSS variables as everything else, the labels are set in the page's own font, and the
whole report weighs less than one of the images it replaces.

Everything here returns a string of markup. No matplotlib, no file, no state.
"""

from __future__ import annotations

import html

import numpy as np

# The window worth showing for a metric whose floor is chance, not zero. A bar drawn
# from 0 makes 0.49 and 0.90 look like neighbours; drawn from here they do not.
AUC_FLOOR = 0.30


def _ticks(lo: float, hi: float, want: int = 5) -> list[float]:
    """Round numbers covering [lo, hi], about `want` of them."""

    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return [lo] if np.isfinite(lo) else []
    raw = (hi - lo) / max(want, 2)
    magnitude = 10.0 ** np.floor(np.log10(raw))
    step = min((m * magnitude for m in (1, 2, 2.5, 5, 10) if m * magnitude >= raw),
               default=magnitude * 10)
    first = np.ceil(lo / step) * step
    return [round(first + i * step, 10)
            for i in range(int((hi - first) / step) + 1)]


def _fmt(value: float) -> str:
    text = f"{value:g}"
    return text


class _Scale:
    """Data units to pixels, one axis."""

    def __init__(self, lo: float, hi: float, start: float, end: float):
        self.lo, self.hi = (lo, hi) if hi > lo else (lo, lo + 1.0)
        self.start, self.end = start, end

    def __call__(self, value: float) -> float:
        frac = (value - self.lo) / (self.hi - self.lo)
        return self.start + frac * (self.end - self.start)


def _panel(series: list[dict], title: str, width: int = 520, height: int = 250,
           chance: float | None = None) -> str:
    """One line chart. `series` entries carry x, y, colour, label and an optional mark.

    Drawn inside a `viewBox` and released at `width:100%`, so the browser rasterises it
    at whatever the display actually is. That is the whole reason this is not a PNG.
    """

    left, right, top, bottom = 44, 10, 28, 30
    values = [v for s in series for v in s["y"] if v is not None and np.isfinite(v)]
    xs = [v for s in series for v in s["x"]]
    if not values or not xs:
        return ""

    lo, hi = min(values), max(values)
    pad = (hi - lo) * 0.12 or 0.05
    if chance is not None:
        lo, hi = min(lo, chance), max(hi, chance)
    y = _Scale(lo - pad, hi + pad, height - bottom, top)
    x = _Scale(min(xs), max(xs), left, width - right)

    parts = [f'<svg class="chart" viewBox="0 0 {width} {height}" '
             f'role="img" aria-label="{html.escape(title)}">',
             f'<text class="ct" x="{left}" y="16">{html.escape(title)}</text>']

    for value in _ticks(lo - pad, hi + pad):
        py = y(value)
        parts.append(f'<line class="grid" x1="{left}" y1="{py:.1f}" '
                     f'x2="{width - right}" y2="{py:.1f}"/>')
        parts.append(f'<text class="tk" x="{left - 7}" y="{py + 3.5:.1f}" '
                     f'text-anchor="end">{_fmt(value)}</text>')

    for value in _ticks(min(xs), max(xs), want=6):
        if value != int(value):
            continue
        px = x(value)
        parts.append(f'<text class="tk" x="{px:.1f}" y="{height - bottom + 16:.1f}" '
                     f'text-anchor="middle">{int(value)}</text>')
    parts.append(f'<text class="tk" x="{(left + width - right) / 2:.0f}" '
                 f'y="{height - 4}" text-anchor="middle">epoch</text>')

    if chance is not None:
        parts.append(f'<line class="chance" x1="{left}" y1="{y(chance):.1f}" '
                     f'x2="{width - right}" y2="{y(chance):.1f}"/>')

    for s in series:
        points = " ".join(f"{x(a):.1f},{y(b):.1f}" for a, b in zip(s["x"], s["y"])
                          if b is not None and np.isfinite(b))
        if not points:
            continue
        parts.append(f'<polyline class="ln" points="{points}" '
                     f'stroke="{s["colour"]}"/>')
        mark = s.get("mark")
        if mark is not None and np.isfinite(mark[1]):
            parts.append(f'<circle class="dot" cx="{x(mark[0]):.1f}" '
                         f'cy="{y(mark[1]):.1f}" r="4" fill="{s["colour"]}"/>')

    parts.append("</svg>")
    return "".join(parts)


def curves(records: list) -> str:
    """Training loss and holdout AUC per epoch, every fold overlaid.

    Read together: a loss still falling while the AUC has flattened is a model
    memorising the training studies, and the gap between the two is the only place
    that shows.
    """

    loss, auc, legend = [], [], []
    for record in records:
        colour = f"var(--f{record.fold % 5})"
        epochs = [e["epoch"] + 1 for e in record.history]
        best = record.best_epoch
        aucs = [e.get("holdout_auc") for e in record.history]
        loss.append({"x": epochs, "y": [e.get("loss") for e in record.history],
                     "colour": colour})
        auc.append({"x": epochs, "y": aucs, "colour": colour,
                    "mark": (best + 1, aucs[best]) if 0 <= best < len(aucs) else None})
        legend.append(f'<span class="key"><i style="background:{colour}"></i>'
                      f'fold {record.fold}</span>')

    return (f'<div class="panels">{_panel(loss, "training loss")}'
            f'{_panel(auc, "holdout macro AUC", chance=0.5)}</div>'
            f'<div class="legend">{"".join(legend)}'
            f'<span class="key"><i class="ring"></i>epoch kept</span></div>')


def spread(fold_aucs: np.ndarray, pooled: float, interval: tuple) -> str:
    """The folds as points, the pooled figure and its interval behind them.

    Here to make one thing hard to miss: if the interval is wide enough to swallow the
    difference you are trying to measure, the run has not measured it.
    """

    width, height = 720, 96
    left, right = 34, 24
    x = _Scale(AUC_FLOOR, 1.0, left, width - right)
    axis = height - 30

    parts = [f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" '
             f'aria-label="fold spread">']
    lo, hi = interval
    if np.isfinite(lo) and np.isfinite(hi):
        parts.append(f'<rect class="band" x="{x(lo):.1f}" y="{axis - 26:.1f}" '
                     f'width="{x(hi) - x(lo):.1f}" height="52" rx="4"/>')
    parts.append(f'<line class="axis" x1="{left}" y1="{axis}" '
                 f'x2="{width - right}" y2="{axis}"/>')
    parts.append(f'<line class="chance" x1="{x(0.5):.1f}" y1="{axis - 30:.1f}" '
                 f'x2="{x(0.5):.1f}" y2="{axis + 6:.1f}"/>')
    if np.isfinite(pooled):
        parts.append(f'<line class="pooled" x1="{x(pooled):.1f}" y1="{axis - 32:.1f}" '
                     f'x2="{x(pooled):.1f}" y2="{axis + 6:.1f}"/>')

    for value in _ticks(AUC_FLOOR, 1.0, want=8):
        parts.append(f'<text class="tk" x="{x(value):.1f}" y="{axis + 20:.1f}" '
                     f'text-anchor="middle">{_fmt(value)}</text>')

    # Folds that land on the same value would overprint; nudge each one up a row.
    taken: list[float] = []
    for fold, value in enumerate(fold_aucs):
        if not np.isfinite(value):
            continue
        px = x(value)
        row = sum(1 for t in taken if abs(t - px) < 22)
        taken.append(px)
        cy = axis - 9 - row * 15
        parts.append(f'<circle class="dot" cx="{px:.1f}" cy="{cy:.1f}" r="5.5" '
                     f'fill="var(--f{fold % 5})"/>')
        parts.append(f'<text class="fl" x="{px + 9:.1f}" y="{cy + 3.5:.1f}">'
                     f'f{fold}</text>')
    parts.append("</svg>")
    return "".join(parts)


def target_bar(row: dict, lo: float = AUC_FLOOR, hi: float = 1.0) -> str:
    """One pathology's AUC as a bar, with its confidence interval drawn around it.

    The bar is the estimate; the lighter span behind it is where the true value
    plausibly sits. Drawing both is the point — with a dozen positives the span covers
    a third of the axis, and a bar on its own would invite reading a difference that
    the run never measured. A span crossing the chance mark means nothing was found,
    whichever side of it the bar happens to end.
    """

    def pct(value: float) -> float:
        return max(0.0, min(1.0, (value - lo) / (hi - lo))) * 100.0

    if not np.isfinite(row["auc"]):
        return '<div class="track"><span class="none">no score</span></div>'

    kind = "flat" if row.get("flat") else "thin" if row["thin"] else "good"
    span = ""
    if np.isfinite(row.get("lo", np.nan)) and np.isfinite(row.get("hi", np.nan)):
        left = pct(row["lo"])
        span = (f'<i class="span" style="left:{left:.2f}%;'
                f'width:{max(pct(row["hi"]) - left, 0.4):.2f}%"></i>')
    return (f'<div class="track">{span}'
            f'<i class="fill {kind}" style="width:{pct(row["auc"]):.2f}%"></i>'
            f'<i class="chance-mark" style="left:{pct(0.5):.2f}%"></i></div>')
