"""A scrollable page for any set of series, to look at one thing next to another.

    python -m tools.atlas.series_viewer --series <study_tail>:<series_tail> ... -o page.html

The atlas builds a curated page per pathology. This is the blunt instrument: name a few
series, get one HTML with a slider for each, captioned with the geometry that matters —
slice spacing, millimetres per pixel, physical extent. Useful whenever a claim about the
data would be easier to check by looking than by reading a table.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from rsna.config import Config                                   # noqa: E402
from tools.atlas import page as P                                # noqa: E402
from tools.atlas.render import stack_to_jpegs                    # noqa: E402
from tools.atlas.study import (TRAIN_SERIES, load_series,        # noqa: E402
                               series_headers, side_of, stack_orientation)


def find(study_tail: str, series_tail: str | None):
    """A study and one of its series, addressed by the tails of their UIDs."""

    hits = [d for d in TRAIN_SERIES.iterdir() if d.name.endswith(study_tail)]
    if len(hits) != 1:
        raise ValueError(f"{study_tail!r} matches {len(hits)} studies")
    headers = series_headers(hits[0].name)
    if series_tail:
        rows = headers[headers["SeriesInstanceUID"].str.endswith(series_tail)]
        if not len(rows):
            raise ValueError(f"no series ending {series_tail!r} in {study_tail}")
        return hits[0].name, headers, rows.iloc[0]
    # No series named: take the one with the most slices, which is the interesting one.
    return hits[0].name, headers, headers.sort_values("n_slices",
                                                      ascending=False).iloc[0]


def block(uid: str, headers, row, index: int, config: Config) -> str:
    series = load_series(row, config)
    side, how = side_of(headers)
    left, right = stack_orientation(series.plane, side)
    frames, _ = stack_to_jpegs(series.volume)

    steps = np.abs(np.diff([p for p in series.positions if np.isfinite(p)]))
    spacing = float(np.median(steps)) if len(steps) else float("nan")
    extent = float(np.ptp([p for p in series.positions if np.isfinite(p)])) \
        if len(steps) else float("nan")

    caption = (f"{P._esc(series.label)} &middot; <b>{series.n} slices</b> &middot; "
               f"{series.mm_per_px:.3f} mm/px in plane &middot; "
               f"<b>{spacing:.2f} mm between slices</b> &middot; "
               f"{series.thickness:.2f} mm thick &middot; {extent:.0f} mm covered "
               f"&middot; anisotropy <b>{spacing / series.mm_per_px:.1f}:1</b>")

    return f"""
<h3>&hellip;{P._esc(uid[-11:])} &mdash; {P._esc(series.description)}</h3>
{P.viewer(f"v{index}", frames, left, right, series.mm_per_px, caption=caption)}"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--series", nargs="+", required=True,
                    help="study_tail[:series_tail], repeatable")
    ap.add_argument("-o", "--out", required=True, type=Path)
    ap.add_argument("--title", default="Series viewer")
    args = ap.parse_args()

    config = Config()
    body = [f"<header><h1>{P._esc(args.title)}</h1>"
            f'<p class="lede">Scroll each stack. The captions carry the geometry: what '
            f'separates these series is not what they show but how finely they sample '
            f'it.</p></header>']
    for i, spec in enumerate(args.series, 1):
        study_tail, _, series_tail = spec.partition(":")
        uid, headers, row = find(study_tail, series_tail or None)
        body.append(block(uid, headers, row, i, config))
        print(f"  {spec} done", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(P.page(args.title, "".join(body)))
    print(f"\n{args.out}  ({args.out.stat().st_size/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
