"""Sample every annotated study once, so training costs no DICOM.

    python -m scripts.build_landmark_cache \
        --annotations data/manual_annotations/landmarks-pf.csv \
        --config '{"points": ["pf_centre"]}' \
        --out /data/mgr/rsna-knee/landmark-cache-pf

The cache tag names the sampling geometry and nothing else, so two landmarks sampled the
same way would write the same filenames. `cache.load` refuses a cache built under a
different `LandmarkConfig` and `points` is part of it, so the mistake is caught rather
than silently read — but it is still one cache per landmark, in its own directory.

The series each study is sampled from is the one the annotator actually clicked on, read
from the table's `series` column. That is what makes this plane-agnostic: a sagittal
meniscus point and an axial patellofemoral one go through the same code because neither
the sampler nor the target encoder has an opinion about which plane it is looking at.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from rsna.landmark import cache                                      # noqa: E402
from rsna.landmark.config import LandmarkConfig                      # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--annotations", required=True, type=Path,
                    help="a table from tools.annotate.to_mm")
    ap.add_argument("--dicom-root", type=Path,
                    default=Path("/data/mgr/rsna-knee/extracted/train_series"))
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--config", default="{}",
                    help="JSON overriding LandmarkConfig fields, e.g. "
                         "'{\"points\": [\"pf_centre\"]}'")
    args = ap.parse_args()

    config = LandmarkConfig.from_dict(json.loads(args.config))
    print(f"{config.points} at {config.img}px / {config.fov_mm:.0f}mm, "
          f"{config.slices} slots, decimating towards {config.decimate_to_mm} mm")
    cache.build(args.annotations, args.dicom_root, args.out, config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
