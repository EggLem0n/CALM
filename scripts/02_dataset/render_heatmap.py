#!/usr/bin/env python
"""Render one MP4 per dataset episode: congestion heatmap + AMR positions.

    python scripts/02_dataset/render_heatmap.py --dataset data/heatmap_dataset/<run>
    python scripts/02_dataset/render_heatmap.py --dataset <dir> --episodes 0 209 --fps 60
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # scripts/, for _bootstrap
import _bootstrap  # noqa: E402,F401  (repo root on sys.path; must precede `calm`)

import argparse  # noqa: E402

from calm.dataset.viz import add_render_args, run_render  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_render_args(ap)
    run_render(ap.parse_args())


if __name__ == "__main__":
    main()
