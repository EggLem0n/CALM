#!/usr/bin/env python
"""Per-horizon (t+1 .. t+10) accuracy of the congestion predictor: CSV + plot.

    python scripts/03_forecast/per_frame_accuracy.py
    python scripts/03_forecast/per_frame_accuracy.py --no-plot
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # scripts/, for _bootstrap
import _bootstrap  # noqa: E402,F401  (repo root on sys.path; must precede `calm`)

import argparse  # noqa: E402

from calm.forecast.per_frame_accuracy import (  # noqa: E402
    add_accuracy_args,
    run_accuracy,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_accuracy_args(ap)
    run_accuracy(ap.parse_args())


if __name__ == "__main__":
    main()
