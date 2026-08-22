#!/usr/bin/env python
"""Every figure a finished training run can produce: loss curve, a GT-vs-prediction
image, and one GT|prediction video per test episode.

Safe to run at any point -- each step is independent and skips cleanly when the arrays
it needs have not been written yet.

    python scripts/03_forecast/visualize.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # scripts/, for _bootstrap
import _bootstrap  # noqa: E402,F401  (repo root on sys.path; must precede `calm`)

import argparse  # noqa: E402

from calm.forecast.viz.figures import run_figures  # noqa: E402


def main() -> None:
    argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    run_figures()


if __name__ == "__main__":
    main()
