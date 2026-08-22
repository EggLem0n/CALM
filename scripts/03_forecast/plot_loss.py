#!/usr/bin/env python
"""Train/val loss curve, optionally truncated to an early epoch.

Truncating is useful for showing the start of a run without the later epochs
squashing the y-axis.

    python scripts/03_forecast/plot_loss.py       # every logged epoch
    python scripts/03_forecast/plot_loss.py 20    # epochs <= 20
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # scripts/, for _bootstrap
import _bootstrap  # noqa: E402,F401  (repo root on sys.path; must precede `calm`)

import argparse  # noqa: E402

from calm.forecast.viz.loss_curve import (  # noqa: E402
    add_loss_curve_args,
    run_loss_curve,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_loss_curve_args(ap)
    run_loss_curve(ap.parse_args())


if __name__ == "__main__":
    main()
