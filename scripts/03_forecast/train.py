#!/usr/bin/env python
"""Train the SimVP congestion predictor (past 10 frames -> next 10 frames).

    python scripts/03_forecast/train.py
    TEST_ONLY=1 python scripts/03_forecast/train.py   # test the best checkpoint
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # scripts/, for _bootstrap
import _bootstrap  # noqa: E402,F401  (repo root on sys.path; must precede `calm`)

import argparse  # noqa: E402


def main() -> None:
    argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    # imported here, not at module scope: torch + the vendored OpenSTL tree cost seconds
    # to load, and --help must not pay for them.
    from calm.forecast.train import run_training

    run_training()


if __name__ == "__main__":
    main()
