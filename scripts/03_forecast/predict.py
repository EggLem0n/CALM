#!/usr/bin/env python
"""Load the trained checkpoint and check it reproduces the saved predictions.

    python scripts/03_forecast/predict.py
    python scripts/03_forecast/predict.py --episode data/.../episode_0000.npz
    python scripts/03_forecast/predict.py --recompute-y-scale
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # scripts/, for _bootstrap
import _bootstrap  # noqa: E402,F401  (repo root on sys.path; must precede `calm`)

import argparse  # noqa: E402


def main() -> None:
    # imported here, not at module scope: the module pulls in torch and the vendored
    # OpenSTL tree, and --help must not pay for them.
    from calm.forecast.predict import add_predict_args, run_predict

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_predict_args(ap)
    run_predict(ap.parse_args())


if __name__ == "__main__":
    main()
