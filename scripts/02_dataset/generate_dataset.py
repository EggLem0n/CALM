#!/usr/bin/env python
"""Sweep the PIBT solver over the (AMR count) x (dispersion) grid and write the
congestion dataset.

    python scripts/02_dataset/generate_dataset.py
    python scripts/02_dataset/generate_dataset.py --rounds 4 --seconds 3600 --num_of_process 8
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # scripts/, for _bootstrap
import _bootstrap  # noqa: E402,F401  (repo root on sys.path; must precede `calm`)

import argparse  # noqa: E402

from calm.dataset import (  # noqa: E402
    add_generate_args,
    run_generation,
    validate_generate_args,
)
from calm.dataset.generate import DESCRIPTION  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=DESCRIPTION)
    add_generate_args(ap)
    args = ap.parse_args()
    validate_generate_args(ap, args)
    run_generation(args)


if __name__ == "__main__":
    main()
