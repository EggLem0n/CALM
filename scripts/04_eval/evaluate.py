#!/usr/bin/env python
"""Congestion-aware PIBT vs plain PIBT across the sweep grid.

    python scripts/04_eval/evaluate.py                    # full grid, videos on
    python scripts/04_eval/evaluate.py --no-video         # metrics only
    python scripts/04_eval/evaluate.py --video-only --from-run <run>   # videos only
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # scripts/, for _bootstrap
import _bootstrap  # noqa: E402,F401  (repo root on sys.path; must precede `calm`)

import argparse  # noqa: E402

from calm.eval import (  # noqa: E402
    add_eval_args,
    run_sweep,
    validate_eval_args,
)
from calm.eval.cli import DESCRIPTION  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=DESCRIPTION)
    add_eval_args(ap)
    args = ap.parse_args()
    validate_eval_args(ap, args)
    run_sweep(args)


if __name__ == "__main__":
    main()
