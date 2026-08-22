#!/usr/bin/env python
"""Build the factory map, print a summary, and save the arrays + a preview PNG.

    python scripts/01_pibt/preview_map.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # scripts/, for _bootstrap
import _bootstrap  # noqa: E402,F401  (repo root on sys.path; must precede `calm`)

import argparse  # noqa: E402

from calm.pibt.viz.factory_map import run_preview  # noqa: E402


def main() -> None:
    argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    run_preview()


if __name__ == "__main__":
    main()
