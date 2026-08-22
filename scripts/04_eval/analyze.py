#!/usr/bin/env python
"""Rebuild analysis_summary.md + metrics_glossary.md for evaluation run folders.

    python scripts/04_eval/analyze.py reports/CALM_comparison/<run> [<run> ...]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # scripts/, for _bootstrap
import _bootstrap  # noqa: E402,F401  (repo root on sys.path; must precede `calm`)

import argparse  # noqa: E402

from calm.eval.make_analysis_summary import write_run_reports  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", nargs="+", help="evaluation run folder holding metrics.csv")
    for d in ap.parse_args().run_dir:
        for written in write_run_reports(d):
            print(f"wrote {written}")


if __name__ == "__main__":
    main()
