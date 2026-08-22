"""Sub-package containing the congestion-aware PIBT evaluation: sweep the grid, run
it, score it, report it.

The pieces are importable on their own -- ``solve``/``evaluate`` to score a single
scenario, ``run_job`` for one grid point, ``run_sweep`` to drive a whole grid from an
already-parsed argument namespace. Nothing here reads ``sys.argv``: the parser is built
and parsed by ``scripts/04_eval/evaluate.py``, out of the flag definitions ``add_eval_args``
contributes.
"""
from .cli import add_eval_args, validate_eval_args
from .pipeline import run_sweep
from .runner import evaluate, run_job, solve

__all__ = ["add_eval_args", "validate_eval_args", "run_sweep",
           "evaluate", "solve", "run_job"]
