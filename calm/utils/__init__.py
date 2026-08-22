"""Sub-package containing utilities for common operations and helper functions.

These are the pieces every stage shares -- the live console boards, the process-pool
fan-out, the run-output paths, and the sweep grid -- kept free of any dependency on the
solver, the predictor, or the evaluator so that importing them stays cheap and cannot
cycle back through the packages that use them.
"""

from .console import AnsiBoard, bar, boxed, enable_ansi, live
from .parallel import ignore_sigint, imap_unordered
from .paths import (
    CALM_CONFIGS_DIR,
    CALM_DATA_DIR,
    CALM_MODELS_DIR,
    CALM_REPORTS_DIR,
    CALM_ROOT_DIR,
    run_dir,
    timestamp,
)
from .sweep import agent_count_sweep, frac_sweep, grid_cells

__all__ = [
    # console
    "AnsiBoard",
    "bar",
    "boxed",
    "enable_ansi",
    "live",
    # parallel
    "ignore_sigint",
    "imap_unordered",
    # paths
    "CALM_ROOT_DIR",
    "CALM_DATA_DIR",
    "CALM_MODELS_DIR",
    "CALM_REPORTS_DIR",
    "CALM_CONFIGS_DIR",
    "run_dir",
    "timestamp",
    # sweep
    "agent_count_sweep",
    "frac_sweep",
    "grid_cells",
]
