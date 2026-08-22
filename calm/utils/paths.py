"""Sub-module defining where things live, and where a run writes its output.

Every tool used to recompute the repo root from its own ``__file__`` depth and build a
``<kind>/<yymmdd_hhmm>`` folder by hand. Both are here now, so a tool that moves between
packages doesn't silently point at the wrong root.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

# calm/utils/paths.py -> calm/utils/ -> calm/ -> repo root
CALM_ROOT_DIR = Path(__file__).resolve().parents[2]
"""Path to the repository root, which holds ``data/``, ``models/``, ``reports/`` and ``configs/``."""

CALM_DATA_DIR = CALM_ROOT_DIR / "data"
"""Path to the ``data`` directory, holding the generated congestion heatmap datasets."""

CALM_MODELS_DIR = CALM_ROOT_DIR / "models"
"""Path to the ``models`` directory, holding the trained SimVP checkpoints."""

CALM_REPORTS_DIR = CALM_ROOT_DIR / "reports"
"""Path to the ``reports`` directory, holding one folder per evaluation or figure run."""

CALM_CONFIGS_DIR = CALM_ROOT_DIR / "configs"
"""Path to the ``configs`` directory, holding ``default.yaml``."""


def timestamp() -> str:
    """The ``yymmdd_hhmm`` stamp every run folder is named with."""
    return datetime.now().strftime("%y%m%d_%H%M")


def run_dir(kind: str, *, root: Path | None = None, create: bool = True) -> Path:
    """A fresh ``<root>/<kind>/<yymmdd_hhmm>`` folder for one run's output.

    ``reports/`` is shared by several tools, so each writes under its own ``kind``
    (``congestion_prediction``, ``CALM_comparison``, ...) -- that keeps a run
    attributable to the code that produced it and to when it ran.

    Args:
        kind: Sub-folder naming the tool that owns the run.
        root: Directory the ``kind`` folder is created under. Defaults to
            :attr:`CALM_REPORTS_DIR`.
        create: Whether to create the folder. Set to False to only compute the path.

    Returns:
        The path to the run folder.
    """
    path = (root or CALM_REPORTS_DIR) / kind / timestamp()
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path
