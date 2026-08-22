"""Package containing the core framework.

CALM -- Congestion-Aware Lookahead Mobility -- is three stages that close a loop:

* :mod:`calm.pibt` -- lifelong PIBT MAPF solver, near-linear in agent count
* :mod:`calm.dataset` -- sweeps the solver to build the congestion dataset
* :mod:`calm.forecast` -- trains a SimVP forecaster and feeds it back into
  the solver as a move-cost penalty (``congestion_weight``)
* :mod:`calm.eval` -- scores congestion-aware runs against the plain-PIBT baseline

:mod:`calm.utils` holds what all four share. The commands that drive them live in
``scripts/``.

The names below are the entry points you actually reach for, gathered so a notebook or
a script needs one import instead of knowing the layout::

    import calm

    env = calm.build_map()                      # the 50x80 factory
    paths, summary = calm.plan(starts, ...)     # run the solver
    labels = calm.congestion_labels(positions, H, W)

Each is resolved on first attribute access (PEP 562), which buys three things:
``import calm`` costs nothing; torch and matplotlib stay out of the vanilla-PIBT paths;
and a sub-package importing from :mod:`calm.utils` cannot cycle back through here.
Sub-packages are always importable directly (``from calm import pibt as mapf``).
"""

import importlib
from typing import Any

from .utils.paths import (
    CALM_CONFIGS_DIR,
    CALM_DATA_DIR,
    CALM_MODELS_DIR,
    CALM_REPORTS_DIR,
    CALM_ROOT_DIR,
)

__version__ = "0.1.0"
"""Version of the CALM package, matching ``pyproject.toml``."""

# Public name -> "module:attribute". One line per entry point; the module each points at
# carries the real documentation.
_ENTRY_POINTS = {
    # -- stage 1: the map and the solver --
    "build_map": "calm.pibt.factory_map_generator:build_factory_map",
    "plan": "calm.pibt:plan_pibt_repeated_tasks",
    "select_starts": "calm.pibt:select_start_goal_pairs",
    "walkable_points": "calm.pibt:walkable_points",
    "congestion_labels": "calm.pibt:build_additive_congestion_label_sequence",
    "occupancy": "calm.pibt:build_occupancy_sequence",
    "positions_from_paths": "calm.pibt:paths_to_agent_positions",
    "count_collisions": "calm.pibt:compute_collision_count",
    "load_config": "calm.pibt:load_config",
    "Config": "calm.pibt:Config",
    "MAPFConfig": "calm.pibt:MAPFConfig",
    "DistanceFieldCache": "calm.pibt:DistanceFieldCache",
    # -- stage 2: the congestion forecaster (pulls in torch + OpenSTL) --
    "CongestionPredictor": "calm.forecast.predict:CongestionPredictor",
    # -- stage 3: evaluation --
    "solve": "calm.eval.runner:solve",
    "score_run": "calm.eval.runner:evaluate",
    "run_job": "calm.eval.runner:run_job",
    # -- rendering (pulls in matplotlib) --
    "animate": "calm.pibt.viz:animate_paths",
}

__all__ = [
    "__version__",
    "CALM_ROOT_DIR",
    "CALM_DATA_DIR",
    "CALM_MODELS_DIR",
    "CALM_REPORTS_DIR",
    "CALM_CONFIGS_DIR",
    *_ENTRY_POINTS,
]


def __getattr__(name: str) -> Any:
    """Resolve an entry point on first access, then cache it in the module globals."""
    if name not in _ENTRY_POINTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, _, attribute = _ENTRY_POINTS[name].partition(":")
    value = getattr(importlib.import_module(module), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
