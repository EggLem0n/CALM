"""Sub-package containing path animation for PIBT runs.

Adapted from the MACPF project (``macpf/classical_mapf/viz.py``). The kinodynamic
renderer it shipped with -- continuous speed integration, local AMR avoidance,
priority yielding, visual collision guards -- has been removed: PIBT produces discrete
grid paths and always passed ``agent_states=None``, so that branch never executed.
What remains linearly interpolates between grid cells.

Deliberately NOT re-exported from ``calm.pibt``'s own ``__init__``, so importing the
solver never pulls in matplotlib. Import explicitly::

    from calm.pibt import viz
"""
from .background import plot_map_background
from .animate import animate_paths

__all__ = ["animate_paths", "plot_map_background"]
