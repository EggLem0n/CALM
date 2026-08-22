"""Sub-package containing the dataset renderer: congestion heatmap + AMR overlay to MP4.

Rendering is post-hoc -- it reads the ``episode_*.npz`` shards
:mod:`calm.dataset.generate` already wrote and never re-simulates.

Deliberately NOT re-exported from :mod:`calm.dataset`'s own ``__init__``, so
importing the generator never pulls in matplotlib::

    from calm.dataset import viz
"""
from .heatmap_video import add_render_args, render_congestion_video, run_render

__all__ = ["render_congestion_video", "add_render_args", "run_render"]
