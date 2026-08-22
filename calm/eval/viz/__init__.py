"""Sub-package containing the comparison videos and the metrics table image.

A finished sweep draws two things: one side-by-side MP4 per selected cell (plain PIBT
against congestion-aware, re-rendered from the cached paths rather than re-simulated),
and a PNG of the by-lambda metrics aggregate. ``sched`` is what keeps several NVENC
encodes from colliding.

Deliberately NOT re-exported from :mod:`calm.eval`'s own ``__init__``, so scoring
a sweep never pulls in matplotlib::

    from calm.eval import viz
"""
from .table import save_metrics_table_png
from .video import video_job

__all__ = ["save_metrics_table_png", "video_job"]
