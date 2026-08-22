"""Sub-package containing every figure the congestion forecaster produces.

Four independent pieces, each reading arrays that training already wrote: the
train/val loss curve, the ground-truth-vs-prediction stills and videos, the
error-vs-horizon curve that accompanies the per-horizon accuracy CSV, and the
single-horizon strip that checks the standalone checkpoint wrapper.

``window_strip`` is imported lazily by the functions that need it rather than here,
because annotating a predictor would otherwise pull torch into this package.

Deliberately NOT re-exported from :mod:`calm.forecast`'s own ``__init__``,
so nothing here is dragged in alongside the predictor::

    from calm.forecast import viz
"""
from .accuracy_curve import plot_curve
from .loss_curve import plot_loss_curve

__all__ = ["plot_curve", "plot_loss_curve"]
