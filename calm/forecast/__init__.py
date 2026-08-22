"""Sub-package containing the SimVP congestion forecaster.

Trains on the shards :mod:`calm.dataset` writes (past 10 frames -> next 10)
and serves the trained checkpoint back to the solver as a move-cost penalty.

Nothing is re-exported here: every module in this sub-package pulls in torch and the
vendored OpenSTL tree, and the vanilla-PIBT paths must stay free of both. Import the
module you need directly, e.g. ``from calm.forecast.predict import
CongestionPredictor`` -- or reach it lazily through ``calm.CongestionPredictor``.

Nothing here reads ``sys.argv`` either: the launchers under
``scripts/03_forecast/`` own their parsers and build them from the
``add_*_args`` helpers each module contributes.
"""
