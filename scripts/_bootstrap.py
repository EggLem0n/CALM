"""Import this first, before anything from ``calm``.

Every launcher in this folder needs the same two things, and both have to happen
before the first ``import calm``:

* the repo root on ``sys.path``, so the scripts run from a clone without ``pip install``
* ``KMP_DUPLICATE_LIB_OK``, because torch (libiomp) and numpy/opencv (MKL) can load
  two OpenMP runtimes into one process

Doing it here rather than in each script keeps the launchers to their docstring and
one import -- and keeps the ``# noqa: E402`` clutter out of them.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
