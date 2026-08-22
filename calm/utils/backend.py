"""Sub-module for the matplotlib backend setup every drawing module shares.

Imported for its side effects, and imported FIRST by any module that draws: the Agg
backend must be selected before ``pyplot`` is touched, or a worker process without a
display will fail. Pointing matplotlib at the ffmpeg bundled with imageio-ffmpeg is
what lets the renderers write MP4 under multiprocessing with no system ffmpeg::

    from calm.utils import backend  # noqa: F401  -- selects Agg before pyplot
    import matplotlib.pyplot as plt

Deliberately NOT re-exported from :mod:`calm.utils`' own ``__init__``, because importing
it pulls in matplotlib -- and the solver paths that use :mod:`calm.utils.parallel` or
:mod:`calm.utils.paths` must stay free of it. Import this module by name.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Optional

import matplotlib

matplotlib.use("Agg")  # headless; safe inside worker processes


@lru_cache(maxsize=1)
def ffmpeg_exe() -> Optional[str]:
    """Path to the ffmpeg binary bundled with imageio-ffmpeg, or None if unavailable.

    Cached because resolving it touches the filesystem and the video tools ask per clip.
    Callers that shell out to ffmpeg directly (rather than going through matplotlib's
    writer) use this so there is one answer to "which ffmpeg" in the whole repo.
    """
    try:
        import imageio_ffmpeg
    except Exception:
        return None
    try:
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


# Hand the same binary to matplotlib's FFMpegWriter, so animation.save() and the direct
# ffmpeg calls in calm.eval.video agree on the executable.
_exe = ffmpeg_exe()
if _exe is not None:
    matplotlib.rcParams["animation.ffmpeg_path"] = _exe
