"""Sub-module for in-place ANSI status boards used by long parallel runs.

The dataset generator and the grid evaluator both print a live status block that
redraws itself instead of scrolling away. The shared machinery -- enabling ANSI on
Windows consoles, repainting N lines with a cursor-up jump, clearing the tail when a
frame shrinks, and the ticker thread that keeps elapsed/ETA moving while workers are
busy -- lives here. Each tool subclasses :class:`AnsiBoard` and supplies its own
:meth:`body`.
"""
from __future__ import annotations

import os
import sys
import threading
from contextlib import contextmanager
from typing import Iterator, List


def enable_ansi() -> None:
    """Make Windows consoles interpret ANSI escapes. No-op elsewhere."""
    if os.name == "nt":
        os.system("")


class AnsiBoard:
    """A block of console lines that repaints itself in place.

    Subclasses implement :meth:`body`; call :meth:`draw` (or drive it with
    :func:`live`) to repaint. Drawing is locked so a ticker thread and the main
    thread cannot interleave escape sequences.
    """

    def __init__(self) -> None:
        self._lines_drawn = 0
        self._lock = threading.RLock()
        enable_ansi()
        try:  # box-drawing / check marks must not crash a cp949 console
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    def body(self) -> List[str]:
        """The lines to display, top to bottom. Recomputed on every repaint."""
        raise NotImplementedError

    def draw(self) -> None:
        with self._lock:
            rows = self.body()
            out = f"\x1b[{self._lines_drawn}A" if self._lines_drawn else ""
            out += "".join("\x1b[2K" + row + "\n" for row in rows)
            extra = self._lines_drawn - len(rows)
            if extra > 0:          # a taller previous frame would leave stale lines behind
                out += "\x1b[2K\n" * extra + f"\x1b[{extra}A"
            sys.stdout.write(out)
            sys.stdout.flush()
            self._lines_drawn = len(rows)


@contextmanager
def live(board: AnsiBoard, interval: float = 0.5) -> Iterator[AnsiBoard]:
    """Repaint ``board`` every ``interval`` seconds for the duration of the block.

    Paints once on entry and once more on exit, so the final state is always the
    last thing on screen even if the body only changed between ticks.
    """
    stop = threading.Event()

    def _tick() -> None:
        while not stop.wait(interval):
            board.draw()

    ticker = threading.Thread(target=_tick, daemon=True)
    board.draw()
    ticker.start()
    try:
        yield board
    finally:
        stop.set()
        ticker.join(timeout=2.0)
        board.draw()


def bar(fraction: float, width: int) -> str:
    """A ``[####----]``-style progress bar body, clamped to [0, 1]."""
    filled = int(width * min(1.0, max(0.0, fraction)))
    return "#" * filled + "-" * (width - filled)


def boxed(lines: List[str], inner_width: int) -> List[str]:
    """Wrap ``lines`` in a +---+ ASCII frame, truncating/padding to ``inner_width``."""
    rule = "+" + "-" * (inner_width + 2) + "+"
    return [rule] + ["| " + t[:inner_width].ljust(inner_width) + " |" for t in lines] + [rule]
