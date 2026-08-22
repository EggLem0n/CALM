"""The live grid board: one mark per (AMR count, dispersion fraction) cell."""
from __future__ import annotations

import time

from calm.utils.console import AnsiBoard


class GridBoard(AnsiBoard):
    """In-place board of the whole (AMR count) x (dispersion frac) grid: every cell shows
    done / running / pending, so you can SEE which cells are in flight."""
    MARKS = {0: "\u00b7", 1: "\u25b6", 2: "\u2713"}   # pending / running / done

    def __init__(self, counts, fracs, total_units, status, counter, running=None):
        super().__init__()
        self.counts, self.fracs = counts, fracs
        self.status, self.counter = status, counter
        self.running = running
        self.total_cells = len(counts) * len(fracs)
        self.total_units = max(1, total_units)
        self.started = time.perf_counter()

    def body(self):
        nf = len(self.fracs)
        lines = ["       " + "".join(f"{f:>5.1f}" for f in self.fracs)]
        for ci, c in enumerate(self.counts):
            row = f" n{c:<5}"
            for fi in range(nf):
                row += f"{self.MARKS.get(self.status[ci * nf + fi], '?'):^5}"
            lines.append(row)
        done = sum(1 for i in range(self.total_cells) if self.status[i] == 2)
        run = sum(1 for i in range(self.total_cells) if self.status[i] == 1)
        units = min(self.counter.value, self.total_units)
        elapsed = time.perf_counter() - self.started
        frac = units / self.total_units
        eta = (elapsed / frac - elapsed) if frac > 0 else 0.0
        lines.append("")
        lines.append(f" cells {done}/{self.total_cells} done, {run} running  |  "
                     f"steps {units}/{self.total_units} ({frac * 100:4.0f}%)  |  "
                     f"elapsed {elapsed / 60:5.1f}m  eta {eta / 60:5.1f}m")
        try:
            active = sorted(self.running.values()) if self.running is not None else []
        except RuntimeError:                          # dict mutated mid-iteration; skip this frame
            active = []
        if active:
            lines.append(f" running now ({len(active)}):")
            lines.extend(f"   {lab}" for lab in active)
        return lines
