"""The static factory-map layer every animation is drawn on top of."""
from __future__ import annotations

from typing import Dict, Optional

from calm.utils import backend  # noqa: F401  -- selects Agg before pyplot is imported

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb


def plot_map_background(
    ax: plt.Axes,
    factory_map: np.ndarray,
    walkable_map: np.ndarray,
    colors: Optional[Dict[int, str]] = None,
) -> None:
    """Paint the zone map (or a plain walkable/blocked mask) and set up the cell grid.

    ``colors`` maps a zone code to a hex colour; without it the map falls back to
    white = walkable, dark = blocked. ``interpolation="nearest"`` keeps cells crisp
    rather than smeared at video resolution.
    """
    h, w = walkable_map.shape[:2]
    if colors:
        background = np.zeros((h, w, 3), dtype=float)
        for raw_code in np.unique(factory_map):
            code = int(raw_code)
            background[factory_map == code] = to_rgb(colors.get(code, "#f2f2f2"))
        ax.imshow(background, origin="upper", interpolation="nearest")
    else:
        background = np.where(walkable_map, 1.0, 0.15)
        ax.imshow(background, cmap="gray", origin="upper", vmin=0.0, vmax=1.0,
                  interpolation="nearest")
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)
    ax.set_xticks(np.arange(-0.5, w, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, h, 1), minor=True)
    ax.grid(which="minor", color="lightgray", linewidth=0.25)
    ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
