"""Preview image and console summary for the factory map.

Split out of ``factory_map_generator`` so that module stays pure map construction
with no matplotlib in sight -- importing the solver must not drag in a plotting stack.
"""
from __future__ import annotations


import numpy as np

from calm.utils.paths import CALM_DATA_DIR, run_dir
from ..factory_map_generator import (
    BUFFER, CHARGING, COLORS, DELIVERY, FREE, H, INSPECTION,
    MACHINE, OBSTACLE, PICKUP, W, build_factory_map,
)

# Legend order (COLORS is keyed by zone code, so it carries no order).
ZONE_ORDER = (FREE, OBSTACLE, PICKUP, DELIVERY, INSPECTION, MACHINE, CHARGING, BUFFER)


def visualize_factory_map(data=None, save_path="factory_map_preview.png", show=True):
    """Visualize the map. matplotlib is imported lazily so data generation stays light."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgb
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    if data is None:
        data = build_factory_map()
    factory_map = data["factory_map"]

    rgb = np.zeros((H, W, 3), dtype=float)
    for code, color in COLORS.items():
        rgb[factory_map == code] = to_rgb(color)

    fig, ax = plt.subplots(figsize=(16, 10))
    ax.imshow(rgb, origin="upper")
    ax.set_xticks(np.arange(-0.5, W, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, H, 1), minor=True)
    ax.grid(which="minor", color="gray", linewidth=0.25, alpha=0.45)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_xticks(np.arange(0, W, 5))
    ax.set_yticks(np.arange(0, H, 5))
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5)
    ax.set_title("Automotive Assembly Line Parts-Supply Grid Map", fontsize=16, pad=20)
    ax.set_xlabel("x coordinate")
    ax.set_ylabel("y coordinate")

    def scatter_points(points, marker, label, size=80):
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        ax.scatter(xs, ys, marker=marker, s=size, edgecolors="black", linewidths=0.8, label=label)

    scatter_points(data["pickup_points"], "P", "Pickup Points")
    scatter_points(data["delivery_points"], "D", "Delivery Points", size=55)
    scatter_points(data["charging_points"], "^", "Charging Points")

    legend_elements = [
        Patch(facecolor=COLORS[FREE], edgecolor="black", label="Free / Road"),
        Patch(facecolor=COLORS[OBSTACLE], edgecolor="black", label="Obstacle / Conveyor / Equipment"),
        Patch(facecolor=COLORS[PICKUP], edgecolor="black", label="Parts Warehouse / Pickup"),
        Patch(facecolor=COLORS[DELIVERY], edgecolor="black", label="Line-side Delivery Zone"),
        Patch(facecolor=COLORS[INSPECTION], edgecolor="black", label="Inspection Zone"),
        Patch(facecolor=COLORS[MACHINE], edgecolor="black", label="Sequencing / Kitting / Supermarket"),
        Patch(facecolor=COLORS[CHARGING], edgecolor="black", label="Charging / Waiting"),
        Patch(facecolor=COLORS[BUFFER], edgecolor="black", label="Vehicle / Parts Buffer"),
        Line2D([0], [0], marker="P", color="w", markeredgecolor="black", markerfacecolor="black", label="Pickup Point", markersize=9),
        Line2D([0], [0], marker="D", color="w", markeredgecolor="black", markerfacecolor="black", label="Delivery Point", markersize=9),
        Line2D([0], [0], marker="^", color="w", markeredgecolor="black", markerfacecolor="black", label="Charging Point", markersize=9),
    ]
    ax.legend(handles=legend_elements, bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.0)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Visualization saved to: {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def print_map_summary(data):
    factory_map = data["factory_map"]
    walkable_map = data["walkable_map"]
    obstacle_map = data["obstacle_map"]
    print("=== Automotive Assembly Parts-Supply Map Summary ===")
    print(f"Map shape          : {factory_map.shape}  # (height, width)")
    print(f"Total cells        : {factory_map.size}")
    print(f"Walkable cells     : {int(walkable_map.sum())}")
    print(f"Obstacle cells     : {int(obstacle_map.sum())}")
    print(f"Pickup points      : {len(data['pickup_points'])}")
    print(f"Delivery points    : {len(data['delivery_points'])} points")
    print(f"Start candidates   : {data['start_candidates']}")


def run_preview() -> None:
    """Build the map, print its summary, and save the arrays plus a preview PNG."""
    data = build_factory_map()
    print_map_summary(data)
    maps_dir = run_dir("maps", root=CALM_DATA_DIR)
    for name in ("factory_map", "walkable_map", "obstacle_map"):
        np.save(maps_dir / f"{name}.npy", data[name])
    print(f"Saved numpy arrays to: {maps_dir}")
    visualize_factory_map(data, save_path=maps_dir / "factory_map_preview.png", show=False)
