"""Live task overlay: each AMR's current pickup/delivery target and the dashed line to it.

The solver's ``summary["task_assignments"]`` is a per-agent list of completed
assignments in order. The video has no notion of which one is "current", so this walks
each agent's list forward as its dot reaches each target -- that walk, and the artists
it drives, are what this module owns.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

TARGET_REACHED_RADIUS = 0.45   # cells; how close counts as "arrived" for the overlay


def _target_xy(assignment: Optional[Dict[str, Any]]) -> Optional[List[float]]:
    """The assignment's [x, y], or None when it carries no usable target."""
    if assignment is None:
        return None
    target = assignment.get("target")
    if not isinstance(target, list) or len(target) < 2:
        return None
    return [float(target[0]), float(target[1])]


class TaskOverlay:
    """Owns the target markers and route lines, and tracks which assignment is live."""

    def __init__(self, ax, n_agents: int, task_assignments: Sequence[Sequence[Dict[str, Any]]],
                 id_cmap, config) -> None:
        self.n_agents = n_agents
        self.assignments = task_assignments
        self.id_cmap = id_cmap
        self._index = [0] * n_agents          # per agent: how far along its assignment list
        self._last_frame = -1
        self._empty = np.empty((0, 2), dtype=np.float32)

        target_size = getattr(config, "viz_target_size", 150)
        self.pickup_scat = ax.scatter(
            self._empty[:, 0], self._empty[:, 1], marker="P", s=target_size,
            color="white", edgecolor="black", linewidth=0.9,
            label="Current pickup target", zorder=5)
        self.delivery_scat = ax.scatter(
            self._empty[:, 0], self._empty[:, 1], marker="*", s=target_size * 1.2,
            color="white", edgecolor="black", linewidth=0.9,
            label="Current delivery target", zorder=5)
        self.route_lines = [
            ax.plot([], [], linestyle="--",
                    linewidth=getattr(config, "viz_route_linewidth", 1.4),
                    color=id_cmap(agent_id % 20), alpha=0.72, zorder=4)[0]
            for agent_id in range(n_agents)
        ]

    # -- assignment tracking ---------------------------------------------------
    def advance(self, frame: int, positions: np.ndarray) -> None:
        """Step each agent past every target it has now reached. Frame-guarded because
        matplotlib may call the updater more than once for the same frame."""
        if frame <= self._last_frame:
            return
        self._last_frame = frame
        for agent_id in range(min(self.n_agents, len(self.assignments))):
            assignments = self.assignments[agent_id]
            while self._index[agent_id] < len(assignments):
                target = _target_xy(assignments[self._index[agent_id]])
                if target is None:                    # malformed entry: skip past it
                    self._index[agent_id] += 1
                    continue
                distance = float(np.linalg.norm(positions[agent_id] - np.asarray(target, np.float32)))
                if distance > TARGET_REACHED_RADIUS:
                    break
                self._index[agent_id] += 1

    def active_for(self, agent_id: int) -> Optional[Dict[str, Any]]:
        """The assignment this agent is currently heading to, if any."""
        if agent_id >= len(self.assignments):
            return None
        assignments = self.assignments[agent_id]
        if self._index[agent_id] >= len(assignments):
            return None
        return assignments[self._index[agent_id]]

    # -- artists ---------------------------------------------------------------
    def _target_offsets(self) -> Tuple[np.ndarray, List[Any], np.ndarray, List[Any]]:
        pickup_offsets: List[List[float]] = []
        pickup_colors: List[Any] = []
        delivery_offsets: List[List[float]] = []
        delivery_colors: List[Any] = []
        for agent_id in range(len(self.assignments)):
            active = self.active_for(agent_id)
            target = _target_xy(active)
            if target is None:
                continue
            color = self.id_cmap(agent_id % 20)
            if active.get("action") == "pickup":
                pickup_offsets.append(target)
                pickup_colors.append(color)
            else:
                delivery_offsets.append(target)
                delivery_colors.append(color)
        pickup = np.asarray(pickup_offsets, np.float32) if pickup_offsets else self._empty
        delivery = np.asarray(delivery_offsets, np.float32) if delivery_offsets else self._empty
        return pickup, pickup_colors, delivery, delivery_colors

    def _update_route_lines(self, positions: np.ndarray) -> None:
        for agent_id, line in enumerate(self.route_lines):
            target = _target_xy(self.active_for(agent_id))
            if target is None:
                line.set_data([], [])
                continue
            line.set_data([float(positions[agent_id, 0]), target[0]],
                          [float(positions[agent_id, 1]), target[1]])

    def update(self, frame: int, positions: np.ndarray) -> None:
        """Advance the assignment cursors, then repoint every artist at this frame."""
        self.advance(frame, positions)
        self._update_route_lines(positions)
        pickup, pickup_colors, delivery, delivery_colors = self._target_offsets()
        self.pickup_scat.set_offsets(pickup)
        self.pickup_scat.set_color(pickup_colors)
        self.delivery_scat.set_offsets(delivery)
        self.delivery_scat.set_color(delivery_colors)

    @property
    def artists(self) -> Tuple[Any, ...]:
        return (self.pickup_scat, self.delivery_scat, *self.route_lines)
