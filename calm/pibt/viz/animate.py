"""Turn PIBT grid paths into an MP4.

Three stages, one function each: build the static figure, produce a frame, encode.
``animate_paths`` wires them together.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

from calm.utils import backend  # noqa: F401  -- selects Agg before pyplot is imported

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter

from ..config import MAPFConfig
from ..metrics import paths_to_agent_positions
from ..types import Coord, PathType
from .background import plot_map_background
from .overlays import TaskOverlay

FPS = 30  # fixed playback frame rate


def _build_figure(env, paths, starts, goals, config, id_cmap, task_summary):
    """The static layer: map background, start squares, and the faint planned routes."""
    fig, ax = plt.subplots(figsize=(10, 8))
    plot_map_background(ax, np.asarray(env["factory_map"]),
                        np.asarray(env["walkable_map"]), colors=env.get("colors"))

    for agent_id, (start, goal) in enumerate(zip(starts, goals)):
        color = id_cmap(agent_id % 20)
        ax.scatter([start[0]], [start[1]], marker="s",
                   s=getattr(config, "viz_start_size", 50),
                   color=color, edgecolor="black", alpha=0.8)
        if task_summary is None:      # no live targets to draw -> show the fixed goal instead
            ax.scatter([goal[0]], [goal[1]], marker="*", s=90,
                       color=color, edgecolor="black", alpha=0.8)

    # Each agent's full planned route, drawn once as a faint static underlay so the video
    # shows where every robot is headed. Toggle with the `show_planned_routes` key.
    if getattr(config, "show_planned_routes", True):
        for agent_id, path in enumerate(paths):
            if len(path) < 2:
                continue
            ax.plot([p[0] for p in path], [p[1] for p in path],
                    color=id_cmap(agent_id % 20),
                    linewidth=getattr(config, "viz_planned_route_linewidth", 1.0),
                    alpha=0.30,
                    zorder=0.6,       # under the robots, route lines, and target markers
                    solid_capstyle="round")
    return fig, ax


def _interpolate(positions: np.ndarray, frame: int, subframes: int,
                 makespan: int) -> Tuple[np.ndarray, float]:
    """Linear interpolation between the two grid cells bracketing this frame."""
    if frame == 0:
        return positions[0], 0.0
    base_t = min((frame - 1) // subframes + 1, makespan)
    alpha = ((frame - 1) % subframes + 1) / subframes
    prev_positions, curr_positions = positions[base_t - 1], positions[base_t]
    return prev_positions + (curr_positions - prev_positions) * alpha, float(base_t)


def _encode(anim: FuncAnimation, output_dir: Path, config: MAPFConfig) -> None:
    """Encode on the GPU (NVENC H.264) at high quality; fall back to CPU x264, then GIF --
    so a box without an NVIDIA GPU (or a fresh clone without ffmpeg) still gets output."""
    dpi = int(getattr(config, "viz_video_dpi", 200))   # figsize (10,8) @ dpi 200 -> 2000x1600
    cq = int(getattr(config, "viz_video_cq", 15))      # NVENC constant quality (lower = better)
    encoders = [
        ("h264_nvenc", ["-preset", "p7", "-tune", "hq", "-rc", "vbr",
                        "-cq", str(cq), "-b:v", "0", "-pix_fmt", "yuv420p"]),
        ("libx264", ["-preset", "slow", "-crf", str(cq + 1), "-pix_fmt", "yuv420p"]),
    ]
    gif = output_dir / "classical_mapf_animation.gif"
    if FFMpegWriter.isAvailable():
        mp4 = output_dir / "classical_mapf_animation.mp4"
        for codec, extra in encoders:
            try:
                anim.save(mp4, writer=FFMpegWriter(fps=FPS, codec=codec, extra_args=extra), dpi=dpi)
                return
            except Exception:                          # encoder unavailable -> try the next one
                continue
    anim.save(gif, writer=PillowWriter(fps=FPS), dpi=dpi)


def animate_paths(
    env: Dict[str, Any],
    paths: Sequence[PathType],
    starts: Sequence[Coord],
    goals: Sequence[Coord],
    output_dir: Path,
    config: MAPFConfig,
    task_summary: Optional[Dict[str, Any]] = None,
) -> None:
    """Write ``classical_mapf_animation.mp4`` (or .gif) for one PIBT run.

    Pass the solver's ``summary`` as ``task_summary`` to draw each AMR's live
    pickup/delivery target and the dashed line to it; without it, the fixed goals are
    drawn as stars instead.
    """
    if not paths:
        return

    makespan = max(len(path) - 1 for path in paths)
    positions = paths_to_agent_positions(paths, makespan).astype(np.float32)
    id_cmap = plt.get_cmap("tab20")

    fig, ax = _build_figure(env, paths, starts, goals, config, id_cmap, task_summary)
    overlay = TaskOverlay(ax, len(paths),
                          task_summary.get("task_assignments", []) if task_summary else [],
                          id_cmap, config)
    scat = ax.scatter([], [], s=getattr(config, "viz_robot_size", 95),
                      marker="o", edgecolor="black", linewidth=0.7)
    title = ax.set_title("")

    subframes = max(1, int(config.animation_subframes))
    agent_colors = [id_cmap(i % 20) for i in range(len(paths))]

    def update(frame: int):
        frame_positions, visual_t = _interpolate(positions, frame, subframes, makespan)
        scat.set_offsets(frame_positions)
        scat.set_color(agent_colors)
        overlay.update(frame, frame_positions)
        title.set_text(f"Classical MAPF agent positions, t={visual_t:.2f}")
        return (scat, title, *overlay.artists)

    anim = FuncAnimation(fig, update, frames=makespan * subframes + 1,
                         interval=config.animation_interval_ms, blit=False)
    _encode(anim, output_dir, config)
    plt.close(fig)
