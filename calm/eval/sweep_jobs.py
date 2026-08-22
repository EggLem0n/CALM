"""Expanding the sweep axes into the flat list of planner runs.

One "job" is one grid point: a cell, a seed (repeat), and one full config
(lambda x gamma x depth mode x min depth x horizon x predict_every x cap).
Jobs are interleaved across cells so every cell has work in the first wave and no
worker sits idle until the tail.
"""
from __future__ import annotations

from itertools import zip_longest


def build_run_jobs(grid, args, axes, *, videos_on):
    """Return the interleaved job list for the whole sweep."""
    gammas, min_depths, depth_modes = axes["gammas"], axes["min_depths"], axes["depth_modes"]
    horizons, predict_everys, caps = axes["horizons"], axes["predict_everys"], axes["caps"]
    weights_pos, n_repeats = axes["weights_pos"], axes["n_repeats"]
    total = len(grid)
    # A run's trajectory is cached (for the video phase) iff it is a video candidate:
    # the baseline, or a PRIMARY-config congestion run (the gamma x lambda grid the videos
    # pick the best lambda from). Off when videos are disabled.
    videos_on = (not args.no_video) and bool(weights_pos)
    prim0 = (depth_modes[0], min_depths[0], predict_everys[0], horizons[0], caps[0])
    per_cell_jobs = []
    for ci, (count, frac) in enumerate(grid):
        jl = []
        for rep in range(n_repeats):
            # rep 0 reuses the original per-cell seed (base_seed + ci); each extra rep shifts by a
            # whole grid so every (cell, rep) gets a distinct seed. The seed is shared by all configs
            # within one (cell, rep), so each repeat stays a controlled A/B (only lambda/config varies).
            seed = args.base_seed + rep * total + ci
            save0 = videos_on and rep == 0       # only the representative seed feeds the video cache
            jl.append({"cell_idx": ci, "count": count, "frac": frac, "seed": seed, "rep": rep,
                       "baseline": True, "weight": 0.0, "gamma": gammas[0], "horizon": horizons[0],
                       "min_depth": min_depths[0], "depth_mode": depth_modes[0],
                       "predict_every": predict_everys[0], "cap": caps[0], "save_paths": save0})
            for mode in depth_modes:
                for md in min_depths:
                    for g in gammas:
                        for h in horizons:
                            for w in weights_pos:
                                for pe in predict_everys:
                                    for cap in caps:
                                        is_prim = (mode, md, pe, h, cap) == prim0
                                        jl.append({"cell_idx": ci, "count": count, "frac": frac,
                                                   "seed": seed, "rep": rep, "baseline": False,
                                                   "weight": w, "gamma": g, "horizon": h,
                                                   "min_depth": md, "depth_mode": mode,
                                                   "predict_every": pe, "cap": cap,
                                                   "save_paths": save0 and is_prim})
        per_cell_jobs.append(jl)
    return [j for wave in zip_longest(*per_cell_jobs) for j in wave if j is not None]
