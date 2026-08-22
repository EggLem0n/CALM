"""The evaluation run, end to end.

Resolve the sweep axes, build the job list, run the metrics phase across a process
pool, render the video phase, then write metadata and the summary tables. Each phase
is its own function so a caller can drive them separately.
"""
from __future__ import annotations

import contextlib
import csv
import json
import os
import sys
import time
from multiprocessing import Array, Manager, Value
from pathlib import Path


from calm.utils import console, parallel
from calm.utils.paths import run_dir as make_run_dir
from calm.utils.sweep import agent_count_sweep, frac_sweep, grid_cells
from .sweep_jobs import build_run_jobs
from .board import GridBoard
from .report import _typed_metric_rows, print_summaries
from .viz.table import save_metrics_table_png
from . import runner
from .runner import _init_worker, run_job
from .viz.sched import _build_vjobs, _prune_paths, _render_videos


def render_videos_only(args):
    """Standalone video (re-)render for an existing run: read its metadata.json + metrics.csv,
    rebuild the video jobs for the cells that fully completed, and encode straight from the
    cached sim paths (videos/.paths/). No metrics, no re-simulation. Idempotent -- safe to
    re-run after a crashed render."""
    run_dir = Path(args.from_run) if args.from_run else None
    if run_dir is None or not run_dir.is_dir():
        raise SystemExit("--video-only requires --from-run <existing run dir>")
    meta_path, csv_path = run_dir / "metadata.json", run_dir / "metrics.csv"
    if not meta_path.exists() or not csv_path.exists():
        raise SystemExit(f"missing metadata.json / metrics.csv under {run_dir}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    # Run-defining params come from metadata so cfg / any fallback re-sim is deterministic;
    # rendering knobs (dpi, cq, grid, movement_only, ...) keep whatever this CLI invocation set.
    args.seconds = int(meta["seconds"])
    args.base_seed = int(meta["base_seed"])
    args.center_value = float(meta["congestion_center_value"])
    args.step_value = float(meta["congestion_step_value"])
    gammas = list(meta["gammas"])
    prim = {"mode": meta["depth_modes"][0], "md": meta["min_depths"][0],
            "pe": meta["predict_everys"][0], "h": meta["horizons"][0], "cap": meta["caps"][0]}

    rows = _typed_metric_rows(csv_path)
    rows_by_cell = {}
    cells = {}
    for r in rows:
        ci = r["episode"]
        rows_by_cell.setdefault(ci, []).append(r)
        cells[ci] = (r["num_agents"], r["frac"], args.base_seed + ci)
    runs_per_cell = max(1, int(meta["runs"]) // max(1, int(meta["cells"])))
    completed = {ci for ci, rs in rows_by_cell.items() if len(rs) >= runs_per_cell}

    vjobs = _build_vjobs(cells, rows_by_cell, gammas, prim, completed)
    if not vjobs:
        raise SystemExit(f"no completed cells with a primary-config run to render in {run_dir}")
    (run_dir / "videos").mkdir(parents=True, exist_ok=True)
    vworkers = max(1, min(int(args.video_workers), len(vjobs)))
    print(f"[video-only] {run_dir}", flush=True)
    print(f"rendering videos for {len(vjobs)} completed cell(s) with {vworkers} worker(s) "
          f"(vanilla | best-lambda per gamma; from cached sim paths)...", flush=True)
    episode_videos = dict(meta.get("episode_videos", {}))
    vdone = [0]

    def absorb_v(res):
        episode_videos[str(res["cell_idx"])] = res["videos"]
        vdone[0] += 1
        print(f"  [{vdone[0]:>2}/{len(vjobs)}] cell {res['cell_idx']}: {len(res['videos'])} videos",
              flush=True)
    try:
        _render_videos(vjobs, args, str(run_dir), vworkers, absorb_v)
    except KeyboardInterrupt:
        print("\n[interrupted] video render stopped.", flush=True)
    if args.keep_paths != "best":          # --video-only defaults to leaving the cache intact
        _prune_paths(run_dir, vjobs, args.keep_paths)
    meta["episode_videos"] = episode_videos
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\nvideos -> {run_dir / 'videos'}", flush=True)


def run_sweep(args):
    """Run the whole evaluation grid for already-parsed `args`.

    Takes the parsed namespace rather than reading ``sys.argv``, so a notebook can drive
    a sweep without faking a command line. ``scripts/04_eval/evaluate.py`` is the launcher.
    """
    if args.video_only:
        render_videos_only(args)
        return
    args.horizon = max(1, min(10, int(args.horizon)))
    if args.predict_every is None:
        args.predict_every = max(1, 11 - args.horizon)   # full-depth lookahead (handoff section 4)
    # predict_every sweep axis: explicit --predict-everys, else the single --predict-every value.
    # Clamp to [1, 10] (>10 leaves late steps with no forecast frame; the solver clamps too).
    predict_everys = sorted({max(1, min(10, int(pe)))
                             for pe in (args.predict_everys or [args.predict_every])})
    # H (descent depth) sweep axis: explicit --horizons, else the single --horizon. Clamp [1, 10].
    horizons = sorted({max(1, min(10, int(h))) for h in (args.horizons or [args.horizon])})
    # congestion_max_penalty (clip) sweep axis: explicit --caps, else a single 0 (off). Clamp >= 0.
    caps = sorted({max(0.0, float(c)) for c in (args.caps or [0.0])})
    n_repeats = max(1, int(args.repeats))   # run every condition across this many distinct seeds
    counts, fracs = agent_count_sweep(args), frac_sweep(args)
    grid = grid_cells(args)
    weights = sorted(args.weights)
    weights_pos = [w for w in weights if w > 0]
    gammas, min_depths, depth_modes = args.gammas, args.min_depths, args.depth_modes
    axes = {"gammas": gammas, "min_depths": min_depths, "depth_modes": depth_modes,
            "horizons": horizons, "predict_everys": predict_everys, "caps": caps,
            "weights_pos": weights_pos, "n_repeats": n_repeats}

    # group comparison runs under reports/CALM_comparison/<yymmdd_hhmm>/ (attributable to this code)
    out_dir = make_run_dir("CALM_comparison", create=False)
    (out_dir / "videos").mkdir(parents=True, exist_ok=True)

    # per cell: 1 baseline (lambda=0)
    #   + (depth_mode x min_depth x gamma x horizon x lambda>0 x predict_every x cap) congestion runs
    cong_per_cell = (len(depth_modes) * len(min_depths) * len(gammas) * len(horizons)
                     * len(weights_pos) * len(predict_everys) * len(caps))
    runs_per_cell = 1 + cong_per_cell
    board_runs_per_cell = runs_per_cell * n_repeats   # a (count,frac) cell holds every config across all seeds
    n_vid = 0 if (args.no_video or not weights_pos) else len(grid) * len(gammas)
    print(f"grid: counts {counts} x fracs {fracs} | lambdas {weights} x gammas {gammas} "
          f"| min_depths {min_depths} x depth_modes {depth_modes} | horizons {horizons} "
          f"| predict_everys {predict_everys} | caps {caps} | repeats {n_repeats}")
    seeds_note = f" x {n_repeats} seeds" if n_repeats > 1 else ""
    print(f"  = {len(grid)} cells x {runs_per_cell} runs{seeds_note} = "
          f"{len(grid) * board_runs_per_cell} planner runs"
          f"{'' if args.no_video else f'  (+{n_vid} MP4s: vanilla vs best-lambda per gamma)'}", flush=True)
    print(f"output -> {out_dir}\n", flush=True)

    all_rows, episode_videos = [], {}
    workers = max(1, int(args.num_of_process))
    t0 = time.perf_counter()

    # metrics.csv is written INCREMENTALLY (one cell's rows appended as it finishes), so a
    # long 550-run sweep that gets interrupted still keeps every completed cell's metrics
    # (and the per-cell MP4s are likewise already on disk).
    csv_path = out_dir / "metrics.csv"
    csv_fields = ["episode", "num_agents", "frac", "gamma", "horizon", "min_depth", "depth_mode",
                  "weight", "predict_every", "cap", "seed", "rep", "deliveries", "energy", "energy_per_delivery",
                  "density_uniformity", "occ_cv", "mean_robot_cong", "p99_cong", "peak_cong",
                  "collisions", "preds", "wall_s"]
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        csv.DictWriter(fh, fieldnames=csv_fields).writeheader()

    total = len(grid)

    # WORK UNIT = one planner run (not a whole cell): there are runs_per_cell runs per cell,
    # times every cell, which vastly outnumbers the workers, so no worker ever sits idle until
    # the very tail. Jobs are interleaved across cells (round-robin) so every cell has work in
    # the first wave. The board keeps its current form (one mark per (count, frac) cell); a
    # cell flips to done once all of its own runs complete.
    videos_on = (not args.no_video) and bool(weights_pos)
    jobs = build_run_jobs(grid, args, axes, videos_on=videos_on)
    total_runs = len(jobs)

    # shared "currently running" map (pid -> label) so the board shows what every worker is on now.
    # Manager dict for the pool; a plain dict (exposed as the module global) for workers==1.
    if workers == 1:
        running = {}
        runner.set_running(running)
        mgr = None
    else:
        mgr = Manager()
        running = mgr.dict()

    counter = Value("i", 0)
    status = Array("b", total)               # per cell; 0=pending 1=running 2=done
    for i in range(total):
        status[i] = 1                        # every cell is in flight from the start (no idle worker)

    # Live grid board (unchanged form): a ✓/▶/· per (count, frac) cell, redrawn ~2x/s by a ticker
    # thread. --verbose falls back to a scrolling per-cell line.
    board = None if args.verbose else GridBoard(counts, fracs, total_runs, status, counter, running)

    cell_done = [0] * total
    rows_by_cell = {ci: [] for ci in range(total)}

    def absorb(res):
        ci, row = res["cell_idx"], res["row"]
        all_rows.append(row)
        rows_by_cell[ci].append(row)
        with open(csv_path, "a", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, fieldnames=csv_fields).writerow(row)
        cell_done[ci] += 1
        counter.value += 1
        if cell_done[ci] >= board_runs_per_cell:
            status[ci] = 2                   # all of this cell's runs (every seed) done -> check mark
            if board is None:
                nd = sum(1 for d in cell_done if d >= board_runs_per_cell)
                el = time.perf_counter() - t0
                print(f"[{nd:>2}/{total}] cell {ci} ({row['num_agents']} AMRs frac {row['frac']:.1f}) "
                      f"done | {counter.value}/{total_runs} runs | elapsed {el / 60:.1f}m", flush=True)

    interrupted = False
    job_args = [(job, args, str(out_dir)) for job in jobs]
    stack = contextlib.ExitStack()
    with stack:
        if board is not None:
            stack.enter_context(console.live(board))
        try:
            for res in parallel.imap_unordered(run_job, job_args, workers, star=True,
                                              initializer=_init_worker, initargs=(running,)):
                absorb(res)
        except KeyboardInterrupt:
            interrupted = True
            print("\n[interrupted] stopped; partial metrics.csv is kept.", flush=True)
    if board is not None:
        board.draw()
        print()                                  # drop below the board for the summary
    cells_done = sum(1 for d in cell_done if d >= board_runs_per_cell)

    # ---- videos (separate phase): per (cell, gamma) best-lambda for the PRIMARY config.
    # Encodes from the trajectories the metrics phase already cached (videos/.paths/) -- no
    # re-simulation. Only FULLY-completed cells are rendered, so an interrupted sweep still
    # gets videos for the cells that finished. Crash-isolated across cells.
    if not args.no_video and weights_pos:
        prim = {"mode": depth_modes[0], "md": min_depths[0], "pe": predict_everys[0],
                "h": horizons[0], "cap": caps[0]}
        cells = {ci: (count, frac, args.base_seed + ci) for ci, (count, frac) in enumerate(grid)}
        completed = {ci for ci in range(total) if cell_done[ci] >= board_runs_per_cell}
        vjobs = _build_vjobs(cells, rows_by_cell, gammas, prim, completed)
        # The video phase opens a GPU NVENC encoder per worker; consumer GPUs cap concurrent NVENC
        # sessions (too many -> a worker dies). Cap it SEPARATELY from the metrics-phase
        # --num_of_process (which only does inference, no NVENC). Never exceed the number of videos.
        vworkers = max(1, min(int(args.video_workers), len(vjobs))) if vjobs else 0
        print(f"\nrendering videos for {len(vjobs)} completed cell(s) with {vworkers} worker(s) "
              f"(vanilla | best-lambda per gamma; from cached sim paths)...", flush=True)
        vdone = [0]

        def absorb_v(res):
            episode_videos[res["cell_idx"]] = res["videos"]
            vdone[0] += 1
            print(f"  [{vdone[0]:>2}/{len(vjobs)}] cell {res['cell_idx']}: {len(res['videos'])} videos",
                  flush=True)
        try:
            _render_videos(vjobs, args, str(out_dir), vworkers, absorb_v)
        except KeyboardInterrupt:
            print("\n[interrupted] video phase stopped (metrics already saved).", flush=True)
        _prune_paths(out_dir, vjobs, args.keep_paths)

    # ---- metadata.json ----
    (out_dir / "metadata.json").write_text(json.dumps({
        # "grid_eval" is kept verbatim: it is the tool tag every finished report in
        # reports/CALM_comparison/ already carries. The module was since split into
        # pipeline/runner/report, but changing the tag would split the run history.
        "tool": "grid_eval", "solver": "pibt_lifelong",
        "counts": counts, "fracs": fracs, "weights": weights,
        "gammas": gammas, "min_depths": min_depths, "depth_modes": depth_modes,
        "horizon": args.horizon, "horizons": horizons, "predict_every": args.predict_every,
        "predict_everys": predict_everys, "caps": caps,
        "seconds": args.seconds, "base_seed": args.base_seed, "repeats": n_repeats,
        "congestion_center_value": args.center_value, "congestion_step_value": args.step_value,
        "cells": total, "cells_completed": cells_done, "interrupted": interrupted,
        "runs": total_runs,
        "elapsed_min": (time.perf_counter() - t0) / 60.0,
        "episode_videos": episode_videos,
    }, indent=2), encoding="utf-8")

    print_summaries(all_rows, weights, weights_pos, grid, rows_by_cell, cells_done, total)

    # save the by-lambda table as an image too (out_dir; rides along when moved below)
    if all_rows:
        try:
            save_metrics_table_png(all_rows, weights, cells_done, out_dir / "metrics_table.png")
        except Exception as exc:  # noqa: BLE001
            print(f"[metrics table png skipped] {exc!r}")

    # per-run write-up + glossary, dropped into this run's dated folder (never fatal)
    summary_md = None
    if all_rows:
        try:
            from .make_analysis_summary import write_run_reports
            summary_md, _ = write_run_reports(out_dir)
        except Exception as exc:  # noqa: BLE001
            print(f"[analysis summary skipped] {exc!r}")

    print(f"\nmetrics -> {out_dir / 'metrics.csv'}")
    print(f"table   -> {out_dir / 'metrics_table.png'}")
    if summary_md is not None:
        print(f"summary -> {summary_md}")
        print(f"glossary-> {out_dir / 'metrics_glossary.md'}")
    if not args.no_video:
        print(f"videos  -> {out_dir / 'videos'}")
    print(f"report  -> {out_dir}")
    print(f"{'INTERRUPTED — ' if interrupted else ''}elapsed "
          f"{(time.perf_counter() - t0) / 60.0:.1f} min "
          f"({cells_done}/{total} cells)")
    if interrupted:
        # hard-exit so a half-torn-down process pool can't hang the interpreter at shutdown
        sys.stdout.flush()
        os._exit(130)
