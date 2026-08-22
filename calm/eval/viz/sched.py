"""Deciding which videos to make, and rendering them without one crash losing the rest.

Encoding opens a GPU NVENC session per worker and consumer GPUs cap how many can be
open at once; too many and a worker simply dies. So the scheduler backs off a worker
and retries the survivors, then falls back to one cell per fresh single-worker pool --
a hard crash is contained to the cell that caused it.
"""
from __future__ import annotations

import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

from ..runner import _init_worker, _paths_file
from .video import video_job


def _build_vjobs(cells, rows_by_cell, gammas, prim, completed):
    """One video job per COMPLETED cell that has at least one primary-config run.
    `cells` maps cell_idx -> (count, frac, seed); `prim` carries the primary config
    (mode/md/pe/h/cap). Each gamma's lambda is the one that maximised deliveries."""
    vjobs = []
    for ci in sorted(cells):
        if ci not in completed:
            continue
        count, frac, seed = cells[ci]
        best_by_gamma = {}
        for g in gammas:
            cand = [r for r in rows_by_cell.get(ci, [])
                    if r.get("rep", 0) == 0                       # videos render rep 0 only
                    and r["depth_mode"] == prim["mode"] and r["min_depth"] == prim["md"]
                    and r["gamma"] == g and r["predict_every"] == prim["pe"]
                    and r["horizon"] == prim["h"] and r["cap"] == prim["cap"]]
            if cand:
                best_by_gamma[g] = max(cand, key=lambda r: r["deliveries"])["weight"]
        if best_by_gamma:
            vjobs.append({"cell_idx": ci, "count": count, "frac": frac, "seed": seed,
                          "best_by_gamma": best_by_gamma, "prim_mode": prim["mode"],
                          "prim_md": prim["md"], "prim_pe": prim["pe"],
                          "prim_cap": prim["cap"], "horizon": prim["h"]})
    return vjobs


def _render_videos(vjobs, args, out_dir_str, vworkers, on_done):
    """Crash-isolated video rendering. A worker death (e.g. a consumer-GPU NVENC session
    limit) only loses the cell that was on it: survivors are retried with one fewer worker,
    and the single-worker tail puts each cell in its own pool so a hard crash can't poison
    the rest. Per-cell soft failures are logged and skipped; finished cells keep their MP4s."""
    if not vjobs:
        return
    vworkers = max(1, min(int(vworkers), len(vjobs)))
    remaining = list(vjobs)
    # Stage 1: multi-worker rounds. On a worker death, back off a worker and retry survivors.
    while remaining and vworkers > 1:
        batch, remaining = remaining, []
        vpool = ProcessPoolExecutor(max_workers=vworkers, initializer=_init_worker)
        broke = False
        try:
            fut_to_job = {vpool.submit(video_job, vj, args, out_dir_str): vj for vj in batch}
            for fut in as_completed(fut_to_job):
                vj = fut_to_job[fut]
                try:
                    on_done(fut.result())
                except BrokenProcessPool:
                    broke = True
                    remaining.append(vj)               # survivor: retry next round
                except Exception as exc:               # noqa: BLE001 (one bad cell, keep the rest)
                    print(f"  [skip] cell {vj['cell_idx']}: video failed ({exc!r})", flush=True)
        finally:
            vpool.shutdown(wait=False, cancel_futures=True)
        if broke and remaining:
            vworkers -= 1
            print(f"  [recover] a video worker died; retrying {len(remaining)} cell(s) "
                  f"with {vworkers} worker(s)...", flush=True)
    # Stage 2: one cell per fresh single-worker pool -> a hard crash is contained to that cell.
    for vj in remaining:
        vpool = ProcessPoolExecutor(max_workers=1, initializer=_init_worker)
        try:
            on_done(vpool.submit(video_job, vj, args, out_dir_str).result())
        except Exception as exc:                       # noqa: BLE001 (incl. BrokenProcessPool)
            print(f"  [skip] cell {vj['cell_idx']}: video failed ({exc!r})", flush=True)
        finally:
            vpool.shutdown(wait=False, cancel_futures=True)


def _prune_paths(out_dir, vjobs, keep):
    """Trim the videos/.paths/ trajectory cache after rendering.
    keep='all' keeps every candidate (re-pick a different lambda later via --video-only),
    'none' drops the whole cache, 'best' keeps only the baseline + best-lambda files that
    were actually encoded (final footprint == the old 'only the best' behaviour)."""
    pdir = Path(out_dir) / "videos" / ".paths"
    if not pdir.exists() or keep == "all":
        return
    if keep == "none":
        shutil.rmtree(pdir, ignore_errors=True)
        return
    vdir = Path(out_dir) / "videos"
    keepset = set()
    for vj in vjobs:
        ci = vj["cell_idx"]
        keepset.add(_paths_file(vdir, ci, None, 0.0, True).name)
        for g, w in vj["best_by_gamma"].items():
            keepset.add(_paths_file(vdir, ci, g, w, False).name)
    for f in pdir.glob("*.npz"):
        if f.name not in keepset:
            f.unlink()
