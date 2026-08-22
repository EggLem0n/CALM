"""One planner run: set it up, solve it, score it, and cache its trajectory.

This is the work unit the evaluator parallelises over -- a single grid point
(cell x gamma x lambda x depth mode x horizon x predict_every x cap x seed).
``run_job`` is what a worker process actually calls.

The env, the base config and the SimVP predictor are per-process singletons: built
once per worker and reused across every run that worker handles.
"""
from __future__ import annotations

import os
import time
import warnings
from pathlib import Path

import numpy as np

from calm import pibt as mapf
from calm.utils import parallel
from calm.pibt import factory_map_generator as fmg


def get_predictor():
    """Per-process SimVP predictor singleton. Inference always runs on the GPU."""
    global _PREDICTOR
    if _PREDICTOR is None:
        # local import: keeps torch out of the vanilla-only paths
        from calm.forecast.predict import CongestionPredictor
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError(
                "GPU inference is required but CUDA is not available. Install a CUDA build of torch "
                "(e.g. pip install torch --index-url https://download.pytorch.org/whl/cu121).")
        _PREDICTOR = CongestionPredictor(device="cuda")
    return _PREDICTOR


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def evaluate(paths, summary, walkable, config, wall):
    H, W = walkable.shape[:2]
    ap = mapf.paths_to_agent_positions(paths, config.max_time)            # (T, N, 2)
    T = ap.shape[0]
    cong = mapf.build_additive_congestion_label_sequence(
        ap, H, W, config.congestion_center_value, config.congestion_step_value)

    tidx = np.arange(T)[:, None]
    robot_cong = cong[tidx, ap[..., 1], ap[..., 0]]                       # (T, N)
    energy = int(np.abs(np.diff(ap.astype(np.int32), axis=0)).sum())      # total cells moved
    deliveries = int(summary["total_completed_deliveries"])

    occ = mapf.build_occupancy_sequence(ap, H, W).astype(np.float64)      # (T, H, W)
    occ_mean = occ.mean(axis=0)[walkable]
    total = occ_mean.sum()
    if total > 0:
        p = occ_mean[occ_mean > 0] / total
        uniformity = float(-(p * np.log(p)).sum() / np.log(int(walkable.sum())))
        occ_cv = float(occ_mean.std() / (occ_mean.mean() + 1e-12))
    else:
        uniformity, occ_cv = 0.0, 0.0

    return {
        "deliveries": deliveries,
        "energy": energy,
        "energy_per_delivery": (energy / deliveries) if deliveries else float("nan"),
        "density_uniformity": uniformity,
        "occ_cv": occ_cv,
        "mean_robot_cong": float(robot_cong.mean()),
        "p99_cong": float(np.percentile(cong, 99.0)),
        "peak_cong": float(cong.max()),
        "collisions": int(mapf.compute_collision_count(ap)),
        "preds": int(summary["congestion_prediction_count"]),
        "wall_s": float(wall),
    }


def solve(weight, env, config, starts, predict_every, *,
          gamma=0.73, horizon=10, min_depth=2, depth_mode="peaked", max_penalty=0.0):
    walkable = np.asarray(env["walkable_map"]).astype(bool)
    pickup = mapf.walkable_points(env, "pickup_points", walkable)
    delivery = mapf.walkable_points(env, "delivery_points", walkable)
    predictor = get_predictor() if weight > 0 else None
    t0 = time.perf_counter()
    paths, summary = mapf.plan_pibt_repeated_tasks(
        starts, pickup, delivery, walkable, config,
        pickup_point_groups=mapf.normalize_point_groups(env.get("pickup_point_groups")),
        delivery_point_groups=mapf.normalize_point_groups(env.get("delivery_point_groups")),
        congestion_predictor=predictor, congestion_weight=weight,
        congestion_gamma=gamma, congestion_horizon=horizon,
        congestion_min_depth=min_depth, congestion_depth_mode=depth_mode,
        congestion_max_penalty=max_penalty, predict_every=predict_every,
    )
    return paths, summary, evaluate(paths, summary, walkable, config, time.perf_counter() - t0)


# ---------------------------------------------------------------------------
# work unit = ONE planner run (a single grid point), so every worker stays busy:
# the total run count vastly outnumbers the workers. env / base config are
# per-process singletons (built once per worker, reused across that worker's runs).
# ---------------------------------------------------------------------------
_ENV = None
_BASE_CFG = None
_RUNNING = None        # pid -> label of the run this worker is doing now (for the live board)


def _job_label(job):
    """Short label of a run: 'n300 f0.0  peaked md1 g0.61 H10 λ0.5 pe1 cap0' / '... baseline'."""
    head = f"n{job['count']:>3} f{job['frac']:.1f}"
    if job["baseline"]:
        return f"{head}  baseline"
    return (f"{head}  {job['depth_mode']:>9} md{job['min_depth']} g{job['gamma']:g} "
            f"H{job['horizon']:g} λ{job['weight']:g} pe{job['predict_every']:g} cap{job['cap']:g}")


# A congestion run's config = everything that distinguishes it within a cell EXCEPT lambda
# (the swept congestion weight). Used to group the per-config summary tables.
_CFG_AXES = ("depth_mode", "min_depth", "gamma", "horizon", "predict_every", "cap")


def _cfg_key(row):
    return tuple(row[a] for a in _CFG_AXES)


def _cfg_label(combo):
    mode, md, g, h, pe, cap = combo
    return f"mode={mode} md={md} g={g:g} H={h:g} pe={pe:g} cap={cap:g}"


def get_env():
    global _ENV
    if _ENV is None:
        _ENV = fmg.build_factory_map()
    return _ENV


def get_base_cfg():
    global _BASE_CFG
    if _BASE_CFG is None:
        _BASE_CFG = mapf.load_config()
    return _BASE_CFG


def _cell_cfg(count, frac, seed, args):
    """Deterministic per-cell config (same seed -> same starts & run, sharable across runs)."""
    return get_base_cfg().replace(
        num_agents=count, distributed_fraction=frac, seed=seed, max_time=args.seconds,
        congestion_center_value=args.center_value, congestion_step_value=args.step_value,
        show_planning_progress=False)


# ---------------------------------------------------------------------------
# Saved-simulation paths: the video phase no longer re-runs PIBT. The metrics
# phase already simulates every candidate run (baseline + the primary-config
# gamma x lambda grid the videos pick from), so each such run dumps its agent
# trajectory once (compact int16, zlib-compressed) into videos/.paths/. The
# video phase then just loads them. Saving piggybacks on a sim that ran anyway
# (~tens of ms each), so it adds <1% to the metrics phase yet removes the whole
# (1 + #gammas) x #cells re-simulation the old video phase did.
# ---------------------------------------------------------------------------
def _paths_file(vdir, ci, gamma, weight, baseline):
    """Per-run trajectory cache file under <out>/videos/.paths/."""
    d = Path(vdir) / ".paths"
    if baseline:
        return d / f"ep{ci:03d}_baseline.npz"
    return d / f"ep{ci:03d}_g{gamma:g}_w{weight:g}.npz"


def _save_run_paths(file, paths, summary, max_t):
    """Persist one run's full trajectory (T+1, N, 2) int16 + its task summary, atomically."""
    file.parent.mkdir(parents=True, exist_ok=True)
    pos = mapf.paths_to_agent_positions(paths, max_t).astype(np.int16)
    tmp = file.with_name(file.name + ".tmp")
    with open(tmp, "wb") as fh:            # write to a handle so numpy doesn't re-append ".npz"
        np.savez_compressed(fh, pos=pos, summary=np.array(summary, dtype=object))
    tmp.replace(file)


def _pos_to_paths(pos):
    """Rebuild MACPF's List[List[(x, y)]] paths from a saved (T+1, N, 2) array."""
    pos = np.asarray(pos)
    T, N = pos.shape[0], pos.shape[1]
    return [[(int(pos[t, i, 0]), int(pos[t, i, 1])) for t in range(T)] for i in range(N)]


def _load_run_paths(vdir, ci, *, gamma, weight, baseline):
    """Return (paths, summary) from the trajectory cache, or None if absent/corrupt."""
    f = _paths_file(vdir, ci, gamma, weight, baseline)
    if not f.exists():
        return None
    try:
        with np.load(f, allow_pickle=True) as d:
            return _pos_to_paths(d["pos"]), d["summary"].item()
    except Exception:                                   # noqa: BLE001 (corrupt/partial -> re-sim)
        return None


def run_job(job, args, out_dir_str=None):
    """One planner run -> its CSV row. Candidate runs (job['save_paths']) also dump their
    trajectory to videos/.paths/ so the video phase can encode without re-simulating."""
    if _RUNNING is not None:
        _RUNNING[os.getpid()] = _job_label(job)        # tell the board what this worker is on now
    try:
        env = get_env()
        walkable = np.asarray(env["walkable_map"]).astype(bool)
        cfg = _cell_cfg(job["count"], job["frac"], job["seed"], args)
        starts, _ = mapf.select_start_goal_pairs(env, walkable, cfg)
        weight = 0.0 if job["baseline"] else job["weight"]
        paths, summary, m = solve(weight, env, cfg, starts, job["predict_every"],
                                  gamma=job["gamma"], horizon=job["horizon"],
                                  min_depth=job["min_depth"], depth_mode=job["depth_mode"],
                                  max_penalty=job["cap"])
        if job.get("save_paths") and out_dir_str is not None:
            try:
                _save_run_paths(_paths_file(Path(out_dir_str) / "videos", job["cell_idx"],
                                            job["gamma"], weight, job["baseline"]),
                                paths, summary, args.seconds)
            except Exception as exc:                    # noqa: BLE001 (cache is best-effort)
                print(f"[paths not cached] cell {job['cell_idx']}: {exc!r}", flush=True)
        if job["baseline"]:
            row = {"episode": job["cell_idx"], "num_agents": job["count"], "frac": job["frac"],
                   "gamma": "", "horizon": job["horizon"], "min_depth": "", "depth_mode": "baseline",
                   "weight": 0.0, "predict_every": "", "cap": "", "seed": job["seed"],
                   "rep": job["rep"], **m}
        else:
            row = {"episode": job["cell_idx"], "num_agents": job["count"], "frac": job["frac"],
                   "gamma": job["gamma"], "horizon": job["horizon"], "min_depth": job["min_depth"],
                   "depth_mode": job["depth_mode"], "weight": job["weight"],
                   "predict_every": job["predict_every"], "cap": job["cap"], "seed": job["seed"],
                   "rep": job["rep"], **m}
        return {"cell_idx": job["cell_idx"], "row": row}
    finally:
        if _RUNNING is not None:
            _RUNNING.pop(os.getpid(), None)


def set_running(running):
    """Point the module at the shared pid->label map (workers==1 path).

    With a pool, ``_init_worker`` does this inside each worker; running in-process
    there is no initializer, so the caller sets it here instead.
    """
    global _RUNNING
    _RUNNING = running


def _init_worker(running=None):
    global _RUNNING
    _RUNNING = running
    parallel.ignore_sigint()                       # main process handles Ctrl+C
    warnings.filterwarnings("ignore", message=r".*pkg_resources is deprecated.*")
