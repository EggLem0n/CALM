"""Building one comparison video: render the panels, then stitch them.

A video is assembled from trajectories the metrics phase already cached, so nothing
here re-runs the solver. Panels are produced by ``calm.pibt.viz.animate_paths`` and by
the congestion-heatmap renderer, then combined with ffmpeg (``hstack`` for two,
``xstack`` for a grid).
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np

from calm import pibt as mapf
from ..runner import _cell_cfg, _load_run_paths, get_env, get_predictor, solve


# ---------------------------------------------------------------------------
# movement video: calm.pibt.viz.animate_paths, one panel per run;
# vanilla | congestion are hstacked with ffmpeg.
# ---------------------------------------------------------------------------
def _anim_config(base, args):
    """Config carrying the animation knobs + the adjustable robot/line sizes."""
    return base.replace(
        animation_subframes=args.anim_subframes,
        animation_interval_ms=35,
        show_planned_routes=args.planned_routes,
        viz_robot_size=args.robot_size,
        viz_start_size=args.robot_size,
        viz_route_linewidth=args.route_linewidth,
        viz_planned_route_linewidth=args.planned_route_linewidth,
        viz_target_size=args.target_size,
        show_planning_progress=False,
        viz_video_dpi=args.video_dpi,
        viz_video_cq=args.video_cq,
    )


def _animate_scenario(env, starts, paths, summary, anim_cfg, out_mp4, tmp_dir):
    """Render ONE scenario with calm.pibt.viz.animate_paths -> out_mp4."""
    from calm.pibt import viz
    tmp_dir.mkdir(parents=True, exist_ok=True)
    viz.animate_paths(env, paths, list(starts), list(starts), tmp_dir, anim_cfg, task_summary=summary)
    src = tmp_dir / "classical_mapf_animation.mp4"
    if not src.exists():                       # ffmpeg missing -> animate_paths fell back to gif
        src = tmp_dir / "classical_mapf_animation.gif"
        out_mp4 = out_mp4.with_suffix(".gif")
    if out_mp4.exists():
        out_mp4.unlink()
    shutil.move(str(src), str(out_mp4))
    return out_mp4


def _hstack(left_mp4, right_mp4, out_mp4, cq=15):
    """Combine two equal-size clips side by side (ffmpeg hstack). Re-encodes on the GPU
    (NVENC H.264) at high quality so this second pass barely adds generation loss; falls
    back to high-quality CPU x264 if NVENC is unavailable on this machine."""
    from calm.utils.backend import ffmpeg_exe
    ff = ffmpeg_exe()
    base = [ff, "-y", "-loglevel", "error", "-i", str(left_mp4), "-i", str(right_mp4),
            "-filter_complex", "[0:v][1:v]hstack=inputs=2"]
    nvenc = ["-c:v", "h264_nvenc", "-preset", "p7", "-tune", "hq", "-rc", "vbr",
             "-cq", str(cq), "-b:v", "0", "-pix_fmt", "yuv420p", str(out_mp4)]
    x264 = ["-c:v", "libx264", "-preset", "slow", "-crf", str(cq + 1),
            "-pix_fmt", "yuv420p", str(out_mp4)]
    try:
        subprocess.run(base + nvenc, check=True)
    except subprocess.CalledProcessError:
        subprocess.run(base + x264, check=True)


def _predict_congestion_sequence(cong, predictor, predict_every, *, chunk=64):
    """Replay the SimVP predictor over a trajectory's per-step congestion to get the
    predicted-congestion FIELD the planner would consult at each step -- usable for the
    vanilla run too (the predictor never ran during vanilla planning, but we can still
    show what it WOULD have forecast).

    ``cong`` is (T,H,W) raw instantaneous congestion. The predictor is re-run every
    ``predict_every`` steps on the trailing 10 frames (the planner's cadence); the first
    9 steps have no 10-frame history yet, so they stay zero. Predictions are batched.
    Returns (T,H,W) raw predicted congestion."""
    cong = np.asarray(cong, np.float32)
    T = cong.shape[0]
    out = np.zeros_like(cong)
    if T < 10:
        return out
    step = max(1, int(predict_every))
    bases = list(range(9, T, step))                       # each base sees frames base-9..base
    futs = []
    for s in range(0, len(bases), chunk):                 # batch to bound GPU memory
        pasts = np.stack([cong[b - 9:b + 1] for b in bases[s:s + chunk]])[:, :, None]  # (B,10,1,H,W)
        futs.append(np.asarray(predictor.predict(pasts))[:, :, 0])                      # (B,10,H,W)
    fut = np.concatenate(futs, axis=0)
    for i, b in enumerate(bases):
        nb = bases[i + 1] if i + 1 < len(bases) else T
        for t in range(b, nb):
            out[t] = fut[i, min(t - b, 9)]                # forecast for this step from base b
    return out


def _xstack_grid(panels, out_mp4, *, cols, rows, cell_w, cell_h, cq=15):
    """Tile equal-cell MP4s into a cols x rows grid (ffmpeg xstack). Each input is fps-
    synced and letterboxed into a cell_w x cell_h cell (no distortion), so panels of
    different native aspect (movement 10x8 vs heatmap 8x5) line up. `panels` is row-major.
    Re-encodes on the GPU (NVENC), falling back to CPU x264."""
    from calm.utils.backend import ffmpeg_exe
    ff = ffmpeg_exe()
    ins = []
    for p in panels:
        ins += ["-i", str(p)]
    pre = [
        f"[{i}:v]fps=30,scale={cell_w}:{cell_h}:force_original_aspect_ratio=decrease,"
        f"pad={cell_w}:{cell_h}:({cell_w}-iw)/2:({cell_h}-ih)/2:color=black,setsar=1[v{i}]"
        for i in range(len(panels))
    ]
    layout = "|".join(f"{(i % cols) * cell_w}_{(i // cols) * cell_h}" for i in range(len(panels)))
    chain = (";".join(pre) + ";" + "".join(f"[v{i}]" for i in range(len(panels)))
             + f"xstack=inputs={len(panels)}:layout={layout}[v]")
    base = [ff, "-y", "-loglevel", "error", *ins, "-filter_complex", chain, "-map", "[v]"]
    nvenc = ["-c:v", "h264_nvenc", "-preset", "p7", "-tune", "hq", "-rc", "vbr",
             "-cq", str(cq), "-b:v", "0", "-pix_fmt", "yuv420p", str(out_mp4)]
    x264 = ["-c:v", "libx264", "-preset", "slow", "-crf", str(cq + 1),
            "-pix_fmt", "yuv420p", str(out_mp4)]
    try:
        subprocess.run(base + nvenc, check=True)
    except subprocess.CalledProcessError:
        subprocess.run(base + x264, check=True)


def video_job(vjob, args, out_dir_str):
    """Render one MP4 per gamma for a cell (primary config, best lambda). Default is a
    cols x rows composite per gamma -- row 1 vanilla, row 2 the congestion-aware run;
    col 1 movement, col 2 the ACTUAL congestion field the AMRs produced, col 3 the
    SimVP-PREDICTED field (replayed post-hoc, so the vanilla run gets one too). With
    --movement-only it falls back to the plain vanilla|congestion movement clip.
    Loads the baseline + each chosen config's trajectory from the metrics-phase cache
    (videos/.paths/); only if a cache file is missing does it deterministically re-sim."""
    from calm.dataset.viz.heatmap_video import render_congestion_video
    env = get_env()
    walkable = np.asarray(env["walkable_map"]).astype(bool)
    H, W = walkable.shape[:2]
    obstacle = np.asarray(env.get("obstacle_map", (~walkable).astype(np.uint8)), dtype=np.float32)
    cfg = _cell_cfg(vjob["count"], vjob["frac"], vjob["seed"], args)
    starts, _ = mapf.select_start_goal_pairs(env, walkable, cfg)
    prim_mode, prim_md, horizon = vjob["prim_mode"], vjob["prim_md"], vjob["horizon"]
    prim_pe = vjob.get("prim_pe", args.predict_every)   # the swept predict_every the videos use
    prim_cap = vjob.get("prim_cap", 0.0)                # the swept congestion clip the videos use
    best_by_gamma = vjob["best_by_gamma"]

    vdir = Path(out_dir_str) / "videos"
    tmp = vdir / f".tmp_ep{vjob['cell_idx']:03d}"
    anim_cfg = _anim_config(cfg, args)
    vs = args.video_seconds if 0 < args.video_seconds < args.seconds else (args.seconds + 1)
    clip = lambda paths: [p[:vs + 1] for p in paths]
    cv, sv = cfg.congestion_center_value, cfg.congestion_step_value
    composite = not args.movement_only
    predictor = get_predictor() if composite else None
    name_for = lambda g, w: (f"ep{vjob['cell_idx']:03d}_n{vjob['count']}_f{vjob['frac']:.1f}_"
                             f"{prim_mode}_md{prim_md}_g{g:g}_w{w:g}.mp4")

    # vanilla (lambda 0) -- shared across every gamma in this cell.
    # Prefer the metrics-phase trajectory cache; re-sim only if it's missing.
    loaded = _load_run_paths(vdir, vjob["cell_idx"], gamma=None, weight=0.0, baseline=True)
    if loaded is not None:
        base_paths, base_summary = loaded
    else:
        base_paths, base_summary, _ = solve(
            0.0, env, cfg, starts, prim_pe,
            gamma=next(iter(best_by_gamma)), horizon=horizon, min_depth=prim_md,
            depth_mode=prim_mode, max_penalty=prim_cap)
    van_move = _animate_scenario(env, starts, clip(base_paths), base_summary,
                                 anim_cfg, tmp / "van_move.mp4", tmp / "vm")
    ap_base = mapf.paths_to_agent_positions(base_paths, vs)
    van_act = van_pred = None
    if composite:
        van_act = mapf.build_additive_congestion_label_sequence(ap_base, H, W, cv, sv)
        van_pred = _predict_congestion_sequence(van_act, predictor, prim_pe)

    # each gamma's congestion-aware run (movement now; fields kept for a shared colour scale)
    runs = []
    for g in sorted(best_by_gamma):
        w = best_by_gamma[g]
        loaded = _load_run_paths(vdir, vjob["cell_idx"], gamma=g, weight=w, baseline=False)
        if loaded is not None:
            cpaths, csumm = loaded
        else:
            cpaths, csumm, _ = solve(w, env, cfg, starts, prim_pe,
                                     gamma=g, horizon=horizon, min_depth=prim_md,
                                     depth_mode=prim_mode, max_penalty=prim_cap)
        app_move = _animate_scenario(env, starts, clip(cpaths), csumm, anim_cfg,
                                     tmp / f"app_move_g{g:g}.mp4", tmp / f"am{g:g}")
        d = {"g": g, "w": w, "move": app_move}
        if composite:
            d["ap"] = mapf.paths_to_agent_positions(cpaths, vs)
            d["act"] = mapf.build_additive_congestion_label_sequence(d["ap"], H, W, cv, sv)
            d["pred"] = _predict_congestion_sequence(d["act"], predictor, prim_pe)
        runs.append(d)

    names = []
    if not composite:                                   # plain vanilla|congestion movement clip
        for d in runs:
            name = name_for(d["g"], d["w"])
            _hstack(van_move, d["move"], vdir / name, args.video_cq)
            names.append(name)
        shutil.rmtree(tmp, ignore_errors=True)
        return {"cell_idx": vjob["cell_idx"], "videos": names}

    # one colour scale across ALL heatmap fields in the cell, so every panel is comparable
    fields = [van_act, van_pred] + [d["act"] for d in runs] + [d["pred"] for d in runs]
    cat = np.concatenate([f[f > 0].ravel() for f in fields])
    vmax = float(np.percentile(cat, args.heatmap_vmax_pct)) if cat.size else 1.0
    hfps = max(1, round(30 / max(1, args.anim_subframes)))   # match the movement clip's duration
    hm = lambda field, ap, out, label: render_congestion_video(
        field, ap, obstacle, out, fps=hfps, dpi=args.video_dpi, vmax=vmax,
        label=label, title_fmt="t={t}/{T}")

    van_act_mp4, van_pred_mp4 = tmp / "van_act.mp4", tmp / "van_pred.mp4"
    hm(van_act, ap_base, van_act_mp4, "vanilla λ0  actual")
    hm(van_pred, ap_base, van_pred_mp4, "vanilla λ0  predicted")
    cols, rows = (2, 3) if args.video_grid == "2x3" else (3, 2)
    for d in runs:
        g, w = d["g"], d["w"]
        lab = f"{prim_mode} md{prim_md} g{g:g} λ{w:g}"
        app_act_mp4, app_pred_mp4 = tmp / f"app_act_g{g:g}.mp4", tmp / f"app_pred_g{g:g}.mp4"
        hm(d["act"], d["ap"], app_act_mp4, lab + "  actual")
        hm(d["pred"], d["ap"], app_pred_mp4, lab + "  predicted")
        if cols == 2:                                   # 2 cols x 3 rows: col=condition, row=view
            panels = [van_move, d["move"], van_act_mp4, app_act_mp4, van_pred_mp4, app_pred_mp4]
        else:                                           # 3 cols x 2 rows: row=condition, col=view
            panels = [van_move, van_act_mp4, van_pred_mp4, d["move"], app_act_mp4, app_pred_mp4]
        _xstack_grid(panels, vdir / name_for(g, w), cols=cols, rows=rows,
                     cell_w=800, cell_h=500, cq=args.video_cq)
        names.append(name_for(g, w))
    shutil.rmtree(tmp, ignore_errors=True)
    return {"cell_idx": vjob["cell_idx"], "videos": names}
