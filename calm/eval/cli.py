"""The flag definitions for the grid evaluation.

Which flags exist is this package's business, so the definitions live here; the
parsing itself is done by ``scripts/04_eval/evaluate.py``, which owns the parser. Importing
this module pulls in neither numpy, torch, nor matplotlib, so a launcher can show
``--help`` without paying for any of them.
"""
from __future__ import annotations

import argparse

from calm import pibt as mapf


# ---------------------------------------------------------------------------
# CLI (generate.py-style argument names)
# ---------------------------------------------------------------------------
DESCRIPTION = ("Grid A/B eval of congestion-aware PIBT over AMR count x dispersion frac x "
               "congestion weight (lambda) x depth gamma (+ optional min_depth / depth_mode "
               "ablations), with a per-(cell, gamma) vanilla-vs-best-lambda MP4.")


def add_eval_args(ap: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add every grid-evaluation flag to `ap` and return it.

    Defaults for the solver flags come from ``configs/default.yaml``, so the file stays
    the single source of truth even for values the CLI can override.
    """
    base = mapf.load_config()
    ap.add_argument("--num_of_process", type=int, default=1,
                    help="Parallel cell jobs. NOTE: lambda>0 runs use the GPU; many processes "
                         "share one GPU (each loads the 220MB model). Raise only if VRAM allows.")
    ap.add_argument("--base-seed", type=int, default=42, help="cell i uses base_seed + i (shared across lambdas).")
    ap.add_argument("--repeats", type=int, default=1,
                    help="repeat the WHOLE grid across this many distinct seeds (default 1 = old "
                         "behaviour). Repeat r of cell i uses seed base_seed + r*cells + i, so rep 0 "
                         "reproduces the single-seed run exactly and extra reps add seed diversity. "
                         "Within one (cell, rep) the seed is still shared across lambdas, so each "
                         "repeat stays a controlled A/B. Every metric table then averages over seeds "
                         "and the +-sd column shows the across-seed spread. Videos use rep 0 only.")
    ap.add_argument("--seconds", type=int, default=900,
                    help="episode length in steps (metrics use the full length; dataset used 1800).")
    ap.add_argument("--video-seconds", type=int, default=900,
                    help="render only the first N steps in the MP4s (0 or >= --seconds = full). "
                         "Default 900 = full when --seconds is also 900.")
    ap.add_argument("--weights", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0],
                    help="congestion-weight (lambda) axis; 0 = vanilla PIBT (computed once per cell).")
    ap.add_argument("--gammas", type=float, nargs="+", default=[0.73],
                    help="depth-weight peak r axis (handoff usable band [0.607, 0.730]); "
                         "default [0.73] = single value (old behaviour). e.g. 0.61 0.66 0.70 0.73")
    ap.add_argument("--horizon", type=int, default=10, help="H: max descent depth read (<=10). "
                    "Single value; also the basis for the default --predict-every (11 - H).")
    ap.add_argument("--horizons", type=int, nargs="+", default=None,
                    help="sweep axis for H (max descent depth), e.g. '4 7 10'. When given, every "
                         "congestion run is also solved at each H (multiplies the grid). Omitted = the "
                         "single --horizon. Each value clamped to [1, 10]. NOTE: --predict-every's "
                         "default (11 - H) is derived from the single --horizon, not per-swept-H; pass "
                         "--predict-everys explicitly to sweep that too.")
    ap.add_argument("--min-depths", type=int, nargs="+", default=[2],
                    help="k_start ablation axis (handoff 7-1: '1 2' to compare include/exclude k=1).")
    ap.add_argument("--depth-modes", nargs="+", default=["peaked"], choices=["peaked", "frontload"],
                    help="depth-weight shape ablation axis (handoff 7-2: 'peaked frontload').")
    ap.add_argument("--min-agents", type=int, default=300)
    ap.add_argument("--max-agents", type=int, default=500)
    ap.add_argument("--agent-step", type=int, default=100)
    ap.add_argument("--min-frac", type=float, default=0.0)
    ap.add_argument("--max-frac", type=float, default=1.0)
    ap.add_argument("--frac-step", type=float, default=0.5)
    ap.add_argument("--center-value", type=float, default=base.congestion_center_value)
    ap.add_argument("--step-value", type=float, default=base.congestion_step_value)
    ap.add_argument("--predict-every", type=int, default=None,
                    help="MPC re-predict period; default = 11 - horizon (keeps full-depth lookahead, "
                         "handoff section 4). Raising it is cheaper but the penalty fades late in "
                         "each window.")
    ap.add_argument("--predict-everys", type=int, nargs="+", default=None,
                    help="sweep axis for the MPC re-predict period, e.g. '1 2 5 10'. When given, every "
                         "congestion run is also solved at each value (multiplies the grid like gammas / "
                         "min-depths). Omitted = a single value taken from --predict-every. Deliveries are "
                         "largely flat across this axis while wall time drops, so it mainly maps the "
                         "speed/quality trade-off. Each value is clamped to [1, 10].")
    ap.add_argument("--caps", type=float, nargs="+", default=None,
                    help="sweep axis for congestion_max_penalty: clip lambda*cong to this many "
                         "distance-cells, e.g. '0 1 2'. 0 = off (unbounded, the default). A cap <= 1 keeps "
                         "the distance-minimizing move first (PIBT progress guaranteed; no congestion-"
                         "induced backtracking blow-up) while still allowing dodge/wait among near-equal "
                         "cells. Omitted = a single 0 (off). Each value clamped to >= 0.")
    ap.add_argument("--no-video", action="store_true", help="metrics only, skip MP4s")
    ap.add_argument("--keep-paths", choices=["best", "all", "none"], default="best",
                    help="what to do with the videos/.paths/ trajectory cache after rendering. "
                         "'best' (default): keep only the baseline + best-lambda files that were "
                         "encoded (final footprint == the old 'only the best' behaviour). 'all': keep "
                         "every cached candidate so --video-only can re-pick a different lambda / "
                         "re-render later. 'none': delete the whole cache to reclaim disk.")
    ap.add_argument("--video-only", action="store_true",
                    help="skip the metrics phase entirely: re-render the videos for an existing run "
                         "(--from-run) straight from its cached sim paths (no re-simulation). Safe to "
                         "re-run if a previous render crashed; needs the run to have been produced with "
                         "the trajectory cache (i.e. not --keep-paths none).")
    ap.add_argument("--from-run", type=str, default=None,
                    help="run directory (reports/CALM_comparison/<yymmdd_hhmm>) for --video-only.")
    ap.add_argument("--video-workers", type=int, default=4,
                    help="parallel workers for the VIDEO phase only (the metrics phase uses "
                         "--num_of_process). Capped separately because each video worker opens a GPU "
                         "NVENC encoder session, and consumer GPUs allow only a few concurrent sessions "
                         "(too many -> 'OpenEncodeSessionEx failed / No capable devices found' and the "
                         "whole video phase dies). Default 4 is safe; raise it if your GPU/driver handled "
                         "more (composite mode also loads the 220MB predictor per worker, so watch VRAM).")
    # --- MACPF animate_paths knobs (videos) ---
    ap.add_argument("--anim-subframes", type=int, default=1,
                    help="interpolated frames per sim-step at 30fps. 30=realtime/smooth but ~30x "
                         "slower to render; 1=cell-to-cell jumps, fastest. (MACPF default 30)")
    ap.add_argument("--robot-size", type=float, default=8.0,
                    help="robot marker area (matplotlib s=). MACPF default 95 is for tens of agents; "
                         "shrink for hundreds.")
    ap.add_argument("--route-linewidth", type=float, default=0.6,
                    help="dashed robot->current-target line width.")
    ap.add_argument("--planned-route-linewidth", type=float, default=0.3,
                    help="faint full planned-route underlay width (only if --planned-routes).")
    ap.add_argument("--target-size", type=float, default=40.0, help="pickup/delivery target marker area.")
    ap.add_argument("--video-dpi", type=int, default=200,
                    help="MP4 render resolution: the figure is 10x8 in, so dpi 200 -> 2000x1600 px "
                         "(the old default was effectively 100 = 1000x800). Higher = crisper dots but "
                         "slower to render and larger files.")
    ap.add_argument("--video-cq", type=int, default=15,
                    help="NVENC constant-quality level (0-51, lower = higher quality / larger file). "
                         "15 is near-visually-lossless for this flat dots-on-map content; push to ~10 for "
                         "max quality. The CPU x264 fallback uses crf = cq + 1. Encoding is on the GPU "
                         "(h264_nvenc); both the per-panel render and the side-by-side hstack use it.")
    ap.add_argument("--planned-routes", action="store_true",
                    help="also draw each robot's full planned route as a faint underlay "
                         "(off by default: 300-750 such polylines clutter the frame).")
    ap.add_argument("--movement-only", action="store_true",
                    help="render the plain vanilla|congestion MOVEMENT clip only (the old default). "
                         "Without this each gamma gets the full composite: row 1 vanilla / row 2 "
                         "congestion-aware, col 1 movement / col 2 actual congestion / col 3 "
                         "SimVP-predicted congestion. The composite needs the GPU predictor.")
    ap.add_argument("--video-grid", choices=("3x2", "2x3"), default="3x2",
                    help="composite tiling: 3x2 = cols are movement|actual|predicted, rows are "
                         "vanilla/applied (wide); 2x3 = cols are vanilla|applied, rows are the three "
                         "views (near-square, keeps the old left-right vanilla|applied split).")
    ap.add_argument("--heatmap-vmax-pct", type=float, default=99.0,
                    help="percentile of positive congestion used as the shared heatmap colour-scale "
                         "max across ALL panels in a composite (lower = brighter / more saturated).")
    ap.add_argument("--verbose", action="store_true",
                    help="scrolling per-cell/per-video log instead of the tqdm progress bar")
    return ap


def validate_eval_args(ap: argparse.ArgumentParser, args) -> None:
    """Reject flag combinations argparse cannot express, via ``ap.error`` (exit code 2)."""
    if args.min_agents < 1 or args.min_agents > args.max_agents:
        ap.error("require 1 <= --min-agents <= --max-agents")
    if not (0.0 <= args.min_frac <= args.max_frac <= 1.0):
        ap.error("require 0.0 <= --min-frac <= --max-frac <= 1.0")
    if args.repeats < 1:
        ap.error("require --repeats >= 1")
