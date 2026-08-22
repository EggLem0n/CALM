"""Sub-module defining the (AMR count) x (dispersion fraction) grid that both the
dataset generator and the evaluator sweep over.

These three helpers used to live in `generate_heatmap/generate.py` *and* in
`evaluation/pipeline.py` as byte-identical copies. Keeping one definition means the
grid a dataset was generated on and the grid it is evaluated on cannot drift apart.

The evaluator layers its own extra axes (gamma, lambda, depth mode, horizon, cap,
repeats) on top of these two; those stay in the evaluator because the generator has
no notion of them.
"""
from __future__ import annotations

import argparse
from typing import List, Tuple


def agent_count_sweep(args: argparse.Namespace) -> List[int]:
    """AMR counts on one grid axis: --min-agents to --max-agents (inclusive),
    stepping by --agent-step. e.g. 300..750 step 50 -> [300, 350, ..., 750]."""
    return list(range(args.min_agents, args.max_agents + 1, args.agent_step))


def frac_sweep(args: argparse.Namespace) -> List[float]:
    """Dispersion fractions on the other axis: --min-frac to --max-frac (inclusive),
    stepping by --frac-step. e.g. 0..1 step 0.1 -> [0.0, 0.1, ..., 1.0]. Each value is
    the share of that episode's AMRs spawned spread across the whole map; the rest
    spawn in the charging/staging aisles."""
    n = int(round((args.max_frac - args.min_frac) / args.frac_step)) + 1
    out = []
    for i in range(n):
        f = round(args.min_frac + i * args.frac_step, 6)
        if f <= args.max_frac + 1e-9:
            out.append(min(1.0, max(0.0, f)))
    return out


def grid_cells(args: argparse.Namespace) -> List[Tuple[int, float]]:
    """The full (AMR count, dispersion fraction) grid. Outer loop = count,
    inner loop = fraction."""
    return [(c, f) for c in agent_count_sweep(args) for f in frac_sweep(args)]
