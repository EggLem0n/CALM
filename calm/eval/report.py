"""Turning the collected metric rows into console tables.

Kept free of matplotlib: the PNG form of the by-lambda aggregate is drawn by
:mod:`calm.eval.viz.table`.
"""
from __future__ import annotations

import csv

import numpy as np

from .runner import _cfg_key, _cfg_label


def _print_lambda_table(rows, weights, indent="", base_mean=None):
    """Print a per-lambda aggregate of `rows` (mean over whatever cells/configs they span).

    Columns mirror the CSV metrics, plus the across-row std of deliveries (±sd) and,
    when `base_mean` is given, the delivery gain over baseline (d-base)."""
    has_base = base_mean is not None
    cols = [("deliv", 7), ("±sd", 5)]
    if has_base:
        cols.append(("d-base", 7))
    cols += [("energy", 8), ("e/dlv", 7), ("unifrm", 6), ("occCV", 5),
             ("cong@r", 6), ("p99", 6), ("peak", 6), ("preds", 6), ("wall", 6), ("coll", 4)]
    hdr = f"{indent}{'lam':>5} | " + " ".join(f"{n:>{w}}" for n, w in cols)
    print(hdr)
    print(f"{indent}{'-' * (len(hdr) - len(indent))}")
    for lam in weights:
        sub = [r for r in rows if r["weight"] == lam]
        if not sub:
            continue
        dv = [r["deliveries"] for r in sub]

        def mean(key, group=sub):
            return float(np.mean([r[key] for r in group]))

        sd = float(np.std(dv)) if len(dv) > 1 else 0.0
        vals = {
            "deliv":  f"{float(np.mean(dv)):>7.1f}",
            "±sd":    f"{sd:>5.1f}",
            "d-base": f"{(float(np.mean(dv)) - base_mean):>+7.1f}" if has_base else "",
            "energy": f"{mean('energy'):>8.0f}",
            "e/dlv":  f"{mean('energy_per_delivery'):>7.2f}",
            "unifrm": f"{mean('density_uniformity'):>6.3f}",
            "occCV":  f"{mean('occ_cv'):>5.2f}",
            "cong@r": f"{mean('mean_robot_cong'):>6.1f}",
            "p99":    f"{mean('p99_cong'):>6.0f}",
            "peak":   f"{mean('peak_cong'):>6.0f}",
            "preds":  f"{mean('preds'):>6.0f}",
            "wall":   f"{mean('wall_s'):>6.1f}",
            "coll":   f"{int(sum(r['collisions'] for r in sub)):>4}",
        }
        print(f"{indent}{lam:>5.2f} | " + " ".join(vals[n] for n, _ in cols))


def _typed_metric_rows(csv_path):
    """Read metrics.csv back with the same Python types run_job produced (so the best-lambda
    selection in --video-only matches the inline phase). Empty cells (baseline) stay ''."""
    def num(s, cast):
        return cast(s) if s not in ("", None) else ""
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows.append({
                "episode": int(r["episode"]),
                "num_agents": int(r["num_agents"]),
                "frac": float(r["frac"]),
                "gamma": num(r["gamma"], float),
                "horizon": num(r["horizon"], int),
                "min_depth": num(r["min_depth"], int),
                "depth_mode": r["depth_mode"],
                "weight": float(r["weight"]) if r["weight"] not in ("", None) else 0.0,
                "predict_every": num(r["predict_every"], int),
                "cap": num(r["cap"], float),
                "rep": int(r["rep"]) if r.get("rep") not in ("", None) else 0,  # absent in pre-repeats CSVs
                "deliveries": float(r["deliveries"]) if r["deliveries"] not in ("", None) else 0.0,
            })
    return rows


def print_summaries(all_rows, weights, weights_pos, grid, rows_by_cell, cells_done, total):
    """The four console tables a finished sweep prints: overall by lambda, per cell,
    best lambda per config, and the full lambda sweep per config."""
    cong_rows = [r for r in all_rows if r["depth_mode"] != "baseline"]
    base_deliveries = [r["deliveries"] for r in all_rows if r["depth_mode"] == "baseline"]
    base_mean = float(np.mean(base_deliveries)) if base_deliveries else None

    # (1) headline: mean over every completed cell, by congestion weight
    if all_rows:
        print(f"\n=== mean over {cells_done}/{total} completed cells, by congestion weight ===")
        _print_lambda_table(all_rows, weights, base_mean=base_mean)

    # (2) per (count, frac) cell: by congestion weight (mean over mode/md/gamma within the cell)
    if all_rows:
        print("\n=== per (count, frac) cell, by congestion weight ===")
        for ci in sorted(rows_by_cell):
            crows = rows_by_cell[ci]
            if not crows:
                continue
            count, frac = grid[ci]
            cbase = [r["deliveries"] for r in crows if r["depth_mode"] == "baseline"]
            cbm = float(np.mean(cbase)) if cbase else None
            tag = f"  [count={count}  frac={frac:g}]"
            if cbm is not None:
                tag += f"  baseline deliv={cbm:.1f}"
            print(f"\n{tag}")
            _print_lambda_table(crows, weights, indent="    ", base_mean=cbm)

    # (3) best lambda per config (mode, md, gamma, H, predict_every, cap): compact winner summary
    if cong_rows:
        bm = base_mean if base_mean is not None else float("nan")
        print(f"\n=== best lambda per config (mode,md,gamma,H,pe,cap) "
              f"(mean deliveries over cells; baseline {bm:.1f}) ===")
        for combo in sorted({_cfg_key(r) for r in cong_rows}):
            by_w = {}
            for r in cong_rows:
                if _cfg_key(r) == combo:
                    by_w.setdefault(r["weight"], []).append(r["deliveries"])
            best_w = max(by_w, key=lambda w: float(np.mean(by_w[w])))
            best_d = float(np.mean(by_w[best_w]))
            print(f"  [{_cfg_label(combo)}]  best λ={best_w:g}  deliv={best_d:.1f}  d-base={best_d - bm:+.1f}")

    # (4) full lambda sweep per config (mode, md, gamma, H, predict_every, cap): all lambdas, full metrics
    if cong_rows:
        print("\n=== full lambda sweep per config (mode,md,gamma,H,pe,cap) (mean over cells) ===")
        for combo in sorted({_cfg_key(r) for r in cong_rows}):
            grp = [r for r in cong_rows if _cfg_key(r) == combo]
            print(f"\n  [{_cfg_label(combo)}]")
            _print_lambda_table(grp, weights_pos, indent="    ", base_mean=base_mean)
