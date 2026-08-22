"""Sub-module for rendering the by-lambda metrics aggregate as a table image.

The console form of the same aggregate lives in :mod:`calm.eval.report`, which
stays free of matplotlib; only this module draws.
"""
from __future__ import annotations

import numpy as np


def save_metrics_table_png(rows, weights, n_cells, out_path):
    """Render the by-lambda mean metrics as a table image (matplotlib)."""
    from calm.utils import backend  # noqa: F401  -- selects Agg before pyplot
    import matplotlib.pyplot as plt

    cols = ["lambda", "deliveries", "±sd", "energy", "energy/deliv", "uniformity",
            "occ_cv", "cong@robot", "p99 cong", "peak cong", "preds", "collisions"]
    body = []
    for w in weights:
        sub = [r for r in rows if r["weight"] == w]
        if not sub:
            continue

        def m(key, group=sub):
            return float(np.mean([r[key] for r in group]))

        dv = [r["deliveries"] for r in sub]
        sd = float(np.std(dv)) if len(dv) > 1 else 0.0
        body.append([f"{w:g}", f"{m('deliveries'):.1f}", f"{sd:.1f}", f"{m('energy'):.0f}",
                     f"{m('energy_per_delivery'):.1f}", f"{m('density_uniformity'):.3f}",
                     f"{m('occ_cv'):.2f}", f"{m('mean_robot_cong'):.1f}",
                     f"{m('p99_cong'):.0f}", f"{m('peak_cong'):.0f}", f"{m('preds'):.0f}",
                     f"{int(sum(r['collisions'] for r in sub))}"])

    fig, ax = plt.subplots(figsize=(1.35 * len(cols), 0.7 + 0.45 * (len(body) + 1)))
    ax.axis("off")
    tbl = ax.table(cellText=body, colLabels=cols, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1, 1.6)
    for j in range(len(cols)):                          # header styling
        c = tbl[0, j]; c.set_facecolor("#40466e"); c.set_text_props(color="white", weight="bold")
    for i in range(1, len(body) + 1):                   # zebra rows
        for j in range(len(cols)):
            tbl[i, j].set_facecolor("#f2f2f7" if i % 2 else "#ffffff")
    ax.set_title(f"Congestion-aware PIBT - metrics by lambda  (mean over {n_cells} cells)",
                 fontsize=12, pad=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
