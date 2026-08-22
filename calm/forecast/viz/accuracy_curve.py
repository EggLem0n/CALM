"""Sub-module for the error-vs-horizon curve of the congestion predictor.

Draws what :mod:`calm.forecast.per_frame_accuracy` computes: MSE and MAE
against the prediction horizon t+1 .. t+10, on twin y-axes because the two live on
different scales. The computation itself stays in that module, free of matplotlib.
"""
from __future__ import annotations

import os

import numpy as np


def plot_curve(path: str, m: dict) -> None:
    """Write the per-horizon error curve to `path`. `m` is per_frame_metrics()'s dict."""
    from calm.utils import backend  # noqa: F401  -- selects Agg before pyplot
    import matplotlib.pyplot as plt

    T = len(m["mse"])
    xs = np.arange(1, T + 1)
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.plot(xs, m["mse"], marker="o", color="tab:red", label="MSE (norm)")
    ax1.set_xlabel("prediction horizon  (t+k)")
    ax1.set_ylabel("MSE (normalized units)", color="tab:red")
    ax1.tick_params(axis="y", labelcolor="tab:red")
    ax1.set_xticks(xs)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(xs, m["mae"], marker="s", color="tab:blue", label="MAE (norm)")
    ax2.set_ylabel("MAE (normalized units)", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")

    ax1.set_title("Per-frame prediction error vs horizon\n(higher k = further into the future)")
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
