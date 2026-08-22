# -*- coding: utf-8 -*-
"""Train/val loss curve from an OpenSTL training log.

``visualize.py`` (full run) and ``plot_loss_until.py`` (truncated to an early epoch)
drew the same curve from byte-identical code; the only real differences were the
epoch cut-off, the title, and the output filename. Those are parameters now.

Works mid-run: it only needs the per-epoch lines the trainer appends to
``work_dirs/<ex_name>/train_*.log``.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
from pathlib import Path
from typing import List, Optional, Tuple

from calm.utils.paths import run_dir

EPOCH_LINE = re.compile(
    r"Epoch\s+(\d+):.*?Train Loss:\s+([\d.eE+-]+).*?Vali Loss:\s+([\d.eE+-]+)")


def read_loss_log(log_dir: str, max_epoch: Optional[int] = None
                  ) -> Tuple[List[int], List[float], List[float], Optional[str]]:
    """Parse the newest ``train_*.log`` in ``log_dir``.

    Returns ``(epochs, train, val, log_path)``; ``log_path`` is None when no log
    exists. With ``max_epoch`` set, epochs above it are dropped so the early part of
    a run can be shown without later epochs squashing the y-axis.
    """
    logs = sorted(glob.glob(os.path.join(log_dir, "train_*.log")))
    if not logs:
        return [], [], [], None
    log_path = logs[-1]
    epochs: List[int] = []
    train: List[float] = []
    val: List[float] = []
    with open(log_path, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            m = EPOCH_LINE.search(line)
            if not m:
                continue
            epoch = int(m.group(1))
            if max_epoch is not None and epoch > max_epoch:
                continue
            epochs.append(epoch)
            train.append(float(m.group(2)))
            val.append(float(m.group(3)))
    return epochs, train, val, log_path


def plot_loss_curve(log_dir: str, out_path: str, max_epoch: Optional[int] = None) -> bool:
    """Draw the train/val curve to ``out_path``. Returns False (and explains why) when
    the log is missing or has no epoch lines yet."""
    from calm.utils import backend  # noqa: F401  -- selects Agg before pyplot
    import matplotlib.pyplot as plt

    epochs, train, val, log_path = read_loss_log(log_dir, max_epoch)
    if log_path is None:
        print("[loss curve skipped] no train_*.log in", log_dir)
        return False
    if not epochs:
        scope = "" if max_epoch is None else f"<= {max_epoch} "
        print(f"[loss curve skipped] no epoch {scope}loss lines in", log_path)
        return False

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(epochs, train, marker='o', ms=3, label="train loss")
    ax.plot(epochs, val, marker='s', ms=3, label="val loss")
    best = min(range(len(val)), key=lambda i: val[i])
    ax.axvline(epochs[best], color='gray', ls='--', lw=1)
    ax.annotate(f"best val {val[best]:.4g}\n@ epoch {epochs[best]}",
                xy=(epochs[best], val[best]),
                xytext=(0.98, 0.95), textcoords='axes fraction',
                ha='right', va='top', fontsize=9,
                arrowprops=dict(arrowstyle='->', color='gray'))
    ax.set_xlabel("epoch"); ax.set_ylabel("loss (MSE)")
    scope = "" if max_epoch is None else f"epoch <= {max_epoch}, "
    ax.set_title(f"Train / Val loss  ({scope}{os.path.basename(log_path)})")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    scope = "" if max_epoch is None else f" (<= {max_epoch})"
    print(f"[loss curve] {len(epochs)} epochs{scope} "
          f"from {os.path.basename(log_path)} -> {out_path}")
    return True


EX_NAME = "custom_exp_congestion"  # matches train.py


def add_loss_curve_args(ap: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add the loss-curve flags to `ap` and return it."""
    ap.add_argument("max_epoch", nargs="?", type=int, default=None,
                    help="drop epochs above this (default: plot every logged epoch)")
    return ap


def run_loss_curve(args) -> None:
    """Draw this run's loss curve into a fresh report folder."""
    max_epoch = args.max_epoch
    log_dir = Path(__file__).resolve().parents[1] / "work_dirs" / EX_NAME
    name = "loss_curve.png" if max_epoch is None else f"loss_curve_e{max_epoch}.png"
    plot_loss_curve(str(log_dir), str(run_dir("congestion_prediction") / name),
                    max_epoch=max_epoch)
