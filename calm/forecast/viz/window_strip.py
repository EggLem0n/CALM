"""Sub-module drawing one 10-frame horizon as a ground-truth-over-prediction strip.

Two rows (GT / predicted) by ten future frames on a shared color scale. This is the
visual counterpart to the numeric check in :mod:`calm.forecast.predict`:
it confirms the standalone checkpoint wrapper produces the right picture, not just
the right error.
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional

import numpy as np

from calm.utils.paths import run_dir

if TYPE_CHECKING:                       # avoids importing torch just to type-annotate
    from calm.forecast.predict import CongestionPredictor


def visualize_window(predictor: "CongestionPredictor", episode_path: str,
                     window: int = 0, out_path: Optional[str] = None) -> None:
    """Save a GT-vs-standalone-prediction strip for one 10-frame horizon."""
    from calm.utils import backend  # noqa: F401  -- selects Agg before pyplot
    import matplotlib.pyplot as plt

    from calm.forecast.predict import AFT_SEQ_LENGTH, PRE_SEQ_LENGTH

    y = np.load(episode_path)["y"].astype(np.float32)    # (T,1,50,80)
    stride = PRE_SEQ_LENGTH + AFT_SEQ_LENGTH
    s = window * stride
    assert s + stride <= y.shape[0], f"window {window} out of range for {y.shape[0]} frames"
    past = y[s:s + PRE_SEQ_LENGTH]                        # (10,1,50,80) raw
    true = y[s + PRE_SEQ_LENGTH:s + stride, 0]            # (10,50,80) raw
    pred = predictor.predict(past)[:, 0]                  # (10,50,80) raw
    vmax = float(true.max()) or 1.0

    fig, axes = plt.subplots(2, AFT_SEQ_LENGTH, figsize=(AFT_SEQ_LENGTH * 1.5, 3.4))
    for j in range(AFT_SEQ_LENGTH):
        axes[0, j].imshow(true[j], vmin=0, vmax=vmax, cmap="turbo"); axes[0, j].axis("off")
        axes[1, j].imshow(pred[j], vmin=0, vmax=vmax, cmap="turbo"); axes[1, j].axis("off")
        axes[0, j].set_title(f"t+{j + 1}", fontsize=7)
    axes[0, 0].set_ylabel("GT", fontsize=9); axes[1, 0].set_ylabel("Pred", fontsize=9)
    fig.suptitle(f"{os.path.basename(episode_path)}  window {window}  "
                 f"(standalone best.ckpt, raw vmax={vmax:.0f})", fontsize=9)
    fig.tight_layout()
    if out_path is None:
        out_path = str(run_dir("congestion_prediction") / "standalone_pred_check.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] saved {out_path}")
