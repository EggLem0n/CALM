# -*- coding: utf-8 -*-
"""Standalone inference wrapper for the trained SimVP congestion predictor.

Loads ``work_dirs/custom_exp_congestion/checkpoints/best.ckpt`` WITHOUT OpenSTL's
Lightning training harness (no dataloaders, no BaseExperiment), exposing a plain
``torch.nn.Module`` that maps 10 past congestion frames -> 10 future frames. This
is the piece PIBT will call online to steer agents away from predicted congestion.

The checkpoint is a Lightning checkpoint: its ``state_dict`` keys are prefixed
``model.`` (the SimVP net lives at ``Base_method.model``). We rebuild ``SimVP_Model``
with the exact training config (hparams.yaml) and load only those ``model.*`` weights.

Normalization: training divided every congestion frame by ``Y_SCALE`` (the train
split's windowed max) into ~[0, 1]; this value was NOT persisted, so we hard-code
the recomputed constant here and expose ``compute_y_scale()`` to reproduce it.
PIBT only needs the relative ranking of cells, so the scale cancels out there --
but ``predict()`` denormalizes so callers also get congestion in real units.

Run (OpenSTL conda env):
    conda activate OpenSTL
    python predict.py                       # verify wrapper == exp.test() saved/preds.npy
    python predict.py --episode data/heatmap_dataset/260623_0907/episode_0000.npz
    python predict.py --recompute-y-scale   # re-derive Y_SCALE from the train split
"""
from __future__ import annotations

import argparse
import os
# Windows/conda OpenMP shim (torch libiomp + numpy MKL); must precede numpy/torch.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import warnings
# Harmless: PyTorch Lightning still imports the deprecated pkg_resources API. Mute that
# one message (must precede the openstl/lightning import below).
warnings.filterwarnings("ignore", message=r".*pkg_resources is deprecated.*")
import sys
import glob
import random

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
OPENSTL_ROOT = os.path.join(HERE, "OpenSTL")
if OPENSTL_ROOT not in sys.path:
    sys.path.insert(0, OPENSTL_ROOT)

# SimVP_Model is imported inside CongestionPredictor.__init__ rather than here: pulling
# it in drags the whole vendored OpenSTL tree (timm, fvcore, lightning) into the process,
# and a launcher showing --help, or a caller that only wants the constants below, must
# not pay for that. sys.path is already prepared above, so the import works from anywhere.

# ---------------------------------------------------------------------------
# Constants -- must match train.py / hparams.yaml
# ---------------------------------------------------------------------------
PRE_SEQ_LENGTH = 10
AFT_SEQ_LENGTH = 10

MODEL_CFG = dict(
    in_shape=[PRE_SEQ_LENGTH, 1, 50, 80],
    hid_S=64,
    hid_T=256,
    N_S=2,
    N_T=8,
    model_type="gSTA",
)

# Train split's windowed max; recompute with compute_y_scale() / --recompute-y-scale.
Y_SCALE = 1100.0

# Everything shared sits at the repo root, two levels up from this package: the
# checkpoint in models/, the dataset in data/. OpenSTL is vendored alongside this file,
# so OPENSTL_ROOT above needs no adjustment. SAVED_DIR (the arrays exp.test() writes) is
# produced by training here; until then verify_against_saved() just skips.
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
CKPT_PATH = os.path.join(REPO_ROOT, "models", "best.ckpt")
DATA_DIR = os.path.join(REPO_ROOT, "data", "heatmap_dataset", "260623_0907")
SAVED_DIR = os.path.join(HERE, "work_dirs", "custom_exp_congestion", "saved")


class CongestionPredictor:
    """past 10 congestion frames -> predicted future 10 frames (raw congestion units)."""

    def __init__(self, ckpt_path: str = CKPT_PATH, device: str | None = None,
                 y_scale: float = Y_SCALE):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.y_scale = float(y_scale)

        from openstl.models import SimVP_Model  # heavy; see the note at the imports

        model = SimVP_Model(**MODEL_CFG)
        # Lightning ckpt holds pickled hparams/callbacks -> torch>=2.6 needs weights_only=False.
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = ckpt["state_dict"]
        model_state = {k[len("model."):]: v for k, v in state.items() if k.startswith("model.")}
        missing, unexpected = model.load_state_dict(model_state, strict=True)
        assert not missing and not unexpected, (missing, unexpected)
        model.eval().to(self.device)
        self.model = model

    @torch.no_grad()
    def predict_normalized(self, past_norm: np.ndarray) -> np.ndarray:
        """past_norm: (B, 10, 1, 50, 80) already in [0, 1] -> (B, 10, 1, 50, 80) in [0, 1]."""
        x = torch.as_tensor(np.asarray(past_norm, np.float32), device=self.device)
        return self.model(x).cpu().numpy()

    @torch.no_grad()
    def predict(self, past_congestion: np.ndarray) -> np.ndarray:
        """Raw congestion in, raw congestion out.

        Accepts (10, 1, 50, 80) or batched (B, 10, 1, 50, 80); returns the same rank.
        """
        a = np.asarray(past_congestion, np.float32)
        single = a.ndim == 4
        if single:
            a = a[None]
        out = self.predict_normalized(a / self.y_scale) * self.y_scale
        return out[0] if single else out


# ---------------------------------------------------------------------------
# y_scale reproduction (deterministic; mirrors train.py)
# ---------------------------------------------------------------------------
def compute_y_scale(data_dir: str = DATA_DIR, ratio=(19, 7, 7), seed: int = 0,
                    pre: int = PRE_SEQ_LENGTH, aft: int = AFT_SEQ_LENGTH) -> float:
    """Reproduce train.py's ``y_scale`` from the train split.

    Same stratified-by-robot-count split (seed 0) and same non-overlapping
    (stride = pre+aft) sliding windows; y_scale = max over the train windows.
    """
    files = sorted(glob.glob(os.path.join(data_dir, "episode_*.npz")))
    assert files, f"no episode_*.npz in {data_dir}"

    def robot_count(f):
        with np.load(f) as z:
            return int(z["starts"].shape[0])

    groups: dict[int, list[str]] = {}
    for f in files:
        groups.setdefault(robot_count(f), []).append(f)

    r_tr, _, _ = ratio
    denom = sum(ratio)
    rng = random.Random(seed)
    train: list[str] = []
    for n in sorted(groups):
        g = sorted(groups[n])
        rng.shuffle(g)
        train += g[: len(g) * r_tr // denom]

    stride = pre + aft
    y_scale = 1.0
    for f in sorted(train):
        y = np.load(f)["y"].astype(np.float32)
        for s in range(0, y.shape[0] - stride + 1, stride):
            m = float(y[s:s + stride].max())
            if m > y_scale:
                y_scale = m
    return y_scale


# ---------------------------------------------------------------------------
# Verifications / demos
# ---------------------------------------------------------------------------
def verify_against_saved(predictor: CongestionPredictor, n: int = 64) -> None:
    """The strongest check: reproduce exp.test()'s saved/preds.npy from saved/inputs.npy.

    inputs/preds are already normalized [0, 1], so this bypasses y_scale and asks
    only: does the standalone load reproduce the trained model's predictions?
    """
    inp = os.path.join(SAVED_DIR, "inputs.npy")
    prd = os.path.join(SAVED_DIR, "preds.npy")
    if not (os.path.exists(inp) and os.path.exists(prd)):
        print("[verify] saved/inputs.npy or preds.npy missing -- skip (run train's test first).")
        return
    inputs = np.load(inp, mmap_mode="r")[:n]          # (n,10,1,50,80) normalized
    ref = np.load(prd, mmap_mode="r")[:n]
    out = predictor.predict_normalized(np.ascontiguousarray(inputs))
    diff = np.abs(out - ref)
    print(f"[verify] standalone vs exp.test() on {n} test windows:")
    print(f"         max|Δ| = {diff.max():.3e}   mean|Δ| = {diff.mean():.3e}")
    ok = diff.max() < 1e-3
    print("         => REPRODUCES saved/preds.npy" if ok
          else "         => MISMATCH (check MODEL_CFG / checkpoint)")


def roll_episode(predictor: CongestionPredictor, episode_path: str) -> None:
    """Slide the 10->10 predictor over one episode; report error vs ground truth."""
    y = np.load(episode_path)["y"].astype(np.float32)   # (T,1,50,80) raw congestion
    T = y.shape[0]
    stride = PRE_SEQ_LENGTH + AFT_SEQ_LENGTH
    pasts, futures = [], []
    for s in range(0, T - stride + 1, stride):
        pasts.append(y[s:s + PRE_SEQ_LENGTH])
        futures.append(y[s + PRE_SEQ_LENGTH:s + stride])
    if not pasts:
        print(f"[roll] episode too short ({T} frames) for a {stride}-frame window.")
        return
    past = np.stack(pasts)                                # (B,10,1,50,80) raw
    true = np.stack(futures)
    pred = predictor.predict(past)                       # raw units
    mse = float(np.mean((pred - true) ** 2))
    mae = float(np.mean(np.abs(pred - true)))
    base = os.path.basename(episode_path)
    print(f"[roll] {base}: {past.shape[0]} windows | raw-unit  MSE {mse:.2f}  MAE {mae:.2f}"
          f"  (y_scale={predictor.y_scale:g})")
    nmse = mse / predictor.y_scale ** 2
    print(f"       normalized MSE {nmse:.3e} (compare train val_loss ~8e-4)")


def add_predict_args(ap: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add the checkpoint-verification flags to `ap` and return it."""
    ap.add_argument("--ckpt", default=CKPT_PATH)
    ap.add_argument("--episode", default=None, help="roll the predictor over this episode_*.npz")
    ap.add_argument("--viz", default=None, help="save GT-vs-pred image for this episode_*.npz")
    ap.add_argument("--window", type=int, default=0, help="which 10-frame window to visualize")
    ap.add_argument("--device", default=None)
    ap.add_argument("--recompute-y-scale", action="store_true",
                    help="re-derive Y_SCALE from the train split and exit")
    return ap


def run_predict(args) -> None:
    """Do whichever check `args` selected: y-scale, a strip, an episode roll, or the default."""
    if args.recompute_y_scale:
        ys = compute_y_scale()
        print(f"recomputed y_scale = {ys:g}  (hard-coded Y_SCALE = {Y_SCALE:g})")
        return

    predictor = CongestionPredictor(ckpt_path=args.ckpt, device=args.device)
    print(f"loaded {os.path.basename(args.ckpt)} on {predictor.device} | y_scale={predictor.y_scale:g}")

    if args.viz:
        from calm.forecast.viz.window_strip import visualize_window

        visualize_window(predictor, args.viz, window=args.window)
    elif args.episode:
        roll_episode(predictor, args.episode)
    else:
        verify_against_saved(predictor)
