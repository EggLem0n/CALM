"""Sub-package containing the congestion heatmap dataset generator (PIBT solver).

Sweeps the solver over the (AMR count) x (dispersion fraction) grid and writes one
``episode_*.npz`` shard per cell.

Nothing here reads ``sys.argv``: ``scripts/02_dataset/generate_dataset.py`` owns the parser and
builds it from the flag definitions :func:`generate.add_generate_args` contributes.
"""
from .generate import add_generate_args, run_generation, validate_generate_args

__all__ = ["add_generate_args", "validate_generate_args", "run_generation"]
