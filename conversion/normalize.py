"""
Threshold normalization: compute per-layer IF thresholds from calibration data.

Supports both a single global percentile and per-layer percentiles, and
reports the clip fraction (share of observed activations exceeding threshold).

Usage
-----
python conversion/normalize.py --stats calibration_stats.pkl
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import PERCENTILE


def compute_thresholds(
    stats_path: str | dict[str, np.ndarray] = "calibration_stats.pkl",
    percentile: float | Sequence[float] = PERCENTILE,
    verbose:    bool = True,
    return_clip_fracs: bool = False,
) -> dict[str, float] | tuple[dict[str, float], dict[str, float]]:
    """
    Load calibration stats (or accept a preloaded dict) and compute one
    threshold per layer.

    Parameters
    ----------
    stats_path : str | dict
        Path to calibration_stats.pkl, or already-loaded stats dict.
    percentile : float | Sequence[float]
        Single percentile or one percentile per layer in sorted layer order.
    verbose : bool
        Print summary table.
    return_clip_fracs : bool
        If True, return (thresholds, clip_fractions).
    """
    if isinstance(stats_path, dict):
        stats = stats_path
    else:
        with open(stats_path, "rb") as f:
            stats = pickle.load(f)

    layer_names = sorted(stats.keys())
    if isinstance(percentile, (list, tuple, np.ndarray)):
        assert len(percentile) == len(layer_names), (
            f"Expected {len(layer_names)} percentiles, got {len(percentile)}"
        )
        pct_list = [float(p) for p in percentile]
    else:
        pct_list = [float(percentile)] * len(layer_names)

    thresholds: dict[str, float] = {}
    clip_fracs: dict[str, float] = {}

    if verbose:
        pct_desc = f"{pct_list[0]}th" if len(set(pct_list)) == 1 else str(pct_list)
        print(f"Computing thresholds at {pct_desc} percentile:\n")
        print(f"  {'Layer':<12}  {'Max act':>10}  {'Threshold':>10}  {'Ratio':>8}  {'Clip %':>8}")
        print("  " + "-" * 58)

    for name, pct in zip(layer_names, pct_list):
        acts  = stats[name]
        t     = float(np.percentile(acts, pct))
        t     = max(t, 1e-6)
        ratio = t / (float(acts.max()) + 1e-9)
        clip  = float(np.mean(acts > t))
        thresholds[name] = t
        clip_fracs[name] = clip
        if verbose:
            print(f"  {name:<12}  {acts.max():>10.4f}  {t:>10.4f}  {ratio:>7.1%}  {clip*100:>7.3f}%")

    if verbose:
        print(f"\nThresholds (ordered): {list(thresholds.values())}")

    if return_clip_fracs:
        return thresholds, clip_fracs
    return thresholds


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--stats",      type=str,   default="calibration_stats.pkl")
    p.add_argument("--percentile", type=float, nargs="+", default=[PERCENTILE])
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    pct = args.percentile[0] if len(args.percentile) == 1 else args.percentile
    compute_thresholds(args.stats, pct)
