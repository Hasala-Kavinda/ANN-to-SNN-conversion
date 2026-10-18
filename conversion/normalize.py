"""
Threshold normalization: compute per-layer IF thresholds from calibration data.

The threshold for layer i is the P-th percentile of all activations
observed at that layer's ReLU output during calibration gameplay.

Using the high percentile (99.9th by default) instead of the absolute max:
  - The max is too sensitive to outlier hot pixels in rare frames.
  - The 99.9th clips only the rarest activations, keeping spike rates in
    a usable range while preventing most neurons from being silenced.

Usage
-----
python conversion/normalize.py --stats calibration_stats.pkl
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import PERCENTILE


def compute_thresholds(
    stats_path: str = "calibration_stats.pkl",
    percentile: float = PERCENTILE,
) -> dict[str, float]:
    """
    Load calibration stats and compute one threshold per layer.

    Returns
    -------
    dict[layer_name, threshold_value]
    Ordered as relu_0, relu_1, relu_2, relu_3 (conv1, conv2, conv3, fc1).
    """
    with open(stats_path, "rb") as f:
        stats: dict[str, np.ndarray] = pickle.load(f)

    thresholds: dict[str, float] = {}
    print(f"Computing thresholds at {percentile}th percentile:\n")
    print(f"  {'Layer':<12}  {'Max act':>10}  {'Threshold':>10}  {'Ratio':>8}")
    print("  " + "-" * 48)

    for name in sorted(stats.keys()):
        acts  = stats[name]
        t     = float(np.percentile(acts, percentile))
        t     = max(t, 1e-6)    # guard against zero thresholds
        ratio = t / (acts.max() + 1e-9)
        thresholds[name] = t
        print(f"  {name:<12}  {acts.max():>10.4f}  {t:>10.4f}  {ratio:>7.1%}")

    print(f"\nThresholds (ordered): {list(thresholds.values())}")
    return thresholds


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--stats",      type=str,   default="calibration_stats.pkl")
    p.add_argument("--percentile", type=float, default=PERCENTILE)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    compute_thresholds(args.stats, args.percentile)
