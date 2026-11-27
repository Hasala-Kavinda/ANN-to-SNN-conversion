"""
Statistical helpers for paired evaluation and bootstrap confidence intervals.
"""

from __future__ import annotations

import numpy as np


def bootstrap_ci(
    values: np.ndarray | list[float],
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """
    Compute a two-sided percentile bootstrap confidence interval for the mean.

    Returns (ci_lo, ci_hi). For <= 1 sample, returns (mean, mean).
    """
    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return 0.0, 0.0
    if n == 1:
        m = float(arr[0])
        return m, m

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = arr[idx].mean(axis=1)
    lo = float(np.percentile(means, 100.0 * (alpha / 2.0)))
    hi = float(np.percentile(means, 100.0 * (1.0 - alpha / 2.0)))
    return lo, hi


def paired_summary(
    ann_rewards: np.ndarray | list[float],
    snn_rewards: np.ndarray | list[float],
    n_boot: int = 2000,
    seed: int = 0,
) -> dict:
    """
    Compute paired difference (SNN - ANN) with 95% bootstrap CI and 95% non-inferiority check.

    Non-inferiority at 95% holds when the lower 95% CI bound of (SNN - ANN) is
    >= -0.05 * |mean(ANN)| (or >= -5.0 if |mean(ANN)| is near zero).
    """
    ann = np.asarray(ann_rewards, dtype=np.float64)
    snn = np.asarray(snn_rewards, dtype=np.float64)
    if len(ann) != len(snn):
        raise ValueError(f"Paired arrays must have equal length, got {len(ann)} vs {len(snn)}")
    if len(ann) == 0:
        raise ValueError("Paired arrays must not be empty")

    diff = snn - ann
    mean_diff = float(diff.mean())
    ci_lo, ci_hi = bootstrap_ci(diff, n_boot=n_boot, seed=seed)
    ann_mean = float(ann.mean())
    snn_mean = float(snn.mean())

    ratio = (snn_mean / ann_mean) if ann_mean > 1e-6 else None
    margin = 0.05 * abs(ann_mean) if abs(ann_mean) > 1e-6 else 5.0
    non_inferior_95 = bool(ci_lo >= -margin)

    return {
        "n": len(ann),
        "ann_mean": ann_mean,
        "snn_mean": snn_mean,
        "mean_diff": mean_diff,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "ratio": ratio,
        "non_inferior_95": non_inferior_95,
    }
