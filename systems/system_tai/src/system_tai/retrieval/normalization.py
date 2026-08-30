"""
Population-Level Score Normalization Module
===========================================
Computes deterministic Robust Z-scores across the full corpus population (180k vectors)
using Median and Median Absolute Deviation (MAD).
"""

from __future__ import annotations

import math
from typing import NamedTuple, Sequence
import numpy as np


class NormalizationStats(NamedTuple):
    population_size: int
    median: float
    mad: float
    is_degenerate: bool  # True if MAD == 0


def compute_population_mad_stats(
    scores: np.ndarray,
    epsilon: float = 1e-6,
) -> tuple[np.ndarray, NormalizationStats]:
    """
    Computes robust population z-scores:
        z = (scores - median) / (1.4826 * mad + epsilon)
    
    Uses float64 for stable deterministic reduction across different OS / BLAS threads.
    Handles NaN/Inf and degenerate zero-dispersion populations safely.
    """
    if scores.size == 0:
        return scores.copy(), NormalizationStats(0, 0.0, 0.0, True)

    # Cast to float64 for deterministic reductions
    arr_f64 = np.asarray(scores, dtype=np.float64).ravel()
    finite_mask = np.isfinite(arr_f64)
    finite_vals = arr_f64[finite_mask]

    if finite_vals.size == 0:
        # All values non-finite: fail closed
        z_out = np.full(scores.shape, -1e9, dtype=np.float32)
        return z_out, NormalizationStats(0, 0.0, 0.0, True)

    med = float(np.median(finite_vals))
    dev = np.abs(finite_vals - med)
    mad = float(np.median(dev))

    is_degenerate = mad < 1e-9

    if is_degenerate:
        # Fallback to standard deviation if MAD is 0 but finite scores vary slightly
        std = float(np.std(finite_vals))
        scale = max(std, epsilon)
    else:
        scale = 1.4826 * mad + epsilon

    z_scores = np.full_like(arr_f64, -1e9)
    z_scores[finite_mask] = (finite_vals - med) / scale
    
    # Reshape back to original shape in float32
    z_out = z_scores.reshape(scores.shape).astype(np.float32)

    stats = NormalizationStats(
        population_size=arr_f64.size,
        median=med,
        mad=mad,
        is_degenerate=is_degenerate,
    )
    return z_out, stats
