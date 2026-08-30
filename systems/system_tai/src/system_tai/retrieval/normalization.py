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
    
    # Clean non-finite values if present
    if not np.all(np.isfinite(arr_f64)):
        arr_f64 = np.nan_to_num(arr_f64, nan=0.0, posinf=1.0, neginf=-1.0)

    med = float(np.median(arr_f64))
    dev = np.abs(arr_f64 - med)
    mad = float(np.median(dev))

    is_degenerate = mad < 1e-9

    if is_degenerate:
        # Fallback to standard deviation if MAD is 0 but scores vary slightly
        std = float(np.std(arr_f64))
        scale = max(std, epsilon)
    else:
        scale = 1.4826 * mad + epsilon

    z_scores = (arr_f64 - med) / scale
    
    # Reshape back to original shape in float32
    z_out = z_scores.reshape(scores.shape).astype(np.float32)

    stats = NormalizationStats(
        population_size=arr_f64.size,
        median=med,
        mad=mad,
        is_degenerate=is_degenerate,
    )
    return z_out, stats
