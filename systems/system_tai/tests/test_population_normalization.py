"""
Unit test for population-level robust Z-score normalization.
"""

import numpy as np
import pytest
from system_tai.retrieval.normalization import (
    compute_population_mad_stats,
    NormalizationStats,
)


def test_population_mad_stats_standard():
    # Normal distribution test
    rng = np.random.default_rng(seed=42)
    scores = rng.normal(loc=0.5, scale=0.1, size=(1000,)).astype(np.float32)

    z_scores, stats = compute_population_mad_stats(scores)

    assert stats.population_size == 1000
    assert not stats.is_degenerate
    assert abs(stats.median - 0.5) < 0.02
    assert stats.mad > 0.0
    assert abs(float(np.median(z_scores))) < 1e-5
    assert z_scores.shape == (1000,)
    assert z_scores.dtype == np.float32


def test_population_mad_stats_degenerate_zeros():
    # All scores are identical
    scores = np.ones((500,), dtype=np.float32) * 0.42

    z_scores, stats = compute_population_mad_stats(scores)

    assert stats.population_size == 500
    assert stats.is_degenerate
    assert np.all(z_scores == 0.0)


def test_population_mad_stats_nan_handling():
    # Array contains NaNs and Infs
    scores = np.array([0.1, np.nan, 0.3, np.inf, 0.2], dtype=np.float32)

    z_scores, stats = compute_population_mad_stats(scores)

    assert stats.population_size == 5
    assert np.all(np.isfinite(z_scores))
