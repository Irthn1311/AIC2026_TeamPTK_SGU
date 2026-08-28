"""Unit tests for DP temporal chain solver and soft-AND multi-clause scoring."""

from __future__ import annotations

import math
import pytest

from system_tai.kis.video_first import (
    compute_soft_and_joint_score,
    solve_temporal_chain,
)


def test_single_scene_chain_returns_best_peak() -> None:
    peaks = [[(100, 0.85), (200, 0.92), (300, 0.70)]]
    weights = [1.0]
    valid, frames, score = solve_temporal_chain(peaks, weights, min_gap=60)
    assert valid is True
    assert frames == (200,)
    assert math.isclose(score, 0.92, rel_tol=1e-5)


def test_two_scene_valid_chronological_chain() -> None:
    # Scene 1 peaks: frame 100 (.85), frame 1000 (.95)
    # Scene 2 peaks: frame 500 (.90), frame 1200 (.75)
    # Valid chain 1: 100 -> 500 (score: 0.85 + 0.90 = 1.75 / 2 = 0.875)
    # Valid chain 2: 100 -> 1200 (score: 0.85 + 0.75 = 1.60 / 2 = 0.80)
    # Valid chain 3: 1000 -> 1200 (score: 0.95 + 0.75 = 1.70 / 2 = 0.85)
    # Best valid chain: (100, 500) with avg score 0.875!
    # (Notice independent argmax would pick 1000 and 500 which is INVALID (1000 > 500)).
    peaks = [
        [(100, 0.85), (1000, 0.95)],
        [(500, 0.90), (1200, 0.75)],
    ]
    weights = [1.0, 1.0]
    valid, frames, score = solve_temporal_chain(peaks, weights, min_gap=60)
    assert valid is True
    assert frames == (100, 500)
    assert math.isclose(score, 0.875, rel_tol=1e-5)


def test_two_scene_strictly_invalid_sequence_fails() -> None:
    # Scene 1 peak only at frame 2000
    # Scene 2 peak only at frame 500
    # No chronological sequence possible where f1 + 60 <= f2
    peaks = [
        [(2000, 0.95)],
        [(500, 0.90)],
    ]
    weights = [1.0, 1.0]
    valid, frames, score = solve_temporal_chain(peaks, weights, min_gap=60)
    assert valid is False
    assert frames == ()
    assert score == 0.0


def test_three_scene_chronological_chain_dp() -> None:
    # T1 -> T2 -> T3
    peaks = [
        [(100, 0.80), (800, 0.90)],
        [(400, 0.85), (1200, 0.70)],
        [(700, 0.95), (1500, 0.88)],
    ]
    weights = [1.0, 1.0, 1.0]
    valid, frames, score = solve_temporal_chain(peaks, weights, min_gap=60)
    assert valid is True
    # Valid chains:
    # 100 -> 400 -> 700: (.80 + .85 + .95) / 3 = 2.60 / 3 = 0.8667
    # 100 -> 400 -> 1500: (.80 + .85 + .88) / 3 = 2.53 / 3 = 0.8433
    # 100 -> 1200 -> 1500: (.80 + .70 + .88) / 3 = 2.38 / 3 = 0.7933
    # 800 -> 1200 -> 1500: (.90 + .70 + .88) / 3 = 2.48 / 3 = 0.8267
    assert frames == (100, 400, 700)
    assert math.isclose(score, 2.60 / 3.0, rel_tol=1e-5)


def test_soft_and_joint_score_demotes_single_clause_distractor() -> None:
    weights = [1.0, 1.0]
    # Single-clause distractor: high T1, near-zero T2
    distractor_scores = [0.45, 0.05]
    distractor_soft_and = compute_soft_and_joint_score(distractor_scores, weights)

    # True multi-clause video: balanced moderate T1 and T2
    target_scores = [0.30, 0.28]
    target_soft_and = compute_soft_and_joint_score(target_scores, weights)

    # Target multi-clause soft-AND must beat single-clause distractor
    assert target_soft_and > distractor_soft_and
    # Verify exact geometric mean properties
    expected_target = math.sqrt((0.30 + 1e-4) * (0.28 + 1e-4))
    assert math.isclose(target_soft_and, expected_target, rel_tol=1e-4)
