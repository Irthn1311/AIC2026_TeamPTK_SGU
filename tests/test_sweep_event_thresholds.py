"""
Unit tests for Stage 3C Threshold Sweep Diagnostic Tool
=========================================================
"""

import sys
from pathlib import Path
import pandas as pd
import pytest

TESTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TESTS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from scripts.sweep_event_thresholds import (
        analyze_similarity_modalities,
        evaluate_single_threshold,
    )
except BaseException:
    import importlib
    mod = importlib.import_module("scripts.sweep_event_thresholds")
    analyze_similarity_modalities = mod.analyze_similarity_modalities
    evaluate_single_threshold = mod.evaluate_single_threshold


def test_analyze_similarity_modalities():
    """Test distribution calculation for visual, semantic, and combined similarities."""
    df_sim = pd.DataFrame({
        "visual_similarity": [0.3, 0.5, 0.7],
        "semantic_similarity": [0.4, 0.6, 0.8],
        "fused_similarity": [0.35, 0.55, 0.75],
    })

    stats = analyze_similarity_modalities(df_sim)
    assert "visual_similarity" in stats
    assert "semantic_similarity" in stats
    assert "combined_similarity" in stats
    assert pytest.approx(stats["visual_similarity"]["mean"], 1e-4) == 0.5
    assert pytest.approx(stats["combined_similarity"]["q50_median"], 1e-4) == 0.55


def test_evaluate_single_threshold():
    """Test evaluation of a single threshold over sample shots."""
    df_shots = pd.DataFrame([
        {"video_id": "V1", "shot_id": 0, "start_sec": 0.0},
        {"video_id": "V1", "shot_id": 1, "start_sec": 5.0},
        {"video_id": "V1", "shot_id": 2, "start_sec": 10.0},
    ])
    df_sim = pd.DataFrame([
        {"video_id": "V1", "shot_id": 0, "fused_similarity": 0.60},
        {"video_id": "V1", "shot_id": 1, "fused_similarity": 0.40},
    ])

    res = evaluate_single_threshold(df_sim, df_shots, threshold=0.50)

    assert res["threshold"] == 0.50
    assert res["total_boundaries"] == 1
    assert res["total_events"] == 2
    assert res["events_per_video_mean"] == 2.0
    assert res["shots_per_event_mean"] == 1.5
