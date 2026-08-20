"""
Unit tests for Stage 3C Robust Calibrated Sweep Diagnostic Tool
=================================================================
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

TESTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TESTS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from scripts.build_event_boundaries import apply_robust_calibrated_fusion
    from scripts.sweep_event_thresholds import evaluate_calibrated_configuration
except BaseException:
    import importlib
    mod_b = importlib.import_module("scripts.build_event_boundaries")
    mod_s = importlib.import_module("scripts.sweep_event_thresholds")
    apply_robust_calibrated_fusion = mod_b.apply_robust_calibrated_fusion
    evaluate_calibrated_configuration = mod_s.evaluate_calibrated_configuration


def test_apply_robust_calibrated_fusion():
    """Test Robust Z-score normalization & Sigmoidal Boundary Evidence calculation."""
    df_sim = pd.DataFrame({
        "video_id": ["V1", "V1", "V1"],
        "shot_i": [0, 1, 2],
        "shot_next": [1, 2, 3],
        "visual_similarity": [0.0, 0.5, 1.0],
        "semantic_similarity": [0.90, 0.95, 1.0],
    })

    df_cal = apply_robust_calibrated_fusion(df_sim, vis_weight=0.5, sem_weight=0.5)

    assert "visual_z" in df_cal.columns
    assert "semantic_z" in df_cal.columns
    assert "visual_boundary_evidence" in df_cal.columns
    assert "semantic_boundary_evidence" in df_cal.columns
    assert "boundary_score" in df_cal.columns

    # Boundary score should be strictly bounded between 0.0 and 1.0
    scores = df_cal["boundary_score"].to_numpy()
    assert np.all((scores >= 0.0) & (scores <= 1.0))


def test_evaluate_calibrated_configuration():
    """Test evaluation of a single calibrated configuration over sample shots."""
    df_shots = pd.DataFrame([
        {"video_id": "V1", "shot_id": 0, "start_sec": 0.0},
        {"video_id": "V1", "shot_id": 1, "start_sec": 5.0},
        {"video_id": "V1", "shot_id": 2, "start_sec": 10.0},
    ])
    df_sim = pd.DataFrame([
        {"video_id": "V1", "shot_i": 0, "shot_next": 1, "boundary_score": 0.80},
        {"video_id": "V1", "shot_i": 1, "shot_next": 2, "boundary_score": 0.30},
    ])

    res = evaluate_calibrated_configuration(
        df_sim, df_shots, threshold=0.60, vis_weight=0.5, sem_weight=0.5
    )

    assert res["threshold"] == 0.60
    assert res["total_boundaries"] == 1
    assert res["total_events"] == 2
    assert res["events_per_video_mean"] == 2.0
    assert res["shots_per_event_mean"] == 1.5
