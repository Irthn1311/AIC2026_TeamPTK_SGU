"""
Unit tests for EventGraph Stages 3B, 3C, 3D Pipeline
=====================================================
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
    from scripts.build_event_boundaries import (
        compute_vector_similarity,
        compute_video_adjacent_similarities,
        detect_video_event_boundaries,
    )
except BaseException:
    import importlib
    mod = importlib.import_module("scripts.build_event_boundaries")
    compute_vector_similarity = mod.compute_vector_similarity
    compute_video_adjacent_similarities = mod.compute_video_adjacent_similarities
    detect_video_event_boundaries = mod.detect_video_event_boundaries


def test_compute_vector_similarity():
    """Test vector cosine similarity calculation."""
    v1 = [1.0, 0.0, 0.0]
    v2 = [1.0, 0.0, 0.0]
    v3 = [0.0, 1.0, 0.0]

    assert pytest.approx(compute_vector_similarity(v1, v2), 1e-5) == 1.0
    assert pytest.approx(compute_vector_similarity(v1, v3), 1e-5) == 0.0


def test_compute_video_adjacent_similarities():
    """Test adjacent shot similarity calculation within a single video."""
    df_shots = pd.DataFrame([
        {
            "video_id": "V001",
            "shot_id": 0,
            "start_sec": 0.0,
            "end_sec": 5.0,
            "visual_embedding": [1.0, 0.0],
            "semantic_embedding": [1.0, 0.0],
        },
        {
            "video_id": "V001",
            "shot_id": 1,
            "start_sec": 5.0,
            "end_sec": 10.0,
            "visual_embedding": [1.0, 0.0],
            "semantic_embedding": [1.0, 0.0],
        },
        {
            "video_id": "V001",
            "shot_id": 2,
            "start_sec": 10.0,
            "end_sec": 15.0,
            "visual_embedding": [0.0, 1.0],
            "semantic_embedding": [0.0, 1.0],
        },
    ])

    df_sim = compute_video_adjacent_similarities("V001", df_shots)

    assert len(df_sim) == 2
    # Pair (0, 1) should have similarity 1.0
    assert pytest.approx(df_sim.iloc[0]["fused_similarity"], 1e-5) == 1.0
    # Pair (1, 2) should have similarity 0.0
    assert pytest.approx(df_sim.iloc[1]["fused_similarity"], 1e-5) == 0.0


def test_detect_video_event_boundaries():
    """Test Event boundary detection and Event construction."""
    df_shots = pd.DataFrame([
        {
            "video_id": "V001",
            "shot_id": 0,
            "start_frame": 0,
            "end_frame": 100,
            "start_sec": 0.0,
            "end_sec": 5.0,
            "visual_embedding": [1.0, 0.0],
            "semantic_embedding": [1.0, 0.0],
            "representative_keyframe": "kf_0",
        },
        {
            "video_id": "V001",
            "shot_id": 1,
            "start_frame": 101,
            "end_frame": 200,
            "start_sec": 5.0,
            "end_sec": 10.0,
            "visual_embedding": [1.0, 0.0],
            "semantic_embedding": [1.0, 0.0],
            "representative_keyframe": "kf_1",
        },
        {
            "video_id": "V001",
            "shot_id": 2,
            "start_frame": 201,
            "end_frame": 300,
            "start_sec": 10.0,
            "end_sec": 15.0,
            "visual_embedding": [0.0, 1.0],
            "semantic_embedding": [0.0, 1.0],
            "representative_keyframe": "kf_2",
        },
    ])

    df_sim = compute_video_adjacent_similarities("V001", df_shots)
    df_sim_updated, events = detect_video_event_boundaries(
        df_shots, df_sim, boundary_threshold=0.50
    )

    assert len(events) == 2
    
    # First event should contain shots 0 and 1
    event_0 = events[0]
    assert event_0["event_id"] == "V001_E000"
    assert event_0["start_shot_id"] == 0
    assert event_0["end_shot_id"] == 1
    assert event_0["num_shots"] == 2
    assert event_0["shot_ids"] == [0, 1]
    assert event_0["representative_keyframes"] == ["kf_0", "kf_1"]

    # Second event should contain shot 2
    event_1 = events[1]
    assert event_1["event_id"] == "V001_E001"
    assert event_1["start_shot_id"] == 2
    assert event_1["end_shot_id"] == 2
    assert event_1["num_shots"] == 1
    assert event_1["shot_ids"] == [2]
