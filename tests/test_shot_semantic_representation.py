"""
Unit tests for Stage 3A Shot Semantic Representation Builder
============================================================
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
    from scripts.build_shot_semantic_representation import (
        aggregate_visual_embeddings,
        compute_temporal_attention_weights,
        process_shot_group,
        ShotFeatureExtractor,
    )
except BaseException:
    # Fallback import if torch DLL causes top-level issue on Windows
    import importlib
    mod = importlib.import_module("scripts.build_shot_semantic_representation")
    aggregate_visual_embeddings = mod.aggregate_visual_embeddings
    compute_temporal_attention_weights = mod.compute_temporal_attention_weights
    process_shot_group = mod.process_shot_group
    ShotFeatureExtractor = mod.ShotFeatureExtractor


def test_compute_temporal_attention_weights():
    """Test Gaussian temporal decay attention weights calculation."""
    timestamps = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    center_sec = 1.0
    duration_sec = 2.0

    weights = compute_temporal_attention_weights(timestamps, center_sec, duration_sec)

    assert len(weights) == 3
    assert np.isclose(np.sum(weights), 1.0)
    # Center frame (timestamp=1.0) must have highest weight
    assert weights[1] > weights[0]
    assert weights[1] > weights[2]


def test_aggregate_visual_embeddings():
    """Test weighted mean pooling and L2 normalization."""
    embeds = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=np.float32)
    weights = np.array([0.5, 0.5], dtype=np.float32)

    agg = aggregate_visual_embeddings(embeds, weights)

    assert agg.shape == (3,)
    # Verify unit L2 norm
    norm = np.linalg.norm(agg)
    assert np.isclose(norm, 1.0)


def test_process_shot_group():
    """Test end-to-end processing of a single shot group."""
    shot_row = pd.Series({
        "video_id": "L21_V001",
        "shot_id": 0,
        "start_sec": 0.0,
        "end_sec": 5.0,
        "duration_sec": 5.0,
    })

    group_df = pd.DataFrame([
        {
            "keyframe_id": "kf_0",
            "keyframe_timestamp_sec": 1.0,
            "ocr_text": "CẢNH BÁO",
            "objects": ["car", "person"],
        },
        {
            "keyframe_id": "kf_1",
            "keyframe_timestamp_sec": 2.5,
            "ocr_text": "CẢNH BÁO NGUY HIỂM",
            "objects": ["person"],
        },
    ])

    extractor = ShotFeatureExtractor(device="cpu")
    result = process_shot_group(shot_row, group_df, extractor)

    assert result["video_id"] == "L21_V001"
    assert result["shot_id"] == 0
    assert result["num_keyframes"] == 2
    assert result["representative_keyframe"] == "kf_1"  # 2.5s is closest to center 2.5s
    assert len(result["visual_embedding"]) == 512
    assert len(result["semantic_embedding"]) == 768
    assert "CẢNH BÁO" in result["ocr_text"]
    assert "car" in result["objects"]
    assert "person" in result["objects"]
