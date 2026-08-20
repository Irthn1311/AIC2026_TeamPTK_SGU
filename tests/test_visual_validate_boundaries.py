"""
Unit tests for Stage 3C Visual Validation & Adaptive Merging Tool
===================================================================
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
    from scripts.visual_validate_boundaries import adaptive_merge_short_events
except BaseException:
    import importlib
    mod = importlib.import_module("scripts.visual_validate_boundaries")
    adaptive_merge_short_events = mod.adaptive_merge_short_events


def test_adaptive_merge_short_events():
    """Test adaptive merging of 1-shot events into nearest neighbor."""
    df_shots = pd.DataFrame([
        {"video_id": "V1", "shot_id": 0, "start_sec": 0.0, "end_sec": 5.0, "representative_keyframe": "kf0"},
        {"video_id": "V1", "shot_id": 1, "start_sec": 5.0, "end_sec": 7.0, "representative_keyframe": "kf1"},
        {"video_id": "V1", "shot_id": 2, "start_sec": 7.0, "end_sec": 12.0, "representative_keyframe": "kf2"},
    ])
    # Pair (0, 1) boundary score 0.80 > 0.65 -> Trigger Boundary
    # Pair (1, 2) boundary score 0.40 < 0.65 -> No boundary
    df_sim = pd.DataFrame([
        {"shot_i": 0, "shot_next": 1, "boundary_score": 0.80},
        {"shot_i": 1, "shot_next": 2, "boundary_score": 0.40},
    ])

    events, stats = adaptive_merge_short_events("V1", df_shots, df_sim, threshold=0.65)

    assert len(events) >= 1
    assert "raw_1_shot_cnt" in stats
    assert "merged_cnt" in stats
