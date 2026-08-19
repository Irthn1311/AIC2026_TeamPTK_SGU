"""
Unit Tests for Stage 2 Shot ↔ BTC Keyframe Temporal Alignment
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_shot_btc_alignment_873 import align_video_keyframes, normalize_btc_map


def test_alignment_logic_various_cases():
    # Setup shots (2 adjacent shots)
    shots_df = pd.DataFrame([
        {
            "video_id": "L21_V001",
            "shot_id": 0,
            "start_frame": 0,
            "end_frame": 99,
            "start_sec": 0.0,
            "end_sec": 3.3,
            "source_fps": 30.0,
        },
        {
            "video_id": "L21_V001",
            "shot_id": 1,
            "start_frame": 100,
            "end_frame": 199,
            "start_sec": 3.333,
            "end_sec": 6.633,
            "source_fps": 30.0,
        },
    ])

    # Setup keyframes for test cases
    raw_keyframes_df = pd.DataFrame([
        # Case 1: Inside shot 0 -> TIME=True, FRAME=True -> AGREE
        {"video_id": "L21_V001", "global_id": "kf_0", "frame_idx": 30, "timestamp_sec": 1.0},
        # Case 2: Exact boundary of shot 0 -> AGREE
        {"video_id": "L21_V001", "global_id": "kf_1", "frame_idx": 0, "timestamp_sec": 0.0},
        # Case 3: Inside shot 1 -> AGREE
        {"video_id": "L21_V001", "global_id": "kf_2", "frame_idx": 150, "timestamp_sec": 5.0},
        # Case 4: Outside any shot -> UNALIGNED
        {"video_id": "L21_V001", "global_id": "kf_3", "frame_idx": 300, "timestamp_sec": 10.0},
        # Case 5: Disagreement case (TIME in shot 0, FRAME in shot 1 due to offset/skew)
        {"video_id": "L21_V001", "global_id": "kf_4", "frame_idx": 105, "timestamp_sec": 2.5},
    ])

    keyframes_df = normalize_btc_map(raw_keyframes_df)

    df_aligned, df_unassigned, df_shots_no_kf, df_disagreements, stats = align_video_keyframes(
        "L21_V001", shots_df, keyframes_df
    )

    # Check total aligned keyframes (4 aligned, 1 unassigned)
    assert len(df_aligned) == 4, f"Expected 4 aligned keyframes, got {len(df_aligned)}"
    assert len(df_unassigned) == 1, f"Expected 1 unassigned keyframe, got {len(df_unassigned)}"
    assert df_unassigned.iloc[0]["keyframe_id"] == "kf_3"

    # Check duplicate assignment prevention
    assert not df_aligned.duplicated(subset=["keyframe_id"]).any(), "Found duplicate keyframe assignments!"

    # Check AGREE / DISAGREE classification
    agree_kf_ids = set(df_aligned[df_aligned["alignment_status"] == "AGREE"]["keyframe_id"])
    disagree_kf_ids = set(df_aligned[df_aligned["alignment_status"] == "DISAGREE"]["keyframe_id"])

    assert "kf_0" in agree_kf_ids, "kf_0 should AGREE"
    assert "kf_1" in agree_kf_ids, "kf_1 should AGREE"
    assert "kf_2" in agree_kf_ids, "kf_2 should AGREE"
    assert "kf_4" in disagree_kf_ids, "kf_4 should DISAGREE"

    print("[TEST] test_alignment_logic_various_cases: PASS")


def test_fps_variations():
    for fps in [25.0, 29.97, 30.0]:
        shots_df = pd.DataFrame([
            {
                "video_id": "TEST_FPS",
                "shot_id": 0,
                "start_frame": 0,
                "end_frame": int(fps * 2) - 1,
                "start_sec": 0.0,
                "end_sec": 2.0,
                "source_fps": fps,
            }
        ])
        raw_kf = pd.DataFrame([
            {"video_id": "TEST_FPS", "global_id": "kf_a", "frame_idx": int(fps), "timestamp_sec": 1.0}
        ])
        kf_df = normalize_btc_map(raw_kf)
        df_aligned, _, _, _, _ = align_video_keyframes("TEST_FPS", shots_df, kf_df)
        assert len(df_aligned) == 1
        assert df_aligned.iloc[0]["alignment_status"] == "AGREE"
        assert abs(df_aligned.iloc[0]["relative_position"] - 0.5) < 1e-3

    print("[TEST] test_fps_variations (25.0, 29.97, 30.0): PASS")


if __name__ == "__main__":
    test_alignment_logic_various_cases()
    test_fps_variations()
    print("All Stage 2 alignment unit tests PASSED successfully!")
