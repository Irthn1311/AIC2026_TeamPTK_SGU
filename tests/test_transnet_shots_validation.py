"""
Unit Test for TransNetV2 Shot Detection Validation and Formatting
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_transnet_shots_873 import validate_video_shots, natural_video_key


def test_natural_video_key():
    items = ["L21_V010", "L21_V002", "L21_V001", "L30_V005"]
    sorted_items = sorted(items, key=natural_video_key)
    assert sorted_items == ["L21_V001", "L21_V002", "L21_V010", "L30_V005"], f"Unexpected sort: {sorted_items}"
    print("[TEST] natural_video_key: PASS")


def test_validate_video_shots():
    # Valid dataframe
    df_valid = pd.DataFrame([
        {
            "video_id": "L21_V001",
            "shot_id": 0,
            "start_frame": 0,
            "end_frame": 100,
            "start_sec": 0.0,
            "end_sec": 3.333,
            "num_frames": 101,
            "duration_sec": 3.367,
            "source_fps": 30.0,
            "frame_count": 300,
            "detector_backend": "transnetv2",
            "confidence": 0.95,
            "video_path": "/fake/L21_V001.mp4",
        },
        {
            "video_id": "L21_V001",
            "shot_id": 1,
            "start_frame": 101,
            "end_frame": 299,
            "start_sec": 3.367,
            "end_sec": 9.967,
            "num_frames": 199,
            "duration_sec": 6.633,
            "source_fps": 30.0,
            "frame_count": 300,
            "detector_backend": "transnetv2",
            "confidence": 0.88,
            "video_path": "/fake/L21_V001.mp4",
        },
    ])
    is_valid, errs = validate_video_shots(df_valid)
    assert is_valid, f"Expected valid dataframe but got errors: {errs}"
    print("[TEST] validate_video_shots (valid): PASS")

    # Invalid dataframe (overlapping frames)
    df_invalid = df_valid.copy()
    df_invalid.loc[1, "start_frame"] = 50  # Overlap!
    is_valid, errs = validate_video_shots(df_invalid)
    assert not is_valid, "Expected invalid dataframe due to overlap"
    print("[TEST] validate_video_shots (invalid overlap): PASS")


if __name__ == "__main__":
    test_natural_video_key()
    test_validate_video_shots()
    print("All unit tests passed!")
