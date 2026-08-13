from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class MappingRecord:
    video_id: str
    feature_index: int
    keyframe_name: str
    keyframe_path: str
    frame_idx: int
    timestamp_seconds: float
    fps: float | None


def load_keyframe_mapping(mapping_path: str | Path, keyframe_root: str | Path) -> pd.DataFrame:
    mapping_path = Path(mapping_path)
    df = pd.read_csv(mapping_path)
    if "n" not in df.columns:
        raise ValueError(f"Missing 'n' column in mapping file: {mapping_path}")
    if "frame_idx" not in df.columns:
        raise ValueError(f"Missing 'frame_idx' column in mapping file: {mapping_path}")
    if "pts_time" not in df.columns:
        raise ValueError(f"Missing 'pts_time' column in mapping file: {mapping_path}")
    video_id = mapping_path.stem
    keyframe_root = Path(keyframe_root)
    df = df.copy()
    df["video_id"] = video_id
    df["feature_index"] = df["n"].astype(int) - 1
    df["keyframe_name"] = df["n"].astype(int).map(lambda x: f"{x:03d}.jpg")
    df["keyframe_directory"] = str(keyframe_root / video_id)
    df["keyframe_path"] = df["keyframe_name"].map(lambda x: str(keyframe_root / video_id / x))
    df["timestamp_seconds"] = df["pts_time"].astype(float)
    df["frame_idx"] = df["frame_idx"].astype(int)
    if "fps" in df.columns:
        df["fps"] = df["fps"].astype(float)
    return df

