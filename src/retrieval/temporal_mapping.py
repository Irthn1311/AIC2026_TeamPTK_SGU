from __future__ import annotations

from typing import Any, Literal

import pandas as pd


TemporalMappingMode = Literal["interval", "start_time_window"]


def normalize_temporal_interval(start_time: float, end_time: float) -> tuple[float, float]:
    """Return an ordered [start, end] interval even when input metadata is reversed."""
    start = float(start_time)
    end = float(end_time)
    if end < start:
        start, end = end, start
    return start, end


def keyframe_matches_ocr_segment(
    keyframe_timestamp: float,
    segment_start_time: float,
    segment_end_time: float,
    margin_sec: float = 1.0,
    mode: TemporalMappingMode = "interval",
) -> bool:
    """Check whether a keyframe timestamp should receive an OCR segment score."""
    t = float(keyframe_timestamp)
    margin = max(0.0, float(margin_sec))
    start, end = normalize_temporal_interval(segment_start_time, segment_end_time)

    if mode == "start_time_window":
        return abs(t - start) <= margin
    if mode != "interval":
        raise ValueError(f"Unsupported OCR temporal mapping mode: {mode}")
    return (start - margin) <= t <= (end + margin)


def ocr_segment_time_mask(
    keyframes_df: pd.DataFrame,
    video_id: str,
    segment: dict[str, Any],
    margin_sec: float = 1.0,
    mode: TemporalMappingMode = "interval",
) -> pd.Series:
    """Build a boolean mask selecting keyframes covered by an OCR segment."""
    if "video_id" not in keyframes_df.columns or "timestamp_seconds" not in keyframes_df.columns:
        raise ValueError("keyframes_df must contain video_id and timestamp_seconds columns")

    start = float(segment.get("start_time", 0.0))
    end = float(segment.get("end_time", start))
    margin = max(0.0, float(margin_sec))
    same_video = keyframes_df["video_id"].astype(str) == str(video_id)

    if mode == "start_time_window":
        return same_video & ((keyframes_df["timestamp_seconds"].astype(float) - start).abs() <= margin)
    if mode != "interval":
        raise ValueError(f"Unsupported OCR temporal mapping mode: {mode}")

    start, end = normalize_temporal_interval(start, end)
    ts = keyframes_df["timestamp_seconds"].astype(float)
    return same_video & (ts >= start - margin) & (ts <= end + margin)


def ocr_context_interval_mask(
    ocr_segments_df: pd.DataFrame,
    video_id: str,
    target_timestamp: float,
    window_sec: float = 10.0,
) -> pd.Series:
    """Select OCR segments whose intervals overlap the timestamp context window."""
    if "video_id" not in ocr_segments_df.columns or "start_time" not in ocr_segments_df.columns:
        raise ValueError("ocr_segments_df must contain video_id and start_time columns")

    same_video = ocr_segments_df["video_id"].astype(str) == str(video_id)
    start = ocr_segments_df["start_time"].astype(float)
    end = ocr_segments_df["end_time"].astype(float) if "end_time" in ocr_segments_df.columns else start
    left = float(target_timestamp) - max(0.0, float(window_sec))
    right = float(target_timestamp) + max(0.0, float(window_sec))
    return same_video & (start <= right) & (end >= left)
