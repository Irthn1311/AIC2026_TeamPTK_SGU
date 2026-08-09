"""Stage 2A runtime result records and non-mutating video view."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


def grouped_video_view(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group by first raw occurrence while preserving the raw ranking unchanged."""

    grouped: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for frame in frames:
        video_id = str(frame["video_id"])
        if video_id not in grouped:
            grouped[video_id] = {
                "video_id": video_id,
                "best_frame_rank": int(frame["rank"]),
                "best_global_row": int(frame["global_row"]),
                "best_n": int(frame["n"]),
                "best_original_frame_idx": int(frame["original_frame_idx"]),
                "best_score": float(frame["score"]),
                "frames_in_raw_results": 0,
            }
            order.append(video_id)
        grouped[video_id]["frames_in_raw_results"] += 1
    return [
        {"video_rank": rank, **grouped[video_id]} for rank, video_id in enumerate(order, start=1)
    ]


@dataclass(frozen=True)
class QueryResult:
    query_id: str
    request: dict[str, Any]
    language_resolution: dict[str, Any]
    encoding: dict[str, Any]
    ranked_frames: list[dict[str, Any]]
    ranked_videos: list[dict[str, Any]]
    latencies_ms: dict[str, float]
    output_root: Path


__all__ = ["QueryResult", "grouped_video_view"]
