"""Full-corpus RT1 aggregation over the frozen Stage 1 catalog."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .dante import dante_monotonic_dp


@dataclass(frozen=True)
class VideoRows:
    video_id: str
    rows: np.ndarray
    source_was_contiguous: bool


def build_video_row_groups(catalog: Any) -> list[VideoRows]:
    """Build canonical per-video row order from ``n`` then global row."""

    video_index = np.asarray(catalog.video_index, dtype=np.int64)
    ordinals = np.asarray(catalog.n, dtype=np.int64)
    if video_index.ndim != 1 or ordinals.shape != video_index.shape:
        raise ValueError("Stage 1 catalog arrays are inconsistent")
    if len(video_index) == 0 or np.any(video_index < 0):
        raise ValueError("Stage 1 catalog has invalid video indexes")
    if int(video_index.max()) >= len(catalog.video_table):
        raise ValueError("Stage 1 catalog references an unknown video")
    groups = []
    covered = np.zeros(len(video_index), dtype=np.uint8)
    for table_index in np.unique(video_index):
        rows = np.flatnonzero(video_index == table_index).astype(np.int64)
        covered[rows] += 1
        source_was_contiguous = bool(len(rows) == 1 or np.all(np.diff(rows) == 1))
        order = np.lexsort((rows, ordinals[rows]))
        rows = rows[order]
        video_id = str(catalog.video_table[int(table_index)]["video_id"])
        groups.append(VideoRows(video_id, rows, source_was_contiguous))
    if not np.all(covered == 1):
        raise RuntimeError("Stage 1 catalog rows were not partitioned exactly once")
    return sorted(groups, key=lambda group: group.video_id)


def _anchor(
    catalog: Any,
    *,
    global_row: int,
    catalog_position: int,
    event_id: str,
    similarity: float,
) -> dict[str, Any]:
    mapped = catalog.map_row(global_row)
    return {
        "event_id": event_id,
        "catalog_position": catalog_position,
        "global_row": global_row,
        "n": int(mapped["n"]),
        "original_frame_idx": int(mapped["original_frame_idx"]),
        "keyframe_relative_path": mapped.get("keyframe_relative_path"),
        "event_similarity": similarity,
    }


def rank_unordered_event_max(
    scores: np.ndarray,
    event_ids: list[str],
    groups: list[VideoRows],
    catalog: Any,
) -> list[dict[str, Any]]:
    """Rank videos by mean independent per-event maximum similarity."""

    matrix = np.asarray(scores, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != len(event_ids):
        raise ValueError("event score matrix does not match event IDs")
    output = []
    for group in groups:
        if len(group.rows) == 0 or int(group.rows[-1]) >= matrix.shape[1]:
            raise ValueError("video rows exceed the full score matrix")
        local = matrix[:, group.rows]
        positions = np.argmax(local, axis=1)
        anchors = [
            _anchor(
                catalog,
                global_row=int(group.rows[position]),
                catalog_position=int(position),
                event_id=event_ids[event_index],
                similarity=float(local[event_index, position]),
            )
            for event_index, position in enumerate(positions)
        ]
        monotonic = all(
            int(left) < int(right)
            for left, right in zip(positions[:-1], positions[1:], strict=True)
        )
        output.append(
            {
                "video_id": group.video_id,
                "unordered_score": float(np.mean(np.max(local, axis=1), dtype=np.float64)),
                "event_count": len(event_ids),
                "event_best": anchors,
                "independent_argmax_order_is_monotonic": monotonic,
            }
        )
    ordered = sorted(output, key=lambda item: (-item["unordered_score"], item["video_id"]))
    return [{"video_rank": rank, **item} for rank, item in enumerate(ordered, start=1)]


def rank_dante_dp(
    scores: np.ndarray,
    event_ids: list[str],
    groups: list[VideoRows],
    catalog: Any,
    *,
    distance_lambda: float,
) -> list[dict[str, Any]]:
    """Rank videos using strict monotonic DANTE alignment over canonical positions."""

    matrix = np.asarray(scores, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != len(event_ids):
        raise ValueError("event score matrix does not match event IDs")
    output = []
    for group in groups:
        if len(group.rows) == 0 or int(group.rows[-1]) >= matrix.shape[1]:
            raise ValueError("video rows exceed the full score matrix")
        local = matrix[:, group.rows]
        alignment = dante_monotonic_dp(local, distance_lambda)
        if alignment is None:
            continue
        chain = [
            _anchor(
                catalog,
                global_row=int(group.rows[position]),
                catalog_position=position,
                event_id=event_ids[event_index],
                similarity=float(local[event_index, position]),
            )
            for event_index, position in enumerate(alignment.positions)
        ]
        output.append(
            {
                "video_id": group.video_id,
                "dante_score": alignment.score,
                "lambda": distance_lambda,
                "event_count": len(event_ids),
                "chain": chain,
                "first_position": alignment.positions[0],
                "last_position": alignment.positions[-1],
                "span_in_keyframes": alignment.positions[-1] - alignment.positions[0],
                "strictly_increasing_positions": True,
            }
        )
    ordered = sorted(output, key=lambda item: (-item["dante_score"], item["video_id"]))
    return [{"video_rank": rank, **item} for rank, item in enumerate(ordered, start=1)]


def top_k_video_overlap(
    left: list[dict[str, Any]], right: list[dict[str, Any]], cutoffs: tuple[int, ...]
) -> dict[str, dict[str, float | int]]:
    output: dict[str, dict[str, float | int]] = {}
    for cutoff in cutoffs:
        left_ids = {str(item["video_id"]) for item in left[:cutoff]}
        right_ids = {str(item["video_id"]) for item in right[:cutoff]}
        overlap = len(left_ids & right_ids)
        union = len(left_ids | right_ids)
        output[str(cutoff)] = {
            "overlap_count": overlap,
            "left_count": len(left_ids),
            "right_count": len(right_ids),
            "jaccard": overlap / union if union else 0.0,
        }
    return output


__all__ = [
    "VideoRows",
    "build_video_row_groups",
    "rank_dante_dp",
    "rank_unordered_event_max",
    "top_k_video_overlap",
]
