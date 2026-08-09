"""Bounded structural and paired-language diagnostics for Stage 1C."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

DIAGNOSTIC_CUTOFFS = (5, 10, 20, 50)


def _prefix(items: list[dict[str, Any]], cutoff: int) -> list[dict[str, Any]]:
    return items[: min(cutoff, len(items))]


def initial_frame_diagnostics(frames: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for cutoff in DIAGNOSTIC_CUTOFFS:
        selected = _prefix(frames, cutoff)
        count = sum(
            int(item["n"]) == 1 and int(item["original_frame_idx"]) == 0
            for item in selected
        )
        result[f"initial_frame_count_top{cutoff}"] = count
        result[f"initial_frame_rate_top{cutoff}"] = count / len(selected) if selected else 0.0
    return result


def video_concentration_diagnostics(frames: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for cutoff in DIAGNOSTIC_CUTOFFS:
        selected = _prefix(frames, cutoff)
        counts = Counter(str(item["video_id"]) for item in selected)
        result[f"unique_videos_top{cutoff}"] = len(counts)
        if cutoff in {20, 50}:
            maximum = max(counts.values(), default=0)
            result[f"max_frames_from_single_video_top{cutoff}"] = maximum
            result[f"top_video_share_top{cutoff}"] = (
                maximum / len(selected) if selected else 0.0
            )
    return result


def exact_vector_diagnostics(vectors: np.ndarray) -> dict[str, Any]:
    matrix = np.asarray(vectors)
    if matrix.ndim != 2:
        raise ValueError("Returned stored vectors must be a rank-2 matrix")
    result: dict[str, Any] = {"equality_basis": "EXACT_CANONICAL_STORED_VECTOR_BYTES"}
    for cutoff in DIAGNOSTIC_CUTOFFS:
        selected = matrix[: min(cutoff, len(matrix))]
        fingerprints = [np.ascontiguousarray(row).tobytes() for row in selected]
        unique = len(set(fingerprints))
        result[f"unique_exact_vectors_top{cutoff}"] = unique
        if cutoff in {20, 50}:
            result[f"exact_duplicate_rows_top{cutoff}"] = len(selected) - unique
            result[f"exact_duplicate_rate_top{cutoff}"] = (
                (len(selected) - unique) / len(selected) if len(selected) else 0.0
            )
    return result


def score_diagnostics(frames: list[dict[str, Any]]) -> dict[str, Any]:
    scores = np.asarray([item["score"] for item in frames], dtype=np.float64)
    result: dict[str, Any] = {"top1_score": float(scores[0]) if len(scores) else None}
    for cutoff in DIAGNOSTIC_CUTOFFS:
        selected = scores[: min(cutoff, len(scores))]
        result[f"top{cutoff}_score_mean"] = (
            float(np.mean(selected)) if len(selected) else None
        )
    for target in (2, 5, 20):
        result[f"score_gap_1_{target}"] = (
            float(scores[0] - scores[target - 1]) if len(scores) >= target else None
        )
    return result


def query_diagnostics(
    frames: list[dict[str, Any]], stored_vectors: np.ndarray
) -> dict[str, Any]:
    return {
        **initial_frame_diagnostics(frames),
        **video_concentration_diagnostics(frames),
        **exact_vector_diagnostics(stored_vectors),
        **score_diagnostics(frames),
    }


def _overlap(left: list[Any], right: list[Any], cutoff: int) -> tuple[int, float]:
    left_set, right_set = set(left[:cutoff]), set(right[:cutoff])
    intersection = len(left_set & right_set)
    union = len(left_set | right_set)
    return intersection, intersection / union if union else 0.0


def paired_language_diagnostic(
    pair_id: str,
    en_embedding: np.ndarray,
    vi_embedding: np.ndarray,
    en_frames: list[dict[str, Any]],
    vi_frames: list[dict[str, Any]],
) -> dict[str, Any]:
    left = np.asarray(en_embedding, dtype=np.float32)
    right = np.asarray(vi_embedding, dtype=np.float32)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    cosine = float(np.dot(left, right) / denominator) if denominator else None
    result: dict[str, Any] = {
        "pair_id": pair_id,
        "en_query_id": en_frames[0]["query_id"] if en_frames else None,
        "vi_query_id": vi_frames[0]["query_id"] if vi_frames else None,
        "text_embedding_cosine_en_vi": cosine,
        "diagnostic_only": True,
    }
    en_rows = [item["global_row"] for item in en_frames]
    vi_rows = [item["global_row"] for item in vi_frames]
    en_videos = [item["video_id"] for item in en_frames]
    vi_videos = [item["video_id"] for item in vi_frames]
    for cutoff in (5, 10, 20, 50):
        count, jaccard = _overlap(en_rows, vi_rows, cutoff)
        result[f"top{cutoff}_global_row_overlap_count"] = count
        if cutoff <= 20:
            result[f"top{cutoff}_global_row_jaccard"] = jaccard
    for cutoff in (5, 10, 20):
        count, jaccard = _overlap(en_videos, vi_videos, cutoff)
        result[f"top{cutoff}_video_overlap_count"] = count
        result[f"top{cutoff}_video_jaccard"] = jaccard
    return result


def numeric_summary(values: list[float]) -> dict[str, float | None]:
    matrix = np.asarray(values, dtype=np.float64)
    if not len(matrix):
        return {"min": None, "max": None, "mean": None, "median": None}
    return {
        "min": float(np.min(matrix)),
        "max": float(np.max(matrix)),
        "mean": float(np.mean(matrix)),
        "median": float(np.median(matrix)),
    }

