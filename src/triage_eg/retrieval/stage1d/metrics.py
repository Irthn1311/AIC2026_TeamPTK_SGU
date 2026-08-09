"""Three-arm ranking and CLIP text-space diagnostics for Stage 1D."""

from __future__ import annotations

from typing import Any

import numpy as np

from triage_eg.retrieval.stage1c.metrics import numeric_summary

CUTOFFS = (5, 10, 20, 50)


def _overlap(left: list[Any], right: list[Any], cutoff: int) -> dict[str, Any]:
    left_set = set(left[:cutoff])
    right_set = set(right[:cutoff])
    overlap = len(left_set & right_set)
    union = len(left_set | right_set)
    return {"overlap_count": overlap, "jaccard": overlap / union if union else 0.0}


def arm_overlap(
    left_frames: list[dict[str, Any]],
    right_frames: list[dict[str, Any]],
) -> dict[str, Any]:
    left_rows = [item["global_row"] for item in left_frames]
    right_rows = [item["global_row"] for item in right_frames]
    left_videos = [item["video_id"] for item in left_frames]
    right_videos = [item["video_id"] for item in right_frames]
    return {
        "frame": {
            f"top{cutoff}": _overlap(left_rows, right_rows, cutoff)
            for cutoff in CUTOFFS
        },
        "video": {
            f"top{cutoff}": _overlap(left_videos, right_videos, cutoff)
            for cutoff in CUTOFFS
        },
    }


def _cosine(left: np.ndarray, right: np.ndarray) -> float | None:
    a = np.asarray(left, dtype=np.float32)
    b = np.asarray(right, dtype=np.float32)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator else None


def pair_comparison(
    *,
    pair_id: str,
    category: str,
    difficulty: str,
    en_text: str,
    vi_text: str,
    translated_text: str,
    en_embedding: np.ndarray,
    vi_embedding: np.ndarray,
    translated_embedding: np.ndarray,
    en_frames: list[dict[str, Any]],
    vi_frames: list[dict[str, Any]],
    translated_frames: list[dict[str, Any]],
    structural: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    en_vi = arm_overlap(en_frames, vi_frames)
    en_translated = arm_overlap(en_frames, translated_frames)
    vi_translated = arm_overlap(vi_frames, translated_frames)
    frame_delta = (
        en_translated["frame"]["top20"]["jaccard"]
        - en_vi["frame"]["top20"]["jaccard"]
    )
    video_delta = (
        en_translated["video"]["top20"]["jaccard"]
        - en_vi["video"]["top20"]["jaccard"]
    )
    return {
        "pair_id": pair_id,
        "category": category,
        "difficulty": difficulty,
        "en_text": en_text,
        "vi_text": vi_text,
        "translated_en_text": translated_text,
        "text_space": {
            "clip_text_cosine_en_vi": _cosine(en_embedding, vi_embedding),
            "clip_text_cosine_en_translated": _cosine(
                en_embedding, translated_embedding
            ),
            "clip_text_cosine_vi_translated": _cosine(
                vi_embedding, translated_embedding
            ),
            "diagnostic_only": True,
        },
        "ranking_alignment": {
            "en_direct_vs_vi_direct": en_vi,
            "en_direct_vs_vi_translated_en": en_translated,
            "vi_direct_vs_vi_translated_en": vi_translated,
            "en_vs_vi_top20_frame_jaccard": en_vi["frame"]["top20"]["jaccard"],
            "en_vs_translated_top20_frame_jaccard": en_translated["frame"]["top20"][
                "jaccard"
            ],
            "en_vs_vi_top20_video_jaccard": en_vi["video"]["top20"]["jaccard"],
            "en_vs_translated_top20_video_jaccard": en_translated["video"]["top20"][
                "jaccard"
            ],
            "translated_minus_vi_frame_jaccard": frame_delta,
            "translated_minus_vi_video_jaccard": video_delta,
            "interpretation": "RANKING_ALIGNMENT_DIAGNOSTIC",
        },
        "structural_summaries": structural,
        "human_review_status": "NOT_REVIEWED",
    }


def extended_numeric_summary(values: list[float]) -> dict[str, float | None]:
    base = numeric_summary(values)
    matrix = np.asarray(values, dtype=np.float64)
    return {
        **base,
        "p05": float(np.percentile(matrix, 5)) if len(matrix) else None,
        "p95": float(np.percentile(matrix, 95)) if len(matrix) else None,
    }


def aggregate_comparisons(comparisons: list[dict[str, Any]]) -> dict[str, Any]:
    def values(path: tuple[str, ...]) -> list[float]:
        output = []
        for item in comparisons:
            value: Any = item
            for name in path:
                value = value[name]
            output.append(float(value))
        return output

    return {
        "pairs_completed": len(comparisons),
        "text_space": {
            key: extended_numeric_summary(values(("text_space", key)))
            for key in (
                "clip_text_cosine_en_vi",
                "clip_text_cosine_en_translated",
                "clip_text_cosine_vi_translated",
            )
        },
        "frame_overlap": {
            "en_vs_vi_top20_jaccard": extended_numeric_summary(
                values(("ranking_alignment", "en_vs_vi_top20_frame_jaccard"))
            ),
            "en_vs_translated_top20_jaccard": extended_numeric_summary(
                values(("ranking_alignment", "en_vs_translated_top20_frame_jaccard"))
            ),
            "translated_minus_vi_top20_jaccard": extended_numeric_summary(
                values(("ranking_alignment", "translated_minus_vi_frame_jaccard"))
            ),
        },
        "video_overlap": {
            "en_vs_vi_top20_jaccard": extended_numeric_summary(
                values(("ranking_alignment", "en_vs_vi_top20_video_jaccard"))
            ),
            "en_vs_translated_top20_jaccard": extended_numeric_summary(
                values(("ranking_alignment", "en_vs_translated_top20_video_jaccard"))
            ),
            "translated_minus_vi_top20_jaccard": extended_numeric_summary(
                values(("ranking_alignment", "translated_minus_vi_video_jaccard"))
            ),
        },
    }

