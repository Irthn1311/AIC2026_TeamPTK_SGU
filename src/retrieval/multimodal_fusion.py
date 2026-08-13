from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from rapidfuzz import fuzz


def minmax_normalize(series: pd.Series, eps: float = 1e-12) -> pd.Series:
    if series.empty:
        return series
    mn = float(series.min())
    mx = float(series.max())
    if abs(mx - mn) < eps:
        return pd.Series([1.0] * len(series), index=series.index)
    return (series - mn) / (mx - mn + eps)


def lexical_similarity(query: str, text: str) -> float:
    if not query or not text:
        return 0.0
    return float(fuzz.token_sort_ratio(query, text)) / 100.0


def fuse_candidates(visual_df: pd.DataFrame, ocr_df: pd.DataFrame, query: str, visual_weight: float = 0.7, ocr_weight: float = 0.25, lexical_weight: float = 0.05, mode: str = "weighted_sum", rrf_k: int = 60) -> pd.DataFrame:
    if visual_df is None:
        visual_df = pd.DataFrame()
    if ocr_df is None:
        ocr_df = pd.DataFrame()
    if visual_df.empty and ocr_df.empty:
        return pd.DataFrame()

    vis = visual_df.copy()
    ocr = ocr_df.copy()
    if vis.empty and "global_id" not in vis.columns:
        vis = pd.DataFrame(columns=["global_id"])
    if ocr.empty and "global_id" not in ocr.columns:
        ocr = pd.DataFrame(columns=["global_id"])
    if not vis.empty:
        vis["visual_normalized_score"] = minmax_normalize(vis["visual_raw_score"])
    if not ocr.empty:
        ocr["ocr_normalized_score"] = minmax_normalize(ocr["ocr_raw_score"])
        ocr["lexical_score"] = ocr["ocr_text"].fillna("").map(lambda t: lexical_similarity(query, str(t)))

    merged = pd.merge(vis, ocr, on="global_id", how="outer", suffixes=("_visual", "_ocr"))
    for col in ["visual_raw_score", "visual_rank", "visual_normalized_score"]:
        if col not in merged:
            merged[col] = 0.0
    for col in ["ocr_raw_score", "ocr_rank", "ocr_normalized_score", "lexical_score", "ocr_text"]:
        if col not in merged:
            merged[col] = 0.0 if col != "ocr_text" else ""

    if "ocr_text_ocr" in merged.columns or "ocr_text_visual" in merged.columns:
        merged["ocr_text"] = merged.get("ocr_text_ocr", pd.Series([""] * len(merged))).fillna("").astype(str)
        if "ocr_text_visual" in merged.columns:
            merged["ocr_text"] = merged["ocr_text"].where(merged["ocr_text"].str.strip() != "", merged["ocr_text_visual"].fillna("").astype(str))
    if "video_id_ocr" in merged.columns or "video_id_visual" in merged.columns:
        merged["video_id"] = merged.get("video_id_visual", pd.Series([None] * len(merged))).fillna(merged.get("video_id_ocr"))
    if "timestamp_seconds_ocr" in merged.columns or "timestamp_seconds_visual" in merged.columns:
        merged["timestamp_seconds"] = merged.get("timestamp_seconds_visual", pd.Series([np.nan] * len(merged))).fillna(merged.get("timestamp_seconds_ocr"))
    if "timestamp_text_ocr" in merged.columns or "timestamp_text_visual" in merged.columns:
        merged["timestamp_text"] = merged.get("timestamp_text_visual", pd.Series([""] * len(merged))).fillna(merged.get("timestamp_text_ocr"))
    if "frame_idx_ocr" in merged.columns or "frame_idx_visual" in merged.columns:
        merged["frame_idx"] = merged.get("frame_idx_visual", pd.Series([np.nan] * len(merged))).fillna(merged.get("frame_idx_ocr"))
    if "keyframe_path_ocr" in merged.columns or "keyframe_path_visual" in merged.columns:
        merged["keyframe_path"] = merged.get("keyframe_path_visual", pd.Series([""] * len(merged))).fillna(merged.get("keyframe_path_ocr"))
    if "video_path_ocr" in merged.columns or "video_path_visual" in merged.columns:
        merged["video_path"] = merged.get("video_path_visual", pd.Series([""] * len(merged))).fillna(merged.get("video_path_ocr"))

    if mode == "weighted_rrf":
        merged["fused_score"] = (
            visual_weight / (rrf_k + merged["visual_rank"].fillna(1e9))
            + ocr_weight / (rrf_k + merged["ocr_rank"].fillna(1e9))
        )
    else:
        merged["fused_score"] = (
            visual_weight * merged["visual_normalized_score"].fillna(0.0)
            + ocr_weight * merged["ocr_normalized_score"].fillna(0.0)
            + lexical_weight * merged["lexical_score"].fillna(0.0)
        )
    merged["visual_score"] = merged["visual_raw_score"].fillna(0.0)
    merged["ocr_score"] = merged["ocr_raw_score"].fillna(0.0)
    merged["lexical_score"] = merged["lexical_score"].fillna(0.0)
    merged = merged.sort_values("fused_score", ascending=False).reset_index(drop=True)
    merged["rank"] = range(1, len(merged) + 1)
    return merged
