"""
EventGraph Pipeline: Stages 3B, 3C, 3D — Robust Calibrated Event Boundary Detection
====================================================================================
Senior ML Engineer Implementation for AI Challenge 2026 Multimodal Video Retrieval.

Robust Calibrated Fusion Mechanism:
  1. Robust Z-score: z_m = (similarity_m - Median_m) / (IQR_m + 1e-8)
  2. Boundary Evidence: b_m = sigmoid(-z_m) = 1 / (1 + exp(z_m))
  3. Multimodal Fusion: boundary_score = w_vis * b_vis + w_sem * b_sem
  4. Boundary Decision: boundary_score > threshold

Artifact Outputs:
  - artifacts/event_graph/boundaries/adjacent_similarities.parquet
  - artifacts/event_graph/events/all_events.parquet
  - artifacts/event_graph/events/boundary_summary_report.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from _bootstrap import PROJECT_ROOT
from src.retrieval.logging_utils import setup_logger

logger = setup_logger("event-pipeline-3b-3d")


def compute_vector_similarity(
    vec1: Optional[List[float] | np.ndarray], vec2: Optional[List[float] | np.ndarray]
) -> float:
    """Compute cosine similarity between two vectors (assumes L2 normalized vectors)."""
    if vec1 is None or vec2 is None:
        return 0.0
    v1 = np.array(vec1, dtype=np.float32)
    v2 = np.array(vec2, dtype=np.float32)
    if v1.size == 0 or v2.size == 0:
        return 0.0

    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 < 1e-8 or norm2 < 1e-8:
        return 0.0

    v1_norm = v1 / norm1
    v2_norm = v2 / norm2
    dot_val = float(np.dot(v1_norm, v2_norm))
    return float(np.clip(dot_val, -1.0, 1.0))


def compute_video_raw_similarities(
    video_id: str,
    df_video_shots: pd.DataFrame,
) -> pd.DataFrame:
    """Stage 3B: Compute raw visual and semantic cosine similarities strictly within a video."""
    df_sorted = df_video_shots.sort_values("start_sec").reset_index(drop=True)
    num_shots = len(df_sorted)

    records = []
    for i in range(num_shots - 1):
        shot_i = df_sorted.iloc[i]
        shot_next = df_sorted.iloc[i + 1]

        vis_emb_i = shot_i.get("visual_embedding")
        vis_emb_next = shot_next.get("visual_embedding")
        sem_emb_i = shot_i.get("semantic_embedding")
        sem_emb_next = shot_next.get("semantic_embedding")

        vis_sim = compute_vector_similarity(vis_emb_i, vis_emb_next)
        sem_sim = compute_vector_similarity(sem_emb_i, sem_emb_next)

        records.append({
            "video_id": video_id,
            "shot_i": int(shot_i["shot_id"]),
            "shot_next": int(shot_next["shot_id"]),
            "shot_i_start_sec": float(shot_i["start_sec"]),
            "shot_i_end_sec": float(shot_i["end_sec"]),
            "shot_next_start_sec": float(shot_next["start_sec"]),
            "shot_next_end_sec": float(shot_next["end_sec"]),
            "visual_similarity": float(vis_sim),
            "semantic_similarity": float(sem_sim),
        })

    return pd.DataFrame(records)


def apply_robust_calibrated_fusion(
    df_sim: pd.DataFrame,
    vis_weight: float = 0.5,
    sem_weight: float = 0.5,
) -> pd.DataFrame:
    """Apply Robust Z-score normalization and Sigmoidal Boundary Evidence Fusion."""
    df = df_sim.copy()
    if df.empty:
        return df

    # 1. Modality Statistics (Median & IQR)
    vis_median = float(df["visual_similarity"].median())
    vis_q25 = float(df["visual_similarity"].quantile(0.25))
    vis_q75 = float(df["visual_similarity"].quantile(0.75))
    vis_iqr = max(vis_q75 - vis_q25, 1e-8)

    sem_median = float(df["semantic_similarity"].median())
    sem_q25 = float(df["semantic_similarity"].quantile(0.25))
    sem_q75 = float(df["semantic_similarity"].quantile(0.75))
    sem_iqr = max(sem_q75 - sem_q25, 1e-8)

    # 2. Robust Z-score
    df["visual_z"] = (df["visual_similarity"] - vis_median) / vis_iqr
    df["semantic_z"] = (df["semantic_similarity"] - sem_median) / sem_iqr

    # 3. Boundary Evidence: b = sigmoid(-z) = 1 / (1 + exp(z))
    df["visual_boundary_evidence"] = 1.0 / (1.0 + np.exp(df["visual_z"].to_numpy(dtype=np.float64)))
    df["semantic_boundary_evidence"] = 1.0 / (1.0 + np.exp(df["semantic_z"].to_numpy(dtype=np.float64)))

    # 4. Multimodal Fusion Boundary Score
    w_sum = vis_weight + sem_weight
    w_v = vis_weight / w_sum if w_sum > 0 else 0.5
    w_s = sem_weight / w_sum if w_sum > 0 else 0.5

    df["boundary_score"] = (w_v * df["visual_boundary_evidence"]) + (w_s * df["semantic_boundary_evidence"])

    # Legacy fallback calculation for comparison
    df["fused_similarity"] = (w_v * df["visual_similarity"]) + (w_s * df["semantic_similarity"])

    return df


def detect_video_event_boundaries(
    df_video_shots: pd.DataFrame,
    df_adj_sim: pd.DataFrame,
    boundary_threshold: float = 0.70,
    min_event_shots: int = 1,
    fusion_method: str = "robust_calibrated",
    merge_threshold: float = 0.55,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """Stage 3C & 3D: Detect event boundaries and construct Events for one video."""
    df_sorted = df_video_shots.sort_values("start_sec").reset_index(drop=True)
    video_id = str(df_sorted.iloc[0]["video_id"])
    num_shots = len(df_sorted)

    if num_shots == 0:
        return pd.DataFrame(), []

    # Map (shot_i, shot_next) to boundary decision score
    score_map = {}
    if not df_adj_sim.empty:
        for _, row in df_adj_sim.iterrows():
            pair_key = (int(row["shot_i"]), int(row["shot_next"]))
            if fusion_method == "robust_calibrated":
                score_map[pair_key] = float(row.get("boundary_score", 0.0))
            else:
                score_map[pair_key] = 1.0 - float(row.get("fused_similarity", 1.0))

    boundaries_flags = []
    for i in range(num_shots - 1):
        shot_i_id = int(df_sorted.iloc[i]["shot_id"])
        shot_next_id = int(df_sorted.iloc[i + 1]["shot_id"])
        score_val = score_map.get((shot_i_id, shot_next_id), 0.0)

        is_boundary = score_val > boundary_threshold if fusion_method == "robust_calibrated" else score_val > (1.0 - boundary_threshold)
        boundaries_flags.append(is_boundary)

    # Initial Event Construction
    raw_events: List[List[pd.Series]] = []
    current_shots: List[pd.Series] = [df_sorted.iloc[0]]

    for i in range(num_shots - 1):
        is_b = boundaries_flags[i]
        shot_next = df_sorted.iloc[i + 1]

        if is_b:
            raw_events.append(current_shots)
            current_shots = [shot_next]
        else:
            current_shots.append(shot_next)

    if current_shots:
        raw_events.append(current_shots)

    # Conditional Threshold-Gated Adaptive Neighbor Merging for 1-shot events
    final_events_shots: List[List[pd.Series]] = []
    e_idx = 0
    while e_idx < len(raw_events):
        evt = raw_events[e_idx]
        if len(evt) == 1 and len(raw_events) > 1:
            shot = evt[0]
            s_id = int(shot["shot_id"])

            left_score = float("inf")
            right_score = float("inf")

            if final_events_shots:
                left_last_s_id = int(final_events_shots[-1][-1]["shot_id"])
                left_score = score_map.get((left_last_s_id, s_id), float("inf"))

            if e_idx + 1 < len(raw_events):
                right_first_s_id = int(raw_events[e_idx + 1][0]["shot_id"])
                right_score = score_map.get((s_id, right_first_s_id), float("inf"))

            min_boundary_score = min(left_score, right_score)

            # ONLY merge if similarity with a neighbor is sufficiently high (boundary_score < merge_threshold)
            if min_boundary_score < merge_threshold:
                if left_score <= right_score and final_events_shots:
                    final_events_shots[-1].append(shot)
                elif e_idx + 1 < len(raw_events):
                    raw_events[e_idx + 1].insert(0, shot)
                else:
                    final_events_shots.append(evt)
            else:
                # Keep as a valid 1-shot event because it is distinctly different from both sides!
                final_events_shots.append(evt)
        else:
            final_events_shots.append(evt)
        e_idx += 1

    events: List[Dict[str, Any]] = [
        build_single_event_record(video_id, idx, e_shots)
        for idx, e_shots in enumerate(final_events_shots)
    ]

    df_adj_updated = df_adj_sim.copy()
    if not df_adj_updated.empty:
        if fusion_method == "robust_calibrated":
            df_adj_updated["is_boundary"] = df_adj_updated["boundary_score"] > boundary_threshold
        else:
            df_adj_updated["is_boundary"] = df_adj_updated["fused_similarity"] < boundary_threshold

    return df_adj_updated, events


def build_single_event_record(
    video_id: str, event_idx: int, event_shots: List[pd.Series]
) -> Dict[str, Any]:
    """Stage 3D: Build exact schema required for an Event."""
    event_id = f"{video_id}_E{event_idx:03d}"

    start_shot_id = int(event_shots[0]["shot_id"])
    end_shot_id = int(event_shots[-1]["shot_id"])
    shot_ids = [int(s["shot_id"]) for s in event_shots]

    start_frame = int(event_shots[0].get("start_frame", 0))
    end_frame = int(event_shots[-1].get("end_frame", 0))

    start_sec = float(event_shots[0]["start_sec"])
    end_sec = float(event_shots[-1]["end_sec"])
    duration_sec = end_sec - start_sec

    rep_keyframes = []
    for s in event_shots:
        rk = str(s.get("representative_keyframe", ""))
        if rk and rk not in rep_keyframes:
            rep_keyframes.append(rk)

    return {
        "event_id": event_id,
        "video_id": video_id,
        "event_index": event_idx,
        "start_shot_id": start_shot_id,
        "end_shot_id": end_shot_id,
        "shot_ids": shot_ids,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "start_sec": start_sec,
        "end_sec": end_sec,
        "duration_sec": duration_sec,
        "num_shots": len(shot_ids),
        "representative_keyframes": rep_keyframes,
    }


def generate_pipeline_summary_report(
    df_similarities: pd.DataFrame, df_events: pd.DataFrame, threshold: float
) -> Dict[str, Any]:
    """Compute comprehensive statistical metrics for dataset summary report."""
    total_shots = len(df_similarities) + df_events["video_id"].nunique()
    total_videos = df_events["video_id"].nunique()
    total_events = len(df_events)
    total_boundaries = int(df_similarities["is_boundary"].sum()) if "is_boundary" in df_similarities else 0

    score_series = df_similarities["boundary_score"] if "boundary_score" in df_similarities else df_similarities.get("fused_similarity", pd.Series([0.0]))
    shots_per_event = df_events["num_shots"] if not df_events.empty else pd.Series([0])
    events_per_video = df_events.groupby("video_id").size() if not df_events.empty else pd.Series([0])

    summary = {
        "dataset_statistics": {
            "total_videos": total_videos,
            "total_shots": total_shots,
            "total_events": total_events,
            "total_adjacent_pairs": len(df_similarities),
            "boundary_threshold": threshold,
            "total_boundaries_detected": total_boundaries,
            "average_events_per_video": float(events_per_video.mean()),
            "average_shots_per_event": float(shots_per_event.mean()),
        },
        "score_distribution": {
            "mean": float(score_series.mean()),
            "std": float(score_series.std()),
            "min": float(score_series.min()),
            "max": float(score_series.max()),
            "percentiles": {
                "25%": float(score_series.quantile(0.25)),
                "50% (median)": float(score_series.median()),
                "75%": float(score_series.quantile(0.75)),
                "90%": float(score_series.quantile(0.90)),
                "95%": float(score_series.quantile(0.95)),
            },
        },
        "events_per_video_distribution": {
            "mean": float(events_per_video.mean()),
            "min": int(events_per_video.min()),
            "max": int(events_per_video.max()),
            "median": float(events_per_video.median()),
        },
        "shots_per_event_distribution": {
            "mean": float(shots_per_event.mean()),
            "min": int(shots_per_event.min()),
            "max": int(shots_per_event.max()),
            "median": float(shots_per_event.median()),
        },
    }
    return summary


def run_event_pipeline(
    features_path: Path,
    output_dir: Path,
    boundary_threshold: float = 0.60,
    vis_weight: float = 0.5,
    sem_weight: float = 0.5,
    min_event_shots: int = 1,
    fusion_method: str = "robust_calibrated",
) -> Tuple[Path, Path, Path]:
    """Execute Stages 3B, 3C, 3D end-to-end with Robust Calibrated Fusion."""
    output_dir.mkdir(parents=True, exist_ok=True)
    boundaries_dir = output_dir.parent / "boundaries"
    boundaries_dir.mkdir(parents=True, exist_ok=True)

    out_sim_path = boundaries_dir / "adjacent_similarities.parquet"
    out_events_path = output_dir / "all_events.parquet"
    out_summary_path = output_dir / "boundary_summary_report.json"

    if not features_path.exists():
        raise FileNotFoundError(f"Input shot features parquet not found at: {features_path}")

    logger.info("==================================================================")
    logger.info("🎬 STAGES 3B, 3C, 3D: ROBUST CALIBRATED EVENT BOUNDARY ENGINE")
    logger.info("==================================================================")
    logger.info("Loading input features from: %s", features_path)
    df_features = pd.read_parquet(features_path)

    total_shots = len(df_features)
    video_groups = list(df_features.groupby("video_id"))
    total_videos = len(video_groups)

    logger.info("Loaded %d shots across %d videos", total_shots, total_videos)
    logger.info("Parameters: method=%s, threshold=%.2f, vis_weight=%.2f, sem_weight=%.2f, min_shots=%d",
                fusion_method, boundary_threshold, vis_weight, sem_weight, min_event_shots)

    all_sim_dfs = []
    start_time = time.time()

    # Pass 1: Compute Raw Similarities across all videos
    for idx, (video_id, df_video_shots) in enumerate(video_groups, start=1):
        df_sim = compute_video_raw_similarities(str(video_id), df_video_shots)
        all_sim_dfs.append(df_sim)

    df_raw_sim = pd.concat(all_sim_dfs, ignore_index=True) if all_sim_dfs else pd.DataFrame()

    # Apply Robust Calibrated Normalization & Fusion over full dataset
    logger.info("Applying Robust Z-score Normalization & Sigmoidal Fusion...")
    df_calibrated_sim = apply_robust_calibrated_fusion(df_raw_sim, vis_weight=vis_weight, sem_weight=sem_weight)

    # Pass 2: Detect Boundaries & Build Events
    grouped_calibrated_sim = df_calibrated_sim.groupby("video_id")
    all_sim_updated_dfs = []
    all_events = []

    for idx, (video_id, df_video_shots) in enumerate(video_groups, start=1):
        v_sim = grouped_calibrated_sim.get_group(video_id) if video_id in grouped_calibrated_sim.groups else pd.DataFrame()
        df_sim_upd, video_events = detect_video_event_boundaries(
            df_video_shots, v_sim, boundary_threshold=boundary_threshold,
            min_event_shots=min_event_shots, fusion_method=fusion_method
        )
        all_sim_updated_dfs.append(df_sim_upd)
        all_events.extend(video_events)

    df_all_sim = pd.concat(all_sim_updated_dfs, ignore_index=True) if all_sim_updated_dfs else pd.DataFrame()
    df_all_events = pd.DataFrame(all_events)

    logger.info("Saving calibrated adjacent similarities to: %s", out_sim_path)
    df_all_sim.to_parquet(out_sim_path, index=False)

    logger.info("Saving constructed events to: %s", out_events_path)
    df_all_events.to_parquet(out_events_path, index=False)

    summary = generate_pipeline_summary_report(df_all_sim, df_all_events, boundary_threshold)
    with open(out_summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 75)
    print("🎉 STAGES 3B, 3C, 3D CALIBRATED PIPELINE SUMMARY REPORT")
    print("=" * 75)
    print(f" • Total Videos Processed     : {summary['dataset_statistics']['total_videos']:,}")
    print(f" • Total Shots Evaluated    : {summary['dataset_statistics']['total_shots']:,}")
    print(f" • Total Boundaries Detected: {summary['dataset_statistics']['total_boundaries_detected']:,}")
    print(f" • Total Events Constructed  : {summary['dataset_statistics']['total_events']:,}")
    print(f" • Boundary Threshold       : {summary['dataset_statistics']['boundary_threshold']}")
    print(f" • Avg Events / Video       : {summary['dataset_statistics']['average_events_per_video']:.2f}")
    print(f" • Avg Shots / Event       : {summary['dataset_statistics']['average_shots_per_event']:.2f}")
    print("-" * 75)
    print("📊 BOUNDARY SCORE STATISTICAL DISTRIBUTION:")
    print(f" • Mean ± Std               : {summary['score_distribution']['mean']:.4f} ± {summary['score_distribution']['std']:.4f}")
    print(f" • Min / Max                : {summary['score_distribution']['min']:.4f} / {summary['score_distribution']['max']:.4f}")
    print(f" • Quantiles (25% / 50% / 75% / 95%): {summary['score_distribution']['percentiles']['25%']:.4f} / {summary['score_distribution']['percentiles']['50% (median)']:.4f} / {summary['score_distribution']['percentiles']['75%']:.4f} / {summary['score_distribution']['percentiles']['95%']:.4f}")
    print("=" * 75 + "\n")

    return out_sim_path, out_events_path, out_summary_path


def main():
    parser = argparse.ArgumentParser(
        description="EventGraph Stages 3B-3D: Robust Calibrated Event Boundary Detection Engine"
    )
    parser.add_argument(
        "--features",
        default=str(PROJECT_ROOT / "artifacts" / "event_graph" / "features" / "shot_features.parquet"),
        help="Path to shot_features.parquet",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "artifacts" / "event_graph" / "events"),
        help="Directory to save output all_events.parquet",
    )
    parser.add_argument(
        "--boundary-threshold",
        type=float,
        default=0.60,
        help="Boundary score threshold above which an event boundary is triggered (default: 0.60)",
    )
    parser.add_argument(
        "--vis-weight",
        type=float,
        default=0.5,
        help="Weight for visual boundary evidence (default: 0.5)",
    )
    parser.add_argument(
        "--sem-weight",
        type=float,
        default=0.5,
        help="Weight for semantic boundary evidence (default: 0.5)",
    )
    parser.add_argument(
        "--min-event-shots",
        type=int,
        default=1,
        help="Minimum number of shots allowed in an event (default: 1)",
    )
    parser.add_argument(
        "--fusion-method",
        choices=["robust_calibrated", "legacy"],
        default="robust_calibrated",
        help="Fusion method for boundary detection (default: robust_calibrated)",
    )
    args = parser.parse_args()

    run_event_pipeline(
        features_path=Path(args.features),
        output_dir=Path(args.output_dir),
        boundary_threshold=args.boundary_threshold,
        vis_weight=args.vis_weight,
        sem_weight=args.sem_weight,
        min_event_shots=args.min_event_shots,
        fusion_method=args.fusion_method,
    )


if __name__ == "__main__":
    main()
