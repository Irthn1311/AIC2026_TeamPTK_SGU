"""
EventGraph Pipeline: Stages 3B, 3C, 3D — Adjacent Shot Similarity & Event Boundary Construction
=================================================================================================
Senior ML Engineer Implementation for AI Challenge 2026 Multimodal Video Retrieval.

Stages:
  - Stage 3B: Adjacent Shot Similarity (Visual, Semantic, Fused cosine similarity within videos)
  - Stage 3C: Event Boundary Detection (Threshold-based boundary detection with noise filtering)
  - Stage 3D: Event Construction (Group contiguous shots into high-level events with metadata)

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


# =============================================================================
# STAGE 3B: ADJACENT SHOT SIMILARITY
# =============================================================================
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
    
    # Check if vectors are unit norm; if not, normalize
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 < 1e-8 or norm2 < 1e-8:
        return 0.0
    
    v1_norm = v1 / norm1
    v2_norm = v2 / norm2
    dot_val = float(np.dot(v1_norm, v2_norm))
    return float(np.clip(dot_val, -1.0, 1.0))


def compute_video_adjacent_similarities(
    video_id: str,
    df_video_shots: pd.DataFrame,
    vis_weight: float = 0.5,
    sem_weight: float = 0.5,
) -> pd.DataFrame:
    """Stage 3B: Compute adjacent shot similarities strictly within a single video."""
    # Ensure chronological sorting
    df_sorted = df_video_shots.sort_values("start_sec").reset_index(drop=True)
    num_shots = len(df_sorted)

    records = []
    for i in range(num_shots - 1):
        shot_i = df_sorted.iloc[i]
        shot_next = df_sorted.iloc[i + 1]

        # Extract embeddings
        vis_emb_i = shot_i.get("visual_embedding")
        vis_emb_next = shot_next.get("visual_embedding")
        sem_emb_i = shot_i.get("semantic_embedding")
        sem_emb_next = shot_next.get("semantic_embedding")

        vis_sim = compute_vector_similarity(vis_emb_i, vis_emb_next)
        sem_sim = compute_vector_similarity(sem_emb_i, sem_emb_next)
        
        # Weighted fusion similarity score
        fused_sim = (vis_weight * vis_sim) + (sem_weight * sem_sim)

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
            "fused_similarity": float(fused_sim),
        })

    return pd.DataFrame(records)


# =============================================================================
# STAGE 3C & 3D: EVENT BOUNDARY DETECTION & EVENT CONSTRUCTION
# =============================================================================
def detect_video_event_boundaries(
    df_video_shots: pd.DataFrame,
    df_adj_sim: pd.DataFrame,
    boundary_threshold: float = 0.50,
    min_event_shots: int = 1,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """Stage 3C & 3D: Detect event boundaries and construct Events for one video."""
    df_sorted = df_video_shots.sort_values("start_sec").reset_index(drop=True)
    video_id = str(df_sorted.iloc[0]["video_id"])
    num_shots = len(df_sorted)

    if num_shots == 0:
        return pd.DataFrame(), []

    # Map (shot_i, shot_next) to fused_similarity
    sim_map = {}
    if not df_adj_sim.empty:
        for _, row in df_adj_sim.iterrows():
            sim_map[(int(row["shot_i"]), int(row["shot_next"]))] = float(row["fused_similarity"])

    # Determine boundaries
    boundaries_flags = []
    for i in range(num_shots - 1):
        shot_i_id = int(df_sorted.iloc[i]["shot_id"])
        shot_next_id = int(df_sorted.iloc[i + 1]["shot_id"])
        sim_val = sim_map.get((shot_i_id, shot_next_id), 1.0)
        
        # Candidate boundary if similarity < threshold
        is_boundary = sim_val < boundary_threshold
        boundaries_flags.append(is_boundary)

    # Group contiguous shots into Events with min_event_shots filtering
    events: List[Dict[str, Any]] = []
    current_shots: List[pd.Series] = [df_sorted.iloc[0]]
    event_idx = 0

    for i in range(num_shots - 1):
        is_boundary = boundaries_flags[i]
        shot_next = df_sorted.iloc[i + 1]

        if is_boundary:
            # Check noise filtering condition
            if len(current_shots) < min_event_shots and i < num_shots - 2:
                # Merge small noise shot into next event instead of creating tiny boundary
                current_shots.append(shot_next)
            else:
                # Construct Event
                event_rec = build_single_event_record(video_id, event_idx, current_shots)
                events.append(event_rec)
                event_idx += 1
                current_shots = [shot_next]
        else:
            current_shots.append(shot_next)

    # Add final event
    if current_shots:
        event_rec = build_single_event_record(video_id, event_idx, current_shots)
        events.append(event_rec)

    # Update adjacent similarity DataFrame with boundary flag column
    df_adj_updated = df_adj_sim.copy()
    if not df_adj_updated.empty:
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

    # Handle optional start_frame / end_frame
    start_frame = int(event_shots[0].get("start_frame", 0))
    end_frame = int(event_shots[-1].get("end_frame", 0))

    start_sec = float(event_shots[0]["start_sec"])
    end_sec = float(event_shots[-1]["end_sec"])
    duration_sec = end_sec - start_sec

    # Collect representative keyframes across shots in the event
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


# =============================================================================
# STATISTICAL SUMMARY & AUDIT ENGINE
# =============================================================================
def generate_pipeline_summary_report(
    df_similarities: pd.DataFrame, df_events: pd.DataFrame, threshold: float
) -> Dict[str, Any]:
    """Compute comprehensive statistical metrics for dataset summary report."""
    total_shots = len(df_similarities) + df_events["video_id"].nunique()
    total_videos = df_events["video_id"].nunique()
    total_events = len(df_events)
    total_boundaries = int(df_similarities["is_boundary"].sum()) if "is_boundary" in df_similarities else 0

    sim_series = df_similarities["fused_similarity"] if not df_similarities.empty else pd.Series([0.0])
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
        "similarity_distribution": {
            "mean": float(sim_series.mean()),
            "std": float(sim_series.std()),
            "min": float(sim_series.min()),
            "max": float(sim_series.max()),
            "percentiles": {
                "25%": float(sim_series.quantile(0.25)),
                "50% (median)": float(sim_series.median()),
                "75%": float(sim_series.quantile(0.75)),
                "90%": float(sim_series.quantile(0.90)),
                "95%": float(sim_series.quantile(0.95)),
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


# =============================================================================
# MAIN PIPELINE FUNCTION
# =============================================================================
def run_event_pipeline(
    features_path: Path,
    output_dir: Path,
    boundary_threshold: float = 0.50,
    vis_weight: float = 0.5,
    sem_weight: float = 0.5,
    min_event_shots: int = 1,
) -> Tuple[Path, Path, Path]:
    """Execute Stages 3B, 3C, 3D end-to-end."""
    output_dir.mkdir(parents=True, exist_ok=True)
    boundaries_dir = output_dir.parent / "boundaries"
    boundaries_dir.mkdir(parents=True, exist_ok=True)

    out_sim_path = boundaries_dir / "adjacent_similarities.parquet"
    out_events_path = output_dir / "all_events.parquet"
    out_summary_path = output_dir / "boundary_summary_report.json"

    if not features_path.exists():
        raise FileNotFoundError(f"Input shot features parquet not found at: {features_path}")

    logger.info("==================================================================")
    logger.info("🎬 STARTING STAGES 3B, 3C, 3D: EVENT BOUNDARY & CONSTRUCTION")
    logger.info("==================================================================")
    logger.info("Loading input features from: %s", features_path)
    df_features = pd.read_parquet(features_path)

    total_shots = len(df_features)
    video_groups = list(df_features.groupby("video_id"))
    total_videos = len(video_groups)

    logger.info("Loaded %d shots across %d videos", total_shots, total_videos)
    logger.info("CLI Parameters: threshold=%.2f, vis_weight=%.2f, sem_weight=%.2f, min_shots=%d",
                boundary_threshold, vis_weight, sem_weight, min_event_shots)

    all_sim_dfs = []
    all_events = []
    start_time = time.time()

    for idx, (video_id, df_video_shots) in enumerate(video_groups, start=1):
        # Stage 3B: Compute Adjacent Similarities
        df_sim = compute_video_adjacent_similarities(
            str(video_id), df_video_shots, vis_weight=vis_weight, sem_weight=sem_weight
        )
        
        # Stage 3C & 3D: Detect Boundaries & Construct Events
        df_sim_updated, video_events = detect_video_event_boundaries(
            df_video_shots, df_sim, boundary_threshold=boundary_threshold, min_event_shots=min_event_shots
        )

        all_sim_dfs.append(df_sim_updated)
        all_events.extend(video_events)

        if idx % 100 == 0 or idx == total_videos:
            elapsed = time.time() - start_time
            rate = idx / (elapsed + 1e-5)
            eta_sec = (total_videos - idx) / (rate + 1e-5)
            logger.info("Progress: [%d/%d vids] (%.1f%%) | Events: %d | Rate: %.1f vids/sec | ETA: %.1fs",
                        idx, total_videos, (idx / total_videos) * 100, len(all_events), rate, eta_sec)

    df_all_sim = pd.concat(all_sim_dfs, ignore_index=True) if all_sim_dfs else pd.DataFrame()
    df_all_events = pd.DataFrame(all_events)

    logger.info("Saving adjacent similarities to: %s", out_sim_path)
    df_all_sim.to_parquet(out_sim_path, index=False)

    logger.info("Saving constructed events to: %s", out_events_path)
    df_all_events.to_parquet(out_events_path, index=False)

    # Generate Statistical Summary Report
    summary = generate_pipeline_summary_report(df_all_sim, df_all_events, boundary_threshold)
    with open(out_summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info("Saved summary report to: %s", out_summary_path)

    # Print Final Summary Table to Terminal
    print("\n" + "=" * 75)
    print("🎉 STAGES 3B, 3C, 3D PIPELINE SUMMARY REPORT")
    print("=" * 75)
    print(f" • Total Videos Processed     : {summary['dataset_statistics']['total_videos']:,}")
    print(f" • Total Shots Evaluated    : {summary['dataset_statistics']['total_shots']:,}")
    print(f" • Total Boundaries Detected: {summary['dataset_statistics']['total_boundaries_detected']:,}")
    print(f" • Total Events Constructed  : {summary['dataset_statistics']['total_events']:,}")
    print(f" • Boundary Threshold       : {summary['dataset_statistics']['boundary_threshold']}")
    print(f" • Avg Events / Video       : {summary['dataset_statistics']['average_events_per_video']:.2f}")
    print(f" • Avg Shots / Event       : {summary['dataset_statistics']['average_shots_per_event']:.2f}")
    print("-" * 75)
    print("📊 ADJACENT SIMILARITY STATISTICAL DISTRIBUTION:")
    print(f" • Mean ± Std               : {summary['similarity_distribution']['mean']:.4f} ± {summary['similarity_distribution']['std']:.4f}")
    print(f" • Min / Max                : {summary['similarity_distribution']['min']:.4f} / {summary['similarity_distribution']['max']:.4f}")
    print(f" • Quantiles (25% / 50% / 75% / 95%): {summary['similarity_distribution']['percentiles']['25%']:.4f} / {summary['similarity_distribution']['percentiles']['50% (median)']:.4f} / {summary['similarity_distribution']['percentiles']['75%']:.4f} / {summary['similarity_distribution']['percentiles']['95%']:.4f}")
    print("=" * 75 + "\n")

    return out_sim_path, out_events_path, out_summary_path


def main():
    parser = argparse.ArgumentParser(
        description="EventGraph Stages 3B-3D: Adjacent Similarity & Event Boundary Detection"
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
        default=0.50,
        help="Similarity threshold below which an event boundary is triggered (default: 0.50)",
    )
    parser.add_argument(
        "--vis-weight",
        type=float,
        default=0.5,
        help="Weight for visual embedding cosine similarity",
    )
    parser.add_argument(
        "--sem-weight",
        type=float,
        default=0.5,
        help="Weight for semantic embedding cosine similarity",
    )
    parser.add_argument(
        "--min-event-shots",
        type=int,
        default=1,
        help="Minimum number of shots allowed in an event to filter noise splits",
    )
    args = parser.parse_args()

    run_event_pipeline(
        features_path=Path(args.features),
        output_dir=Path(args.output_dir),
        boundary_threshold=args.boundary_threshold,
        vis_weight=args.vis_weight,
        sem_weight=args.sem_weight,
        min_event_shots=args.min_event_shots,
    )


if __name__ == "__main__":
    main()
