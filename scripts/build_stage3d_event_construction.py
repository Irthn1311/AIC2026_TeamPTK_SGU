"""
Stage 3D: Boundary Refinement & Shot-to-Event Construction (AI Challenge 2026 EventGraph)
========================================================================================
Senior Computer Vision & ML Engineer Implementation.

Refinement & Event Construction Pipeline:
1. Analyzes raw boundary evidence (Visual Z-score, Semantic Z-score, Boundary Score).
2. Performs Multimodal Boundary Refinement:
   - Evaluates Background / Layout Constancy (S_layout) and Context Continuity (S_context).
   - Identifies False Positive patterns (e.g., Slide / Infographic text changes with static layout).
   - Calculates `final_boundary_score` using adaptive layout-suppression weighting.
3. Groups contiguous shots between refined strong boundaries into cohesive Events.
4. Performs Adaptive Micro-Event Merging for isolated single-shot fragments with weak boundaries.
5. Saves production event dataset `all_events.parquet` and summary statistics.

Output Artifacts:
  - artifacts/event_graph/boundaries/refined_boundaries.parquet
  - artifacts/event_graph/events/all_events.parquet
  - artifacts/event_graph/events/stage3d_refinement_report.json
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

import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("stage3d-event-construction")


def compute_vector_similarity(
    vec1: Optional[List[float] | np.ndarray], vec2: Optional[List[float] | np.ndarray]
) -> float:
    """Compute cosine similarity between two vectors."""
    if vec1 is None or vec2 is None:
        return 0.0
    v1 = np.array(vec1, dtype=np.float32)
    v2 = np.array(vec2, dtype=np.float32)
    if v1.size == 0 or v2.size == 0:
        return 0.0

    norm1 = float(np.linalg.norm(v1))
    norm2 = float(np.linalg.norm(v2))
    if norm1 < 1e-8 or norm2 < 1e-8:
        return 0.0

    dot_val = float(np.dot(v1 / norm1, v2 / norm2))
    return float(np.clip(dot_val, -1.0, 1.0))


def compute_layout_constancy(row: pd.Series) -> float:
    """
    Estimate background / layout constancy metric S_layout [0, 1].
    Slide / Infographic transitions have high visual similarity or similar background structure
    despite high text/semantic divergence.
    """
    vis_sim = float(row.get("visual_similarity", 0.0))
    sem_sim = float(row.get("semantic_similarity", 0.0))
    vis_evd = float(row.get("visual_boundary_evidence", 0.5))
    sem_evd = float(row.get("semantic_boundary_evidence", 0.5))

    # Layout constancy is high when visual similarity is high relative to semantic similarity divergence
    # or when semantic boundary evidence is high but visual evidence is low
    layout_score = float(np.clip(vis_sim + max(0.0, sem_evd - vis_evd) * 0.5, 0.0, 1.0))
    return layout_score


def refine_boundary_scores(
    df_boundaries: pd.DataFrame,
    threshold: float = 0.70,
    suppression_weight: float = 0.35,
) -> pd.DataFrame:
    """
    Stage 3D Multimodal Refinement:
    Calculates `final_boundary_score` by penalizing false boundaries caused by slide/infographic
    text changes with static background/layout.
    """
    df = df_boundaries.copy()
    if df.empty:
        return df

    final_scores = []
    suppression_deltas = []
    is_boundary_final = []

    for _, row in df.iterrows():
        base_score = float(row.get("boundary_score", 0.0))
        vis_evd = float(row.get("visual_boundary_evidence", 0.5))
        sem_evd = float(row.get("semantic_boundary_evidence", 0.5))
        vis_sim = float(row.get("visual_similarity", 0.0))
        sem_sim = float(row.get("semantic_similarity", 0.0))

        layout_constancy = compute_layout_constancy(row)

        # Refinement logic: If semantic evidence is high (slide text changed) but visual background layout
        # is constant (high layout_constancy or high vis_sim), apply adaptive suppression.
        sem_gap = max(0.0, sem_evd - vis_evd)
        
        # Slide pattern detection score
        slide_pattern = layout_constancy * sem_gap
        delta = float(np.clip(suppression_weight * slide_pattern, 0.0, 0.35))

        refined_score = float(np.clip(base_score - delta, 0.0, 1.0))
        
        final_scores.append(refined_score)
        suppression_deltas.append(delta)
        is_boundary_final.append(refined_score > threshold)

    df["layout_constancy"] = [compute_layout_constancy(r) for _, r in df.iterrows()]
    df["suppression_delta"] = suppression_deltas
    df["final_boundary_score"] = final_scores
    df["is_boundary_refined"] = is_boundary_final

    return df


def construct_video_events(
    df_video_shots: pd.DataFrame,
    df_video_boundaries: pd.DataFrame,
    threshold: float = 0.70,
    merge_threshold: float = 0.55,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """Construct cohesive Event records for one video after boundary refinement."""
    df_sorted = df_video_shots.sort_values("start_sec").reset_index(drop=True)
    if df_sorted.empty:
        return pd.DataFrame(), []

    video_id = str(df_sorted.iloc[0]["video_id"])
    num_shots = len(df_sorted)

    # Map (shot_i, shot_next) to final_boundary_score
    score_map = {}
    if not df_video_boundaries.empty:
        for _, row in df_video_boundaries.iterrows():
            pair_key = (int(row["shot_i"]), int(row["shot_next"]))
            score_map[pair_key] = float(row.get("final_boundary_score", row.get("boundary_score", 0.0)))

    # Identify refined boundary flags
    boundaries_flags = []
    for i in range(num_shots - 1):
        shot_i_id = int(df_sorted.iloc[i]["shot_id"])
        shot_next_id = int(df_sorted.iloc[i + 1]["shot_id"])
        score_val = score_map.get((shot_i_id, shot_next_id), 0.0)
        boundaries_flags.append(score_val > threshold)

    # Partition shots into raw events
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

    # Adaptive merging of single-shot micro-events with weak boundary scores
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

            if min_boundary_score < merge_threshold:
                if left_score <= right_score and final_events_shots:
                    final_events_shots[-1].append(shot)
                elif e_idx + 1 < len(raw_events):
                    raw_events[e_idx + 1].insert(0, shot)
                else:
                    final_events_shots.append(evt)
            else:
                final_events_shots.append(evt)
        else:
            final_events_shots.append(evt)
        e_idx += 1

    # Build Event records with schema:
    # event_id, video_id, start_shot, end_shot, start_sec, end_sec, shot_ids, representative_keyframes, boundary_confidence
    event_records = []
    for idx, e_shots in enumerate(final_events_shots):
        e_rec = build_stage3d_event_record(video_id, idx, e_shots, score_map)
        event_records.append(e_rec)

    return df_video_boundaries, event_records


def build_stage3d_event_record(
    video_id: str,
    event_idx: int,
    event_shots: List[pd.Series],
    score_map: Dict[Tuple[int, int], float],
) -> Dict[str, Any]:
    """Construct exact Stage 3D Event Record schema."""
    event_id = f"{video_id}_E{event_idx:03d}"

    start_shot = int(event_shots[0]["shot_id"])
    end_shot = int(event_shots[-1]["shot_id"])
    shot_ids = [int(s["shot_id"]) for s in event_shots]

    start_frame = int(event_shots[0].get("start_frame", 0))
    end_frame = int(event_shots[-1].get("end_frame", 0))

    start_sec = float(event_shots[0]["start_sec"])
    end_sec = float(event_shots[-1]["end_sec"])
    duration_sec = round(end_sec - start_sec, 3)

    rep_keyframes = []
    for s in event_shots:
        rk = str(s.get("representative_keyframe", ""))
        if rk and rk not in rep_keyframes:
            rep_keyframes.append(rk)

    # Calculate boundary_confidence (average boundary score at start and end of event)
    event_boundary_scores = []
    if start_shot > 0:
        prev_score = score_map.get((start_shot - 1, start_shot))
        if prev_score is not None:
            event_boundary_scores.append(prev_score)
    next_score = score_map.get((end_shot, end_shot + 1))
    if next_score is not None:
        event_boundary_scores.append(next_score)

    boundary_confidence = float(np.mean(event_boundary_scores)) if event_boundary_scores else 0.85

    return {
        "event_id": event_id,
        "video_id": video_id,
        "event_index": event_idx,
        "start_shot": start_shot,
        "end_shot": end_shot,
        "start_sec": start_sec,
        "end_sec": end_sec,
        "duration_sec": duration_sec,
        "num_shots": len(shot_ids),
        "shot_ids": shot_ids,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "representative_keyframes": rep_keyframes,
        "boundary_confidence": round(boundary_confidence, 4),
    }


def main():
    parser = argparse.ArgumentParser(description="Stage 3D: Boundary Refinement & Event Construction")
    parser.add_argument(
        "--boundaries-in",
        type=str,
        default=str(PROJECT_ROOT / "artifacts" / "event_graph" / "boundaries" / "adjacent_similarities.parquet"),
        help="Path to Stage 3B adjacent similarities parquet",
    )
    parser.add_argument(
        "--shots-in",
        type=str,
        default=str(PROJECT_ROOT / "artifacts" / "event_graph" / "features" / "shot_features.parquet"),
        help="Path to Stage 3A shot features parquet",
    )
    parser.add_argument(
        "--events-out",
        type=str,
        default=str(PROJECT_ROOT / "artifacts" / "event_graph" / "events" / "all_events.parquet"),
        help="Path to output all_events.parquet",
    )
    parser.add_argument(
        "--boundaries-out",
        type=str,
        default=str(PROJECT_ROOT / "artifacts" / "event_graph" / "boundaries" / "refined_boundaries.parquet"),
        help="Path to output refined boundaries parquet",
    )
    parser.add_argument(
        "--report-out",
        type=str,
        default=str(PROJECT_ROOT / "artifacts" / "event_graph" / "events" / "stage3d_refinement_report.json"),
        help="Path to output stage3d refinement report JSON",
    )
    parser.add_argument("--threshold", type=float, default=0.70, help="Refined boundary detection threshold")
    parser.add_argument("--merge-threshold", type=float, default=0.55, help="Single-shot micro-event merge threshold")
    parser.add_argument("--suppression-weight", type=float, default=0.35, help="Slide false positive suppression weight")
    args = parser.parse_args()

    logger.info("==================================================================")
    logger.info("⚡ STAGE 3D: BOUNDARY REFINEMENT & SHOT-TO-EVENT CONSTRUCTION")
    logger.info("==================================================================")

    boundaries_path = Path(args.boundaries_in)
    shots_path = Path(args.shots_in)

    if not boundaries_path.exists():
        logger.error("Boundaries file not found: %s", boundaries_path)
        sys.exit(1)
    if not shots_path.exists():
        logger.error("Shot features file not found: %s", shots_path)
        sys.exit(1)

    logger.info("Loading Stage 3B boundaries from: %s", boundaries_path)
    df_boundaries = pd.read_parquet(boundaries_path)

    logger.info("Loading Stage 3A shot features from: %s", shots_path)
    df_shots = pd.read_parquet(shots_path)

    logger.info("Refining boundary scores (Slide/Infographic Suppression)...")
    t0 = time.time()
    df_refined_boundaries = refine_boundary_scores(
        df_boundaries,
        threshold=args.threshold,
        suppression_weight=args.suppression_weight,
    )

    # Compute boundary statistics before and after refinement
    raw_b_count = int(df_boundaries["is_boundary"].sum()) if "is_boundary" in df_boundaries else 0
    refined_b_count = int(df_refined_boundaries["is_boundary_refined"].sum())
    suppressed_count = int(((df_boundaries.get("is_boundary", False)) & (~df_refined_boundaries["is_boundary_refined"])).sum())

    logger.info("Raw Boundaries (>%.2f): %d", args.threshold, raw_b_count)
    logger.info("Refined Boundaries (>%.2f): %d (Suppressed False Positives: %d)", args.threshold, refined_b_count, suppressed_count)

    # Construct Events per video
    logger.info("Constructing Events for %d videos...", df_shots["video_id"].nunique())
    all_event_records = []
    
    for vid, df_v_shots in df_shots.groupby("video_id"):
        df_v_bound = df_refined_boundaries[df_refined_boundaries["video_id"] == vid]
        _, v_events = construct_video_events(
            df_v_shots,
            df_v_bound,
            threshold=args.threshold,
            merge_threshold=args.merge_threshold,
        )
        all_event_records.extend(v_events)

    df_events = pd.DataFrame(all_event_records)
    t_elapsed = time.time() - t0

    # Save output artifacts
    events_out_path = Path(args.events_out)
    events_out_path.parent.mkdir(parents=True, exist_ok=True)
    df_events.to_parquet(events_out_path, index=False)
    logger.info("Saved %d constructed events to: %s", len(df_events), events_out_path)

    refined_out_path = Path(args.boundaries_out)
    refined_out_path.parent.mkdir(parents=True, exist_ok=True)
    df_refined_boundaries.to_parquet(refined_out_path, index=False)
    logger.info("Saved refined boundaries to: %s", refined_out_path)

    # Generate Refinement & Event Statistics Report
    durations = df_events["duration_sec"].to_numpy()
    num_shots_list = df_events["num_shots"].to_numpy()

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "execution_time_sec": round(t_elapsed, 2),
        "total_videos": int(df_shots["video_id"].nunique()),
        "total_shots": int(len(df_shots)),
        "total_events": int(len(df_events)),
        "boundary_statistics": {
            "threshold": args.threshold,
            "raw_boundary_count": raw_b_count,
            "refined_boundary_count": refined_b_count,
            "suppressed_false_positives": suppressed_count,
            "suppression_percentage": round((suppressed_count / max(1, raw_b_count)) * 100, 2),
        },
        "event_duration_statistics": {
            "mean_sec": round(float(np.mean(durations)), 2),
            "median_sec": round(float(np.median(durations)), 2),
            "min_sec": round(float(np.min(durations)), 2),
            "max_sec": round(float(np.max(durations)), 2),
            "p25_sec": round(float(np.percentile(durations, 25)), 2),
            "p75_sec": round(float(np.percentile(durations, 75)), 2),
        },
        "shots_per_event_statistics": {
            "mean": round(float(np.mean(num_shots_list)), 2),
            "median": int(np.median(num_shots_list)),
            "min": int(np.min(num_shots_list)),
            "max": int(np.max(num_shots_list)),
        },
    }

    report_out_path = Path(args.report_out)
    report_out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 80)
    print("🎉 STAGE 3D BOUNDARY REFINEMENT & EVENT CONSTRUCTION COMPLETED!")
    print("=" * 80)
    print(f" • Total Videos Processed    : {report['total_videos']:,}")
    print(f" • Total Shots               : {report['total_shots']:,}")
    print(f" • Total Events Constructed  : {report['total_events']:,}")
    print(f" • Raw Boundaries            : {raw_b_count:,}")
    print(f" • Refined Boundaries        : {refined_b_count:,} (Suppressed {suppressed_count} slide FPs)")
    print(f" • Mean Event Duration       : {report['event_duration_statistics']['mean_sec']}s")
    print(f" • Mean Shots / Event        : {report['shots_per_event_statistics']['mean']}")
    print(f" • Events Output Path        : {events_out_path}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
