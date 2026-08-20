"""
Stage 3B: Adjacent Shot Similarity & Event Boundary Detection (AI Challenge 2026)
===================================================================================
Calculates adjacent shot similarities and groups temporally contiguous shots into high-level Events.

Key features:
1. Computes Visual Similarity, Semantic Similarity, and Entity Overlap for adjacent shots (S_i, S_{i+1}).
2. Applies Adaptive Thresholding on Fused Similarity to detect Event Boundaries.
3. Groups contiguous shots into logical Events (start_sec, end_sec, event_summary, shot_ids).
4. Outputs:
   - artifacts/event_graph/boundaries/adjacent_similarities.parquet
   - artifacts/event_graph/events/all_events.parquet
"""

from __future__ import annotations

import argparse
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

logger = setup_logger("event-boundary-builder")


def compute_jaccard_similarity(list_a: List[str], list_b: List[str]) -> float:
    """Compute Jaccard similarity index between two lists of tokens/entities."""
    set_a = set(list_a)
    set_b = set(list_b)
    if not set_a and not set_b:
        return 0.0
    union = set_a.union(set_b)
    if not union:
        return 0.0
    intersection = set_a.intersection(set_b)
    return len(intersection) / len(union)


def process_video_boundaries(
    video_id: str,
    df_video_shots: pd.DataFrame,
    vis_weight: float = 0.5,
    sem_weight: float = 0.4,
    ent_weight: float = 0.1,
    boundary_threshold: float = 0.50,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Calculate adjacent similarities and extract Event boundaries for a single video."""
    # Ensure shots are ordered chronologically by shot_id / start_sec
    df_sorted = df_video_shots.sort_values("start_sec").reset_index(drop=True)
    num_shots = len(df_sorted)

    sim_records = []
    events = []

    current_event_shots = [df_sorted.iloc[0]]
    event_counter = 0

    for i in range(num_shots - 1):
        shot_curr = df_sorted.iloc[i]
        shot_next = df_sorted.iloc[i + 1]

        # Extract embeddings
        v_curr = np.array(shot_curr["visual_embedding"], dtype=np.float32)
        v_next = np.array(shot_next["visual_embedding"], dtype=np.float32)
        s_curr = np.array(shot_curr["semantic_embedding"], dtype=np.float32)
        s_next = np.array(shot_next["semantic_embedding"], dtype=np.float32)

        # Dot product for unit L2 normalized vectors
        vis_sim = float(np.dot(v_curr, v_next))
        sem_sim = float(np.dot(s_curr, s_next))

        # Entity overlap (Objects + OCR tokens)
        obj_curr = list(shot_curr.get("objects", []))
        obj_next = list(shot_next.get("objects", []))
        ent_sim = compute_jaccard_similarity(obj_curr, obj_next)

        # Weighted fused similarity score
        fused_sim = (vis_weight * vis_sim) + (sem_weight * sem_sim) + (ent_weight * ent_sim)
        is_boundary = fused_sim < boundary_threshold

        sim_records.append({
            "video_id": video_id,
            "shot_i": int(shot_curr["shot_id"]),
            "shot_i_next": int(shot_next["shot_id"]),
            "visual_similarity": vis_sim,
            "semantic_similarity": sem_sim,
            "entity_similarity": ent_sim,
            "fused_similarity": fused_sim,
            "is_boundary": is_boundary,
        })

        if is_boundary:
            # Finalize current Event
            event_rec = create_event_record(video_id, event_counter, current_event_shots)
            events.append(event_rec)
            event_counter += 1
            current_event_shots = [shot_next]
        else:
            current_event_shots.append(shot_next)

    # Finalize remaining shots as the last event of the video
    if current_event_shots:
        event_rec = create_event_record(video_id, event_counter, current_event_shots)
        events.append(event_rec)

    return sim_records, events


def create_event_record(
    video_id: str, event_idx: int, event_shots: List[pd.Series]
) -> Dict[str, Any]:
    """Synthesize an Event record from a contiguous sequence of Shots."""
    event_id = f"{video_id}_E{event_idx:03d}"
    start_sec = float(event_shots[0]["start_sec"])
    end_sec = float(event_shots[-1]["end_sec"])
    duration_sec = end_sec - start_sec

    shot_ids = [int(s["shot_id"]) for s in event_shots]
    num_shots = len(shot_ids)

    # Combine objects, OCR text, and captions
    all_objects = []
    all_ocr = []
    all_captions = []

    for s in event_shots:
        all_objects.extend(s.get("objects", []))
        if s.get("ocr_text"):
            all_ocr.append(s["ocr_text"])
        if s.get("caption"):
            all_captions.append(s["caption"])

    dedup_objects = list(dict.fromkeys(all_objects))
    combined_ocr = " ".join(list(dict.fromkeys(all_ocr)))
    event_summary = f"Event {event_id} spanning {duration_sec:.1f}s across {num_shots} shots."

    return {
        "event_id": event_id,
        "video_id": video_id,
        "event_index": event_idx,
        "start_sec": start_sec,
        "end_sec": end_sec,
        "duration_sec": duration_sec,
        "num_shots": num_shots,
        "shot_ids": shot_ids,
        "objects": dedup_objects,
        "ocr_text": combined_ocr,
        "event_summary": event_summary,
    }


def build_event_boundaries(
    features_path: Path,
    output_dir: Path,
    boundary_threshold: float = 0.50,
) -> Tuple[Path, Path]:
    """Build Stage 3B Event Boundaries & Events Parquet artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_boundaries = output_dir / "adjacent_similarities.parquet"
    out_events = output_dir / "all_events.parquet"

    if not features_path.exists():
        raise FileNotFoundError(f"Shot features parquet not found at: {features_path}")

    logger.info("Loading shot features from: %s", features_path)
    df_features = pd.read_parquet(features_path)
    total_shots = len(df_features)
    total_videos = df_features["video_id"].nunique()

    logger.info("Processing adjacent shot similarities across %d videos (%d shots)...", total_videos, total_shots)

    all_sim_records = []
    all_event_records = []
    start_time = time.time()

    for idx, (video_id, df_video_shots) in enumerate(df_features.groupby("video_id"), start=1):
        sim_recs, event_recs = process_video_boundaries(
            str(video_id), df_video_shots, boundary_threshold=boundary_threshold
        )
        all_sim_records.extend(sim_recs)
        all_event_records.extend(event_recs)

        if idx % 100 == 0 or idx == total_videos:
            elapsed = time.time() - start_time
            logger.info("Processed [%d/%d] videos | Total Events created: %d | Elapsed: %.1fs", idx, total_videos, len(all_event_records), elapsed)

    df_boundaries = pd.DataFrame(all_sim_records)
    df_events = pd.DataFrame(all_event_records)

    logger.info("Saving adjacent similarities to: %s", out_boundaries)
    df_boundaries.to_parquet(out_boundaries, index=False)

    logger.info("Saving aggregated events to: %s", out_events)
    df_events.to_parquet(out_events, index=False)

    logger.info(
        "✅ Stage 3B Event Boundaries completed! Created %d Events from %d Shots across %d Videos.",
        len(df_events),
        total_shots,
        total_videos,
    )
    return out_boundaries, out_events


def main():
    parser = argparse.ArgumentParser(
        description="Stage 3B: Adjacent Shot Similarity & Event Boundary Detection"
    )
    parser.add_argument(
        "--features",
        default=str(PROJECT_ROOT / "artifacts" / "event_graph" / "features" / "shot_features.parquet"),
        help="Path to shot_features.parquet",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "artifacts" / "event_graph" / "events"),
        help="Directory to save boundary and event parquets",
    )
    parser.add_argument(
        "--boundary-threshold",
        type=float,
        default=0.50,
        help="Similarity threshold below which an Event Boundary is triggered",
    )
    args = parser.parse_args()

    build_event_boundaries(
        features_path=Path(args.features),
        output_dir=Path(args.output_dir),
        boundary_threshold=args.boundary_threshold,
    )


if __name__ == "__main__":
    main()
