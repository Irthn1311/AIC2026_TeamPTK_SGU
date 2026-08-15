"""Post-inference BTC, score, T3, and G1 attribution for one semantic unit."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from aic2026_eval.contracts import accepted_intervals
from triage_eg.e2eg1.ranking import canonical_coarse_candidates
from triage_eg.experiments.t2d_ceiling import stable_event_ranking
from triage_eg.experiments.t3_diverse_temporal import build_diverse_event_pool

from .contracts import SINGLE_EVENT_REASONS, D1Settings, SemanticUnitSnapshot


def frame_distance_to_intervals(frame: int, intervals: list[tuple[int, int]]) -> int:
    value = int(frame)
    return min(
        0 if start <= value <= end else min(abs(value - start), abs(value - end))
        for start, end in intervals
    )


def frame_inside_intervals(frame: int, intervals: list[tuple[int, int]]) -> bool:
    return any(start <= int(frame) <= end for start, end in intervals)


def _inverse_ranks(order: np.ndarray) -> np.ndarray:
    output = np.empty(len(order), dtype=np.int64)
    output[order] = np.arange(1, len(order) + 1, dtype=np.int64)
    return output


def _video_ranking(scores: np.ndarray, groups: list[Any], catalog: Any) -> list[dict[str, Any]]:
    rows = []
    for group in groups:
        local = np.asarray(scores[group.rows], dtype=np.float32)
        order = stable_event_ranking(local)
        position = int(order[0])
        global_row = int(group.rows[position])
        rows.append(
            {
                "video_id": group.video_id,
                "video_score": float(local[position]),
                "best_global_row": global_row,
                "best_frame_idx": int(catalog.original_idx[global_row]),
            }
        )
    ordered = sorted(rows, key=lambda row: (-row["video_score"], row["video_id"]))
    return [{"video_rank": rank, **row} for rank, row in enumerate(ordered, 1)]


def audit_event_unit(
    unit: SemanticUnitSnapshot,
    *,
    correct_video: str,
    interval_value: Any,
    groups: list[Any],
    group_by_video: dict[str, Any],
    catalog: Any,
) -> dict[str, Any]:
    """Measure one event without altering its finalized frozen score vector."""

    scores = np.asarray(unit.scores, dtype=np.float32)
    if (
        scores.ndim != 1
        or len(scores) != len(catalog.original_idx)
        or not np.isfinite(scores).all()
    ):
        raise ValueError("D1 requires one finalized finite full-corpus score vector")
    if correct_video not in group_by_video:
        raise ValueError(f"correct video absent from frozen catalog: {correct_video}")
    intervals = accepted_intervals(interval_value)
    group = group_by_video[correct_video]
    global_rows = np.asarray(group.rows, dtype=np.int64)
    frames = np.asarray(catalog.original_idx[global_rows], dtype=np.int64)
    local_scores = scores[global_rows]
    distances = np.asarray(
        [frame_distance_to_intervals(int(frame), intervals) for frame in frames], dtype=np.int64
    )
    target_positions = np.flatnonzero(distances == 0)
    nearest_position = min(
        range(len(global_rows)), key=lambda index: (int(distances[index]), int(global_rows[index]))
    )
    local_order = stable_event_ranking(local_scores)
    local_ranks = _inverse_ranks(local_order)
    best_position = int(local_order[0])
    best_target_position = (
        min((int(value) for value in target_positions), key=lambda value: int(local_ranks[value]))
        if len(target_positions)
        else None
    )
    global_order = stable_event_ranking(scores)
    global_ranks = _inverse_ranks(global_order)
    videos = _video_ranking(scores, groups, catalog)
    correct_video_row = next(row for row in videos if row["video_id"] == correct_video)

    fps_values = np.asarray(catalog.mapping_fps[global_rows], dtype=np.float64)
    valid_fps = fps_values[np.isfinite(fps_values) & (fps_values > 0)]
    if not len(valid_fps) or not np.allclose(valid_fps, valid_fps[0], rtol=0, atol=1e-6):
        raise RuntimeError(f"MAPPING_FPS_INVALID: {correct_video}")
    t3_pool = build_diverse_event_pool(
        unit.event_id or "E1", local_scores, frames, float(valid_fps[0])
    )
    t3_rows = [
        {
            "pool_rank": rank,
            "event_id": item.event_id,
            "event_region_id": item.event_region_id,
            "catalog_position": int(item.catalog_position),
            "global_row": int(global_rows[item.catalog_position]),
            "original_frame_idx": int(item.original_frame_idx),
            "score": float(item.similarity),
            "distance_to_gt": frame_distance_to_intervals(item.original_frame_idx, intervals),
        }
        for rank, item in enumerate(t3_pool, 1)
    ]
    t3_hits = [row for row in t3_rows if row["distance_to_gt"] == 0]
    nearest_t3 = min(
        t3_rows,
        key=lambda row: (row["distance_to_gt"], row["pool_rank"], row["global_row"]),
    )

    unique_best: dict[int, tuple[float, int]] = {}
    for position, frame in enumerate(frames):
        candidate = float(local_scores[position]), int(global_rows[position])
        current = unique_best.get(int(frame))
        if current is None or (-candidate[0], candidate[1]) < (-current[0], current[1]):
            unique_best[int(frame)] = candidate
    unique_order = sorted(unique_best.items(), key=lambda item: (-item[1][0], item[1][1]))
    unique_rank = {frame: rank for rank, (frame, _) in enumerate(unique_order, 1)}

    best_target_score = (
        float(local_scores[best_target_position]) if best_target_position is not None else None
    )
    best_target_frame = (
        int(frames[best_target_position]) if best_target_position is not None else None
    )
    return {
        "unit_id": unit.unit_id,
        "query_id": unit.query_id,
        "task": unit.task,
        "event_id": unit.event_id,
        "correct_video": correct_video,
        "correct_video_btc_row_count": len(global_rows),
        "target_btc_rows_inside_gt": int(len(target_positions)),
        "target_unique_frame_ids_inside_gt": int(len(set(frames[target_positions].tolist()))),
        "nearest_btc_distance_to_gt_frames": int(distances[nearest_position]),
        "nearest_btc_frame_idx": int(frames[nearest_position]),
        "has_btc_keyframe_inside_gt": bool(len(target_positions)),
        "has_btc_target": bool(len(target_positions)),
        "best_target_score": best_target_score,
        "best_target_global_row": (
            int(global_rows[best_target_position]) if best_target_position is not None else None
        ),
        "best_target_frame_idx": best_target_frame,
        "best_target_within_video_rank": (
            int(local_ranks[best_target_position]) if best_target_position is not None else None
        ),
        "best_target_score_rank_among_unique_frames": (
            unique_rank[best_target_frame] if best_target_frame is not None else None
        ),
        "correct_video_best_score": float(local_scores[best_position]),
        "correct_video_best_frame_idx": int(frames[best_position]),
        "semantic_target_score_gap": (
            float(local_scores[best_position]) - best_target_score
            if best_target_score is not None
            else None
        ),
        "best_target_global_frame_rank": (
            int(global_ranks[int(global_rows[best_target_position])])
            if best_target_position is not None
            else None
        ),
        "correct_video_rank_by_best_frame": int(correct_video_row["video_rank"]),
        "correct_video_score_by_best_frame": float(correct_video_row["video_score"]),
        "t3_pool": t3_rows,
        "t3_pool_has_exact_gt_hit": bool(t3_hits),
        "t3_pool_has_target": bool(t3_hits),
        "t3_best_distance_to_gt": int(nearest_t3["distance_to_gt"]),
        "t3_pool_target_rank": min((row["pool_rank"] for row in t3_hits), default=None),
        "nearest_t3_candidate_frame": int(nearest_t3["original_frame_idx"]),
        "target_frames": tuple(sorted(set(int(value) for value in frames[target_positions]))),
        "t3_candidates": tuple(t3_pool),
        "intervals": tuple(intervals),
    }


def classify_single_event(row: dict[str, Any], settings: D1Settings) -> str:
    if row["g1_has_target"]:
        return "SUCCESS_G1_TARGET_HIT"
    if not row["has_btc_target"]:
        return "BTC_REPRESENTATION_GAP"
    if (
        not row["t3_pool_has_target"]
        and row["target_within_video_rank"] is not None
        and row["target_within_video_rank"] > settings.t3_pool_limit
    ):
        return "TARGET_SEMANTIC_SCORE_WEAK"
    if not row["t3_pool_has_target"]:
        return "T3_REGION_REPRESENTATIVE_GAP"
    if (
        not row["entered_final_top100"]
        and row["correct_video_rank"] > settings.g1_coverage_video_limit
        and (row["g0_rank"] is None or row["g0_rank"] > 100)
    ):
        return "GLOBAL_VIDEO_RANKING_GAP"
    if row["t3_pool_has_target"] and not row["entered_final_top100"]:
        return "G1_ALLOCATION_GAP"
    return "UNCLASSIFIED_SINGLE_EVENT"


def audit_single_event(
    unit: SemanticUnitSnapshot,
    *,
    ground_truth: dict[str, Any],
    predictions: list[dict[str, Any]],
    source_pool: tuple[dict[str, Any], ...],
    g1_allocation: tuple[dict[str, Any], ...],
    groups: list[Any],
    group_by_video: dict[str, Any],
    catalog: Any,
    settings: D1Settings,
    full_qa_correct: bool | None = None,
) -> dict[str, Any]:
    correct_video = str(ground_truth["correct_video"])
    core = audit_event_unit(
        unit,
        correct_video=correct_video,
        interval_value=ground_truth["acceptable_intervals"],
        groups=groups,
        group_by_video=group_by_video,
        catalog=catalog,
    )
    intervals = list(core["intervals"])
    query_predictions = sorted(predictions, key=lambda row: row["rank"])
    g1_hits = [
        row
        for row in query_predictions
        if row["video_id"] == correct_video and frame_inside_intervals(row["frame_id"], intervals)
    ]
    canonical = canonical_coarse_candidates(source_pool)
    source_by_global = {int(row["global_row"]): row for row in canonical}
    allocation_by_global = {int(row["global_row"]): row for row in g1_allocation}
    target_t3 = [row for row in core["t3_pool"] if row["distance_to_gt"] == 0]
    useful = min(target_t3, key=lambda row: row["pool_rank"], default=None)
    source = source_by_global.get(int(useful["global_row"])) if useful else None
    if useful and source is None:
        source = next(
            (
                item
                for item in canonical
                if item["video_id"] == correct_video
                and int(item["original_frame_idx"]) == int(useful["original_frame_idx"])
            ),
            None,
        )
    allocated = allocation_by_global.get(int(useful["global_row"])) if useful else None
    if useful and allocated is None:
        allocated = next(
            (
                row
                for row in g1_allocation
                if row["video_id"] == correct_video
                and int(row["original_frame_idx"]) == int(useful["original_frame_idx"])
            ),
            None,
        )
    row = {
        key: value
        for key, value in core.items()
        if key not in {"target_frames", "t3_candidates", "intervals"}
    }
    row.update(
        {
            "gt_intervals": [list(interval) for interval in intervals],
            "grounding_correct": bool(g1_hits),
            "full_qa_correct": full_qa_correct,
            "g1_has_target": bool(g1_hits),
            "g1_best_target_rank": min((item["rank"] for item in g1_hits), default=None),
            "g1_top100_correct_video_exists": any(
                item["video_id"] == correct_video for item in query_predictions
            ),
            "g0_rank": int(source["g0_rank"]) if source else None,
            "g1_rank": int(allocated["coverage_rank"]) if allocated else None,
            "video_hypothesis_rank": (
                int(allocated["video_hypothesis_rank"])
                if allocated
                else core["correct_video_rank_by_best_frame"]
            ),
            "within_video_region_rank": int(useful["pool_rank"]) if useful else None,
            "was_in_coverage_block": bool(allocated and allocated["was_in_coverage_block"]),
            "entered_final_top100": allocated is not None,
            "G1_ALLOCATION_MISS": bool(target_t3 and not g1_hits),
            "target_within_video_rank": core["best_target_within_video_rank"],
            "target_global_rank": core["best_target_global_frame_rank"],
            "correct_video_rank": core["correct_video_rank_by_best_frame"],
            "nearest_btc_distance": core["nearest_btc_distance_to_gt_frames"],
            "nearest_t3_distance": core["t3_best_distance_to_gt"],
            "best_useful_t3_global_row": int(useful["global_row"]) if useful else None,
            "best_useful_t3_frame_idx": (int(useful["original_frame_idx"]) if useful else None),
            "g1_top_prediction": query_predictions[0] if query_predictions else None,
        }
    )
    row["primary_failure_reason"] = classify_single_event(row, settings)
    return row


def _distribution(rows: list[dict[str, Any]], field: str) -> dict[str, float | None]:
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    if not values:
        return {"median": None, "p75": None, "p90": None}
    return {
        "median": float(np.median(values)),
        "p75": float(np.percentile(values, 75)),
        "p90": float(np.percentile(values, 90)),
    }


def summarize_single_events(rows: list[dict[str, Any]], task: str) -> dict[str, Any]:
    members = [row for row in rows if row["task"] == task]
    counts = Counter(row["primary_failure_reason"] for row in members)
    return {
        "task": task,
        "query_count": len(members),
        "primary_failure_counts": {reason: int(counts[reason]) for reason in SINGLE_EVENT_REASONS},
        "g1_successful_target_hits": counts["SUCCESS_G1_TARGET_HIT"],
        "grounding_correct_count": sum(bool(row["grounding_correct"]) for row in members),
        "full_qa_correct_count": (
            sum(bool(row["full_qa_correct"]) for row in members) if task == "QA" else None
        ),
        "correct_video_rank": _distribution(members, "correct_video_rank"),
        "best_target_within_video_rank": _distribution(members, "target_within_video_rank"),
        "best_target_global_rank": _distribution(members, "target_global_rank"),
        "nearest_btc_distance_to_gt": _distribution(members, "nearest_btc_distance"),
        "nearest_t3_distance_to_gt": _distribution(members, "nearest_t3_distance"),
    }


__all__ = [
    "audit_event_unit",
    "audit_single_event",
    "classify_single_event",
    "frame_distance_to_intervals",
    "frame_inside_intervals",
    "summarize_single_events",
]
