"""Pure scoring diagnostics for T2-D candidate and order-constrained ceilings."""

from __future__ import annotations

from collections import Counter
from statistics import mean
from time import monotonic
from typing import Any

import numpy as np

from triage_eg.experiments.reference_rt1 import dante_monotonic_dp
from triage_eg.experiments.reference_rt2 import RT2BenchmarkQuery
from triage_eg.experiments.temporal_t2 import TemporalPath

TOLERANCE_SECONDS = (3, 6, 9, 12)
EVENT_ONLY_CUTOFFS = (1, 3, 5, 10, 20, 50)
K_VALUES = (1, 3, 5)


def stable_event_ranking(scores: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float32)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("event-only scores must be a finite non-empty vector")
    return np.lexsort((np.arange(len(values), dtype=np.int64), -values))


def reference_neighborhood_mask(
    original_frames: np.ndarray,
    reference_original_frame_idx: int,
    fps: float,
    tolerance_seconds: int,
) -> np.ndarray:
    frames = np.asarray(original_frames, dtype=np.int64)
    if frames.ndim != 1 or len(frames) == 0 or fps <= 0 or tolerance_seconds <= 0:
        raise ValueError("reference-neighborhood inputs are invalid")
    return np.abs(frames - int(reference_original_frame_idx)) / fps <= tolerance_seconds


def masked_monotonic_dp(scores: np.ndarray, allowed: np.ndarray) -> TemporalPath | None:
    """Exact lambda-zero DP where each event has an explicit allowed-position mask."""

    matrix = np.asarray(scores, dtype=np.float32)
    mask = np.asarray(allowed, dtype=bool)
    if matrix.ndim != 2 or matrix.shape != mask.shape or not np.isfinite(matrix).all():
        raise ValueError("masked DP requires aligned finite score and mask matrices")
    event_count, position_count = matrix.shape
    if event_count == 0 or position_count == 0 or position_count < event_count:
        return None
    previous = np.full(position_count, -np.inf, dtype=np.float64)
    previous[mask[0]] = matrix[0, mask[0]].astype(np.float64)
    pointers = np.full((event_count, position_count), -1, dtype=np.int64)
    for event_index in range(1, event_count):
        current = np.full(position_count, -np.inf, dtype=np.float64)
        best_score = -np.inf
        best_position = -1
        for position in range(1, position_count):
            predecessor = position - 1
            if previous[predecessor] > best_score:
                best_score = previous[predecessor]
                best_position = predecessor
            if mask[event_index, position] and best_position >= 0:
                current[position] = best_score + float(matrix[event_index, position])
                pointers[event_index, position] = best_position
        previous = current
    final_position = int(np.argmax(previous))
    if not np.isfinite(previous[final_position]):
        return None
    positions = [final_position]
    for event_index in range(event_count - 1, 0, -1):
        positions.append(int(pointers[event_index, positions[-1]]))
    positions.reverse()
    if any(left >= right for left, right in zip(positions[:-1], positions[1:], strict=True)):
        raise RuntimeError("masked diagnostic DP violated strict order")
    return TemporalPath(float(previous[final_position]), tuple(positions))


def _relative_gap(best_score: float, constrained_score: float) -> float | None:
    denominator = abs(best_score)
    return (best_score - constrained_score) / denominator if denominator > 1e-12 else None


def _path_anchors(
    query: RT2BenchmarkQuery,
    path: TemporalPath,
    scores: np.ndarray,
    source_rows: np.ndarray,
    catalog: Any,
) -> list[dict[str, Any]]:
    anchors = []
    for event_index, (event, position) in enumerate(
        zip(query.events, path.positions, strict=True)
    ):
        global_row = int(source_rows[position])
        mapped = catalog.map_row(global_row)
        anchors.append(
            {
                "event_id": event.event_id,
                "catalog_position": int(position),
                "global_row": global_row,
                "n": int(mapped["n"]),
                "original_frame_idx": int(mapped["original_frame_idx"]),
                "event_similarity": float(scores[event_index, position]),
            }
        )
    return anchors


def _source_arrays(catalog: Any, source_rows: np.ndarray) -> tuple[np.ndarray, float]:
    original_frames = np.asarray(catalog.original_idx[source_rows], dtype=np.int64)
    fps_values = np.asarray(catalog.mapping_fps[source_rows], dtype=np.float64)
    if (
        original_frames.shape != source_rows.shape
        or fps_values.shape != source_rows.shape
        or not np.isfinite(fps_values).all()
        or np.any(fps_values <= 0)
        or not np.allclose(fps_values, fps_values[0], rtol=0.0, atol=1e-6)
    ):
        raise ValueError("source-video frame identity/FPS arrays are invalid")
    return original_frames, float(fps_values[0])


def diagnose_source_query(
    query: RT2BenchmarkQuery,
    source_scores: np.ndarray,
    source_rows: np.ndarray,
    catalog: Any,
    t2_paths: tuple[TemporalPath, ...],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, float],
]:
    """Run D1, D2, D3 and reproduce T2 coverage from one reused score matrix."""

    scores = np.asarray(source_scores, dtype=np.float32)
    if scores.shape != (len(query.events), len(source_rows)) or not np.isfinite(scores).all():
        raise ValueError("source score matrix shape is invalid")
    if len(t2_paths) < 5:
        raise ValueError("T2-D requires five T2 paths for every benchmark query")
    original_frames, fps = _source_arrays(catalog, source_rows)
    neighborhood_masks = {
        tolerance: np.stack(
            [
                reference_neighborhood_mask(
                    original_frames,
                    event.reference_original_frame_idx,
                    fps,
                    tolerance,
                )
                for event in query.events
            ]
        )
        for tolerance in TOLERANCE_SECONDS
    }
    if any(not mask.any(axis=1).all() for mask in neighborhood_masks.values()):
        raise RuntimeError("NO_CANONICAL_REFERENCE_NEIGHBORHOOD_CANDIDATE")

    event_only_started = monotonic()
    event_ceiling: list[dict[str, Any]] = []
    for event_index, event in enumerate(query.events):
        ranking = stable_event_ranking(scores[event_index])
        top_position = int(ranking[0])
        by_tolerance = {}
        for tolerance in TOLERANCE_SECONDS:
            inside = neighborhood_masks[tolerance][event_index]
            best_rank_zero = next(
                (rank for rank, position in enumerate(ranking) if inside[position]), None
            )
            if best_rank_zero is None:
                raise RuntimeError("NO_CANONICAL_REFERENCE_NEIGHBORHOOD_CANDIDATE")
            position = int(ranking[best_rank_zero])
            score = float(scores[event_index, position])
            top_score = float(scores[event_index, top_position])
            gap = top_score - score
            by_tolerance[str(tolerance)] = {
                "best_neighborhood_rank": best_rank_zero + 1,
                "best_neighborhood_catalog_position": position,
                "best_neighborhood_original_frame_idx": int(original_frames[position]),
                "best_neighborhood_score": score,
                "score_gap_from_top1": gap,
                "relative_score_gap_from_top1": (
                    gap / abs(top_score) if abs(top_score) > 1e-12 else None
                ),
                "rank_percentile": (best_rank_zero + 1) / len(source_rows),
            }
        primary = by_tolerance["6"]
        event_ceiling.append(
            {
                "query_id": query.query_id,
                "event_id": event.event_id,
                "source_video_id": query.source_video_id,
                "event_count": len(query.events),
                "source_video_keyframe_count": len(source_rows),
                "source_video_fps": fps,
                "reference_original_frame_idx": event.reference_original_frame_idx,
                "reference_catalog_position": event.reference_catalog_position,
                "top1_event_only_score": float(scores[event_index, top_position]),
                **primary,
                "by_tolerance_seconds": by_tolerance,
            }
        )
    event_only_ms = (monotonic() - event_only_started) * 1000

    oracle_started = monotonic()
    unconstrained = dante_monotonic_dp(scores, distance_lambda=0.0)
    if unconstrained is None:
        raise RuntimeError("UNCONSTRAINED_DP_INFEASIBLE")
    if t2_paths[0].positions != unconstrained.positions or not np.isclose(
        t2_paths[0].score, unconstrained.score, rtol=1e-7, atol=1e-7
    ):
        raise RuntimeError("T2_K1_DP_REPRODUCTION_FAILED")
    oracle = masked_monotonic_dp(scores, neighborhood_masks[6])
    oracle_anchors = (
        _path_anchors(query, oracle, scores, source_rows, catalog) if oracle else []
    )
    oracle_gap = unconstrained.score - oracle.score if oracle else None
    query_oracle = {
        "query_id": query.query_id,
        "source_video_id": query.source_video_id,
        "event_count": len(query.events),
        "oracle_path_feasible": oracle is not None,
        "oracle_infeasible_reason": (
            None if oracle else "NO_STRICT_MONOTONIC_REFERENCE_WINDOW_PATH"
        ),
        "unconstrained_best_score": unconstrained.score,
        "unconstrained_path_positions": list(unconstrained.positions),
        "oracle_window_path_score": oracle.score if oracle else None,
        "oracle_absolute_score_gap": oracle_gap,
        "oracle_relative_score_gap": (
            _relative_gap(unconstrained.score, oracle.score) if oracle else None
        ),
        "oracle_path_anchors": oracle_anchors,
        "all_oracle_anchors_inside_event_neighborhoods": bool(
            oracle
            and all(
                neighborhood_masks[6][index, anchor["catalog_position"]]
                for index, anchor in enumerate(oracle_anchors)
            )
        ),
        "strictly_monotonic": bool(
            oracle
            and all(
                left < right
                for left, right in zip(
                    oracle.positions[:-1], oracle.positions[1:], strict=True
                )
            )
        ),
    }
    oracle_ms = (monotonic() - oracle_started) * 1000

    forced_started = monotonic()
    forced_rows: list[dict[str, Any]] = []
    for event_index, event in enumerate(query.events):
        allowed = np.ones_like(scores, dtype=bool)
        allowed[event_index] = neighborhood_masks[6][event_index]
        forced = masked_monotonic_dp(scores, allowed)
        anchors = _path_anchors(query, forced, scores, source_rows, catalog) if forced else []
        gap = unconstrained.score - forced.score if forced else None
        forced_rows.append(
            {
                "query_id": query.query_id,
                "event_id": event.event_id,
                "source_video_id": query.source_video_id,
                "event_count": len(query.events),
                "constrained_event_index": event_index,
                "forced_event_path_feasible": forced is not None,
                "forced_event_infeasible_reason": (
                    None if forced else "NO_STRICT_MONOTONIC_FORCED_EVENT_PATH"
                ),
                "unconstrained_best_score": unconstrained.score,
                "forced_event_best_score": forced.score if forced else None,
                "forced_event_absolute_score_gap": gap,
                "forced_event_relative_score_gap": (
                    _relative_gap(unconstrained.score, forced.score) if forced else None
                ),
                "forced_anchor": anchors[event_index] if forced else None,
                "full_forced_path": anchors,
                "strictly_monotonic": bool(
                    forced
                    and all(
                        left < right
                        for left, right in zip(
                            forced.positions[:-1], forced.positions[1:], strict=True
                        )
                    )
                ),
            }
        )
    forced_ms = (monotonic() - forced_started) * 1000

    t2_join: list[dict[str, Any]] = []
    for event_index, event in enumerate(query.events):
        positions = [int(path.positions[event_index]) for path in t2_paths[:5]]
        errors = [
            abs(int(original_frames[position]) - event.reference_original_frame_idx) / fps
            for position in positions
        ]
        t2_join.append(
            {
                "query_id": query.query_id,
                "event_id": event.event_id,
                "t2_k5_reachable_6s": min(errors) <= 6.0,
                "t2_k5_unique_anchor_count": len(set(positions)),
                "t2_k5_minimum_seconds_error": min(errors),
                "t2_k5_anchor_positions": positions,
            }
        )

    single_path = {}
    for k in K_VALUES:
        single_path[str(k)] = any(
            all(
                neighborhood_masks[6][event_index, path.positions[event_index]]
                for event_index in range(len(query.events))
            )
            for path in t2_paths[:k]
        )
    query_t2 = {
        "query_id": query.query_id,
        "event_count": len(query.events),
        "single_path_all_events_reachable_6s": single_path,
        "k5_anchor_diversity_per_event": [
            len({path.positions[event_index] for path in t2_paths[:5]})
            for event_index in range(len(query.events))
        ],
    }
    return (
        event_ceiling,
        forced_rows,
        query_oracle,
        t2_join,
        query_t2,
        {
            "event_only_diagnostic_ms": event_only_ms,
            "oracle_dp_ms": oracle_ms,
            "forced_event_dp_ms": forced_ms,
        },
    )


def _rank_bucket(rank: int) -> str:
    if rank == 1:
        return "1"
    if rank <= 3:
        return "2_TO_3"
    if rank <= 5:
        return "4_TO_5"
    if rank <= 10:
        return "6_TO_10"
    if rank <= 20:
        return "11_TO_20"
    if rank <= 50:
        return "21_TO_50"
    return "GT_50"


def build_t2d_metrics(
    event_ceiling: list[dict[str, Any]],
    forced_rows: list[dict[str, Any]],
    query_oracles: list[dict[str, Any]],
    t2_join: list[dict[str, Any]],
    query_t2: list[dict[str, Any]],
) -> dict[str, Any]:
    if len(event_ceiling) != 74 or len(t2_join) != 74 or len(query_t2) != 24:
        raise ValueError("T2-D metrics require exactly 74 events and 24 queries")
    ceiling_by_tolerance = {}
    for tolerance in TOLERANCE_SECONDS:
        ranks = [
            row["by_tolerance_seconds"][str(tolerance)]["best_neighborhood_rank"]
            for row in event_ceiling
        ]
        ceiling_by_tolerance[str(tolerance)] = {
            f"EVENT_ONLY_WINDOW_RECALL@{cutoff}": mean(rank <= cutoff for rank in ranks)
            for cutoff in EVENT_ONLY_CUTOFFS
        }
    primary_ranks = [int(row["best_neighborhood_rank"]) for row in event_ceiling]
    missed_ids = {
        (str(row["query_id"]), str(row["event_id"]))
        for row in t2_join
        if not row["t2_k5_reachable_6s"]
    }
    missed_forced = [
        row
        for row in forced_rows
        if (str(row["query_id"]), str(row["event_id"])) in missed_ids
    ]
    missed_ceiling = [
        row
        for row in event_ceiling
        if (str(row["query_id"]), str(row["event_id"])) in missed_ids
    ]
    rank_cumulative = {
        "RANK_LE_5": sum(row["best_neighborhood_rank"] <= 5 for row in missed_ceiling),
        "RANK_LE_10": sum(row["best_neighborhood_rank"] <= 10 for row in missed_ceiling),
        "RANK_LE_20": sum(row["best_neighborhood_rank"] <= 20 for row in missed_ceiling),
        "RANK_LE_50": sum(row["best_neighborhood_rank"] <= 50 for row in missed_ceiling),
        "RANK_GT_50": sum(row["best_neighborhood_rank"] > 50 for row in missed_ceiling),
    }
    gap_counts = Counter()
    for row in missed_forced:
        value = row["forced_event_relative_score_gap"]
        if value is None:
            gap_counts["UNDEFINED"] += 1
        elif value <= 0.005:
            gap_counts["LE_0_5_PERCENT"] += 1
        elif value <= 0.01:
            gap_counts["GT_0_5_TO_1_PERCENT"] += 1
        elif value <= 0.02:
            gap_counts["GT_1_TO_2_PERCENT"] += 1
        elif value <= 0.05:
            gap_counts["GT_2_TO_5_PERCENT"] += 1
        else:
            gap_counts["GT_5_PERCENT"] += 1
    event_diversities = [
        value for row in query_t2 for value in row["k5_anchor_diversity_per_event"]
    ]
    query_means = [mean(row["k5_anchor_diversity_per_event"]) for row in query_t2]
    return {
        "BENCHMARK_TYPE": "AI_CURATED_INTERNAL_PSEUDO_GT",
        "primary_tolerance_seconds": 6,
        "D1_EVENT_ONLY_CEILING": {
            "by_tolerance_seconds": ceiling_by_tolerance,
            "best_neighborhood_rank_distribution_6s": dict(
                sorted(Counter(_rank_bucket(rank) for rank in primary_ranks).items())
            ),
        },
        "D2_ORDER_CONSTRAINED_ORACLE": {
            "query_count": len(query_oracles),
            "feasible_count": sum(row["oracle_path_feasible"] for row in query_oracles),
            "infeasible_count": sum(not row["oracle_path_feasible"] for row in query_oracles),
        },
        "D3_FORCED_EVENT": {
            "event_count": len(forced_rows),
            "feasible_count": sum(row["forced_event_path_feasible"] for row in forced_rows),
            "k5_miss_forced_gap_buckets": dict(sorted(gap_counts.items())),
        },
        "T2_REPRODUCTION": {
            "k5_reachable_6s_count": sum(row["t2_k5_reachable_6s"] for row in t2_join),
            "k5_missed_6s_count": len(missed_ids),
            "single_path_all_events_reachable_6s_counts": {
                str(k): sum(row["single_path_all_events_reachable_6s"][str(k)] for row in query_t2)
                for k in K_VALUES
            },
        },
        "K5_FAILURE_DESCRIPTIVE_BUCKETS": {
            "event_only_rank_cumulative": rank_cumulative,
            "forced_event_relative_score_gap_exclusive": dict(sorted(gap_counts.items())),
        },
        "K5_PATH_DIVERSITY": {
            "QUERY_WEIGHTED_MEAN_ANCHOR_DIVERSITY": mean(query_means),
            "EVENT_WEIGHTED_MEAN_ANCHOR_DIVERSITY": mean(event_diversities),
            "event_level_unique_anchor_distribution": dict(
                sorted(Counter(str(value) for value in event_diversities).items())
            ),
        },
        "INTERPRETATION_MATRIX": {
            "status": "DESCRIPTIVE_HINTS_ONLY_NO_AUTOMATIC_ASSIGNMENT",
            "patterns": [
                "PATH_SELECTION_OR_DIVERSITY_LIMITATION",
                "TEMPORAL_COMPATIBILITY_OR_SEQUENCE_SCORING_LIMITATION",
                "EVENT_REPRESENTATION_OR_QUERY_RETRIEVAL_LIMITATION",
            ],
        },
        "optional_larger_k": "NOT_RUN_REQUIRES_CHANGE_TO_FROZEN_T2_MAX_BEAM_WIDTH",
    }


def validate_expected_t2_reproduction(metrics: dict[str, Any]) -> None:
    reproduced = metrics["T2_REPRODUCTION"]
    if reproduced["k5_reachable_6s_count"] != 54 or reproduced["k5_missed_6s_count"] != 20:
        raise RuntimeError("T2_K5_REPRODUCTION_FAILED: expected 54/74 and 20 misses")
    if reproduced["single_path_all_events_reachable_6s_counts"] != {
        "1": 5,
        "3": 7,
        "5": 9,
    }:
        raise RuntimeError("T2_SINGLE_PATH_REPRODUCTION_FAILED: expected K1/K3/K5 = 5/7/9")


__all__ = [
    "EVENT_ONLY_CUTOFFS",
    "K_VALUES",
    "TOLERANCE_SECONDS",
    "build_t2d_metrics",
    "diagnose_source_query",
    "masked_monotonic_dp",
    "reference_neighborhood_mask",
    "stable_event_ranking",
    "validate_expected_t2_reproduction",
]
