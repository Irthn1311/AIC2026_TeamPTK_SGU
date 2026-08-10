"""Coarse-window reachability and path-diversity metrics for T2."""

from __future__ import annotations

from statistics import mean
from typing import Any


def event_is_reachable(distance_frames: int, fps: float, tolerance_seconds: int) -> bool:
    if distance_frames < 0 or fps <= 0 or tolerance_seconds <= 0:
        raise ValueError("distance, FPS, and tolerance are invalid")
    return distance_frames <= tolerance_seconds * fps


def query_all_events_reachable(event_rows: list[dict[str, Any]], k: int, tolerance: int) -> bool:
    if not event_rows:
        raise ValueError("query reachability requires at least one event")
    return all(bool(row["by_k"][str(k)]["reachable_seconds"][str(tolerance)]) for row in event_rows)


def _slice_metrics(
    event_rows: list[dict[str, Any]],
    query_rows: list[dict[str, Any]],
    k_values: tuple[int, ...],
    tolerances: tuple[int, ...],
) -> dict[str, Any]:
    if not event_rows or not query_rows:
        return {"event_count": len(event_rows), "query_count": len(query_rows), "status": "EMPTY"}
    sweep: dict[str, Any] = {}
    for tolerance in tolerances:
        sweep[str(tolerance)] = {
            **{
                f"EVENT_WINDOW_RECALL@{k}": mean(
                    bool(row["by_k"][str(k)]["reachable_seconds"][str(tolerance)])
                    for row in event_rows
                )
                for k in k_values
            },
            **{
                f"QUERY_ALL_EVENTS_REACHABLE@{k}": mean(
                    bool(row["by_k"][str(k)]["all_events_reachable_seconds"][str(tolerance)])
                    for row in query_rows
                )
                for k in k_values
            },
        }
    primary = sweep["6"]
    return {
        "event_count": len(event_rows),
        "query_count": len(query_rows),
        "PRIMARY_TOLERANCE_SECONDS": 6,
        "PRIMARY_6_SECONDS": primary,
        "TOLERANCE_SWEEP_SECONDS": sweep,
    }


def build_t2_metrics(
    event_rows: list[dict[str, Any]],
    query_rows: list[dict[str, Any]],
    *,
    k_values: tuple[int, ...] = (1, 3, 5),
    tolerances: tuple[int, ...] = (3, 6, 9, 12),
) -> dict[str, Any]:
    overall = _slice_metrics(event_rows, query_rows, k_values, tolerances)
    by_event_count = {}
    for count in (2, 3, 4):
        selected_queries = [row for row in query_rows if int(row["event_count"]) == count]
        query_ids = {str(row["query_id"]) for row in selected_queries}
        selected_events = [row for row in event_rows if str(row["query_id"]) in query_ids]
        by_event_count[str(count)] = _slice_metrics(
            selected_events, selected_queries, k_values, tolerances
        )

    diversity = {}
    for k in k_values:
        values = [row["path_diversity"][str(k)] for row in query_rows]
        diversity[str(k)] = {
            "mean_unique_path_count": mean(item["unique_path_count"] for item in values),
            "mean_duplicate_path_rate": mean(item["duplicate_path_rate"] for item in values),
            "mean_anchor_position_diversity_per_event": mean(
                mean(item["anchor_position_diversity_per_event"])
                for item in values
            ),
            "requested_hypothesis_count": k,
        }
    return {
        "BENCHMARK_TYPE": "AI_CURATED_INTERNAL_PSEUDO_GT",
        "metric_scope": "COARSE_WINDOW_RECALL_NOT_EXACT_FRAME_LOCALIZATION",
        "k_values": list(k_values),
        "tolerance_seconds": list(tolerances),
        "OVERALL": overall,
        "BY_QUERY_EVENT_COUNT": by_event_count,
        "PATH_DIVERSITY": diversity,
    }


def validate_recall_monotonicity(metrics: dict[str, Any]) -> None:
    slices = [metrics["OVERALL"], *metrics["BY_QUERY_EVENT_COUNT"].values()]
    for item in slices:
        if item.get("status") == "EMPTY":
            continue
        for tolerance_values in item["TOLERANCE_SWEEP_SECONDS"].values():
            event_values = [tolerance_values[f"EVENT_WINDOW_RECALL@{k}"] for k in (1, 3, 5)]
            query_values = [
                tolerance_values[f"QUERY_ALL_EVENTS_REACHABLE@{k}"] for k in (1, 3, 5)
            ]
            if event_values != sorted(event_values) or query_values != sorted(query_values):
                raise RuntimeError("T2 Recall@K must be monotonic non-decreasing")


__all__ = [
    "build_t2_metrics",
    "event_is_reachable",
    "query_all_events_reachable",
    "validate_recall_monotonicity",
]
