"""Post-inference TRAKE event and chain attribution over frozen T3 pools."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import numpy as np

from triage_eg.e2eg1.contracts import T3_SELECTED_DELTA
from triage_eg.experiments.t3_diverse_temporal import (
    DiverseTemporalPath,
    build_diverse_event_pool,
    enumerate_feasible_paths,
    select_coverage_aware,
)

from .contracts import TRAKE_REASONS, D1Settings, SemanticUnitSnapshot
from .single_event import audit_event_unit, frame_inside_intervals


def strict_target_chain_exists(frame_pools: list[list[int] | tuple[int, ...]]) -> bool:
    """Greedy existence test for one strictly increasing raw-frame choice per event."""

    previous = -1
    for pool in frame_pools:
        candidates = sorted({int(value) for value in pool if int(value) > previous})
        if not candidates:
            return False
        previous = candidates[0]
    return True


def _t3_path_exists(event_rows: list[dict[str, Any]], *, target_only: bool) -> bool:
    pools = []
    lookups = []
    for row in event_rows:
        candidates = tuple(
            candidate
            for candidate in row["t3_candidates"]
            if not target_only
            or frame_inside_intervals(candidate.original_frame_idx, list(row["intervals"]))
        )
        if not candidates:
            return False
        pools.append(candidates)
        lookups.append({candidate.catalog_position: candidate for candidate in candidates})
    feasible, _ = enumerate_feasible_paths(tuple(pools))
    for path in feasible:
        frames = [
            lookups[index][position].original_frame_idx
            for index, position in enumerate(path.positions)
        ]
        if all(left < right for left, right in zip(frames, frames[1:], strict=False)):
            return True
    return False


def _global_t3_chain_ranking(
    units: list[SemanticUnitSnapshot], groups: list[Any], catalog: Any
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Reproduce the frozen T3 global chain order without changing predictions."""

    paths: list[DiverseTemporalPath] = []
    records: dict[tuple[int, ...], dict[str, Any]] = {}
    for group in groups:
        global_rows = np.asarray(group.rows, dtype=np.int64)
        frames = np.asarray(catalog.original_idx[global_rows], dtype=np.int64)
        fps_values = np.asarray(catalog.mapping_fps[global_rows], dtype=np.float64)
        valid_fps = fps_values[np.isfinite(fps_values) & (fps_values > 0)]
        if not len(valid_fps) or not np.allclose(valid_fps, valid_fps[0], rtol=0, atol=1e-6):
            raise RuntimeError(f"MAPPING_FPS_INVALID: {group.video_id}")
        pools = tuple(
            build_diverse_event_pool(
                f"{unit.event_id}@{group.video_id}",
                np.asarray(unit.scores[global_rows], dtype=np.float32),
                frames,
                float(valid_fps[0]),
            )
            for unit in units
        )
        feasible, _ = enumerate_feasible_paths(pools)
        candidate_lookup = [
            {candidate.catalog_position: candidate for candidate in pool} for pool in pools
        ]
        for local_path in feasible:
            global_positions = tuple(
                int(global_rows[position]) for position in local_path.positions
            )
            output_frames = tuple(
                int(candidate_lookup[index][position].original_frame_idx)
                for index, position in enumerate(local_path.positions)
            )
            if any(
                left >= right for left, right in zip(output_frames, output_frames[1:], strict=False)
            ):
                continue
            path = DiverseTemporalPath(
                local_path.score,
                global_positions,
                local_path.region_ids,
                local_path.event_scores,
            )
            paths.append(path)
            records[global_positions] = {
                "video_id": group.video_id,
                "frame_ids": output_frames,
                "score": float(path.score),
            }
    frozen_prefix = select_coverage_aware(tuple(paths), T3_SELECTED_DELTA)
    selected = {path.positions for path in frozen_prefix}
    ordered = [*frozen_prefix]
    ordered.extend(
        path
        for path in sorted(paths, key=lambda item: (-item.score, item.positions))
        if path.positions not in selected
    )

    def deduplicate(sequence: list[DiverseTemporalPath]) -> list[dict[str, Any]]:
        output, seen = [], set()
        for path in sequence:
            row = records[path.positions]
            key = str(row["video_id"]), tuple(int(value) for value in row["frame_ids"])
            if key in seen:
                continue
            seen.add(key)
            output.append({"rank": len(output) + 1, **row})
        return output

    # The frozen pipeline truncates the raw ordered path list to 100 before
    # frame-tuple deduplication. Keep that visible prefix for reproduction and
    # the full deduplicated order only for the post-GT diagnostic ceiling.
    return deduplicate(ordered), deduplicate(ordered[:100])


def classify_trake(row: dict[str, Any], settings: D1Settings) -> str:
    if row["g1_top100_full_target_chain_exists"]:
        return "SUCCESS_FULL_CHAIN"
    if not row["btc_target_chain_exists"]:
        return "BTC_EVENT_REPRESENTATION_GAP"
    if any(
        event["has_btc_target"]
        and not event["t3_pool_has_target"]
        and event["target_within_video_rank"] is not None
        and event["target_within_video_rank"] > settings.t3_pool_limit
        for event in row["events"]
    ):
        return "EVENT_SEMANTIC_SCORE_GAP"
    if all(
        event["target_within_video_rank"] is not None
        and event["target_within_video_rank"] <= settings.t3_pool_limit
        for event in row["events"]
    ) and any(not event["t3_pool_has_target"] for event in row["events"]):
        return "T3_EVENT_POOL_GAP"
    if (
        all(event["t3_pool_has_target"] for event in row["events"])
        and not row["t3_target_chain_exists"]
    ):
        return "MONOTONIC_COMPOSITION_GAP"
    if row["t3_target_chain_exists"] and not row["g1_top100_full_target_chain_exists"]:
        return "GLOBAL_CHAIN_RANKING_GAP"
    return "UNCLASSIFIED_TRAKE"


def audit_trake_query(
    units: list[SemanticUnitSnapshot],
    *,
    ground_truth: dict[str, Any],
    predictions: list[dict[str, Any]],
    groups: list[Any],
    group_by_video: dict[str, Any],
    catalog: Any,
    settings: D1Settings,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not units or any(unit.query_id != units[0].query_id for unit in units):
        raise ValueError("TRAKE units must be non-empty and belong to one query")
    intervals = ground_truth["event_intervals"]
    if len(intervals) != len(units):
        raise ValueError("TRAKE event intervals do not match semantic units")
    correct_video = str(ground_truth["correct_video"])
    internal_events = [
        audit_event_unit(
            unit,
            correct_video=correct_video,
            interval_value=[interval],
            groups=groups,
            group_by_video=group_by_video,
            catalog=catalog,
        )
        for unit, interval in zip(units, intervals, strict=True)
    ]
    event_rows = []
    for row in internal_events:
        public = {
            key: value
            for key, value in row.items()
            if key not in {"target_frames", "t3_candidates", "intervals"}
        }
        public["gt_intervals"] = [list(interval) for interval in row["intervals"]]
        event_rows.append(public)
    target_frame_pools = [list(row["target_frames"]) for row in internal_events]
    btc_chain = strict_target_chain_exists(target_frame_pools)
    t3_target_chain = _t3_path_exists(internal_events, target_only=True)
    t3_feasible_chain = _t3_path_exists(internal_events, target_only=False)
    ordered_predictions = sorted(predictions, key=lambda row: row["rank"])
    global_chains, reproduced_output = _global_t3_chain_ranking(units, groups, catalog)
    predicted_tuples = [
        (str(row["video_id"]), tuple(int(value) for value in row["frame_ids"]))
        for row in ordered_predictions
    ]
    reconstructed_tuples = [
        (str(row["video_id"]), tuple(int(value) for value in row["frame_ids"]))
        for row in reproduced_output
    ]
    if predicted_tuples != reconstructed_tuples:
        raise RuntimeError("D1_T3_GLOBAL_RANKING_REPRODUCTION_MISMATCH")
    correct_video_predictions = [
        row for row in ordered_predictions if row["video_id"] == correct_video
    ]
    correct_video_global = [row for row in global_chains if row["video_id"] == correct_video]
    target_global = [
        row
        for row in correct_video_global
        if all(
            frame_inside_intervals(frame, [tuple(interval)])
            for frame, interval in zip(row["frame_ids"], intervals, strict=True)
        )
    ]
    full_target_predictions = [
        row
        for row in correct_video_predictions
        if all(
            frame_inside_intervals(frame, [tuple(interval)])
            for frame, interval in zip(row["frame_ids"], intervals, strict=True)
        )
    ]
    query_row = {
        "query_id": units[0].query_id,
        "task": "TRAKE",
        "correct_video": correct_video,
        "event_count": len(units),
        "event_ids": [unit.event_id for unit in units],
        "events": event_rows,
        "btc_target_chain_exists": btc_chain,
        "t3_target_chain_exists": t3_target_chain,
        "t3_correct_video_feasible_chain_exists": t3_feasible_chain,
        "best_correct_video_chain_global_rank": min(
            (row["rank"] for row in correct_video_global), default=None
        ),
        "best_correct_video_chain_global_rank_status": (
            "WITHIN_TOP100"
            if correct_video_predictions
            else "BEYOND_TOP100"
            if correct_video_global
            else "NO_FEASIBLE_CHAIN"
        ),
        "best_full_target_chain_global_rank": min(
            (row["rank"] for row in target_global), default=None
        ),
        "t3_global_chain_count": len(global_chains),
        "t3_global_ranking_reproduced": True,
        "g1_top100_correct_video_exists": bool(correct_video_predictions),
        "g1_top100_full_target_chain_exists": bool(full_target_predictions),
        "g1_best_full_target_chain_rank": min(
            (row["rank"] for row in full_target_predictions), default=None
        ),
        "g1_top_prediction": ordered_predictions[0] if ordered_predictions else None,
        "g1_best_correct_video_prediction": (
            correct_video_predictions[0] if correct_video_predictions else None
        ),
    }
    query_row["primary_failure_reason"] = classify_trake(query_row, settings)
    return event_rows, query_row


def summarize_trake(
    event_rows: list[dict[str, Any]], query_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    counts = Counter(row["primary_failure_reason"] for row in query_rows)
    positions: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for query in query_rows:
        for event in query["events"]:
            positions[str(event["event_id"])].append(event)
    position_summary = {}
    for event_id, rows in sorted(positions.items()):
        ranks = [
            float(row["best_target_within_video_rank"])
            for row in rows
            if row["best_target_within_video_rank"] is not None
        ]
        position_summary[event_id] = {
            "query_count": len(rows),
            "btc_target_coverage": sum(row["has_btc_target"] for row in rows) / len(rows),
            "t3_target_coverage": sum(row["t3_pool_has_target"] for row in rows) / len(rows),
            "median_target_within_video_rank": float(np.median(ranks)) if ranks else None,
        }
    return {
        "query_count": len(query_rows),
        "event_count_total": len(event_rows),
        "events_with_btc_target_keyframes": sum(row["has_btc_target"] for row in event_rows),
        "events_with_t3_target_hit": sum(row["t3_pool_has_target"] for row in event_rows),
        "queries_with_btc_target_chain": sum(row["btc_target_chain_exists"] for row in query_rows),
        "queries_with_t3_target_chain": sum(row["t3_target_chain_exists"] for row in query_rows),
        "queries_with_correct_video_feasible_t3_chain": sum(
            row["t3_correct_video_feasible_chain_exists"] for row in query_rows
        ),
        "queries_with_correct_video_in_top100": sum(
            row["g1_top100_correct_video_exists"] for row in query_rows
        ),
        "queries_with_full_target_chain_in_top100": sum(
            row["g1_top100_full_target_chain_exists"] for row in query_rows
        ),
        "primary_failure_counts": {reason: int(counts[reason]) for reason in TRAKE_REASONS},
        "per_event_position": position_summary,
    }


__all__ = [
    "audit_trake_query",
    "classify_trake",
    "strict_target_chain_exists",
    "summarize_trake",
]
