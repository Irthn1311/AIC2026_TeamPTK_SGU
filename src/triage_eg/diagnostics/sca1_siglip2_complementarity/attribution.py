"""Post-GT paired complementarity and oracle-union diagnostics."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from triage_eg.diagnostics.d1_grounding_attribution import strict_target_chain_exists

from .contracts import (
    FUSION_RESCUE_THRESHOLD,
    FUSION_TRAKE_CHAIN_DELTA_THRESHOLD,
    FUSION_TRAKE_EVENT_DELTA_THRESHOLD,
)


def _rank_change(a0: int | None, s1: int | None) -> str:
    if a0 == s1:
        return "SAME"
    if a0 is None:
        return "IMPROVED" if s1 is not None else "SAME"
    if s1 is None:
        return "WORSENED"
    return "IMPROVED" if s1 < a0 else "WORSENED"


def _unit_rows(audit: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = [*audit["single_rows"], *audit["trake_event_rows"]]
    return {str(row["unit_id"]): row for row in rows}


def paired_unit_deltas(a0_audit: dict[str, Any], s1_audit: dict[str, Any]) -> list[dict[str, Any]]:
    a0, s1 = _unit_rows(a0_audit), _unit_rows(s1_audit)
    if set(a0) != set(s1) or len(a0) != 100:
        raise RuntimeError("SCA1_PAIRED_SEMANTIC_UNIT_SET_MISMATCH")
    output = []
    for unit_id in sorted(a0):
        left, right = a0[unit_id], s1[unit_id]
        row = {
            "unit_id": unit_id,
            "query_id": left["query_id"],
            "task": left["task"],
            "event_id": left.get("event_id"),
            "t3_target_hit_a0": bool(left["t3_pool_has_target"]),
            "t3_target_hit_s1": bool(right["t3_pool_has_target"]),
            "t3_target_hit_u1": bool(left["t3_pool_has_target"] or right["t3_pool_has_target"]),
            "nearest_t3_distance_a0": left.get("nearest_t3_distance"),
            "nearest_t3_distance_s1": right.get("nearest_t3_distance"),
            "primary_failure_a0": left.get("primary_failure_reason"),
            "primary_failure_s1": right.get("primary_failure_reason"),
        }
        for field in ("target_within_video_rank", "target_global_rank", "correct_video_rank"):
            row[f"{field}_a0"] = left.get(field)
            row[f"{field}_s1"] = right.get(field)
            row[f"{field}_change"] = _rank_change(left.get(field), right.get(field))
        a0_top100 = (
            left.get("target_global_rank") is not None and int(left["target_global_rank"]) <= 100
        )
        s1_top100 = (
            right.get("target_global_rank") is not None and int(right["target_global_rank"]) <= 100
        )
        row["target_global_top100_a0"] = a0_top100
        row["target_global_top100_s1"] = s1_top100
        row["target_global_top100_rescue"] = bool(not a0_top100 and s1_top100)
        row["target_global_top100_loss"] = bool(a0_top100 and not s1_top100)
        row["t3_target_rescue"] = bool(
            not left["t3_pool_has_target"] and right["t3_pool_has_target"]
        )
        row["t3_target_loss"] = bool(left["t3_pool_has_target"] and not right["t3_pool_has_target"])
        output.append(row)
    return output


def _target_t3_frames(event: dict[str, Any]) -> set[int]:
    return {
        int(row["original_frame_idx"])
        for row in event.get("t3_pool", [])
        if int(row.get("distance_to_gt", -1)) == 0
    }


def oracle_union_diagnostics(
    a0_audit: dict[str, Any],
    s1_audit: dict[str, Any],
    unit_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    a0_queries = {row["query_id"]: row for row in a0_audit["trake_query_rows"]}
    s1_queries = {row["query_id"]: row for row in s1_audit["trake_query_rows"]}
    a0_events: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    s1_events: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in a0_audit["trake_event_rows"]:
        a0_events[row["query_id"]].append(row)
    for row in s1_audit["trake_event_rows"]:
        s1_events[row["query_id"]].append(row)
    query_rows = []
    for query_id in sorted(a0_queries):
        left, right = a0_queries[query_id], s1_queries[query_id]
        if left["btc_target_chain_exists"] != right["btc_target_chain_exists"]:
            raise RuntimeError(f"SCA1_BTC_REPRESENTATION_CEILING_CHANGED: {query_id}")
        left_events = sorted(a0_events[query_id], key=lambda row: str(row["event_id"]))
        right_events = sorted(s1_events[query_id], key=lambda row: str(row["event_id"]))
        if [row["event_id"] for row in left_events] != [row["event_id"] for row in right_events]:
            raise RuntimeError(f"SCA1_TRAKE_EVENT_ORDER_MISMATCH: {query_id}")
        union_pools = [
            sorted(_target_t3_frames(a0) | _target_t3_frames(s1))
            for a0, s1 in zip(left_events, right_events, strict=True)
        ]
        u1_chain = strict_target_chain_exists(union_pools)
        patterns = []
        for a0, s1 in zip(left_events, right_events, strict=True):
            a0_hit, s1_hit = bool(a0["t3_pool_has_target"]), bool(s1["t3_pool_has_target"])
            patterns.append(
                {
                    "event_id": a0["event_id"],
                    "a0_t3_hit": a0_hit,
                    "s1_t3_hit": s1_hit,
                    "u1_t3_hit": a0_hit or s1_hit,
                    "pattern": "RESCUE"
                    if not a0_hit and s1_hit
                    else "LOSS"
                    if a0_hit and not s1_hit
                    else "BOTH_HIT"
                    if a0_hit and s1_hit
                    else "BOTH_MISS",
                }
            )
        query_rows.append(
            {
                "query_id": query_id,
                "btc_target_chain_a0": bool(left["btc_target_chain_exists"]),
                "btc_target_chain_s1": bool(right["btc_target_chain_exists"]),
                "t3_target_chain_a0": bool(left["t3_target_chain_exists"]),
                "t3_target_chain_s1": bool(right["t3_target_chain_exists"]),
                "t3_target_chain_u1": bool(u1_chain),
                "full_target_chain_top100_a0": bool(left["g1_top100_full_target_chain_exists"]),
                "full_target_chain_top100_s1": bool(right["g1_top100_full_target_chain_exists"]),
                "event_patterns": patterns,
                "union_target_frame_pools": union_pools,
            }
        )
    trake_unit_rows = [row for row in unit_rows if row["task"] == "TRAKE"]
    all_counts = {
        arm: sum(row[f"t3_target_hit_{arm}"] for row in unit_rows) for arm in ("a0", "s1", "u1")
    }
    trake_event_counts = {
        arm: sum(row[f"t3_target_hit_{arm}"] for row in trake_unit_rows)
        for arm in ("a0", "s1", "u1")
    }
    chain_counts = {
        "a0": sum(row["t3_target_chain_a0"] for row in query_rows),
        "s1": sum(row["t3_target_chain_s1"] for row in query_rows),
        "u1": sum(row["t3_target_chain_u1"] for row in query_rows),
    }
    return {
        "semantic_unit_t3_hit_counts": all_counts,
        "trake_t3_event_hit_counts": trake_event_counts,
        "trake_target_chain_counts": chain_counts,
        "trake_queries": query_rows,
    }


def classify_complementarity(
    unit_rows: list[dict[str, Any]], oracle: dict[str, Any]
) -> tuple[str, dict[str, Any]]:
    global_rescues = sum(row["target_global_top100_rescue"] for row in unit_rows)
    trake_events = oracle["trake_t3_event_hit_counts"]
    chains = oracle["trake_target_chain_counts"]
    event_delta = trake_events["u1"] - trake_events["a0"]
    chain_delta = chains["u1"] - chains["a0"]
    conditions = {
        "global_top100_rescues_at_least_5": global_rescues >= FUSION_RESCUE_THRESHOLD,
        "trake_u1_event_delta_at_least_3": event_delta >= FUSION_TRAKE_EVENT_DELTA_THRESHOLD,
        "trake_u1_chain_delta_at_least_2": chain_delta >= FUSION_TRAKE_CHAIN_DELTA_THRESHOLD,
    }
    met = sum(conditions.values())
    if met >= 2:
        classification = "OPEN_BOUNDED_FUSION"
    elif met == 0 and event_delta <= 1 and chain_delta <= 1:
        classification = "NO_USEFUL_COMPLEMENTARITY"
    else:
        classification = "LIMITED_OR_MIXED_COMPLEMENTARITY"
    return classification, {
        "predeclared_conditions": conditions,
        "conditions_met": met,
        "global_top100_rescues": global_rescues,
        "trake_u1_event_delta_over_a0": event_delta,
        "trake_u1_chain_delta_over_a0": chain_delta,
        "fusion_gate": "OPEN" if classification == "OPEN_BOUNDED_FUSION" else "CLOSED",
    }


def summarize_paired(unit_rows: list[dict[str, Any]], oracle: dict[str, Any]) -> dict[str, Any]:
    classification, decision = classify_complementarity(unit_rows, oracle)
    return {
        "status": "COMPLETE",
        "complementarity_classification": classification,
        **decision,
        "unique_s1_target_global_top100_rescues": sum(
            row["target_global_top100_rescue"] for row in unit_rows
        ),
        "unique_a0_target_global_top100_hits_lost_by_s1": sum(
            row["target_global_top100_loss"] for row in unit_rows
        ),
        "unique_s1_t3_event_rescues": sum(row["t3_target_rescue"] for row in unit_rows),
        "unique_a0_t3_event_hits_lost_by_s1": sum(row["t3_target_loss"] for row in unit_rows),
        "rank_change_counts": {
            field: dict(Counter(row[f"{field}_change"] for row in unit_rows))
            for field in (
                "target_within_video_rank",
                "target_global_rank",
                "correct_video_rank",
            )
        },
        **{
            key: oracle[key]
            for key in (
                "semantic_unit_t3_hit_counts",
                "trake_t3_event_hit_counts",
                "trake_target_chain_counts",
            )
        },
        "production_policy_changed": False,
        "fusion_implemented": False,
    }


__all__ = [
    "classify_complementarity",
    "oracle_union_diagnostics",
    "paired_unit_deltas",
    "summarize_paired",
]
