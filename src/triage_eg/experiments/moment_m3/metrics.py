"""Interval-first evaluation and frozen M3 KEEP/DROP policy."""

from __future__ import annotations

from collections import Counter
from statistics import mean, median
from typing import Any

TOLERANCES = (1, 3, 5, 10)
ARMS = ("m1", "m3_a1", "m3_a2")


def distance_to_intervals(frame: int, intervals: list[list[int]]) -> int:
    if not intervals:
        raise ValueError("M3 evaluation requires at least one accepted interval")
    distances = []
    for start, end in intervals:
        if start > end:
            raise ValueError("M3 accepted interval is invalid")
        distances.append(
            0 if start <= frame <= end else start - frame if frame < start else frame - end
        )
    return min(distances)


def evaluate_predictions_only(
    registry_row: dict[str, Any], predictions: dict[str, int]
) -> dict[str, Any]:
    """Evaluation-only boundary: this is the only layer that consumes GT intervals."""
    intervals = registry_row.get("accepted_intervals")
    if not isinstance(intervals, list) or not intervals:
        raise ValueError("M3 evaluation row has no accepted intervals")
    output = {
        "case_id": registry_row["case_id"],
        "video_id": registry_row["video_id"],
        "moment_type": registry_row["moment_type"],
        "primary_gate": bool(registry_row["primary_gate"]),
        "conditional": bool(registry_row["conditional"]),
        "accepted_intervals": intervals,
    }
    for arm in ARMS:
        frame = int(predictions[arm])
        distance = distance_to_intervals(frame, intervals)
        output[f"{arm}_prediction"] = frame
        output[f"{arm}_distance"] = distance
        output[f"{arm}_hit"] = distance == 0
    return output


def _arm_metrics(rows: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    distances = [int(row[f"{arm}_distance"]) for row in rows]
    if not distances:
        return {"case_count": 0, "status": "EMPTY_SLICE"}
    return {
        "case_count": len(rows),
        "interval_hit_count": sum(value == 0 for value in distances),
        "interval_hit_rate": mean(value == 0 for value in distances),
        "mean_distance_to_interval": mean(distances),
        "median_distance_to_interval": median(distances),
        **{
            f"within_{limit}_frame": sum(value <= limit for value in distances)
            for limit in TOLERANCES
        },
    }


def _paired(rows: list[dict[str, Any]], left: str, right: str) -> dict[str, int]:
    outcomes = Counter()
    for row in rows:
        left_distance = int(row[f"{left}_distance"])
        right_distance = int(row[f"{right}_distance"])
        outcomes[
            "wins"
            if left_distance < right_distance
            else "losses"
            if left_distance > right_distance
            else "ties"
        ] += 1
    return {name: outcomes[name] for name in ("wins", "ties", "losses")}


def aggregate_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "case_count": len(rows),
        "A0_M1": _arm_metrics(rows, "m1"),
        "A1_M3_STATE_TRANSITION": _arm_metrics(rows, "m3_a1"),
        "A2_M3_MOTION_TIEBREAKER": _arm_metrics(rows, "m3_a2"),
        "A1_VS_M1": _paired(rows, "m3_a1", "m1"),
        "A2_VS_M1": _paired(rows, "m3_a2", "m1"),
        "A2_VS_A1": _paired(rows, "m3_a2", "m3_a1"),
    }


def build_metrics(
    rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    primary = [row for row in rows if row["primary_gate"] and not row["conditional"]]
    secondary = [row for row in rows if row["conditional"]]
    moment_types = sorted({str(row["moment_type"]) for row in rows})
    by_type = {
        moment_type: {
            **aggregate_metrics([row for row in primary if row["moment_type"] == moment_type]),
            "coverage_status": (
                "ELIGIBLE_FOR_TYPE_DECISION"
                if sum(row["moment_type"] == moment_type for row in primary) >= 3
                else "INSUFFICIENT_TYPE_COVERAGE"
            ),
        }
        for moment_type in moment_types
    }
    return aggregate_metrics(primary), by_type, aggregate_metrics(secondary)


def _type_decision(metrics: dict[str, Any], moment_type: str) -> str:
    if moment_type == "EXTREMUM":
        return "UNSUPPORTED" if metrics.get("case_count", 0) else "INSUFFICIENT_COVERAGE"
    if metrics.get("case_count", 0) < 3:
        return "INSUFFICIENT_COVERAGE"
    m1 = metrics["A0_M1"]
    m3 = metrics["A1_M3_STATE_TRANSITION"]
    paired = metrics["A1_VS_M1"]
    interval_improved = m3["interval_hit_count"] >= m1["interval_hit_count"] + 1
    median_improved = m3["median_distance_to_interval"] <= m1["median_distance_to_interval"] - 1
    keep = (
        m3["interval_hit_rate"] >= m1["interval_hit_rate"]
        and paired["wins"] >= paired["losses"]
        and (median_improved or interval_improved)
    )
    return "KEEP" if keep else "DROP"


def decide_m3(
    primary: dict[str, Any], by_type: dict[str, Any], *, primary_case_count: int
) -> dict[str, Any]:
    type_names = ("ONSET", "CONTACT", "FIRST_OCCURRENCE", "SEPARATION", "EXTREMUM")
    type_decisions = {
        name: _type_decision(by_type.get(name, {"case_count": 0}), name) for name in type_names
    }
    a0 = primary["A0_M1"]
    a1 = primary["A1_M3_STATE_TRANSITION"]
    a2 = primary["A2_M3_MOTION_TIEBREAKER"]
    a1_pair = primary["A1_VS_M1"]
    a2_a1 = primary["A2_VS_A1"]
    global_keep = bool(
        primary_case_count
        and a1.get("interval_hit_rate", -1) >= a0.get("interval_hit_rate", 0)
        and a1.get("median_distance_to_interval", float("inf"))
        < a0.get("median_distance_to_interval", float("inf"))
        and a1_pair["wins"] > a1_pair["losses"]
    )
    motion_keep = bool(
        primary_case_count
        and a2.get("median_distance_to_interval", float("inf"))
        < a1.get("median_distance_to_interval", float("inf"))
        and a2.get("interval_hit_count", 0) >= a1.get("interval_hit_count", 0)
        and a2_a1["losses"] <= a2_a1["wins"]
    )
    kept_types = [name for name, value in type_decisions.items() if value == "KEEP"]
    if primary_case_count < 8:
        global_decision = "DIAGNOSTIC_ONLY"
    elif global_keep:
        global_decision = "KEEP_AS_TYPE_SPECIFIC_BOUNDARY_REFINER"
    elif kept_types:
        global_decision = "PARTIAL_KEEP"
    else:
        global_decision = "DROP"
    return {
        "M3_BENCHMARK_COVERAGE": (
            "TOO_SMALL_FOR_KEEP_DROP"
            if primary_case_count < 8
            else "ADEQUATE_FOR_BOUNDED_KEEP_DROP"
        ),
        **{f"M3_{name}": value for name, value in type_decisions.items()},
        "M3_MOTION_TIEBREAKER": "KEEP" if motion_keep else "DROP",
        "M3_GLOBAL": global_decision,
        "M1_GENERAL_LOCAL_REFINER": "KEEP",
        "PRODUCTION_ROUTER_CHANGE_REQUIRED": "YES"
        if kept_types and primary_case_count >= 8
        else "NO",
        "M3_FURTHER_RESEARCH_REQUIRED": "NO",
        "NEXT_STEP": (
            "INTEGRATE_KEPT_M3_TYPES_WITH_T3_M1"
            if kept_types and primary_case_count >= 8
            else "KEEP_M1_ONLY_AND_MOVE_TO_END_TO_END"
        ),
        "RETURN_TO_MAIN_PIPELINE": "YES",
    }


__all__ = [
    "ARMS",
    "TOLERANCES",
    "aggregate_metrics",
    "build_metrics",
    "decide_m3",
    "distance_to_intervals",
    "evaluate_predictions_only",
]
