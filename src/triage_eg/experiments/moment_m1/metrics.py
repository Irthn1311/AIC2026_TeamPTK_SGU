"""Metrics for the bounded M1 local raw-frame refinement experiment."""

from __future__ import annotations

from statistics import mean, median
from typing import Any

HIT_THRESHOLDS = (1, 5, 10, 30)


def failure_diagnostics(
    *, reference_reachable: bool, coarse_error_frames: int, refined_error_frames: int
) -> list[str]:
    """Return only the descriptive M1 diagnostic codes allowed by the contract."""

    diagnostics = []
    if not reference_reachable:
        diagnostics.append("COARSE_REFERENCE_OUTSIDE_WINDOW")
    if refined_error_frames < coarse_error_frames:
        diagnostics.append("LOCAL_REFINEMENT_IMPROVED")
    elif refined_error_frames == coarse_error_frames:
        diagnostics.append("LOCAL_REFINEMENT_TIED")
    else:
        diagnostics.append("LOCAL_REFINEMENT_REGRESSED")
    return diagnostics


def _arm_metrics(records: list[dict[str, Any]], error_field: str) -> dict[str, Any]:
    errors = [int(item[error_field]) for item in records]
    if not errors:
        return {
            "event_count": 0,
            "MEAN_ABSOLUTE_ERROR_FRAMES": None,
            "MEDIAN_ABSOLUTE_ERROR_FRAMES": None,
            **{f"HIT_WITHIN_{threshold}_FRAMES": None for threshold in HIT_THRESHOLDS},
        }
    return {
        "event_count": len(errors),
        "MEAN_ABSOLUTE_ERROR_FRAMES": mean(errors),
        "MEDIAN_ABSOLUTE_ERROR_FRAMES": median(errors),
        **{
            f"HIT_WITHIN_{threshold}_FRAMES": mean(error <= threshold for error in errors)
            for threshold in HIT_THRESHOLDS
        },
    }


def aggregate_refinement_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate M0/M1 localization and paired refinement outcomes."""

    wins = sum(
        int(item["refined_error_frames"]) < int(item["coarse_error_frames"]) for item in records
    )
    ties = sum(
        int(item["refined_error_frames"]) == int(item["coarse_error_frames"]) for item in records
    )
    regressions = len(records) - wins - ties
    denominator = len(records)
    return {
        "event_count": denominator,
        "M0_BTC_TECHNICAL_KEYFRAME_ANCHOR": _arm_metrics(records, "coarse_error_frames"),
        "M1_LOCAL_RAW_CLIP_COARSE_TO_FINE": _arm_metrics(records, "refined_error_frames"),
        "REFINEMENT_WIN_COUNT": wins,
        "REFINEMENT_WIN_RATE": wins / denominator if denominator else None,
        "REFINEMENT_TIE_COUNT": ties,
        "REFINEMENT_TIE_RATE": ties / denominator if denominator else None,
        "REFINEMENT_REGRESSION_COUNT": regressions,
        "REFINEMENT_REGRESSION_RATE": regressions / denominator if denominator else None,
        "MEAN_ERROR_DELTA_FRAMES": (
            mean(int(item["error_delta"]) for item in records) if records else None
        ),
    }


def build_m1_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the two mandated M1 metric scopes."""

    reachable = [item for item in records if bool(item["reference_reachable"])]
    return {
        "metric_scope": "AI_CURATED_INTERNAL_PSEUDO_GT_NOT_OFFICIAL",
        "arms": [
            "M0_BTC_TECHNICAL_KEYFRAME_ANCHOR",
            "M1_LOCAL_RAW_CLIP_COARSE_TO_FINE",
        ],
        "ALL_EVENTS": aggregate_refinement_metrics(records),
        "REFERENCE_REACHABLE_EVENTS": aggregate_refinement_metrics(reachable),
    }


__all__ = [
    "HIT_THRESHOLDS",
    "aggregate_refinement_metrics",
    "build_m1_metrics",
    "failure_diagnostics",
]
