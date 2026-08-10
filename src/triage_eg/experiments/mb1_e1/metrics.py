"""Interval-first metrics for the bounded MB1-E1 comparison."""

from __future__ import annotations

from collections import Counter
from statistics import mean, median
from typing import Any

import numpy as np

TOLERANCES = (1, 5, 10, 30)


def interval_hit(frame: int, start: int, end: int) -> bool:
    if start > end:
        raise ValueError("interval start must not exceed interval end")
    return start <= frame <= end


def distance_to_interval(frame: int, start: int, end: int) -> int:
    """Return zero inside the interval, otherwise distance to its nearest boundary."""

    if start > end:
        raise ValueError("interval start must not exceed interval end")
    if frame < start:
        return start - frame
    if frame > end:
        return frame - end
    return 0


def _arm_metrics(rows: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    distances = [int(row[f"{prefix}_distance_to_interval"]) for row in rows]
    preferred = [abs(int(row[f"{prefix}_frame"]) - int(row["preferred_frame"])) for row in rows]
    return {
        "event_count": len(rows),
        "INTERVAL_HIT_RATE": mean(value == 0 for value in distances),
        "MEAN_DISTANCE_TO_INTERVAL": mean(distances),
        "MEDIAN_DISTANCE_TO_INTERVAL": median(distances),
        **{
            f"HIT_WITHIN_{tolerance}_RAW_FRAMES_RATE": mean(
                value <= tolerance for value in distances
            )
            for tolerance in TOLERANCES
        },
        "preferred_frame_MAE": mean(preferred),
        "preferred_frame_metric_role": "SECONDARY_DIAGNOSTIC_ONLY",
    }


def _correlation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    x = np.asarray([row["score_gain"] for row in rows], dtype=np.float64)
    y = np.asarray([row["interval_error_improvement"] for row in rows], dtype=np.float64)
    if len(rows) < 2:
        return {"pearson_r": None, "status": "UNDEFINED_TOO_FEW_SAMPLES"}
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return {"pearson_r": None, "status": "UNDEFINED_ZERO_VARIANCE"}
    return {"pearson_r": float(np.corrcoef(x, y)[0, 1]), "status": "DEFINED"}


def aggregate_interval_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"event_count": 0, "status": "EMPTY_SLICE"}
    outcomes = Counter(str(row["pairwise_outcome"]) for row in rows)
    return {
        "event_count": len(rows),
        "M0_SOURCE_ANCHOR_FRAME": _arm_metrics(rows, "m0"),
        "M1_LOCAL_RAW_CLIP_COARSE_TO_FINE": _arm_metrics(rows, "m1"),
        "pairwise": {
            "M1_WINS": outcomes["M1_WINS"],
            "M0_WINS": outcomes["M0_WINS"],
            "TIES": outcomes["TIES"],
        },
        "clip_score_gain_vs_interval_error_improvement": _correlation(rows),
        "small_slice_warning": len(rows) < 5,
    }


def build_mb1_e1_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    high = [row for row in rows if row["annotation_confidence"] == "HIGH"]
    confidence_values = sorted({str(row["annotation_confidence"]) for row in rows})
    moment_types = sorted({str(row["moment_type"]) for row in rows})
    return {
        "primary_metric_semantics": "DISTANCE_TO_ACCEPTABLE_INTERVAL",
        "preferred_frame_semantics": "SECONDARY_DIAGNOSTIC_ONLY",
        "primary_decision_slice": "ALL_HIGH_MEDIUM",
        "ALL_HIGH_MEDIUM": aggregate_interval_metrics(rows),
        "HIGH_CONFIDENCE_ONLY": aggregate_interval_metrics(high),
        "BY_ANNOTATION_CONFIDENCE": {
            value: aggregate_interval_metrics(
                [row for row in rows if row["annotation_confidence"] == value]
            )
            for value in confidence_values
        },
        "BY_MOMENT_TYPE": {
            value: aggregate_interval_metrics(
                [row for row in rows if row["moment_type"] == value]
            )
            for value in moment_types
        },
    }
