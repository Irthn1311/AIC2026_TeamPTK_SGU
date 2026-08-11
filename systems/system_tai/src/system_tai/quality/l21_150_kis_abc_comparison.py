"""Paired internal A/B/C ablation for the L21-150 KIS DEV benchmark.

Input: three evaluator JSON objects for VI-only, VI+EN RRF, and EN-only.
Output: deterministic paired DEV diagnostics without an official or causal claim.
Status: experimental quality-analysis tooling; it never calls retrieval or models.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping
from typing import Any

K_VALUES = (1, 5, 20, 50, 100)
EXPECTED_Q2_DEV_QUERY_COUNT = 38


class L21150KISABCComparisonError(ValueError):
    """The three reports do not form the frozen paired KIS DEV ablation."""


def _finite_number(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise L21150KISABCComparisonError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise L21150KISABCComparisonError(f"{context} must be finite")
    return result


def _metric_map(row: Mapping[str, Any], field: str, query_id: str) -> dict[int, float]:
    payload = row.get(field)
    if not isinstance(payload, Mapping):
        raise L21150KISABCComparisonError(f"query {query_id} lacks {field}")
    result: dict[int, float] = {}
    for cutoff in K_VALUES:
        key = str(cutoff)
        if key not in payload:
            raise L21150KISABCComparisonError(
                f"query {query_id} lacks {field}[{cutoff}]"
            )
        value = _finite_number(payload[key], f"query {query_id} {field}[{cutoff}]")
        if not 0.0 <= value <= 1.0:
            raise L21150KISABCComparisonError(
                f"query {query_id} {field}[{cutoff}] must be in [0, 1]"
            )
        result[cutoff] = value
    return result


def _load_arm(
    report: Mapping[str, Any],
    arm_name: str,
) -> dict[str, Mapping[str, Any]]:
    if type(report.get("benchmark_id")) is not str or not report["benchmark_id"]:
        raise L21150KISABCComparisonError(f"{arm_name} has no valid benchmark_id")
    rows = report.get("query_reports")
    if type(rows) is not list:
        raise L21150KISABCComparisonError(f"{arm_name} has no query_reports array")

    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise L21150KISABCComparisonError(f"{arm_name} contains a malformed query report")
        if row.get("task") != "kis":
            raise L21150KISABCComparisonError(
                f"{arm_name} comparison accepts KIS query reports only"
            )
        query_id = row.get("query_id")
        if type(query_id) is not str or not query_id:
            raise L21150KISABCComparisonError(f"{arm_name} KIS row has no valid query_id")
        if query_id in result:
            raise L21150KISABCComparisonError(
                f"{arm_name} has duplicate KIS query report: {query_id}"
            )
        if row.get("split") != "DEV":
            raise L21150KISABCComparisonError(
                f"{arm_name} comparison is restricted to KIS DEV"
            )
        _metric_map(row, "video_recall_at_k", query_id)
        _metric_map(row, "frame_recall_at_k", query_id)
        prediction_count = row.get("prediction_count")
        if (
            type(prediction_count) is not int
            or prediction_count < 0
            or prediction_count > 100
        ):
            raise L21150KISABCComparisonError(
                f"{arm_name} query {query_id} has invalid prediction_count"
            )
        first_rank = row.get("first_video_hit_rank")
        if first_rank is not None and (
            type(first_rank) is not int or first_rank < 1 or first_rank > 100
        ):
            raise L21150KISABCComparisonError(
                f"{arm_name} query {query_id} has invalid first_video_hit_rank"
            )
        _finite_number(row.get("final_score"), f"{arm_name} query {query_id} final_score")
        result[query_id] = row

    if len(result) != EXPECTED_Q2_DEV_QUERY_COUNT:
        raise L21150KISABCComparisonError(
            f"{arm_name} must contain exactly {EXPECTED_Q2_DEV_QUERY_COUNT} KIS DEV queries; "
            f"found {len(result)}"
        )
    overall = report.get("overall")
    if not isinstance(overall, Mapping):
        raise L21150KISABCComparisonError(f"{arm_name} has no overall object")
    duplicate_count = overall.get("duplicate_count")
    if type(duplicate_count) is not int or duplicate_count < 0:
        raise L21150KISABCComparisonError(
            f"{arm_name} overall.duplicate_count must be a non-negative integer"
        )
    return result


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _metric_at_k(
    rows: Mapping[str, Mapping[str, Any]], field: str, cutoff: int
) -> float:
    return _mean([float(row[field][str(cutoff)]) for row in rows.values()])


def _metric_comparison(
    arm_a: Mapping[str, Mapping[str, Any]],
    arm_b: Mapping[str, Mapping[str, Any]],
    arm_c: Mapping[str, Mapping[str, Any]],
    field: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for cutoff in K_VALUES:
        a_value = _metric_at_k(arm_a, field, cutoff)
        b_value = _metric_at_k(arm_b, field, cutoff)
        c_value = _metric_at_k(arm_c, field, cutoff)
        result[str(cutoff)] = {
            "arm_a": a_value,
            "arm_b": b_value,
            "arm_c": c_value,
            "delta_b_minus_a": b_value - a_value,
            "delta_c_minus_a": c_value - a_value,
            "delta_c_minus_b": c_value - b_value,
        }
    return result


def _hit_ids(rows: Mapping[str, Mapping[str, Any]], cutoff: int = 100) -> list[str]:
    return sorted(
        query_id
        for query_id, row in rows.items()
        if float(row["video_recall_at_k"][str(cutoff)]) > 0.0
    )


def _depth_distribution(rows: Mapping[str, Mapping[str, Any]]) -> dict[str, int]:
    counts = Counter(int(row["prediction_count"]) for row in rows.values())
    return {str(depth): count for depth, count in sorted(counts.items())}


def _duplicate_count(report: Mapping[str, Any]) -> int:
    overall = report["overall"]
    assert isinstance(overall, Mapping)
    value = overall["duplicate_count"]
    assert type(value) is int
    return value


def compare_l21_150_kis_abc_arms(
    arm_a_report: Mapping[str, Any],
    arm_b_report: Mapping[str, Any],
    arm_c_report: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare the frozen paired DEV arms without inferring causality."""

    arm_a = _load_arm(arm_a_report, "arm_a")
    arm_b = _load_arm(arm_b_report, "arm_b")
    arm_c = _load_arm(arm_c_report, "arm_c")
    benchmark_ids = {
        arm_a_report["benchmark_id"],
        arm_b_report["benchmark_id"],
        arm_c_report["benchmark_id"],
    }
    if len(benchmark_ids) != 1:
        raise L21150KISABCComparisonError("benchmark_id mismatch")
    if set(arm_a) != set(arm_b) or set(arm_a) != set(arm_c):
        raise L21150KISABCComparisonError(
            "paired KIS query IDs differ: "
            f"a_not_b={sorted(set(arm_a) - set(arm_b))}, "
            f"b_not_a={sorted(set(arm_b) - set(arm_a))}, "
            f"a_not_c={sorted(set(arm_a) - set(arm_c))}, "
            f"c_not_a={sorted(set(arm_c) - set(arm_a))}"
        )

    hits_a = set(_hit_ids(arm_a))
    hits_b = set(_hit_ids(arm_b))
    hits_c = set(_hit_ids(arm_c))
    rank_comparisons: list[dict[str, Any]] = []
    for query_id in sorted(hits_b & hits_c):
        rank_a = arm_a[query_id].get("first_video_hit_rank")
        rank_b = arm_b[query_id].get("first_video_hit_rank")
        rank_c = arm_c[query_id].get("first_video_hit_rank")
        if type(rank_b) is not int or type(rank_c) is not int:
            raise L21150KISABCComparisonError(
                f"query {query_id} hit by B and C lacks a valid first-hit rank"
            )
        rank_comparisons.append(
            {
                "query_id": query_id,
                "arm_a_first_video_hit_rank": rank_a,
                "arm_b_first_video_hit_rank": rank_b,
                "arm_c_first_video_hit_rank": rank_c,
                "delta_c_minus_b": rank_c - rank_b,
            }
        )

    final_a = _mean([float(row["final_score"]) for row in arm_a.values()])
    final_b = _mean([float(row["final_score"]) for row in arm_b.values()])
    final_c = _mean([float(row["final_score"]) for row in arm_c.values()])
    return {
        "schema_version": 1,
        "benchmark_id": arm_a_report["benchmark_id"],
        "comparison_role": "PAIRED_KIS_DEV_ABC_ABLATION",
        "semantic_gt_authority": "SOURCE_PROPOSED_INTERNAL",
        "official_competition_claim": False,
        "causal_translation_claim": False,
        "holdout_used": False,
        "arm_a": "VI_ONLY",
        "arm_b": "TRANSLATION_AUGMENTED_RRF",
        "arm_c": "EN_ONLY",
        "paired_query_count": len(arm_a),
        "video_recall_at_k": _metric_comparison(
            arm_a, arm_b, arm_c, "video_recall_at_k"
        ),
        "target_video_hit_query_ids": {
            "arm_a": sorted(hits_a),
            "arm_b": sorted(hits_b),
            "arm_c": sorted(hits_c),
            "b_rescued_vs_a": sorted(hits_b - hits_a),
            "b_regressed_vs_a": sorted(hits_a - hits_b),
            "c_rescued_vs_a": sorted(hits_c - hits_a),
            "c_regressed_vs_a": sorted(hits_a - hits_c),
            "c_unique_vs_b": sorted(hits_c - hits_b),
            "b_unique_vs_c": sorted(hits_b - hits_c),
            "shared_b_c_hits": sorted(hits_b & hits_c),
        },
        "first_target_video_hit_rank_comparisons": rank_comparisons,
        "output_depth_distribution": {
            "arm_a": _depth_distribution(arm_a),
            "arm_b": _depth_distribution(arm_b),
            "arm_c": _depth_distribution(arm_c),
        },
        "duplicate_diagnostics": {
            "arm_a_duplicate_count": _duplicate_count(arm_a_report),
            "arm_b_duplicate_count": _duplicate_count(arm_b_report),
            "arm_c_duplicate_count": _duplicate_count(arm_c_report),
        },
        "secondary_metrics": {
            "frame_recall_at_k": _metric_comparison(
                arm_a, arm_b, arm_c, "frame_recall_at_k"
            ),
            "btc_like_final_score": {
                "arm_a": final_a,
                "arm_b": final_b,
                "arm_c": final_c,
                "delta_b_minus_a": final_b - final_a,
                "delta_c_minus_a": final_c - final_a,
                "delta_c_minus_b": final_c - final_b,
            },
        },
    }
