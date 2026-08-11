"""Paired internal comparison for L21-150 KIS VI-only and translation arms."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from typing import Any

K_VALUES = (1, 5, 20, 50, 100)


class L21150KISComparisonError(ValueError):
    """The two reports are not a valid paired KIS comparison."""


def _kis_reports(report: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows = report.get("query_reports")
    if type(rows) is not list:
        raise L21150KISComparisonError("evaluation report has no query_reports array")
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping) or row.get("task") != "kis":
            continue
        query_id = row.get("query_id")
        if type(query_id) is not str or not query_id:
            raise L21150KISComparisonError("KIS query report has no valid query_id")
        if query_id in result:
            raise L21150KISComparisonError(f"duplicate KIS query report: {query_id}")
        result[query_id] = row
    return result


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _metric_at_k(
    rows: Mapping[str, Mapping[str, Any]], field: str, cutoff: int
) -> float:
    values: list[float] = []
    for row in rows.values():
        metric = row.get(field)
        if not isinstance(metric, Mapping) or str(cutoff) not in metric:
            raise L21150KISComparisonError(
                f"query {row.get('query_id')} lacks {field}[{cutoff}]"
            )
        values.append(float(metric[str(cutoff)]))
    return _mean(values)


def _hit_ids(rows: Mapping[str, Mapping[str, Any]], cutoff: int) -> list[str]:
    return sorted(
        query_id
        for query_id, row in rows.items()
        if float(row["video_recall_at_k"][str(cutoff)]) > 0.0
    )


def compare_l21_150_kis_arms(
    arm_a_report: Mapping[str, Any],
    arm_b_report: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare paired DEV results; this makes no official or causal claim."""

    if arm_a_report.get("benchmark_id") != arm_b_report.get("benchmark_id"):
        raise L21150KISComparisonError("benchmark_id mismatch")
    arm_a = _kis_reports(arm_a_report)
    arm_b = _kis_reports(arm_b_report)
    if set(arm_a) != set(arm_b):
        raise L21150KISComparisonError(
            "paired KIS query IDs differ: "
            f"only_a={sorted(set(arm_a) - set(arm_b))}, "
            f"only_b={sorted(set(arm_b) - set(arm_a))}"
        )
    query_ids = sorted(arm_a)
    for query_id in query_ids:
        if arm_a[query_id].get("split") != "DEV" or arm_b[query_id].get("split") != "DEV":
            raise L21150KISComparisonError("comparison is restricted to KIS DEV")

    primary: dict[str, Any] = {}
    for cutoff in K_VALUES:
        a_value = _metric_at_k(arm_a, "video_recall_at_k", cutoff)
        b_value = _metric_at_k(arm_b, "video_recall_at_k", cutoff)
        primary[str(cutoff)] = {
            "arm_a": a_value,
            "arm_b": b_value,
            "delta_b_minus_a": b_value - a_value,
        }
    secondary_frame: dict[str, Any] = {}
    for cutoff in K_VALUES:
        a_value = _metric_at_k(arm_a, "frame_recall_at_k", cutoff)
        b_value = _metric_at_k(arm_b, "frame_recall_at_k", cutoff)
        secondary_frame[str(cutoff)] = {
            "arm_a": a_value,
            "arm_b": b_value,
            "delta_b_minus_a": b_value - a_value,
        }

    hits_a = _hit_ids(arm_a, 100)
    hits_b = _hit_ids(arm_b, 100)
    comparable_ranks = []
    for query_id in query_ids:
        rank_a = arm_a[query_id].get("first_video_hit_rank")
        rank_b = arm_b[query_id].get("first_video_hit_rank")
        if type(rank_a) is int and type(rank_b) is int:
            comparable_ranks.append(
                {
                    "query_id": query_id,
                    "arm_a_first_video_hit_rank": rank_a,
                    "arm_b_first_video_hit_rank": rank_b,
                    "delta_b_minus_a": rank_b - rank_a,
                }
            )

    depth_a = Counter(int(row.get("prediction_count", 0)) for row in arm_a.values())
    depth_b = Counter(int(row.get("prediction_count", 0)) for row in arm_b.values())
    final_a = _mean([float(row.get("final_score", 0.0)) for row in arm_a.values()])
    final_b = _mean([float(row.get("final_score", 0.0)) for row in arm_b.values()])
    return {
        "schema_version": 1,
        "benchmark_id": arm_a_report.get("benchmark_id"),
        "comparison_role": "PAIRED_KIS_DEV_EXPERIMENT_COMPARISON",
        "semantic_gt_authority": "SOURCE_PROPOSED_INTERNAL",
        "official_competition_claim": False,
        "causal_translation_claim": False,
        "arm_a": "VI_ONLY",
        "arm_b": "TRANSLATION_AUGMENTED_RRF",
        "paired_query_count": len(query_ids),
        "video_recall_at_k": primary,
        "target_video_hit_query_ids": {
            "arm_a": hits_a,
            "arm_b": hits_b,
            "rescued_by_arm_b": sorted(set(hits_b) - set(hits_a)),
            "regressed_in_arm_b": sorted(set(hits_a) - set(hits_b)),
        },
        "first_video_hit_rank_comparisons": comparable_ranks,
        "output_depth_distribution": {
            "arm_a": {str(depth): count for depth, count in sorted(depth_a.items())},
            "arm_b": {str(depth): count for depth, count in sorted(depth_b.items())},
        },
        "duplicate_diagnostics": {
            "arm_a_duplicate_count": arm_a_report.get("overall", {}).get(
                "duplicate_count"
            ),
            "arm_b_duplicate_count": arm_b_report.get("overall", {}).get(
                "duplicate_count"
            ),
        },
        "secondary_metrics": {
            "frame_recall_at_k": secondary_frame,
            "btc_like_final_score": {
                "arm_a": final_a,
                "arm_b": final_b,
                "delta_b_minus_a": final_b - final_a,
            },
        },
    }
