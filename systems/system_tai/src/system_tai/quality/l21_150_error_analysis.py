"""Mechanical error taxonomy for L21-150 diagnostic evaluation reports."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any


def classify_query_report(
    report: Mapping[str, Any],
    *,
    failure_reason: str | None = None,
) -> tuple[str, ...]:
    categories: list[str] = []
    task = report["task"]
    if not report.get("result_valid", True):
        categories.append("OUTPUT_INVALID")

    if task == "kis":
        if not report.get("video_hit", False):
            categories.append("VIDEO_MISS")
        elif not report.get("frame_hit", False):
            categories.append("VIDEO_HIT_FRAME_MISS")
        elif (report.get("first_relevant_rank") or 0) > 5:
            categories.append("FRAME_HIT_LOW_RANK")
        if any(
            "duplicate candidate" in str(error)
            for error in report.get("validation_errors", [])
        ):
            categories.append("DUPLICATE_PRESSURE")
    elif task == "qa":
        if not report.get("video_hit", False):
            categories.append("VIDEO_MISS")
        elif not report.get("frame_hit", False):
            categories.append("FRAME_MISS")
        if report.get("frame_hit", False) and not report.get(
            "answer_hit_given_grounding", False
        ):
            categories.append("ANSWER_MISS")
        if report.get("answer_hit", False) and not report.get("frame_hit", False):
            categories.append("ANSWER_RIGHT_GROUNDING_WRONG")
        if failure_reason and "unsupported" in failure_reason.casefold():
            categories.append("UNSUPPORTED_ANSWER_TYPE")
    elif task == "trake":
        if not report.get("video_hit", False):
            categories.append("VIDEO_MISS")
        if report.get("video_hit", False) and report.get("event_coverage", 0.0) < 1.0:
            categories.append("EVENT_MISS")
        if report.get("video_hit", False) and not report.get("event_order_valid", False):
            categories.append("ORDER_FAIL")
        if report.get("chain_completeness", 0.0) < 1.0:
            categories.append("PARTIAL_CHAIN")
        if report.get("full_chain_accuracy", False) and (
            report.get("first_relevant_rank") or 0
        ) > 5:
            categories.append("FULL_CHAIN_LOW_RANK")
    else:
        categories.append("OUTPUT_INVALID")
    return tuple(dict.fromkeys(categories))


def analyze_l21_150_errors(
    evaluation_report: Mapping[str, Any],
    *,
    failures: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    failure_by_query = {
        str(failure.get("query_id")): str(failure.get("failure_reason", ""))
        for failure in failures
        if failure.get("query_id") is not None
    }
    rows: list[dict[str, Any]] = []
    aggregate_fields = ("task", "branch", "difficulty", "video_id", "split")
    aggregates: dict[str, dict[str, Counter[str]]] = {
        field: defaultdict(Counter) for field in aggregate_fields
    }
    total_counts: Counter[str] = Counter()

    for report in evaluation_report.get("query_reports", []):
        query_id = str(report["query_id"])
        categories = classify_query_report(
            report,
            failure_reason=failure_by_query.get(query_id),
        )
        row = {
            "query_id": query_id,
            "task": report["task"],
            "branch": report["branch"],
            "difficulty": report["difficulty"],
            "video_id": report["video_id"],
            "split": report["split"],
            "categories": list(categories),
            "failure_reason": failure_by_query.get(query_id),
        }
        rows.append(row)
        total_counts.update(categories)
        for field in aggregate_fields:
            aggregates[field][str(row[field])].update(categories)

    return {
        "schema_version": 1,
        "benchmark_id": evaluation_report.get("benchmark_id"),
        "analysis_role": "MECHANICAL_ERROR_TAXONOMY",
        "causal_claims_made": False,
        "query_count": len(rows),
        "category_counts": dict(sorted(total_counts.items())),
        "aggregates": {
            field: {
                group: dict(sorted(counts.items()))
                for group, counts in sorted(groups.items())
            }
            for field, groups in aggregates.items()
        },
        "query_errors": rows,
    }
