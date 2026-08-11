"""BTC-like diagnostic evaluator for the internal L21-150 benchmark."""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from .l21_150_answers import answer_matches
from .l21_150_schema import (
    L21150Benchmark,
    L21150KISQuery,
    L21150QAQuery,
    L21150Query,
)

OFFICIAL_K = (1, 5, 20, 50, 100)
GT_POLICIES = {"proposed", "validated-only"}


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _validated_query_ids(mapping_report: Mapping[str, Any] | None) -> set[str]:
    if mapping_report is None:
        return set()
    records = mapping_report.get("records")
    if type(records) is not list:
        raise ValueError("mapping validation report must contain a records array")
    statuses: dict[str, list[str]] = defaultdict(list)
    for record in records:
        if type(record) is not dict:
            raise ValueError("mapping validation record must be an object")
        query_id = record.get("query_id")
        status = record.get("status")
        if type(query_id) is not str or type(status) is not str:
            raise ValueError("mapping validation records require query_id and status")
        statuses[query_id].append(status)
    return {
        query_id
        for query_id, query_statuses in statuses.items()
        if query_statuses and all(status == "VALIDATED" for status in query_statuses)
    }


def _candidate_identity(task_type: str, candidate: Mapping[str, Any]) -> tuple[Any, ...]:
    if task_type == "trake":
        frames = candidate.get("actual_frame_ids")
        return (candidate.get("video_id"), tuple(frames) if type(frames) is list else frames)
    if task_type == "qa":
        return (
            candidate.get("video_id"),
            candidate.get("actual_frame_id"),
            candidate.get("answer"),
        )
    return (candidate.get("video_id"), candidate.get("actual_frame_id"))


def _validate_candidate(
    candidate: Any,
    query: L21150Query,
) -> tuple[dict[str, Any] | None, str | None]:
    if type(candidate) is not dict:
        return None, "prediction must be an object"
    if candidate.get("query_id") != query.query_id:
        return None, "query_id mismatch"
    if candidate.get("task") != query.task_type:
        return None, "task mismatch"
    rank = candidate.get("rank")
    if type(rank) is not int or not 1 <= rank <= 100:
        return None, "rank must be an integer in [1, 100]"
    video_id = candidate.get("video_id")
    if type(video_id) is not str or not video_id.strip():
        return None, "video_id must be non-empty"
    if query.task_type in {"kis", "qa"}:
        frame_id = candidate.get("actual_frame_id")
        if type(frame_id) is not int or frame_id < 0:
            return None, "actual_frame_id must be a non-negative integer"
        if query.task_type == "qa":
            answer = candidate.get("answer")
            if type(answer) is not str or not answer.strip():
                return None, "QA answer must be non-empty"
    else:
        frame_ids = candidate.get("actual_frame_ids")
        if type(frame_ids) is not list or not frame_ids:
            return None, "TRAKE actual_frame_ids must be a non-empty array"
        if any(type(frame_id) is not int or frame_id < 0 for frame_id in frame_ids):
            return None, "TRAKE actual_frame_ids must contain non-negative integers"
    latency = candidate.get("latency_seconds")
    if latency is not None and (
        type(latency) not in {int, float} or not math.isfinite(float(latency)) or latency < 0
    ):
        return None, "latency_seconds must be finite and non-negative"
    return dict(candidate), None


def _prepare_predictions(
    query: L21150Query,
    raw_candidates: Sequence[Any],
) -> tuple[list[dict[str, Any]], list[str], int, int, int]:
    valid: list[dict[str, Any]] = []
    errors: list[str] = []
    invalid_count = 0
    for index, raw_candidate in enumerate(raw_candidates):
        candidate, error = _validate_candidate(raw_candidate, query)
        if error is not None:
            errors.append(f"record {index + 1}: {error}")
            invalid_count += 1
        else:
            valid.append(candidate)

    valid.sort(key=lambda candidate: candidate["rank"])
    ranks = [candidate["rank"] for candidate in valid]
    duplicate_record_indexes: set[int] = set()
    if len(ranks) != len(set(ranks)):
        errors.append("duplicate ranks")
        seen_ranks: set[int] = set()
        for index, rank in enumerate(ranks):
            if rank in seen_ranks:
                duplicate_record_indexes.add(index)
            seen_ranks.add(rank)
    if ranks and ranks != list(range(1, len(ranks) + 1)):
        errors.append("ranks are not contiguous from 1")
    identities = [_candidate_identity(query.task_type, candidate) for candidate in valid]
    duplicate_count = len(identities) - len(set(identities))
    if duplicate_count:
        errors.append(f"duplicate candidate identities: {duplicate_count}")
        seen_identities: set[tuple[Any, ...]] = set()
        for index, identity in enumerate(identities):
            if identity in seen_identities:
                duplicate_record_indexes.add(index)
            seen_identities.add(identity)
    duplicate_record_count = len(duplicate_record_indexes)
    return (
        valid,
        errors,
        invalid_count + duplicate_record_count,
        duplicate_count,
        duplicate_record_count,
    )


def _prefix_max(values: Sequence[float], ranks: Sequence[int], cutoff: int) -> float:
    return max(
        (value for value, rank in zip(values, ranks) if rank <= cutoff),
        default=0.0,
    )


def _query_report(query: L21150Query, candidates: Sequence[dict[str, Any]]) -> dict[str, Any]:
    ranks = [candidate["rank"] for candidate in candidates]
    scores: list[float] = []
    video_scores: list[float] = []
    frame_scores: list[float] = []
    answer_scores: list[float] = []
    order_scores: list[float] = []
    completeness_scores: list[float] = []
    event_hits_by_candidate: list[list[bool]] = []

    for candidate in candidates:
        video_hit = candidate["video_id"] == query.video_id
        video_scores.append(float(video_hit))
        if isinstance(query, L21150KISQuery):
            frame_hit = video_hit and (
                query.proposed_interval.start_frame_id
                <= candidate["actual_frame_id"]
                <= query.proposed_interval.end_frame_id
            )
            frame_scores.append(float(frame_hit))
            scores.append(float(frame_hit))
        elif isinstance(query, L21150QAQuery):
            frame_hit = video_hit and (
                query.proposed_interval.start_frame_id
                <= candidate["actual_frame_id"]
                <= query.proposed_interval.end_frame_id
            )
            answer_hit = answer_matches(candidate["answer"], query.accepted_answers)
            frame_scores.append(float(frame_hit))
            answer_scores.append(float(answer_hit))
            scores.append(float(frame_hit and answer_hit))
        else:
            frame_ids = candidate["actual_frame_ids"]
            required_count = len(query.events)
            hits = [
                video_hit
                and event.proposed_interval.start_frame_id
                <= frame_id
                <= event.proposed_interval.end_frame_id
                for frame_id, event in zip(frame_ids, query.events)
            ]
            hits.extend([False] * (required_count - len(hits)))
            hits = hits[:required_count]
            event_hits_by_candidate.append(hits)
            score = sum(hits) / required_count if video_hit else 0.0
            scores.append(score)
            frame_scores.append(score)
            completeness = min(len(frame_ids), required_count) / required_count
            completeness_scores.append(completeness)
            order_scores.append(
                float(
                    len(frame_ids) == required_count
                    and all(left < right for left, right in zip(frame_ids, frame_ids[1:]))
                )
            )

    r_at_k = {str(cutoff): _prefix_max(scores, ranks, cutoff) for cutoff in OFFICIAL_K}
    final_score = statistics.fmean(r_at_k.values())
    first_relevant_rank = next(
        (rank for rank, score in zip(ranks, scores) if score == 1.0),
        None,
    )
    common = {
        "query_id": query.query_id,
        "task": query.task_type,
        "video_id": query.video_id,
        "branch": query.branch,
        "difficulty": query.difficulty,
        "split": query.split,
        "prediction_count": len(candidates),
        "r_at_k": r_at_k,
        "final_score": final_score,
        "first_relevant_rank": first_relevant_rank,
        "reciprocal_rank": 0.0 if first_relevant_rank is None else 1.0 / first_relevant_rank,
        "video_hit": bool(max(video_scores, default=0.0)),
        "frame_hit": bool(max(frame_scores, default=0.0)),
        "full_hit": bool(max(scores, default=0.0) == 1.0),
    }
    if isinstance(query, L21150KISQuery):
        common["video_recall_at_k"] = {
            str(cutoff): _prefix_max(video_scores, ranks, cutoff) for cutoff in OFFICIAL_K
        }
        common["frame_recall_at_k"] = dict(r_at_k)
    elif isinstance(query, L21150QAQuery):
        grounded_answers = [
            answer
            for answer, frame in zip(answer_scores, frame_scores)
            if frame == 1.0
        ]
        common.update(
            {
                "answer_hit": bool(max(answer_scores, default=0.0)),
                "answer_hit_given_grounding": bool(max(grounded_answers, default=0.0)),
                "full_tuple_hit": bool(max(scores, default=0.0)),
            }
        )
    else:
        required_count = len(query.events)
        per_event = [
            max(
                (float(hits[index]) for hits in event_hits_by_candidate),
                default=0.0,
            )
            for index in range(required_count)
        ]
        common.update(
            {
                "event_count": required_count,
                "per_event_accuracy": per_event,
                "event_coverage": max(scores, default=0.0),
                "event_order_valid": bool(max(order_scores, default=0.0)),
                "full_chain_accuracy": bool(
                    max(scores, default=0.0) == 1.0
                    and max(order_scores, default=0.0) == 1.0
                ),
                "chain_completeness": max(completeness_scores, default=0.0),
            }
        )
    return common


def _mean(values: Iterable[float]) -> float:
    collected = list(values)
    return statistics.fmean(collected) if collected else 0.0


def _summary(reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "query_count": len(reports),
        "r_at_k": {
            str(cutoff): _mean(report["r_at_k"][str(cutoff)] for report in reports)
            for cutoff in OFFICIAL_K
        },
        "final_score": _mean(report["final_score"] for report in reports),
    }


def _slices(reports: Sequence[Mapping[str, Any]], field: str) -> dict[str, Any]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for report in reports:
        groups[str(report[field])].append(report)
    return {key: _summary(groups[key]) for key in sorted(groups)}


def evaluate_l21_150(
    benchmark: L21150Benchmark,
    predictions: Sequence[Mapping[str, Any]],
    *,
    gt_policy: str = "proposed",
    mapping_validation_report: Mapping[str, Any] | None = None,
    split: str = "all",
    task: str = "all",
) -> dict[str, Any]:
    if gt_policy not in GT_POLICIES:
        raise ValueError("gt_policy must be proposed or validated-only")
    if split not in {"all", "dev", "holdout"}:
        raise ValueError("split must be all, dev, or holdout")
    if task not in {"all", "kis", "qa", "trake"}:
        raise ValueError("task must be all, kis, qa, or trake")

    validated_ids = _validated_query_ids(mapping_validation_report)
    selected = [
        query
        for query in benchmark.queries
        if (split == "all" or query.split.casefold() == split)
        and (task == "all" or query.task_type == task)
    ]
    excluded_gt = [
        query.query_id
        for query in selected
        if gt_policy == "validated-only" and query.query_id not in validated_ids
    ]
    if gt_policy == "validated-only":
        if mapping_validation_report is None:
            raise ValueError("validated-only requires a mapping validation report")
        selected = [query for query in selected if query.query_id in validated_ids]
        if not selected:
            raise ValueError("validated-only selected zero MAPPING_VALIDATED_GT queries")

    benchmark_by_id = {query.query_id: query for query in selected}
    raw_by_query: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    orphan_count = 0
    for candidate in predictions:
        query_id = candidate.get("query_id") if isinstance(candidate, Mapping) else None
        if query_id in benchmark_by_id:
            raw_by_query[str(query_id)].append(candidate)
        else:
            orphan_count += 1

    query_reports: list[dict[str, Any]] = []
    total_valid_records = 0
    total_invalid_records = orphan_count
    duplicate_count = 0
    latencies: list[float] = []
    for query in selected:
        (
            prepared,
            errors,
            invalid_count,
            query_duplicate_count,
            duplicate_record_count,
        ) = _prepare_predictions(query, raw_by_query.get(query.query_id, []))
        total_valid_records += len(prepared) - duplicate_record_count
        total_invalid_records += invalid_count
        duplicate_count += query_duplicate_count
        query_latency = next(
            (
                float(candidate["latency_seconds"])
                for candidate in prepared
                if candidate.get("latency_seconds") is not None
            ),
            None,
        )
        if query_latency is not None:
            latencies.append(query_latency)
        report = _query_report(query, prepared)
        report["result_valid"] = not errors
        report["validation_errors"] = errors
        report["invalid_result_count"] = invalid_count
        query_reports.append(report)

    task_reports = {
        task_name: [report for report in query_reports if report["task"] == task_name]
        for task_name in ("kis", "qa", "trake")
    }
    kis_reports = task_reports["kis"]
    qa_reports = task_reports["qa"]
    trake_reports = task_reports["trake"]

    task_metrics = {
        task_name: _summary(reports) for task_name, reports in task_reports.items()
    }
    task_metrics["kis"].update(
        {
            "mrr": _mean(report["reciprocal_rank"] for report in kis_reports),
            "video_recall_at_k": {
                str(cutoff): _mean(
                    report["video_recall_at_k"][str(cutoff)] for report in kis_reports
                )
                for cutoff in OFFICIAL_K
            },
            "frame_recall_at_k": {
                str(cutoff): _mean(
                    report["frame_recall_at_k"][str(cutoff)] for report in kis_reports
                )
                for cutoff in OFFICIAL_K
            },
            "video_miss_count": sum(not report["video_hit"] for report in kis_reports),
            "video_hit_frame_miss_count": sum(
                report["video_hit"] and not report["frame_hit"] for report in kis_reports
            ),
            "frame_hit_low_rank_count": sum(
                report["frame_hit"] and (report["first_relevant_rank"] or 0) > 5
                for report in kis_reports
            ),
        }
    )
    task_metrics["qa"].update(
        {
            "video_accuracy": _mean(float(report["video_hit"]) for report in qa_reports),
            "grounding_accuracy": _mean(
                float(report["frame_hit"]) for report in qa_reports
            ),
            "answer_accuracy": _mean(
                float(report["answer_hit"]) for report in qa_reports
            ),
            "full_tuple_accuracy": _mean(
                float(report["full_tuple_hit"]) for report in qa_reports
            ),
            "answer_right_grounding_wrong_count": sum(
                report["answer_hit"] and not report["frame_hit"] for report in qa_reports
            ),
        }
    )
    task_metrics["trake"].update(
        {
            "video_accuracy": _mean(
                float(report["video_hit"]) for report in trake_reports
            ),
            "event_coverage": _mean(report["event_coverage"] for report in trake_reports),
            "event_order_accuracy": _mean(
                float(report["event_order_valid"]) for report in trake_reports
            ),
            "full_chain_accuracy": _mean(
                float(report["full_chain_accuracy"]) for report in trake_reports
            ),
            "chain_completeness": _mean(
                report["chain_completeness"] for report in trake_reports
            ),
            "per_event_accuracy": [
                _mean(
                    report["per_event_accuracy"][event_index]
                    for report in trake_reports
                    if event_index < len(report["per_event_accuracy"])
                )
                for event_index in range(
                    max(
                        (len(report["per_event_accuracy"]) for report in trake_reports),
                        default=0,
                    )
                )
            ],
        }
    )

    return {
        "schema_version": 1,
        "benchmark_id": benchmark.benchmark_id,
        "benchmark_role": benchmark.benchmark_role,
        "official_ground_truth": False,
        "gt_policy": gt_policy,
        "gt_evidence_mode": (
            "SOURCE_PROPOSED_GT" if gt_policy == "proposed" else "MAPPING_VALIDATED_GT"
        ),
        "semantic_accuracy_claim": False,
        "selected_query_count": len(selected),
        "excluded_unvalidated_query_count": len(excluded_gt),
        "excluded_unvalidated_query_ids": excluded_gt,
        "overall": {
            **_summary(query_reports),
            "valid_result_count": total_valid_records,
            "invalid_result_count": total_invalid_records,
            "top100_depth": max(
                (report["prediction_count"] for report in query_reports), default=0
            ),
            "mean_top100_depth": _mean(
                report["prediction_count"] for report in query_reports
            ),
            "duplicate_count": duplicate_count,
            "duplicate_rate": (
                duplicate_count / (total_valid_records + total_invalid_records)
                if total_valid_records + total_invalid_records
                else 0.0
            ),
            "latency_p50_seconds": _percentile(latencies, 0.5),
            "latency_p95_seconds": _percentile(latencies, 0.95),
        },
        "task_metrics": task_metrics,
        "slices": {
            "task": _slices(query_reports, "task"),
            "branch": _slices(query_reports, "branch"),
            "difficulty": _slices(query_reports, "difficulty"),
            "video": _slices(query_reports, "video_id"),
            "split": _slices(query_reports, "split"),
        },
        "query_reports": query_reports,
    }
