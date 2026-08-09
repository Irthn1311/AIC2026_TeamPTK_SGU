"""Unified quality evaluation that delegates official metrics to P0-A."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from statistics import fmean
from typing import Any

from system_tai.preliminary.evaluation import (
    OFFICIAL_K,
    QueryEvaluationReport,
    evaluate_dataset,
    evaluate_ranked_query,
)
from system_tai.preliminary.matching import NormalizedAliasAnswerMatcher
from system_tai.preliminary.scoring import (
    score_kis_prediction,
    score_qa_prediction,
    score_trake_prediction,
)
from system_tai.preliminary.top100 import RankedTop100Query

from .schema import (
    AnnotationStatus,
    Difficulty,
    KISQualityQuery,
    LabelOrigin,
    QAQualityQuery,
    QualityBenchmark,
    QualityQuery,
    QualityTaskType,
    TRAKEQualityQuery,
)


@dataclass(frozen=True, slots=True)
class QualityQueryReport:
    query_id: str
    task_type: QualityTaskType
    difficulty: Difficulty
    tags: tuple[str, ...]
    label_origin: LabelOrigin
    evaluation: QueryEvaluationReport


@dataclass(frozen=True, slots=True)
class QualityTaskSummary:
    task_type: QualityTaskType
    query_count: int
    mean_r_at_1: float
    mean_r_at_5: float
    mean_r_at_20: float
    mean_r_at_50: float
    mean_r_at_100: float
    mean_final_score: float


@dataclass(frozen=True, slots=True)
class QualityBreakdown:
    dimension: str
    value: str
    query_count: int
    mean_final_score: float


@dataclass(frozen=True, slots=True)
class QualityEvaluationReport:
    benchmark_id: str
    include_synthetic: bool
    scored_query_count: int
    skipped_draft_count: int
    skipped_synthetic_count: int
    query_reports: tuple[QualityQueryReport, ...]
    task_summaries: tuple[QualityTaskSummary, ...]
    breakdowns: tuple[QualityBreakdown, ...]
    overall_query_macro_score: float
    task_macro_score: float


def _eligible_queries(
    benchmark: QualityBenchmark,
    *,
    include_synthetic: bool,
) -> tuple[tuple[QualityQuery, ...], int, int]:
    scored: list[QualityQuery] = []
    skipped_drafts = 0
    skipped_synthetic = 0
    for query in benchmark.queries:
        if query.annotation_status is AnnotationStatus.DRAFT:
            skipped_drafts += 1
            continue
        if query.label_origin is LabelOrigin.SYNTHETIC and not include_synthetic:
            skipped_synthetic += 1
            continue
        scored.append(query)
    return tuple(scored), skipped_drafts, skipped_synthetic


def _scorer(query: QualityQuery) -> Callable[[Any, Any], float]:
    if type(query) is KISQualityQuery:
        return score_kis_prediction
    if type(query) is TRAKEQualityQuery:
        return score_trake_prediction
    if type(query) is QAQualityQuery:
        matcher = NormalizedAliasAnswerMatcher(strip_punctuation=True)
        return lambda prediction, ground_truth: score_qa_prediction(
            prediction,
            ground_truth,
            matcher,
        )
    raise TypeError(f"unsupported quality query type: {type(query).__name__}")


def _mean_metric(
    reports: Sequence[QualityQueryReport],
    attribute: str,
) -> float:
    if not reports:
        return 0.0
    return fmean(getattr(report.evaluation, attribute) for report in reports)


def _task_summary(
    task_type: QualityTaskType,
    reports: Sequence[QualityQueryReport],
) -> QualityTaskSummary:
    means = {k: _mean_metric(reports, f"r_at_{k}") for k in OFFICIAL_K}
    return QualityTaskSummary(
        task_type=task_type,
        query_count=len(reports),
        mean_r_at_1=means[1],
        mean_r_at_5=means[5],
        mean_r_at_20=means[20],
        mean_r_at_50=means[50],
        mean_r_at_100=means[100],
        mean_final_score=_mean_metric(reports, "final_score"),
    )


def _breakdowns(reports: Sequence[QualityQueryReport]) -> tuple[QualityBreakdown, ...]:
    values: list[QualityBreakdown] = []
    for difficulty in Difficulty:
        grouped = [report for report in reports if report.difficulty is difficulty]
        if grouped:
            values.append(
                QualityBreakdown(
                    "difficulty",
                    difficulty.value,
                    len(grouped),
                    _mean_metric(grouped, "final_score"),
                )
            )
    tags = sorted({tag for report in reports for tag in report.tags})
    for tag in tags:
        grouped = [report for report in reports if tag in report.tags]
        values.append(
            QualityBreakdown(
                "tag",
                tag,
                len(grouped),
                _mean_metric(grouped, "final_score"),
            )
        )
    return tuple(values)


def evaluate_quality_benchmark(
    benchmark: QualityBenchmark,
    prediction_queries: Sequence[RankedTop100Query],
    *,
    include_synthetic: bool = False,
) -> QualityEvaluationReport:
    """Evaluate exact eligible queries through the frozen P0-A evaluator."""

    if type(benchmark) is not QualityBenchmark:
        raise TypeError("benchmark must be QualityBenchmark")
    predictions = tuple(prediction_queries)
    if any(type(query) is not RankedTop100Query for query in predictions):
        raise TypeError("prediction_queries must contain RankedTop100Query values")
    prediction_ids = tuple(query.query_id for query in predictions)
    if len(set(prediction_ids)) != len(prediction_ids):
        raise ValueError("duplicate prediction query_id")

    scored, skipped_drafts, skipped_synthetic = _eligible_queries(
        benchmark,
        include_synthetic=include_synthetic,
    )
    expected_ids = tuple(query.query_id for query in scored)
    missing = sorted(set(expected_ids) - set(prediction_ids))
    unexpected = sorted(set(prediction_ids) - set(expected_ids))
    if missing or unexpected:
        raise ValueError(
            f"prediction query set mismatch: missing={missing}, unexpected={unexpected}"
        )
    prediction_by_id = {query.query_id: query for query in predictions}

    query_reports: list[QualityQueryReport] = []
    for query in scored:
        prediction_query = prediction_by_id[query.query_id]
        if prediction_query.task_type != query.task_type.value:
            raise ValueError(
                f"task mismatch for {query.query_id!r}: benchmark={query.task_type.value}, "
                f"prediction={prediction_query.task_type}"
            )
        if query.ground_truth is None:
            raise ValueError(f"scored query {query.query_id!r} has no ground truth")
        evaluation = evaluate_ranked_query(
            query.query_id,
            query.task_type.value,
            list(prediction_query.predictions),
            query.ground_truth,
            _scorer(query),
        )
        query_reports.append(
            QualityQueryReport(
                query.query_id,
                query.task_type,
                query.difficulty,
                query.tags,
                query.label_origin,
                evaluation,
            )
        )

    task_summaries = tuple(
        _task_summary(
            task_type,
            [report for report in query_reports if report.task_type is task_type],
        )
        for task_type in QualityTaskType
        if any(report.task_type is task_type for report in query_reports)
    )
    dataset_report = evaluate_dataset([report.evaluation for report in query_reports])
    task_macro = (
        fmean(summary.mean_final_score for summary in task_summaries)
        if task_summaries
        else 0.0
    )
    return QualityEvaluationReport(
        benchmark_id=benchmark.benchmark_id,
        include_synthetic=include_synthetic,
        scored_query_count=len(query_reports),
        skipped_draft_count=skipped_drafts,
        skipped_synthetic_count=skipped_synthetic,
        query_reports=tuple(query_reports),
        task_summaries=task_summaries,
        breakdowns=_breakdowns(query_reports),
        overall_query_macro_score=dataset_report.mean_query_final_score,
        task_macro_score=task_macro,
    )
