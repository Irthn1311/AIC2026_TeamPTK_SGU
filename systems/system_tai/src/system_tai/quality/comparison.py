"""Deterministic query-by-query experiment comparison for quality reports."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .evaluator import QualityEvaluationReport
from .schema import QualityTaskType


class DeltaClassification(StrEnum):
    IMPROVED = "IMPROVED"
    TIED = "TIED"
    REGRESSED = "REGRESSED"


@dataclass(frozen=True, slots=True)
class QualityQueryDelta:
    query_id: str
    task_type: QualityTaskType
    baseline_final_score: float
    candidate_final_score: float
    delta: float
    classification: DeltaClassification


@dataclass(frozen=True, slots=True)
class QualityTaskDelta:
    task_type: QualityTaskType
    baseline_mean_final_score: float
    candidate_mean_final_score: float
    delta: float


@dataclass(frozen=True, slots=True)
class QualityComparisonReport:
    benchmark_id: str
    baseline_label: str
    candidate_label: str
    query_count: int
    baseline_overall_query_macro_score: float
    candidate_overall_query_macro_score: float
    overall_delta: float
    baseline_task_macro_score: float
    candidate_task_macro_score: float
    task_macro_delta: float
    improved_count: int
    tied_count: int
    regressed_count: int
    task_deltas: tuple[QualityTaskDelta, ...]
    query_deltas: tuple[QualityQueryDelta, ...]


def _label(value: str, name: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _classification(delta: float, tolerance: float) -> DeltaClassification:
    if delta > tolerance:
        return DeltaClassification.IMPROVED
    if delta < -tolerance:
        return DeltaClassification.REGRESSED
    return DeltaClassification.TIED


def compare_quality_reports(
    baseline: QualityEvaluationReport,
    candidate: QualityEvaluationReport,
    *,
    baseline_label: str = "baseline",
    candidate_label: str = "candidate",
    tolerance: float = 1e-12,
) -> QualityComparisonReport:
    """Compare aligned reports without making an experiment acceptance decision."""

    if type(baseline) is not QualityEvaluationReport:
        raise TypeError("baseline must be QualityEvaluationReport")
    if type(candidate) is not QualityEvaluationReport:
        raise TypeError("candidate must be QualityEvaluationReport")
    baseline_label = _label(baseline_label, "baseline_label")
    candidate_label = _label(candidate_label, "candidate_label")
    if type(tolerance) not in (int, float) or tolerance < 0:
        raise ValueError("tolerance must be a non-negative number")
    if baseline.benchmark_id != candidate.benchmark_id:
        raise ValueError("benchmark_id mismatch")

    base_by_id = {report.query_id: report for report in baseline.query_reports}
    cand_by_id = {report.query_id: report for report in candidate.query_reports}
    if len(base_by_id) != len(baseline.query_reports):
        raise ValueError("baseline contains duplicate query IDs")
    if len(cand_by_id) != len(candidate.query_reports):
        raise ValueError("candidate contains duplicate query IDs")
    if set(base_by_id) != set(cand_by_id):
        raise ValueError("scored query ID set mismatch")

    query_deltas: list[QualityQueryDelta] = []
    for baseline_query in baseline.query_reports:
        candidate_query = cand_by_id[baseline_query.query_id]
        if baseline_query.task_type is not candidate_query.task_type:
            raise ValueError(f"task mismatch for query {baseline_query.query_id!r}")
        base_score = baseline_query.evaluation.final_score
        candidate_score = candidate_query.evaluation.final_score
        delta = candidate_score - base_score
        query_deltas.append(
            QualityQueryDelta(
                baseline_query.query_id,
                baseline_query.task_type,
                base_score,
                candidate_score,
                delta,
                _classification(delta, float(tolerance)),
            )
        )

    base_tasks = {summary.task_type: summary for summary in baseline.task_summaries}
    cand_tasks = {summary.task_type: summary for summary in candidate.task_summaries}
    if set(base_tasks) != set(cand_tasks):
        raise ValueError("task summary set mismatch")
    task_deltas = tuple(
        QualityTaskDelta(
            task,
            base_tasks[task].mean_final_score,
            cand_tasks[task].mean_final_score,
            cand_tasks[task].mean_final_score - base_tasks[task].mean_final_score,
        )
        for task in QualityTaskType
        if task in base_tasks
    )
    classifications = [delta.classification for delta in query_deltas]
    return QualityComparisonReport(
        benchmark_id=baseline.benchmark_id,
        baseline_label=baseline_label,
        candidate_label=candidate_label,
        query_count=len(query_deltas),
        baseline_overall_query_macro_score=baseline.overall_query_macro_score,
        candidate_overall_query_macro_score=candidate.overall_query_macro_score,
        overall_delta=(
            candidate.overall_query_macro_score - baseline.overall_query_macro_score
        ),
        baseline_task_macro_score=baseline.task_macro_score,
        candidate_task_macro_score=candidate.task_macro_score,
        task_macro_delta=candidate.task_macro_score - baseline.task_macro_score,
        improved_count=classifications.count(DeltaClassification.IMPROVED),
        tied_count=classifications.count(DeltaClassification.TIED),
        regressed_count=classifications.count(DeltaClassification.REGRESSED),
        task_deltas=task_deltas,
        query_deltas=tuple(query_deltas),
    )
