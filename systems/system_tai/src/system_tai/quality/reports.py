"""Deterministic UTF-8 serializers for quality evaluation and comparison reports."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from system_tai.preliminary.evaluation import OFFICIAL_K

from .comparison import QualityComparisonReport
from .evaluator import QualityEvaluationReport, QualityQueryReport

TAG_DELIMITER = "|"


def _query_payload(report: QualityQueryReport) -> dict[str, object]:
    evaluation = report.evaluation
    return {
        "query_id": report.query_id,
        "task_type": report.task_type.value,
        "difficulty": report.difficulty.value,
        "tags": list(report.tags),
        "label_origin": report.label_origin.value,
        "prediction_count": evaluation.prediction_count,
        "per_rank_r_scores": list(evaluation.per_rank_r_scores),
        **{f"r_at_{k}": getattr(evaluation, f"r_at_{k}") for k in OFFICIAL_K},
        "final_score": evaluation.final_score,
    }


def _quality_payload(report: QualityEvaluationReport) -> dict[str, object]:
    return {
        "benchmark_id": report.benchmark_id,
        "include_synthetic": report.include_synthetic,
        "scored_query_count": report.scored_query_count,
        "skipped_draft_count": report.skipped_draft_count,
        "skipped_synthetic_count": report.skipped_synthetic_count,
        "overall_query_macro_score": report.overall_query_macro_score,
        "task_macro_score": report.task_macro_score,
        "query_reports": [_query_payload(query) for query in report.query_reports],
        "task_summaries": [
            {
                "task_type": summary.task_type.value,
                "query_count": summary.query_count,
                **{
                    f"mean_r_at_{k}": getattr(summary, f"mean_r_at_{k}")
                    for k in OFFICIAL_K
                },
                "mean_final_score": summary.mean_final_score,
            }
            for summary in report.task_summaries
        ],
        "breakdowns": [
            {
                "dimension": breakdown.dimension,
                "value": breakdown.value,
                "query_count": breakdown.query_count,
                "mean_final_score": breakdown.mean_final_score,
            }
            for breakdown in report.breakdowns
        ],
    }


def _comparison_payload(report: QualityComparisonReport) -> dict[str, object]:
    return {
        "benchmark_id": report.benchmark_id,
        "baseline_label": report.baseline_label,
        "candidate_label": report.candidate_label,
        "query_count": report.query_count,
        "baseline_overall_query_macro_score": (
            report.baseline_overall_query_macro_score
        ),
        "candidate_overall_query_macro_score": (
            report.candidate_overall_query_macro_score
        ),
        "overall_delta": report.overall_delta,
        "baseline_task_macro_score": report.baseline_task_macro_score,
        "candidate_task_macro_score": report.candidate_task_macro_score,
        "task_macro_delta": report.task_macro_delta,
        "improved_count": report.improved_count,
        "tied_count": report.tied_count,
        "regressed_count": report.regressed_count,
        "task_deltas": [
            {
                "task_type": delta.task_type.value,
                "baseline_mean_final_score": delta.baseline_mean_final_score,
                "candidate_mean_final_score": delta.candidate_mean_final_score,
                "delta": delta.delta,
            }
            for delta in report.task_deltas
        ],
        "query_deltas": [
            {
                "query_id": delta.query_id,
                "task_type": delta.task_type.value,
                "baseline_final_score": delta.baseline_final_score,
                "candidate_final_score": delta.candidate_final_score,
                "delta": delta.delta,
                "classification": delta.classification.value,
            }
            for delta in report.query_deltas
        ],
    }


def _write_json(payload: dict[str, object], destination: Path) -> Path:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def write_quality_report_json(
    report: QualityEvaluationReport,
    destination: Path,
) -> Path:
    return _write_json(_quality_payload(report), destination)


def write_quality_report_csv(
    report: QualityEvaluationReport,
    destination: Path,
) -> Path:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "benchmark_id",
        "query_id",
        "task_type",
        "difficulty",
        "tags",
        "label_origin",
        "prediction_count",
        *(f"r_at_{k}" for k in OFFICIAL_K),
        "final_score",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for query in report.query_reports:
            evaluation = query.evaluation
            writer.writerow(
                {
                    "benchmark_id": report.benchmark_id,
                    "query_id": query.query_id,
                    "task_type": query.task_type.value,
                    "difficulty": query.difficulty.value,
                    "tags": TAG_DELIMITER.join(query.tags),
                    "label_origin": query.label_origin.value,
                    "prediction_count": evaluation.prediction_count,
                    **{
                        f"r_at_{k}": getattr(evaluation, f"r_at_{k}")
                        for k in OFFICIAL_K
                    },
                    "final_score": evaluation.final_score,
                }
            )
    return path


def write_quality_comparison_json(
    report: QualityComparisonReport,
    destination: Path,
) -> Path:
    return _write_json(_comparison_payload(report), destination)
