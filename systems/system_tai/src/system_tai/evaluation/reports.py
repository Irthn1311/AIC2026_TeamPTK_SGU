"""Deterministic JSON, CSV, and Markdown benchmark reports."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from system_tai.evaluation.kis_benchmark import KISBenchmarkReport


@dataclass(frozen=True, slots=True)
class BenchmarkReportPaths:
    json_path: Path
    csv_path: Path
    markdown_path: Path


def write_benchmark_reports(
    report: KISBenchmarkReport,
    output_directory: Path,
) -> BenchmarkReportPaths:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    paths = BenchmarkReportPaths(
        json_path=output / "kis_benchmark_report.json",
        csv_path=output / "kis_benchmark_summary.csv",
        markdown_path=output / "kis_benchmark_report.md",
    )
    paths.json_path.write_text(
        json.dumps(asdict(report), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(report, paths.csv_path)
    paths.markdown_path.write_text(_markdown(report), encoding="utf-8")
    return paths


def _write_csv(report: KISBenchmarkReport, path: Path) -> None:
    recall_fields = [f"recall_at_{cutoff}" for cutoff in report.top_ks]
    coverage_fields = [
        f"ground_truth_coverage_at_{cutoff}" for cutoff in report.top_ks
    ]
    hit_fields = [f"hit_count_at_{cutoff}" for cutoff in report.top_ks]
    video_fields = [
        f"relevant_video_coverage_at_{cutoff}" for cutoff in report.top_ks
    ]
    fields = [
        "query_id",
        "language",
        "variant_type",
        "semantic_group_id",
        "relevant_label_count",
        "first_relevant_rank",
        "reciprocal_rank",
        *recall_fields,
        *coverage_fields,
        *hit_fields,
        *video_fields,
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for metric in sorted(report.query_metrics, key=lambda item: item.query_id):
            row: dict[str, object] = {
                "query_id": metric.query_id,
                "language": metric.language,
                "variant_type": metric.variant_type,
                "semantic_group_id": metric.semantic_group_id,
                "relevant_label_count": metric.relevant_label_count,
                "first_relevant_rank": metric.first_relevant_rank,
                "reciprocal_rank": metric.reciprocal_rank,
            }
            row.update(
                {f"recall_at_{cutoff}": value for cutoff, value in metric.recall_at_k}
            )
            row.update(
                {
                    f"ground_truth_coverage_at_{cutoff}": value
                    for cutoff, value in metric.ground_truth_coverage_at_k
                }
            )
            row.update(
                {
                    f"hit_count_at_{cutoff}": value
                    for cutoff, value in metric.hit_count_at_k
                }
            )
            if metric.relevant_video_coverage_at_k is not None:
                row.update(
                    {
                        f"relevant_video_coverage_at_{cutoff}": value
                        for cutoff, value in metric.relevant_video_coverage_at_k
                    }
                )
            writer.writerow(row)


def _markdown(report: KISBenchmarkReport) -> str:
    lines = [
        "# KIS Benchmark Report",
        "",
        f"- Evaluation state: `{report.evaluation_state}`",
        f"- Benchmark: `{report.benchmark_id}` (schema {report.schema_version})",
        f"- Evaluated verified queries: {report.evaluated_query_count}",
        f"- Excluded draft queries: {report.excluded_draft_query_count}",
        f"- Validation-invalid query count: {report.invalid_query_count}",
        f"- Source scope: {', '.join(report.source_video_scope)}",
        f"- Model: `{report.model_identifier}`",
        f"- Device: `{report.device}`",
        f"- Retrieval: `{report.retrieval_implementation}`",
        f"- Canonical unsuppressed: `{str(report.canonical_unsuppressed).lower()}`",
        "",
        "## Aggregate metrics",
        "",
        "| Group | Queries | MRR | Mean first rank | Missing hits | Recall@K | "
        "GT coverage@K |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for aggregate in report.aggregates:
        recall = ", ".join(
            f"R@{cutoff}={value:.4f}"
            for cutoff, value in aggregate.mean_recall_at_k
        )
        coverage = ", ".join(
            f"C@{cutoff}={value:.4f}"
            for cutoff, value in aggregate.mean_ground_truth_coverage_at_k
        )
        first_rank = (
            "-"
            if aggregate.mean_first_relevant_rank is None
            else f"{aggregate.mean_first_relevant_rank:.4f}"
        )
        lines.append(
            f"| {aggregate.group_type}:{aggregate.group_value} | "
            f"{aggregate.query_count} | {aggregate.mean_reciprocal_rank:.4f} | "
            f"{first_rank} | {aggregate.queries_without_relevant_hit} | {recall} | "
            f"{coverage} |"
        )
    lines.extend(
        [
            "",
            "## Paired comparisons",
            "",
            "| Semantic group | English variant | Status | Vietnamese | English | "
            "First-rank delta (EN-VI) | First-rank outcome | Recall outcomes |",
            "|---|---|---|---|---|---:|---|---|",
        ]
    )
    for pair in report.paired_comparisons:
        delta = (
            "-"
            if pair.first_relevant_rank_delta_english_minus_vietnamese is None
            else str(pair.first_relevant_rank_delta_english_minus_vietnamese)
        )
        recall_outcomes = ", ".join(
            f"R@{cutoff}:{outcome}"
            for cutoff, outcome in pair.recall_outcome_at_k
        )
        lines.append(
            f"| {pair.semantic_group_id} | {pair.comparison_variant_type} | "
            f"{pair.status} | {pair.vietnamese_query_id or '-'} | "
            f"{pair.comparison_query_id or '-'} | {delta} | "
            f"{pair.first_relevant_rank_outcome or '-'} | "
            f"{recall_outcomes or '-'} |"
        )
    lines.extend(["", "## Metric definitions", ""])
    for name, definition in sorted(report.metric_definitions.items()):
        lines.append(f"- **{name}:** {definition}")
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {limitation}" for limitation in report.limitations)
    return "\n".join(lines) + "\n"
