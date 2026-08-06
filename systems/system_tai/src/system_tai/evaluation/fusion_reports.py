"""Deterministic JSON, CSV, and Markdown reports for Weighted RRF pilots."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from system_tai.evaluation.fusion_benchmark import FusionBenchmarkReport


@dataclass(frozen=True, slots=True)
class FusionReportPaths:
    json_path: Path
    csv_path: Path
    markdown_path: Path


def write_fusion_reports(
    report: FusionBenchmarkReport,
    output_directory: Path,
) -> FusionReportPaths:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    paths = FusionReportPaths(
        json_path=output / "kis_fusion_pilot_report.json",
        csv_path=output / "kis_fusion_pilot_summary.csv",
        markdown_path=output / "kis_fusion_pilot_report.md",
    )
    paths.json_path.write_text(
        json.dumps(asdict(report), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(report, paths.csv_path)
    paths.markdown_path.write_text(_markdown(report), encoding="utf-8")
    return paths


def _write_csv(report: FusionBenchmarkReport, path: Path) -> None:
    fields = [
        "semantic_group_id",
        "contributing_variant_ids",
        "contributing_variant_count",
        "relevant_label_count",
        "first_relevant_rank",
        "reciprocal_rank",
        *[f"recall_at_{cutoff}" for cutoff in report.top_ks],
        *[f"ground_truth_coverage_at_{cutoff}" for cutoff in report.top_ks],
        *[f"hit_count_at_{cutoff}" for cutoff in report.top_ks],
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for metric in sorted(report.group_metrics, key=lambda item: item.semantic_group_id):
            row: dict[str, object] = {
                "semantic_group_id": metric.semantic_group_id,
                "contributing_variant_ids": "|".join(metric.contributing_variant_ids),
                "contributing_variant_count": metric.contributing_variant_count,
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
            writer.writerow(row)


def _markdown(report: FusionBenchmarkReport) -> str:
    lines = [
        "# KIS Weighted RRF Pilot Report",
        "",
        f"- State: `{report.evaluation_state}`",
        f"- Benchmark: `{report.benchmark_id}`",
        f"- Comparable groups: {report.evaluated_group_count}",
        f"- Verified variants evaluated: {report.evaluated_verified_query_count}",
        f"- Excluded drafts: {report.excluded_draft_query_count}",
        f"- Source scope: {', '.join(report.source_video_scope)}",
        f"- Per-variant retrieval: `{report.retrieval_implementation}`",
        f"- Fusion: `{report.fusion_method}`, k={report.rrf_constant}",
        "- Canonical per-variant output unsuppressed: "
        f"`{str(report.canonical_per_variant_unsuppressed).lower()}`",
        "",
        "## Group metrics",
        "",
        "| Group | Variants | Count | First rank | RR | Recall@K | GT coverage@K |",
        "|---|---|---:|---:|---:|---|---|",
    ]
    for metric in report.group_metrics:
        first_rank = "-" if metric.first_relevant_rank is None else str(metric.first_relevant_rank)
        recall = ", ".join(
            f"R@{cutoff}={value:.4f}" for cutoff, value in metric.recall_at_k
        )
        coverage = ", ".join(
            f"C@{cutoff}={value:.4f}"
            for cutoff, value in metric.ground_truth_coverage_at_k
        )
        lines.append(
            f"| {metric.semantic_group_id} | {', '.join(metric.contributing_variant_ids)} | "
            f"{metric.contributing_variant_count} | {first_rank} | "
            f"{metric.reciprocal_rank:.6f} | {recall} | {coverage} |"
        )
    lines.extend(["", "## Metric definitions", ""])
    for name, definition in sorted(report.metric_definitions.items()):
        lines.append(f"- **{name}:** {definition}")
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {limitation}" for limitation in report.limitations)
    return "\n".join(lines) + "\n"
