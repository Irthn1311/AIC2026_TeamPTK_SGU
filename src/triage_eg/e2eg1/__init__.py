"""TRIAGE-EG E2E-G1 safe hypothesis preservation experiment."""

from .contracts import VARIANTS, E2EG1Settings
from .pipeline import SafeCoveragePipeline, filter_machine_ids, is_opaque_machine_id
from .ranking import coverage_order, g0_order, rank_video_hypotheses, safe_alternative_order
from .runner import (
    combine_prediction_variants,
    compare_variants,
    create_bundle,
    decisions,
    evaluate_finalized,
    extract_development_bundle,
    formal_report_lines,
    materialize_inference_only,
    post_inference_diagnostics,
    render_cross_review,
    run_prediction_variant,
    runtime_summary,
    write_manifests,
)

__all__ = [
    "E2EG1Settings",
    "SafeCoveragePipeline",
    "VARIANTS",
    "combine_prediction_variants",
    "compare_variants",
    "coverage_order",
    "create_bundle",
    "decisions",
    "evaluate_finalized",
    "extract_development_bundle",
    "filter_machine_ids",
    "formal_report_lines",
    "g0_order",
    "is_opaque_machine_id",
    "materialize_inference_only",
    "post_inference_diagnostics",
    "rank_video_hypotheses",
    "render_cross_review",
    "run_prediction_variant",
    "runtime_summary",
    "safe_alternative_order",
    "write_manifests",
]
