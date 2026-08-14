"""TRIAGE-EG E2E-1 canonical integration baseline."""

from .contracts import E2E1Settings, QueryPlan
from .pipeline import CanonicalTriagePipeline, PredictionResult
from .planning import plan_queries, plan_query
from .qa import OptionalTesseract, numeric_tokens, route_intent
from .runner import (
    combine_prediction_variants,
    compare_variants,
    create_e2e1_bundle,
    evaluate_finalized,
    extract_development_bundle,
    failure_taxonomy,
    formal_report_lines,
    materialize_inference_only,
    render_cross_review,
    run_prediction_variant,
    run_predictions_only,
    runtime_summary,
    write_manifests,
)

__all__ = [
    "CanonicalTriagePipeline",
    "E2E1Settings",
    "OptionalTesseract",
    "PredictionResult",
    "QueryPlan",
    "numeric_tokens",
    "plan_queries",
    "plan_query",
    "route_intent",
    "create_e2e1_bundle",
    "compare_variants",
    "combine_prediction_variants",
    "evaluate_finalized",
    "extract_development_bundle",
    "failure_taxonomy",
    "formal_report_lines",
    "materialize_inference_only",
    "render_cross_review",
    "run_prediction_variant",
    "run_predictions_only",
    "runtime_summary",
    "write_manifests",
]
