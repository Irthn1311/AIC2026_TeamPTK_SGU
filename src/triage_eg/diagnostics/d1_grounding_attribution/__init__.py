"""TRIAGE-EG D1 grounding and candidate error attribution audit."""

from .contracts import D1Settings, InferenceSnapshot, SemanticUnitSnapshot
from .runner import (
    EXPECTED_E2EG1_CROSS_G1_SHA256,
    EXPECTED_E2EG1_SHA256,
    capture_inference_snapshot,
    create_bundle,
    formal_report_lines,
    run_g1_reproduction,
    run_post_gt_attribution,
    verify_historical_reproduction,
    write_blind_translation_artifacts,
    write_manifests,
)
from .single_event import (
    audit_event_unit,
    audit_single_event,
    classify_single_event,
    frame_distance_to_intervals,
    summarize_single_events,
)
from .trake import audit_trake_query, classify_trake, strict_target_chain_exists, summarize_trake
from .translation import (
    blind_translation_rows,
    translation_provenance_rows,
    translation_review_instructions,
    translation_surface_checks,
    translation_surface_summary,
)
from .visuals import render_review_sheets, select_review_cases

__all__ = [
    "D1Settings",
    "EXPECTED_E2EG1_CROSS_G1_SHA256",
    "EXPECTED_E2EG1_SHA256",
    "InferenceSnapshot",
    "SemanticUnitSnapshot",
    "audit_event_unit",
    "audit_single_event",
    "audit_trake_query",
    "blind_translation_rows",
    "capture_inference_snapshot",
    "classify_single_event",
    "classify_trake",
    "create_bundle",
    "formal_report_lines",
    "frame_distance_to_intervals",
    "render_review_sheets",
    "run_g1_reproduction",
    "run_post_gt_attribution",
    "select_review_cases",
    "strict_target_chain_exists",
    "summarize_single_events",
    "summarize_trake",
    "translation_review_instructions",
    "translation_provenance_rows",
    "translation_surface_checks",
    "translation_surface_summary",
    "verify_historical_reproduction",
    "write_blind_translation_artifacts",
    "write_manifests",
]
