"""Public API for TRIAGE-EG M3."""

from .metrics import build_metrics, decide_m3, distance_to_intervals, evaluate_predictions_only
from .registry import (
    EXPECTED_AI_QC_SHA256,
    NEW_TRANSITIONS,
    build_case_registry,
    inference_case_from_registry,
    validate_trusted_registry_row,
)
from .runner import (
    M3Config,
    create_m3_bundle,
    formal_report_lines,
    preflight_m3,
    run_m3,
)
from .solver import (
    M3InferenceCase,
    M3Settings,
    TransitionSolution,
    adjacent_embedding_motion,
    build_state_signals,
    local_window,
    moving_median,
    solve_state_transition,
)

__all__ = [
    "EXPECTED_AI_QC_SHA256",
    "M3Config",
    "M3InferenceCase",
    "M3Settings",
    "NEW_TRANSITIONS",
    "TransitionSolution",
    "adjacent_embedding_motion",
    "build_case_registry",
    "build_metrics",
    "build_state_signals",
    "create_m3_bundle",
    "decide_m3",
    "distance_to_intervals",
    "evaluate_predictions_only",
    "formal_report_lines",
    "inference_case_from_registry",
    "local_window",
    "moving_median",
    "preflight_m3",
    "run_m3",
    "solve_state_transition",
    "validate_trusted_registry_row",
]
