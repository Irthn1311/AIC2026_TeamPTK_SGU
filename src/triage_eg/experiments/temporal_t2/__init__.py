"""Public API for T2 k-best monotonic temporal hypotheses."""

from .metrics import (
    build_t2_metrics,
    event_is_reachable,
    query_all_events_reachable,
    validate_recall_monotonicity,
)
from .runner import (
    K_VALUES,
    T2_METHOD,
    T2_VERSION,
    TOLERANCE_SECONDS,
    T2RunnerConfig,
    T2Settings,
    create_t2_bundle,
    evaluate_source_paths,
    preflight_t2,
    run_t2,
)
from .solver import MAX_BEAM_WIDTH, TemporalPath, k_best_monotonic_paths

__all__ = [
    "K_VALUES",
    "MAX_BEAM_WIDTH",
    "T2RunnerConfig",
    "T2Settings",
    "T2_METHOD",
    "T2_VERSION",
    "TOLERANCE_SECONDS",
    "TemporalPath",
    "build_t2_metrics",
    "create_t2_bundle",
    "evaluate_source_paths",
    "event_is_reachable",
    "k_best_monotonic_paths",
    "preflight_t2",
    "query_all_events_reachable",
    "run_t2",
    "validate_recall_monotonicity",
]
