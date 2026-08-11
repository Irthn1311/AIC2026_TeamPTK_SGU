"""Public API for the bounded T2-D ceiling diagnostic."""

from .diagnostics import (
    EVENT_ONLY_CUTOFFS,
    K_VALUES,
    TOLERANCE_SECONDS,
    build_t2d_metrics,
    diagnose_source_query,
    masked_monotonic_dp,
    reference_neighborhood_mask,
    stable_event_ranking,
    validate_expected_t2_reproduction,
)
from .runner import (
    EXPECTED_BENCHMARK_SHA256,
    EXPECTED_STAGE1_FINGERPRINT,
    T2D_VERSION,
    T2DRunnerConfig,
    create_t2d_bundle,
    preflight_t2d,
    run_t2d,
)

__all__ = [
    "EVENT_ONLY_CUTOFFS",
    "EXPECTED_BENCHMARK_SHA256",
    "EXPECTED_STAGE1_FINGERPRINT",
    "K_VALUES",
    "T2DRunnerConfig",
    "T2D_VERSION",
    "TOLERANCE_SECONDS",
    "build_t2d_metrics",
    "create_t2d_bundle",
    "diagnose_source_query",
    "masked_monotonic_dp",
    "preflight_t2d",
    "reference_neighborhood_mask",
    "run_t2d",
    "stable_event_ranking",
    "validate_expected_t2_reproduction",
]
