"""Public API for T3 coverage-aware diverse temporal hypotheses."""

from .hypotheses import (
    FINAL_PATH_LIMIT,
    MAX_RAW_COMBINATIONS,
    POOL_LIMIT,
    REGION_RADIUS_SECONDS,
    DiverseTemporalPath,
    EventCandidate,
    build_diverse_event_pool,
    enumerate_feasible_paths,
    event_region_novelty,
    relative_score_gap,
    select_coverage_aware,
    select_score_top_k,
)
from .runner import (
    DELTA_GRID,
    T3_VERSION,
    T3RunnerConfig,
    create_t3_bundle,
    preflight_t3,
    run_t3,
    validate_a0_reproduction,
)

__all__ = [
    "DELTA_GRID",
    "DiverseTemporalPath",
    "EventCandidate",
    "FINAL_PATH_LIMIT",
    "MAX_RAW_COMBINATIONS",
    "POOL_LIMIT",
    "REGION_RADIUS_SECONDS",
    "T3RunnerConfig",
    "T3_VERSION",
    "build_diverse_event_pool",
    "create_t3_bundle",
    "enumerate_feasible_paths",
    "event_region_novelty",
    "preflight_t3",
    "relative_score_gap",
    "run_t3",
    "select_coverage_aware",
    "select_score_top_k",
    "validate_a0_reproduction",
]
