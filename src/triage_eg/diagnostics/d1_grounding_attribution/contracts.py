"""Frozen contracts for D1 grounding and candidate error attribution."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from triage_eg.e2eg1.contracts import (
    COVERAGE_REGIONS_PER_VIDEO,
    COVERAGE_VIDEO_LIMIT,
    PROTECTED_GLOBAL_PREFIX,
)
from triage_eg.experiments.t3_diverse_temporal import POOL_LIMIT, REGION_RADIUS_SECONDS

PRIMARY_BENCHMARK = "DEV_CROSS_60"
SELECTED_GROUNDING_POLICY = "G1_COVERAGE_COARSE"
MAX_REVIEW_CASES = 18
SINGLE_EVENT_REASONS = (
    "SUCCESS_G1_TARGET_HIT",
    "BTC_REPRESENTATION_GAP",
    "TARGET_SEMANTIC_SCORE_WEAK",
    "T3_REGION_REPRESENTATIVE_GAP",
    "GLOBAL_VIDEO_RANKING_GAP",
    "G1_ALLOCATION_GAP",
    "UNCLASSIFIED_SINGLE_EVENT",
)
TRAKE_REASONS = (
    "SUCCESS_FULL_CHAIN",
    "BTC_EVENT_REPRESENTATION_GAP",
    "EVENT_SEMANTIC_SCORE_GAP",
    "T3_EVENT_POOL_GAP",
    "MONOTONIC_COMPOSITION_GAP",
    "GLOBAL_CHAIN_RANKING_GAP",
    "UNCLASSIFIED_TRAKE",
)
FORBIDDEN_BLIND_QC_FIELDS = frozenset(
    {
        "correct_video",
        "acceptable_intervals",
        "accepted_intervals",
        "event_intervals",
        "retrieval_rank",
        "retrieval_score",
        "success",
        "failure",
        "difficulty",
        "accepted_answers",
        "ground_truth",
        "gt",
    }
)


@dataclass(frozen=True)
class D1Settings:
    primary_benchmark: str = PRIMARY_BENCHMARK
    use_selected_g1: bool = True
    t3_pool_limit: int = POOL_LIMIT
    t3_region_radius_seconds: float = REGION_RADIUS_SECONDS
    g1_protected_prefix: int = PROTECTED_GLOBAL_PREFIX
    g1_coverage_video_limit: int = COVERAGE_VIDEO_LIMIT
    g1_coverage_regions: int = COVERAGE_REGIONS_PER_VIDEO
    max_review_cases: int = MAX_REVIEW_CASES
    run_m1: bool = False
    use_m2: bool = False
    use_m3: bool = False
    use_graph: bool = False
    use_vlm: bool = False
    use_agent: bool = False
    parameter_sweep: bool = False

    def __post_init__(self) -> None:
        frozen = (
            self.primary_benchmark == PRIMARY_BENCHMARK
            and self.use_selected_g1
            and self.t3_pool_limit == POOL_LIMIT == 10
            and self.t3_region_radius_seconds == REGION_RADIUS_SECONDS == 3.0
            and self.g1_protected_prefix == PROTECTED_GLOBAL_PREFIX == 5
            and self.g1_coverage_video_limit == COVERAGE_VIDEO_LIMIT == 5
            and self.g1_coverage_regions == COVERAGE_REGIONS_PER_VIDEO == 10
            and self.max_review_cases == MAX_REVIEW_CASES
        )
        disabled = not any(
            (
                self.run_m1,
                self.use_m2,
                self.use_m3,
                self.use_graph,
                self.use_vlm,
                self.use_agent,
                self.parameter_sweep,
            )
        )
        if not frozen or not disabled:
            raise ValueError("D1 settings are frozen and diagnostic-only")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SemanticUnitSnapshot:
    unit_id: str
    query_id: str
    task: str
    event_id: str | None
    source_language: str
    source_text: str
    embedding: np.ndarray
    scores: np.ndarray
    encoding: dict[str, Any]


@dataclass(frozen=True)
class InferenceSnapshot:
    prediction_sha256: str
    units: dict[str, SemanticUnitSnapshot]
    unit_ids_by_query: dict[str, tuple[str, ...]]
    single_event_pools: dict[str, tuple[dict[str, Any], ...]]
    g1_allocations: dict[str, tuple[dict[str, Any], ...]]
    trake_chains: dict[str, tuple[dict[str, Any], ...]]


__all__ = [
    "D1Settings",
    "FORBIDDEN_BLIND_QC_FIELDS",
    "InferenceSnapshot",
    "MAX_REVIEW_CASES",
    "PRIMARY_BENCHMARK",
    "SELECTED_GROUNDING_POLICY",
    "SINGLE_EVENT_REASONS",
    "SemanticUnitSnapshot",
    "TRAKE_REASONS",
]
