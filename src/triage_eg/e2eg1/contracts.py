"""Frozen contracts for E2E-G1 safe temporal hypothesis allocation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from triage_eg.experiments.t3_diverse_temporal import POOL_LIMIT, REGION_RADIUS_SECONDS

MAX_PREDICTIONS = 100
PROTECTED_GLOBAL_PREFIX = 5
COVERAGE_VIDEO_LIMIT = 5
COVERAGE_REGIONS_PER_VIDEO = POOL_LIMIT
M1_SINGLE_EVENT_BUDGET = 10
M1_TRAKE_SOURCE_CHAINS = 5
T3_SELECTED_DELTA = 0.05
OCR_MAX_GROUNDING_RANKS = 20
VARIANTS = ("G0_E2E1_COARSE", "G1_COVERAGE_COARSE", "G2_SAFE_M1")
VARIANT_SLUGS = {
    "G0_E2E1_COARSE": "g0",
    "G1_COVERAGE_COARSE": "g1",
    "G2_SAFE_M1": "g2",
}


@dataclass(frozen=True)
class E2EG1Settings:
    max_predictions: int = MAX_PREDICTIONS
    protected_global_prefix: int = PROTECTED_GLOBAL_PREFIX
    coverage_video_limit: int = COVERAGE_VIDEO_LIMIT
    coverage_regions_per_video: int = COVERAGE_REGIONS_PER_VIDEO
    m1_single_event_budget: int = M1_SINGLE_EVENT_BUDGET
    m1_trake_source_chains: int = M1_TRAKE_SOURCE_CHAINS
    t3_selected_delta: float = T3_SELECTED_DELTA
    ocr_max_grounding_ranks: int = OCR_MAX_GROUNDING_RANKS
    use_m2: bool = False
    use_m3: bool = False
    use_event_graph: bool = False
    use_vlm: bool = False
    use_agent: bool = False
    use_nvdec_default: bool = False

    def __post_init__(self) -> None:
        expected = (
            self.max_predictions == MAX_PREDICTIONS
            and self.protected_global_prefix == PROTECTED_GLOBAL_PREFIX
            and self.coverage_video_limit == COVERAGE_VIDEO_LIMIT
            and self.coverage_regions_per_video == POOL_LIMIT == 10
            and self.m1_single_event_budget == M1_SINGLE_EVENT_BUDGET
            and self.m1_trake_source_chains == M1_TRAKE_SOURCE_CHAINS
            and self.t3_selected_delta == T3_SELECTED_DELTA
            and self.ocr_max_grounding_ranks == OCR_MAX_GROUNDING_RANKS
            and REGION_RADIUS_SECONDS == 3.0
        )
        disabled = not any(
            (
                self.use_m2,
                self.use_m3,
                self.use_event_graph,
                self.use_vlm,
                self.use_agent,
                self.use_nvdec_default,
            )
        )
        if not expected or not disabled:
            raise ValueError("E2E-G1 settings are frozen and cannot be tuned")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


__all__ = [
    "COVERAGE_REGIONS_PER_VIDEO",
    "COVERAGE_VIDEO_LIMIT",
    "E2EG1Settings",
    "M1_SINGLE_EVENT_BUDGET",
    "M1_TRAKE_SOURCE_CHAINS",
    "MAX_PREDICTIONS",
    "PROTECTED_GLOBAL_PREFIX",
    "T3_SELECTED_DELTA",
    "VARIANTS",
    "VARIANT_SLUGS",
]
