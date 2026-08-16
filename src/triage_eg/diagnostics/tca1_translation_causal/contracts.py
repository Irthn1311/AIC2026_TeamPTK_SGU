"""Frozen contracts for TCA-1 translation causal ablation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

FROZEN_REVIEW_VERSION = "AI_BLIND_TRANSLATION_REVIEW_V1_2026-08-16"
FROZEN_REVIEW_PROTOCOL = "BLIND_SEMANTIC_FIDELITY_NO_GT_NO_RANK_NO_OUTCOME"
FROZEN_REVIEW_SHA256 = "6ab80bd34e84c58135ede83fdd6d5a0252fff872c640dc1248e72cf203862013"
FROZEN_OVERRIDE_SHA256 = "a2bad0e1dfc5397521ba09cdddf99aaf71751079e66f9ce3f4deacae01bd1105"
FROZEN_SOURCE_QC_SHA256 = "e0fa9d53d46ed2eff4f814045d74f186b024c4aa0e06d4d456293da07e76d730"
FROZEN_ZIP_SHA256 = "f2af3a682024d8987b0999ce5a79a9d344580512fe4589cb4d87c3f07f6e5ad3"
EXPECTED_A0_PREDICTION_SHA256 = "8a774e25aae0d4e23eafa905e468b25baeabc0b2ed74ba16491a1138b099ef9e"
EXPECTED_COUNTS = {"PASS": 49, "CONDITIONAL": 34, "FAIL": 17}
EXPECTED_BY_TASK = {
    "KIS": {"PASS": 10, "CONDITIONAL": 9, "FAIL": 1},
    "QA": {"PASS": 10, "CONDITIONAL": 8, "FAIL": 2},
    "TRAKE": {"PASS": 29, "CONDITIONAL": 17, "FAIL": 14},
}
EXPECTED_UNIT_COUNT = 100
EXPECTED_FAIL_COUNT = 17
PRIMARY_BENCHMARK = "DEV_CROSS_60"
PRIMARY_VARIANT = "G1_COVERAGE_COARSE"

FORBIDDEN_REVIEW_FIELDS = frozenset(
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
class TCA1Settings:
    """Predeclared diagnostic-only TCA-1 protocol."""

    benchmark: str = PRIMARY_BENCHMARK
    baseline_arm: str = "A0_CANONICAL_OPUS"
    intervention_arm: str = "A1_FAIL_REFERENCE_EN"
    selected_variant: str = PRIMARY_VARIANT
    expected_unit_count: int = EXPECTED_UNIT_COUNT
    expected_fail_count: int = EXPECTED_FAIL_COUNT
    negative_control_atol: float = 1e-6
    negative_control_rtol: float = 0.0
    run_m1: bool = False
    use_m2: bool = False
    use_m3: bool = False
    use_graph: bool = False
    use_vlm: bool = False
    use_agent: bool = False
    parameter_sweep: bool = False
    production_policy_changed: bool = False

    def __post_init__(self) -> None:
        frozen = (
            self.benchmark == PRIMARY_BENCHMARK
            and self.baseline_arm == "A0_CANONICAL_OPUS"
            and self.intervention_arm == "A1_FAIL_REFERENCE_EN"
            and self.selected_variant == PRIMARY_VARIANT
            and self.expected_unit_count == EXPECTED_UNIT_COUNT
            and self.expected_fail_count == EXPECTED_FAIL_COUNT
            and self.negative_control_atol == 1e-6
            and self.negative_control_rtol == 0.0
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
                self.production_policy_changed,
            )
        )
        if not frozen or not disabled:
            raise ValueError("TCA-1 protocol is frozen and diagnostic-only")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


__all__ = [
    "EXPECTED_A0_PREDICTION_SHA256",
    "EXPECTED_BY_TASK",
    "EXPECTED_COUNTS",
    "EXPECTED_FAIL_COUNT",
    "EXPECTED_UNIT_COUNT",
    "FORBIDDEN_REVIEW_FIELDS",
    "FROZEN_OVERRIDE_SHA256",
    "FROZEN_REVIEW_PROTOCOL",
    "FROZEN_REVIEW_SHA256",
    "FROZEN_REVIEW_VERSION",
    "FROZEN_SOURCE_QC_SHA256",
    "FROZEN_ZIP_SHA256",
    "PRIMARY_BENCHMARK",
    "PRIMARY_VARIANT",
    "TCA1Settings",
]
