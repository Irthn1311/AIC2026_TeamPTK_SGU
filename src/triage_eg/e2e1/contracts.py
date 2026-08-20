"""Frozen contracts for the TRIAGE-EG E2E-1 integration baseline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

MAX_PREDICTIONS = 100
M1_REFINE_TOP_SINGLE_EVENT = 10
M1_REFINE_TOP_CHAINS = 5
T3_SELECTED_DELTA = 0.05
OCR_MAX_GROUNDING_RANKS = 20
VARIANTS = ("P0_COARSE", "P1_CANONICAL")
FORBIDDEN_INFERENCE_FIELDS = frozenset(
    {
        "gt",
        "accepted_intervals",
        "acceptable_intervals",
        "correct_video",
        "accepted_answers",
        "aliases",
        "event_intervals",
        "annotation_audit",
    }
)


@dataclass(frozen=True)
class E2E1Settings:
    max_predictions: int = MAX_PREDICTIONS
    m1_refine_top_single_event: int = M1_REFINE_TOP_SINGLE_EVENT
    m1_refine_top_chains: int = M1_REFINE_TOP_CHAINS
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
            self.max_predictions == 100
            and self.m1_refine_top_single_event == 10
            and self.m1_refine_top_chains == 5
            and self.t3_selected_delta == 0.05
            and self.ocr_max_grounding_ranks == 20
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
            raise ValueError("E2E-1 settings are frozen and cannot be tuned")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class QueryPlan:
    query_id: str
    task: str
    language: str
    grounding_text: str
    question: str | None
    events: tuple[tuple[str, str], ...]
    answer_type: str | None = None
    compiled_routing: tuple[str, ...] = ()
    answer_policy: str | None = None
    evidence_provenance: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "task": self.task,
            "language": self.language,
            "grounding_text": self.grounding_text,
            "question": self.question,
            "events": [
                {"event_id": event_id, "description": text} for event_id, text in self.events
            ],
            "answer_type": self.answer_type,
            "compiled_routing": list(self.compiled_routing),
            "answer_policy": self.answer_policy,
            "evidence_provenance": list(self.evidence_provenance),
            "gt_available_to_inference": False,
        }


__all__ = [
    "E2E1Settings",
    "FORBIDDEN_INFERENCE_FIELDS",
    "MAX_PREDICTIONS",
    "M1_REFINE_TOP_CHAINS",
    "M1_REFINE_TOP_SINGLE_EVENT",
    "OCR_MAX_GROUNDING_RANKS",
    "QueryPlan",
    "T3_SELECTED_DELTA",
    "VARIANTS",
]
