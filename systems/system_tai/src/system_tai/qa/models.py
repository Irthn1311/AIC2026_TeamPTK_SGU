import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from system_tai.preliminary.schemas import QAPrediction

from .question_types import QuestionType


@dataclass(frozen=True, slots=True)
class QAQuery:
    query_id: str
    event_description: str
    question: str
    event_description_en: str | None = None
    question_en: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.query_id, str) or not self.query_id.strip():
            raise ValueError("query_id must be a non-empty string")
        if not isinstance(self.event_description, str) or not self.event_description.strip():
            raise ValueError("event_description must be a non-empty string")
        if not isinstance(self.question, str) or not self.question.strip():
            raise ValueError("question must be a non-empty string")


@dataclass(frozen=True, slots=True)
class QAEvidenceCandidate:
    query_id: str
    rank: int
    video_id: str
    frame_id: int
    retrieval_score: float
    evidence_score: float | None = None
    source_status: str = "ok"
    timestamp_seconds: float | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if type(self.rank) is not int or self.rank < 1:
            raise TypeError("rank must be an integer >= 1")
        if type(self.frame_id) is not int or self.frame_id < 0:
            raise TypeError("frame_id must be an integer >= 0")
        if not isinstance(self.query_id, str) or not self.query_id.strip():
            raise ValueError("query_id must be a non-empty string")
        if not isinstance(self.video_id, str) or not self.video_id.strip():
            raise ValueError("video_id must be a non-empty string")

        if type(self.retrieval_score) is bool or not isinstance(
            self.retrieval_score, (int, float)
        ):
            raise TypeError("retrieval_score must be a finite float or int")
        if not math.isfinite(float(self.retrieval_score)):
            raise ValueError("retrieval_score must be finite (not NaN or Inf)")

        if self.evidence_score is not None:
            if type(self.evidence_score) is bool or not isinstance(
                self.evidence_score, (int, float)
            ):
                raise TypeError("evidence_score must be a finite float or int when provided")
            if not math.isfinite(float(self.evidence_score)):
                raise ValueError("evidence_score must be finite (not NaN or Inf)")

        if self.timestamp_seconds is not None:
            if type(self.timestamp_seconds) is bool or not isinstance(
                self.timestamp_seconds, (int, float)
            ):
                raise TypeError(
                    "timestamp_seconds must be a non-negative finite float or int when provided"
                )
            if (
                not math.isfinite(float(self.timestamp_seconds))
                or float(self.timestamp_seconds) < 0.0
            ):
                raise ValueError("timestamp_seconds must be non-negative and finite")

        prov_dict = dict(self.provenance) if self.provenance is not None else {}
        object.__setattr__(self, "provenance", MappingProxyType(prov_dict))


@dataclass(frozen=True, slots=True)
class AnswerHypothesis:
    canonical_answer: str
    aliases: tuple[str, ...] = field(default_factory=tuple)
    visual_prompts: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not isinstance(self.canonical_answer, str) or not self.canonical_answer.strip():
            raise ValueError("canonical_answer must be a non-empty string")
        if type(self.aliases) is not tuple:
            raise TypeError("aliases must be a tuple")
        if any(not isinstance(a, str) or not a.strip() for a in self.aliases):
            raise ValueError("all aliases must be non-empty strings")
        if type(self.visual_prompts) is not tuple:
            raise TypeError("visual_prompts must be a tuple")
        if any(not isinstance(vp, str) or not vp.strip() for vp in self.visual_prompts):
            raise ValueError("all visual_prompts must be non-empty strings")


@dataclass(frozen=True, slots=True)
class QAResult:
    query_id: str
    question_type: QuestionType
    predictions: list[QAPrediction]
    unsupported_reason: str | None = None
    warnings: list[str] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)
