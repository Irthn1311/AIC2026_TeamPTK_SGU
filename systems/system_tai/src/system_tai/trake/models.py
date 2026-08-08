import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from system_tai.preliminary.schemas import TRAKEPrediction


@dataclass(frozen=True, slots=True)
class TRAKEEvent:
    event_index: int
    description: str
    description_en: str | None = None

    def __post_init__(self) -> None:
        if type(self.event_index) is not int or self.event_index < 0:
            raise TypeError("event_index must be a non-negative integer (bool not allowed)")
        if not isinstance(self.description, str) or not self.description.strip():
            raise ValueError("description must be a non-empty string")
        if self.description_en is not None:
            if not isinstance(self.description_en, str) or not self.description_en.strip():
                raise ValueError("description_en must be None or a non-empty string")


@dataclass(frozen=True, slots=True)
class TRAKEQuery:
    query_id: str
    events: tuple[TRAKEEvent, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.query_id, str) or not self.query_id.strip():
            raise ValueError("query_id must be a non-empty string")
        if type(self.events) is not tuple:
            raise TypeError("events must be a tuple")
        if len(self.events) < 1:
            raise ValueError("events tuple must contain at least one event")
        for idx, ev in enumerate(self.events):
            if type(ev) is not TRAKEEvent:
                raise TypeError(f"Element at index {idx} in events must be TRAKEEvent instance")
            if ev.event_index != idx:
                raise ValueError(
                    f"Event index mismatch at pos {idx}: expected {idx}, got {ev.event_index}"
                )


@dataclass(frozen=True, slots=True)
class TRAKEEventCandidate:
    query_id: str
    event_index: int
    rank: int
    video_id: str
    frame_id: int
    retrieval_score: float
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.query_id, str) or not self.query_id.strip():
            raise ValueError("query_id must be a non-empty string")
        if type(self.event_index) is not int or self.event_index < 0:
            raise TypeError("event_index must be a non-negative integer (bool not allowed)")
        if type(self.rank) is not int or self.rank < 1:
            raise TypeError("rank must be an integer >= 1 (bool not allowed)")
        if not isinstance(self.video_id, str) or not self.video_id.strip():
            raise ValueError("video_id must be a non-empty string")
        if type(self.frame_id) is not int or self.frame_id < 0:
            raise TypeError("frame_id must be a non-negative integer (bool not allowed)")
        if type(self.retrieval_score) is bool or not isinstance(
            self.retrieval_score, (int, float)
        ):
            raise TypeError("retrieval_score must be a finite float or int")
        if not math.isfinite(float(self.retrieval_score)):
            raise ValueError("retrieval_score must be finite")
        if not isinstance(self.provenance, Mapping):
            raise TypeError("provenance must be a Mapping")
        if not isinstance(self.provenance, MappingProxyType):
            object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))


@dataclass(frozen=True, slots=True)
class TRAKEResult:
    query_id: str
    event_count: int
    predictions: tuple[TRAKEPrediction, ...]
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.query_id, str) or not self.query_id.strip():
            raise ValueError("query_id must be a non-empty string")
        if type(self.event_count) is not int or self.event_count < 1:
            raise TypeError("event_count must be an integer >= 1 (bool not allowed)")
        if type(self.predictions) is not tuple:
            raise TypeError("predictions must be a tuple")
        for p in self.predictions:
            if not isinstance(p, TRAKEPrediction):
                raise TypeError("predictions must contain TRAKEPrediction instances")
        if not isinstance(self.diagnostics, Mapping):
            raise TypeError("diagnostics must be a Mapping")
        if not isinstance(self.diagnostics, MappingProxyType):
            object.__setattr__(self, "diagnostics", MappingProxyType(dict(self.diagnostics)))
