"""Small, explicit contracts for reference experiment RT1."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

RT1_VERSION = "0.1.0"
RT1_ARMS = ("WHOLE_QUERY", "UNORDERED_EVENT_MAX", "DANTE_DP")
REFERENCE_ALGORITHM = "DANTE_STYLE_MONOTONIC_DP"
IMPLEMENTATION_TYPE = "INDEPENDENT_ADAPTATION"
LAMBDA_SOURCE = "PAPER_REFERENCE_NOT_TUNED_ON_AIC2026"


def _safe_id(value: str, name: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value):
        raise ValueError(f"{name} must be a safe non-empty identifier")
    return value


@dataclass(frozen=True)
class RT1Event:
    event_id: str
    text: str

    def __post_init__(self) -> None:
        _safe_id(self.event_id, "event_id")
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("event text must be non-empty")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> RT1Event:
        if set(value) != {"event_id", "text"}:
            raise ValueError("event fields must be exactly event_id and text")
        return cls(str(value["event_id"]), str(value["text"]))

    def as_dict(self) -> dict[str, str]:
        return {"event_id": self.event_id, "text": self.text}


@dataclass(frozen=True)
class RT1Query:
    query_id: str
    language: str
    narrative_text: str
    events: tuple[RT1Event, ...]
    source: str

    def __post_init__(self) -> None:
        _safe_id(self.query_id, "query_id")
        if self.language not in {"en", "vi"}:
            raise ValueError("RT1 language must be explicit en or vi")
        if not isinstance(self.narrative_text, str) or not self.narrative_text.strip():
            raise ValueError("narrative_text must be non-empty")
        if len(self.events) < 2:
            raise ValueError("RT1 requires at least two ordered events")
        event_ids = [event.event_id for event in self.events]
        if len(set(event_ids)) != len(event_ids):
            raise ValueError("event_id values must be unique within a query")
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("source must be non-empty")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> RT1Query:
        required = {"query_id", "language", "narrative_text", "events", "source"}
        if set(value) != required or not isinstance(value["events"], list):
            raise ValueError(f"RT1 query fields must be exactly {sorted(required)}")
        return cls(
            str(value["query_id"]),
            str(value["language"]),
            str(value["narrative_text"]),
            tuple(RT1Event.from_dict(item) for item in value["events"]),
            str(value["source"]),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "language": self.language,
            "narrative_text": self.narrative_text,
            "events": [event.as_dict() for event in self.events],
            "source": self.source,
        }


@dataclass(frozen=True)
class RT1Settings:
    dante_lambda: float = 0.001
    control_top_k: int = 100
    chain_export_top_k: int = 20
    visual_top_k: int = 3
    review_seed: int = 2026

    def __post_init__(self) -> None:
        if self.dante_lambda < 0:
            raise ValueError("DANTE lambda must be non-negative")
        if not 20 <= self.control_top_k <= 100:
            raise ValueError("control_top_k must be between 20 and 100")
        if min(self.chain_export_top_k, self.visual_top_k) <= 0:
            raise ValueError("RT1 output limits must be positive")


def load_rt1_queries(path: str | Path) -> list[RT1Query]:
    source = Path(path).expanduser().resolve(strict=True)
    queries: list[RT1Query] = []
    try:
        for line in source.read_text(encoding="utf-8").splitlines():
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise TypeError("query row is not an object")
                queries.append(RT1Query.from_dict(value))
    except (json.JSONDecodeError, OSError, TypeError) as error:
        raise ValueError(f"Invalid RT1 query suite {source}: {error}") from error
    if not queries or len({query.query_id for query in queries}) != len(queries):
        raise ValueError("RT1 query suite must contain unique queries")
    return queries


def load_rt1_settings(path: str | Path) -> RT1Settings:
    source = Path(path).expanduser().resolve(strict=True)
    value = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("reference_experiment") != "RT1":
        raise ValueError("Invalid RT1 experiment configuration")
    dante = value.get("dante", {})
    output = value.get("output", {})
    return RT1Settings(
        dante_lambda=float(dante.get("lambda", 0.001)),
        control_top_k=int(value.get("control", {}).get("top_k", 100)),
        chain_export_top_k=int(output.get("chain_top_k", 20)),
        visual_top_k=int(output.get("visual_top_k", 3)),
        review_seed=int(output.get("review_seed", 2026)),
    )


__all__ = [
    "IMPLEMENTATION_TYPE",
    "LAMBDA_SOURCE",
    "REFERENCE_ALGORITHM",
    "RT1_ARMS",
    "RT1_VERSION",
    "RT1Event",
    "RT1Query",
    "RT1Settings",
    "load_rt1_queries",
    "load_rt1_settings",
]
