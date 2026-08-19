"""Ground-truth-free ordered QueryEvent compilation."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class CompiledEvent:
    query_id: str
    event_index: int
    event_text: str
    event_count: int
    provenance: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _explicit(query: dict[str, Any]) -> list[str]:
    for key in ("events", "query_events", "event_descriptions"):
        value = query.get(key)
        if isinstance(value, list) and all(
            isinstance(item, str) and item.strip() for item in value
        ):
            return [item.strip() for item in value]
    return []


def _deterministic_split(text: str, count: int) -> list[str]:
    pieces = [
        item.strip(" ,.;")
        for item in re.split(
            r"\b(?:then|after that|next|sau đó|tiếp theo)\b|[;。]", text, flags=re.I
        )
        if item.strip(" ,.;")
    ]
    if len(pieces) == count:
        return pieces
    return [f"event {index + 1}/{count}: {text.strip()}" for index in range(count)]


def compile_query_events(
    query: dict[str, Any],
    b0_rows: list[dict[str, Any]],
    *,
    text_decomposer: Any | None = None,
) -> tuple[CompiledEvent, ...]:
    task, query_id = str(query["task"]).upper(), str(query["query_id"])
    text = str(query.get("query", query.get("description", ""))).strip()
    if task != "TRAKE":
        return (CompiledEvent(query_id, 0, text, 1, "SINGLE_EVENT_TASK"),)
    explicit = _explicit(query)
    structure_count = next(
        (len(row["frame_ids"]) for row in b0_rows if isinstance(row.get("frame_ids"), list)),
        None,
    )
    count = int(query.get("event_count") or structure_count or len(explicit))
    if count < 2:
        raise RuntimeError("TRAKE_EVENT_COUNT_MUST_BE_AT_LEAST_TWO")
    if explicit and len(explicit) == count:
        values, provenance = explicit, "EXPLICIT_QUERY_SCHEMA"
    elif text_decomposer is not None:
        values = list(text_decomposer(text, count))
        provenance = "DETERMINISTIC_QWEN_TEXT_ONLY"
    else:
        values, provenance = _deterministic_split(text, count), "DETERMINISTIC_TEXT_FALLBACK"
    if len(values) != count or any(not str(value).strip() for value in values):
        raise RuntimeError("TRAKE_EVENT_DECOMPOSITION_COUNT_MISMATCH")
    return tuple(
        CompiledEvent(query_id, index, str(value).strip(), count, provenance)
        for index, value in enumerate(values)
    )
