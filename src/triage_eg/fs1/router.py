"""Deterministic query and per-event modality routing."""

from __future__ import annotations

import re
from collections.abc import Iterable

from .contracts import RouteDecision

_PATTERNS = {
    "ocr": re.compile(r"\b(text|sign|logo|label|number|written|read|chữ|biển|số|nhãn)\b", re.I),
    "asr": re.compile(r"\b(say|said|speak|speech|mention|listen|hear|nói|đề cập|nghe)\b", re.I),
    "action": re.compile(r"\b(action|then|before|after|while|doing|đang|sau đó|trước khi)\b", re.I),
    "object": re.compile(
        r"\b(object|color|count|many|left|right|near|wearing|màu|bao nhiêu|bên trái|bên phải)\b",
        re.I,
    ),
}


def route_query(
    task: str, text: str, *, available: Iterable[str], event_index: int | None = None
) -> RouteDecision:
    available_set = {str(value).casefold() for value in available}
    modalities, reasons = ["b0_visual"], ["B0_VISUAL_ALWAYS_ON"]
    for modality in ("ocr", "asr", "action", "object"):
        if _PATTERNS[modality].search(text or "") and modality in available_set:
            modalities.append(modality)
            reasons.append(f"{modality.upper()}_INTENT")
    return RouteDecision(str(task).upper(), tuple(modalities), tuple(reasons), event_index)


def route_events(
    task: str, events: Iterable[str], *, available: Iterable[str]
) -> tuple[RouteDecision, ...]:
    return tuple(
        route_query(task, text, available=available, event_index=index)
        for index, text in enumerate(events)
    )
