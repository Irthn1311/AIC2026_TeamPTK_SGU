"""Deterministic query and per-event modality routing."""

from __future__ import annotations

import re
from collections.abc import Iterable

from .contracts import RouteDecision

_PATTERNS = {
    "ocr": re.compile(
        r"\b(text|sign|logo|label|number|written|read|chữ|biển|số|nhãn|tên|tiêu đề|"
        r"công thức|câu thơ|nội dung|bảng|khẩu hiệu|ghi|viết|trên giấy|trên bảng|phông nền)\b",
        re.I,
    ),
    "asr": re.compile(
        r"\b(say|said|speak|speech|mention|listen|hear|nói|đọc|nhắc|đề cập|nghe|"
        r"phát biểu|giới thiệu|tường thuật|mẩu tin|bản tin|nghiên cứu|chương trình|sự kiện)\b",
        re.I,
    ),
    "action": re.compile(r"\b(action|then|before|after|while|doing|đang|sau đó|trước khi)\b", re.I),
    "object": re.compile(
        r"\b(object|color|count|many|left|right|near|wearing|màu|bao nhiêu|bên trái|"
        r"bên phải|hai|ba|bốn|năm|nhiều|ở giữa|xung quanh|phía sau|bên cạnh)\b|\b\d+\b",
        re.I,
    ),
}

_KIS_NEWS_TOPIC = re.compile(
    r"\b(mẩu tin|bản tin|tường thuật|giới thiệu|nghiên cứu|chương trình|sự kiện|"
    r"đại học|bệnh viện|câu lạc bộ|thị trấn|địa phương|tỉnh|huyện|xã|lễ hội|"
    r"đạo diễn|bộ phim|tin tức)\b",
    re.I,
)

_ANSWER_PATTERNS = (
    ("COUNT", r"\b(how many|count|bao nhiêu|mấy)\b"),
    ("COLOR", r"\b(color|colour|màu gì|màu nào)\b"),
    ("TITLE", r"\b(tiêu đề|tên món|title)\b"),
    ("QUOTE_OR_VISIBLE_TEXT", r"\b(câu thơ|khẩu hiệu|nội dung|ghi gì|viết gì|chữ gì)\b"),
    (
        "LOCATION_NAME",
        r"\b(tên (?:xã|phường|huyện|tỉnh|thành phố|địa điểm)|(?:xã|phường|huyện|tỉnh|"
        r"thành phố|địa điểm).{0,24}(?:tên|là gì)|place name|location name)\b",
    ),
    ("SPEECH", r"\b(nói gì|đọc gì|nhắc gì|đề cập gì|phát biểu|what .* say)\b"),
    ("PERSON", r"\b(who|ai|người nào)\b"),
    ("YES_NO", r"^(is|are|does|do|did|có phải|có)\b"),
    ("ACTION", r"\b(doing|happen|làm gì|đang làm gì|hành động gì)\b"),
    ("OBJECT", r"\b(what object|what item|vật gì|đồ gì|món gì)\b"),
    ("LOCATION_NAME", r"\b(where|ở đâu|địa điểm nào)\b"),
)


def classify_answer_type(question: str) -> str:
    text = " ".join(str(question).strip().split())
    for answer_type, pattern in _ANSWER_PATTERNS:
        if re.search(pattern, text, re.I):
            return answer_type
    return "OTHER"


def route_query(
    task: str,
    text: str,
    *,
    available: Iterable[str],
    event_index: int | None = None,
    answer_type: str | None = None,
) -> RouteDecision:
    available_set = {str(value).casefold() for value in available}
    modalities, reasons = ["b0_visual"], ["B0_VISUAL_ALWAYS_ON"]
    normalized_task = str(task).upper()
    kind = str(answer_type or "").upper()
    requested: list[tuple[str, str]] = []
    if normalized_task == "TRAKE" and "action" in available_set:
        requested.append(("action", "TRAKE_ACTION_DEFAULT"))
    if normalized_task == "QA":
        if kind in {"LOCATION_NAME", "TITLE", "QUOTE_OR_VISIBLE_TEXT"}:
            requested.extend((("ocr", f"QA_{kind}"), ("asr", f"QA_{kind}")))
        elif kind == "SPEECH":
            requested.append(("asr", "QA_SPEECH"))
        elif kind in {"COUNT", "COLOR", "OBJECT"}:
            requested.extend((("object", f"QA_{kind}"), ("qwen", f"QA_{kind}")))
    if normalized_task == "KIS" and _KIS_NEWS_TOPIC.search(text or ""):
        requested.extend((("asr", "KIS_NEWS_TOPIC"), ("ocr", "KIS_NEWS_TOPIC")))
    for modality, reason in requested:
        if modality in available_set and modality not in modalities:
            modalities.append(modality)
            reasons.append(reason)
    for modality in ("ocr", "asr", "action", "object"):
        if (
            _PATTERNS[modality].search(text or "")
            and modality in available_set
            and modality not in modalities
        ):
            modalities.append(modality)
            reasons.append(f"{modality.upper()}_INTENT")
    return RouteDecision(normalized_task, tuple(modalities), tuple(reasons), event_index)


def route_events(
    task: str, events: Iterable[str], *, available: Iterable[str]
) -> tuple[RouteDecision, ...]:
    return tuple(
        route_query(task, text, available=available, event_index=index)
        for index, text in enumerate(events)
    )
