"""Deterministic QA type routing and shortest-answer canonicalization."""

from __future__ import annotations

import re

from .contracts import QWEN_ANSWER_LIMIT

_TYPES = (
    ("count", r"\b(how many|count|bao nhiêu|mấy)\b"),
    ("color", r"\b(color|colour|màu)\b"),
    ("visible_text", r"\b(text|written|sign|read|chữ|biển)\b"),
    ("speech", r"\b(say|said|mention|nói|đề cập)\b"),
    ("location", r"\b(where|location|ở đâu)\b"),
    ("person", r"\b(who|person|ai|người nào)\b"),
    ("yes_no", r"^(is|are|does|do|did|có phải|có)\b"),
    ("action", r"\b(doing|happen|action|làm gì|đang làm)\b"),
    ("object", r"\b(what object|what item|vật gì|đồ gì)\b"),
)


def answer_type(question: str) -> str:
    for name, pattern in _TYPES:
        if re.search(pattern, question.strip(), re.I):
            return name
    return "other"


def canonical_short_answer(value: str, kind: str) -> str:
    answer = " ".join(str(value).strip().split())
    if kind == "count":
        words = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "năm": "5"}
        for token in re.findall(r"\w+", answer.casefold()):
            if token.isdigit():
                answer = token
                break
            if token in words:
                answer = words[token]
                break
    if kind == "color":
        answer = re.sub(r"^(?:màu|color is|the color is)\s+", "", answer, flags=re.I)
    if kind == "object":
        answer = re.sub(r"^(?:một|an?|the)\s+", "", answer, flags=re.I)
    if len(answer.encode("utf-8")) > QWEN_ANSWER_LIMIT:
        raise ValueError("QA_CANONICAL_ANSWER_EXCEEDS_100_BYTES")
    return answer
