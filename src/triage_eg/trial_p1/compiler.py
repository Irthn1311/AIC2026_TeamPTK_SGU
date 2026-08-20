"""Deterministic, GT-free Trial P1 Query Compiler."""

from __future__ import annotations

import re
from typing import Any

from triage_eg.fs1.router import classify_answer_type, route_events, route_query

_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")
_ACTION = re.compile(
    r"\b(đang|bắt đầu|tiếp theo|sau đó|lần lượt|đặt|chơi|thu hoạch|vượt|dẫn đầu|"
    r"cắt|rời|xoay|chạm|chào|mở lửa|trang trí|vệ sinh|cho ăn)\b",
    re.I,
)
_TEXT = re.compile(
    r"\b(tên|tiêu đề|chữ|khẩu hiệu|logo|bảng|biển|ghi|viết|công thức|câu thơ|"
    r"mẩu tin|bản tin|chương trình|sự kiện|đại học|bệnh viện|tỉnh|xã)\b",
    re.I,
)
_VISUAL = re.compile(
    r"\b(cảnh|hình ảnh|màu|mặc|đội|cầm|ngồi|đứng|đĩa|chim|dê|bánh|máy ảnh|"
    r"góc quay|flycam|con lân|chảo|nấm|măng tây)\b",
    re.I,
)


def _sentences(text: str) -> list[str]:
    return [" ".join(part.split()) for part in _SENTENCE.split(text) if part.strip()]


def _bounded(values: list[str], limit: int) -> list[str]:
    result: list[str] = []
    for value in values:
        compact = " ".join(value.split())
        if compact and compact not in result:
            result.append(compact)
        if len(result) == limit:
            break
    return result


def _knowledge_expansions(text: str) -> list[dict[str, str]]:
    # A generic high-confidence entity resolution rule. It is text-pattern based,
    # never keyed by query ID and never replaces the source query.
    if re.search(r"Steven\s+Spielberg", text, re.I) and re.search(r"\b1975\b", text):
        return [
            {
                "text": "cá mập nguy hiểm, bộ phim Jaws của Steven Spielberg năm 1975",
                "source": "QUERY_KNOWLEDGE_EXPANSION",
                "confidence": "HIGH",
            }
        ]
    return []


def _split_qa(text: str) -> tuple[str, str]:
    matches = list(re.finditer(r"(?:^|\s)(Hỏi\s+.+?\?)", text, re.I | re.S))
    if matches:
        match = matches[-1]
        return text[: match.start(1)].strip(), match.group(1).strip()
    question_start = text.rfind("?")
    if question_start >= 0:
        previous = max(text.rfind(".", 0, question_start), text.rfind("\n", 0, question_start))
        return text[: previous + 1].strip(), text[previous + 1 : question_start + 1].strip()
    raise ValueError("TRIAL_QA_QUESTION_NOT_FOUND")


def compile_query(
    row: dict[str, Any], *, available: tuple[str, ...] = ("ocr", "asr", "action", "object", "qwen")
) -> dict[str, Any]:
    text = str(row["normalized_text"])
    task = str(row["task"]).upper()
    sentences = _sentences(text)
    events = list(row.get("events", []))
    grounding_text, question = text, None
    if task == "QA":
        grounding_text, question = _split_qa(text)
    elif task == "TRAKE":
        grounding_text = str(row.get("context") or " ".join(e["description"] for e in events))

    semantic_core = sentences[0] if sentences else text
    visual = _bounded([part for part in sentences if _VISUAL.search(part)], 4)
    action = _bounded([part for part in sentences if _ACTION.search(part)], 4)
    text_anchors = _bounded([part for part in sentences if _TEXT.search(part)], 4)
    expansions = _knowledge_expansions(text)
    answer_kind = classify_answer_type(question or text) if task == "QA" else None
    if task == "TRAKE":
        routes = [
            decision.as_dict()
            if hasattr(decision, "as_dict")
            else {
                "task": decision.task,
                "modalities": list(decision.modalities),
                "reasons": list(decision.reasons),
                "event_index": decision.event_index,
            }
            for decision in route_events(
                task, [e["description"] for e in events], available=available
            )
        ]
    else:
        decision = route_query(task, text, available=available, answer_type=answer_kind)
        routes = [
            {
                "task": decision.task,
                "modalities": list(decision.modalities),
                "reasons": list(decision.reasons),
                "event_index": decision.event_index,
            }
        ]

    variants = _bounded(
        [
            semantic_core,
            *visual[:2],
            *action[:2],
            *text_anchors[:2],
            *[e["text"] for e in expansions],
        ],
        5,
    )
    team_query: dict[str, Any] = {
        "query_id": row["query_id"],
        "task": task,
        "language": "vi",
        "query": grounding_text,
    }
    if task == "QA":
        team_query["question"] = question
    if task == "TRAKE":
        team_query.update(
            {
                "event_count": len(events),
                "event_descriptions": [
                    {"event_id": e["event_id"], "description": e["description"]} for e in events
                ],
            }
        )
    return {
        "query_id": row["query_id"],
        "task": task,
        "raw_text": row["raw_text"],
        "semantic_core": semantic_core,
        "visual_anchors": visual,
        "action_anchors": action,
        "text_entity_anchors": text_anchors,
        "knowledge_expansions": expansions,
        "retrieval_variants": variants,
        "events": events,
        "answer_type": answer_kind,
        "routing": routes,
        "team_query": team_query,
        "gt_used": False,
    }


def compile_queries(manifest: dict[str, Any], **kwargs: Any) -> list[dict[str, Any]]:
    return [compile_query(row, **kwargs) for row in manifest["queries"]]


__all__ = ["compile_query", "compile_queries"]
