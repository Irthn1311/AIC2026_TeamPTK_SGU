"""Deterministic, source-traceable query views for Prelim R5."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable
from typing import Any

VIEW_NAMES = (
    "ORIGINAL_VI",
    "TRANSLATED_EN",
    "ENTITY_DISTINCTIVE",
    "ACTION_OBJECT",
    "CONTEXT_ANCHORS",
)

_TOKEN = re.compile(r"[^\W_]+(?:-[^\W_]+)*", re.UNICODE)
_QUOTED = re.compile(r"[\"“”']([^\"“”']{2,80})[\"“”']")
_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")
_ACTION = re.compile(
    r"\b(đang|bắt đầu|tiếp theo|sau đó|đặt|chơi|thu hoạch|vượt|dẫn đầu|"
    r"cắt|rời|xoay|chạm|mở|đóng|nấu|chiên|chạy|đi|cầm|ngồi|đứng|cho ăn)\b",
    re.I,
)
_CONSTRAINT = re.compile(
    r"\b(đen|trắng|đỏ|cam|vàng|xanh|tím|hồng|nâu|xám|một|hai|ba|bốn|năm|"
    r"sáu|đầu tiên|cuối cùng|trước|sau|trái|phải)\b|\b\d+\b",
    re.I,
)
_ENTITY_UNIT = re.compile(
    r"\b(xã|phường|thị trấn|huyện|quận|tỉnh|thành phố|đại học|bệnh viện|"
    r"công ty|trung tâm|lễ hội|chương trình)\b",
    re.I,
)
_GENERIC = frozenset(
    {
        "cảnh",
        "clip",
        "cần",
        "có",
        "của",
        "đây",
        "đoạn",
        "được",
        "hình",
        "khi",
        "là",
        "một",
        "này",
        "người",
        "những",
        "phần",
        "sau",
        "thấy",
        "theo",
        "tìm",
        "trong",
        "trên",
        "video",
        "việc",
        "với",
    }
)


def _compact(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(value)).split())


def _tokens(value: str) -> list[str]:
    return _TOKEN.findall(_compact(value))


def _sentences(value: str) -> list[str]:
    return [_compact(part) for part in _SENTENCE.split(value) if _compact(part)]


def _source_text(query: dict[str, Any], event_index: int | None) -> str:
    if str(query["task"]).upper() != "TRAKE":
        return _compact(str(query["query"]))
    descriptions = query.get("event_descriptions", [])
    if event_index is None or not 0 <= event_index < len(descriptions):
        raise ValueError("TRAKE R5 view requires a valid ordinal event_index")
    event = descriptions[event_index]
    return _compact(str(event["description"]))


def _entity_distinctive(text: str) -> tuple[str, list[str]]:
    quoted = [_compact(value) for value in _QUOTED.findall(text)]
    sentences = _sentences(text)
    entity_sentences = [sentence for sentence in sentences if _ENTITY_UNIT.search(sentence)]
    rare = []
    for token in _tokens(text):
        folded = token.casefold()
        if len(folded) >= 4 and folded not in _GENERIC and token not in rare:
            rare.append(token)
    parts = [*quoted, *entity_sentences, " ".join(rare[:12])]
    values = [part for index, part in enumerate(parts) if part and part not in parts[:index]]
    return _compact(" | ".join(values[:3])) or text, values[:3]


def _action_object(text: str) -> tuple[str, list[str]]:
    sentences = _sentences(text)
    selected = [
        sentence
        for sentence in sentences
        if _ACTION.search(sentence) or _CONSTRAINT.search(sentence)
    ]
    if not selected:
        selected = sentences[:1]
    return _compact(" | ".join(selected[:2])) or text, selected[:2]


def _context_anchors(text: str) -> tuple[str, list[str]]:
    sentences = _sentences(text)
    scored = sorted(
        enumerate(sentences),
        key=lambda item: (
            -sum(bool(pattern.search(item[1])) for pattern in (_ACTION, _CONSTRAINT, _ENTITY_UNIT)),
            -len(set(_tokens(item[1]))),
            item[0],
        ),
    )
    selected = [sentence for _, sentence in scored[:2]]
    return _compact(" | ".join(selected)) or text, selected


def build_query_views(
    query: dict[str, Any],
    *,
    translator: Callable[[str], str],
) -> list[dict[str, Any]]:
    """Create exactly five non-hallucinated views per query or ordinal TRAKE event."""

    task = str(query["task"]).upper()
    event_count = int(query.get("event_count", 1)) if task == "TRAKE" else 1
    rows = []
    for event_index in range(event_count):
        ordinal = event_index if task == "TRAKE" else None
        original = _source_text(query, ordinal)
        translated = _compact(translator(original))
        if not translated:
            raise RuntimeError("R5_TRANSLATED_VIEW_EMPTY")
        entity, entity_sources = _entity_distinctive(original)
        action, action_sources = _action_object(original)
        context, context_sources = _context_anchors(original)
        definitions = (
            ("ORIGINAL_VI", original, "vi", [original], "ORIGINAL_QUERY"),
            ("TRANSLATED_EN", translated, "en", [original], "FROZEN_OPUS_TRANSLATION"),
            (
                "ENTITY_DISTINCTIVE",
                entity,
                "vi",
                entity_sources,
                "DETERMINISTIC_ORIGINAL_SUBSPANS",
            ),
            (
                "ACTION_OBJECT",
                action,
                "vi",
                action_sources,
                "DETERMINISTIC_ORIGINAL_SUBSPANS",
            ),
            (
                "CONTEXT_ANCHORS",
                context,
                "vi",
                context_sources,
                "EXISTING_COMPILER_STYLE_HIGH_MEDIUM_ANCHORS",
            ),
        )
        for name, text, language, sources, derivation in definitions:
            rows.append(
                {
                    "query_id": str(query["query_id"]),
                    "task": task,
                    "view": name,
                    "event_index": ordinal,
                    "text": text,
                    "language": language,
                    "source_text": original,
                    "source_spans": sources,
                    "derivation": derivation,
                    "gt_used": False,
                }
            )
    expected = event_count * len(VIEW_NAMES)
    if len(rows) != expected:
        raise RuntimeError("R5_QUERY_VIEW_CARDINALITY_FAILED")
    return rows


def materialize_view_queries(
    query: dict[str, Any], views: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    """Produce one TEAM query per view while preserving every TRAKE ordinal slot."""

    query_id, task = str(query["query_id"]), str(query["task"]).upper()
    output = {}
    for name in VIEW_NAMES:
        selected = [row for row in views if row["view"] == name]
        if task == "TRAKE":
            selected.sort(key=lambda row: int(row["event_index"]))
            if [row["event_index"] for row in selected] != list(range(int(query["event_count"]))):
                raise RuntimeError(f"R5_TRAKE_ORDINAL_VIEW_COLLAPSE:{query_id}:{name}")
            descriptions = [
                {"event_id": f"E{index + 1}", "description": row["text"]}
                for index, row in enumerate(selected)
            ]
            transformed = {
                **query,
                "query": " ".join(row["text"] for row in selected),
                "event_descriptions": descriptions,
                "language": selected[0]["language"],
            }
        else:
            if len(selected) != 1:
                raise RuntimeError(f"R5_NON_TRAKE_VIEW_CARDINALITY:{query_id}:{name}")
            transformed = {
                **query,
                "query": selected[0]["text"],
                "language": selected[0]["language"],
            }
        transformed["query_id"] = f"{query_id}__r5__{name.casefold()}"
        output[name] = transformed
    return output


__all__ = ["VIEW_NAMES", "build_query_views", "materialize_view_queries"]
