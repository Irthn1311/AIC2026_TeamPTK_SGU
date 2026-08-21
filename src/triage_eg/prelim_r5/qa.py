"""Deterministic evidence-first QA extraction with frozen BCF1 fallback."""

from __future__ import annotations

import re
import unicodedata
from typing import Any

from triage_eg.fs1.router import classify_answer_type

_LOCATION = re.compile(
    r"\b(xã|phường|thị\s+trấn|huyện|quận|tỉnh|thành\s+phố)\s+"
    r"([A-ZÀ-ỸĐ][\wÀ-ỹĐđ-]*(?:\s+[A-ZÀ-ỸĐ][\wÀ-ỹĐđ-]*){0,4})",
    re.UNICODE,
)
_NUMBER = re.compile(r"(?<!\w)(\d{1,6})(?!\w)")
_NUMBER_WORD = re.compile(r"\b(một|hai|ba|bốn|năm|sáu|bảy|tám|chín|mười|mười một|mười hai)\b", re.I)
_COLOR = re.compile(r"\b(đen|trắng|đỏ|cam|vàng|xanh(?: lá| dương)?|tím|hồng|nâu|xám)\b", re.I)
_TITLE = re.compile(
    r"(?:tên(?:\s+gọi)?(?:\s+của\s+[^,.]{1,40})?\s+là|có\s+tên(?:\s+gọi)?\s+là)"
    r"\s*[:\-]?\s*([^,.!?;]{2,100})",
    re.I,
)
_QUOTED = re.compile(r"[\"“”']([^\"“”']{2,100})[\"“”']")
_WRAPPER = re.compile(
    r"^(?:tên(?:\s+gọi)?\s+là|có\s+tên(?:\s+gọi)?\s+là|món\s+ăn\s+có\s+tên\s+gọi\s+là)\s*[:\-]?\s*",
    re.I,
)


def _compact(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(value)).split())


def _clean_candidate(value: str) -> str:
    return _compact(value).strip(" \t\r\n\"'“”.,:;!?-–—")


def _answer_type(query: dict[str, Any]) -> str:
    explicit = str(query.get("answer_type") or "").upper()
    return explicit or classify_answer_type(str(query.get("question") or query.get("query")))


def _requested_location_unit(question: str) -> str | None:
    folded = question.casefold()
    for unit in ("xã", "phường", "thị trấn", "huyện", "quận", "tỉnh", "thành phố"):
        if re.search(rf"\b{re.escape(unit)}\b", folded):
            return unit
    return None


def _extract(text: str, kind: str, question: str) -> list[tuple[str, str]]:
    output = []
    if kind == "LOCATION_NAME":
        requested = _requested_location_unit(question)
        for match in _LOCATION.finditer(text):
            unit, name = _compact(match.group(1)).casefold(), _compact(match.group(2))
            if requested and unit != requested:
                continue
            output.append((name, "EXPLICIT_LOCATION_UNIT_LINK"))
    elif kind in {"COUNT", "NUMBER"}:
        output.extend((match.group(1), "LOCAL_NUMERIC_SPAN") for match in _NUMBER.finditer(text))
        output.extend(
            (_compact(match.group(1)), "LOCAL_NUMBER_WORD_SPAN")
            for match in _NUMBER_WORD.finditer(text)
        )
    elif kind in {"COLOR", "COLOUR"}:
        output.extend(
            (_compact(match.group(1)), "CLOSED_COLOR_VOCAB_EVIDENCE")
            for match in _COLOR.finditer(text)
        )
    elif kind in {"TITLE", "NAME"}:
        for pattern, reason in (
            (_TITLE, "EXPLICIT_TITLE_PATTERN"),
            (_QUOTED, "QUOTED_TITLE_OR_HEADER"),
        ):
            output.extend(
                (_clean_candidate(_WRAPPER.sub("", match.group(1))), reason)
                for match in pattern.finditer(text)
            )
    elif kind in {"QUOTE_OR_VISIBLE_TEXT", "SPEECH", "TEXT_PRESERVING"}:
        output.extend(
            (_clean_candidate(match.group(1)), "NEAR_VERBATIM_QUOTED_EVIDENCE")
            for match in _QUOTED.finditer(text)
        )
    elif kind == "YES_NO":
        question_terms = {
            token
            for token in re.findall(r"\w+", question.casefold(), re.UNICODE)
            if len(token) >= 4
        }
        text_terms = set(re.findall(r"\w+", text.casefold(), re.UNICODE))
        if len(question_terms & text_terms) >= 2:
            if re.search(r"\b(không|chưa|chẳng)\b", text, re.I):
                output.append(("không", "EXPLICIT_NEGATED_PREDICATE"))
            elif re.search(r"\b(có|đúng|đã)\b", text, re.I):
                output.append(("có", "EXPLICIT_SUPPORTED_PREDICATE"))
    return [(answer, reason) for answer, reason in output if 0 < len(answer) <= 100]


def build_deterministic_qa_rows(
    query: dict[str, Any],
    evidence: list[dict[str, Any]],
    bcf1_fallback: list[dict[str, Any]],
    *,
    context_videos: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Rank supported answer tuples or copy the complete BCF1 fallback exactly."""

    query_id = str(query["query_id"])
    kind = _answer_type(query)
    question = str(query.get("question") or query.get("query"))
    allowed = set(context_videos[:20])
    candidates: dict[tuple[str, int, str], dict[str, Any]] = {}
    audits = []
    for source_rank, row in enumerate(evidence, 1):
        video_id = str(row.get("video_id", ""))
        frame_id = row.get("frame_id")
        text = _compact(str(row.get("text") or row.get("normalized_text") or ""))
        context_relevant = video_id in allowed and isinstance(frame_id, int) and bool(text)
        extracted = _extract(text, kind, question) if context_relevant else []
        audits.append(
            {
                "query_id": query_id,
                "answer_type": kind,
                "source_id": row.get("source_id") or row.get("chunk_id"),
                "source": row.get("source") or row.get("modality"),
                "video_id": video_id or None,
                "frame_id": frame_id,
                "context_relevant": context_relevant,
                "candidate_count": len(extracted),
                "candidates": [answer for answer, _ in extracted],
                "gt_used": False,
            }
        )
        for answer, reason in extracted:
            key = video_id, int(frame_id), answer.casefold()
            current = candidates.get(key)
            value = {
                "query_id": query_id,
                "video_id": video_id,
                "frame_id": int(frame_id),
                "answer": answer,
                "source_rank": source_rank,
                "support_reason": reason,
                "supporting_text": text,
                "answer_type": kind,
            }
            if current is None or source_rank < int(current["source_rank"]):
                candidates[key] = value
    supported = sorted(
        candidates.values(),
        key=lambda row: (
            int(row["source_rank"]),
            str(row["video_id"]),
            int(row["frame_id"]),
            str(row["answer"]).casefold(),
        ),
    )
    if not supported:
        return [dict(row) for row in bcf1_fallback], [
            *audits,
            {
                "query_id": query_id,
                "answer_type": kind,
                "decision": "EXACT_BCF1_FALLBACK_NO_DETERMINISTIC_SUFFICIENT_CANDIDATE",
                "fallback_row_count": len(bcf1_fallback),
                "gt_used": False,
            },
        ]
    output = []
    seen = set()
    for row in [*supported, *bcf1_fallback]:
        key = str(row["video_id"]), int(row["frame_id"]), str(row["answer"]).casefold()
        if key in seen:
            continue
        seen.add(key)
        output.append(
            {
                "query_id": query_id,
                "video_id": row["video_id"],
                "frame_id": int(row["frame_id"]),
                "answer": str(row["answer"])[:100],
                "rank": len(output) + 1,
                "system_variant": "SAFE_R5_QE_DETERMINISTIC_QA",
            }
        )
        if len(output) == 100:
            break
    if len(output) != 100:
        raise RuntimeError(f"R5_QA_TOP100_INCOMPLETE:{query_id}:{len(output)}")
    audits.append(
        {
            "query_id": query_id,
            "answer_type": kind,
            "decision": "DETERMINISTIC_EVIDENCE_FIRST_WITH_BCF1_TAIL",
            "supported_candidate_count": len(supported),
            "gt_used": False,
        }
    )
    return output, audits


__all__ = ["build_deterministic_qa_rows"]
