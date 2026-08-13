from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from enum import Enum


class QuestionType(Enum):
    COLOR = "COLOR"
    COUNT = "COUNT"
    YES_NO = "YES_NO"
    DIRECTION = "DIRECTION"
    OBJECT_ENTITY = "OBJECT_ENTITY"
    OBJECT_COUNT = "OBJECT_COUNT"
    UNSUPPORTED = "UNSUPPORTED"


@dataclass(frozen=True, slots=True)
class QuestionClassification:
    question_type: QuestionType
    reason: str


_OCR_PATTERNS = (
    r"\bbien so\b",
    r"\bdoc (?:chu|dong|noi dung)\b",
    r"\bchu gi\b",
    r"\bso tien\b",
    r"\bgia (?:bao nhieu|la bao nhieu)\b",
    r"\bbao nhieu tien\b",
    r"\blicense plate\b",
    r"\bwhat (?:does|do) .* say\b",
    r"\bwhat (?:text|number) (?:is|was) (?:shown|written|displayed)\b",
    r"\bread (?:the )?(?:text|sign|number)\b",
)

_OBJECT_COUNT_PATTERNS = (
    r"\bbao nhieu (?:nguoi|xe|vat|do vat|con|chai|qua|mon do)\b",
    r"\bso luong (?:nguoi|xe|vat|do vat|con|chai|qua|mon do)\b",
    r"\bhow many (?:people|persons|cars|vehicles|objects|items|bottles|animals)\b",
    r"\bnumber of (?:people|persons|cars|vehicles|objects|items|bottles|animals)\b",
)

_OBJECT_ENTITY_PATTERNS = (
    r"\bday la vat gi\b",
    r"\bvat (?:the )?gi\b",
    r"\bvat gi\b",
    r"\b(?:nguoi|anh ay|co ay|nguoi nay) .*\bcam gi\b",
    r"\bdang cam gi\b",
    r"\btrong .* co vat gi\b",
    r"\bwhat object\b",
    r"\bwhat is the object\b",
    r"\bwhich object\b",
    r"\bwhat item\b",
    r"\bwhat (?:is|was) (?:he|she|the person|the man|the woman) holding\b",
)

_COLOR_PATTERNS = (
    r"\bmau gi\b",
    r"\bmau nao\b",
    r"\bco mau\b",
    r"\bmau sac\b",
    r"\bwhat color\b",
    r"\bwhich color\b",
    r"\bcolor of\b",
)

_COUNT_PATTERNS = (
    r"\bbao nhieu\b",
    r"\bmay\b",
    r"\bso luong\b",
    r"\bhow many\b",
    r"\bhow much\b",
    r"\bnumber of\b",
)

_YES_NO_PATTERNS = (
    r"\bco .* khong\b",
    r"\bco phai\b",
    r"\bphai khong\b",
    r"^\s*is there\b",
    r"^\s*are there\b",
    r"^\s*(?:is|are|was|were)\s+",
    r"^\s*(?:does|did|do|can)\b",
)

_DIRECTION_PATTERNS = (
    r"\bben trai\b",
    r"\bben phai\b",
    r"\bhuong nao\b",
    r"\bben nao\b",
    r"\bphia nao\b",
    r"\bwhich direction\b",
    r"\bleft or right\b",
    r"\bwhich side\b",
    r"\bon the left\b",
    r"\bon the right\b",
)


def _normalize(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    without_marks = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return " ".join(without_marks.split())


def _matching_pattern(text: str, patterns: tuple[str, ...]) -> str | None:
    for pattern in patterns:
        if re.search(pattern, text):
            return pattern
    return None


def classify_question(
    question: str, question_en: str | None = None
) -> QuestionClassification:
    texts = [("question", _normalize(question))]
    if question_en:
        texts.append(("question_en", _normalize(question_en)))

    precedence = (
        (_OCR_PATTERNS, QuestionType.UNSUPPORTED, "OCR_PATTERN_PROVIDER_MISSING"),
        (_OBJECT_COUNT_PATTERNS, QuestionType.OBJECT_COUNT, "OBJECT_COUNT_PATTERN"),
        (_OBJECT_ENTITY_PATTERNS, QuestionType.OBJECT_ENTITY, "OBJECT_ENTITY_PATTERN"),
        (_COUNT_PATTERNS, QuestionType.COUNT, "LEGACY_COUNT_PATTERN"),
        (_YES_NO_PATTERNS, QuestionType.YES_NO, "LEGACY_YES_NO_PATTERN"),
        (_DIRECTION_PATTERNS, QuestionType.DIRECTION, "LEGACY_DIRECTION_PATTERN"),
        (_COLOR_PATTERNS, QuestionType.COLOR, "LEGACY_COLOR_PATTERN"),
    )
    for source, text in texts:
        for patterns, q_type, reason in precedence:
            pattern = _matching_pattern(text, patterns)
            if pattern is not None:
                return QuestionClassification(
                    question_type=q_type,
                    reason=f"{reason}:{source}:{pattern}",
                )
    return QuestionClassification(
        question_type=QuestionType.UNSUPPORTED,
        reason="NO_SUPPORTED_QUESTION_PATTERN",
    )


def classify_question_type(question: str, question_en: str | None = None) -> QuestionType:
    """Backward-compatible question-type-only classification API."""

    classified = classify_question(question, question_en).question_type
    # Preserve the public Phase-P0 classifier contract for legacy callers. The
    # capability-aware runtime consumes ``classify_question`` directly and can
    # fail closed for artifact-only OBJECT_COUNT until a defensible provider exists.
    return QuestionType.COUNT if classified is QuestionType.OBJECT_COUNT else classified
