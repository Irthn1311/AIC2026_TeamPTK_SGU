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
    OCR = "OCR"
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

_QA_A3_ADDITIONAL_OCR_PATTERNS = (
    r"\bbien (?:hieu|bao|quang cao) (?:ghi|viet|co) (?:chu )?gi\b",
    r"\bdong chu .* (?:la )?gi\b",
    r"\bnoi dung .* (?:man hinh|bien|bang|poster).* (?:la )?gi\b",
    r"\bcon so .* (?:hien thi|ghi|viet).* (?:la )?gi\b",
    r"\bgia .* (?:ghi|hien thi).* bao nhieu\b",
    r"\bwhat (?:is|was) (?:written|displayed) (?:on|in) "
    r"(?:the )?(?:sign|screen|board|poster)\b",
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
    r"\bcay trong\b.*\bla gi\b",
    r"\bloai xe gi\b",
    r"\bthiet bi gi\b",
    r"\bwhat crop\b",
    r"\bwhat type of vehicle(?:s)?\b",
    r"\bwhat device\b",
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
    r"\bmay (?:nguoi|con|cai|chiec|xe|vat|do vat|chai|qua|mon|lan)\b",
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

_LEGACY_COLOR_PATTERNS = (
    r"màu\s+gì",
    r"màu\s+nào",
    r"có\s+màu",
    r"màu\s+sắc",
    r"what\s+color",
    r"which\s+color",
    r"color\s+of",
)

_LEGACY_COUNT_PATTERNS = (
    r"bao\s+nhiêu",
    r"\bmấy\b",
    r"số\s+lượng",
    r"how\s+many",
    r"how\s+much",
    r"number\s+of",
)

_LEGACY_YES_NO_PATTERNS = (
    r"có\s+.*\s+không",
    r"có\s+phải",
    r"phải\s+không",
    r"^\s*is\s+there\b",
    r"^\s*are\s+there\b",
    r"^\s*is\s+",
    r"^\s*are\s+",
    r"^\s*was\s+",
    r"^\s*were\s+",
    r"^\s*does\b",
    r"^\s*did\b",
    r"^\s*do\b",
    r"^\s*can\b",
)

_LEGACY_DIRECTION_PATTERNS = (
    r"bên\s+trái",
    r"bên\s+phải",
    r"hướng\s+nào",
    r"bên\s+nào",
    r"phía\s+nào",
    r"which\s+direction",
    r"left\s+or\s+right",
    r"which\s+side",
    r"on\s+the\s+left",
    r"on\s+the\s+right",
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


def classify_question_legacy(
    question: str, question_en: str | None = None
) -> QuestionClassification:
    """Reproduce the frozen Phase-P0 classifier at parent ``88b8f2f``."""

    texts = [("question", question.lower())]
    if question_en:
        texts.append(("question_en", question_en.lower()))
    precedence = (
        (_LEGACY_COUNT_PATTERNS, QuestionType.COUNT, "LEGACY_COUNT_PATTERN"),
        (_LEGACY_YES_NO_PATTERNS, QuestionType.YES_NO, "LEGACY_YES_NO_PATTERN"),
        (_LEGACY_DIRECTION_PATTERNS, QuestionType.DIRECTION, "LEGACY_DIRECTION_PATTERN"),
        (_LEGACY_COLOR_PATTERNS, QuestionType.COLOR, "LEGACY_COLOR_PATTERN"),
    )
    for source, text in texts:
        for patterns, question_type, reason in precedence:
            pattern = _matching_pattern(text, patterns)
            if pattern is not None:
                return QuestionClassification(
                    question_type=question_type,
                    reason=f"{reason}:{source}:{pattern}",
                )
    return QuestionClassification(
        question_type=QuestionType.UNSUPPORTED,
        reason="LEGACY_NO_SUPPORTED_QUESTION_PATTERN",
    )


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


def classify_ocr_question(
    question: str,
    question_en: str | None = None,
) -> QuestionClassification | None:
    """Classify high-precision OCR intent only for an enabled QA-A3 provider."""

    texts = [("question", _normalize(question))]
    if question_en:
        texts.append(("question_en", _normalize(question_en)))
    for source, text in texts:
        pattern = _matching_pattern(
            text,
            _OCR_PATTERNS + _QA_A3_ADDITIONAL_OCR_PATTERNS,
        )
        if pattern is not None:
            return QuestionClassification(
                question_type=QuestionType.OCR,
                reason=f"OCR_PATTERN_PROVIDER_ENABLED:{source}:{pattern}",
            )
    return None


def classify_question_type(question: str, question_en: str | None = None) -> QuestionType:
    """Return the frozen Phase-P0 legacy classification contract."""

    return classify_question_legacy(question, question_en).question_type
