import re
from enum import Enum


class QuestionType(Enum):
    COLOR = "COLOR"
    COUNT = "COUNT"
    YES_NO = "YES_NO"
    DIRECTION = "DIRECTION"
    UNSUPPORTED = "UNSUPPORTED"


_COLOR_PATTERNS = [
    r"màu\s+gì",
    r"màu\s+nào",
    r"có\s+màu",
    r"màu\s+sắc",
    r"what\s+color",
    r"which\s+color",
    r"color\s+of",
]

_COUNT_PATTERNS = [
    r"bao\s+nhiêu",
    r"\bmấy\b",
    r"số\s+lượng",
    r"how\s+many",
    r"how\s+much",
    r"number\s+of",
]

_YES_NO_PATTERNS = [
    r"có\s+.*\s+không",
    r"có\s+phải",
    r"phải\s+không",
    r"is\s+there",
    r"are\s+there",
    r"\bdoes\b",
    r"\bdid\b",
    r"is\s+the\b",
    r"was\s+there",
]

_DIRECTION_PATTERNS = [
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
]


def _matches_any(text: str, patterns: list[str]) -> bool:
    text_lower = text.lower()
    for pat in patterns:
        if re.search(pat, text_lower):
            return True
    return False


def classify_question_type(question: str, question_en: str | None = None) -> QuestionType:
    q_texts = [question]
    if question_en:
        q_texts.append(question_en)

    for q in q_texts:
        if _matches_any(q, _COLOR_PATTERNS):
            return QuestionType.COLOR
        if _matches_any(q, _COUNT_PATTERNS):
            return QuestionType.COUNT
        if _matches_any(q, _DIRECTION_PATTERNS):
            return QuestionType.DIRECTION
        if _matches_any(q, _YES_NO_PATTERNS):
            return QuestionType.YES_NO

    return QuestionType.UNSUPPORTED
