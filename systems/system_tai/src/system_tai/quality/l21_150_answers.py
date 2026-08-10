"""Conservative, deterministic Q&A answer normalization for L21-150."""

from __future__ import annotations

import re
import unicodedata

_SPACES = re.compile(r"\s+")
_OUTER_PUNCTUATION = re.compile(r"^[\s.,;:!?\"'“”‘’()\[\]{}]+|[\s.,;:!?\"'“”‘’()\[\]{}]+$")
_THOUSANDS = re.compile(r"(?<=\d)[.,](?=\d{3}(?:\D|$))")
_DECIMAL_COMMA = re.compile(r"(?<=\d),(?=\d{1,2}(?:\D|$))")
_PERCENT_SPACING = re.compile(r"\s*%")
_RANGE_SPACING = re.compile(r"\s*[-–—]\s*")
_ALTERNATIVE_SEPARATOR = re.compile(r"\s*/\s*")


def normalize_answer(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("answer must be a non-empty string")
    text = unicodedata.normalize("NFKC", value).casefold()
    text = text.replace("\u00a0", " ").replace("’", "'").replace("‘", "'")
    text = _SPACES.sub(" ", text).strip()
    text = _OUTER_PUNCTUATION.sub("", text)
    text = _THOUSANDS.sub("", text)
    text = _DECIMAL_COMMA.sub(".", text)
    text = _PERCENT_SPACING.sub("%", text)
    text = _RANGE_SPACING.sub("-", text)
    return _SPACES.sub(" ", text).strip()


def source_answer_aliases(source_answer: str) -> tuple[str, ...]:
    if not isinstance(source_answer, str) or not source_answer.strip():
        raise ValueError("source_answer must be a non-empty string")
    aliases = tuple(part.strip() for part in _ALTERNATIVE_SEPARATOR.split(source_answer))
    if any(not alias for alias in aliases):
        raise ValueError("source answer contains an empty explicit alternative")
    return aliases


def answer_matches(prediction: str, accepted_answers: tuple[str, ...]) -> bool:
    normalized_prediction = normalize_answer(prediction)
    return normalized_prediction in {normalize_answer(answer) for answer in accepted_answers}
