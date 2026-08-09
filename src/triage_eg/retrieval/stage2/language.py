"""Conservative deterministic language routing for Stage 2A."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from .contracts import QueryRequest, Stage2RuntimeError

VIETNAMESE_UNICODE = frozenset(
    "ăâđêôơưĂÂĐÊÔƠƯ"
    "áàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩị"
    "óòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ"
    "ÁÀẢÃẠẤẦẨẪẬẮẰẲẴẶÉÈẺẼẸẾỀỂỄỆÍÌỈĨỊ"
    "ÓÒỎÕỌỐỒỔỖỘỚỜỞỠỢÚÙỦŨỤỨỪỬỮỰÝỲỶỸỴ"
)
VI_ASCII_MARKERS = {
    "mot",
    "nguoi",
    "dang",
    "nau",
    "chiec",
    "mau",
    "choi",
    "bong",
    "trong",
    "ngoai",
    "xung",
    "quanh",
    "cai",
}
ENGLISH_MARKERS = {
    "a",
    "the",
    "person",
    "people",
    "car",
    "red",
    "cooking",
    "kitchen",
    "playing",
    "football",
    "sitting",
    "table",
    "indoor",
    "outdoor",
    "with",
}


@dataclass(frozen=True)
class LanguageResolution:
    requested_language: str
    resolved_language: str
    resolution_basis: str
    language_path: str

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


def resolve_language(request: QueryRequest) -> LanguageResolution:
    if request.language == "en":
        return LanguageResolution("en", "en", "EXPLICIT", "DIRECT_CLIP")
    if request.language == "vi":
        return LanguageResolution("vi", "vi", "EXPLICIT", "VI_TO_EN_THEN_CLIP")
    text = request.text.strip()
    if any(character in VIETNAMESE_UNICODE for character in text):
        return LanguageResolution(
            "auto", "vi", "VIETNAMESE_UNICODE_HEURISTIC", "VI_TO_EN_THEN_CLIP"
        )
    if not text.isascii():
        raise Stage2RuntimeError("LANGUAGE_AMBIGUOUS", "unsupported Unicode evidence")
    tokens = re.findall(r"[a-z]+", text.lower())
    if len(set(tokens) & VI_ASCII_MARKERS) >= 2:
        raise Stage2RuntimeError(
            "LANGUAGE_AMBIGUOUS", "Vietnamese without diacritics requires explicit language"
        )
    if len(set(tokens) & ENGLISH_MARKERS) >= 2:
        return LanguageResolution("auto", "en", "ENGLISH_LEXICAL_HEURISTIC", "DIRECT_CLIP")
    raise Stage2RuntimeError("LANGUAGE_AMBIGUOUS", "provide explicit en or vi")


__all__ = ["LanguageResolution", "resolve_language"]
