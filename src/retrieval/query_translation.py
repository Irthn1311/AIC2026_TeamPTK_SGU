from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


logger = logging.getLogger("aic.query_translation")

_VIETNAMESE_CHARS = set(
    "ăâđêôơư"
    "áàảãạắằẳẵặấầẩẫậéèẻẽẹếềểễệíìỉĩị"
    "óòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ"
)
_VI_MARKERS = {
    "mot",
    "nhieu",
    "nguoi",
    "trong",
    "ngoai",
    "mau",
    "duoi",
    "tren",
    "canh",
    "dong",
    "nuoc",
    "cay",
    "xe",
    "dang",
    "phia",
    "voi",
}


@dataclass(frozen=True)
class TranslationResult:
    text: str
    source: str
    usable: bool


def _ascii_fold(text: str) -> str:
    import unicodedata

    text = unicodedata.normalize("NFD", str(text or ""))
    return "".join(ch for ch in text if unicodedata.category(ch) != "Mn").replace("đ", "d").replace("Đ", "D")


def looks_vietnamese(text: str) -> bool:
    raw = str(text or "").strip().lower()
    if not raw:
        return False
    if any(ch in _VIETNAMESE_CHARS for ch in raw):
        return True
    folded = re.sub(r"[^a-z0-9\s]", " ", _ascii_fold(raw).lower())
    words = set(re.sub(r"\s+", " ", folded).strip().split())
    return bool(words.intersection(_VI_MARKERS))


def looks_like_usable_english(query: str, translated: str) -> bool:
    translated = str(translated or "").strip()
    query = str(query or "").strip()
    if not translated or translated == query:
        return False
    # Visible OCR text should be preserved verbatim, often inside quotes.
    # Do not reject an otherwise-English translation just because quoted text
    # still contains Vietnamese accents.
    without_quotes = re.sub(r'"[^"]*"|“[^”]*”|\'[^\']*\'', " ", translated)
    if looks_vietnamese(without_quotes):
        return False
    latin_chars = sum(ch.isascii() and ch.isalpha() for ch in translated)
    return latin_chars >= 4


class CachedQueryTranslator:
    """VI->EN query translator with transparent cache and no hand-written semantic expansion."""

    def __init__(self, cache_path: Path | None = None, enable_google: bool = True):
        self.cache_path = cache_path
        self.enable_google = enable_google
        self._cache: dict[str, str] = {}
        self._google: Any = None
        if cache_path is not None and cache_path.exists():
            try:
                payload = json.loads(cache_path.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    self._cache = {str(k): str(v) for k, v in payload.items() if str(v).strip()}
            except Exception as exc:
                logger.warning("Could not read translation cache %s: %s", cache_path, exc)

    def initialize_google(self) -> None:
        if not self.enable_google or self._google is not None:
            return
        try:
            from deep_translator import GoogleTranslator

            self._google = GoogleTranslator(source="auto", target="en")
            logger.info("GoogleTranslator initialized for query translation.")
        except Exception as exc:
            logger.warning("GoogleTranslator not available; using cache/original query only: %s", exc)

    def translate(self, query: str) -> TranslationResult:
        q = str(query or "").strip()
        if not q:
            return TranslationResult(text=q, source="empty", usable=False)

        cached = self._cache.get(q)
        if cached:
            usable = looks_like_usable_english(q, cached)
            return TranslationResult(text=cached if usable else q, source="cache", usable=usable)

        self.initialize_google()
        if self._google is not None:
            try:
                translated = str(self._google.translate(q) or "").strip()
                if looks_like_usable_english(q, translated):
                    self._cache[q] = translated
                    self._save_cache()
                    return TranslationResult(text=translated, source="google", usable=True)
            except Exception as exc:
                logger.debug("Google translation skipped: %s", exc)

        return TranslationResult(text=q, source="original", usable=False)

    def _save_cache(self) -> None:
        if self.cache_path is None:
            return
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(
                json.dumps(self._cache, ensure_ascii=False, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.debug("Could not save translation cache %s: %s", self.cache_path, exc)
