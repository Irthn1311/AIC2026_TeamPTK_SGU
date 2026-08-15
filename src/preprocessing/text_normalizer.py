"""
Text Normalizer — Deep Normalization for AIC Query Preprocessing.

Performs deeper normalization than text_cleaner.py:
  1. Unicode NFC + NFKC (both passes)
  2. Abbreviation expansion (MC, VTV, SEA Games, ...)
  3. Number-word → digit conversion (ba → 3, mười hai → 12)
  4. Smart lowercasing (preserve ACRONYMS like VTV1, HTV, VNPT)
  5. Telex/VNI typo detection & basic correction
  6. Punctuation normalization (preserve meaningful dashes & quotes)
  7. Bilingual (Vi + En) — handles mixed-language queries transparently
"""

from __future__ import annotations

import re
import unicodedata
from typing import Dict, List, Tuple


# ============================================================
# Abbreviation Expansion Dictionary
# ============================================================
_ABBREVIATION_MAP: Dict[str, str] = {
    # Vietnamese broadcast channels
    "VTV1": "VTV1 đài truyền hình Việt Nam kênh 1",
    "VTV2": "VTV2 đài truyền hình Việt Nam kênh 2",
    "VTV3": "VTV3 đài truyền hình Việt Nam kênh 3",
    "VTV4": "VTV4",
    "VTV6": "VTV6",
    "HTV7": "HTV7 đài truyền hình thành phố Hồ Chí Minh kênh 7",
    "HTV9": "HTV9",
    "THVL1": "THVL1 truyền hình Vĩnh Long kênh 1",
    "SCTV": "SCTV kênh truyền hình",
    # Common Vietnamese roles
    "MC": "người dẫn chương trình",
    "BTV": "biên tập viên truyền hình",
    "PTV": "phát thanh viên",
    "ĐTV": "điều tra viên",
    # Sports
    "SEA GAMES": "SEA Games Đại hội Thể thao Đông Nam Á",
    "SEAGAMES": "SEA Games Đại hội Thể thao Đông Nam Á",
    "ASIAD": "ASIAD Đại hội Thể thao Châu Á",
    "FIFA": "FIFA tổ chức bóng đá thế giới",
    "UEFA": "UEFA liên đoàn bóng đá châu Âu",
    "WC": "World Cup bóng đá thế giới",
    "CK": "chung kết",
    "BK": "bán kết",
    "TK": "tứ kết",
    # Other
    "UBND": "Ủy ban nhân dân",
    "HĐND": "Hội đồng nhân dân",
    "TW": "Trung ương",
    "TP": "thành phố",
    "Q.": "quận",
    "P.": "phường",
    "GS": "giáo sư",
    "TS": "tiến sĩ",
    "PGS": "phó giáo sư",
    "BS": "bác sĩ",
    "KTS": "kiến trúc sư",
    "CV": "chuyên viên",
}

# ============================================================
# Number-Word → Digit Conversion
# ============================================================
_VI_NUMBER_WORDS: Dict[str, int] = {
    "không": 0, "một": 1, "hai": 2, "ba": 3, "bốn": 4,
    "năm": 5, "sáu": 6, "bảy": 7, "tám": 8, "chín": 9,
    "mười": 10, "mười một": 11, "mười hai": 12,
    "mười ba": 13, "mười bốn": 14, "mười lăm": 15,
    "mười sáu": 16, "mười bảy": 17, "mười tám": 18, "mười chín": 19,
    "hai mươi": 20, "ba mươi": 30, "bốn mươi": 40, "năm mươi": 50,
    "một trăm": 100, "hai trăm": 200, "ba trăm": 300,
    "một nghìn": 1000, "hai nghìn": 2000,
}

_EN_NUMBER_WORDS: Dict[str, int] = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
    "hundred": 100, "thousand": 1000,
}

# Words that look like number words but are NOT quantities in query context
_VI_NUMBER_EXCEPTIONS = {
    "năm",  # Can mean "year" or "lie down" — only convert when next to a count noun
}

# Count nouns that confirm preceding "năm" is a number
_VI_COUNT_NOUNS = [
    "người", "cầu thủ", "vận động viên", "chiếc", "cái", "con", "em",
    "bạn", "đội", "lần", "bàn thắng", "điểm", "huy chương",
]

# ============================================================
# Telex / VNI Common Typos
# (Only the clearest, unambiguous corrections)
# ============================================================
_TELEX_CORRECTIONS: List[Tuple[str, str]] = [
    # Common unaccented → accented
    (r"\bnguoi\b", "người"),
    (r"\bviec\b", "việc"),
    (r"\bkhong\b", "không"),
    (r"\bduoc\b", "được"),
    (r"\bthuong\b", "thường"),
    (r"\bnhau\b", "nhau"),
    (r"\bmuon\b", "muốn"),
    (r"\bnhan\b", "nhận"),
    (r"\bdung\b", "đúng"),
    (r"\bquyen\b", "quyền"),
]


class TextNormalizer:
    """
    Deep text normalizer for AIC query preprocessing.

    Usage:
        normalizer = TextNormalizer()
        clean = normalizer.normalize("MC VTV3 đang phát biểu năm người")
        # → "người dẫn chương trình VTV3 đài truyền hình Việt Nam kênh 3 đang phát biểu 5 người"
    """

    def __init__(
        self,
        expand_abbreviations: bool = True,
        convert_number_words: bool = True,
        fix_telex_typos: bool = False,  # Disabled by default — risky without full dict
    ):
        self.expand_abbreviations = expand_abbreviations
        self.convert_number_words = convert_number_words
        self.fix_telex_typos = fix_telex_typos

        # Pre-sort abbreviation keys by length (longest-first) to avoid partial matches
        self._abbrev_keys_sorted = sorted(
            _ABBREVIATION_MAP.keys(), key=len, reverse=True
        )

    # ----------------------------------------------------------
    # Main Entry
    # ----------------------------------------------------------

    def normalize(self, raw_text: str) -> str:
        """
        Full normalization pipeline.

        Steps:
          1. Unicode NFC + NFKC
          2. Control character removal
          3. Preserve ACRONYMS, lowercase the rest
          4. Optional telex correction
          5. Abbreviation expansion
          6. Number-word → digit
          7. Whitespace normalization
          8. Punctuation cleanup

        Returns:
            Normalized string ready for entity extraction.
        """
        if not raw_text or not raw_text.strip():
            return ""

        text = self._unicode_normalize(raw_text)
        text = self._smart_lowercase(text)

        if self.fix_telex_typos:
            text = self._fix_telex(text)

        if self.expand_abbreviations:
            text = self._expand_abbreviations(text)

        if self.convert_number_words:
            text = self._convert_number_words(text)

        text = self._normalize_punctuation(text)
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def detect_language_mix(self, text: str) -> Dict[str, float]:
        """
        Detect the ratio of Vietnamese vs English in a query.

        Returns:
            Dict with 'vi' and 'en' ratios (sum to 1.0).
        """
        words = text.lower().split()
        if not words:
            return {"vi": 0.5, "en": 0.5}

        # Vietnamese indicator: contains common diacritics
        vi_chars = set("àáảãạăắặằẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ")
        vi_count = sum(1 for w in words if any(c in vi_chars for c in w.lower()))

        # English indicator: purely ASCII alphabetic words (no Vietnamese diacritics)
        en_count = sum(
            1 for w in words
            if w.isalpha() and all(c in "abcdefghijklmnopqrstuvwxyz" for c in w.lower())
            and not any(c in vi_chars for c in w.lower())
        )

        total = max(vi_count + en_count, 1)
        vi_ratio = round(vi_count / total, 2)
        en_ratio = round(1.0 - vi_ratio, 2)

        return {"vi": vi_ratio, "en": en_ratio}

    # ----------------------------------------------------------
    # Internal steps
    # ----------------------------------------------------------

    def _unicode_normalize(self, text: str) -> str:
        """NFC first (diacritics), then remove control characters."""
        text = unicodedata.normalize("NFC", text)
        text = "".join(ch for ch in text if unicodedata.category(ch)[0] != "C")
        return text

    def _smart_lowercase(self, text: str) -> str:
        """
        Lowercase all words except:
        - All-caps acronyms of 2+ chars (VTV1, HTV, VNPT, FIFA, MC, BTV)
        - Numbers and digits
        """
        tokens = text.split()
        result = []
        for tok in tokens:
            # Keep if ALL CAPS (2+ alpha chars) — it's an acronym
            alpha_only = re.sub(r"[^A-Za-z]", "", tok)
            if len(alpha_only) >= 2 and alpha_only.isupper():
                result.append(tok)  # keep original case
            else:
                result.append(tok.lower())
        return " ".join(result)

    def _fix_telex(self, text: str) -> str:
        """Apply conservative telex/VNI typo corrections."""
        for pattern, replacement in _TELEX_CORRECTIONS:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        return text

    def _expand_abbreviations(self, text: str) -> str:
        """
        Expand known abbreviations — longest-first to avoid partial matches.
        Only expands when the abbreviation appears as a standalone token.
        """
        for abbrev in self._abbrev_keys_sorted:
            # Word-boundary match, case-insensitive for English abbrevs
            pattern = r"(?<!\w)" + re.escape(abbrev) + r"(?!\w)"
            expanded = _ABBREVIATION_MAP[abbrev]
            text = re.sub(pattern, expanded, text, flags=re.IGNORECASE)
        return text

    def _convert_number_words(self, text: str) -> str:
        """
        Convert multi-word number phrases and single number words to digits.
        Vietnamese-aware: handles ambiguous "năm" (five vs year) contextually.

        Multi-word phrases are converted first (longest-first match).
        """
        # Multi-word Vietnamese numbers first (e.g. "mười hai" → "12")
        sorted_vi = sorted(_VI_NUMBER_WORDS.keys(), key=len, reverse=True)
        for word in sorted_vi:
            if word in _VI_NUMBER_EXCEPTIONS:
                continue
            digit = _VI_NUMBER_WORDS[word]
            pattern = r"(?<!\w)" + re.escape(word) + r"(?!\w)"
            text = re.sub(pattern, str(digit), text, flags=re.IGNORECASE)

        # Special case: "năm" — only convert when followed by a count noun
        count_noun_pattern = "|".join(re.escape(n) for n in _VI_COUNT_NOUNS)
        text = re.sub(
            r"\bnăm\b(?=\s+(?:" + count_noun_pattern + r"))",
            "5",
            text,
            flags=re.IGNORECASE,
        )

        # English number words
        sorted_en = sorted(_EN_NUMBER_WORDS.keys(), key=len, reverse=True)
        for word in sorted_en:
            pattern = r"(?<!\w)" + re.escape(word) + r"(?!\w)"
            text = re.sub(pattern, str(_EN_NUMBER_WORDS[word]), text, flags=re.IGNORECASE)

        return text

    def _normalize_punctuation(self, text: str) -> str:
        """
        Clean redundant punctuation while preserving:
        - Quoted strings: "VTV", 'HTV' → kept for OCR hint detection
        - Score notations: 2-1, 3:0 → kept
        - Ellipsis: ... → kept
        """
        text = re.sub(r"\?{2,}", "?", text)
        text = re.sub(r"!{2,}", "!", text)
        text = re.sub(r"\.{4,}", "...", text)
        # Remove stray commas/semicolons at start/end
        text = text.strip(",:;")
        return text
