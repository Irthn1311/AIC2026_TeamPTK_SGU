"""Semantic sanity validator for Vietnamese-to-English translation verification."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Any

from .models import SemanticValidationIssue, ValidationResult

VALIDATOR_VERSION: str = "sem_sanity_v1"

# Vietnamese function / grammatical words indicating untranslated sentences
VI_FUNCTION_WORDS = frozenset(
    {
        "là",
        "và",
        "của",
        "những",
        "các",
        "có",
        "trong",
        "được",
        "đã",
        "sẽ",
        "đang",
        "người",
        "này",
        "đó",
        "để",
        "với",
        "sau",
        "chuyển",
        "sang",
        "cảnh",
        "đoạn",
        "phim",
        "một",
    }
)

# Unaccented Vietnamese words for detecting untranslated unaccented text
VI_UNACCENTED_WORDS = frozenset(
    {
        "nguoi",
        "dan",
        "ong",
        "phu",
        "nu",
        "tre",
        "em",
        "dung",
        "ngoi",
        "chay",
        "di",
        "ben",
        "trai",
        "phai",
        "tren",
        "duoi",
        "trong",
        "ngoai",
        "khong",
        "co",
        "la",
        "va",
        "cua",
        "nhung",
        "cac",
        "mot",
        "hai",
        "ba",
        "bon",
        "nam",
        "mau",
        "do",
        "xanh",
        "vang",
        "trang",
        "den",
        "ao",
        "quan",
        "xe",
        "may",
        "canh",
        "doan",
        "phim",
        "khung",
        "hinh",
        "quay",
        "sau",
        "truoc",
        "tiep",
        "theo",
    }
)

# Cardinal English number words mapped to integer values
EN_CARDINAL_WORDS: dict[str, int] = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
    "hundred": 100,
    "thousand": 1000,
}

# English ordinal words
EN_ORDINAL_WORDS: dict[str, int] = {
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "sixth": 6,
    "seventh": 7,
    "eighth": 8,
    "ninth": 9,
    "tenth": 10,
    "twelfth": 12,
    "twentieth": 20,
}


def _strip_diacritics(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", text)
    stripped = "".join(c for c in nfkd if not unicodedata.combining(c))
    return stripped.replace("đ", "d").replace("Đ", "D")


def _has_vietnamese_diacritics(text: str) -> bool:
    vi_chars = set("àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ")
    return any(c in vi_chars for c in text.lower())


# Color mappings for clause-entity attribute resolution
VN_COLOR_MAP: list[tuple[str, str]] = [
    ("xanh lá", "green"),
    ("xanh lục", "green"),
    ("xanh đậm", "dark_blue"),
    ("xanh thẫm", "dark_blue"),
    ("xanh dương", "blue"),
    ("xanh da trời", "blue"),
    ("hồng tím", "purple"),
    ("đỏ", "red"),
    ("vàng", "yellow"),
    ("trắng", "white"),
    ("đen", "black"),
    ("hồng", "pink"),
    ("tím", "purple"),
    ("cam", "orange"),
    ("xanh", "green_or_blue"),
]

EN_COLOR_MAP: list[tuple[str, str]] = [
    ("dark blue", "dark_blue"),
    ("navy", "dark_blue"),
    ("pink-purple", "purple"),
    ("green", "green"),
    ("red", "red"),
    ("blue", "blue"),
    ("yellow", "yellow"),
    ("white", "white"),
    ("black", "black"),
    ("pink", "pink"),
    ("purple", "purple"),
    ("violet", "purple"),
    ("orange", "orange"),
]


def _color_matches(vn_color: str, en_color: str) -> bool:
    if vn_color == en_color:
        return True
    if vn_color == "green_or_blue" and en_color in ("green", "blue"):
        return True
    return False


VN_ENTITIES: list[tuple[str, str]] = [
    ("man", r"\b(?:người\s+đàn\s+ông|đàn\s+ông|chàng\s+trai|bé\s+trai)\b"),
    ("woman", r"\b(?:phụ\s+nữ|cô\s+gái|bé\s+gái)\b"),
    ("employee", r"\b(?:nhân\s+viên|bồi\s+bàn|đầu\s+bếp)\b"),
    ("guest", r"\b(?:du\s+khách|khách)\b"),
    ("child", r"\b(?:trẻ\s+em|đứa\s+trẻ|em\s+bé)\b"),
    ("person", r"\b(?:người)\b"),
    ("dog", r"\b(?:con\s+chó|đàn\s+chó|chó)\b"),
    ("lion", r"\b(?:đàn\s+sư\s+tử|sư\s+tử)\b"),
    ("cat", r"\b(?:con\s+mèo|mèo)\b"),
    ("horse", r"\b(?:con\s+ngựa|ngựa)\b"),
    ("hat", r"\b(?:nón\s+lá|nón\s+bảo\s+hiểm|mũ\s+bảo\s+hiểm|nón|mũ)\b"),
    ("shirt", r"\b(?:áo\s+sơ\s+mi|sơ\s+mi|áo\s+thun|áo\s+phông|áo\s+vest|vest|áo)\b"),
    ("headscarf", r"\b(?:khăn\s+trùm\s+đầu|khăn)\b"),
    ("pepper", r"\b(?:ớt)\b"),
    ("map", r"\b(?:bản\s+đồ)\b"),
    ("dam", r"\b(?:đập\s+nước|đập)\b"),
    ("gemstone", r"\b(?:khối\s+đá\s+quý|đá\s+quý)\b"),
    ("squid", r"\b(?:mực)\b"),
    ("pea", r"\b(?:đậu\s+hà\s+lan|đậu)\b"),
    ("onion", r"\b(?:hành\s+tây|hành)\b"),
    ("chair", r"\b(?:chiếc\s+ghế|cái\s+ghế|ghế)\b"),
    ("glasses", r"\b(?:cặp\s+kính|mắt\s+kính|kính)\b"),
]

EN_ENTITIES: list[tuple[str, str]] = [
    ("man", r"\b(?:man|men|boy|boys|gentleman)\b"),
    ("woman", r"\b(?:woman|women|girl|girls|lady)\b"),
    ("employee", r"\b(?:employee|employees|staff|worker|workers|chef|waiter)\b"),
    ("guest", r"\b(?:guest|guests|visitor|visitors)\b"),
    ("child", r"\b(?:child|children|kid|kids|baby|babies)\b"),
    ("person", r"\b(?:person|people)\b"),
    ("dog", r"\b(?:dog|dogs)\b"),
    ("lion", r"\b(?:lion|lions|pride\s+of\s+lions)\b"),
    ("cat", r"\b(?:cat|cats)\b"),
    ("horse", r"\b(?:horse|horses)\b"),
    ("hat", r"\b(?:hat|hats|cap|caps|helmet|helmets)\b"),
    ("shirt", r"\b(?:shirt|shirts|suit|suits|vest|vests|jacket|jackets|attire)\b"),
    ("headscarf", r"\b(?:headscarf|scarf)\b"),
    ("pepper", r"\b(?:pepper|peppers|chili|chilies)\b"),
    ("map", r"\b(?:map|maps)\b"),
    ("dam", r"\b(?:dam|dams)\b"),
    ("gemstone", r"\b(?:gemstone|gemstones)\b"),
    ("squid", r"\b(?:squid|squids)\b"),
    ("pea", r"\b(?:pea|peas)\b"),
    ("onion", r"\b(?:onion|onions)\b"),
    ("chair", r"\b(?:chair|chairs|seat|seats)\b"),
    ("glasses", r"\b(?:glasses|spectacles|eyewear)\b"),
]

ENTITY_GROUPS: dict[str, str] = {
    "man": "person",
    "woman": "person",
    "employee": "person",
    "guest": "person",
    "child": "person",
    "person": "person",
}

COLOR_BEARING_ENTITIES: frozenset[str] = frozenset({"hat", "shirt", "headscarf", "pepper"})


@dataclass
class _ClauseEntityMention:
    clause_idx: int
    entity_type: str
    raw_text: str
    count: int | None = None
    color: str | None = None
    spatial: str | None = None
    negated: bool = False
    has_comparator: bool = False
    comparator_type: str | None = None
    start_pos: int = 0


class SemanticSanityValidator:
    """Validates structural and semantic fidelity of Vietnamese-to-English translations.

    Separates issues strictly into two tiers:
    - ERROR: Definite failure that compromises retrieval (triggers fallback).
    - WARNING: Stylistic variation, proper noun, or implicit marker.
    """

    version: str = VALIDATOR_VERSION

    def validate(self, source_text: str, target_text: str) -> ValidationResult:
        issues: list[SemanticValidationIssue] = []
        extracted_entities: dict[str, Any] = {}

        src_clean = " ".join((source_text or "").split())
        tgt_clean = " ".join((target_text or "").split())
        src_lower = src_clean.lower()
        tgt_lower = tgt_clean.lower()

        # 1. Degenerate Output Check (ERROR)
        if not tgt_clean:
            issues.append(
                SemanticValidationIssue(
                    severity="ERROR",
                    code="degenerate_output",
                    message="Target translation is empty or whitespace only",
                )
            )
            return ValidationResult(is_valid=False, issues=tuple(issues), extracted_entities={})

        tgt_words = tgt_lower.split()
        if len(tgt_words) >= 4:
            # Check for repetitive loops e.g. "dam dam dam dam"
            repeated = False
            for i in range(len(tgt_words) - 3):
                if tgt_words[i] == tgt_words[i + 1] == tgt_words[i + 2] == tgt_words[i + 3]:
                    repeated = True
                    break
            if not repeated and len(tgt_words) >= 8:
                for i in range(len(tgt_words) - 7):
                    even_match = (
                        tgt_words[i] == tgt_words[i + 2] == tgt_words[i + 4] == tgt_words[i + 6]
                    )
                    odd_match = (
                        tgt_words[i + 1] == tgt_words[i + 3] == tgt_words[i + 5] == tgt_words[i + 7]
                    )
                    if even_match and odd_match:
                        repeated = True
                        break
            if repeated:
                issues.append(
                    SemanticValidationIssue(
                        severity="ERROR",
                        code="degenerate_output",
                        message="Target translation contains degenerate repetitive token loop",
                    )
                )

        # 2. Negation Preservation (ERROR)
        self._validate_negation(src_lower, tgt_lower, issues)

        # 3. Spatial / Directional Polarity & Direction Drop (ERROR)
        self._validate_spatial_and_direction(src_lower, tgt_lower, issues)

        # 4. Entity Count & Cardinal vs Ordinal (ERROR)
        src_numbers = self._extract_vietnamese_numbers(src_clean)
        tgt_numbers = self._extract_english_numbers(tgt_clean)
        tgt_numbers_multiset = self._extract_english_numbers_multiset(tgt_clean)
        tgt_ordinals = self._extract_english_ordinals(tgt_clean)
        extracted_entities["src_numbers"] = src_numbers
        extracted_entities["tgt_numbers"] = tgt_numbers

        src_strict_vals = [
            val for val, _original_form, is_strict, _is_paired in src_numbers if is_strict
        ]
        src_counts = Counter(src_strict_vals)
        tgt_counts = Counter(tgt_numbers_multiset)

        for val, req_count in src_counts.items():
            actual_count = tgt_counts.get(val, 0)
            if actual_count < req_count:
                if val in tgt_ordinals:
                    issues.append(
                        SemanticValidationIssue(
                            severity="ERROR",
                            code="cardinal_ordinal_mismatch",
                            message=(
                                f"Cardinal count {val} was translated as an ordinal in target "
                                "(e.g. second/third instead of two/three)"
                            ),
                        )
                    )
                else:
                    issues.append(
                        SemanticValidationIssue(
                            severity="ERROR",
                            code="count_dropped",
                            message=f"Count {val} dropped in translation",
                        )
                    )

        # 5. Comparator Preservation & Polarity (ERROR)
        self._validate_comparators(src_lower, tgt_lower, issues)

        # 6. Temporal Sequence Ordering (ERROR)
        self._validate_temporal_sequence(src_lower, tgt_lower, issues)

        # 7. Perspective / Viewpoint Integrity (ERROR)
        self._validate_perspective(src_lower, tgt_lower, issues)

        # 8. Clause-Entity Binding & Multi-Attribute Integrity (ERROR)
        self._validate_clause_entities(src_clean, tgt_clean, issues)

        # 9. Untranslated Vietnamese (Accented & Unaccented)
        self._validate_untranslated_vietnamese(src_clean, tgt_clean, issues)

        is_valid = not any(issue.severity == "ERROR" for issue in issues)
        return ValidationResult(
            is_valid=is_valid,
            issues=tuple(issues),
            extracted_entities=extracted_entities,
        )

    def _validate_negation(
        self,
        src_lower: str,
        tgt_lower: str,
        issues: list[SemanticValidationIssue],
    ) -> None:
        """Verify negation polarity is preserved without dropping or introducing negation."""
        src_clean_neg = re.sub(r"\bkhông\s+khí\b", "", src_lower)
        src_clean_neg = re.sub(r"\bkhông\s+những\b.*?\bmà\s+còn\b", "", src_clean_neg)
        src_clean_neg = re.sub(r"\bsố\s+không\b", "", src_clean_neg)

        src_negs = re.findall(
            r"\b(không\s+(?!khí\b|những\b|gian\b)[a-zà-ỹ]+|chưa\s+[a-zà-ỹ]+|chẳng\s+[a-zà-ỹ]+|"
            r"đừng\s+[a-zà-ỹ]+|vắng bóng|không một ai)\b",
            src_clean_neg,
        )
        has_src_negation = bool(src_negs)

        tgt_clean_neg = re.sub(r"\bnot\s+only\b.*?\bbut\s+(?:also\b)?", "", tgt_lower)
        tgt_negs = re.findall(
            r"\b(no|not|neither|nor|none|nobody|nothing|nowhere|never|without)\b|"
            r"\b\w+n't\b|\bcannot\b",
            tgt_clean_neg,
        )
        has_tgt_negation = bool(tgt_negs)

        if has_src_negation and not has_tgt_negation:
            issues.append(
                SemanticValidationIssue(
                    severity="ERROR",
                    code="negation_dropped",
                    message=(
                        "Negation dropped: source specifies absence/negation but target is positive"
                    ),
                )
            )
        elif not has_src_negation and has_tgt_negation:
            issues.append(
                SemanticValidationIssue(
                    severity="ERROR",
                    code="negation_introduced",
                    message=(
                        "Negation introduced: source is affirmative but target expresses negation"
                    ),
                )
            )
        elif len(src_negs) >= 2 and len(tgt_negs) < len(src_negs):
            issues.append(
                SemanticValidationIssue(
                    severity="ERROR",
                    code="negation_dropped",
                    message="Negation dropped for one or more clauses in translation",
                )
            )

    def _validate_spatial_and_direction(
        self,
        src_lower: str,
        tgt_lower: str,
        issues: list[SemanticValidationIssue],
    ) -> None:
        """Validate left/right/top/bottom spatial polarity and direction preservation."""
        has_src_left = bool(re.search(r"\b(bên trái|phía trái|ở trái|tay trái)\b", src_lower))
        has_src_right = bool(re.search(r"\b(bên phải|phía phải|ở phải|tay phải)\b", src_lower))
        has_src_top = bool(re.search(r"\b(bên trên|phía trên|ở trên)\b", src_lower))
        has_src_bottom = bool(re.search(r"\b(bên dưới|phía dưới|ở dưới)\b", src_lower))

        has_tgt_left = bool(re.search(r"\bleft\b", tgt_lower))
        has_tgt_right = bool(re.search(r"\bright\b", tgt_lower))
        has_tgt_top = bool(re.search(r"\b(top|above|upper|up)\b", tgt_lower))
        has_tgt_bottom = bool(re.search(r"\b(bottom|below|lower|down)\b", tgt_lower))

        # 1. Drops when multiple or single directions exist
        if has_src_left and not has_tgt_left:
            issues.append(
                SemanticValidationIssue(
                    severity="ERROR",
                    code="direction_dropped",
                    message="Directional marker 'left' dropped in translation",
                )
            )
        if has_src_right and not has_tgt_right:
            issues.append(
                SemanticValidationIssue(
                    severity="ERROR",
                    code="direction_dropped",
                    message="Directional marker 'right' dropped in translation",
                )
            )
        if has_src_top and not has_tgt_top:
            issues.append(
                SemanticValidationIssue(
                    severity="ERROR",
                    code="direction_dropped",
                    message="Directional marker 'top/above' dropped in translation",
                )
            )
        if has_src_bottom and not has_tgt_bottom:
            issues.append(
                SemanticValidationIssue(
                    severity="ERROR",
                    code="direction_dropped",
                    message="Directional marker 'bottom/below' dropped in translation",
                )
            )

        # 2. Inversions
        if has_src_left and not has_src_right and has_tgt_right and not has_tgt_left:
            issues.append(
                SemanticValidationIssue(
                    severity="ERROR",
                    code="spatial_inversion",
                    message="Spatial polarity inverted: source left translated as right",
                )
            )
        elif has_src_right and not has_src_left and has_tgt_left and not has_tgt_right:
            issues.append(
                SemanticValidationIssue(
                    severity="ERROR",
                    code="spatial_inversion",
                    message="Spatial polarity inverted: source right translated as left",
                )
            )

        if has_src_top and not has_src_bottom and has_tgt_bottom and not has_tgt_top:
            issues.append(
                SemanticValidationIssue(
                    severity="ERROR",
                    code="spatial_inversion",
                    message="Spatial polarity inverted: source top translated as bottom",
                )
            )
        elif has_src_bottom and not has_src_top and has_tgt_top and not has_tgt_bottom:
            issues.append(
                SemanticValidationIssue(
                    severity="ERROR",
                    code="spatial_inversion",
                    message="Spatial polarity inverted: source bottom translated as top",
                )
            )

    def _validate_comparators(
        self,
        src_lower: str,
        tgt_lower: str,
        issues: list[SemanticValidationIssue],
    ) -> None:
        """Validate comparators requiring bound quantities to avoid false matches."""
        num_pat = (
            r"(\d+|một|hai|ba|bốn|năm|sáu|bảy|tám|chín|mười|mốt|lăm|chục|trăm|nghìn|ngàn|triệu|tỷ)"
        )
        src_at_least = re.search(
            r"\b(?:ít nhất|tối thiểu|không dưới)\s+" + num_pat + r"\b", src_lower
        )
        src_more_than = re.search(r"\b(?:nhiều hơn|hơn|vượt quá)\s+" + num_pat + r"\b", src_lower)
        src_at_most = re.search(
            r"\b(?:không quá|tối đa|nhiều nhất)\s+" + num_pat + r"\b", src_lower
        )
        src_less_than = re.search(r"\b(?:ít hơn|thấp hơn|dưới)\s+" + num_pat + r"\b", src_lower)

        en_num_pat = (
            r"(\d+|one|two|three|four|five|six|seven|eight|nine|ten|"
            r"twelve|twenty|thirty|hundred|a|an)"
        )
        tgt_at_least = re.search(
            r"\b(?:at least|no less than|no fewer than|minimum of)\s+" + en_num_pat + r"\b",
            tgt_lower,
        )
        tgt_more_than = re.search(
            r"\b(?:more than|greater than|over|in excess of|exceeding)\s+" + en_num_pat + r"\b",
            tgt_lower,
        )
        tgt_at_most = re.search(
            r"\b(?:at most|no more than|maximum of)\s+" + en_num_pat + r"\b", tgt_lower
        ) or re.search(r"\bup to\s+" + en_num_pat + r"\b", tgt_lower)
        tgt_less_than = re.search(
            r"\b(?:less than|fewer than|under|below)\s+" + en_num_pat + r"\b", tgt_lower
        )

        def _get_num_val(m: re.Match[str] | None, is_vi: bool) -> int | None:
            if not m:
                return None
            val_str = m.group(1).lower()
            if val_str.isdigit():
                return int(val_str)
            if is_vi:
                vi_map = {
                    "một": 1,
                    "mốt": 1,
                    "hai": 2,
                    "ba": 3,
                    "bốn": 4,
                    "tư": 4,
                    "năm": 5,
                    "lăm": 5,
                    "sáu": 6,
                    "bảy": 7,
                    "tám": 8,
                    "chín": 9,
                    "mười": 10,
                }
                return vi_map.get(val_str)
            return EN_CARDINAL_WORDS.get(val_str, 1 if val_str in ("a", "an") else None)

        if src_at_least:
            if tgt_more_than or tgt_less_than or tgt_at_most:
                issues.append(
                    SemanticValidationIssue(
                        severity="ERROR",
                        code="comparator_polarity_inversion",
                        message=(
                            "Comparator polarity inverted: 'ít nhất' (at least) "
                            "translated as greater or less"
                        ),
                    )
                )
            elif not tgt_at_least:
                issues.append(
                    SemanticValidationIssue(
                        severity="ERROR",
                        code="comparator_dropped",
                        message="Comparative quantifier 'ít nhất' (at least) was dropped",
                    )
                )
            else:
                v_val = _get_num_val(src_at_least, is_vi=True)
                e_val = _get_num_val(tgt_at_least, is_vi=False)
                if v_val is not None and e_val is not None and v_val != e_val:
                    issues.append(
                        SemanticValidationIssue(
                            severity="ERROR",
                            code="comparator_mismatch",
                            message=(
                                f"Comparator bound to different number in target: "
                                f"{v_val} vs {e_val}"
                            ),
                        )
                    )
        elif src_more_than:
            if tgt_at_least or tgt_at_most or tgt_less_than:
                issues.append(
                    SemanticValidationIssue(
                        severity="ERROR",
                        code="comparator_polarity_inversion",
                        message=(
                            "Comparator polarity inverted: 'hơn/nhiều hơn' (more than) "
                            "translated as at least or at most"
                        ),
                    )
                )
            elif not tgt_more_than:
                issues.append(
                    SemanticValidationIssue(
                        severity="ERROR",
                        code="comparator_dropped",
                        message="Comparative quantifier 'hơn/nhiều hơn' was dropped",
                    )
                )
            else:
                v_val = _get_num_val(src_more_than, is_vi=True)
                e_val = _get_num_val(tgt_more_than, is_vi=False)
                if v_val is not None and e_val is not None and v_val != e_val:
                    issues.append(
                        SemanticValidationIssue(
                            severity="ERROR",
                            code="comparator_mismatch",
                            message=(
                                f"Comparator bound to different number in target: "
                                f"{v_val} vs {e_val}"
                            ),
                        )
                    )
        elif src_at_most:
            if tgt_more_than or tgt_at_least:
                issues.append(
                    SemanticValidationIssue(
                        severity="ERROR",
                        code="comparator_polarity_inversion",
                        message=(
                            "Comparator polarity inverted: 'không quá/tối đa' (at most) "
                            "translated as more than or at least"
                        ),
                    )
                )
            elif not tgt_at_most:
                issues.append(
                    SemanticValidationIssue(
                        severity="ERROR",
                        code="comparator_dropped",
                        message="Comparative quantifier 'không quá/tối đa' was dropped",
                    )
                )
            else:
                v_val = _get_num_val(src_at_most, is_vi=True)
                e_val = _get_num_val(tgt_at_most, is_vi=False)
                if v_val is not None and e_val is not None and v_val != e_val:
                    issues.append(
                        SemanticValidationIssue(
                            severity="ERROR",
                            code="comparator_mismatch",
                            message=(
                                f"Comparator bound to different number in target: "
                                f"{v_val} vs {e_val}"
                            ),
                        )
                    )
        elif src_less_than:
            if tgt_more_than or tgt_at_least or tgt_at_most:
                issues.append(
                    SemanticValidationIssue(
                        severity="ERROR",
                        code="comparator_polarity_inversion",
                        message=(
                            "Comparator polarity inverted: 'ít hơn/dưới' (less than) "
                            "translated as more than"
                        ),
                    )
                )
            elif not tgt_less_than:
                issues.append(
                    SemanticValidationIssue(
                        severity="ERROR",
                        code="comparator_dropped",
                        message="Comparative quantifier 'ít hơn/dưới' was dropped",
                    )
                )
            else:
                v_val = _get_num_val(src_less_than, is_vi=True)
                e_val = _get_num_val(tgt_less_than, is_vi=False)
                if v_val is not None and e_val is not None and v_val != e_val:
                    issues.append(
                        SemanticValidationIssue(
                            severity="ERROR",
                            code="comparator_mismatch",
                            message=(
                                f"Comparator bound to different number in target: "
                                f"{v_val} vs {e_val}"
                            ),
                        )
                    )

    def _validate_temporal_sequence(
        self,
        src_lower: str,
        tgt_lower: str,
        issues: list[SemanticValidationIssue],
    ) -> None:
        idx_src_begin = self._find_first_pattern(src_lower, [r"\bbắt đầu\b", r"\bkhởi đầu\b"])
        idx_src_next = self._find_first_pattern(
            src_lower, [r"\bsau đó\b", r"\btiếp theo\b", r"\btiếp đến\b", r"\bkế tiếp\b"]
        )

        if idx_src_begin is not None and idx_src_next is not None and idx_src_begin < idx_src_next:
            idx_tgt_begin = self._find_first_pattern(
                tgt_lower, [r"\bbegin", r"\bstart", r"\binitially\b", r"\bfirst\b"]
            )
            idx_tgt_next = self._find_first_pattern(
                tgt_lower,
                [r"\bthen\b", r"\bafter\b", r"\bsubsequently\b", r"\bnext\b", r"\bfollowed by\b"],
            )
            if idx_tgt_begin is not None and idx_tgt_next is not None:
                if idx_tgt_begin > idx_tgt_next:
                    issues.append(
                        SemanticValidationIssue(
                            severity="ERROR",
                            code="temporal_inversion",
                            message="Temporal sequence markers inverted in translation",
                        )
                    )
            elif idx_tgt_begin is None and idx_tgt_next is None:
                issues.append(
                    SemanticValidationIssue(
                        severity="WARNING",
                        code="implicit_temporal_order",
                        message="Temporal sequence translated implicitly without explicit words",
                    )
                )

    def _validate_perspective(
        self,
        src_lower: str,
        tgt_lower: str,
        issues: list[SemanticValidationIssue],
    ) -> None:
        """Validate perspective/viewpoint markers like 'từ trên cao' (aerial/from above)."""
        has_src_high = bool(
            re.search(r"\b(từ trên cao|từ trên không|nhìn từ trên cao)\b", src_lower)
        )
        if has_src_high:
            has_tgt_high = bool(
                re.search(
                    r"\b(from above|aerial|overhead|top-down|high angle|bird's eye)\b",
                    tgt_lower,
                )
            )
            if not has_tgt_high:
                issues.append(
                    SemanticValidationIssue(
                        severity="ERROR",
                        code="perspective_dropped",
                        message=(
                            "Perspective marker 'từ trên cao' (aerial/from above) "
                            "was dropped in translation"
                        ),
                    )
                )

    def _extract_clause_entities_vi(self, text: str) -> list[_ClauseEntityMention]:
        mentions: list[_ClauseEntityMention] = []
        clause_pat = r"[,;.]|\b(?:và|sau đó|tiếp theo|tiếp đến|kế tiếp|trong khi|nhưng|rồi)\b"
        clauses = []
        last_end = 0
        for m in re.finditer(clause_pat, text, flags=re.IGNORECASE):
            span = text[last_end : m.start()].strip()
            if span:
                clauses.append((span, last_end))
            last_end = m.end()
        tail = text[last_end:].strip()
        if tail:
            clauses.append((tail, last_end))

        matched_ranges: list[tuple[int, int]] = []

        for c_idx, (clause, c_offset) in enumerate(clauses):
            c_lower = clause.lower()
            clause_negated = bool(re.search(r"\b(không|chưa|chẳng|đừng|vắng bóng)\b", c_lower))

            for etype, epat in VN_ENTITIES:
                for ematch in re.finditer(epat, c_lower):
                    global_start = c_offset + ematch.start()
                    global_end = c_offset + ematch.end()
                    if any(s <= global_start and global_end <= e for s, e in matched_ranges):
                        continue
                    matched_ranges.append((global_start, global_end))

                    # Count: preceding in clause
                    prefix = c_lower[max(0, ematch.start() - 25) : ematch.start()].strip()
                    count = None
                    m_dig = re.search(r"\b(\d+)\s*(?:con|người|chiếc|cái|khối|đàn)?$", prefix)
                    if m_dig:
                        count = int(m_dig.group(1))
                    else:
                        extracted_nums = self._extract_vietnamese_numbers(prefix)
                        if extracted_nums:
                            # Do not attribute paired body parts (is_paired=True e.g. hai tay)
                            # as the count for non-body entities
                            valid_nums = [n for n in extracted_nums if not n[3]]
                            if valid_nums:
                                count = valid_nums[-1][0]

                    # Color: only bind to color-bearing entities
                    color = None
                    if etype in COLOR_BEARING_ENTITIES:
                        window = c_lower[
                            max(0, ematch.start() - 15) : min(len(c_lower), ematch.end() + 20)
                        ]
                        for c_name, c_canon in VN_COLOR_MAP:
                            if re.search(r"\b" + re.escape(c_name) + r"\b", window):
                                color = c_canon
                                break

                    # Spatial: in clause or near entity
                    spatial = None
                    if re.search(r"\b(?:bên\s+trái|phía\s+trái|ở\s+trái|tay\s+trái)\b", c_lower):
                        spatial = "left"
                    elif re.search(r"\b(?:bên\s+phải|phía\s+phải|ở\s+phải|tay\s+phải)\b", c_lower):
                        spatial = "right"
                    elif re.search(r"\b(?:bên\s+trên|phía\s+trên|ở\s+trên)\b", c_lower):
                        spatial = "top"
                    elif re.search(r"\b(?:bên\s+dưới|phía\s+dưới|ở\s+dưới)\b", c_lower):
                        spatial = "bottom"

                    # Comparator
                    has_comparator = False
                    comparator_type = None
                    comp_m = re.search(
                        r"\b(hơn|nhiều hơn|vượt quá|ít nhất|tối thiểu|không dưới|"
                        r"không quá|tối đa|nhiều nhất|ít hơn|thấp hơn|dưới)\b",
                        prefix,
                    )
                    if comp_m:
                        c_w = comp_m.group(1)
                        has_comparator = True
                        if c_w in ("hơn", "nhiều hơn", "vượt quá"):
                            comparator_type = "more_than"
                        elif c_w in ("ít nhất", "tối thiểu", "không dưới"):
                            comparator_type = "at_least"
                        elif c_w in ("không quá", "tối đa", "nhiều nhất"):
                            comparator_type = "at_most"
                        elif c_w in ("ít hơn", "thấp hơn", "dưới"):
                            comparator_type = "less_than"

                    # Predicate / clause negation
                    pred_window = c_lower[max(0, ematch.start() - 25) : ematch.end()]
                    mention_negated = clause_negated or bool(
                        re.search(r"\b(không|chưa|chẳng|đừng|vắng bóng)\b", pred_window)
                    )

                    mentions.append(
                        _ClauseEntityMention(
                            clause_idx=c_idx,
                            entity_type=etype,
                            raw_text=ematch.group(0),
                            count=count,
                            color=color,
                            spatial=spatial,
                            negated=mention_negated,
                            has_comparator=has_comparator,
                            comparator_type=comparator_type,
                            start_pos=global_start,
                        )
                    )

        mentions.sort(key=lambda m: m.start_pos)
        return mentions

    def _extract_clause_entities_en(self, text: str) -> list[_ClauseEntityMention]:
        mentions: list[_ClauseEntityMention] = []
        clause_pat = r"[,;.]|\b(?:and|then|after that|next|subsequently|while|but)\b"
        clauses = []
        last_end = 0
        for m in re.finditer(clause_pat, text, flags=re.IGNORECASE):
            span = text[last_end : m.start()].strip()
            if span:
                clauses.append((span, last_end))
            last_end = m.end()
        tail = text[last_end:].strip()
        if tail:
            clauses.append((tail, last_end))

        matched_ranges: list[tuple[int, int]] = []

        for c_idx, (clause, c_offset) in enumerate(clauses):
            c_lower = clause.lower()
            clause_negated = bool(
                re.search(
                    r"\b(not|no|neither|without|never|nobody|nothing|wears\s+no|\w+n't|cannot)\b",
                    c_lower,
                )
            )

            for etype, epat in EN_ENTITIES:
                for ematch in re.finditer(epat, c_lower):
                    global_start = c_offset + ematch.start()
                    global_end = c_offset + ematch.end()
                    if any(s <= global_start and global_end <= e for s, e in matched_ranges):
                        continue
                    matched_ranges.append((global_start, global_end))

                    # Count
                    prefix = c_lower[max(0, ematch.start() - 25) : ematch.start()].strip()
                    count = None
                    m_dig = re.search(r"\b(\d+)\s*$", prefix)
                    if m_dig:
                        count = int(m_dig.group(1))
                    else:
                        extracted_en = self._extract_english_numbers(prefix)
                        if extracted_en:
                            count = max(extracted_en)
                    if count is None:
                        if re.search(r"\b(?:only\s+one|a\s+single|single)\s*$", prefix):
                            count = 1
                        elif re.search(r"\b(?:only\s+one\b.*?\b(?:wears?|has?|with)?)\b", c_lower):
                            count = 1

                    # Color: only bind to color-bearing entities
                    color = None
                    if etype in COLOR_BEARING_ENTITIES:
                        window = c_lower[
                            max(0, ematch.start() - 20) : min(len(c_lower), ematch.end() + 10)
                        ]
                        for c_name, c_canon in EN_COLOR_MAP:
                            if re.search(r"\b" + re.escape(c_name) + r"\b", window):
                                color = c_canon
                                break

                    # Spatial
                    spatial = None
                    if re.search(r"\b(?:left|on the left)\b", c_lower):
                        spatial = "left"
                    elif re.search(r"\b(?:right|on the right)\b", c_lower):
                        spatial = "right"
                    elif re.search(r"\b(?:top|above|upper|on top)\b", c_lower):
                        spatial = "top"
                    elif re.search(r"\b(?:bottom|below|lower|at the bottom)\b", c_lower):
                        spatial = "bottom"

                    # Comparator
                    has_comparator = False
                    comparator_type = None
                    comp_m = re.search(
                        r"\b(more than|greater than|over|at least|no less than|"
                        r"no fewer than|minimum of|at most|no more than|maximum of|"
                        r"up to|less than|fewer than|under|below)\b",
                        prefix,
                    )
                    if comp_m:
                        c_w = comp_m.group(1)
                        has_comparator = True
                        if c_w in ("more than", "greater than", "over"):
                            comparator_type = "more_than"
                        elif c_w in ("at least", "no less than", "no fewer than", "minimum of"):
                            comparator_type = "at_least"
                        elif c_w in ("at most", "no more than", "maximum of", "up to"):
                            comparator_type = "at_most"
                        elif c_w in ("less than", "fewer than", "under", "below"):
                            comparator_type = "less_than"

                    # Predicate / clause negation
                    pred_window = c_lower[max(0, ematch.start() - 25) : ematch.end()]
                    mention_negated = clause_negated or bool(
                        re.search(
                            r"\b(not|no|neither|without|never|nobody|nothing|wears\s+no|\w+n't|cannot)\b",
                            pred_window,
                        )
                    )

                    mentions.append(
                        _ClauseEntityMention(
                            clause_idx=c_idx,
                            entity_type=etype,
                            raw_text=ematch.group(0),
                            count=count,
                            color=color,
                            spatial=spatial,
                            negated=mention_negated,
                            has_comparator=has_comparator,
                            comparator_type=comparator_type,
                            start_pos=global_start,
                        )
                    )

        mentions.sort(key=lambda m: m.start_pos)
        return mentions

    def _validate_clause_entities(
        self,
        src_clean: str,
        tgt_clean: str,
        issues: list[SemanticValidationIssue],
    ) -> None:
        """Validate clause/entity representations using mention-level bipartite alignment.

        Prevents global over-rejection while catching count swaps, dropped numbers,
        color drops/swaps, spatial swaps, and temporal sequence entity order.
        """
        v_mentions = self._extract_clause_entities_vi(src_clean)
        e_mentions = self._extract_clause_entities_en(tgt_clean)

        # 0. Initial scene drop check (scene_dropped)
        m_start = re.search(r"\b(?:bắt đầu|khởi đầu)\b", src_clean.lower())
        if m_start:
            m_trans = re.search(
                r"\b(?:sau đó|tiếp theo|tiếp đến|kế tiếp|rồi|trong khi)\b", src_clean.lower()
            )
            trans_pos = m_trans.start() if m_trans else len(src_clean)
            init_v_mentions = [m for m in v_mentions if m.start_pos < trans_pos]
            if init_v_mentions:
                init_types = {m.entity_type for m in init_v_mentions}
                init_broad = {
                    ENTITY_GROUPS.get(m.entity_type, m.entity_type) for m in init_v_mentions
                }
                tgt_types = {m.entity_type for m in e_mentions}
                tgt_broad = {ENTITY_GROUPS.get(m.entity_type, m.entity_type) for m in e_mentions}
                if not (init_types & tgt_types or init_broad & tgt_broad):
                    issues.append(
                        SemanticValidationIssue(
                            severity="ERROR",
                            code="scene_dropped",
                            message="Initial scene entity was dropped in translation",
                        )
                    )

        # Score candidate pairs (v, e) for bipartite alignment
        scored_pairs: list[tuple[float, int, int]] = []
        for i, v in enumerate(v_mentions):
            for j, e in enumerate(e_mentions):
                # Entity type compatibility
                if v.entity_type == e.entity_type:
                    score = 60.0
                elif ENTITY_GROUPS.get(v.entity_type) == ENTITY_GROUPS.get(e.entity_type):
                    score = 20.0
                else:
                    continue  # Incompatible entity category

                # Color compatibility
                if v.color is not None and e.color is not None:
                    if _color_matches(v.color, e.color):
                        score += 30.0
                    else:
                        score -= 30.0
                elif v.color is not None or e.color is not None:
                    score -= 5.0

                # Count compatibility
                if v.count is not None and e.count is not None:
                    if v.count == e.count:
                        score += 10.0
                    else:
                        score -= 10.0

                # Spatial compatibility
                if v.spatial is not None and e.spatial is not None:
                    if v.spatial == e.spatial:
                        score += 15.0
                    else:
                        score -= 10.0

                # Sequence alignment bonus
                score -= abs(v.clause_idx - e.clause_idx) * 2.0
                scored_pairs.append((score, i, j))

        scored_pairs.sort(key=lambda x: x[0], reverse=True)
        matched_v: set[int] = set()
        matched_e: set[int] = set()
        matched_pairs: list[tuple[_ClauseEntityMention, _ClauseEntityMention]] = []

        for score, i, j in scored_pairs:
            if i not in matched_v and j not in matched_e and score > 0:
                matched_v.add(i)
                matched_e.add(j)
                matched_pairs.append((v_mentions[i], e_mentions[j]))

        # 1. Count-entity swap across pairs (ERROR)
        for p1_idx in range(len(matched_pairs)):
            for p2_idx in range(p1_idx + 1, len(matched_pairs)):
                v1, e1 = matched_pairs[p1_idx]
                v2, e2 = matched_pairs[p2_idx]
                if (
                    v1.count is not None
                    and v2.count is not None
                    and e1.count is not None
                    and e2.count is not None
                    and v1.count != v2.count
                ):
                    if e1.count == v2.count and e2.count == v1.count:
                        issues.append(
                            SemanticValidationIssue(
                                severity="ERROR",
                                code="count_entity_swap",
                                message=(
                                    f"Count swapped across entities: {v1.entity_type} has "
                                    f"{v1.count} in source but {e1.count} in target; "
                                    f"{v2.entity_type} has {v2.count} vs {e2.count}"
                                ),
                            )
                        )

        # 2. Dropped count on an entity (ERROR)
        for v, e in matched_pairs:
            if v.count is not None and v.count >= 2:
                if e.count != v.count:
                    src_vals = [
                        val
                        for val, _, is_strict, _ in self._extract_vietnamese_numbers(src_clean)
                        if is_strict
                    ]
                    tgt_vals = self._extract_english_numbers_multiset(tgt_clean)
                    if Counter(tgt_vals).get(v.count, 0) < Counter(src_vals).get(v.count, 0):
                        if not any(iss.code == "count_dropped" for iss in issues):
                            issues.append(
                                SemanticValidationIssue(
                                    severity="ERROR",
                                    code="count_dropped",
                                    message=(
                                        f"Count {v.count} for entity '{v.entity_type}' "
                                        "was dropped in translation"
                                    ),
                                )
                            )

        # 3. Entity color binding & drop/mismatch (ERROR)
        for v in v_mentions:
            if v.color is not None:
                pair = next((p for p in matched_pairs if p[0] is v), None)
                if pair:
                    _, e = pair
                    if e.color is not None:
                        if not _color_matches(v.color, e.color):
                            issues.append(
                                SemanticValidationIssue(
                                    severity="ERROR",
                                    code="color_mismatch",
                                    message=(
                                        f"Entity '{v.entity_type}' color mismatch: "
                                        f"expected {v.color}, got {e.color}"
                                    ),
                                )
                            )
                    else:
                        issues.append(
                            SemanticValidationIssue(
                                severity="ERROR",
                                code="color_dropped",
                                message=(
                                    f"Color '{v.color}' bound to entity '{v.entity_type}' "
                                    "was dropped in translation"
                                ),
                            )
                        )
                else:
                    e_same_type = [
                        m
                        for m in e_mentions
                        if m.entity_type == v.entity_type
                        or ENTITY_GROUPS.get(m.entity_type) == ENTITY_GROUPS.get(v.entity_type)
                    ]
                    if e_same_type:
                        if not any(
                            m.color and _color_matches(v.color, m.color) for m in e_same_type
                        ):
                            issues.append(
                                SemanticValidationIssue(
                                    severity="ERROR",
                                    code="color_dropped",
                                    message=(
                                        f"Color '{v.color}' bound to entity '{v.entity_type}' "
                                        "was dropped in translation"
                                    ),
                                )
                            )

        # 4. Spatial orientation swap & drop across entities (ERROR)
        for p1_idx in range(len(matched_pairs)):
            for p2_idx in range(p1_idx + 1, len(matched_pairs)):
                v1, e1 = matched_pairs[p1_idx]
                v2, e2 = matched_pairs[p2_idx]
                if (
                    v1.spatial is not None
                    and v2.spatial is not None
                    and e1.spatial is not None
                    and e2.spatial is not None
                    and v1.spatial != v2.spatial
                ):
                    if e1.spatial == v2.spatial and e2.spatial == v1.spatial:
                        issues.append(
                            SemanticValidationIssue(
                                severity="ERROR",
                                code="spatial_swap",
                                message=(
                                    f"Spatial orientation swapped across entities: "
                                    f"{v1.entity_type} expected {v1.spatial} got {e1.spatial}, "
                                    f"{v2.entity_type} expected {v2.spatial} got {e2.spatial}"
                                ),
                            )
                        )

        for v, e in matched_pairs:
            if v.spatial is not None and e.spatial is None:
                if not any(iss.code in ("direction_dropped", "spatial_dropped") for iss in issues):
                    issues.append(
                        SemanticValidationIssue(
                            severity="ERROR",
                            code="spatial_dropped",
                            message=(
                                f"Spatial direction for entity '{v.entity_type}' was "
                                f"dropped in translation"
                            ),
                        )
                    )

        # 5. Misplaced negation across entities (ERROR)
        for p1_idx in range(len(matched_pairs)):
            for p2_idx in range(p1_idx + 1, len(matched_pairs)):
                v1, e1 = matched_pairs[p1_idx]
                v2, e2 = matched_pairs[p2_idx]
                if v1.negated != v2.negated:
                    if e1.negated == v2.negated and e2.negated == v1.negated:
                        issues.append(
                            SemanticValidationIssue(
                                severity="ERROR",
                                code="negation_entity_misplaced",
                                message=(
                                    "Negation polarity misplaced across entities in translation"
                                ),
                            )
                        )

        # 6. Comparator binding across entities (ERROR)
        for p1_idx in range(len(matched_pairs)):
            for p2_idx in range(p1_idx + 1, len(matched_pairs)):
                v1, e1 = matched_pairs[p1_idx]
                v2, e2 = matched_pairs[p2_idx]
                if v1.has_comparator and not v2.has_comparator:
                    if e2.has_comparator and not e1.has_comparator:
                        issues.append(
                            SemanticValidationIssue(
                                severity="ERROR",
                                code="comparator_mismatch",
                                message="Comparator quantifier swapped across entities",
                            )
                        )

        for v, e in matched_pairs:
            if v.has_comparator and e.has_comparator:
                if v.comparator_type != e.comparator_type:
                    issues.append(
                        SemanticValidationIssue(
                            severity="ERROR",
                            code="comparator_mismatch",
                            message="Comparator quantifier type mismatched across entities",
                        )
                    )

        # 7. Generalized temporal sequence of entities (ERROR)
        m_b = re.search(r"\b(?:bắt đầu|khởi đầu)\b", src_clean.lower())
        m_t = re.search(r"\b(?:sau đó|tiếp theo|tiếp đến|kế tiếp)\b", src_clean.lower())
        if m_b and m_t and m_b.start() < m_t.start():
            initial_entities = {m.entity_type for m in v_mentions if m.start_pos < m_t.start()}
            subsequent_entities = {m.entity_type for m in v_mentions if m.start_pos > m_t.start()}
            exclusive_init = initial_entities - subsequent_entities
            exclusive_sub = subsequent_entities - initial_entities
            for e_init in exclusive_init:
                for e_sub in exclusive_sub:
                    pos_init = next(
                        (m.start_pos for m in e_mentions if m.entity_type == e_init), None
                    )
                    pos_sub = next(
                        (m.start_pos for m in e_mentions if m.entity_type == e_sub), None
                    )
                    if pos_init is not None and pos_sub is not None:
                        if pos_sub < pos_init:
                            issues.append(
                                SemanticValidationIssue(
                                    severity="ERROR",
                                    code="temporal_inversion",
                                    message=(
                                        f"Temporal sequence of entities inverted: {e_sub} "
                                        f"appears before {e_init}"
                                    ),
                                )
                            )

    def _validate_untranslated_vietnamese(
        self,
        src_clean: str,
        tgt_clean: str,
        issues: list[SemanticValidationIssue],
    ) -> None:
        src_lower = src_clean.lower()
        tgt_lower = tgt_clean.lower()
        tgt_words = tgt_lower.split()

        # Check for unaccented Vietnamese copy:
        # e.g. "nguoi dan ong dung ben trai" -> "nguoi dan ong dung ben trai"
        src_stripped = _strip_diacritics(src_lower)
        if src_stripped == tgt_lower:
            issues.append(
                SemanticValidationIssue(
                    severity="ERROR",
                    code="untranslated_vietnamese",
                    message="Translation is identical to source unaccented Vietnamese text",
                )
            )
            return

        unaccented_match_count = sum(1 for w in tgt_words if w in VI_UNACCENTED_WORDS)
        unaccented_ratio = unaccented_match_count / max(len(tgt_words), 1)
        if len(tgt_words) >= 3 and unaccented_ratio >= 0.5:
            issues.append(
                SemanticValidationIssue(
                    severity="ERROR",
                    code="untranslated_vietnamese",
                    message="Target text predominantly contains unaccented Vietnamese words",
                )
            )
            return

        if _has_vietnamese_diacritics(tgt_clean):
            vi_func_count = sum(1 for word in tgt_words if word in VI_FUNCTION_WORDS)
            src_tokens = set(src_lower.split())
            overlap_count = sum(1 for word in tgt_words if word in src_tokens)
            overlap_ratio = overlap_count / max(len(tgt_words), 1)

            if vi_func_count >= 3 or overlap_ratio > 0.45:
                issues.append(
                    SemanticValidationIssue(
                        severity="ERROR",
                        code="untranslated_vietnamese",
                        message="Translation contains untranslated Vietnamese grammatical phrases",
                    )
                )
            else:
                issues.append(
                    SemanticValidationIssue(
                        severity="WARNING",
                        code="diacritic_proper_noun",
                        message="Target text retains Vietnamese diacritics in phrase",
                    )
                )

    def _extract_vietnamese_numbers(self, text: str) -> list[tuple[int, str, bool, bool]]:
        """Parse explicit digits and compound Vietnamese numbers into integer values.

        Returns tuple of (val, original_form, is_strict_cardinal, is_paired).
        """
        indexed_results: list[tuple[int, tuple[int, str, bool, bool]]] = []
        lower = text.lower()

        # 1. Explicit digits e.g. 5, 12, 100 (excluding 4-digit calendar years 1900-2099)
        for match in re.finditer(r"\b(\d+)\b", text):
            val_str = match.group(1)
            val_int = int(val_str)
            if len(val_str) == 4 and 1900 <= val_int <= 2099:
                continue
            indexed_results.append((match.start(), (val_int, val_str, True, False)))

        # 2. Compound and single Vietnamese number phrases
        tens_units = {
            "một": 1,
            "mốt": 1,
            "hai": 2,
            "ba": 3,
            "bốn": 4,
            "tư": 4,
            "năm": 5,
            "lăm": 5,
            "nhăm": 5,
            "sáu": 6,
            "bảy": 7,
            "tám": 8,
            "chín": 9,
        }
        tens_prefixes = {
            "hai mươi": 20,
            "ba mươi": 30,
            "bốn mươi": 40,
            "năm mươi": 50,
            "sáu mươi": 60,
            "bảy mươi": 70,
            "tám mươi": 80,
            "chín mươi": 90,
            "hai chục": 20,
            "ba chục": 30,
            "bốn chục": 40,
            "năm chục": 50,
        }

        # Track spans to avoid overlapping matches
        matched_spans: list[tuple[int, int]] = []

        # Check teens: "mười [một..chín]"
        for u_word, u_val in tens_units.items():
            pat = r"\bmười\s+" + re.escape(u_word) + r"\b"
            for m in re.finditer(pat, lower):
                indexed_results.append((m.start(), (10 + u_val, m.group(0), True, False)))
                matched_spans.append(m.span())

        # Check 20-99 compounds: "[hai..chín] mươi [mốt..chín]"
        for t_word, t_val in tens_prefixes.items():
            for u_word, u_val in tens_units.items():
                pat = r"\b" + re.escape(t_word) + r"\s+" + re.escape(u_word) + r"\b"
                for m in re.finditer(pat, lower):
                    indexed_results.append((m.start(), (t_val + u_val, m.group(0), True, False)))
                    matched_spans.append(m.span())

            # Check exact tens: "hai mươi", "ba mươi", etc.
            pat_exact = r"\b" + re.escape(t_word) + r"\b"
            for m in re.finditer(pat_exact, lower):
                if not any(s <= m.start() and m.end() <= e for s, e in matched_spans):
                    indexed_results.append((m.start(), (t_val, m.group(0), True, False)))
                    matched_spans.append(m.span())

        # Check exact 10: "mười"
        for m in re.finditer(r"\bmười\b", lower):
            if not any(s <= m.start() and m.end() <= e for s, e in matched_spans):
                indexed_results.append((m.start(), (10, "mười", True, False)))
                matched_spans.append(m.span())

        # Check single digits: "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín"
        singles = {
            "một": 1,
            "hai": 2,
            "đôi": 2,
            "cặp": 2,
            "ba": 3,
            "tam": 3,
            "bốn": 4,
            "tư": 4,
            "năm": 5,
            "sáu": 6,
            "bảy": 7,
            "tám": 8,
            "chín": 9,
        }
        for word, val in singles.items():
            pat = r"\b" + re.escape(word) + r"\b"
            for m in re.finditer(pat, lower):
                if any(s <= m.start() and m.end() <= e for s, e in matched_spans):
                    continue
                start, end = m.span()
                # Disambiguate "năm" (year context vs count 5)
                if word == "năm":
                    pre = lower[max(0, start - 10) : start].strip()
                    post = lower[end : min(len(lower), end + 15)].strip()
                    # Year context vs noun/classifier e.g. "năm người", "năm con", etc.
                    noun_pat = (
                        r"^(?:người|con|chiếc|cái|khối|bản|bộ|loại|thành viên|đứa|học sinh|bạn)\b"
                    )
                    is_noun_following = bool(re.match(noun_pat, post))
                    if not is_noun_following:
                        if (
                            pre.endswith("vào")
                            or pre.endswith("hàng")
                            or re.match(r"^(?:qua|nay|tới|trước|sau|\d{4})\b", post)
                        ):
                            continue
                if word == "một":
                    pre = lower[max(0, start - 20) : start].strip()
                    post = lower[end : min(len(lower), end + 20)].strip()
                    # Only treat 'một' as strict cardinal if preceded by strong cues
                    # like 'chỉ có', 'chỉ', 'duy nhất', 'đúng' or followed by 'lần'
                    is_strict = bool(
                        re.search(r"(chỉ có|chỉ|duy nhất|đúng)$", pre)
                        or re.match(r"^(lần|người duy nhất)\b", post)
                    )
                    indexed_results.append((start, (val, word, is_strict, False)))
                    matched_spans.append(m.span())
                    continue

                is_paired = False
                if val == 2:
                    post = lower[end : min(len(lower), end + 15)].strip()
                    if post.startswith(
                        ("tay", "mắt", "tai", "chân", "bên", "bàn tay", "cánh tay", "đầu gối")
                    ) or word in ("đôi", "cặp"):
                        is_paired = True
                indexed_results.append((start, (val, word, True, is_paired)))
                matched_spans.append(m.span())

        indexed_results.sort(key=lambda x: x[0])
        return [item[1] for item in indexed_results]

    def _extract_english_numbers_multiset(self, text: str) -> list[int]:
        """Extract all numeric values from English text as an ordered multiset."""
        indexed_results: list[tuple[int, int]] = []
        lower = text.lower()

        # 1. Digits
        for match in re.finditer(r"\b(\d+)\b", text):
            val_str = match.group(1)
            val_int = int(val_str)
            if len(val_str) == 4 and 1900 <= val_int <= 2099:
                continue
            indexed_results.append((match.start(), val_int))

        # 2. Compound words: "twenty-one", "twenty two", etc.
        tens_map = {
            "twenty": 20,
            "thirty": 30,
            "forty": 40,
            "fifty": 50,
            "sixty": 60,
            "seventy": 70,
            "eighty": 80,
            "ninety": 90,
        }
        units_map = {
            "one": 1,
            "two": 2,
            "three": 3,
            "four": 4,
            "five": 5,
            "six": 6,
            "seven": 7,
            "eight": 8,
            "nine": 9,
        }

        matched_spans: list[tuple[int, int]] = []
        for t_word, t_val in tens_map.items():
            for u_word, u_val in units_map.items():
                pat = r"\b" + re.escape(t_word) + r"[\s-]+" + re.escape(u_word) + r"\b"
                for m in re.finditer(pat, lower):
                    indexed_results.append((m.start(), t_val + u_val))
                    matched_spans.append(m.span())

        # 3. Single cardinal words
        for word, val in EN_CARDINAL_WORDS.items():
            pat = r"\b" + re.escape(word) + r"\b"
            for m in re.finditer(pat, lower):
                if not any(s <= m.start() and m.end() <= e for s, e in matched_spans):
                    indexed_results.append((m.start(), val))
                    matched_spans.append(m.span())

        # 4. Paired entity representation: 'both', 'pair', 'couple' count as 2
        for m in re.finditer(r"\b(both|pair|couple)\b", lower):
            if not any(s <= m.start() and m.end() <= e for s, e in matched_spans):
                indexed_results.append((m.start(), 2))
                matched_spans.append(m.span())

        # 5. Singularity expressions: 'once', 'single', 'only one' count as 1
        for m in re.finditer(r"\b(once|single|only one)\b", lower):
            if not any(s <= m.start() and m.end() <= e for s, e in matched_spans):
                indexed_results.append((m.start(), 1))
                matched_spans.append(m.span())

        indexed_results.sort(key=lambda x: x[0])
        return [item[1] for item in indexed_results]

    def _extract_english_numbers(self, text: str) -> set[int]:
        """Extract all unique numeric values from English text."""
        return set(self._extract_english_numbers_multiset(text))

    def _extract_english_ordinals(self, text: str) -> set[int]:
        """Extract ordinal values present in English text (e.g. second -> 2, 3rd -> 3)."""
        ordinals: set[int] = set()
        lower = text.lower()

        # Digits with suffix: 1st, 2nd, 3rd, 4th...
        for m in re.finditer(r"\b(\d+)(st|nd|rd|th)\b", lower):
            ordinals.add(int(m.group(1)))

        # Ordinal words
        for word, val in EN_ORDINAL_WORDS.items():
            if re.search(r"\b" + re.escape(word) + r"\b", lower):
                ordinals.add(val)

        return ordinals

    @staticmethod
    def _find_first_pattern(text: str, patterns: list[str]) -> int | None:
        earliest: int | None = None
        for pat in patterns:
            match = re.search(pat, text)
            if match:
                if earliest is None or match.start() < earliest:
                    earliest = match.start()
        return earliest
