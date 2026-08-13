from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any

from src.preprocessing.ocr_temporal_merger import remove_vietnamese_accents


PROTECTED_PATTERN = re.compile(
    r"(?P<phone>\b(?:0|\+84)[\d.\-\s]{7,}\b)"
    r"|(?P<percent>\b\d+(?:[.,]\d+)?\s*%)"
    r"|(?P<date>\b\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?\b)"
    r"|(?P<time>\b\d{1,2}:\d{2}(?::\d{2})?\b)"
    r"|(?P<score>\b\d+\s*[-:]\s*\d+\b)"
    r"|(?P<code>\b[A-Z]{2,}[A-Z0-9-]*\d+[A-Z0-9-]*\b)",
    flags=re.IGNORECASE,
)


@dataclass
class ValidationResult:
    accepted: bool
    reason: str
    proposed_text: str
    fallback_text: str
    protected_source: list[str]
    protected_proposed: list[str]


def normalize_light(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    return " ".join(text.strip().split())


def normalize_match(text: str) -> str:
    return remove_vietnamese_accents(normalize_light(text).lower())


def tokenize(text: str) -> list[str]:
    return re.findall(r"[\w%:/.-]+", normalize_match(text), flags=re.UNICODE)


def protected_tokens(text: str) -> list[str]:
    out = []
    for match in PROTECTED_PATTERN.finditer(normalize_light(text)):
        value = match.group(0)
        if value:
            out.append(re.sub(r"\s+", "", value))
    return out


def token_overlap(source: str, proposed: str) -> float:
    a = {t for t in tokenize(source) if len(t) > 1}
    b = {t for t in tokenize(proposed) if len(t) > 1}
    if not a or not b:
        return 1.0 if not a and not b else 0.0
    return len(a & b) / max(1, min(len(a), len(b)))


def unsupported_caps(source: str, proposed: str) -> list[str]:
    source_norm = {tok.strip(".:-") for tok in tokenize(source)}
    added = []
    for raw_token in re.findall(r"\b[A-ZĐ][A-Z0-9Đ.-]{2,}\b", normalize_light(proposed)):
        norm = normalize_match(raw_token).strip(".:-")
        if norm not in source_norm:
            added.append(raw_token)
    return added


def validate_qwen_correction(
    source_text: str,
    proposed_text: str,
    evidence_texts: list[str] | None = None,
    cfg: dict[str, Any] | None = None,
) -> ValidationResult:
    cfg = cfg or {}
    source = normalize_light(source_text)
    proposed = normalize_light(proposed_text)
    evidence_blob = normalize_light(" ".join([source, *(evidence_texts or [])]))
    fallback = source

    src_protected = protected_tokens(source)
    proposed_protected = protected_tokens(proposed)

    if not proposed:
        return ValidationResult(False, "empty_proposed_text", proposed, fallback, src_protected, proposed_protected)

    if bool(cfg.get("preserve_numbers", True)):
        missing = [tok for tok in src_protected if tok not in proposed_protected]
        if missing:
            return ValidationResult(False, f"missing_protected_tokens:{missing[:4]}", proposed, fallback, src_protected, proposed_protected)

    src_words = source.split()
    proposed_words = proposed.split()
    min_ratio = float(cfg.get("min_length_ratio", 0.35))
    max_ratio = float(cfg.get("max_length_ratio", 2.20))
    if src_words:
        ratio = len(proposed_words) / max(1, len(src_words))
        if ratio < min_ratio or ratio > max_ratio:
            return ValidationResult(False, f"length_ratio_out_of_range:{ratio:.2f}", proposed, fallback, src_protected, proposed_protected)

    overlap = token_overlap(source, proposed)
    min_overlap = float(cfg.get("min_token_overlap", 0.30))
    if len(src_words) >= 4 and overlap < min_overlap:
        return ValidationResult(False, f"low_token_overlap:{overlap:.3f}", proposed, fallback, src_protected, proposed_protected)

    if bool(cfg.get("forbid_unsupported_capital_suffix", True)):
        added_caps = unsupported_caps(evidence_blob, proposed)
        if added_caps:
            return ValidationResult(False, f"unsupported_capital_tokens:{added_caps[:4]}", proposed, fallback, src_protected, proposed_protected)

    return ValidationResult(True, "ok", proposed, fallback, src_protected, proposed_protected)
