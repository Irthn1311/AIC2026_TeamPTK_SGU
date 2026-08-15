"""Blinded translation provenance and deterministic surface-only checks."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from .contracts import FORBIDDEN_BLIND_QC_FIELDS, SemanticUnitSnapshot

_QUOTED = re.compile(r"['\"“”‘’]([^'\"“”‘’]+)['\"“”‘’]")
_NUMBER = re.compile(r"(?<!\w)[+-]?(?:\d+[.,]?\d*|[.,]\d+)(?!\w)")
_ACRONYM = re.compile(r"(?<!\w)[A-Z][A-Z0-9_-]{1,}(?!\w)")


def _normalized_tokens(value: str) -> list[str]:
    return re.findall(r"\w+", value.casefold(), flags=re.UNICODE)


def translation_surface_checks(
    source: str, target: str, *, source_language: str
) -> tuple[bool, list[str]]:
    """Flag only mechanically verifiable surface anomalies, never semantic quality."""

    source_text, target_text = str(source).strip(), str(target).strip()
    reasons: list[str] = []
    if not target_text:
        reasons.append("EMPTY_TRANSLATION")
        return True, reasons
    source_length = max(len(source_text), 1)
    ratio = len(target_text) / source_length
    if ratio < 0.2 or ratio > 4.0:
        reasons.append("EXTREME_LENGTH_RATIO")
    source_numbers = _NUMBER.findall(source_text)
    target_numbers = _NUMBER.findall(target_text)
    if source_numbers != target_numbers:
        reasons.append("NUMBER_LOST_OR_CHANGED")
    target_folded = target_text.casefold()
    lost_quotes = [
        value for value in _QUOTED.findall(source_text) if value.casefold() not in target_folded
    ]
    if lost_quotes:
        reasons.append("QUOTED_LITERAL_LOST_OR_CHANGED")
    lost_acronyms = [
        value for value in _ACRONYM.findall(source_text) if value.casefold() not in target_folded
    ]
    if lost_acronyms:
        reasons.append("ACRONYM_LOST_OR_CHANGED")
    target_tokens = _normalized_tokens(target_text)
    if len(target_tokens) >= 4 and len(set(target_tokens)) == 1:
        reasons.append("UNEXPECTED_DUPLICATED_OUTPUT")
    if source_language == "vi" and source_text.casefold() == target_text.casefold():
        reasons.append("IDENTICAL_TO_VIETNAMESE_SOURCE")
    return bool(reasons), reasons


def blind_translation_rows(
    units: dict[str, SemanticUnitSnapshot],
) -> list[dict[str, Any]]:
    output = []
    for unit_id in sorted(units):
        unit = units[unit_id]
        opus = str(
            unit.encoding.get("translated_text") or unit.encoding.get("clip_input_text") or ""
        )
        anomaly, reasons = translation_surface_checks(
            unit.source_text, opus, source_language=unit.source_language
        )
        row = {
            "unit_id": unit.unit_id,
            "query_id": unit.query_id,
            "task": unit.task,
            "event_id": unit.event_id,
            "source_vi": unit.source_text,
            "opus_en": opus,
            "translation_surface_anomaly": anomaly,
            "translation_surface_reasons": reasons,
        }
        leaked = FORBIDDEN_BLIND_QC_FIELDS & set(row)
        if leaked:
            raise RuntimeError(f"TRANSLATION_BLIND_QC_LEAKAGE: {sorted(leaked)}")
        output.append(row)
    return output


def translation_surface_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    reasons: Counter[str] = Counter(
        reason for row in rows for reason in row["translation_surface_reasons"]
    )
    return {
        "translation_unit_count": len(rows),
        "translation_surface_anomaly_count": sum(
            bool(row["translation_surface_anomaly"]) for row in rows
        ),
        "reason_counts": dict(sorted(reasons.items())),
        "semantic_fidelity_status": "NOT_REVIEWED",
        "surface_checks_are_not_semantic_judgments": True,
    }


def translation_provenance_rows(
    units: dict[str, SemanticUnitSnapshot],
) -> list[dict[str, Any]]:
    """Expose the already-used Stage 2 route without translating a second time."""

    rows = []
    for unit_id in sorted(units):
        unit = units[unit_id]
        encoding = dict(unit.encoding)
        rows.append(
            {
                "unit_id": unit.unit_id,
                "query_id": unit.query_id,
                "task": unit.task,
                "event_id": unit.event_id,
                "source_language": unit.source_language,
                "source_text": unit.source_text,
                "exact_clip_input_text": encoding.get("clip_input_text"),
                "translator_route": encoding.get("translator_route"),
                "translator_model": encoding.get("translator_model"),
                "encoding_provenance": encoding,
            }
        )
    return rows


def translation_review_instructions() -> str:
    return """# AI Translation Review Instructions

Judge ONLY whether `opus_en` faithfully preserves the Vietnamese retrieval meaning in
`source_vi`. This is a blinded review: do not use ground truth, retrieval ranks, scores,
or retrieval outcomes.

For every `unit_id`, return exactly one verdict: `PASS`, `CONDITIONAL`, or `FAIL`.
Optional reason tags are: `ENTITY_LOSS`, `ACTION_LOSS`, `RELATION_LOSS`,
`ATTRIBUTE_LOSS`, `NEGATION_ERROR`, `NUMBER_ERROR`, `COLOR_ERROR`,
`NAMED_TEXT_ERROR`, `WORD_SENSE_ERROR`, or `OTHER`.

Only for `FAIL`, provide one faithful English reference translation. Do not rewrite or
select retrieval predictions. A future bounded causal ablation may use independently
reviewed reference translations; D1 itself makes no causal translation claim.
"""


__all__ = [
    "blind_translation_rows",
    "translation_review_instructions",
    "translation_provenance_rows",
    "translation_surface_checks",
    "translation_surface_summary",
]
