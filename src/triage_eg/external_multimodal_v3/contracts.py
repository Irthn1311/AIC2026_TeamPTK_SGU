"""Frozen labels and small contracts for the external V3 import."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

ASR_SOURCE_TYPE = "ASR_EXTERNAL_V3_VALIDATED"
OCR_SOURCE_TYPE = "OCR_EXTERNAL_PARTIAL"
OBJECT_SOURCE_TYPE = "OBJECT_EXTERNAL_PARTIAL"
PROVENANCE_LEVEL = "VALIDATED_EXTERNAL_NOT_REPRODUCIBLE"
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"


@dataclass(frozen=True)
class ImportResult:
    """Paths and decisions returned by one fail-closed import."""

    output_root: str
    asr_bundle: str
    asr_decision: str
    ocr_decision: str
    object_decision: str
    summary: dict[str, Any]


__all__ = [
    "ASR_SOURCE_TYPE",
    "EMBEDDING_MODEL",
    "ImportResult",
    "OBJECT_SOURCE_TYPE",
    "OCR_SOURCE_TYPE",
    "PROVENANCE_LEVEL",
]
