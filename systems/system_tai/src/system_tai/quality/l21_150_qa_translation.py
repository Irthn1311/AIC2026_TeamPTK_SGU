"""Strict frozen English-question sidecar for the L21-150 QA DEV diagnostic."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .l21_150_schema import L21150Benchmark, L21150QAQuery

SIDECAR_SCHEMA_VERSION = 1
SIDECAR_ROLE = "QA_DEV_LOCALIZATION_TRANSLATION_SIDECAR"
SIDECAR_SCOPE = "DEV_QA_ONLY"
SIDECAR_TASK = "qa"
SOURCE_LANGUAGE = "vi"
TARGET_LANGUAGE = "en"
TRANSLATION_SCOPE = "question_vi_only"
TRANSLATION_STATUS = "MODEL_AUTHORED_FROZEN_NOT_HUMAN_REVIEWED"
EXPECTED_QUERY_COUNT = 38


class QATranslationSidecarError(ValueError):
    """The QA DEV translation sidecar violates its frozen contract."""


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise QATranslationSidecarError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _exact_fields(value: dict[str, Any], expected: set[str], context: str) -> None:
    missing = sorted(expected - set(value))
    extra = sorted(set(value) - expected)
    if missing or extra:
        raise QATranslationSidecarError(
            f"{context} fields mismatch: missing={missing}, extra={extra}"
        )


def _text(value: Any, name: str) -> str:
    if type(value) is not str or not value.strip() or value != value.strip():
        raise QATranslationSidecarError(
            f"{name} must be a non-empty string without outer whitespace"
        )
    return value


@dataclass(frozen=True, slots=True)
class QATranslationEntry:
    query_id: str
    question_vi: str
    question_en: str


@dataclass(frozen=True, slots=True)
class QADevTranslationSidecar:
    schema_version: int
    sidecar_role: str
    benchmark_id: str
    benchmark_sha256: str
    scope: str
    task: str
    query_count: int
    source_language: str
    target_language: str
    translation_scope: str
    translation_status: str
    official_ground_truth: bool
    retrieval_feedback_used: bool
    authoring_provenance: str
    entries: tuple[QATranslationEntry, ...]

    @property
    def translations(self) -> dict[str, str]:
        return {entry.query_id: entry.question_en for entry in self.entries}


TOP_LEVEL_FIELDS = {
    "schema_version",
    "sidecar_role",
    "benchmark_id",
    "benchmark_sha256",
    "scope",
    "task",
    "query_count",
    "source_language",
    "target_language",
    "translation_scope",
    "translation_status",
    "official_ground_truth",
    "retrieval_feedback_used",
    "authoring_provenance",
    "entries",
}
ENTRY_FIELDS = {"query_id", "question_vi", "question_en"}


def validate_qa_dev_translation_payload(
    payload: Any,
    benchmark: L21150Benchmark,
    *,
    benchmark_sha256: str,
) -> QADevTranslationSidecar:
    if type(payload) is not dict:
        raise QATranslationSidecarError("sidecar root must be an object")
    _exact_fields(payload, TOP_LEVEL_FIELDS, "sidecar")
    expected_values = {
        "schema_version": SIDECAR_SCHEMA_VERSION,
        "sidecar_role": SIDECAR_ROLE,
        "benchmark_id": benchmark.benchmark_id,
        "benchmark_sha256": benchmark_sha256,
        "scope": SIDECAR_SCOPE,
        "task": SIDECAR_TASK,
        "query_count": EXPECTED_QUERY_COUNT,
        "source_language": SOURCE_LANGUAGE,
        "target_language": TARGET_LANGUAGE,
        "translation_scope": TRANSLATION_SCOPE,
        "translation_status": TRANSLATION_STATUS,
        "official_ground_truth": False,
        "retrieval_feedback_used": False,
    }
    for field, expected in expected_values.items():
        if payload[field] != expected or type(payload[field]) is not type(expected):
            raise QATranslationSidecarError(f"{field} must equal {expected!r}")

    dev_queries = sorted(
        (
            query
            for query in benchmark.queries
            if isinstance(query, L21150QAQuery) and query.split == "DEV"
        ),
        key=lambda query: query.query_id,
    )
    if len(dev_queries) != EXPECTED_QUERY_COUNT:
        raise QATranslationSidecarError(
            f"benchmark must contain exactly {EXPECTED_QUERY_COUNT} DEV QA queries"
        )
    expected_questions = {query.query_id: query.question_vi for query in dev_queries}
    expected_order = [query.query_id for query in dev_queries]
    holdout_ids = {
        query.query_id
        for query in benchmark.queries
        if isinstance(query, L21150QAQuery) and query.split == "HOLDOUT"
    }

    entries_payload = payload["entries"]
    if type(entries_payload) is not list:
        raise QATranslationSidecarError("entries must be an array")
    if len(entries_payload) != EXPECTED_QUERY_COUNT:
        raise QATranslationSidecarError(
            f"entries must contain exactly {EXPECTED_QUERY_COUNT} DEV QA translations"
        )

    entries: list[QATranslationEntry] = []
    seen: set[str] = set()
    for index, value in enumerate(entries_payload):
        if type(value) is not dict:
            raise QATranslationSidecarError(f"entries[{index}] must be an object")
        _exact_fields(value, ENTRY_FIELDS, f"entries[{index}]")
        query_id = _text(value["query_id"], f"entries[{index}].query_id")
        if query_id in seen:
            raise QATranslationSidecarError(f"duplicate query_id: {query_id}")
        seen.add(query_id)
        if query_id in holdout_ids:
            raise QATranslationSidecarError(f"HOLDOUT query is forbidden: {query_id}")
        expected_question = expected_questions.get(query_id)
        if expected_question is None:
            raise QATranslationSidecarError(f"unknown DEV QA query_id: {query_id}")
        question_vi = _text(value["question_vi"], f"entries[{index}].question_vi")
        if question_vi != expected_question:
            raise QATranslationSidecarError(f"question_vi mismatch for {query_id}")
        entries.append(
            QATranslationEntry(
                query_id=query_id,
                question_vi=question_vi,
                question_en=_text(
                    value["question_en"], f"entries[{index}].question_en"
                ),
            )
        )

    actual_order = [entry.query_id for entry in entries]
    if actual_order != expected_order:
        missing = sorted(set(expected_order) - set(actual_order))
        extra = sorted(set(actual_order) - set(expected_order))
        raise QATranslationSidecarError(
            "entries must exactly follow deterministic DEV QA query_id order: "
            f"missing={missing}, extra={extra}"
        )

    return QADevTranslationSidecar(
        schema_version=payload["schema_version"],
        sidecar_role=payload["sidecar_role"],
        benchmark_id=payload["benchmark_id"],
        benchmark_sha256=payload["benchmark_sha256"],
        scope=payload["scope"],
        task=payload["task"],
        query_count=payload["query_count"],
        source_language=payload["source_language"],
        target_language=payload["target_language"],
        translation_scope=payload["translation_scope"],
        translation_status=payload["translation_status"],
        official_ground_truth=payload["official_ground_truth"],
        retrieval_feedback_used=payload["retrieval_feedback_used"],
        authoring_provenance=_text(
            payload["authoring_provenance"], "authoring_provenance"
        ),
        entries=tuple(entries),
    )


def load_qa_dev_translation_sidecar(
    path: Path,
    benchmark: L21150Benchmark,
    benchmark_path: Path,
) -> QADevTranslationSidecar:
    raw = Path(path).read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        raise QATranslationSidecarError("sidecar must not contain a UTF-8 BOM")
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise QATranslationSidecarError("sidecar must be valid UTF-8") from exc
    try:
        payload = json.loads(text, object_pairs_hook=_reject_duplicate_pairs)
    except json.JSONDecodeError as exc:
        raise QATranslationSidecarError(f"invalid sidecar JSON: {exc}") from exc
    benchmark_sha256 = hashlib.sha256(Path(benchmark_path).read_bytes()).hexdigest()
    return validate_qa_dev_translation_payload(
        payload,
        benchmark,
        benchmark_sha256=benchmark_sha256,
    )


def qa_sidecar_to_payload(sidecar: QADevTranslationSidecar) -> dict[str, Any]:
    return {
        "schema_version": sidecar.schema_version,
        "sidecar_role": sidecar.sidecar_role,
        "benchmark_id": sidecar.benchmark_id,
        "benchmark_sha256": sidecar.benchmark_sha256,
        "scope": sidecar.scope,
        "task": sidecar.task,
        "query_count": sidecar.query_count,
        "source_language": sidecar.source_language,
        "target_language": sidecar.target_language,
        "translation_scope": sidecar.translation_scope,
        "translation_status": sidecar.translation_status,
        "official_ground_truth": sidecar.official_ground_truth,
        "retrieval_feedback_used": sidecar.retrieval_feedback_used,
        "authoring_provenance": sidecar.authoring_provenance,
        "entries": [
            {
                "query_id": entry.query_id,
                "question_vi": entry.question_vi,
                "question_en": entry.question_en,
            }
            for entry in sidecar.entries
        ],
    }


def serialize_qa_dev_translation_sidecar(
    sidecar: QADevTranslationSidecar,
) -> bytes:
    return (
        json.dumps(
            qa_sidecar_to_payload(sidecar),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
