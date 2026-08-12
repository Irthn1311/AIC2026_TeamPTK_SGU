"""Strict frozen English-event sidecar for the L21-150 TRAKE DEV diagnostic."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .l21_150_schema import L21150Benchmark, L21150TRAKEQuery

SIDECAR_SCHEMA_VERSION = 1
SIDECAR_ROLE = "EXPERIMENT_INPUT"
SIDECAR_SCOPE = "DEV_TRAKE_ONLY"
SIDECAR_TASK = "trake"
TRANSLATION_ROLE = "LITERAL_ENGLISH_EVENT_TRANSLATION"
TRANSLATION_POLICY = (
    "FAITHFUL_EVENT_DESCRIPTION_ONLY_WITHOUT_RETRIEVAL_FEEDBACK"
)
TRANSLATION_VERSION = "tr-a2-d0-trake-dev-en-v1"
TRANSLATION_STATUS = "MODEL_AUTHORED_FROZEN_NOT_HUMAN_REVIEWED"
EXPECTED_QUERY_COUNT = 38
EXPECTED_EVENT_COUNT = 114


class TRAKETranslationSidecarError(ValueError):
    """The TRAKE DEV translation sidecar violates its frozen contract."""


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise TRAKETranslationSidecarError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _exact_fields(value: dict[str, Any], expected: set[str], context: str) -> None:
    missing = sorted(expected - set(value))
    extra = sorted(set(value) - expected)
    if missing or extra:
        raise TRAKETranslationSidecarError(
            f"{context} fields mismatch: missing={missing}, extra={extra}"
        )


def _text(value: Any, name: str) -> str:
    if type(value) is not str or not value.strip() or value != value.strip():
        raise TRAKETranslationSidecarError(
            f"{name} must be a non-empty string without outer whitespace"
        )
    return value


@dataclass(frozen=True, slots=True)
class TRAKETranslationRecord:
    query_id: str
    source_event_index: int
    source_vi: str
    translation_en: str


@dataclass(frozen=True, slots=True)
class TRAKEDevTranslationSidecar:
    schema_version: int
    sidecar_role: str
    benchmark_id: str
    benchmark_sha256: str
    scope: str
    task: str
    query_count: int
    event_count: int
    translation_role: str
    translation_policy: str
    translation_version: str
    translation_status: str
    authoring_provenance: str
    retrieval_feedback_used: bool
    records: tuple[TRAKETranslationRecord, ...]

    @property
    def translations(self) -> dict[tuple[str, int], str]:
        return {
            (record.query_id, record.source_event_index): record.translation_en
            for record in self.records
        }


TOP_LEVEL_FIELDS = {
    "schema_version",
    "sidecar_role",
    "benchmark_id",
    "benchmark_sha256",
    "scope",
    "task",
    "query_count",
    "event_count",
    "translation_role",
    "translation_policy",
    "translation_version",
    "translation_status",
    "authoring_provenance",
    "retrieval_feedback_used",
    "records",
}
RECORD_FIELDS = {
    "query_id",
    "source_event_index",
    "source_vi",
    "translation_en",
}


def validate_trake_dev_translation_payload(
    payload: Any,
    benchmark: L21150Benchmark,
    *,
    benchmark_sha256: str,
) -> TRAKEDevTranslationSidecar:
    if type(payload) is not dict:
        raise TRAKETranslationSidecarError("sidecar root must be an object")
    _exact_fields(payload, TOP_LEVEL_FIELDS, "sidecar")
    expected_values = {
        "schema_version": SIDECAR_SCHEMA_VERSION,
        "sidecar_role": SIDECAR_ROLE,
        "benchmark_id": benchmark.benchmark_id,
        "benchmark_sha256": benchmark_sha256,
        "scope": SIDECAR_SCOPE,
        "task": SIDECAR_TASK,
        "query_count": EXPECTED_QUERY_COUNT,
        "event_count": EXPECTED_EVENT_COUNT,
        "translation_role": TRANSLATION_ROLE,
        "translation_policy": TRANSLATION_POLICY,
        "translation_version": TRANSLATION_VERSION,
        "translation_status": TRANSLATION_STATUS,
        "retrieval_feedback_used": False,
    }
    for field, expected in expected_values.items():
        if payload[field] != expected or type(payload[field]) is not type(expected):
            raise TRAKETranslationSidecarError(
                f"{field} must equal {expected!r}"
            )
    authoring_provenance = _text(
        payload["authoring_provenance"], "authoring_provenance"
    )
    records_payload = payload["records"]
    if type(records_payload) is not list:
        raise TRAKETranslationSidecarError("records must be an array")
    if len(records_payload) != EXPECTED_EVENT_COUNT:
        raise TRAKETranslationSidecarError(
            f"records must contain exactly {EXPECTED_EVENT_COUNT} events"
        )

    expected_queries = [
        query
        for query in benchmark.queries
        if isinstance(query, L21150TRAKEQuery) and query.split == "DEV"
    ]
    if len(expected_queries) != EXPECTED_QUERY_COUNT:
        raise TRAKETranslationSidecarError(
            f"benchmark must contain exactly {EXPECTED_QUERY_COUNT} DEV TRAKE queries"
        )
    expected_by_key = {
        (query.query_id, event.event_index): event.description_vi
        for query in expected_queries
        for event in query.events
    }
    expected_order = list(expected_by_key)
    if len(expected_by_key) != EXPECTED_EVENT_COUNT:
        raise TRAKETranslationSidecarError(
            f"benchmark must contain exactly {EXPECTED_EVENT_COUNT} DEV TRAKE events"
        )
    holdout_ids = {
        query.query_id
        for query in benchmark.queries
        if isinstance(query, L21150TRAKEQuery) and query.split == "HOLDOUT"
    }

    records: list[TRAKETranslationRecord] = []
    seen: set[tuple[str, int]] = set()
    for index, value in enumerate(records_payload):
        if type(value) is not dict:
            raise TRAKETranslationSidecarError(f"records[{index}] must be an object")
        _exact_fields(value, RECORD_FIELDS, f"records[{index}]")
        query_id = _text(value["query_id"], f"records[{index}].query_id")
        event_index = value["source_event_index"]
        if type(event_index) is not int or event_index not in {1, 2, 3}:
            raise TRAKETranslationSidecarError(
                f"records[{index}].source_event_index must be 1, 2, or 3"
            )
        key = (query_id, event_index)
        if key in seen:
            raise TRAKETranslationSidecarError(
                f"duplicate query/event pair: {query_id}/{event_index}"
            )
        seen.add(key)
        if query_id in holdout_ids:
            raise TRAKETranslationSidecarError(
                f"HOLDOUT query is forbidden: {query_id}"
            )
        expected_source = expected_by_key.get(key)
        if expected_source is None:
            raise TRAKETranslationSidecarError(
                f"unknown DEV TRAKE query/event pair: {query_id}/{event_index}"
            )
        source_vi = _text(value["source_vi"], f"records[{index}].source_vi")
        if source_vi != expected_source:
            raise TRAKETranslationSidecarError(
                f"source_vi mismatch for {query_id}/{event_index}"
            )
        records.append(
            TRAKETranslationRecord(
                query_id=query_id,
                source_event_index=event_index,
                source_vi=source_vi,
                translation_en=_text(
                    value["translation_en"], f"records[{index}].translation_en"
                ),
            )
        )

    actual_order = [
        (record.query_id, record.source_event_index) for record in records
    ]
    if actual_order != expected_order:
        missing = sorted(set(expected_order) - set(actual_order))
        extra = sorted(set(actual_order) - set(expected_order))
        raise TRAKETranslationSidecarError(
            "records must exactly follow benchmark DEV TRAKE event order: "
            f"missing={missing}, extra={extra}"
        )

    return TRAKEDevTranslationSidecar(
        schema_version=payload["schema_version"],
        sidecar_role=payload["sidecar_role"],
        benchmark_id=payload["benchmark_id"],
        benchmark_sha256=payload["benchmark_sha256"],
        scope=payload["scope"],
        task=payload["task"],
        query_count=payload["query_count"],
        event_count=payload["event_count"],
        translation_role=payload["translation_role"],
        translation_policy=payload["translation_policy"],
        translation_version=payload["translation_version"],
        translation_status=payload["translation_status"],
        authoring_provenance=authoring_provenance,
        retrieval_feedback_used=payload["retrieval_feedback_used"],
        records=tuple(records),
    )


def load_trake_dev_translation_sidecar(
    path: Path,
    benchmark: L21150Benchmark,
    benchmark_path: Path,
) -> TRAKEDevTranslationSidecar:
    raw = Path(path).read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        raise TRAKETranslationSidecarError("sidecar must not contain a UTF-8 BOM")
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise TRAKETranslationSidecarError("sidecar must be valid UTF-8") from exc
    try:
        payload = json.loads(text, object_pairs_hook=_reject_duplicate_pairs)
    except json.JSONDecodeError as exc:
        raise TRAKETranslationSidecarError(f"invalid sidecar JSON: {exc}") from exc
    benchmark_sha256 = hashlib.sha256(Path(benchmark_path).read_bytes()).hexdigest()
    return validate_trake_dev_translation_payload(
        payload,
        benchmark,
        benchmark_sha256=benchmark_sha256,
    )


def sidecar_to_payload(sidecar: TRAKEDevTranslationSidecar) -> dict[str, Any]:
    return {
        "schema_version": sidecar.schema_version,
        "sidecar_role": sidecar.sidecar_role,
        "benchmark_id": sidecar.benchmark_id,
        "benchmark_sha256": sidecar.benchmark_sha256,
        "scope": sidecar.scope,
        "task": sidecar.task,
        "query_count": sidecar.query_count,
        "event_count": sidecar.event_count,
        "translation_role": sidecar.translation_role,
        "translation_policy": sidecar.translation_policy,
        "translation_version": sidecar.translation_version,
        "translation_status": sidecar.translation_status,
        "authoring_provenance": sidecar.authoring_provenance,
        "retrieval_feedback_used": sidecar.retrieval_feedback_used,
        "records": [
            {
                "query_id": record.query_id,
                "source_event_index": record.source_event_index,
                "source_vi": record.source_vi,
                "translation_en": record.translation_en,
            }
            for record in sidecar.records
        ],
    }


def serialize_trake_dev_translation_sidecar(
    sidecar: TRAKEDevTranslationSidecar,
) -> bytes:
    payload = sidecar_to_payload(sidecar)
    records = payload.pop("records")
    lines = ["{"]
    for key, value in payload.items():
        lines.append(
            f"  {json.dumps(key)}: "
            f"{json.dumps(value, ensure_ascii=False)},"
        )
    lines.append('  "records": [')
    for index, record in enumerate(records):
        suffix = "," if index + 1 < len(records) else ""
        lines.append(
            f"    {json.dumps(record, ensure_ascii=False)}{suffix}"
        )
    lines.extend(["  ]", "}"])
    return ("\n".join(lines) + "\n").encode("utf-8")
