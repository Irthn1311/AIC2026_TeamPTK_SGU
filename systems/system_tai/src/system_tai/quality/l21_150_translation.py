"""Strict experiment-input sidecar for the L21-150 KIS DEV Q2 pilot."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .l21_150_schema import L21150Benchmark, L21150KISQuery

SIDECAR_SCHEMA_VERSION = 1
SIDECAR_ROLE = "EXPERIMENT_INPUT"
SIDECAR_TASK = "kis"
SIDECAR_SPLIT = "dev"
TRANSLATION_STATUS = "REVIEWED_FROZEN"
TRANSLATION_POLICY = "FAITHFUL_ENGLISH_TRANSLATION_WITHOUT_RETRIEVAL_FEEDBACK"
TRANSLATION_VERSION = "q2-kis-dev-en-v1"


class KISTranslationSidecarError(ValueError):
    """The sidecar violates its frozen experiment-input contract."""


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise KISTranslationSidecarError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _exact_fields(value: dict[str, Any], expected: set[str], context: str) -> None:
    missing = sorted(expected - set(value))
    extra = sorted(set(value) - expected)
    if missing or extra:
        raise KISTranslationSidecarError(
            f"{context} fields mismatch: missing={missing}, extra={extra}"
        )


def _text(value: Any, name: str) -> str:
    if type(value) is not str or not value.strip() or value != value.strip():
        raise KISTranslationSidecarError(
            f"{name} must be a non-empty string without outer whitespace"
        )
    return value


@dataclass(frozen=True, slots=True)
class KISTranslationRecord:
    query_id: str
    source_vi: str
    translation_en: str


@dataclass(frozen=True, slots=True)
class KISDevTranslationSidecar:
    schema_version: int
    sidecar_role: str
    benchmark_id: str
    benchmark_sha256: str
    task: str
    split: str
    query_count: int
    translation_policy: str
    translation_version: str
    translation_status: str
    retrieval_feedback_used: bool
    records: tuple[KISTranslationRecord, ...]

    @property
    def translations(self) -> dict[str, str]:
        return {record.query_id: record.translation_en for record in self.records}


TOP_LEVEL_FIELDS = {
    "schema_version",
    "sidecar_role",
    "benchmark_id",
    "benchmark_sha256",
    "task",
    "split",
    "query_count",
    "translation_policy",
    "translation_version",
    "translation_status",
    "retrieval_feedback_used",
    "records",
}
RECORD_FIELDS = {"query_id", "source_vi", "translation_en"}


def validate_kis_dev_translation_payload(
    payload: Any,
    benchmark: L21150Benchmark,
    *,
    benchmark_sha256: str,
) -> KISDevTranslationSidecar:
    if type(payload) is not dict:
        raise KISTranslationSidecarError("sidecar root must be an object")
    _exact_fields(payload, TOP_LEVEL_FIELDS, "sidecar")
    if payload["schema_version"] != SIDECAR_SCHEMA_VERSION:
        raise KISTranslationSidecarError(
            f"schema_version must be {SIDECAR_SCHEMA_VERSION}"
        )
    if payload["sidecar_role"] != SIDECAR_ROLE:
        raise KISTranslationSidecarError(f"sidecar_role must be {SIDECAR_ROLE}")
    if payload["benchmark_id"] != benchmark.benchmark_id:
        raise KISTranslationSidecarError("benchmark_id mismatch")
    if payload["benchmark_sha256"] != benchmark_sha256:
        raise KISTranslationSidecarError("benchmark_sha256 mismatch")
    if payload["task"] != SIDECAR_TASK:
        raise KISTranslationSidecarError(f"task must be {SIDECAR_TASK}")
    if payload["split"] != SIDECAR_SPLIT:
        raise KISTranslationSidecarError(f"split must be {SIDECAR_SPLIT}")
    if payload["translation_policy"] != TRANSLATION_POLICY:
        raise KISTranslationSidecarError(
            f"translation_policy must be {TRANSLATION_POLICY}"
        )
    if payload["translation_version"] != TRANSLATION_VERSION:
        raise KISTranslationSidecarError(
            f"translation_version must be {TRANSLATION_VERSION}"
        )
    if payload["translation_status"] != TRANSLATION_STATUS:
        raise KISTranslationSidecarError(
            f"translation_status must be {TRANSLATION_STATUS}"
        )
    if payload["retrieval_feedback_used"] is not False:
        raise KISTranslationSidecarError("retrieval_feedback_used must be false")
    records_payload = payload["records"]
    if type(records_payload) is not list:
        raise KISTranslationSidecarError("records must be an array")
    if type(payload["query_count"]) is not int:
        raise KISTranslationSidecarError("query_count must be an integer")
    if payload["query_count"] != len(records_payload):
        raise KISTranslationSidecarError("query_count does not match records")

    expected_queries = [
        query
        for query in benchmark.queries
        if isinstance(query, L21150KISQuery) and query.split == "DEV"
    ]
    expected_by_id = {query.query_id: query for query in expected_queries}
    holdout_ids = {
        query.query_id
        for query in benchmark.queries
        if isinstance(query, L21150KISQuery) and query.split == "HOLDOUT"
    }
    records: list[KISTranslationRecord] = []
    seen: set[str] = set()
    for index, value in enumerate(records_payload):
        if type(value) is not dict:
            raise KISTranslationSidecarError(f"records[{index}] must be an object")
        _exact_fields(value, RECORD_FIELDS, f"records[{index}]")
        query_id = _text(value["query_id"], f"records[{index}].query_id")
        if query_id in seen:
            raise KISTranslationSidecarError(f"duplicate query_id: {query_id}")
        seen.add(query_id)
        if query_id in holdout_ids:
            raise KISTranslationSidecarError(f"HOLDOUT query is forbidden: {query_id}")
        query = expected_by_id.get(query_id)
        if query is None:
            raise KISTranslationSidecarError(f"extra or unknown DEV KIS query: {query_id}")
        source_vi = _text(value["source_vi"], f"records[{index}].source_vi")
        if source_vi != query.query_vi:
            raise KISTranslationSidecarError(f"source_vi mismatch for {query_id}")
        records.append(
            KISTranslationRecord(
                query_id=query_id,
                source_vi=source_vi,
                translation_en=_text(
                    value["translation_en"], f"records[{index}].translation_en"
                ),
            )
        )
    expected_ids = [query.query_id for query in expected_queries]
    actual_ids = [record.query_id for record in records]
    if actual_ids != expected_ids:
        missing = sorted(set(expected_ids) - set(actual_ids))
        extra = sorted(set(actual_ids) - set(expected_ids))
        raise KISTranslationSidecarError(
            "records must exactly follow benchmark DEV KIS order: "
            f"missing={missing}, extra={extra}"
        )

    return KISDevTranslationSidecar(
        schema_version=payload["schema_version"],
        sidecar_role=payload["sidecar_role"],
        benchmark_id=payload["benchmark_id"],
        benchmark_sha256=payload["benchmark_sha256"],
        task=payload["task"],
        split=payload["split"],
        query_count=payload["query_count"],
        translation_policy=payload["translation_policy"],
        translation_version=payload["translation_version"],
        translation_status=payload["translation_status"],
        retrieval_feedback_used=payload["retrieval_feedback_used"],
        records=tuple(records),
    )


def load_kis_dev_translation_sidecar(
    path: Path,
    benchmark: L21150Benchmark,
    benchmark_path: Path,
) -> KISDevTranslationSidecar:
    raw = Path(path).read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        raise KISTranslationSidecarError("sidecar must not contain a UTF-8 BOM")
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise KISTranslationSidecarError("sidecar must be valid UTF-8") from exc
    try:
        payload = json.loads(text, object_pairs_hook=_reject_duplicate_pairs)
    except json.JSONDecodeError as exc:
        raise KISTranslationSidecarError(f"invalid sidecar JSON: {exc}") from exc
    benchmark_sha256 = hashlib.sha256(Path(benchmark_path).read_bytes()).hexdigest()
    return validate_kis_dev_translation_payload(
        payload,
        benchmark,
        benchmark_sha256=benchmark_sha256,
    )


def sidecar_to_payload(sidecar: KISDevTranslationSidecar) -> dict[str, Any]:
    return {
        "schema_version": sidecar.schema_version,
        "sidecar_role": sidecar.sidecar_role,
        "benchmark_id": sidecar.benchmark_id,
        "benchmark_sha256": sidecar.benchmark_sha256,
        "task": sidecar.task,
        "split": sidecar.split,
        "query_count": sidecar.query_count,
        "translation_policy": sidecar.translation_policy,
        "translation_version": sidecar.translation_version,
        "translation_status": sidecar.translation_status,
        "retrieval_feedback_used": sidecar.retrieval_feedback_used,
        "records": [
            {
                "query_id": record.query_id,
                "source_vi": record.source_vi,
                "translation_en": record.translation_en,
            }
            for record in sidecar.records
        ],
    }


def serialize_kis_dev_translation_sidecar(sidecar: KISDevTranslationSidecar) -> bytes:
    return (
        json.dumps(sidecar_to_payload(sidecar), ensure_ascii=False, indent=2) + "\n"
    ).encode("utf-8")
