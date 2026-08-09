"""Strict immutable contracts for the unified semantic quality benchmark."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, TypeAlias, cast

from system_tai.preliminary.schemas import (
    KISGroundTruth,
    QAGroundTruth,
    TRAKEGroundTruth,
)


class QualityTaskType(StrEnum):
    KIS = "kis"
    QA = "qa"
    TRAKE = "trake"


class AnnotationStatus(StrEnum):
    DRAFT = "draft"
    VERIFIED = "verified"


class LabelOrigin(StrEnum):
    UNLABELED = "unlabeled"
    HUMAN_RAW_VIDEO = "human_raw_video"
    OFFICIAL = "official"
    SYNTHETIC = "synthetic"


class Difficulty(StrEnum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    UNKNOWN = "unknown"


class QualityBenchmarkFormatError(ValueError):
    """Raised when a quality benchmark violates its strict JSON contract."""


def _reject_duplicate_object_pairs(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise QualityBenchmarkFormatError(
                f"duplicate JSON object key: {key!r}"
            )
        result[key] = value
    return result


def _require_text(value: object, name: str, *, allow_empty: bool = False) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be a string")
    resolved = cast(str, value)
    if not allow_empty and not resolved.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return resolved


def _require_optional_text(value: object, name: str) -> str | None:
    if value is None:
        return None
    return _require_text(value, name)


def _validate_common(query: QualityQuery, expected_task: QualityTaskType) -> None:
    _require_text(query.query_id, "query_id")
    if query.task_type is not expected_task:
        raise ValueError(f"task_type must be {expected_task.value!r}")
    if type(query.annotation_status) is not AnnotationStatus:
        raise TypeError("annotation_status must be AnnotationStatus")
    if type(query.label_origin) is not LabelOrigin:
        raise TypeError("label_origin must be LabelOrigin")
    if type(query.difficulty) is not Difficulty:
        raise TypeError("difficulty must be Difficulty")
    if type(query.tags) is not tuple:
        raise TypeError("tags must be a tuple")
    for tag in query.tags:
        _require_text(tag, "tag")
    if len(set(query.tags)) != len(query.tags):
        raise ValueError("tags must be unique")
    _require_text(query.annotation_notes, "annotation_notes", allow_empty=True)
    _require_text(query.source_reference, "source_reference", allow_empty=True)
    if query.annotation_status is AnnotationStatus.VERIFIED:
        if query.ground_truth is None:
            raise ValueError("verified query requires ground_truth")
        if query.label_origin is LabelOrigin.UNLABELED:
            raise ValueError("verified query cannot use label_origin='unlabeled'")
        if not query.source_reference.strip():
            raise ValueError("verified query requires non-empty source_reference")


@dataclass(frozen=True, slots=True)
class KISQualityQuery:
    query_id: str
    task_type: QualityTaskType
    annotation_status: AnnotationStatus
    label_origin: LabelOrigin
    difficulty: Difficulty
    tags: tuple[str, ...]
    annotation_notes: str
    source_reference: str
    query_vi: str
    query_en: str | None
    query_en_expansion: str | None
    ground_truth: KISGroundTruth | None

    def __post_init__(self) -> None:
        _validate_common(self, QualityTaskType.KIS)
        _require_text(self.query_vi, "query_vi")
        _require_optional_text(self.query_en, "query_en")
        _require_optional_text(self.query_en_expansion, "query_en_expansion")
        if self.ground_truth is not None:
            if type(self.ground_truth) is not KISGroundTruth:
                raise TypeError("ground_truth must be KISGroundTruth or None")
            if self.ground_truth.query_id != self.query_id:
                raise ValueError("ground_truth query_id must match query_id")


@dataclass(frozen=True, slots=True)
class QAQualityQuery:
    query_id: str
    task_type: QualityTaskType
    annotation_status: AnnotationStatus
    label_origin: LabelOrigin
    difficulty: Difficulty
    tags: tuple[str, ...]
    annotation_notes: str
    source_reference: str
    event_description: str
    question: str
    event_description_en: str | None
    question_en: str | None
    ground_truth: QAGroundTruth | None

    def __post_init__(self) -> None:
        _validate_common(self, QualityTaskType.QA)
        _require_text(self.event_description, "event_description")
        _require_text(self.question, "question")
        _require_optional_text(self.event_description_en, "event_description_en")
        _require_optional_text(self.question_en, "question_en")
        if self.ground_truth is not None:
            if type(self.ground_truth) is not QAGroundTruth:
                raise TypeError("ground_truth must be QAGroundTruth or None")
            if self.ground_truth.query_id != self.query_id:
                raise ValueError("ground_truth query_id must match query_id")


@dataclass(frozen=True, slots=True)
class QualityTRAKEEvent:
    description: str
    description_en: str | None

    def __post_init__(self) -> None:
        _require_text(self.description, "event description")
        _require_optional_text(self.description_en, "event description_en")


@dataclass(frozen=True, slots=True)
class TRAKEQualityQuery:
    query_id: str
    task_type: QualityTaskType
    annotation_status: AnnotationStatus
    label_origin: LabelOrigin
    difficulty: Difficulty
    tags: tuple[str, ...]
    annotation_notes: str
    source_reference: str
    events: tuple[QualityTRAKEEvent, ...]
    ground_truth: TRAKEGroundTruth | None

    def __post_init__(self) -> None:
        _validate_common(self, QualityTaskType.TRAKE)
        if type(self.events) is not tuple or not self.events:
            raise ValueError("events must be a non-empty tuple")
        if any(type(event) is not QualityTRAKEEvent for event in self.events):
            raise TypeError("events must contain only QualityTRAKEEvent values")
        if self.ground_truth is not None:
            if type(self.ground_truth) is not TRAKEGroundTruth:
                raise TypeError("ground_truth must be TRAKEGroundTruth or None")
            if self.ground_truth.query_id != self.query_id:
                raise ValueError("ground_truth query_id must match query_id")
            if len(self.events) != len(self.ground_truth.event_intervals):
                raise ValueError("TRAKE events and ground-truth intervals must have equal length")


QualityQuery: TypeAlias = KISQualityQuery | QAQualityQuery | TRAKEQualityQuery


@dataclass(frozen=True, slots=True)
class QualityBenchmark:
    schema_version: int
    benchmark_id: str
    description: str
    queries: tuple[QualityQuery, ...]

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("schema_version must be the integer 1")
        _require_text(self.benchmark_id, "benchmark_id")
        _require_text(self.description, "description", allow_empty=True)
        if type(self.queries) is not tuple:
            raise TypeError("queries must be a tuple")
        allowed = (KISQualityQuery, QAQualityQuery, TRAKEQualityQuery)
        if any(type(query) not in allowed for query in self.queries):
            raise TypeError("queries contains an unsupported query type")
        query_ids = tuple(query.query_id for query in self.queries)
        if len(set(query_ids)) != len(query_ids):
            raise ValueError("query_id must be globally unique within benchmark")


_TOP_FIELDS = frozenset({"schema_version", "benchmark_id", "description", "queries"})
_COMMON_QUERY_FIELDS = frozenset(
    {
        "task_type",
        "query_id",
        "annotation_status",
        "label_origin",
        "difficulty",
        "tags",
        "annotation_notes",
        "source_reference",
    }
)
_KIS_FIELDS = _COMMON_QUERY_FIELDS | {
    "query_vi",
    "query_en",
    "query_en_expansion",
    "ground_truth",
}
_QA_FIELDS = _COMMON_QUERY_FIELDS | {
    "event_description",
    "question",
    "event_description_en",
    "question_en",
    "ground_truth",
}
_TRAKE_FIELDS = _COMMON_QUERY_FIELDS | {"events", "ground_truth"}


def _require_object(value: object, context: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise QualityBenchmarkFormatError(f"{context} must be a JSON object")
    result = cast(dict[object, object], value)
    if any(type(key) is not str for key in result):
        raise QualityBenchmarkFormatError(f"{context} field names must be strings")
    return cast(dict[str, Any], value)


def _require_fields(value: dict[str, Any], fields: frozenset[str], context: str) -> None:
    missing = sorted(fields - set(value))
    unknown = sorted(set(value) - fields)
    if missing or unknown:
        raise QualityBenchmarkFormatError(
            f"{context} fields mismatch: missing={missing}, unknown={unknown}"
        )


def _enum_value(enum_type: type[StrEnum], value: object, name: str) -> StrEnum:
    if type(value) is not str:
        raise QualityBenchmarkFormatError(f"{name} must be a string enum value")
    try:
        return enum_type(value)
    except ValueError as exc:
        raise QualityBenchmarkFormatError(f"invalid {name}: {value!r}") from exc


def _strict_int(value: object, name: str) -> int:
    if type(value) is not int:
        raise QualityBenchmarkFormatError(f"{name} must be an integer")
    return cast(int, value)


def _parse_common(raw: dict[str, Any]) -> dict[str, Any]:
    tags_value = raw["tags"]
    if type(tags_value) is not list:
        raise QualityBenchmarkFormatError("tags must be a JSON list")
    tags = tuple(_require_text(tag, "tag") for tag in tags_value)
    return {
        "query_id": _require_text(raw["query_id"], "query_id"),
        "task_type": _enum_value(QualityTaskType, raw["task_type"], "task_type"),
        "annotation_status": _enum_value(
            AnnotationStatus, raw["annotation_status"], "annotation_status"
        ),
        "label_origin": _enum_value(LabelOrigin, raw["label_origin"], "label_origin"),
        "difficulty": _enum_value(Difficulty, raw["difficulty"], "difficulty"),
        "tags": tags,
        "annotation_notes": _require_text(
            raw["annotation_notes"], "annotation_notes", allow_empty=True
        ),
        "source_reference": _require_text(
            raw["source_reference"], "source_reference", allow_empty=True
        ),
    }


def _parse_kis_ground_truth(value: object, query_id: str) -> KISGroundTruth | None:
    if value is None:
        return None
    raw = _require_object(value, "KIS ground_truth")
    fields = frozenset({"video_id", "start_frame_id", "end_frame_id"})
    _require_fields(raw, fields, "KIS ground_truth")
    return KISGroundTruth(
        query_id,
        _require_text(raw["video_id"], "ground_truth.video_id"),
        _strict_int(raw["start_frame_id"], "ground_truth.start_frame_id"),
        _strict_int(raw["end_frame_id"], "ground_truth.end_frame_id"),
    )


def _parse_qa_ground_truth(value: object, query_id: str) -> QAGroundTruth | None:
    if value is None:
        return None
    raw = _require_object(value, "Q&A ground_truth")
    fields = frozenset(
        {"video_id", "start_frame_id", "end_frame_id", "accepted_answers"}
    )
    _require_fields(raw, fields, "Q&A ground_truth")
    answers_value = raw["accepted_answers"]
    if type(answers_value) is not list:
        raise QualityBenchmarkFormatError("accepted_answers must be a JSON list")
    answers = tuple(_require_text(answer, "accepted_answer") for answer in answers_value)
    return QAGroundTruth(
        query_id,
        _require_text(raw["video_id"], "ground_truth.video_id"),
        _strict_int(raw["start_frame_id"], "ground_truth.start_frame_id"),
        _strict_int(raw["end_frame_id"], "ground_truth.end_frame_id"),
        answers,
    )


def _parse_trake_ground_truth(value: object, query_id: str) -> TRAKEGroundTruth | None:
    if value is None:
        return None
    raw = _require_object(value, "TRAKE ground_truth")
    fields = frozenset({"video_id", "event_intervals"})
    _require_fields(raw, fields, "TRAKE ground_truth")
    raw_intervals = raw["event_intervals"]
    if type(raw_intervals) is not list or not raw_intervals:
        raise QualityBenchmarkFormatError("event_intervals must be a non-empty JSON list")
    intervals: list[tuple[int, int]] = []
    for index, raw_interval in enumerate(raw_intervals):
        if type(raw_interval) is not list or len(raw_interval) != 2:
            raise QualityBenchmarkFormatError(
                f"event_intervals[{index}] must be a two-integer JSON list"
            )
        intervals.append(
            (
                _strict_int(raw_interval[0], f"event_intervals[{index}][0]"),
                _strict_int(raw_interval[1], f"event_intervals[{index}][1]"),
            )
        )
    return TRAKEGroundTruth(
        query_id,
        _require_text(raw["video_id"], "ground_truth.video_id"),
        tuple(intervals),
    )


def _parse_quality_query(value: object, index: int) -> QualityQuery:
    raw = _require_object(value, f"queries[{index}]")
    task = _enum_value(QualityTaskType, raw.get("task_type"), "task_type")
    expected_fields = {
        QualityTaskType.KIS: _KIS_FIELDS,
        QualityTaskType.QA: _QA_FIELDS,
        QualityTaskType.TRAKE: _TRAKE_FIELDS,
    }[task]
    _require_fields(raw, frozenset(expected_fields), f"queries[{index}]")
    common = _parse_common(raw)
    query_id = cast(str, common["query_id"])
    if task is QualityTaskType.KIS:
        return KISQualityQuery(
            **common,
            query_vi=_require_text(raw["query_vi"], "query_vi"),
            query_en=_require_optional_text(raw["query_en"], "query_en"),
            query_en_expansion=_require_optional_text(
                raw["query_en_expansion"], "query_en_expansion"
            ),
            ground_truth=_parse_kis_ground_truth(raw["ground_truth"], query_id),
        )
    if task is QualityTaskType.QA:
        return QAQualityQuery(
            **common,
            event_description=_require_text(
                raw["event_description"], "event_description"
            ),
            question=_require_text(raw["question"], "question"),
            event_description_en=_require_optional_text(
                raw["event_description_en"], "event_description_en"
            ),
            question_en=_require_optional_text(raw["question_en"], "question_en"),
            ground_truth=_parse_qa_ground_truth(raw["ground_truth"], query_id),
        )

    events_value = raw["events"]
    if type(events_value) is not list or not events_value:
        raise QualityBenchmarkFormatError("events must be a non-empty JSON list")
    events: list[QualityTRAKEEvent] = []
    for event_index, event_value in enumerate(events_value):
        event = _require_object(event_value, f"events[{event_index}]")
        _require_fields(
            event,
            frozenset({"description", "description_en"}),
            f"events[{event_index}]",
        )
        events.append(
            QualityTRAKEEvent(
                _require_text(event["description"], "event.description"),
                _require_optional_text(event["description_en"], "event.description_en"),
            )
        )
    return TRAKEQualityQuery(
        **common,
        events=tuple(events),
        ground_truth=_parse_trake_ground_truth(raw["ground_truth"], query_id),
    )


def parse_quality_benchmark_payload(payload: object) -> QualityBenchmark:
    """Parse one strict schema-v1 JSON-compatible benchmark object."""

    try:
        raw = _require_object(payload, "benchmark")
        _require_fields(raw, _TOP_FIELDS, "benchmark")
        schema_version = _strict_int(raw["schema_version"], "schema_version")
        raw_queries = raw["queries"]
        if type(raw_queries) is not list:
            raise QualityBenchmarkFormatError("queries must be a JSON list")
        return QualityBenchmark(
            schema_version=schema_version,
            benchmark_id=_require_text(raw["benchmark_id"], "benchmark_id"),
            description=_require_text(raw["description"], "description", allow_empty=True),
            queries=tuple(
                _parse_quality_query(query, index)
                for index, query in enumerate(raw_queries)
            ),
        )
    except QualityBenchmarkFormatError:
        raise
    except (TypeError, ValueError) as exc:
        raise QualityBenchmarkFormatError(str(exc)) from exc


def load_quality_benchmark_json(path: Path) -> QualityBenchmark:
    """Load strict UTF-8 JSON without BOM or schema repair."""

    source = Path(path)
    try:
        payload = source.read_bytes()
    except OSError as exc:
        raise QualityBenchmarkFormatError(f"cannot read benchmark: {exc}") from exc
    try:
        text = payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise QualityBenchmarkFormatError(f"benchmark is not valid UTF-8: {exc}") from exc
    if text.startswith("\ufeff"):
        raise QualityBenchmarkFormatError("UTF-8 BOM is not permitted")
    try:
        decoded = json.loads(text, object_pairs_hook=_reject_duplicate_object_pairs)
    except json.JSONDecodeError as exc:
        raise QualityBenchmarkFormatError(f"invalid benchmark JSON: {exc.msg}") from exc
    return parse_quality_benchmark_payload(decoded)
