"""Strict schema for the internal L21-150 diagnostic benchmark."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeAlias

BENCHMARK_ID = "system_tai-l21-150-diagnostic-v1"
BENCHMARK_ROLE = "DIAGNOSTIC_DEVELOPMENT"
FRAME_GT_STATUS = "PROPOSED_UNTIL_MAPPING_VALIDATED"
TASK_TYPES = ("kis", "qa", "trake")
SPLITS = ("DEV", "HOLDOUT")


class L21150FormatError(ValueError):
    """The benchmark does not satisfy the frozen diagnostic schema."""


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise L21150FormatError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _exact_fields(payload: dict[str, Any], expected: set[str], context: str) -> None:
    actual = set(payload)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise L21150FormatError(
            f"{context} fields mismatch: missing={missing}, extra={extra}"
        )


def _text(value: Any, name: str) -> str:
    if type(value) is not str or not value.strip():
        raise L21150FormatError(f"{name} must be a non-empty string")
    if value != value.strip():
        raise L21150FormatError(f"{name} must not have outer whitespace")
    return value


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int:
        raise L21150FormatError(f"{name} must be an integer")
    if value < minimum:
        raise L21150FormatError(f"{name} must be >= {minimum}")
    return value


def _string_list(value: Any, name: str) -> tuple[str, ...]:
    if type(value) is not list or not value:
        raise L21150FormatError(f"{name} must be a non-empty array")
    result = tuple(_text(item, f"{name}[]") for item in value)
    if len(set(result)) != len(result):
        raise L21150FormatError(f"{name} must not contain duplicates")
    return result


@dataclass(frozen=True, slots=True)
class FrameInterval:
    start_frame_id: int
    end_frame_id: int

    def __post_init__(self) -> None:
        _integer(self.start_frame_id, "start_frame_id")
        _integer(self.end_frame_id, "end_frame_id")
        if self.start_frame_id > self.end_frame_id:
            raise L21150FormatError("start_frame_id must be <= end_frame_id")


@dataclass(frozen=True, slots=True)
class L21150KISQuery:
    query_id: str
    query_vi: str
    video_id: str
    reference_timestamp: str
    proposed_frame_center: int
    proposed_interval: FrameInterval
    branch: str
    difficulty: str
    split: Literal["DEV", "HOLDOUT"]
    task_type: Literal["kis"] = "kis"

    def __post_init__(self) -> None:
        _validate_common_query(self)
        _text(self.query_vi, "query_vi")
        _text(self.reference_timestamp, "reference_timestamp")
        _validate_center(self.proposed_frame_center, self.proposed_interval)


@dataclass(frozen=True, slots=True)
class L21150QAQuery:
    query_id: str
    question_vi: str
    video_id: str
    reference_timestamp: str
    proposed_frame_center: int
    proposed_interval: FrameInterval
    source_answer: str
    canonical_answer: str
    accepted_answers: tuple[str, ...]
    branch: str
    difficulty: str
    split: Literal["DEV", "HOLDOUT"]
    task_type: Literal["qa"] = "qa"

    def __post_init__(self) -> None:
        _validate_common_query(self)
        _text(self.question_vi, "question_vi")
        _text(self.reference_timestamp, "reference_timestamp")
        _validate_center(self.proposed_frame_center, self.proposed_interval)
        _text(self.source_answer, "source_answer")
        _text(self.canonical_answer, "canonical_answer")
        if not isinstance(self.accepted_answers, tuple) or not self.accepted_answers:
            raise L21150FormatError("accepted_answers must be a non-empty tuple")
        for answer in self.accepted_answers:
            _text(answer, "accepted_answers[]")
        if len(set(self.accepted_answers)) != len(self.accepted_answers):
            raise L21150FormatError("accepted_answers must not contain duplicates")


@dataclass(frozen=True, slots=True)
class L21150TRAKEEvent:
    event_index: int
    description_vi: str
    reference_timestamp: str
    proposed_frame_center: int
    proposed_interval: FrameInterval

    def __post_init__(self) -> None:
        _integer(self.event_index, "event_index", minimum=1)
        _text(self.description_vi, "description_vi")
        _text(self.reference_timestamp, "reference_timestamp")
        _validate_center(self.proposed_frame_center, self.proposed_interval)


@dataclass(frozen=True, slots=True)
class L21150TRAKEQuery:
    query_id: str
    video_id: str
    events: tuple[L21150TRAKEEvent, ...]
    branch: str
    difficulty: str
    split: Literal["DEV", "HOLDOUT"]
    task_type: Literal["trake"] = "trake"

    def __post_init__(self) -> None:
        _validate_common_query(self)
        if not isinstance(self.events, tuple) or not self.events:
            raise L21150FormatError("events must be a non-empty tuple")
        if [event.event_index for event in self.events] != list(
            range(1, len(self.events) + 1)
        ):
            raise L21150FormatError("event indexes must be contiguous and ordered")


L21150Query: TypeAlias = L21150KISQuery | L21150QAQuery | L21150TRAKEQuery


def _validate_common_query(query: L21150Query) -> None:
    _text(query.query_id, "query_id")
    _text(query.video_id, "video_id")
    _text(query.branch, "branch")
    _text(query.difficulty, "difficulty")
    if query.split not in SPLITS:
        raise L21150FormatError("split must be DEV or HOLDOUT")


def _validate_center(center: int, interval: FrameInterval) -> None:
    _integer(center, "proposed_frame_center")
    if not isinstance(interval, FrameInterval):
        raise L21150FormatError("proposed_interval must be a FrameInterval")
    if not interval.start_frame_id <= center <= interval.end_frame_id:
        raise L21150FormatError("proposed_frame_center must be inside proposed_interval")


@dataclass(frozen=True, slots=True)
class L21150Benchmark:
    schema_version: int
    benchmark_id: str
    benchmark_role: str
    official_ground_truth: bool
    dataset_scope: str
    frame_gt_status: str
    description: str
    queries: tuple[L21150Query, ...]

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise L21150FormatError("schema_version must be 1")
        if self.benchmark_id != BENCHMARK_ID:
            raise L21150FormatError(f"benchmark_id must be {BENCHMARK_ID}")
        if self.benchmark_role != BENCHMARK_ROLE:
            raise L21150FormatError(f"benchmark_role must be {BENCHMARK_ROLE}")
        if type(self.official_ground_truth) is not bool or self.official_ground_truth:
            raise L21150FormatError("official_ground_truth must be false")
        _text(self.dataset_scope, "dataset_scope")
        if self.frame_gt_status != FRAME_GT_STATUS:
            raise L21150FormatError(f"frame_gt_status must be {FRAME_GT_STATUS}")
        _text(self.description, "description")
        ids = [query.query_id for query in self.queries]
        if len(ids) != len(set(ids)):
            raise L21150FormatError("query_id values must be unique")


def _parse_interval(value: Any, context: str) -> FrameInterval:
    if type(value) is not list or len(value) != 2:
        raise L21150FormatError(f"{context} must be a two-item array")
    return FrameInterval(
        _integer(value[0], f"{context}[0]"),
        _integer(value[1], f"{context}[1]"),
    )


COMMON_FIELDS = {
    "task_type",
    "query_id",
    "video_id",
    "branch",
    "difficulty",
    "split",
}


def _parse_common(payload: dict[str, Any], context: str) -> dict[str, str]:
    split = _text(payload["split"], f"{context}.split")
    if split not in SPLITS:
        raise L21150FormatError(f"{context}.split must be DEV or HOLDOUT")
    return {
        "query_id": _text(payload["query_id"], f"{context}.query_id"),
        "video_id": _text(payload["video_id"], f"{context}.video_id"),
        "branch": _text(payload["branch"], f"{context}.branch"),
        "difficulty": _text(payload["difficulty"], f"{context}.difficulty"),
        "split": split,
    }


def _parse_query(value: Any, index: int) -> L21150Query:
    if type(value) is not dict:
        raise L21150FormatError(f"queries[{index}] must be an object")
    task_type = _text(value.get("task_type"), f"queries[{index}].task_type")
    context = f"queries[{index}]"
    if task_type == "kis":
        _exact_fields(
            value,
            COMMON_FIELDS
            | {
                "query_vi",
                "reference_timestamp",
                "proposed_frame_center",
                "proposed_interval",
            },
            context,
        )
        common = _parse_common(value, context)
        return L21150KISQuery(
            **common,
            query_vi=_text(value["query_vi"], f"{context}.query_vi"),
            reference_timestamp=_text(
                value["reference_timestamp"], f"{context}.reference_timestamp"
            ),
            proposed_frame_center=_integer(
                value["proposed_frame_center"], f"{context}.proposed_frame_center"
            ),
            proposed_interval=_parse_interval(
                value["proposed_interval"], f"{context}.proposed_interval"
            ),
        )
    if task_type == "qa":
        _exact_fields(
            value,
            COMMON_FIELDS
            | {
                "question_vi",
                "reference_timestamp",
                "proposed_frame_center",
                "proposed_interval",
                "source_answer",
                "canonical_answer",
                "accepted_answers",
            },
            context,
        )
        common = _parse_common(value, context)
        return L21150QAQuery(
            **common,
            question_vi=_text(value["question_vi"], f"{context}.question_vi"),
            reference_timestamp=_text(
                value["reference_timestamp"], f"{context}.reference_timestamp"
            ),
            proposed_frame_center=_integer(
                value["proposed_frame_center"], f"{context}.proposed_frame_center"
            ),
            proposed_interval=_parse_interval(
                value["proposed_interval"], f"{context}.proposed_interval"
            ),
            source_answer=_text(value["source_answer"], f"{context}.source_answer"),
            canonical_answer=_text(
                value["canonical_answer"], f"{context}.canonical_answer"
            ),
            accepted_answers=_string_list(
                value["accepted_answers"], f"{context}.accepted_answers"
            ),
        )
    if task_type == "trake":
        _exact_fields(value, COMMON_FIELDS | {"events"}, context)
        common = _parse_common(value, context)
        events_payload = value["events"]
        if type(events_payload) is not list or not events_payload:
            raise L21150FormatError(f"{context}.events must be a non-empty array")
        events: list[L21150TRAKEEvent] = []
        for event_offset, event_payload in enumerate(events_payload):
            event_context = f"{context}.events[{event_offset}]"
            if type(event_payload) is not dict:
                raise L21150FormatError(f"{event_context} must be an object")
            _exact_fields(
                event_payload,
                {
                    "event_index",
                    "description_vi",
                    "reference_timestamp",
                    "proposed_frame_center",
                    "proposed_interval",
                },
                event_context,
            )
            event_index = _integer(
                event_payload["event_index"], f"{event_context}.event_index", minimum=1
            )
            if event_index != event_offset + 1:
                raise L21150FormatError(f"{event_context}.event_index must be contiguous")
            events.append(
                L21150TRAKEEvent(
                    event_index=event_index,
                    description_vi=_text(
                        event_payload["description_vi"],
                        f"{event_context}.description_vi",
                    ),
                    reference_timestamp=_text(
                        event_payload["reference_timestamp"],
                        f"{event_context}.reference_timestamp",
                    ),
                    proposed_frame_center=_integer(
                        event_payload["proposed_frame_center"],
                        f"{event_context}.proposed_frame_center",
                    ),
                    proposed_interval=_parse_interval(
                        event_payload["proposed_interval"],
                        f"{event_context}.proposed_interval",
                    ),
                )
            )
        return L21150TRAKEQuery(**common, events=tuple(events))
    raise L21150FormatError(f"{context}.task_type is unsupported: {task_type}")


TOP_LEVEL_FIELDS = {
    "schema_version",
    "benchmark_id",
    "benchmark_role",
    "official_ground_truth",
    "dataset_scope",
    "frame_gt_status",
    "description",
    "queries",
}


def parse_l21_150_payload(payload: Any) -> L21150Benchmark:
    if type(payload) is not dict:
        raise L21150FormatError("benchmark root must be an object")
    _exact_fields(payload, TOP_LEVEL_FIELDS, "benchmark")
    queries_payload = payload["queries"]
    if type(queries_payload) is not list:
        raise L21150FormatError("queries must be an array")
    return L21150Benchmark(
        schema_version=_integer(payload["schema_version"], "schema_version", minimum=1),
        benchmark_id=_text(payload["benchmark_id"], "benchmark_id"),
        benchmark_role=_text(payload["benchmark_role"], "benchmark_role"),
        official_ground_truth=payload["official_ground_truth"],
        dataset_scope=_text(payload["dataset_scope"], "dataset_scope"),
        frame_gt_status=_text(payload["frame_gt_status"], "frame_gt_status"),
        description=_text(payload["description"], "description"),
        queries=tuple(
            _parse_query(query_payload, index)
            for index, query_payload in enumerate(queries_payload)
        ),
    )


def load_l21_150_benchmark(path: Path) -> L21150Benchmark:
    raw = Path(path).read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        raise L21150FormatError("benchmark JSON must not contain a UTF-8 BOM")
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise L21150FormatError("benchmark JSON must be valid UTF-8") from exc
    try:
        payload = json.loads(text, object_pairs_hook=_reject_duplicate_pairs)
    except json.JSONDecodeError as exc:
        raise L21150FormatError(f"invalid benchmark JSON: {exc}") from exc
    return parse_l21_150_payload(payload)


def _interval_payload(interval: FrameInterval) -> list[int]:
    return [interval.start_frame_id, interval.end_frame_id]


def query_to_payload(query: L21150Query) -> dict[str, Any]:
    common: dict[str, Any] = {
        "task_type": query.task_type,
        "query_id": query.query_id,
        "video_id": query.video_id,
        "branch": query.branch,
        "difficulty": query.difficulty,
        "split": query.split,
    }
    if isinstance(query, L21150KISQuery):
        common.update(
            {
                "query_vi": query.query_vi,
                "reference_timestamp": query.reference_timestamp,
                "proposed_frame_center": query.proposed_frame_center,
                "proposed_interval": _interval_payload(query.proposed_interval),
            }
        )
    elif isinstance(query, L21150QAQuery):
        common.update(
            {
                "question_vi": query.question_vi,
                "reference_timestamp": query.reference_timestamp,
                "proposed_frame_center": query.proposed_frame_center,
                "proposed_interval": _interval_payload(query.proposed_interval),
                "source_answer": query.source_answer,
                "canonical_answer": query.canonical_answer,
                "accepted_answers": list(query.accepted_answers),
            }
        )
    else:
        common["events"] = [
            {
                "event_index": event.event_index,
                "description_vi": event.description_vi,
                "reference_timestamp": event.reference_timestamp,
                "proposed_frame_center": event.proposed_frame_center,
                "proposed_interval": _interval_payload(event.proposed_interval),
            }
            for event in query.events
        ]
    return common


def benchmark_to_payload(benchmark: L21150Benchmark) -> dict[str, Any]:
    return {
        "schema_version": benchmark.schema_version,
        "benchmark_id": benchmark.benchmark_id,
        "benchmark_role": benchmark.benchmark_role,
        "official_ground_truth": benchmark.official_ground_truth,
        "dataset_scope": benchmark.dataset_scope,
        "frame_gt_status": benchmark.frame_gt_status,
        "description": benchmark.description,
        "queries": [query_to_payload(query) for query in benchmark.queries],
    }


def serialize_l21_150_benchmark(benchmark: L21150Benchmark) -> bytes:
    payload = benchmark_to_payload(benchmark)
    return (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def write_l21_150_benchmark(benchmark: L21150Benchmark, path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(serialize_l21_150_benchmark(benchmark))
