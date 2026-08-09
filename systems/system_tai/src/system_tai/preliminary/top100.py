"""Canonical internal Top-100 checkpoint boundary for preliminary tasks."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, TypeAlias, cast

from .schemas import KISPrediction, QAPrediction, TRAKEPrediction
from .validation import TASK_PREDICTION_TYPES, validate_ranked_top100

TaskType: TypeAlias = Literal["kis", "qa", "trake"]
PredictionType: TypeAlias = KISPrediction | QAPrediction | TRAKEPrediction

SUPPORTED_TASKS: tuple[TaskType, ...] = ("kis", "qa", "trake")
TASK_RECORD_FIELDS: Mapping[TaskType, tuple[str, ...]] = MappingProxyType(
    {
        "kis": ("query_id", "rank", "video_id", "frame_id"),
        "qa": ("query_id", "rank", "video_id", "frame_id", "answer"),
        "trake": ("query_id", "rank", "video_id", "frame_ids"),
    }
)


def _require_task(task_type: str) -> TaskType:
    if task_type not in SUPPORTED_TASKS:
        raise ValueError(f"unsupported task_type: {task_type!r}")
    return cast(TaskType, task_type)


def _require_nonempty_text(value: Any, name: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _require_int(value: Any, name: str, *, minimum: int) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


def _expected_prediction_type(task_type: TaskType) -> type[PredictionType]:
    return cast(type[PredictionType], TASK_PREDICTION_TYPES[task_type])


@dataclass(frozen=True, slots=True)
class RankedTop100Query:
    """One query's predictions, preserving caller-provided physical order and ranks."""

    task_type: TaskType
    query_id: str
    predictions: tuple[PredictionType, ...]

    def __post_init__(self) -> None:
        _require_task(self.task_type)
        _require_nonempty_text(self.query_id, "query_id")
        if type(self.predictions) is not tuple:
            raise TypeError("predictions must be a tuple")
        validate_top100_query(self)


@dataclass(frozen=True, slots=True)
class RankedTop100Dataset:
    """An ordered, single-task collection of ranked query predictions."""

    task_type: TaskType
    queries: tuple[RankedTop100Query, ...]

    def __post_init__(self) -> None:
        _require_task(self.task_type)
        if type(self.queries) is not tuple:
            raise TypeError("queries must be a tuple")
        _validate_dataset_structure(self)

    def get_query(self, query_id: str) -> RankedTop100Query:
        for query in self.queries:
            if query.query_id == query_id:
                return query
        raise KeyError(query_id)

    def predictions_for(self, query_id: str) -> tuple[PredictionType, ...]:
        return self.get_query(query_id).predictions


@dataclass(frozen=True, slots=True)
class Top100ExportSummary:
    destination: Path
    task_type: TaskType
    query_count: int
    record_count: int


@dataclass(frozen=True, slots=True)
class Top100ValidationIssue:
    code: str
    message: str
    line_number: int | None = None
    query_id: str | None = None


@dataclass(frozen=True, slots=True)
class Top100ValidationReport:
    valid: bool
    errors: tuple[Top100ValidationIssue, ...]


class Top100FormatError(ValueError):
    """Strict file-boundary error with optional record location."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        line_number: int | None = None,
        query_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.line_number = line_number
        self.query_id = query_id


def validate_top100_query(
    query: RankedTop100Query,
    *,
    expected_trake_event_count: int | None = None,
) -> None:
    """Fail closed using P0-A validation plus optional TRAKE event-count validation."""

    task_type = _require_task(query.task_type)
    expected_type = _expected_prediction_type(task_type)
    for index, prediction in enumerate(query.predictions):
        if type(prediction) is not expected_type:
            raise TypeError(
                f"prediction {index} for task {task_type!r} must be "
                f"{expected_type.__name__}, got {type(prediction).__name__}"
            )
    errors = validate_ranked_top100(
        list(query.predictions),
        expected_task=task_type,
        expected_query_id=query.query_id,
    )
    if errors:
        messages = "; ".join(error.message for error in errors)
        raise ValueError(f"invalid Top-100 query {query.query_id!r}: {messages}")

    if expected_trake_event_count is not None:
        if task_type != "trake":
            raise ValueError("expected_trake_event_count is valid only for task 'trake'")
        expected_count = _require_int(
            expected_trake_event_count,
            "expected_trake_event_count",
            minimum=1,
        )
        for prediction in query.predictions:
            trake_prediction = cast(TRAKEPrediction, prediction)
            if len(trake_prediction.frame_ids) != expected_count:
                raise ValueError(
                    f"TRAKE event-count mismatch for query {query.query_id!r}: "
                    f"expected {expected_count}, got {len(trake_prediction.frame_ids)}"
                )


def _validate_dataset_structure(dataset: RankedTop100Dataset) -> None:
    seen_query_ids: set[str] = set()
    for index, query in enumerate(dataset.queries):
        if type(query) is not RankedTop100Query:
            raise TypeError(
                f"query {index} must be RankedTop100Query, got {type(query).__name__}"
            )
        if query.task_type != dataset.task_type:
            raise ValueError(
                f"query {query.query_id!r} task {query.task_type!r} does not match "
                f"dataset task {dataset.task_type!r}"
            )
        if query.query_id in seen_query_ids:
            raise ValueError(f"duplicate dataset query_id: {query.query_id!r}")
        seen_query_ids.add(query.query_id)


def _validated_expected_query_ids(expected_query_ids: Sequence[str]) -> tuple[str, ...]:
    if isinstance(expected_query_ids, (str, bytes)):
        raise TypeError("expected_query_ids must be a sequence of query IDs")
    resolved = tuple(expected_query_ids)
    for query_id in resolved:
        _require_nonempty_text(query_id, "expected query_id")
    if len(set(resolved)) != len(resolved):
        raise ValueError("expected_query_ids contains duplicates")
    return resolved


def validate_top100_dataset(
    dataset: RankedTop100Dataset,
    *,
    expected_query_ids: Sequence[str] | None = None,
    expected_trake_event_counts: Mapping[str, int] | None = None,
) -> None:
    """Validate task consistency, query membership, and optional TRAKE event counts."""

    _validate_dataset_structure(dataset)
    counts = expected_trake_event_counts
    if counts is not None and dataset.task_type != "trake":
        raise ValueError("expected_trake_event_counts is valid only for task 'trake'")
    if counts is not None and not isinstance(counts, Mapping):
        raise TypeError("expected_trake_event_counts must be a mapping")

    query_ids = tuple(query.query_id for query in dataset.queries)
    if expected_query_ids is not None:
        expected = _validated_expected_query_ids(expected_query_ids)
        missing = sorted(set(expected) - set(query_ids))
        unexpected = sorted(set(query_ids) - set(expected))
        if missing or unexpected:
            raise ValueError(
                "query ID set mismatch: "
                f"missing={missing}, unexpected={unexpected}"
            )

    if counts is not None:
        for query_id, count in counts.items():
            _require_nonempty_text(query_id, "event-count query_id")
            _require_int(count, f"event count for {query_id!r}", minimum=1)
        unknown = sorted(set(counts) - set(query_ids))
        if unknown:
            raise ValueError(f"event-count map contains unknown query IDs: {unknown}")

    for query in dataset.queries:
        expected_count = counts.get(query.query_id) if counts is not None else None
        validate_top100_query(
            query,
            expected_trake_event_count=expected_count,
        )


def prediction_to_record(
    prediction: PredictionType,
    *,
    task_type: TaskType,
) -> dict[str, object]:
    """Convert a P0-A prediction into its exact canonical internal record shape."""

    task = _require_task(task_type)
    expected_type = _expected_prediction_type(task)
    if type(prediction) is not expected_type:
        raise TypeError(
            f"task {task!r} requires {expected_type.__name__}, "
            f"got {type(prediction).__name__}"
        )
    if task == "kis":
        kis = cast(KISPrediction, prediction)
        return {
            "query_id": kis.query_id,
            "rank": kis.rank,
            "video_id": kis.video_id,
            "frame_id": kis.frame_id,
        }
    if task == "qa":
        qa = cast(QAPrediction, prediction)
        return {
            "query_id": qa.query_id,
            "rank": qa.rank,
            "video_id": qa.video_id,
            "frame_id": qa.frame_id,
            "answer": qa.answer,
        }
    trake = cast(TRAKEPrediction, prediction)
    return {
        "query_id": trake.query_id,
        "rank": trake.rank,
        "video_id": trake.video_id,
        "frame_ids": list(trake.frame_ids),
    }


def record_to_prediction(
    record: object,
    *,
    task_type: TaskType,
) -> PredictionType:
    """Strictly parse one task-specific JSON object into a P0-A prediction."""

    task = _require_task(task_type)
    if type(record) is not dict:
        raise TypeError("record must be a JSON object")
    values = cast(dict[str, Any], record)
    if any(type(field) is not str for field in values):
        raise TypeError("record field names must be strings")
    expected_fields = TASK_RECORD_FIELDS[task]
    missing = tuple(field for field in expected_fields if field not in values)
    extra = tuple(sorted(set(values) - set(expected_fields)))
    if missing or extra:
        raise ValueError(
            f"record fields do not match task {task!r}: "
            f"missing={list(missing)}, extra={list(extra)}"
        )

    query_id = _require_nonempty_text(values["query_id"], "query_id")
    rank = _require_int(values["rank"], "rank", minimum=1)
    video_id = _require_nonempty_text(values["video_id"], "video_id")
    if task == "kis":
        frame_id = _require_int(values["frame_id"], "frame_id", minimum=0)
        return KISPrediction(query_id, rank, video_id, frame_id)
    if task == "qa":
        frame_id = _require_int(values["frame_id"], "frame_id", minimum=0)
        answer = _require_nonempty_text(values["answer"], "answer")
        return QAPrediction(query_id, rank, video_id, frame_id, answer)

    raw_frame_ids = values["frame_ids"]
    if type(raw_frame_ids) is not list or not raw_frame_ids:
        raise ValueError("frame_ids must be a non-empty JSON list")
    frame_ids = tuple(
        _require_int(frame_id, "frame_id", minimum=0) for frame_id in raw_frame_ids
    )
    return TRAKEPrediction(query_id, rank, video_id, frame_ids)


def _record_query_id(record: object) -> str | None:
    if type(record) is dict:
        value = cast(dict[str, object], record).get("query_id")
        if type(value) is str:
            return value
    return None


def load_top100_jsonl(
    path: Path,
    *,
    task_type: TaskType,
    expected_query_ids: Sequence[str] | None = None,
    expected_trake_event_counts: Mapping[str, int] | None = None,
) -> RankedTop100Dataset:
    """Load strict UTF-8 JSONL, preserving first-query and physical line order."""

    task = _require_task(task_type)
    source = Path(path)
    try:
        data = source.read_bytes()
    except OSError as exc:
        raise Top100FormatError("FILE_ERROR", f"cannot read checkpoint: {exc}") from exc
    try:
        text = data.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise Top100FormatError("INVALID_UTF8", f"checkpoint is not valid UTF-8: {exc}") from exc
    if text.startswith("\ufeff"):
        raise Top100FormatError("UTF8_BOM", "UTF-8 BOM is not permitted")

    grouped: dict[str, list[PredictionType]] = {}
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise Top100FormatError(
                "INVALID_JSON",
                f"line {line_number}: invalid JSON: {exc.msg}",
                line_number=line_number,
            ) from exc
        try:
            prediction = record_to_prediction(record, task_type=task)
        except (TypeError, ValueError) as exc:
            query_id = _record_query_id(record)
            raise Top100FormatError(
                "INVALID_RECORD",
                f"line {line_number}: {exc}",
                line_number=line_number,
                query_id=query_id,
            ) from exc
        grouped.setdefault(prediction.query_id, []).append(prediction)

    if not grouped:
        raise Top100FormatError("EMPTY_CHECKPOINT", "checkpoint has no prediction records")

    try:
        dataset = RankedTop100Dataset(
            task_type=task,
            queries=tuple(
                RankedTop100Query(task, query_id, tuple(predictions))
                for query_id, predictions in grouped.items()
            ),
        )
        validate_top100_dataset(
            dataset,
            expected_query_ids=expected_query_ids,
            expected_trake_event_counts=expected_trake_event_counts,
        )
    except (TypeError, ValueError) as exc:
        raise Top100FormatError("INVALID_DATASET", str(exc)) from exc
    return dataset


def write_top100_jsonl(
    dataset: RankedTop100Dataset,
    destination: Path,
    *,
    expected_query_ids: Sequence[str] | None = None,
    expected_trake_event_counts: Mapping[str, int] | None = None,
) -> Top100ExportSummary:
    """Validate the entire dataset, then write deterministic internal JSONL once."""

    if type(dataset) is not RankedTop100Dataset:
        raise TypeError("dataset must be RankedTop100Dataset")
    validate_top100_dataset(
        dataset,
        expected_query_ids=expected_query_ids,
        expected_trake_event_counts=expected_trake_event_counts,
    )
    if not dataset.queries:
        raise ValueError("cannot export an empty Top-100 dataset")
    empty_queries = [query.query_id for query in dataset.queries if not query.predictions]
    if empty_queries:
        raise ValueError(
            "record-only JSONL cannot roundtrip queries with zero predictions: "
            f"{empty_queries}"
        )

    lines = [
        json.dumps(
            prediction_to_record(prediction, task_type=dataset.task_type),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        for query in dataset.queries
        for prediction in query.predictions
    ]
    text = "\n".join(lines) + "\n"
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    return Top100ExportSummary(
        destination=path,
        task_type=dataset.task_type,
        query_count=len(dataset.queries),
        record_count=len(lines),
    )


def validate_top100_jsonl(
    path: Path,
    *,
    task_type: TaskType,
    expected_query_ids: Sequence[str] | None = None,
    expected_trake_event_counts: Mapping[str, int] | None = None,
) -> Top100ValidationReport:
    """Return a small structured report without weakening strict loader behavior."""

    try:
        load_top100_jsonl(
            path,
            task_type=task_type,
            expected_query_ids=expected_query_ids,
            expected_trake_event_counts=expected_trake_event_counts,
        )
    except Top100FormatError as exc:
        issue = Top100ValidationIssue(
            code=exc.code,
            message=str(exc),
            line_number=exc.line_number,
            query_id=exc.query_id,
        )
        return Top100ValidationReport(valid=False, errors=(issue,))
    except (TypeError, ValueError) as exc:
        issue = Top100ValidationIssue(code="INVALID_ARGUMENT", message=str(exc))
        return Top100ValidationReport(valid=False, errors=(issue,))
    return Top100ValidationReport(valid=True, errors=())
