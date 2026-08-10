"""Split-blind two-pass human semantic annotation workflow for Q1-B.

The module deliberately depends only on the frozen quality/data schemas and the
accepted suitability queue.  It never imports or executes retrieval/runtime code.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import io
import json
import os
import re
import sys
import uuid
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any, NamedTuple, cast

SYSTEM_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = SYSTEM_ROOT.parents[1]
SOURCE_ROOT = SYSTEM_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from system_tai.preliminary.schemas import (  # noqa: E402
    KISGroundTruth,
    QAGroundTruth,
    TRAKEGroundTruth,
)
from system_tai.quality.schema import (  # noqa: E402
    AnnotationStatus,
    Difficulty,
    KISQualityQuery,
    LabelOrigin,
    QAQualityQuery,
    QualityBenchmark,
    QualityQuery,
    QualityTaskType,
    QualityTRAKEEvent,
    TRAKEQualityQuery,
    load_quality_benchmark_json,
)

BENCHMARK_ID = "system_tai-q1b-b1-diagnostic-v1-draft"
BENCHMARK_DESCRIPTION = (
    "Bootstrap scaffold with no verified semantic GT yet. This is Batch 1 "
    "diagnostic scope only and is not official competition ground truth."
)
PASS1_COMPLETE = "COMPLETE"
PASS2_PENDING = "REVIEW_PENDING"
PASS2_VERIFIED = "VERIFIED"
PASS2_REVISION = "REVISION_REQUIRED"
ALLOWED_PASS2_STATES = frozenset({PASS2_PENDING, PASS2_VERIFIED, PASS2_REVISION})
PILOT15_SIZE = 15

REGISTRY_COLUMNS = (
    "query_id",
    "slot_id",
    "task_type",
    "video_id",
    "query_authored_before_retrieval",
    "gt_authored_before_retrieval",
    "raw_video_reviewed",
    "original_frame_coordinates_verified",
    "annotation_pass1_status",
    "annotation_pass2_status",
    "annotator_id",
    "reviewer_id",
    "semantic_definition",
    "boundary_notes",
    "answer_notes",
    "review_notes",
    "benchmark_included",
)

TRAKE_COLUMNS = (
    "query_id",
    "event_index",
    "event_description",
    "moment_definition",
    "start_frame_id",
    "end_frame_id",
    "annotator_id",
    "reviewer_id",
    "review_status",
    "notes",
)

COMMON_PASS1_FIELDS = frozenset(
    {
        "expect_assignment_rank",
        "expect_slot_id",
        "annotator_id",
        "raw_video_reviewed",
        "query_authored_before_retrieval",
        "gt_authored_before_retrieval",
        "original_frame_coordinates_verified",
        "raw_video_frame_count",
        "difficulty",
        "tags",
        "semantic_definition",
        "annotation_notes",
        "boundary_notes",
        "answer_notes",
    }
)
KIS_PASS1_FIELDS = COMMON_PASS1_FIELDS | {
    "query_vi",
    "query_en",
    "query_en_expansion",
    "start_frame_id",
    "end_frame_id",
}
QA_PASS1_FIELDS = COMMON_PASS1_FIELDS | {
    "event_description",
    "event_description_en",
    "question",
    "question_en",
    "start_frame_id",
    "end_frame_id",
    "accepted_answers",
}
TRAKE_PASS1_FIELDS = COMMON_PASS1_FIELDS | {"events"}
TRAKE_EVENT_FIELDS = frozenset(
    {
        "description",
        "description_en",
        "moment_definition",
        "start_frame_id",
        "end_frame_id",
    }
)

COMMON_PASS2_FIELDS = frozenset(
    {
        "query_id",
        "reviewer_id",
        "decision",
        "raw_video_reviewed",
        "semantic_support_verified",
        "video_id_verified",
        "original_frame_coordinates_verified",
        "intervals_verified",
        "review_notes",
    }
)

SPLIT_TAGS = frozenset({"q1b_dev", "q1b_holdout"})
SLOT_ID_PATTERN = re.compile(r"^(KIS|QA|TRAKE)-(\d{3})$")
SLOT_ID_LIMITS = {"KIS": 25, "QA": 20, "TRAKE": 15}
RESULT_TAG_PATTERN = re.compile(
    r"^(?:rank[-_]?\d+(?:[-_]|$)|retrieval(?:[-_]|$)|model(?:[-_]|$)|"
    r"prediction(?:[-_]|$)|result(?:[-_]|$)|score(?:[-_]|$)|"
    r"rrf(?:[-_]|$)|fusion(?:[-_]|$)|fused(?:[-_]|$)|"
    r"cosine(?:[-_]|$)|similarity(?:[-_]|$)|top[-_]?\d+(?:[-_]|$))",
    flags=re.IGNORECASE,
)
MODEL_RESULT_TAG_PREFIXES = frozenset({"cl" + "ip"})
WINDOWS_ABSOLUTE_PATH_PATTERN = re.compile(r"\b[a-z]:[\\/]", flags=re.IGNORECASE)
UNC_ABSOLUTE_PATH_PATTERN = re.compile(r"\\{2,}[^\\/\s]+\\+[^\\/\s]+")
POSIX_MACHINE_PATH_PATTERN = re.compile(
    r"(?:^|[\s\"'(=,;])/(?:kaggle/input|home|tmp)/", flags=re.IGNORECASE
)
HIDDEN_SPLIT_PATTERN = re.compile(
    r"planned_split|q1b_dev|q1b_holdout|development|holdout", flags=re.IGNORECASE
)


class SemanticAnnotationError(ValueError):
    """Raised when semantic workflow input or cross-artifact state is invalid."""


class SemanticPaths(NamedTuple):
    candidate_manifest: Path
    annotation_plan: Path
    category_codebook: Path
    slot_manifest: Path
    review_log: Path
    benchmark: Path
    registry: Path
    trake_review: Path


class SemanticTarget(NamedTuple):
    assignment_rank: int
    review_sequence: int
    video_id: str
    slot_id: str
    task: str
    target_category: str
    category_name: str
    category_definition: str
    acceptance_guidance: str
    rejection_guidance: str
    suggested_tags: str
    suitability_notes: str
    derived_query_id: str


class RegistryRecord(NamedTuple):
    query_id: str
    slot_id: str
    task_type: str
    video_id: str
    query_authored_before_retrieval: bool
    gt_authored_before_retrieval: bool
    raw_video_reviewed: bool
    original_frame_coordinates_verified: bool
    annotation_pass1_status: str
    annotation_pass2_status: str
    annotator_id: str
    reviewer_id: str
    semantic_definition: str
    boundary_notes: str
    answer_notes: str
    review_notes: str
    benchmark_included: bool


class TRAKEReviewRecord(NamedTuple):
    query_id: str
    event_index: int
    event_description: str
    moment_definition: str
    start_frame_id: int
    end_frame_id: int
    annotator_id: str
    reviewer_id: str
    review_status: str
    notes: str


class SemanticState(NamedTuple):
    targets: tuple[SemanticTarget, ...]
    benchmark: QualityBenchmark
    registry: tuple[RegistryRecord, ...]
    trake_reviews: tuple[TRAKEReviewRecord, ...]


def default_paths() -> SemanticPaths:
    """Return canonical Q1-B artifact paths without exposing machine data paths."""

    root = SYSTEM_ROOT / "benchmarks" / "quality_q1b"
    return SemanticPaths(
        root / "candidate_video_manifest.csv",
        root / "annotation_plan.csv",
        root / "category_codebook.csv",
        root / "slot_assignment_manifest.csv",
        root / "candidate_review_log.csv",
        root / "benchmark.draft.json",
        root / "annotation_registry.csv",
        root / "trake_event_review.csv",
    )


def _load_queue_module() -> Any:
    name = "system_tai_q1b_suitability_queue_for_semantic_workflow"
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    path = Path(__file__).with_name("quality_q1b_annotation_queue.py")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise SemanticAnnotationError("cannot load the frozen suitability queue")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _queue_paths(paths: SemanticPaths) -> Any:
    queue = _load_queue_module()
    return queue.QueuePaths(*paths[:5])


def derive_query_id(slot_id: str) -> str:
    """Derive the immutable semantic query ID from a frozen slot ID."""

    if type(slot_id) is not str:
        raise SemanticAnnotationError("slot_id must be a string")
    match = SLOT_ID_PATTERN.fullmatch(slot_id)
    if match is None:
        raise SemanticAnnotationError("slot_id must use the exact frozen TASK-NNN form")
    task, number_text = match.groups()
    number = int(number_text)
    if not 1 <= number <= SLOT_ID_LIMITS[task]:
        raise SemanticAnnotationError(f"slot_id is outside the frozen {task} slot range")
    return "q1b-" + slot_id.casefold()


def _semantic_targets(paths: SemanticPaths) -> tuple[SemanticTarget, ...]:
    queue = _load_queue_module()
    _candidates, _slots, reviews, codebook = queue.load_queue(_queue_paths(paths))
    assigned = sorted(
        (record for record in reviews if record.decision == queue.ASSIGN),
        key=lambda record: record.assignment_rank,
    )
    targets: list[SemanticTarget] = []
    for expected_rank, record in enumerate(assigned, start=1):
        if record.assignment_rank != expected_rank:
            raise SemanticAnnotationError(
                "ASSIGN records must form a contiguous assignment-rank prefix"
            )
        category = codebook.get(record.target_category)
        if category is None:
            raise SemanticAnnotationError(
                f"missing category for semantic target: {record.target_category}"
            )
        targets.append(
            SemanticTarget(
                record.assignment_rank,
                record.review_sequence,
                record.video_id,
                record.slot_id,
                record.planned_task,
                record.target_category,
                category.category_name,
                category.definition,
                category.acceptance_guidance,
                category.rejection_guidance,
                category.suggested_tags,
                record.notes,
                derive_query_id(record.slot_id),
            )
        )
    query_ids = [target.derived_query_id for target in targets]
    if len(query_ids) != len(set(query_ids)):
        raise SemanticAnnotationError("derived query IDs must be globally unique")
    return tuple(targets)


def _read_utf8(path: Path, label: str) -> str:
    try:
        payload = Path(path).read_bytes()
    except OSError as exc:
        raise SemanticAnnotationError(f"cannot read {label}: {exc}") from exc
    try:
        text = payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise SemanticAnnotationError(f"{label} is not valid UTF-8: {exc}") from exc
    if text.startswith("\ufeff"):
        raise SemanticAnnotationError(f"{label} must not contain a UTF-8 BOM")
    return text


def _strict_csv_rows(
    path: Path,
    columns: tuple[str, ...],
    label: str,
) -> tuple[dict[str, str], ...]:
    parsed = list(csv.reader(io.StringIO(_read_utf8(path, label), newline="")))
    if not parsed:
        raise SemanticAnnotationError(f"{label} is empty")
    if tuple(parsed[0]) != columns:
        raise SemanticAnnotationError(
            f"{label} header mismatch: expected={list(columns)}, actual={parsed[0]}"
        )
    rows: list[dict[str, str]] = []
    for line_number, row in enumerate(parsed[1:], start=2):
        if len(row) != len(columns):
            raise SemanticAnnotationError(
                f"{label} line {line_number} has {len(row)} columns; "
                f"expected {len(columns)}"
            )
        rows.append(dict(zip(columns, row, strict=True)))
    return tuple(rows)


def _parse_bool(value: str, name: str) -> bool:
    if value == "true":
        return True
    if value == "false":
        return False
    raise SemanticAnnotationError(f"{name} must be exactly 'true' or 'false'")


def _parse_nonnegative_int(value: str, name: str) -> int:
    if not value.isascii() or not value.isdecimal():
        raise SemanticAnnotationError(f"{name} must be a canonical non-negative integer")
    resolved = int(value)
    if value != str(resolved):
        raise SemanticAnnotationError(f"{name} must be a canonical non-negative integer")
    return resolved


def _load_registry(path: Path) -> tuple[RegistryRecord, ...]:
    rows = _strict_csv_rows(path, REGISTRY_COLUMNS, "annotation registry")
    return tuple(
        RegistryRecord(
            row["query_id"],
            row["slot_id"],
            row["task_type"],
            row["video_id"],
            _parse_bool(
                row["query_authored_before_retrieval"],
                "query_authored_before_retrieval",
            ),
            _parse_bool(
                row["gt_authored_before_retrieval"],
                "gt_authored_before_retrieval",
            ),
            _parse_bool(row["raw_video_reviewed"], "raw_video_reviewed"),
            _parse_bool(
                row["original_frame_coordinates_verified"],
                "original_frame_coordinates_verified",
            ),
            row["annotation_pass1_status"],
            row["annotation_pass2_status"],
            row["annotator_id"],
            row["reviewer_id"],
            row["semantic_definition"],
            row["boundary_notes"],
            row["answer_notes"],
            row["review_notes"],
            _parse_bool(row["benchmark_included"], "benchmark_included"),
        )
        for row in rows
    )


def _load_trake_reviews(path: Path) -> tuple[TRAKEReviewRecord, ...]:
    rows = _strict_csv_rows(path, TRAKE_COLUMNS, "TRAKE event review")
    return tuple(
        TRAKEReviewRecord(
            row["query_id"],
            _parse_nonnegative_int(row["event_index"], "event_index"),
            row["event_description"],
            row["moment_definition"],
            _parse_nonnegative_int(row["start_frame_id"], "start_frame_id"),
            _parse_nonnegative_int(row["end_frame_id"], "end_frame_id"),
            row["annotator_id"],
            row["reviewer_id"],
            row["review_status"],
            row["notes"],
        )
        for row in rows
    )


def _require_text(value: object, name: str, *, allow_empty: bool = False) -> str:
    if type(value) is not str:
        raise SemanticAnnotationError(f"{name} must be a string")
    text = cast(str, value)
    if not allow_empty and not text.strip():
        raise SemanticAnnotationError(f"{name} must be a non-empty string")
    if allow_empty and text and not text.strip():
        raise SemanticAnnotationError(f"{name} must not contain only whitespace")
    if "\x00" in text:
        raise SemanticAnnotationError(f"{name} must not contain NUL")
    _reject_machine_local_path(text, name)
    return text


def _require_identifier(value: object, name: str) -> str:
    text = _require_text(value, name)
    if text != text.strip():
        raise SemanticAnnotationError(f"{name} must not contain outer whitespace")
    if any(ord(character) < 32 or ord(character) == 127 for character in text):
        raise SemanticAnnotationError(f"{name} must not contain control characters")
    return text


def _contains_machine_local_path(text: str) -> bool:
    return bool(
        WINDOWS_ABSOLUTE_PATH_PATTERN.search(text)
        or UNC_ABSOLUTE_PATH_PATTERN.search(text)
        or POSIX_MACHINE_PATH_PATTERN.search(text)
    )


def _reject_machine_local_path(text: str, name: str) -> None:
    if _contains_machine_local_path(text):
        raise SemanticAnnotationError(f"{name} must not contain a machine-local absolute path")


def _require_optional_text(value: object, name: str) -> str | None:
    if value is None:
        return None
    return _require_text(value, name)


def _strict_int(value: object, name: str, *, positive: bool = False) -> int:
    if type(value) is not int:
        raise SemanticAnnotationError(f"{name} must be an integer")
    resolved = cast(int, value)
    if positive and resolved <= 0:
        raise SemanticAnnotationError(f"{name} must be greater than zero")
    return resolved


def _strict_bool_input(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise SemanticAnnotationError(f"{name} must be a JSON boolean")
    return cast(bool, value)


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SemanticAnnotationError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str) -> None:
    raise SemanticAnnotationError(f"non-finite JSON numeric value is forbidden: {value}")


def load_strict_json(path: Path) -> dict[str, Any]:
    """Load one exact UTF-8 JSON object without BOM, duplicate keys, or NaN."""

    text = _read_utf8(path, "semantic input JSON")
    try:
        payload = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonfinite_constant,
        )
    except json.JSONDecodeError as exc:
        raise SemanticAnnotationError(f"invalid semantic input JSON: {exc.msg}") from exc
    if type(payload) is not dict:
        raise SemanticAnnotationError("semantic input must be one JSON object")
    return cast(dict[str, Any], payload)


def _require_fields(
    payload: Mapping[str, Any], expected: frozenset[str], context: str
) -> None:
    missing = sorted(expected - set(payload))
    unknown = sorted(set(payload) - expected)
    if missing or unknown:
        raise SemanticAnnotationError(
            f"{context} fields mismatch: missing={missing}, unknown={unknown}"
        )


def _validate_tags(value: object) -> tuple[str, ...]:
    if type(value) is not list:
        raise SemanticAnnotationError("tags must be a JSON list")
    tags = tuple(_require_text(tag, "tag") for tag in cast(list[object], value))
    if len(tags) != len(set(tags)):
        raise SemanticAnnotationError("tags must be unique")
    for tag in tags:
        if tag != tag.strip():
            raise SemanticAnnotationError("tag must not contain outer whitespace")
        normalized = tag.casefold()
        if normalized in SPLIT_TAGS:
            raise SemanticAnnotationError(f"split tag is forbidden: {tag!r}")
        if RESULT_TAG_PATTERN.match(normalized) or any(
            normalized == prefix
            or normalized.startswith(prefix + "_")
            or normalized.startswith(prefix + "-")
            for prefix in MODEL_RESULT_TAG_PREFIXES
        ):
            raise SemanticAnnotationError(f"model/result-derived tag is forbidden: {tag!r}")
    return tags


def _validate_interval(start: object, end: object, frame_count: int, label: str) -> tuple[int, int]:
    start_id = _strict_int(start, f"{label}.start_frame_id")
    end_id = _strict_int(end, f"{label}.end_frame_id")
    if start_id < 0 or start_id > end_id or end_id >= frame_count:
        raise SemanticAnnotationError(
            f"{label} requires 0 <= start_frame_id <= end_frame_id < raw_video_frame_count"
        )
    return start_id, end_id


def _validate_common_pass1(payload: Mapping[str, Any], target: SemanticTarget) -> dict[str, Any]:
    rank = _strict_int(payload["expect_assignment_rank"], "expect_assignment_rank")
    slot_id = _require_identifier(payload["expect_slot_id"], "expect_slot_id")
    derive_query_id(slot_id)
    if rank != target.assignment_rank or slot_id != target.slot_id:
        raise SemanticAnnotationError("stale Pass-1 target: assignment rank or slot ID changed")
    confirmations = (
        "raw_video_reviewed",
        "query_authored_before_retrieval",
        "gt_authored_before_retrieval",
        "original_frame_coordinates_verified",
    )
    for field in confirmations:
        if not _strict_bool_input(payload[field], field):
            raise SemanticAnnotationError(f"{field} must be true")
    difficulty_text = _require_text(payload["difficulty"], "difficulty")
    try:
        difficulty = Difficulty(difficulty_text)
    except ValueError as exc:
        raise SemanticAnnotationError(f"invalid difficulty: {difficulty_text!r}") from exc
    return {
        "annotator_id": _require_identifier(payload["annotator_id"], "annotator_id"),
        "frame_count": _strict_int(
            payload["raw_video_frame_count"], "raw_video_frame_count", positive=True
        ),
        "difficulty": difficulty,
        "tags": _validate_tags(payload["tags"]),
        "semantic_definition": _require_text(
            payload["semantic_definition"], "semantic_definition"
        ),
        "annotation_notes": _require_text(
            payload["annotation_notes"], "annotation_notes", allow_empty=True
        ),
        "boundary_notes": _require_text(
            payload["boundary_notes"], "boundary_notes", allow_empty=True
        ),
        "answer_notes": _require_text(
            payload["answer_notes"], "answer_notes", allow_empty=True
        ),
    }


def _build_pass1_record(
    payload: Mapping[str, Any], target: SemanticTarget
) -> tuple[QualityQuery, RegistryRecord, tuple[TRAKEReviewRecord, ...]]:
    expected_fields = {
        "kis": KIS_PASS1_FIELDS,
        "qa": QA_PASS1_FIELDS,
        "trake": TRAKE_PASS1_FIELDS,
    }[target.task]
    _require_fields(payload, frozenset(expected_fields), f"{target.task} Pass-1 input")
    common = _validate_common_pass1(payload, target)
    query_id = target.derived_query_id
    sidecar: tuple[TRAKEReviewRecord, ...] = ()

    query_common = {
        "query_id": query_id,
        "annotation_status": AnnotationStatus.DRAFT,
        "label_origin": LabelOrigin.HUMAN_RAW_VIDEO,
        "difficulty": common["difficulty"],
        "tags": common["tags"],
        "annotation_notes": common["annotation_notes"],
    }
    if target.task == "kis":
        start, end = _validate_interval(
            payload["start_frame_id"],
            payload["end_frame_id"],
            common["frame_count"],
            "KIS interval",
        )
        source_reference = (
            f"raw_video:{target.video_id};reviewed_frames:{start}-{end}"
        )
        query: QualityQuery = KISQualityQuery(
            **query_common,
            task_type=QualityTaskType.KIS,
            source_reference=source_reference,
            query_vi=_require_text(payload["query_vi"], "query_vi"),
            query_en=_require_optional_text(payload["query_en"], "query_en"),
            query_en_expansion=_require_optional_text(
                payload["query_en_expansion"], "query_en_expansion"
            ),
            ground_truth=KISGroundTruth(query_id, target.video_id, start, end),
        )
    elif target.task == "qa":
        start, end = _validate_interval(
            payload["start_frame_id"],
            payload["end_frame_id"],
            common["frame_count"],
            "Q&A interval",
        )
        answers_value = payload["accepted_answers"]
        if type(answers_value) is not list or not answers_value:
            raise SemanticAnnotationError("accepted_answers must be a non-empty JSON list")
        answers = tuple(
            _require_text(answer, "accepted_answer")
            for answer in cast(list[object], answers_value)
        )
        if len(answers) != len(set(answers)):
            raise SemanticAnnotationError("accepted_answers must not contain exact duplicates")
        source_reference = (
            f"raw_video:{target.video_id};reviewed_frames:{start}-{end}"
        )
        query = QAQualityQuery(
            **query_common,
            task_type=QualityTaskType.QA,
            source_reference=source_reference,
            event_description=_require_text(
                payload["event_description"], "event_description"
            ),
            event_description_en=_require_optional_text(
                payload["event_description_en"], "event_description_en"
            ),
            question=_require_text(payload["question"], "question"),
            question_en=_require_optional_text(payload["question_en"], "question_en"),
            ground_truth=QAGroundTruth(
                query_id, target.video_id, start, end, answers
            ),
        )
    else:
        events_value = payload["events"]
        if type(events_value) is not list:
            raise SemanticAnnotationError("events must be a JSON list")
        raw_events = cast(list[object], events_value)
        if not 2 <= len(raw_events) <= 5:
            raise SemanticAnnotationError("TRAKE requires between 2 and 5 events")
        quality_events: list[QualityTRAKEEvent] = []
        intervals: list[tuple[int, int]] = []
        sidecar_rows: list[TRAKEReviewRecord] = []
        previous_start: int | None = None
        for index, event_value in enumerate(raw_events, start=1):
            if type(event_value) is not dict:
                raise SemanticAnnotationError(f"events[{index - 1}] must be an object")
            event = cast(dict[str, Any], event_value)
            _require_fields(event, TRAKE_EVENT_FIELDS, f"events[{index - 1}]")
            description = _require_text(event["description"], "event.description")
            description_en = _require_optional_text(
                event["description_en"], "event.description_en"
            )
            moment_definition = _require_text(
                event["moment_definition"], "event.moment_definition"
            )
            start, end = _validate_interval(
                event["start_frame_id"],
                event["end_frame_id"],
                common["frame_count"],
                f"events[{index - 1}]",
            )
            if previous_start is not None and start <= previous_start:
                raise SemanticAnnotationError(
                    "TRAKE event start positions must be strictly increasing"
                )
            previous_start = start
            quality_events.append(QualityTRAKEEvent(description, description_en))
            intervals.append((start, end))
            sidecar_rows.append(
                TRAKEReviewRecord(
                    query_id,
                    index,
                    description,
                    moment_definition,
                    start,
                    end,
                    common["annotator_id"],
                    "",
                    PASS2_PENDING,
                    "",
                )
            )
        windows = "|".join(f"{start}-{end}" for start, end in intervals)
        source_reference = f"raw_video:{target.video_id};event_windows:{windows}"
        query = TRAKEQualityQuery(
            **query_common,
            task_type=QualityTaskType.TRAKE,
            source_reference=source_reference,
            events=tuple(quality_events),
            ground_truth=TRAKEGroundTruth(query_id, target.video_id, tuple(intervals)),
        )
        sidecar = tuple(sidecar_rows)

    registry = RegistryRecord(
        query_id,
        target.slot_id,
        target.task,
        target.video_id,
        True,
        True,
        True,
        True,
        PASS1_COMPLETE,
        PASS2_PENDING,
        common["annotator_id"],
        "",
        common["semantic_definition"],
        common["boundary_notes"],
        common["answer_notes"],
        "",
        False,
    )
    return query, registry, sidecar


def _query_ground_truth(query: QualityQuery) -> Any:
    if query.ground_truth is None:
        raise SemanticAnnotationError(f"query has no human ground truth: {query.query_id}")
    return query.ground_truth


def _expected_source_reference(query: QualityQuery) -> str:
    ground_truth = _query_ground_truth(query)
    if type(query) in {KISQualityQuery, QAQualityQuery}:
        return (
            f"raw_video:{ground_truth.video_id};reviewed_frames:"
            f"{ground_truth.start_frame_id}-{ground_truth.end_frame_id}"
        )
    intervals = cast(TRAKEGroundTruth, ground_truth).event_intervals
    windows = "|".join(f"{start}-{end}" for start, end in intervals)
    return f"raw_video:{ground_truth.video_id};event_windows:{windows}"


def _query_to_payload(query: QualityQuery) -> dict[str, Any]:
    common: dict[str, Any] = {
        "task_type": query.task_type.value,
        "query_id": query.query_id,
        "annotation_status": query.annotation_status.value,
        "label_origin": query.label_origin.value,
        "difficulty": query.difficulty.value,
        "tags": list(query.tags),
        "annotation_notes": query.annotation_notes,
        "source_reference": query.source_reference,
    }
    ground_truth = _query_ground_truth(query)
    if type(query) is KISQualityQuery:
        common.update(
            {
                "query_vi": query.query_vi,
                "query_en": query.query_en,
                "query_en_expansion": query.query_en_expansion,
                "ground_truth": {
                    "video_id": ground_truth.video_id,
                    "start_frame_id": ground_truth.start_frame_id,
                    "end_frame_id": ground_truth.end_frame_id,
                },
            }
        )
    elif type(query) is QAQualityQuery:
        common.update(
            {
                "event_description": query.event_description,
                "question": query.question,
                "event_description_en": query.event_description_en,
                "question_en": query.question_en,
                "ground_truth": {
                    "video_id": ground_truth.video_id,
                    "start_frame_id": ground_truth.start_frame_id,
                    "end_frame_id": ground_truth.end_frame_id,
                    "accepted_answers": list(ground_truth.accepted_answers),
                },
            }
        )
    else:
        trake_query = cast(TRAKEQualityQuery, query)
        common.update(
            {
                "events": [
                    {
                        "description": event.description,
                        "description_en": event.description_en,
                    }
                    for event in trake_query.events
                ],
                "ground_truth": {
                    "video_id": ground_truth.video_id,
                    "event_intervals": [list(item) for item in ground_truth.event_intervals],
                },
            }
        )
    return common


def _serialize_benchmark(benchmark: QualityBenchmark) -> bytes:
    payload = {
        "schema_version": benchmark.schema_version,
        "benchmark_id": benchmark.benchmark_id,
        "description": benchmark.description,
        "queries": [_query_to_payload(query) for query in benchmark.queries],
    }
    return (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _csv_bytes(columns: tuple[str, ...], rows: Iterable[Sequence[object]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(columns)
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


def _serialize_registry(records: Iterable[RegistryRecord]) -> bytes:
    rows = (
        (
            item.query_id,
            item.slot_id,
            item.task_type,
            item.video_id,
            _bool_text(item.query_authored_before_retrieval),
            _bool_text(item.gt_authored_before_retrieval),
            _bool_text(item.raw_video_reviewed),
            _bool_text(item.original_frame_coordinates_verified),
            item.annotation_pass1_status,
            item.annotation_pass2_status,
            item.annotator_id,
            item.reviewer_id,
            item.semantic_definition,
            item.boundary_notes,
            item.answer_notes,
            item.review_notes,
            _bool_text(item.benchmark_included),
        )
        for item in records
    )
    return _csv_bytes(REGISTRY_COLUMNS, rows)


def _serialize_trake_reviews(records: Iterable[TRAKEReviewRecord]) -> bytes:
    rows = (
        (
            item.query_id,
            item.event_index,
            item.event_description,
            item.moment_definition,
            item.start_frame_id,
            item.end_frame_id,
            item.annotator_id,
            item.reviewer_id,
            item.review_status,
            item.notes,
        )
        for item in records
    )
    return _csv_bytes(TRAKE_COLUMNS, rows)


def _assert_unique(values: Sequence[str], label: str) -> None:
    if len(values) != len(set(values)):
        raise SemanticAnnotationError(f"duplicate {label}")


def _validate_query_contract(
    query: QualityQuery,
    registry: RegistryRecord,
    target: SemanticTarget,
    sidecar: tuple[TRAKEReviewRecord, ...],
) -> None:
    if query.query_id != target.derived_query_id or registry.query_id != query.query_id:
        raise SemanticAnnotationError("query ID is inconsistent with slot ID")
    if registry.slot_id != target.slot_id:
        raise SemanticAnnotationError(f"wrong slot for query {query.query_id}")
    if registry.task_type != target.task or query.task_type.value != target.task:
        raise SemanticAnnotationError(f"wrong task for query {query.query_id}")
    ground_truth = _query_ground_truth(query)
    if registry.video_id != target.video_id or ground_truth.video_id != target.video_id:
        raise SemanticAnnotationError(f"wrong video for query {query.query_id}")
    if query.label_origin is not LabelOrigin.HUMAN_RAW_VIDEO:
        raise SemanticAnnotationError("semantic records require human_raw_video origin")
    if registry.annotation_pass1_status != PASS1_COMPLETE:
        raise SemanticAnnotationError("annotation_pass1_status must be COMPLETE")
    if registry.annotation_pass2_status not in ALLOWED_PASS2_STATES:
        raise SemanticAnnotationError("invalid annotation_pass2_status")
    if not all(
        (
            registry.query_authored_before_retrieval,
            registry.gt_authored_before_retrieval,
            registry.raw_video_reviewed,
            registry.original_frame_coordinates_verified,
        )
    ):
        raise SemanticAnnotationError("Pass-1 confirmation flags must remain true")
    _require_identifier(registry.annotator_id, "registry annotator_id")
    _require_text(registry.semantic_definition, "registry semantic_definition")
    _validate_tags(list(query.tags))
    if query.source_reference != _expected_source_reference(query):
        raise SemanticAnnotationError(f"invalid source_reference for {query.query_id}")

    if registry.annotation_pass2_status == PASS2_PENDING:
        if query.annotation_status is not AnnotationStatus.DRAFT:
            raise SemanticAnnotationError("review-pending query must remain draft")
        if registry.reviewer_id or registry.review_notes or registry.benchmark_included:
            raise SemanticAnnotationError("review-pending registry state is inconsistent")
    elif registry.annotation_pass2_status == PASS2_VERIFIED:
        if query.annotation_status is not AnnotationStatus.VERIFIED:
            raise SemanticAnnotationError("verified registry requires verified query")
        _require_identifier(registry.reviewer_id, "registry reviewer_id")
        if registry.reviewer_id == registry.annotator_id:
            raise SemanticAnnotationError("verified query requires an independent reviewer")
        if not registry.benchmark_included:
            raise SemanticAnnotationError("verified query must be benchmark_included")
    else:
        if query.annotation_status is not AnnotationStatus.DRAFT:
            raise SemanticAnnotationError("revision-required query must remain draft")
        _require_identifier(registry.reviewer_id, "registry reviewer_id")
        if registry.reviewer_id == registry.annotator_id:
            raise SemanticAnnotationError("revision-required query needs independent reviewer")
        if not registry.review_notes.strip() or registry.benchmark_included:
            raise SemanticAnnotationError("revision-required registry state is inconsistent")

    if type(query) is KISQualityQuery:
        if sidecar:
            raise SemanticAnnotationError("KIS query must not own TRAKE sidecar rows")
        if (
            ground_truth.start_frame_id < 0
            or ground_truth.start_frame_id > ground_truth.end_frame_id
        ):
            raise SemanticAnnotationError("invalid KIS interval")
        return
    if type(query) is QAQualityQuery:
        if sidecar:
            raise SemanticAnnotationError("Q&A query must not own TRAKE sidecar rows")
        if (
            ground_truth.start_frame_id < 0
            or ground_truth.start_frame_id > ground_truth.end_frame_id
        ):
            raise SemanticAnnotationError("invalid Q&A interval")
        if not ground_truth.accepted_answers or len(set(ground_truth.accepted_answers)) != len(
            ground_truth.accepted_answers
        ):
            raise SemanticAnnotationError("invalid Q&A accepted answers")
        return

    trake_query = cast(TRAKEQualityQuery, query)
    intervals = ground_truth.event_intervals
    if not 2 <= len(trake_query.events) <= 5 or len(trake_query.events) != len(intervals):
        raise SemanticAnnotationError("TRAKE event/ground-truth count mismatch")
    if len(sidecar) != len(intervals):
        raise SemanticAnnotationError("TRAKE sidecar event count mismatch")
    previous_start: int | None = None
    expected_sidecar_status = registry.annotation_pass2_status
    for index, (event, interval, review) in enumerate(
        zip(trake_query.events, intervals, sidecar, strict=True), start=1
    ):
        start, end = interval
        if start < 0 or start > end:
            raise SemanticAnnotationError("invalid TRAKE interval")
        if previous_start is not None and start <= previous_start:
            raise SemanticAnnotationError("TRAKE event starts must be strictly increasing")
        previous_start = start
        if (
            review.query_id != query.query_id
            or review.event_index != index
            or review.event_description != event.description
            or (review.start_frame_id, review.end_frame_id) != interval
            or not review.moment_definition.strip()
            or review.annotator_id != registry.annotator_id
            or review.reviewer_id != registry.reviewer_id
            or review.review_status != expected_sidecar_status
            or review.notes
        ):
            raise SemanticAnnotationError("TRAKE sidecar content/order is inconsistent")


def _audit_loaded(state: SemanticState, paths: SemanticPaths) -> dict[str, Any]:
    if (
        state.benchmark.schema_version != 1
        or state.benchmark.benchmark_id != BENCHMARK_ID
        or state.benchmark.description != BENCHMARK_DESCRIPTION
    ):
        raise SemanticAnnotationError("benchmark identity/schema changed")
    query_ids = [query.query_id for query in state.benchmark.queries]
    registry_ids = [record.query_id for record in state.registry]
    registry_slots = [record.slot_id for record in state.registry]
    _assert_unique(query_ids, "benchmark query_id")
    _assert_unique(registry_ids, "registry query_id")
    _assert_unique(registry_slots, "registry slot_id")
    if query_ids != registry_ids:
        missing_registry = sorted(set(query_ids) - set(registry_ids))
        missing_benchmark = sorted(set(registry_ids) - set(query_ids))
        raise SemanticAnnotationError(
            "benchmark/registry query mismatch: "
            f"missing_registry={missing_registry}, missing_benchmark={missing_benchmark}"
        )
    target_by_slot = {target.slot_id: target for target in state.targets}
    ranks: list[int] = []
    sidecar_by_query: dict[str, list[TRAKEReviewRecord]] = {}
    for review in state.trake_reviews:
        sidecar_by_query.setdefault(review.query_id, []).append(review)
    orphan_sidecars = sorted(set(sidecar_by_query) - set(query_ids))
    if orphan_sidecars:
        raise SemanticAnnotationError(f"orphan TRAKE sidecar queries: {orphan_sidecars}")
    expected_sidecar_order: list[TRAKEReviewRecord] = []
    for query, registry in zip(state.benchmark.queries, state.registry, strict=True):
        target = target_by_slot.get(registry.slot_id)
        if target is None:
            raise SemanticAnnotationError(
                f"registry slot is not an ASSIGN target: {registry.slot_id}"
            )
        ranks.append(target.assignment_rank)
        query_sidecar = tuple(sidecar_by_query.get(query.query_id, ()))
        _validate_query_contract(query, registry, target, query_sidecar)
        expected_sidecar_order.extend(query_sidecar)
    if ranks != list(range(1, len(ranks) + 1)):
        raise SemanticAnnotationError("semantic registry must be an assignment-rank prefix")
    if list(state.trake_reviews) != expected_sidecar_order:
        raise SemanticAnnotationError("TRAKE sidecar physical ordering is not deterministic")

    split_markers = ("planned_split", "q1b_dev", "q1b_holdout")
    for artifact in (paths.benchmark, paths.registry, paths.trake_review):
        artifact_text = _read_utf8(artifact, artifact.name).casefold()
        if any(marker in artifact_text for marker in split_markers):
            raise SemanticAnnotationError(f"split leakage detected in {artifact.name}")
        if "\x00" in artifact_text or "\\u0000" in artifact_text:
            raise SemanticAnnotationError(f"NUL detected in {artifact.name}")
        if _contains_machine_local_path(artifact_text):
            raise SemanticAnnotationError(
                f"machine-local absolute path detected in {artifact.name}"
            )
    if Path(paths.benchmark).read_bytes() != _serialize_benchmark(state.benchmark):
        raise SemanticAnnotationError("benchmark serialization/order is not canonical")
    if Path(paths.registry).read_bytes() != _serialize_registry(state.registry):
        raise SemanticAnnotationError("registry serialization/order is not canonical")
    if Path(paths.trake_review).read_bytes() != _serialize_trake_reviews(
        state.trake_reviews
    ):
        raise SemanticAnnotationError("TRAKE sidecar serialization/order is not canonical")
    status_counts = Counter(record.annotation_pass2_status for record in state.registry)
    return {
        "status": "AUDIT_PASSED",
        "valid": True,
        "suitability_assign_count": len(state.targets),
        "benchmark_query_count": len(state.benchmark.queries),
        "registry_row_count": len(state.registry),
        "trake_event_row_count": len(state.trake_reviews),
        "pass2_state_counts": dict(sorted(status_counts.items())),
    }


def load_semantic_state(paths: SemanticPaths | None = None) -> SemanticState:
    """Strict-load and cross-audit all semantic workflow artifacts."""

    resolved = default_paths() if paths is None else paths
    try:
        benchmark = load_quality_benchmark_json(resolved.benchmark)
    except Exception as exc:
        raise SemanticAnnotationError(f"invalid benchmark: {exc}") from exc
    state = SemanticState(
        _semantic_targets(resolved),
        benchmark,
        _load_registry(resolved.registry),
        _load_trake_reviews(resolved.trake_review),
    )
    _audit_loaded(state, resolved)
    return state


def audit(paths: SemanticPaths | None = None) -> dict[str, Any]:
    """Return a machine-readable successful cross-artifact audit."""

    resolved = default_paths() if paths is None else paths
    return _audit_loaded(load_semantic_state(resolved), resolved)


def _target_output(target: SemanticTarget) -> dict[str, Any]:
    return {
        "status": "NEXT_PASS1_TARGET",
        "assignment_rank": target.assignment_rank,
        "review_sequence": target.review_sequence,
        "video_id": target.video_id,
        "slot_id": target.slot_id,
        "task": target.task,
        "target_category": target.target_category,
        "category_name": target.category_name,
        "category_definition": target.category_definition,
        "acceptance_guidance": target.acceptance_guidance,
        "rejection_guidance": target.rejection_guidance,
        "suggested_tags": target.suggested_tags,
        "suitability_notes": target.suitability_notes,
        "derived_query_id": target.derived_query_id,
    }


def pass1_next(paths: SemanticPaths | None = None) -> dict[str, Any]:
    state = load_semantic_state(paths)
    if len(state.registry) >= len(state.targets):
        return {"status": "PASS1_COMPLETE_FOR_ASSIGNED_TARGETS"}
    return _target_output(state.targets[len(state.registry)])


def _target_for_registry(state: SemanticState, registry: RegistryRecord) -> SemanticTarget:
    for target in state.targets:
        if target.slot_id == registry.slot_id:
            return target
    raise SemanticAnnotationError(f"no suitability target for slot {registry.slot_id}")


def _query_human_payload(
    query: QualityQuery,
    sidecar: tuple[TRAKEReviewRecord, ...] = (),
) -> dict[str, Any]:
    ground_truth = _query_ground_truth(query)
    if type(query) is KISQualityQuery:
        return {
            "query_vi": query.query_vi,
            "query_en": query.query_en,
            "query_en_expansion": query.query_en_expansion,
            "ground_truth": {
                "video_id": ground_truth.video_id,
                "start_frame_id": ground_truth.start_frame_id,
                "end_frame_id": ground_truth.end_frame_id,
            },
        }
    if type(query) is QAQualityQuery:
        return {
            "event_description": query.event_description,
            "event_description_en": query.event_description_en,
            "question": query.question,
            "question_en": query.question_en,
            "ground_truth": {
                "video_id": ground_truth.video_id,
                "start_frame_id": ground_truth.start_frame_id,
                "end_frame_id": ground_truth.end_frame_id,
                "accepted_answers": list(ground_truth.accepted_answers),
            },
        }
    trake_query = cast(TRAKEQualityQuery, query)
    if len(sidecar) != len(trake_query.events):
        raise SemanticAnnotationError("TRAKE review target is missing sidecar context")
    return {
        "events": [
            {
                "description": event.description,
                "description_en": event.description_en,
                "moment_definition": review.moment_definition,
                "start_frame_id": interval[0],
                "end_frame_id": interval[1],
            }
            for event, interval, review in zip(
                trake_query.events,
                ground_truth.event_intervals,
                sidecar,
                strict=True,
            )
        ]
    }


def _review_target_output(
    state: SemanticState, index: int, status: str
) -> dict[str, Any]:
    query = state.benchmark.queries[index]
    registry = state.registry[index]
    target = _target_for_registry(state, registry)
    sidecar = tuple(
        item for item in state.trake_reviews if item.query_id == query.query_id
    )
    result = {
        "status": status,
        "query_id": query.query_id,
        "assignment_rank": target.assignment_rank,
        "video_id": target.video_id,
        "slot_id": target.slot_id,
        "task": target.task,
        "target_category": target.target_category,
        "category_name": target.category_name,
        "semantic_content": _query_human_payload(query, sidecar),
        "source_reference": query.source_reference,
        "semantic_definition": registry.semantic_definition,
        "boundary_notes": registry.boundary_notes,
        "answer_notes": registry.answer_notes,
        "annotator_id": registry.annotator_id,
    }
    if status == "NEXT_REVISION_TARGET":
        result["reviewer_id"] = registry.reviewer_id
        result["review_notes"] = registry.review_notes
    return result


def pass2_next(paths: SemanticPaths | None = None) -> dict[str, Any]:
    state = load_semantic_state(paths)
    for index, registry in enumerate(state.registry):
        if registry.annotation_pass2_status == PASS2_PENDING:
            return _review_target_output(state, index, "NEXT_PASS2_TARGET")
    return {"status": "PASS2_QUEUE_EMPTY"}


def revision_next(paths: SemanticPaths | None = None) -> dict[str, Any]:
    state = load_semantic_state(paths)
    for index, registry in enumerate(state.registry):
        if registry.annotation_pass2_status == PASS2_REVISION:
            return _review_target_output(state, index, "NEXT_REVISION_TARGET")
    return {"status": "REVISION_QUEUE_EMPTY"}


def semantic_status(paths: SemanticPaths | None = None) -> dict[str, Any]:
    state = load_semantic_state(paths)
    pass2_counts = Counter(record.annotation_pass2_status for record in state.registry)
    task_counts = Counter(record.task_type for record in state.registry)
    query_counts = Counter(query.annotation_status.value for query in state.benchmark.queries)
    trake_query_ids = {record.query_id for record in state.trake_reviews}
    return {
        "status": "SEMANTIC_STATUS",
        "suitability_assign_count": len(state.targets),
        "pass1_complete_count": len(state.registry),
        "pass2_pending_count": pass2_counts[PASS2_PENDING],
        "revision_required_count": pass2_counts[PASS2_REVISION],
        "verified_count": pass2_counts[PASS2_VERIFIED],
        "benchmark_included_count": sum(record.benchmark_included for record in state.registry),
        "task_counts": {
            "kis": task_counts["kis"],
            "qa": task_counts["qa"],
            "trake": task_counts["trake"],
        },
        "benchmark_query_count": len(state.benchmark.queries),
        "draft_query_count": query_counts[AnnotationStatus.DRAFT.value],
        "verified_query_count": query_counts[AnnotationStatus.VERIFIED.value],
        "trake_sidecar_query_count": len(trake_query_ids),
        "trake_sidecar_event_count": len(state.trake_reviews),
    }


def _write_bytes_exclusive(path: Path, payload: bytes) -> None:
    with Path(path).open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


_replace_file: Callable[[Path, Path], object] = os.replace


def _writer_lock_path(paths: SemanticPaths) -> Path:
    return paths.registry.with_name(".quality_q1b_semantic_annotation.lock")


@contextmanager
def _single_writer_lock(paths: SemanticPaths) -> Iterable[None]:
    """Serialize semantic writers; a stale lock is deliberately fail-closed."""

    lock_path = _writer_lock_path(paths)
    try:
        _write_bytes_exclusive(lock_path, f"pid={os.getpid()}\n".encode("ascii"))
    except FileExistsError as exc:
        raise SemanticAnnotationError(
            "semantic writer lock already exists; verify no writer is active before "
            "removing a stale lock"
        ) from exc
    try:
        yield
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass
        except OSError as exc:
            raise SemanticAnnotationError(
                f"semantic writer lock cleanup failed: {lock_path.name}: {exc}"
            ) from exc


def _transactional_write(
    paths: SemanticPaths,
    benchmark: QualityBenchmark,
    registry: tuple[RegistryRecord, ...],
    trake_reviews: tuple[TRAKEReviewRecord, ...],
    *,
    include_trake: bool,
    replace_file: Callable[[Path, Path], object] | None = None,
) -> None:
    replacements: list[tuple[str, Path, bytes]] = [
        ("benchmark", paths.benchmark, _serialize_benchmark(benchmark)),
        ("registry", paths.registry, _serialize_registry(registry)),
    ]
    if include_trake:
        replacements.append(
            ("trake_review", paths.trake_review, _serialize_trake_reviews(trake_reviews))
        )
    token = uuid.uuid4().hex
    staged: dict[str, Path] = {}
    rollback: dict[str, Path] = {}
    originals: dict[str, bytes] = {}
    replaced: list[str] = []
    replacer = _replace_file if replace_file is None else replace_file
    try:
        for field, canonical, payload in replacements:
            originals[field] = canonical.read_bytes()
            stage = canonical.with_name(f".{canonical.name}.{token}.tmp")
            backup = canonical.with_name(f".{canonical.name}.{token}.rollback")
            _write_bytes_exclusive(stage, payload)
            _write_bytes_exclusive(backup, originals[field])
            staged[field] = stage
            rollback[field] = backup
        temporary_paths = paths
        for field, _canonical, _payload in replacements:
            temporary_paths = temporary_paths._replace(**{field: staged[field]})
        load_semantic_state(temporary_paths)

        for field, canonical, _payload in replacements:
            # Record the attempted destination before invoking a replace hook.  A
            # filesystem wrapper may complete the replace and then raise.
            replaced.append(field)
            replacer(staged[field], canonical)
        load_semantic_state(paths)
    except Exception as exc:
        rollback_errors: list[str] = []
        for field, canonical, _payload in reversed(replacements):
            if field not in replaced:
                continue
            try:
                os.replace(rollback[field], canonical)
            except OSError as rollback_exc:
                rollback_errors.append(f"{canonical.name}: {rollback_exc}")
        if rollback_errors:
            raise SemanticAnnotationError(
                "transaction failed and rollback was incomplete: "
                + "; ".join(rollback_errors)
            ) from exc
        if replaced:
            try:
                load_semantic_state(paths)
            except Exception as audit_exc:
                raise SemanticAnnotationError(
                    f"transaction rollback audit failed: {audit_exc}"
                ) from exc
        raise SemanticAnnotationError(f"semantic artifact transaction failed: {exc}") from exc
    finally:
        active_error = sys.exc_info()[1]
        cleanup_errors: list[str] = []
        for temporary in (*staged.values(), *rollback.values()):
            try:
                temporary.unlink(missing_ok=True)
            except OSError as cleanup_exc:
                cleanup_errors.append(f"{temporary.name}: {cleanup_exc}")
        if cleanup_errors:
            error = SemanticAnnotationError(
                "semantic transaction temporary-file cleanup failed: "
                + "; ".join(cleanup_errors)
            )
            if active_error is not None:
                raise error from active_error
            raise error


def _record_pass1_payload(
    payload: Mapping[str, Any],
    paths: SemanticPaths,
    *,
    replace_file: Callable[[Path, Path], object] | None = None,
) -> dict[str, Any]:
    with _single_writer_lock(paths):
        return _record_pass1_payload_locked(payload, paths, replace_file=replace_file)


def _record_pass1_payload_locked(
    payload: Mapping[str, Any],
    paths: SemanticPaths,
    *,
    replace_file: Callable[[Path, Path], object] | None = None,
) -> dict[str, Any]:
    state = load_semantic_state(paths)
    if len(state.registry) >= len(state.targets):
        raise SemanticAnnotationError("no unannotated ASSIGN target remains")
    target = state.targets[len(state.registry)]
    query, registry, sidecar = _build_pass1_record(payload, target)
    benchmark = replace(state.benchmark, queries=state.benchmark.queries + (query,))
    new_registry = state.registry + (registry,)
    new_trake = state.trake_reviews + sidecar
    _transactional_write(
        paths,
        benchmark,
        new_registry,
        new_trake,
        include_trake=bool(sidecar),
        replace_file=replace_file,
    )
    return {
        "status": "PASS1_RECORDED",
        "query_id": query.query_id,
        "assignment_rank": target.assignment_rank,
        "slot_id": target.slot_id,
        "task": target.task,
    }


def pass1_record(
    input_path: Path,
    paths: SemanticPaths | None = None,
    *,
    replace_file: Callable[[Path, Path], object] | None = None,
) -> dict[str, Any]:
    resolved = default_paths() if paths is None else paths
    return _record_pass1_payload(
        load_strict_json(input_path), resolved, replace_file=replace_file
    )


def _pass2_fields(task: str) -> frozenset[str]:
    extra = (
        {"answers_verified"}
        if task == "qa"
        else {"event_order_verified"}
        if task == "trake"
        else set()
    )
    return COMMON_PASS2_FIELDS | extra


def _validate_pass2_input(
    payload: Mapping[str, Any], query: QualityQuery, registry: RegistryRecord
) -> tuple[str, str, str]:
    _require_fields(payload, _pass2_fields(registry.task_type), "Pass-2 input")
    query_id = _require_identifier(payload["query_id"], "query_id")
    if query_id != query.query_id:
        raise SemanticAnnotationError("stale Pass-2 target query_id")
    reviewer = _require_identifier(payload["reviewer_id"], "reviewer_id")
    if reviewer == registry.annotator_id:
        raise SemanticAnnotationError("Pass-2 reviewer must differ from Pass-1 annotator")
    decision = _require_text(payload["decision"], "decision")
    if decision not in {PASS2_VERIFIED, PASS2_REVISION}:
        raise SemanticAnnotationError("decision must be VERIFIED or REVISION_REQUIRED")
    confirmations = [
        "raw_video_reviewed",
        "semantic_support_verified",
        "video_id_verified",
        "original_frame_coordinates_verified",
        "intervals_verified",
    ]
    if registry.task_type == "qa":
        confirmations.append("answers_verified")
    elif registry.task_type == "trake":
        confirmations.append("event_order_verified")
    values = {field: _strict_bool_input(payload[field], field) for field in confirmations}
    if not values["raw_video_reviewed"]:
        raise SemanticAnnotationError("Pass-2 requires personal raw-video review")
    if decision == PASS2_VERIFIED and not all(values.values()):
        raise SemanticAnnotationError("VERIFIED requires every relevant confirmation=true")
    review_notes = _require_text(
        payload["review_notes"], "review_notes", allow_empty=decision == PASS2_VERIFIED
    )
    return decision, reviewer, review_notes


def _record_pass2_payload(
    payload: Mapping[str, Any],
    paths: SemanticPaths,
    *,
    replace_file: Callable[[Path, Path], object] | None = None,
) -> dict[str, Any]:
    with _single_writer_lock(paths):
        return _record_pass2_payload_locked(payload, paths, replace_file=replace_file)


def _record_pass2_payload_locked(
    payload: Mapping[str, Any],
    paths: SemanticPaths,
    *,
    replace_file: Callable[[Path, Path], object] | None = None,
) -> dict[str, Any]:
    state = load_semantic_state(paths)
    pending_index = next(
        (
            index
            for index, record in enumerate(state.registry)
            if record.annotation_pass2_status == PASS2_PENDING
        ),
        None,
    )
    if pending_index is None:
        raise SemanticAnnotationError("no Pass-2 review-pending query remains")
    query = state.benchmark.queries[pending_index]
    registry = state.registry[pending_index]
    decision, reviewer, review_notes = _validate_pass2_input(payload, query, registry)
    new_query = replace(
        query,
        annotation_status=(
            AnnotationStatus.VERIFIED if decision == PASS2_VERIFIED else AnnotationStatus.DRAFT
        ),
    )
    new_registry_record = registry._replace(
        annotation_pass2_status=decision,
        reviewer_id=reviewer,
        review_notes=review_notes,
        benchmark_included=decision == PASS2_VERIFIED,
    )
    queries = list(state.benchmark.queries)
    queries[pending_index] = new_query
    registries = list(state.registry)
    registries[pending_index] = new_registry_record
    trake_changed = registry.task_type == "trake"
    reviews = tuple(
        review._replace(reviewer_id=reviewer, review_status=decision)
        if review.query_id == query.query_id
        else review
        for review in state.trake_reviews
    )
    _transactional_write(
        paths,
        replace(state.benchmark, queries=tuple(queries)),
        tuple(registries),
        reviews,
        include_trake=trake_changed,
        replace_file=replace_file,
    )
    return {"status": decision, "query_id": query.query_id, "reviewer_id": reviewer}


def pass2_record(
    input_path: Path,
    paths: SemanticPaths | None = None,
    *,
    replace_file: Callable[[Path, Path], object] | None = None,
) -> dict[str, Any]:
    resolved = default_paths() if paths is None else paths
    return _record_pass2_payload(
        load_strict_json(input_path), resolved, replace_file=replace_file
    )


def _revise_pass1_payload(
    payload: Mapping[str, Any],
    paths: SemanticPaths,
    *,
    replace_file: Callable[[Path, Path], object] | None = None,
) -> dict[str, Any]:
    with _single_writer_lock(paths):
        return _revise_pass1_payload_locked(payload, paths, replace_file=replace_file)


def _revise_pass1_payload_locked(
    payload: Mapping[str, Any],
    paths: SemanticPaths,
    *,
    replace_file: Callable[[Path, Path], object] | None = None,
) -> dict[str, Any]:
    state = load_semantic_state(paths)
    revision_index = next(
        (
            index
            for index, record in enumerate(state.registry)
            if record.annotation_pass2_status == PASS2_REVISION
        ),
        None,
    )
    if revision_index is None:
        raise SemanticAnnotationError("no revision-required query remains")
    old_registry = state.registry[revision_index]
    target = _target_for_registry(state, old_registry)
    query, registry, sidecar = _build_pass1_record(payload, target)
    old_query = state.benchmark.queries[revision_index]
    if (
        query.query_id != old_query.query_id
        or registry.slot_id != old_registry.slot_id
        or registry.task_type != old_registry.task_type
        or registry.video_id != old_registry.video_id
    ):
        raise SemanticAnnotationError("revision attempted to change immutable identity")
    queries = list(state.benchmark.queries)
    queries[revision_index] = query
    registries = list(state.registry)
    registries[revision_index] = registry
    reviews = tuple(
        review for review in state.trake_reviews if review.query_id != query.query_id
    ) + sidecar
    target_rank = {item.derived_query_id: item.assignment_rank for item in state.targets}
    reviews = tuple(
        sorted(reviews, key=lambda item: (target_rank[item.query_id], item.event_index))
    )
    _transactional_write(
        paths,
        replace(state.benchmark, queries=tuple(queries)),
        tuple(registries),
        reviews,
        include_trake=old_registry.task_type == "trake",
        replace_file=replace_file,
    )
    return {
        "status": "PASS1_REVISED",
        "query_id": query.query_id,
        "assignment_rank": target.assignment_rank,
        "slot_id": target.slot_id,
    }


def pass1_revise(
    input_path: Path,
    paths: SemanticPaths | None = None,
    *,
    replace_file: Callable[[Path, Path], object] | None = None,
) -> dict[str, Any]:
    resolved = default_paths() if paths is None else paths
    return _revise_pass1_payload(
        load_strict_json(input_path), resolved, replace_file=replace_file
    )


def build_template(slot_id: str, paths: SemanticPaths | None = None) -> dict[str, Any]:
    state = load_semantic_state(paths)
    target = next((item for item in state.targets if item.slot_id == slot_id), None)
    if target is None:
        raise SemanticAnnotationError(f"slot is not an accepted ASSIGN target: {slot_id}")
    common: dict[str, Any] = {
        "expect_assignment_rank": target.assignment_rank,
        "expect_slot_id": target.slot_id,
        "annotator_id": "",
        "raw_video_reviewed": False,
        "query_authored_before_retrieval": False,
        "gt_authored_before_retrieval": False,
        "original_frame_coordinates_verified": False,
        "raw_video_frame_count": None,
        "difficulty": "unknown",
        "tags": [],
        "semantic_definition": "",
        "annotation_notes": "",
        "boundary_notes": "",
        "answer_notes": "",
    }
    if target.task == "kis":
        common.update(
            {
                "query_vi": "",
                "query_en": None,
                "query_en_expansion": None,
                "start_frame_id": None,
                "end_frame_id": None,
            }
        )
    elif target.task == "qa":
        common.update(
            {
                "event_description": "",
                "event_description_en": None,
                "question": "",
                "question_en": None,
                "start_frame_id": None,
                "end_frame_id": None,
                "accepted_answers": [],
            }
        )
    else:
        common["events"] = []
    return {
        "status": "PASS1_INPUT_TEMPLATE",
        "context": {
            "assignment_rank": target.assignment_rank,
            "review_sequence": target.review_sequence,
            "video_id": target.video_id,
            "slot_id": target.slot_id,
            "task": target.task,
            "target_category": target.target_category,
            "category_name": target.category_name,
            "category_definition": target.category_definition,
            "acceptance_guidance": target.acceptance_guidance,
            "rejection_guidance": target.rejection_guidance,
            "suggested_tags": target.suggested_tags,
            "suitability_notes": target.suitability_notes,
            "derived_query_id": target.derived_query_id,
        },
        "input": common,
    }


def _pilot_payload(state: SemanticState) -> dict[str, Any]:
    targets = state.targets[:PILOT15_SIZE]
    if len(targets) != PILOT15_SIZE:
        raise SemanticAnnotationError("PILOT15 export requires at least 15 ASSIGN targets")
    counts = Counter(target.task for target in targets)
    return {
        "schema_version": 1,
        "packet_id": "system_tai-q1b-pilot15-semantic-work-v1",
        "target_count": len(targets),
        "task_counts": {
            "kis": counts["kis"],
            "qa": counts["qa"],
            "trake": counts["trake"],
        },
        "targets": [
            {
                "assignment_rank": target.assignment_rank,
                "review_sequence": target.review_sequence,
                "video_id": target.video_id,
                "slot_id": target.slot_id,
                "task": target.task,
                "category": target.target_category,
                "category_name": target.category_name,
                "category_definition": target.category_definition,
                "acceptance_guidance": target.acceptance_guidance,
                "rejection_guidance": target.rejection_guidance,
                "suggested_tags": target.suggested_tags,
                "suitability_notes": target.suitability_notes,
                "derived_query_id": target.derived_query_id,
            }
            for target in targets
        ],
    }


def pilot15_export(output: Path, paths: SemanticPaths | None = None) -> dict[str, Any]:
    resolved_paths = default_paths() if paths is None else paths
    state = load_semantic_state(resolved_paths)
    payload = _pilot_payload(state)
    destination = Path(output)
    lexical_destination = Path(os.path.abspath(destination))
    resolved_destination = destination.resolve(strict=False)
    lexical_repository = Path(os.path.abspath(REPOSITORY_ROOT))
    if lexical_destination.is_relative_to(
        lexical_repository
    ) or resolved_destination.is_relative_to(REPOSITORY_ROOT.resolve()):
        raise SemanticAnnotationError("PILOT15 output must remain outside the repository")
    protected = {
        item.resolve(strict=False)
        for item in resolved_paths
    }
    if resolved_destination in protected:
        raise SemanticAnnotationError("PILOT15 output must not replace a source artifact")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    serialized = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    try:
        _write_bytes_exclusive(temporary, serialized)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "status": "PILOT15_EXPORTED",
        "output": str(destination),
        "target_count": payload["target_count"],
        "task_counts": payload["task_counts"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Split-blind two-pass human semantic annotation workflow for Q1-B."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("pass1-next", help="show the earliest ASSIGN without Pass-1")
    pass1 = subparsers.add_parser("pass1-record", help="record one strict Pass-1 input")
    pass1.add_argument("--input", type=Path, required=True)
    subparsers.add_parser("pass2-next", help="show the earliest review-pending query")
    pass2 = subparsers.add_parser("pass2-record", help="record one strict Pass-2 decision")
    pass2.add_argument("--input", type=Path, required=True)
    subparsers.add_parser("revision-next", help="show the earliest revision-required query")
    revise = subparsers.add_parser("pass1-revise", help="replace one revision-required Pass-1")
    revise.add_argument("--input", type=Path, required=True)
    subparsers.add_parser("status", help="show split-blind aggregate workflow counts")
    subparsers.add_parser("audit", help="run the strict cross-artifact audit")
    pilot = subparsers.add_parser("pilot15-export", help="export 15 suitability targets")
    pilot.add_argument("--output", type=Path, required=True)
    template = subparsers.add_parser("template", help="print a task-specific Pass-1 template")
    template.add_argument("--slot-id", required=True)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.command == "pass1-next":
        return pass1_next()
    if args.command == "pass1-record":
        return pass1_record(args.input)
    if args.command == "pass2-next":
        return pass2_next()
    if args.command == "pass2-record":
        return pass2_record(args.input)
    if args.command == "revision-next":
        return revision_next()
    if args.command == "pass1-revise":
        return pass1_revise(args.input)
    if args.command == "status":
        return semantic_status()
    if args.command == "audit":
        return audit()
    if args.command == "pilot15-export":
        return pilot15_export(args.output)
    if args.command == "template":
        return build_template(args.slot_id)
    raise SemanticAnnotationError(f"unsupported command: {args.command}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = run(args)
    except (SemanticAnnotationError, OSError, ValueError, TypeError) as exc:
        message = str(exc)
        if HIDDEN_SPLIT_PATTERN.search(message):
            message = "workflow validation failed without exposing hidden assignment metadata"
        error = (
            {"status": "AUDIT_FAILED", "valid": False, "error": message}
            if args.command == "audit"
            else {"status": "ERROR", "error": message}
        )
        print(json.dumps(error, ensure_ascii=False, separators=(",", ":")), file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
