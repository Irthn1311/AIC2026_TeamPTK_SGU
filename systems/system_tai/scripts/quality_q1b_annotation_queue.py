"""Deterministic, split-blind human candidate-review queue for Q1-B."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import sys
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import NamedTuple

CANDIDATE_SAMPLING_SEED = "system_tai_q1b_v1"
SLOT_ASSIGNMENT_SEED = "system_tai_q1b_slot_v1"
CANDIDATE_MANIFEST_SHA256 = (
    "d4ef95e0fe51615a436de65760f99c588478f984f27a1cad25337af297aa4661"
)
CANDIDATE_ROW_COUNT = 873
SLOT_ROW_COUNT = 60

CANDIDATE_COLUMNS = ("sample_rank", "video_id", "selection_hash")
PLAN_COLUMNS = (
    "slot_id",
    "planned_task",
    "planned_split",
    "target_category",
    "candidate_video_id",
    "candidate_status",
    "query_id",
    "annotation_status",
    "annotator_id",
    "reviewer_id",
    "notes",
)
CODEBOOK_COLUMNS = (
    "task",
    "category_id",
    "category_name",
    "definition",
    "acceptance_guidance",
    "rejection_guidance",
    "suggested_tags",
)
SLOT_COLUMNS = (
    "assignment_rank",
    "slot_id",
    "planned_task",
    "planned_split",
    "target_category",
    "slot_hash",
)
REVIEW_COLUMNS = (
    "review_sequence",
    "sample_rank",
    "video_id",
    "selection_hash",
    "assignment_rank",
    "slot_id",
    "planned_task",
    "target_category",
    "decision",
    "annotator_id",
    "raw_video_reviewed",
    "reviewed_before_retrieval",
    "skip_reason",
    "notes",
)

ASSIGN = "ASSIGN"
SKIP_NO_SUITABLE_EVENT = "SKIP_NO_SUITABLE_EVENT"
SKIP_TECHNICAL_UNREADABLE = "SKIP_TECHNICAL_UNREADABLE"
ALLOWED_DECISIONS = frozenset(
    {ASSIGN, SKIP_NO_SUITABLE_EVENT, SKIP_TECHNICAL_UNREADABLE}
)

VIDEO_ID_PATTERN = re.compile(r"^[A-Za-z]\d{2}_V\d{3}$")
LOWER_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class AnnotationQueueError(ValueError):
    """Raised when frozen queue artifacts or review state violate the protocol."""


class CandidateRecord(NamedTuple):
    sample_rank: int
    video_id: str
    selection_hash: str


class PlanSlot(NamedTuple):
    slot_id: str
    planned_task: str
    planned_split: str
    target_category: str


class CategoryRecord(NamedTuple):
    task: str
    category_id: str
    category_name: str
    definition: str
    acceptance_guidance: str
    rejection_guidance: str
    suggested_tags: str


class SlotAssignment(NamedTuple):
    assignment_rank: int
    slot_id: str
    planned_task: str
    planned_split: str
    target_category: str
    slot_hash: str


class ReviewRecord(NamedTuple):
    review_sequence: int
    sample_rank: int
    video_id: str
    selection_hash: str
    assignment_rank: int
    slot_id: str
    planned_task: str
    target_category: str
    decision: str
    annotator_id: str
    raw_video_reviewed: bool
    reviewed_before_retrieval: bool
    skip_reason: str
    notes: str


class QueuePaths(NamedTuple):
    candidate_manifest: Path
    annotation_plan: Path
    category_codebook: Path
    slot_manifest: Path
    review_log: Path


class QueueState(NamedTuple):
    consumed_candidate_count: int
    assigned_slot_count: int


def default_paths() -> QueuePaths:
    benchmark_directory = Path(__file__).resolve().parents[1] / "benchmarks" / "quality_q1b"
    return QueuePaths(
        benchmark_directory / "candidate_video_manifest.csv",
        benchmark_directory / "annotation_plan.csv",
        benchmark_directory / "category_codebook.csv",
        benchmark_directory / "slot_assignment_manifest.csv",
        benchmark_directory / "candidate_review_log.csv",
    )


def _strict_csv_rows(path: Path, columns: tuple[str, ...]) -> tuple[dict[str, str], ...]:
    source = Path(path)
    try:
        payload = source.read_bytes()
    except OSError as exc:
        raise AnnotationQueueError(f"cannot read {source.name}: {exc}") from exc
    try:
        text = payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise AnnotationQueueError(f"{source.name} is not valid UTF-8: {exc}") from exc
    if text.startswith("\ufeff"):
        raise AnnotationQueueError(f"{source.name} must not contain a UTF-8 BOM")
    parsed = list(csv.reader(io.StringIO(text, newline="")))
    if not parsed:
        raise AnnotationQueueError(f"{source.name} is empty")
    if tuple(parsed[0]) != columns:
        raise AnnotationQueueError(
            f"{source.name} header mismatch: expected={list(columns)}, actual={parsed[0]}"
        )
    rows: list[dict[str, str]] = []
    for line_number, raw in enumerate(parsed[1:], start=2):
        if len(raw) != len(columns):
            raise AnnotationQueueError(
                f"{source.name} line {line_number} has {len(raw)} columns; "
                f"expected {len(columns)}"
            )
        rows.append(dict(zip(columns, raw, strict=True)))
    return tuple(rows)


def _positive_int(value: str, name: str) -> int:
    if not value.isascii() or not value.isdecimal():
        raise AnnotationQueueError(f"{name} must be a positive canonical integer")
    resolved = int(value)
    if resolved < 1 or value != str(resolved):
        raise AnnotationQueueError(f"{name} must be a positive canonical integer")
    return resolved


def _strict_bool(value: str, name: str) -> bool:
    if value == "true":
        return True
    if value == "false":
        return False
    raise AnnotationQueueError(f"{name} must be exactly 'true' or 'false'")


def _require_nonempty(value: str, name: str) -> str:
    if not value.strip():
        raise AnnotationQueueError(f"{name} must be non-empty")
    return value


def load_candidate_manifest(
    path: Path,
    *,
    expected_sha256: str | None = CANDIDATE_MANIFEST_SHA256,
    expected_rows: int | None = CANDIDATE_ROW_COUNT,
) -> tuple[CandidateRecord, ...]:
    source = Path(path)
    try:
        payload = source.read_bytes()
    except OSError as exc:
        raise AnnotationQueueError(f"cannot read candidate manifest: {exc}") from exc
    digest = hashlib.sha256(payload).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise AnnotationQueueError(
            f"candidate manifest SHA256 mismatch: expected {expected_sha256}, got {digest}"
        )
    raw_rows = _strict_csv_rows(source, CANDIDATE_COLUMNS)
    if expected_rows is not None and len(raw_rows) != expected_rows:
        raise AnnotationQueueError(
            f"candidate manifest row count mismatch: expected {expected_rows}, got {len(raw_rows)}"
        )
    records: list[CandidateRecord] = []
    seen_video_ids: set[str] = set()
    for expected_rank, row in enumerate(raw_rows, start=1):
        rank = _positive_int(row["sample_rank"], "sample_rank")
        if rank != expected_rank:
            raise AnnotationQueueError("candidate sample_rank must be contiguous physical order")
        video_id = row["video_id"]
        if VIDEO_ID_PATTERN.fullmatch(video_id) is None:
            raise AnnotationQueueError(f"malformed candidate video_id: {video_id!r}")
        if video_id in seen_video_ids:
            raise AnnotationQueueError(f"duplicate candidate video_id: {video_id}")
        seen_video_ids.add(video_id)
        selection_hash = row["selection_hash"]
        if LOWER_SHA256_PATTERN.fullmatch(selection_hash) is None:
            raise AnnotationQueueError(f"invalid selection_hash for {video_id}")
        expected_hash = hashlib.sha256(
            f"{CANDIDATE_SAMPLING_SEED}|{video_id}".encode()
        ).hexdigest()
        if selection_hash != expected_hash:
            raise AnnotationQueueError(f"selection_hash mismatch for {video_id}")
        records.append(CandidateRecord(rank, video_id, selection_hash))
    if [(item.selection_hash, item.video_id) for item in records] != sorted(
        (item.selection_hash, item.video_id) for item in records
    ):
        raise AnnotationQueueError("candidate manifest physical order is invalid")
    return tuple(records)


def load_annotation_plan(path: Path) -> tuple[PlanSlot, ...]:
    raw_rows = _strict_csv_rows(path, PLAN_COLUMNS)
    if len(raw_rows) != SLOT_ROW_COUNT:
        raise AnnotationQueueError(f"annotation plan must contain {SLOT_ROW_COUNT} rows")
    records: list[PlanSlot] = []
    seen_slot_ids: set[str] = set()
    for row in raw_rows:
        slot_id = _require_nonempty(row["slot_id"], "slot_id")
        if slot_id in seen_slot_ids:
            raise AnnotationQueueError(f"duplicate annotation-plan slot_id: {slot_id}")
        seen_slot_ids.add(slot_id)
        task = row["planned_task"]
        split = row["planned_split"]
        category = _require_nonempty(row["target_category"], "target_category")
        if task not in {"kis", "qa", "trake"}:
            raise AnnotationQueueError(f"invalid planned_task for {slot_id}")
        if split not in {"development", "holdout"}:
            raise AnnotationQueueError(f"invalid planned_split for {slot_id}")
        if row["candidate_video_id"] or row["query_id"]:
            raise AnnotationQueueError(
                "annotation plan candidate_video_id/query_id must remain empty"
            )
        if row["candidate_status"] != "NOT_SAMPLED":
            raise AnnotationQueueError("annotation plan candidate_status must remain NOT_SAMPLED")
        if row["annotation_status"] != "PLANNED":
            raise AnnotationQueueError("annotation plan annotation_status must remain PLANNED")
        records.append(PlanSlot(slot_id, task, split, category))
    task_counts = Counter(item.planned_task for item in records)
    split_counts = Counter(item.planned_split for item in records)
    if task_counts != Counter({"kis": 25, "qa": 20, "trake": 15}):
        raise AnnotationQueueError(f"annotation plan task counts invalid: {dict(task_counts)}")
    if split_counts != Counter({"development": 41, "holdout": 19}):
        raise AnnotationQueueError(f"annotation plan split counts invalid: {dict(split_counts)}")
    return tuple(records)


def build_slot_assignments(plan: Iterable[PlanSlot]) -> tuple[SlotAssignment, ...]:
    hashed = sorted(
        (
            hashlib.sha256(f"{SLOT_ASSIGNMENT_SEED}|{item.slot_id}".encode()).hexdigest(),
            item.slot_id,
            item,
        )
        for item in plan
    )
    return tuple(
        SlotAssignment(
            rank,
            item.slot_id,
            item.planned_task,
            item.planned_split,
            item.target_category,
            slot_hash,
        )
        for rank, (slot_hash, _slot_id, item) in enumerate(hashed, start=1)
    )


def load_slot_manifest(path: Path, plan: Iterable[PlanSlot]) -> tuple[SlotAssignment, ...]:
    raw_rows = _strict_csv_rows(path, SLOT_COLUMNS)
    records = tuple(
        SlotAssignment(
            _positive_int(row["assignment_rank"], "assignment_rank"),
            row["slot_id"],
            row["planned_task"],
            row["planned_split"],
            row["target_category"],
            row["slot_hash"],
        )
        for row in raw_rows
    )
    expected = build_slot_assignments(plan)
    if records != expected:
        raise AnnotationQueueError(
            "slot assignment manifest disagrees with annotation plan/hash order"
        )
    return records


def load_category_codebook(path: Path, plan: Iterable[PlanSlot]) -> dict[str, CategoryRecord]:
    raw_rows = _strict_csv_rows(path, CODEBOOK_COLUMNS)
    if len(raw_rows) != 30:
        raise AnnotationQueueError("category codebook must contain exactly 30 rows")
    records: dict[str, CategoryRecord] = {}
    for row in raw_rows:
        values = tuple(_require_nonempty(row[column], column) for column in CODEBOOK_COLUMNS)
        record = CategoryRecord(*values)
        if record.category_id in records:
            raise AnnotationQueueError(f"duplicate category_id: {record.category_id}")
        expected_task = (
            "kis"
            if record.category_id.startswith("KIS-")
            else "qa"
            if record.category_id.startswith("QA-")
            else "trake"
            if record.category_id.startswith("TR-")
            else None
        )
        if record.task != expected_task:
            raise AnnotationQueueError(f"task/category mismatch for {record.category_id}")
        records[record.category_id] = record
    expected_ids = {
        *(f"KIS-C{index}" for index in range(1, 11)),
        *(f"QA-C{index}" for index in range(1, 11)),
        *(f"TR-C{index}" for index in range(1, 11)),
    }
    if set(records) != expected_ids:
        raise AnnotationQueueError("category codebook ID set mismatch")
    for slot in plan:
        category = records.get(slot.target_category)
        if category is None or category.task != slot.planned_task:
            raise AnnotationQueueError(f"plan category does not resolve: {slot.target_category}")
    return records


def load_review_log(path: Path) -> tuple[ReviewRecord, ...]:
    raw_rows = _strict_csv_rows(path, REVIEW_COLUMNS)
    return tuple(
        ReviewRecord(
            _positive_int(row["review_sequence"], "review_sequence"),
            _positive_int(row["sample_rank"], "sample_rank"),
            row["video_id"],
            row["selection_hash"],
            _positive_int(row["assignment_rank"], "assignment_rank"),
            row["slot_id"],
            row["planned_task"],
            row["target_category"],
            row["decision"],
            row["annotator_id"],
            _strict_bool(row["raw_video_reviewed"], "raw_video_reviewed"),
            _strict_bool(row["reviewed_before_retrieval"], "reviewed_before_retrieval"),
            row["skip_reason"],
            row["notes"],
        )
        for row in raw_rows
    )


def _validate_decision(record: ReviewRecord) -> None:
    _require_nonempty(record.annotator_id, "annotator_id")
    if record.decision not in ALLOWED_DECISIONS:
        raise AnnotationQueueError(f"unknown decision: {record.decision!r}")
    reason_present = bool(record.skip_reason.strip())
    if record.decision == ASSIGN:
        if not record.raw_video_reviewed:
            raise AnnotationQueueError("ASSIGN requires raw_video_reviewed=true")
        if not record.reviewed_before_retrieval:
            raise AnnotationQueueError("ASSIGN requires reviewed_before_retrieval=true")
        if reason_present:
            raise AnnotationQueueError("ASSIGN requires empty skip_reason")
    elif record.decision == SKIP_NO_SUITABLE_EVENT:
        if not record.raw_video_reviewed:
            raise AnnotationQueueError("semantic skip requires raw_video_reviewed=true")
        if not record.reviewed_before_retrieval:
            raise AnnotationQueueError("semantic skip requires reviewed_before_retrieval=true")
        if not reason_present:
            raise AnnotationQueueError("semantic skip requires non-empty skip_reason")
    else:
        if not reason_present:
            raise AnnotationQueueError("technical skip requires non-empty skip_reason")
        if record.raw_video_reviewed and not record.reviewed_before_retrieval:
            raise AnnotationQueueError(
                "reviewed technical skip requires reviewed_before_retrieval=true"
            )


def validate_review_state(
    candidates: tuple[CandidateRecord, ...],
    slots: tuple[SlotAssignment, ...],
    reviews: tuple[ReviewRecord, ...],
) -> QueueState:
    seen_sequences: set[int] = set()
    seen_ranks: set[int] = set()
    seen_videos: set[str] = set()
    assigned_slots: set[str] = set()
    slot_index = 0
    for row_index, review in enumerate(reviews):
        if review.review_sequence in seen_sequences:
            raise AnnotationQueueError("duplicate review_sequence")
        if review.sample_rank in seen_ranks:
            raise AnnotationQueueError("duplicate sample_rank")
        if review.video_id in seen_videos:
            raise AnnotationQueueError("duplicate reviewed video_id")
        seen_sequences.add(review.review_sequence)
        seen_ranks.add(review.sample_rank)
        seen_videos.add(review.video_id)
        expected_sequence = row_index + 1
        if review.review_sequence != expected_sequence:
            raise AnnotationQueueError("review_sequence must be contiguous from 1")
        if review.sample_rank != expected_sequence:
            raise AnnotationQueueError("sample_rank must equal review_sequence without gaps")
        if row_index >= len(candidates):
            raise AnnotationQueueError("review log consumes beyond candidate inventory")
        if slot_index >= len(slots):
            raise AnnotationQueueError("review row exists after annotation plan completion")
        candidate = candidates[row_index]
        slot = slots[slot_index]
        if (review.video_id, review.selection_hash) != (
            candidate.video_id,
            candidate.selection_hash,
        ):
            raise AnnotationQueueError("review row candidate identity/hash mismatch")
        if (
            review.assignment_rank,
            review.slot_id,
            review.planned_task,
            review.target_category,
        ) != (
            slot.assignment_rank,
            slot.slot_id,
            slot.planned_task,
            slot.target_category,
        ):
            raise AnnotationQueueError("review row does not match the current target slot")
        _validate_decision(review)
        if review.decision == ASSIGN:
            if review.slot_id in assigned_slots:
                raise AnnotationQueueError("one slot cannot be assigned twice")
            assigned_slots.add(review.slot_id)
            slot_index += 1
    return QueueState(len(reviews), slot_index)


def resolve_next_target(
    candidates: tuple[CandidateRecord, ...],
    slots: tuple[SlotAssignment, ...],
    reviews: tuple[ReviewRecord, ...],
    codebook: Mapping[str, CategoryRecord],
) -> dict[str, object]:
    state = validate_review_state(candidates, slots, reviews)
    if state.assigned_slot_count == len(slots):
        return {"status": "ANNOTATION_SLOT_ASSIGNMENT_COMPLETE"}
    if state.consumed_candidate_count >= len(candidates):
        raise AnnotationQueueError("CANDIDATE_INVENTORY_EXHAUSTED_BEFORE_PLAN_COMPLETE")
    candidate = candidates[state.consumed_candidate_count]
    slot = slots[state.assigned_slot_count]
    category = codebook.get(slot.target_category)
    if category is None:
        raise AnnotationQueueError(f"missing category codebook entry: {slot.target_category}")
    return {
        "status": "NEXT_REVIEW_TARGET",
        "review_sequence": state.consumed_candidate_count + 1,
        "sample_rank": candidate.sample_rank,
        "video_id": candidate.video_id,
        "selection_hash": candidate.selection_hash,
        "assignment_rank": slot.assignment_rank,
        "slot_id": slot.slot_id,
        "planned_task": slot.planned_task,
        "target_category": slot.target_category,
        "category_name": category.category_name,
        "category_definition": category.definition,
        "acceptance_guidance": category.acceptance_guidance,
        "rejection_guidance": category.rejection_guidance,
        "suggested_tags": category.suggested_tags,
    }


def load_queue(
    paths: QueuePaths,
) -> tuple[
    tuple[CandidateRecord, ...],
    tuple[SlotAssignment, ...],
    tuple[ReviewRecord, ...],
    dict[str, CategoryRecord],
]:
    candidates = load_candidate_manifest(paths.candidate_manifest)
    plan = load_annotation_plan(paths.annotation_plan)
    codebook = load_category_codebook(paths.category_codebook, plan)
    slots = load_slot_manifest(paths.slot_manifest, plan)
    reviews = load_review_log(paths.review_log)
    validate_review_state(candidates, slots, reviews)
    return candidates, slots, reviews, codebook


def next_target(paths: QueuePaths | None = None) -> dict[str, object]:
    resolved_paths = default_paths() if paths is None else paths
    candidates, slots, reviews, codebook = load_queue(resolved_paths)
    return resolve_next_target(candidates, slots, reviews, codebook)


def _write_review_log(path: Path, records: Iterable[ReviewRecord]) -> None:
    destination = Path(path)
    temporary = destination.with_name(f".{destination.name}.tmp")
    if temporary.exists():
        raise AnnotationQueueError(f"temporary review-log path already exists: {temporary}")
    try:
        with temporary.open("x", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(REVIEW_COLUMNS)
            for record in records:
                writer.writerow(
                    (
                        record.review_sequence,
                        record.sample_rank,
                        record.video_id,
                        record.selection_hash,
                        record.assignment_rank,
                        record.slot_id,
                        record.planned_task,
                        record.target_category,
                        record.decision,
                        record.annotator_id,
                        str(record.raw_video_reviewed).lower(),
                        str(record.reviewed_before_retrieval).lower(),
                        record.skip_reason,
                        record.notes,
                    )
                )
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def record_decision(
    *,
    paths: QueuePaths,
    decision: str,
    annotator_id: str,
    raw_video_reviewed: bool,
    reviewed_before_retrieval: bool,
    skip_reason: str = "",
    notes: str = "",
    expect_sample_rank: int | None = None,
    expect_slot_id: str | None = None,
) -> dict[str, object]:
    candidates, slots, reviews, codebook = load_queue(paths)
    target = resolve_next_target(candidates, slots, reviews, codebook)
    if target["status"] != "NEXT_REVIEW_TARGET":
        raise AnnotationQueueError("annotation slot assignment is already complete")
    if expect_sample_rank is not None and expect_sample_rank != target["sample_rank"]:
        raise AnnotationQueueError("--expect-sample-rank does not match current target")
    if expect_slot_id is not None and expect_slot_id != target["slot_id"]:
        raise AnnotationQueueError("--expect-slot-id does not match current target")
    record = ReviewRecord(
        review_sequence=int(target["review_sequence"]),
        sample_rank=int(target["sample_rank"]),
        video_id=str(target["video_id"]),
        selection_hash=str(target["selection_hash"]),
        assignment_rank=int(target["assignment_rank"]),
        slot_id=str(target["slot_id"]),
        planned_task=str(target["planned_task"]),
        target_category=str(target["target_category"]),
        decision=decision,
        annotator_id=annotator_id,
        raw_video_reviewed=raw_video_reviewed,
        reviewed_before_retrieval=reviewed_before_retrieval,
        skip_reason=skip_reason,
        notes=notes,
    )
    _validate_decision(record)
    updated = reviews + (record,)
    validate_review_state(candidates, slots, updated)
    _write_review_log(paths.review_log, updated)
    return {
        "status": "REVIEW_DECISION_RECORDED",
        "review_sequence": record.review_sequence,
        "sample_rank": record.sample_rank,
        "video_id": record.video_id,
        "assignment_rank": record.assignment_rank,
        "slot_id": record.slot_id,
        "planned_task": record.planned_task,
        "target_category": record.target_category,
        "decision": record.decision,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="operation", required=True)
    subparsers.add_parser("next", help="Print the next split-blind review target")
    record = subparsers.add_parser("record", help="Append exactly one current-target decision")
    record.add_argument("--decision", required=True, choices=tuple(sorted(ALLOWED_DECISIONS)))
    record.add_argument("--annotator-id", required=True)
    record.add_argument("--skip-reason", default="")
    record.add_argument("--notes", default="")
    record.add_argument("--raw-video-reviewed", action="store_true")
    record.add_argument("--reviewed-before-retrieval", action="store_true")
    record.add_argument("--expect-sample-rank", type=int)
    record.add_argument("--expect-slot-id")
    return parser


def run(args: argparse.Namespace) -> int:
    if args.operation == "next":
        result = next_target()
    else:
        result = record_decision(
            paths=default_paths(),
            decision=args.decision,
            annotator_id=args.annotator_id,
            raw_video_reviewed=args.raw_video_reviewed,
            reviewed_before_retrieval=args.reviewed_before_retrieval,
            skip_reason=args.skip_reason,
            notes=args.notes,
            expect_sample_rank=args.expect_sample_rank,
            expect_slot_id=args.expect_slot_id,
        )
    print(json.dumps(result, ensure_ascii=False, separators=(",", ":")))
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        return run(build_parser().parse_args(argv))
    except (AnnotationQueueError, OSError) as exc:
        print(f"Q1-B annotation queue failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
