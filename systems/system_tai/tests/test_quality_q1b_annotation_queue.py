from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

SYSTEM_ROOT = Path(__file__).parents[1]
BENCHMARK_DIR = SYSTEM_ROOT / "benchmarks" / "quality_q1b"
SCRIPT_PATH = SYSTEM_ROOT / "scripts" / "quality_q1b_annotation_queue.py"
SPEC = importlib.util.spec_from_file_location("system_tai_quality_q1b_queue", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
QUEUE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(QUEUE)

PATHS = QUEUE.QueuePaths(
    BENCHMARK_DIR / "candidate_video_manifest.csv",
    BENCHMARK_DIR / "annotation_plan.csv",
    BENCHMARK_DIR / "category_codebook.csv",
    BENCHMARK_DIR / "slot_assignment_manifest.csv",
    BENCHMARK_DIR / "candidate_review_log.csv",
)


@pytest.fixture(scope="module")
def loaded_queue():
    return QUEUE.load_queue(PATHS)


def _review(
    sequence: int,
    candidate,
    slot,
    decision: str = QUEUE.SKIP_NO_SUITABLE_EVENT,
    *,
    raw: bool = True,
    before: bool = True,
    reason: str = "not suitable",
):
    if decision == QUEUE.ASSIGN:
        reason = ""
    if decision == QUEUE.SKIP_TECHNICAL_UNREADABLE and reason == "not suitable":
        reason = "cannot decode raw video"
    return QUEUE.ReviewRecord(
        sequence,
        sequence,
        candidate.video_id,
        candidate.selection_hash,
        slot.assignment_rank,
        slot.slot_id,
        slot.planned_task,
        slot.target_category,
        decision,
        "A01",
        raw,
        before,
        reason,
        "",
    )


def _reviews_for(candidates, slots, decisions):
    records = []
    slot_index = 0
    for index, decision in enumerate(decisions, start=1):
        raw = decision != QUEUE.SKIP_TECHNICAL_UNREADABLE
        before = decision != QUEUE.SKIP_TECHNICAL_UNREADABLE
        record = _review(
            index,
            candidates[index - 1],
            slots[slot_index],
            decision,
            raw=raw,
            before=before,
        )
        records.append(record)
        if decision == QUEUE.ASSIGN:
            slot_index += 1
    return tuple(records)


def _paths_with_temp_log(tmp_path: Path):
    review_log = tmp_path / "candidate_review_log.csv"
    with review_log.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(QUEUE.REVIEW_COLUMNS)
    return PATHS._replace(review_log=review_log)


def test_category_codebook_has_exactly_30_rows() -> None:
    plan = QUEUE.load_annotation_plan(PATHS.annotation_plan)
    assert len(QUEUE.load_category_codebook(PATHS.category_codebook, plan)) == 30


def test_codebook_has_10_categories_per_task() -> None:
    plan = QUEUE.load_annotation_plan(PATHS.annotation_plan)
    codebook = QUEUE.load_category_codebook(PATHS.category_codebook, plan)
    counts = {
        task: sum(item.task == task for item in codebook.values())
        for task in ("kis", "qa", "trake")
    }
    assert counts == {"kis": 10, "qa": 10, "trake": 10}


def test_all_plan_categories_resolve_exactly_once() -> None:
    plan = QUEUE.load_annotation_plan(PATHS.annotation_plan)
    codebook = QUEUE.load_category_codebook(PATHS.category_codebook, plan)
    assert all(slot.target_category in codebook for slot in plan)


def test_slot_assignment_manifest_has_60_rows(loaded_queue) -> None:
    _candidates, slots, _reviews, _codebook = loaded_queue
    assert len(slots) == 60


def test_slot_hashes_match_independent_sha256(loaded_queue) -> None:
    _candidates, slots, _reviews, _codebook = loaded_queue
    for slot in slots:
        expected = hashlib.sha256(
            ("system_tai_q1b_slot_v1|" + slot.slot_id).encode("utf-8")
        ).hexdigest()
        assert slot.slot_hash == expected


def test_slot_assignment_ranks_are_contiguous(loaded_queue) -> None:
    _candidates, slots, _reviews, _codebook = loaded_queue
    assert [slot.assignment_rank for slot in slots] == list(range(1, 61))


def test_slot_order_is_input_order_independent() -> None:
    plan = QUEUE.load_annotation_plan(PATHS.annotation_plan)
    assert QUEUE.build_slot_assignments(plan) == QUEUE.build_slot_assignments(reversed(plan))


def test_candidate_manifest_exact_sha_gate_is_enforced(tmp_path: Path) -> None:
    changed = tmp_path / "candidate.csv"
    changed.write_bytes(PATHS.candidate_manifest.read_bytes() + b"\n")
    with pytest.raises(QUEUE.AnnotationQueueError, match="SHA256 mismatch"):
        QUEUE.load_candidate_manifest(changed)


def test_empty_review_log_returns_first_candidate_and_calculated_slot(loaded_queue) -> None:
    candidates, slots, _canonical_reviews, codebook = loaded_queue
    target = QUEUE.resolve_next_target(candidates, slots, (), codebook)
    assert target["review_sequence"] == 1
    assert target["sample_rank"] == 1
    assert target["video_id"] == "L26_V065"
    assert target["assignment_rank"] == 1
    assert target["slot_id"] == "KIS-015"


def test_next_target_does_not_expose_planned_split(loaded_queue) -> None:
    target = QUEUE.resolve_next_target(*loaded_queue[:3], loaded_queue[3])
    assert "planned_split" not in target
    assert "development" not in json.dumps(target)
    assert "holdout" not in json.dumps(target)


def test_one_semantic_skip_advances_candidate_but_keeps_slot(loaded_queue) -> None:
    candidates, slots, _reviews, codebook = loaded_queue
    reviews = _reviews_for(candidates, slots, [QUEUE.SKIP_NO_SUITABLE_EVENT])
    target = QUEUE.resolve_next_target(candidates, slots, reviews, codebook)
    assert (target["sample_rank"], target["slot_id"]) == (2, slots[0].slot_id)


def test_multiple_skips_keep_same_slot(loaded_queue) -> None:
    candidates, slots, _reviews, codebook = loaded_queue
    reviews = _reviews_for(
        candidates,
        slots,
        [QUEUE.SKIP_NO_SUITABLE_EVENT, QUEUE.SKIP_NO_SUITABLE_EVENT],
    )
    target = QUEUE.resolve_next_target(candidates, slots, reviews, codebook)
    assert (target["sample_rank"], target["slot_id"]) == (3, slots[0].slot_id)


def test_assign_advances_candidate_and_slot(loaded_queue) -> None:
    candidates, slots, _reviews, codebook = loaded_queue
    reviews = _reviews_for(candidates, slots, [QUEUE.ASSIGN])
    target = QUEUE.resolve_next_target(candidates, slots, reviews, codebook)
    assert (target["sample_rank"], target["slot_id"]) == (2, slots[1].slot_id)


def test_skip_assign_next_state_is_correct(loaded_queue) -> None:
    candidates, slots, _reviews, codebook = loaded_queue
    reviews = _reviews_for(
        candidates,
        slots,
        [QUEUE.SKIP_NO_SUITABLE_EVENT, QUEUE.ASSIGN],
    )
    target = QUEUE.resolve_next_target(candidates, slots, reviews, codebook)
    assert (target["sample_rank"], target["slot_id"]) == (3, slots[1].slot_id)


def test_technical_unreadable_skip_keeps_same_slot(loaded_queue) -> None:
    candidates, slots, _reviews, codebook = loaded_queue
    reviews = _reviews_for(candidates, slots, [QUEUE.SKIP_TECHNICAL_UNREADABLE])
    target = QUEUE.resolve_next_target(candidates, slots, reviews, codebook)
    assert (target["sample_rank"], target["slot_id"]) == (2, slots[0].slot_id)


def test_candidate_rank_gap_is_rejected(loaded_queue) -> None:
    candidates, slots, _reviews, _codebook = loaded_queue
    bad = _review(1, candidates[0], slots[0])._replace(sample_rank=2)
    with pytest.raises(QUEUE.AnnotationQueueError, match="sample_rank"):
        QUEUE.validate_review_state(candidates, slots, (bad,))


def test_review_sequence_gap_is_rejected(loaded_queue) -> None:
    candidates, slots, _reviews, _codebook = loaded_queue
    bad = _review(1, candidates[0], slots[0])._replace(review_sequence=2, sample_rank=2)
    with pytest.raises(QUEUE.AnnotationQueueError, match="review_sequence"):
        QUEUE.validate_review_state(candidates, slots, (bad,))


def test_wrong_video_id_is_rejected(loaded_queue) -> None:
    candidates, slots, _reviews, _codebook = loaded_queue
    bad = _review(1, candidates[0], slots[0])._replace(video_id="L21_V001")
    with pytest.raises(QUEUE.AnnotationQueueError, match="identity"):
        QUEUE.validate_review_state(candidates, slots, (bad,))


def test_wrong_selection_hash_is_rejected(loaded_queue) -> None:
    candidates, slots, _reviews, _codebook = loaded_queue
    bad = _review(1, candidates[0], slots[0])._replace(selection_hash="0" * 64)
    with pytest.raises(QUEUE.AnnotationQueueError, match="hash mismatch"):
        QUEUE.validate_review_state(candidates, slots, (bad,))


def test_wrong_slot_id_is_rejected(loaded_queue) -> None:
    candidates, slots, _reviews, _codebook = loaded_queue
    bad = _review(1, candidates[0], slots[0])._replace(slot_id="KIS-999")
    with pytest.raises(QUEUE.AnnotationQueueError, match="target slot"):
        QUEUE.validate_review_state(candidates, slots, (bad,))


def test_wrong_target_category_is_rejected(loaded_queue) -> None:
    candidates, slots, _reviews, _codebook = loaded_queue
    bad = _review(1, candidates[0], slots[0])._replace(target_category="KIS-C1")
    with pytest.raises(QUEUE.AnnotationQueueError, match="target slot"):
        QUEUE.validate_review_state(candidates, slots, (bad,))


def test_wrong_task_is_rejected(loaded_queue) -> None:
    candidates, slots, _reviews, _codebook = loaded_queue
    bad = _review(1, candidates[0], slots[0])._replace(planned_task="qa")
    with pytest.raises(QUEUE.AnnotationQueueError, match="target slot"):
        QUEUE.validate_review_state(candidates, slots, (bad,))


def test_duplicate_candidate_is_rejected(loaded_queue) -> None:
    candidates, slots, _reviews, _codebook = loaded_queue
    first = _review(1, candidates[0], slots[0])
    second = _review(2, candidates[1], slots[0])._replace(video_id=candidates[0].video_id)
    with pytest.raises(QUEUE.AnnotationQueueError, match="duplicate reviewed video_id"):
        QUEUE.validate_review_state(candidates, slots, (first, second))


def test_same_slot_assigned_twice_is_rejected(loaded_queue) -> None:
    candidates, slots, _reviews, _codebook = loaded_queue
    first = _review(1, candidates[0], slots[0], QUEUE.ASSIGN)
    second = _review(2, candidates[1], slots[0], QUEUE.ASSIGN)
    with pytest.raises(QUEUE.AnnotationQueueError, match="target slot"):
        QUEUE.validate_review_state(candidates, slots, (first, second))


def test_unknown_decision_is_rejected(loaded_queue) -> None:
    candidates, slots, _reviews, _codebook = loaded_queue
    bad = _review(1, candidates[0], slots[0])._replace(decision="MAYBE")
    with pytest.raises(QUEUE.AnnotationQueueError, match="unknown decision"):
        QUEUE.validate_review_state(candidates, slots, (bad,))


def test_assign_without_raw_video_review_is_rejected(loaded_queue) -> None:
    candidates, slots, _reviews, _codebook = loaded_queue
    bad = _review(1, candidates[0], slots[0], QUEUE.ASSIGN, raw=False)
    with pytest.raises(QUEUE.AnnotationQueueError, match="raw_video_reviewed"):
        QUEUE.validate_review_state(candidates, slots, (bad,))


def test_assign_without_pre_retrieval_confirmation_is_rejected(loaded_queue) -> None:
    candidates, slots, _reviews, _codebook = loaded_queue
    bad = _review(1, candidates[0], slots[0], QUEUE.ASSIGN, before=False)
    with pytest.raises(QUEUE.AnnotationQueueError, match="reviewed_before_retrieval"):
        QUEUE.validate_review_state(candidates, slots, (bad,))


@pytest.mark.parametrize(("raw", "before"), [(False, True), (True, False), (False, False)])
def test_semantic_skip_without_confirmations_is_rejected(loaded_queue, raw, before) -> None:
    candidates, slots, _reviews, _codebook = loaded_queue
    bad = _review(1, candidates[0], slots[0], raw=raw, before=before)
    with pytest.raises(QUEUE.AnnotationQueueError, match="requires"):
        QUEUE.validate_review_state(candidates, slots, (bad,))


def test_semantic_skip_without_reason_is_rejected(loaded_queue) -> None:
    candidates, slots, _reviews, _codebook = loaded_queue
    bad = _review(1, candidates[0], slots[0], reason="")
    with pytest.raises(QUEUE.AnnotationQueueError, match="skip_reason"):
        QUEUE.validate_review_state(candidates, slots, (bad,))


def test_assign_with_skip_reason_is_rejected(loaded_queue) -> None:
    candidates, slots, _reviews, _codebook = loaded_queue
    bad = _review(1, candidates[0], slots[0], QUEUE.ASSIGN)._replace(skip_reason="reason")
    with pytest.raises(QUEUE.AnnotationQueueError, match="empty skip_reason"):
        QUEUE.validate_review_state(candidates, slots, (bad,))


def test_canonical_candidate_review_log_is_valid_progress_state() -> None:
    candidates, slots, reviews, codebook = QUEUE.load_queue(PATHS)
    expected_sequence = list(range(1, len(reviews) + 1))
    assert [review.review_sequence for review in reviews] == expected_sequence
    assert [review.sample_rank for review in reviews] == expected_sequence
    assert all(review.decision in QUEUE.ALLOWED_DECISIONS for review in reviews)

    state = QUEUE.validate_review_state(candidates, slots, reviews)
    target = QUEUE.resolve_next_target(candidates, slots, reviews, codebook)
    if state.assigned_slot_count == len(slots):
        assert target == {"status": "ANNOTATION_SLOT_ASSIGNMENT_COMPLETE"}
    else:
        assert target["status"] == "NEXT_REVIEW_TARGET"
        assert target["review_sequence"] == len(reviews) + 1
        assert target["sample_rank"] == len(reviews) + 1

    serialized = json.dumps(target)
    assert "planned_split" not in serialized
    assert "development" not in serialized
    assert "holdout" not in serialized


def test_next_operation_preserves_annotation_plan() -> None:
    before = PATHS.annotation_plan.read_bytes()
    QUEUE.next_target(PATHS)
    assert PATHS.annotation_plan.read_bytes() == before


def test_temp_record_preserves_annotation_plan(tmp_path: Path) -> None:
    before = PATHS.annotation_plan.read_bytes()
    QUEUE.record_decision(
        paths=_paths_with_temp_log(tmp_path),
        decision=QUEUE.ASSIGN,
        annotator_id="A01",
        raw_video_reviewed=True,
        reviewed_before_retrieval=True,
    )
    assert PATHS.annotation_plan.read_bytes() == before


@pytest.mark.parametrize(
    "filename",
    [
        "benchmark.draft.json",
        "annotation_registry.csv",
        "trake_event_review.csv",
        "provenance.json",
        "candidate_video_manifest.csv",
    ],
)
def test_next_operation_preserves_frozen_artifact(filename: str) -> None:
    path = BENCHMARK_DIR / filename
    before = path.read_bytes()
    QUEUE.next_target(PATHS)
    assert path.read_bytes() == before


def test_normal_cli_output_never_exposes_split(capsys: pytest.CaptureFixture[str]) -> None:
    assert QUEUE.main(["next"]) == 0
    output = capsys.readouterr().out
    assert "planned_split" not in output
    assert "development" not in output
    assert "holdout" not in output


def test_after_60_assignments_plan_is_complete(loaded_queue) -> None:
    candidates, slots, _reviews, codebook = loaded_queue
    reviews = _reviews_for(candidates, slots, [QUEUE.ASSIGN] * 60)
    assert QUEUE.resolve_next_target(candidates, slots, reviews, codebook) == {
        "status": "ANNOTATION_SLOT_ASSIGNMENT_COMPLETE"
    }


def test_candidate_exhaustion_before_plan_completion_fails_closed(loaded_queue) -> None:
    candidates, slots, _reviews, codebook = loaded_queue
    one_candidate = candidates[:1]
    reviews = _reviews_for(one_candidate, slots, [QUEUE.SKIP_NO_SUITABLE_EVENT])
    with pytest.raises(QUEUE.AnnotationQueueError, match="CANDIDATE_INVENTORY_EXHAUSTED"):
        QUEUE.resolve_next_target(one_candidate, slots, reviews, codebook)


def test_record_operation_appends_exactly_one_row(tmp_path: Path) -> None:
    paths = _paths_with_temp_log(tmp_path)
    QUEUE.record_decision(
        paths=paths,
        decision=QUEUE.ASSIGN,
        annotator_id="A01",
        raw_video_reviewed=True,
        reviewed_before_retrieval=True,
    )
    with paths.review_log.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.reader(stream))
    assert len(rows) == 2
    assert rows[1][0:2] == ["1", "1"]
    assert rows[1][2] == "L26_V065"
    assert rows[1][5] == "KIS-015"
    assert rows[1][8] == QUEUE.ASSIGN


def test_record_cannot_cherry_pick_arbitrary_rank(tmp_path: Path) -> None:
    paths = _paths_with_temp_log(tmp_path)
    with pytest.raises(QUEUE.AnnotationQueueError, match="expect-sample-rank"):
        QUEUE.record_decision(
            paths=paths,
            decision=QUEUE.ASSIGN,
            annotator_id="A01",
            raw_video_reviewed=True,
            reviewed_before_retrieval=True,
            expect_sample_rank=10,
        )


def test_record_cannot_override_current_slot(tmp_path: Path) -> None:
    paths = _paths_with_temp_log(tmp_path)
    with pytest.raises(QUEUE.AnnotationQueueError, match="expect-slot-id"):
        QUEUE.record_decision(
            paths=paths,
            decision=QUEUE.ASSIGN,
            annotator_id="A01",
            raw_video_reviewed=True,
            reviewed_before_retrieval=True,
            expect_slot_id="KIS-001",
        )


def test_review_csv_bytes_are_deterministic_for_equivalent_decisions(tmp_path: Path) -> None:
    first_directory = tmp_path / "first"
    second_directory = tmp_path / "second"
    first_directory.mkdir()
    second_directory.mkdir()
    first = _paths_with_temp_log(first_directory)
    second = _paths_with_temp_log(second_directory)
    for paths in (first, second):
        QUEUE.record_decision(
            paths=paths,
            decision=QUEUE.SKIP_NO_SUITABLE_EVENT,
            annotator_id="A01",
            raw_video_reviewed=True,
            reviewed_before_retrieval=True,
            skip_reason="no suitable event",
            notes="reviewed",
        )
    assert first.review_log.read_bytes() == second.review_log.read_bytes()


def test_queue_script_has_no_forbidden_runtime_imports() -> None:
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    forbidden = (
        "system_tai.kis",
        "system_tai.qa",
        "system_tai.trake",
        "system_tai.retrieval",
        "system_tai.refinement",
        "system_tai.features.query_encoder",
        "OperationalKISRuntime",
    )
    assert not any(name in source for name in forbidden)
