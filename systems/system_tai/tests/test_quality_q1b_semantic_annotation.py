from __future__ import annotations

import ast
import csv
import hashlib
import importlib.util
import json
import os
import random
import shutil
import sys
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

SYSTEM_ROOT = Path(__file__).parents[1]
BENCHMARK_DIR = SYSTEM_ROOT / "benchmarks" / "quality_q1b"
SCRIPT_PATH = SYSTEM_ROOT / "scripts" / "quality_q1b_semantic_annotation.py"
SPEC = importlib.util.spec_from_file_location(
    "system_tai_quality_q1b_semantic_annotation_tests", SCRIPT_PATH
)
assert SPEC is not None and SPEC.loader is not None
SEMANTIC = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SEMANTIC
SPEC.loader.exec_module(SEMANTIC)

EXPECTED_REVIEW_SHA256 = (
    "41ee3117146ed446602b0a36422097173595fbf53f8fff07c7c22f46bc5f8a8e"
)


@pytest.fixture
def semantic_paths(tmp_path: Path):
    copied = tmp_path / "quality_q1b"
    copied.mkdir()
    for filename in (
        "candidate_video_manifest.csv",
        "annotation_plan.csv",
        "category_codebook.csv",
        "slot_assignment_manifest.csv",
        "candidate_review_log.csv",
    ):
        shutil.copy2(BENCHMARK_DIR / filename, copied / filename)

    paths = SEMANTIC.SemanticPaths(
        copied / "candidate_video_manifest.csv",
        copied / "annotation_plan.csv",
        copied / "category_codebook.csv",
        copied / "slot_assignment_manifest.csv",
        copied / "candidate_review_log.csv",
        copied / "benchmark.draft.json",
        copied / "annotation_registry.csv",
        copied / "trake_event_review.csv",
    )
    canonical = SEMANTIC.load_quality_benchmark_json(
        BENCHMARK_DIR / "benchmark.draft.json"
    )
    paths.benchmark.write_bytes(
        SEMANTIC._serialize_benchmark(replace(canonical, queries=tuple()))
    )
    paths.registry.write_bytes(SEMANTIC._serialize_registry(tuple()))
    paths.trake_review.write_bytes(SEMANTIC._serialize_trake_reviews(tuple()))
    return paths


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return path


def _current_target(paths) -> dict[str, Any]:
    target = SEMANTIC.pass1_next(paths)
    assert target["status"] == "NEXT_PASS1_TARGET"
    return target


def _common_payload(target: dict[str, Any]) -> dict[str, Any]:
    return {
        "expect_assignment_rank": target["assignment_rank"],
        "expect_slot_id": target["slot_id"],
        "annotator_id": "annotator-01",
        "raw_video_reviewed": True,
        "query_authored_before_retrieval": True,
        "gt_authored_before_retrieval": True,
        "original_frame_coordinates_verified": True,
        "raw_video_frame_count": 1_000,
        "difficulty": "medium",
        "tags": ["human_visible", "temporal_boundary"],
        "semantic_definition": "Human-authored semantic definition.",
        "annotation_notes": "Authored directly from the raw video.",
        "boundary_notes": "Inclusive original-frame interval.",
        "answer_notes": "Human-authored answer notes.",
    }


def _valid_payload(target: dict[str, Any], *, event_count: int = 2) -> dict[str, Any]:
    payload = _common_payload(target)
    if target["task"] == "kis":
        payload.update(
            {
                "query_vi": "Một người đi qua cửa.",
                "query_en": "A person walks through a door.",
                "query_en_expansion": None,
                "start_frame_id": 10,
                "end_frame_id": 20,
            }
        )
    elif target["task"] == "qa":
        payload.update(
            {
                "event_description": "Một người cầm vật thể.",
                "event_description_en": "A person holds an object.",
                "question": "Vật thể có màu gì?",
                "question_en": "What color is the object?",
                "start_frame_id": 30,
                "end_frame_id": 45,
                "accepted_answers": ["đỏ", "màu đỏ"],
            }
        )
    else:
        payload["events"] = [
            {
                "description": f"Sự kiện {index + 1}",
                "description_en": f"Event {index + 1}",
                "moment_definition": f"Visible event boundary {index + 1}.",
                "start_frame_id": 10 + index * 20,
                "end_frame_id": 15 + index * 20,
            }
            for index in range(event_count)
        ]
    return payload


def _record_current(
    paths,
    tmp_path: Path,
    *,
    event_count: int = 2,
    mutate: Callable[[dict[str, Any]], None] | None = None,
    replace_file=None,
) -> dict[str, Any]:
    target = _current_target(paths)
    payload = _valid_payload(target, event_count=event_count)
    if mutate is not None:
        mutate(payload)
    input_path = _write_json(
        tmp_path / f"pass1-{target['assignment_rank']}-{id(payload)}.json", payload
    )
    return SEMANTIC.pass1_record(input_path, paths, replace_file=replace_file)


def _record_through_task(paths, tmp_path: Path, task: str, *, event_count: int = 2):
    while True:
        target = _current_target(paths)
        result = _record_current(paths, tmp_path, event_count=event_count)
        if target["task"] == task:
            return result


def _pass2_payload(target: dict[str, Any], decision: str = "VERIFIED") -> dict[str, Any]:
    payload: dict[str, Any] = {
        "query_id": target["query_id"],
        "reviewer_id": "reviewer-02",
        "decision": decision,
        "raw_video_reviewed": True,
        "semantic_support_verified": True,
        "video_id_verified": True,
        "original_frame_coordinates_verified": True,
        "intervals_verified": True,
        "review_notes": "" if decision == "VERIFIED" else "Revise the boundary.",
    }
    if target["task"] == "qa":
        payload["answers_verified"] = True
    elif target["task"] == "trake":
        payload["event_order_verified"] = True
    return payload


def _record_pass2(
    paths,
    tmp_path: Path,
    *,
    decision: str = "VERIFIED",
    mutate: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    target = SEMANTIC.pass2_next(paths)
    assert target["status"] == "NEXT_PASS2_TARGET"
    payload = _pass2_payload(target, decision)
    if mutate is not None:
        mutate(payload)
    input_path = _write_json(tmp_path / f"pass2-{target['query_id']}-{id(payload)}.json", payload)
    return SEMANTIC.pass2_record(input_path, paths)


def _advance_pass2_to_task(paths, tmp_path: Path, task: str) -> dict[str, Any]:
    while True:
        target = SEMANTIC.pass2_next(paths)
        assert target["status"] == "NEXT_PASS2_TARGET"
        if target["task"] == task:
            return target
        _record_pass2(paths, tmp_path)


def _artifact_bytes(paths) -> tuple[bytes, bytes, bytes]:
    return (
        paths.benchmark.read_bytes(),
        paths.registry.read_bytes(),
        paths.trake_review.read_bytes(),
    )


def _csv_rows(path: Path) -> list[list[str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.reader(stream))


def _write_csv_rows(path: Path, rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerows(rows)


def _benchmark_payload(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _copy_paths(source: Path, destination: Path):
    shutil.copytree(source, destination)
    return SEMANTIC.SemanticPaths(
        destination / "candidate_video_manifest.csv",
        destination / "annotation_plan.csv",
        destination / "category_codebook.csv",
        destination / "slot_assignment_manifest.csv",
        destination / "candidate_review_log.csv",
        destination / "benchmark.draft.json",
        destination / "annotation_registry.csv",
        destination / "trake_event_review.csv",
    )


def _write_suitability_pattern(paths, decisions: list[str]) -> None:
    with paths.candidate_manifest.open(encoding="utf-8", newline="") as stream:
        candidates = list(csv.DictReader(stream))
    with paths.slot_manifest.open(encoding="utf-8", newline="") as stream:
        slots = list(csv.DictReader(stream))
    header = _csv_rows(paths.review_log)[0]
    rows: list[list[str]] = [header]
    assignment_index = 0
    for sequence, decision in enumerate(decisions, start=1):
        slot = slots[assignment_index]
        candidate = candidates[sequence - 1]
        semantic_skip = decision == "SKIP_NO_SUITABLE_EVENT"
        technical_skip = decision == "SKIP_TECHNICAL_UNREADABLE"
        rows.append(
            [
                str(sequence),
                str(sequence),
                candidate["video_id"],
                candidate["selection_hash"],
                slot["assignment_rank"],
                slot["slot_id"],
                slot["planned_task"],
                slot["target_category"],
                decision,
                "A01",
                "false" if technical_skip else "true",
                "false" if technical_skip else "true",
                "" if decision == "ASSIGN" else "No suitable semantic evidence.",
                "Assigned after raw review." if decision == "ASSIGN" else "Skipped by policy.",
            ]
        )
        if not semantic_skip and not technical_skip:
            assignment_index += 1
    _write_csv_rows(paths.review_log, rows)


def _seed_pass1_prefix(paths, count: int) -> None:
    state = SEMANTIC.load_semantic_state(paths)
    queries = []
    registry = []
    trake_rows = []
    for target in state.targets[:count]:
        query, record, sidecar = SEMANTIC._build_pass1_record(
            _valid_payload(SEMANTIC._target_output(target)), target
        )
        queries.append(query)
        registry.append(record)
        trake_rows.extend(sidecar)
    paths.benchmark.write_bytes(
        SEMANTIC._serialize_benchmark(replace(state.benchmark, queries=tuple(queries)))
    )
    paths.registry.write_bytes(SEMANTIC._serialize_registry(tuple(registry)))
    paths.trake_review.write_bytes(SEMANTIC._serialize_trake_reviews(tuple(trake_rows)))
    SEMANTIC.audit(paths)


def _set_pass2_states(paths, statuses: list[str]) -> None:
    state = SEMANTIC.load_semantic_state(paths)
    assert len(statuses) == len(state.registry)
    queries = []
    records = []
    state_by_query: dict[str, str] = {}
    for query, record, status in zip(
        state.benchmark.queries, state.registry, statuses, strict=True
    ):
        state_by_query[query.query_id] = status
        if status == "VERIFIED":
            queries.append(replace(query, annotation_status=SEMANTIC.AnnotationStatus.VERIFIED))
            records.append(
                record._replace(
                    annotation_pass2_status="VERIFIED",
                    reviewer_id="reviewer-02",
                    review_notes="",
                    benchmark_included=True,
                )
            )
        elif status == "REVISION_REQUIRED":
            queries.append(replace(query, annotation_status=SEMANTIC.AnnotationStatus.DRAFT))
            records.append(
                record._replace(
                    annotation_pass2_status="REVISION_REQUIRED",
                    reviewer_id="reviewer-02",
                    review_notes="Needs revision.",
                    benchmark_included=False,
                )
            )
        else:
            queries.append(query)
            records.append(record)
    reviews = tuple(
        review._replace(
            reviewer_id=(
                "" if state_by_query[review.query_id] == "REVIEW_PENDING" else "reviewer-02"
            ),
            review_status=state_by_query[review.query_id],
        )
        for review in state.trake_reviews
    )
    paths.benchmark.write_bytes(
        SEMANTIC._serialize_benchmark(replace(state.benchmark, queries=tuple(queries)))
    )
    paths.registry.write_bytes(SEMANTIC._serialize_registry(tuple(records)))
    paths.trake_review.write_bytes(SEMANTIC._serialize_trake_reviews(reviews))
    SEMANTIC.audit(paths)


def _assert_cli_json(capsys, expected_code: int, invoke: Callable[[], int]) -> dict[str, Any]:
    assert invoke() == expected_code
    captured = capsys.readouterr()
    selected = captured.out if expected_code == 0 else captured.err
    other = captured.err if expected_code == 0 else captured.out
    assert other == ""
    assert "Traceback" not in selected
    assert selected.count("\n") == 1
    return json.loads(selected)


def test_real_semantic_artifacts_are_valid_and_keep_verified_checkpoint() -> None:
    paths = SEMANTIC.default_paths()
    assert SEMANTIC.audit(paths)["valid"] is True
    state = SEMANTIC.load_semantic_state(paths)

    query = next(
        item for item in state.benchmark.queries if item.query_id == "q1b-kis-015"
    )
    record = next(item for item in state.registry if item.query_id == query.query_id)
    assert query.annotation_status.value == "verified"
    assert query.ground_truth.video_id == "L26_V065"
    assert (query.ground_truth.start_frame_id, query.ground_truth.end_frame_id) == (
        657,
        911,
    )
    assert record.annotation_pass2_status == "VERIFIED"
    assert record.benchmark_included is True


def test_real_review_log_sha_is_frozen() -> None:
    digest = hashlib.sha256((BENCHMARK_DIR / "candidate_review_log.csv").read_bytes())
    assert digest.hexdigest() == EXPECTED_REVIEW_SHA256


def test_deterministic_query_ids() -> None:
    assert SEMANTIC.derive_query_id("KIS-015") == "q1b-kis-015"
    assert SEMANTIC.derive_query_id("QA-015") == "q1b-qa-015"
    assert SEMANTIC.derive_query_id("TRAKE-001") == "q1b-trake-001"


@pytest.mark.parametrize(
    "value",
    [
        "",
        "   ",
        None,
        7,
        "KIS-1",
        "kis-015",
        "QA--015",
        "TRAKE-000",
        "OTHER-001",
        "KIS-015-extra",
        " KIS-015",
        "KIS-015 ",
        "KІS-015",
    ],
)
def test_query_id_rejects_invalid_slot(value: object) -> None:
    with pytest.raises(SEMANTIC.SemanticAnnotationError):
        SEMANTIC.derive_query_id(value)


def test_strict_json_accepts_utf8_without_bom(tmp_path: Path) -> None:
    path = tmp_path / "unicode.json"
    path.write_text('{"text":"Tiếng Việt"}\n', encoding="utf-8")
    assert SEMANTIC.load_strict_json(path) == {"text": "Tiếng Việt"}


def test_strict_json_rejects_bom(tmp_path: Path) -> None:
    path = tmp_path / "bom.json"
    path.write_bytes(b"\xef\xbb\xbf{}")
    with pytest.raises(SEMANTIC.SemanticAnnotationError, match="BOM"):
        SEMANTIC.load_strict_json(path)


def test_strict_json_rejects_invalid_utf8(tmp_path: Path) -> None:
    path = tmp_path / "invalid.json"
    path.write_bytes(b'{"x":"\xff"}')
    with pytest.raises(SEMANTIC.SemanticAnnotationError, match="UTF-8"):
        SEMANTIC.load_strict_json(path)


def test_strict_json_rejects_duplicate_key(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text('{"x":1,"x":2}', encoding="utf-8")
    with pytest.raises(SEMANTIC.SemanticAnnotationError, match="duplicate JSON"):
        SEMANTIC.load_strict_json(path)


@pytest.mark.parametrize(
    "payload",
    [
        '{"events":[{"start":1,"start":2}]}',
        '{"outer":{"answer":"a","answer":"b"}}',
    ],
)
def test_strict_json_rejects_nested_duplicate_keys(tmp_path: Path, payload: str) -> None:
    path = tmp_path / "nested-duplicate.json"
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(SEMANTIC.SemanticAnnotationError, match="duplicate JSON"):
        SEMANTIC.load_strict_json(path)


def test_strict_json_rejects_top_level_list(tmp_path: Path) -> None:
    path = tmp_path / "list.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(SEMANTIC.SemanticAnnotationError, match="one JSON object"):
        SEMANTIC.load_strict_json(path)


@pytest.mark.parametrize(
    "payload",
    [
        b"wrong,header\n",
        b"query_id,query_id\n",
        b"\xef\xbb\xbfquery_id\n",
        b"\xff",
    ],
)
def test_registry_csv_rejects_header_bom_and_encoding_errors(
    tmp_path: Path, payload: bytes
) -> None:
    path = tmp_path / "bad-registry.csv"
    path.write_bytes(payload)
    with pytest.raises(SEMANTIC.SemanticAnnotationError):
        SEMANTIC._load_registry(path)


@pytest.mark.parametrize("row_suffix", [",extra", "", ",", ",,,"])
def test_registry_csv_rejects_wrong_column_counts(tmp_path: Path, row_suffix: str) -> None:
    header = ",".join(SEMANTIC.REGISTRY_COLUMNS)
    base = ["q"] * len(SEMANTIC.REGISTRY_COLUMNS)
    row = ",".join(base[:-1]) + row_suffix
    path = tmp_path / "wrong-columns.csv"
    path.write_text(f"{header}\n{row}\n", encoding="utf-8")
    with pytest.raises(SEMANTIC.SemanticAnnotationError):
        SEMANTIC._load_registry(path)


def test_registry_csv_rejects_blank_extra_physical_row(tmp_path: Path) -> None:
    path = tmp_path / "blank-row.csv"
    path.write_bytes(SEMANTIC._serialize_registry(()) + b"\n")
    with pytest.raises(SEMANTIC.SemanticAnnotationError, match="columns"):
        SEMANTIC._load_registry(path)


def test_csv_quotes_commas_multiline_and_unicode_roundtrip_with_lf(
    semantic_paths, tmp_path: Path
) -> None:
    note = 'ĐỒNG BẰNG SÔNG HỒNG, dấu "ngoặc"; dòng một:\ndòng hai'
    _record_current(
        semantic_paths,
        tmp_path,
        mutate=lambda payload: payload.update(boundary_notes=note, answer_notes=note),
    )
    registry = SEMANTIC.load_semantic_state(semantic_paths).registry[0]
    assert registry.boundary_notes == note
    assert registry.answer_notes == note
    raw = semantic_paths.registry.read_bytes()
    assert b"\r\n" not in raw
    assert raw.startswith(b"query_id,")
    assert "ĐỒNG BẰNG SÔNG HỒNG".encode() in raw


def test_noncanonical_crlf_registry_is_rejected_by_cross_audit(
    semantic_paths, tmp_path: Path
) -> None:
    _record_current(semantic_paths, tmp_path)
    semantic_paths.registry.write_bytes(
        semantic_paths.registry.read_bytes().replace(b"\n", b"\r\n")
    )
    with pytest.raises(SEMANTIC.SemanticAnnotationError, match="canonical"):
        SEMANTIC.audit(semantic_paths)


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_strict_json_rejects_nonfinite_number(tmp_path: Path, constant: str) -> None:
    path = tmp_path / "nonfinite.json"
    path.write_text(f'{{"x":{constant}}}', encoding="utf-8")
    with pytest.raises(SEMANTIC.SemanticAnnotationError, match="non-finite"):
        SEMANTIC.load_strict_json(path)


def test_pass1_initial_target_is_frozen_and_split_blind(semantic_paths) -> None:
    target = SEMANTIC.pass1_next(semantic_paths)
    assert target == {
        "status": "NEXT_PASS1_TARGET",
        "assignment_rank": 1,
        "review_sequence": 1,
        "video_id": "L26_V065",
        "slot_id": "KIS-015",
        "task": "kis",
        "target_category": "KIS-C5",
        "category_name": "SMALL_OBJECT",
        "category_definition": (
            "A semantically important object occupies a relatively small image area or is "
            "difficult to notice."
        ),
        "acceptance_guidance": (
            "The small object is genuinely visible and relevant to the authored query."
        ),
        "rejection_guidance": "Object cannot be verified at native/raw-video evidence.",
        "suggested_tags": "small_object",
        "suitability_notes": (
            "Suitable SMALL_OBJECT example. Raw video was reviewed manually before retrieval. "
            "Around the ingredient presentation, multiple small condiment bowls and chopped "
            "ingredients are clearly visible; for example a small black bowl containing chopped "
            "red chili."
        ),
        "derived_query_id": "q1b-kis-015",
    }
    text = json.dumps(target).casefold()
    assert "planned_split" not in text
    assert "development" not in text
    assert "holdout" not in text


def test_status_initial_counts_and_split_blindness(semantic_paths) -> None:
    status = SEMANTIC.semantic_status(semantic_paths)
    assert status["suitability_assign_count"] == 15
    assert status["pass1_complete_count"] == 0
    assert status["verified_count"] == 0
    assert status["benchmark_query_count"] == 0
    assert "split" not in json.dumps(status).casefold()


@pytest.mark.parametrize("slot_id", ["KIS-015", "QA-015", "TRAKE-001"])
def test_templates_are_task_specific_empty_and_split_blind(semantic_paths, slot_id: str) -> None:
    template = SEMANTIC.build_template(slot_id, semantic_paths)
    assert template["input"]["expect_slot_id"] == slot_id
    assert template["input"]["raw_video_reviewed"] is False
    assert "planned_split" not in json.dumps(template).casefold()
    assert "/kaggle/" not in json.dumps(template)
    assert ":\\" not in json.dumps(template)
    task = template["context"]["task"]
    if task == "kis":
        assert template["input"]["query_vi"] == ""
        assert template["input"]["start_frame_id"] is None
    elif task == "qa":
        assert template["input"]["question"] == ""
        assert template["input"]["accepted_answers"] == []
    else:
        assert template["input"]["events"] == []


def test_fresh_template_input_cannot_be_submitted_unchanged(semantic_paths, tmp_path: Path) -> None:
    template = SEMANTIC.build_template("KIS-015", semantic_paths)
    before = _artifact_bytes(semantic_paths)
    with pytest.raises(SEMANTIC.SemanticAnnotationError):
        SEMANTIC.pass1_record(
            _write_json(tmp_path / "unchanged-template.json", template["input"]), semantic_paths
        )
    assert _artifact_bytes(semantic_paths) == before


def test_pilot15_export_counts_order_determinism_and_split_blindness(
    semantic_paths, tmp_path: Path
) -> None:
    first = tmp_path / "pilot-first.json"
    second = tmp_path / "pilot-second.json"
    SEMANTIC.pilot15_export(first, semantic_paths)
    SEMANTIC.pilot15_export(second, semantic_paths)
    assert first.read_bytes() == second.read_bytes()
    packet = json.loads(first.read_text(encoding="utf-8"))
    assert packet["target_count"] == 15
    assert packet["task_counts"] == {"kis": 8, "qa": 5, "trake": 2}
    assert [item["assignment_rank"] for item in packet["targets"]] == list(range(1, 16))
    text = first.read_text(encoding="utf-8").casefold()
    assert "planned_split" not in text
    assert "ground_truth" not in text
    forbidden_keys = {
        "retrieval_rank",
        "retrieval_score",
        "clip_score",
        "model_prediction",
        "ground_truth",
        "planned_split",
    }
    assert all(forbidden_keys.isdisjoint(item) for item in packet["targets"])


def test_pilot15_export_rejects_repository_destination(semantic_paths) -> None:
    destination = SYSTEM_ROOT / "forbidden-pilot15-export.json"
    assert not destination.exists()
    with pytest.raises(SEMANTIC.SemanticAnnotationError, match="outside the repository"):
        SEMANTIC.pilot15_export(destination, semantic_paths)
    assert not destination.exists()


def test_pilot15_export_rejects_every_canonical_artifact(semantic_paths) -> None:
    before = tuple(path.read_bytes() for path in semantic_paths)
    for path in semantic_paths:
        with pytest.raises(SEMANTIC.SemanticAnnotationError):
            SEMANTIC.pilot15_export(path, semantic_paths)
    assert tuple(path.read_bytes() for path in semantic_paths) == before


def test_pilot15_export_rejects_external_symlink_to_canonical_artifact(
    semantic_paths, tmp_path: Path
) -> None:
    link = tmp_path / "registry-link.csv"
    try:
        link.symlink_to(semantic_paths.registry)
    except OSError:
        pytest.skip("file symlink creation is unavailable on this Windows account")
    before = semantic_paths.registry.read_bytes()
    with pytest.raises(SEMANTIC.SemanticAnnotationError):
        SEMANTIC.pilot15_export(link, semantic_paths)
    assert semantic_paths.registry.read_bytes() == before


def test_valid_kis_pass1_exact_state_and_source_reference(semantic_paths, tmp_path: Path) -> None:
    result = _record_current(semantic_paths, tmp_path)
    assert result == {
        "status": "PASS1_RECORDED",
        "query_id": "q1b-kis-015",
        "assignment_rank": 1,
        "slot_id": "KIS-015",
        "task": "kis",
    }
    state = SEMANTIC.load_semantic_state(semantic_paths)
    query = state.benchmark.queries[0]
    registry = state.registry[0]
    assert query.query_id == "q1b-kis-015"
    assert query.ground_truth.start_frame_id == 10
    assert query.ground_truth.end_frame_id == 20
    assert query.source_reference == "raw_video:L26_V065;reviewed_frames:10-20"
    assert query.annotation_status.value == "draft"
    assert query.label_origin.value == "human_raw_video"
    assert query.tags == ("human_visible", "temporal_boundary")
    assert registry.annotation_pass1_status == "COMPLETE"
    assert registry.annotation_pass2_status == "REVIEW_PENDING"
    assert registry.reviewer_id == ""
    assert registry.review_notes == ""
    assert registry.benchmark_included is False
    assert state.trake_reviews == ()


def test_frame_boundaries_accept_zero_and_last_original_frame(
    semantic_paths, tmp_path: Path
) -> None:
    _record_current(
        semantic_paths,
        tmp_path,
        mutate=lambda payload: payload.update(
            raw_video_frame_count=1_000, start_frame_id=0, end_frame_id=999
        ),
    )
    ground_truth = SEMANTIC.load_semantic_state(semantic_paths).benchmark.queries[0].ground_truth
    assert (ground_truth.start_frame_id, ground_truth.end_frame_id) == (0, 999)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("end_frame_id", 1_000),
        ("end_frame_id", -1),
        ("start_frame_id", 10**100),
        ("start_frame_id", 1.0),
        ("start_frame_id", "1"),
        ("raw_video_frame_count", 1.0),
        ("raw_video_frame_count", "1000"),
    ],
)
def test_frame_coordinate_type_and_upper_bound_fail_closed(
    semantic_paths, tmp_path: Path, field: str, value: object
) -> None:
    payload = _valid_payload(_current_target(semantic_paths))
    payload[field] = value
    before = _artifact_bytes(semantic_paths)
    with pytest.raises(SEMANTIC.SemanticAnnotationError):
        SEMANTIC.pass1_record(_write_json(tmp_path / "frame-invalid.json", payload), semantic_paths)
    assert _artifact_bytes(semantic_paths) == before


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("expect_slot_id", "KIS-999", "frozen KIS slot range"),
        ("expect_assignment_rank", 2, "stale"),
        ("start_frame_id", -1, "0 <="),
        ("start_frame_id", 21, "0 <="),
        ("end_frame_id", 1_000, "0 <="),
        ("raw_video_reviewed", False, "must be true"),
        ("query_authored_before_retrieval", False, "must be true"),
        ("gt_authored_before_retrieval", False, "must be true"),
        ("original_frame_coordinates_verified", False, "must be true"),
        ("raw_video_frame_count", 0, "greater than zero"),
        ("raw_video_frame_count", True, "must be an integer"),
        ("start_frame_id", True, "must be an integer"),
        ("query_vi", "   ", "non-empty"),
        ("difficulty", "trivial", "invalid difficulty"),
        ("tags", ["q1b_dev"], "split tag"),
        ("tags", ["retrieval_bad"], "model/result"),
        ("tags", ["rank_37"], "model/result"),
        ("tags", ["clip_score"], "model/result"),
        ("tags", ["rrf_rank_1"], "model/result"),
        ("tags", ["top100_result"], "model/result"),
        ("tags", ["Q1B_DEV"], "split tag"),
        ("tags", [" q1b_dev "], "outer whitespace"),
        ("tags", ["Rank_37"], "model/result"),
        ("tags", ["rank_001"], "model/result"),
        ("tags", ["retrieval-bad"], "model/result"),
        ("tags", ["model_rank_5"], "model/result"),
        ("tags", ["clip_score_high"], "model/result"),
        ("tags", ["prediction_wrong"], "model/result"),
        ("tags", ["same", "same"], "unique"),
        ("annotator_id", " A01", "outer whitespace"),
        ("annotator_id", "A01 ", "outer whitespace"),
        ("annotator_id", "A01\n", "outer whitespace|control"),
        ("query_vi", "human\x00text", "NUL"),
    ],
)
def test_kis_pass1_rejects_invalid_values_without_mutation(
    semantic_paths, tmp_path: Path, field: str, value: Any, message: str
) -> None:
    before = _artifact_bytes(semantic_paths)
    payload = _valid_payload(_current_target(semantic_paths))
    payload[field] = value
    input_path = _write_json(tmp_path / f"invalid-{field}.json", payload)
    with pytest.raises(SEMANTIC.SemanticAnnotationError, match=message):
        SEMANTIC.pass1_record(input_path, semantic_paths)
    assert _artifact_bytes(semantic_paths) == before


def test_legitimate_rank_language_tag_is_not_over_rejected(semantic_paths, tmp_path: Path) -> None:
    result = _record_current(
        semantic_paths,
        tmp_path,
        mutate=lambda payload: payload.update(tags=["first_ranked_choice"]),
    )
    assert result["status"] == "PASS1_RECORDED"


@pytest.mark.parametrize("mode", ["unknown", "missing"])
def test_kis_pass1_requires_exact_field_set(semantic_paths, tmp_path: Path, mode: str) -> None:
    payload = _valid_payload(_current_target(semantic_paths))
    if mode == "unknown":
        payload["video_id"] = "L99_V999"
    else:
        del payload["query_vi"]
    before = _artifact_bytes(semantic_paths)
    with pytest.raises(SEMANTIC.SemanticAnnotationError, match="fields mismatch"):
        SEMANTIC.pass1_record(_write_json(tmp_path / f"{mode}.json", payload), semantic_paths)
    assert _artifact_bytes(semantic_paths) == before


def test_kis_pass1_roundtrips_frozen_quality_loader(semantic_paths, tmp_path: Path) -> None:
    _record_current(semantic_paths, tmp_path)
    benchmark = SEMANTIC.load_quality_benchmark_json(semantic_paths.benchmark)
    assert len(benchmark.queries) == 1
    assert benchmark.queries[0].query_id == "q1b-kis-015"


@pytest.mark.parametrize(
    "machine_path",
    [
        r"D:\videos\L26_V065.mp4",
        "D:/dataset/video.mp4",
        r"\\server\share\video.mp4",
        "/kaggle/input/x",
        "/home/a/x",
        "/tmp/video.mp4",
    ],
)
def test_pass1_rejects_machine_local_paths_in_canonical_fields(
    semantic_paths, tmp_path: Path, machine_path: str
) -> None:
    payload = _valid_payload(_current_target(semantic_paths))
    payload["boundary_notes"] = machine_path
    before = _artifact_bytes(semantic_paths)
    with pytest.raises(SEMANTIC.SemanticAnnotationError, match="machine-local"):
        SEMANTIC.pass1_record(
            _write_json(tmp_path / "machine-path.json", payload), semantic_paths
        )
    assert _artifact_bytes(semantic_paths) == before


def test_url_like_semantic_text_is_not_misclassified_as_machine_path(
    semantic_paths, tmp_path: Path
) -> None:
    _record_current(
        semantic_paths,
        tmp_path,
        mutate=lambda payload: payload.update(
            boundary_notes="Reference URL https://example.test/home/video is portable."
        ),
    )
    assert (
        SEMANTIC.load_semantic_state(semantic_paths).registry[0].boundary_notes
        == "Reference URL https://example.test/home/video is portable."
    )


def test_valid_qa_pass1_preserves_answers_and_notes(semantic_paths, tmp_path: Path) -> None:
    _record_through_task(semantic_paths, tmp_path, "qa")
    state = SEMANTIC.load_semantic_state(semantic_paths)
    query = state.benchmark.queries[3]
    registry = state.registry[3]
    assert query.query_id == "q1b-qa-015"
    assert query.ground_truth.accepted_answers == ("đỏ", "màu đỏ")
    assert query.source_reference == "raw_video:L25_V051;reviewed_frames:30-45"
    assert registry.answer_notes == "Human-authored answer notes."
    assert state.trake_reviews == ()


def test_qa_exact_alias_rule_keeps_case_distinct_answers(
    semantic_paths, tmp_path: Path
) -> None:
    for _ in range(3):
        _record_current(semantic_paths, tmp_path)
    payload = _valid_payload(_current_target(semantic_paths))
    payload["accepted_answers"] = ["Red", "red"]
    SEMANTIC.pass1_record(
        _write_json(tmp_path / "qa-case-distinct.json", payload), semantic_paths
    )
    query = SEMANTIC.load_semantic_state(semantic_paths).benchmark.queries[3]
    assert query.ground_truth.accepted_answers == ("Red", "red")


def test_qa_aliases_preserve_case_punctuation_and_human_order(
    semantic_paths, tmp_path: Path
) -> None:
    for _ in range(3):
        _record_current(semantic_paths, tmp_path)
    answers = ["White", "white", "white!", "trắng", "Màu trắng."]
    _record_current(
        semantic_paths,
        tmp_path,
        mutate=lambda payload: payload.update(accepted_answers=answers),
    )
    query = SEMANTIC.load_semantic_state(semantic_paths).benchmark.queries[3]
    assert list(query.ground_truth.accepted_answers) == answers


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("accepted_answers", [], "non-empty"),
        ("accepted_answers", ["đỏ", "  "], "non-empty"),
        ("accepted_answers", ["đỏ", "đỏ"], "exact duplicates"),
        ("question", "", "non-empty"),
        ("event_description", " ", "non-empty"),
        ("start_frame_id", -1, "0 <="),
        ("start_frame_id", 50, "0 <="),
        ("end_frame_id", 1_000, "0 <="),
    ],
)
def test_qa_pass1_rejects_invalid_values(
    semantic_paths, tmp_path: Path, field: str, value: Any, message: str
) -> None:
    for _ in range(3):
        _record_current(semantic_paths, tmp_path)
    target = _current_target(semantic_paths)
    assert target["task"] == "qa"
    payload = _valid_payload(target)
    payload[field] = value
    before = _artifact_bytes(semantic_paths)
    with pytest.raises(SEMANTIC.SemanticAnnotationError, match=message):
        SEMANTIC.pass1_record(_write_json(tmp_path / f"qa-{field}.json", payload), semantic_paths)
    assert _artifact_bytes(semantic_paths) == before


@pytest.mark.parametrize("event_count", [2, 5])
def test_valid_trake_pass1_preserves_order_sidecar_and_source(
    semantic_paths, tmp_path: Path, event_count: int
) -> None:
    _record_through_task(semantic_paths, tmp_path, "trake", event_count=event_count)
    state = SEMANTIC.load_semantic_state(semantic_paths)
    query = state.benchmark.queries[7]
    assert len(query.events) == event_count
    assert [event.description for event in query.events] == [
        f"Sự kiện {index + 1}" for index in range(event_count)
    ]
    assert query.ground_truth.event_intervals == tuple(
        (10 + index * 20, 15 + index * 20) for index in range(event_count)
    )
    expected_windows = "|".join(
        f"{10 + index * 20}-{15 + index * 20}" for index in range(event_count)
    )
    assert query.source_reference == f"raw_video:L23_V013;event_windows:{expected_windows}"
    rows = [row for row in state.trake_reviews if row.query_id == query.query_id]
    assert [row.event_index for row in rows] == list(range(1, event_count + 1))
    assert [row.moment_definition for row in rows] == [
        f"Visible event boundary {index + 1}." for index in range(event_count)
    ]
    assert all(row.review_status == "REVIEW_PENDING" for row in rows)


@pytest.mark.parametrize("event_count", [1, 6])
def test_trake_rejects_invalid_event_count(
    semantic_paths, tmp_path: Path, event_count: int
) -> None:
    for _ in range(7):
        _record_current(semantic_paths, tmp_path)
    target = _current_target(semantic_paths)
    payload = _valid_payload(target, event_count=event_count)
    before = _artifact_bytes(semantic_paths)
    with pytest.raises(SEMANTIC.SemanticAnnotationError, match="between 2 and 5"):
        SEMANTIC.pass1_record(_write_json(tmp_path / "trake-count.json", payload), semantic_paths)
    assert _artifact_bytes(semantic_paths) == before


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda events: events[1].update(start_frame_id=5), "strictly increasing"),
        (lambda events: events[0].update(start_frame_id=-1), "0 <="),
        (lambda events: events[0].update(end_frame_id=1_000), "0 <="),
        (lambda events: events[0].update(description=""), "non-empty"),
        (lambda events: events[0].update(moment_definition=" "), "non-empty"),
        (lambda events: events[0].update(extra="forbidden"), "fields mismatch"),
    ],
)
def test_trake_rejects_invalid_events_without_sorting(
    semantic_paths,
    tmp_path: Path,
    mutation: Callable[[list[dict[str, Any]]], None],
    message: str,
) -> None:
    for _ in range(7):
        _record_current(semantic_paths, tmp_path)
    payload = _valid_payload(_current_target(semantic_paths))
    mutation(payload["events"])
    before = _artifact_bytes(semantic_paths)
    with pytest.raises(SEMANTIC.SemanticAnnotationError, match=message):
        SEMANTIC.pass1_record(_write_json(tmp_path / "trake-invalid.json", payload), semantic_paths)
    assert _artifact_bytes(semantic_paths) == before


@pytest.mark.parametrize(
    "intervals",
    [
        [(10, 30), (20, 40)],
        [(10, 10), (11, 11)],
        [(10, 50), (20, 30)],
    ],
)
def test_trake_allows_overlap_adjacency_and_nesting_when_starts_increase(
    semantic_paths, tmp_path: Path, intervals: list[tuple[int, int]]
) -> None:
    for _ in range(7):
        _record_current(semantic_paths, tmp_path)
    payload = _valid_payload(_current_target(semantic_paths), event_count=2)
    for event, (start, end) in zip(payload["events"], intervals, strict=True):
        event["start_frame_id"] = start
        event["end_frame_id"] = end
    SEMANTIC.pass1_record(_write_json(tmp_path / "trake-overlap.json", payload), semantic_paths)
    query = SEMANTIC.load_semantic_state(semantic_paths).benchmark.queries[7]
    assert list(query.ground_truth.event_intervals) == intervals


def test_trake_equal_start_is_rejected_even_when_event_order_is_not_sorted(
    semantic_paths, tmp_path: Path
) -> None:
    for _ in range(7):
        _record_current(semantic_paths, tmp_path)
    payload = _valid_payload(_current_target(semantic_paths), event_count=2)
    payload["events"][1]["start_frame_id"] = payload["events"][0]["start_frame_id"]
    with pytest.raises(SEMANTIC.SemanticAnnotationError, match="strictly increasing"):
        SEMANTIC.pass1_record(_write_json(tmp_path / "equal-start.json", payload), semantic_paths)


def test_prevalidation_benchmark_failure_mutates_nothing(
    semantic_paths, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = SEMANTIC._serialize_benchmark

    def corrupt(benchmark):
        return b"{}\n" if benchmark.queries else original(benchmark)

    monkeypatch.setattr(SEMANTIC, "_serialize_benchmark", corrupt)
    before = _artifact_bytes(semantic_paths)
    with pytest.raises(SEMANTIC.SemanticAnnotationError, match="transaction failed"):
        _record_current(semantic_paths, tmp_path)
    assert _artifact_bytes(semantic_paths) == before


def test_prevalidation_registry_failure_mutates_nothing(
    semantic_paths, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = SEMANTIC._serialize_registry

    def corrupt(records):
        materialized = tuple(records)
        return b"bad\n" if materialized else original(materialized)

    monkeypatch.setattr(SEMANTIC, "_serialize_registry", corrupt)
    before = _artifact_bytes(semantic_paths)
    with pytest.raises(SEMANTIC.SemanticAnnotationError, match="transaction failed"):
        _record_current(semantic_paths, tmp_path)
    assert _artifact_bytes(semantic_paths) == before


def test_prevalidation_trake_failure_mutates_nothing(
    semantic_paths, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for _ in range(7):
        _record_current(semantic_paths, tmp_path)
    original = SEMANTIC._serialize_trake_reviews

    def corrupt(records):
        materialized = tuple(records)
        return b"bad\n" if materialized else original(materialized)

    monkeypatch.setattr(SEMANTIC, "_serialize_trake_reviews", corrupt)
    before = _artifact_bytes(semantic_paths)
    with pytest.raises(SEMANTIC.SemanticAnnotationError, match="transaction failed"):
        _record_current(semantic_paths, tmp_path)
    assert _artifact_bytes(semantic_paths) == before


def test_injected_second_replace_failure_rolls_back_and_cleans_temps(
    semantic_paths, tmp_path: Path
) -> None:
    before = _artifact_bytes(semantic_paths)
    calls = 0

    def fail_second(source: Path, destination: Path) -> object:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected replacement failure")
        return os.replace(source, destination)

    with pytest.raises(SEMANTIC.SemanticAnnotationError, match="transaction failed"):
        _record_current(semantic_paths, tmp_path, replace_file=fail_second)
    assert _artifact_bytes(semantic_paths) == before
    leftovers = [
        path
        for path in semantic_paths.benchmark.parent.iterdir()
        if path.name.startswith(".")
    ]
    assert leftovers == []


def test_replace_then_raise_still_rolls_back_current_destination(
    semantic_paths, tmp_path: Path
) -> None:
    before = _artifact_bytes(semantic_paths)
    calls = 0

    def replace_then_fail(source: Path, destination: Path) -> object:
        nonlocal calls
        calls += 1
        result = os.replace(source, destination)
        if calls == 2:
            raise OSError("injected failure after replacement")
        return result

    with pytest.raises(SEMANTIC.SemanticAnnotationError, match="transaction failed"):
        _record_current(semantic_paths, tmp_path, replace_file=replace_then_fail)
    assert _artifact_bytes(semantic_paths) == before


@pytest.mark.parametrize("fail_call", [1, 2])
@pytest.mark.parametrize("fail_after_replace", [False, True])
def test_kis_transaction_failure_matrix_restores_exact_bytes(
    semantic_paths, tmp_path: Path, fail_call: int, fail_after_replace: bool
) -> None:
    before = _artifact_bytes(semantic_paths)
    before_hashes = tuple(hashlib.sha256(item).hexdigest() for item in before)
    calls = 0

    def injected(source: Path, destination: Path) -> object:
        nonlocal calls
        calls += 1
        if calls == fail_call and not fail_after_replace:
            raise OSError(f"before replace {fail_call}")
        result = os.replace(source, destination)
        if calls == fail_call and fail_after_replace:
            raise OSError(f"after replace {fail_call}")
        return result

    with pytest.raises(SEMANTIC.SemanticAnnotationError, match="transaction failed"):
        _record_current(semantic_paths, tmp_path, replace_file=injected)
    after = _artifact_bytes(semantic_paths)
    assert after == before
    assert tuple(hashlib.sha256(item).hexdigest() for item in after) == before_hashes


@pytest.mark.parametrize("fail_call", [1, 2, 3])
@pytest.mark.parametrize("fail_after_replace", [False, True])
def test_trake_transaction_failure_matrix_restores_exact_bytes(
    semantic_paths, tmp_path: Path, fail_call: int, fail_after_replace: bool
) -> None:
    _seed_pass1_prefix(semantic_paths, 7)
    before = _artifact_bytes(semantic_paths)
    before_hashes = tuple(hashlib.sha256(item).hexdigest() for item in before)
    calls = 0

    def injected(source: Path, destination: Path) -> object:
        nonlocal calls
        calls += 1
        if calls == fail_call and not fail_after_replace:
            raise OSError(f"before replace {fail_call}")
        result = os.replace(source, destination)
        if calls == fail_call and fail_after_replace:
            raise OSError(f"after replace {fail_call}")
        return result

    with pytest.raises(SEMANTIC.SemanticAnnotationError, match="transaction failed"):
        _record_current(semantic_paths, tmp_path, replace_file=injected)
    after = _artifact_bytes(semantic_paths)
    assert after == before
    assert tuple(hashlib.sha256(item).hexdigest() for item in after) == before_hashes


def test_rollback_failure_is_loud_and_names_inconsistent_artifacts(
    semantic_paths, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_replace = os.replace
    forward_calls = 0

    def forward(source: Path, destination: Path) -> object:
        nonlocal forward_calls
        forward_calls += 1
        result = real_replace(source, destination)
        if forward_calls == 2:
            raise OSError("forward failure after registry replace")
        return result

    def rollback_failure(source: Path, destination: Path) -> object:
        if source.name.endswith(".rollback"):
            raise OSError("injected rollback denial")
        return real_replace(source, destination)

    monkeypatch.setattr(SEMANTIC.os, "replace", rollback_failure)
    with pytest.raises(SEMANTIC.SemanticAnnotationError, match="rollback was incomplete") as caught:
        _record_current(semantic_paths, tmp_path, replace_file=forward)
    assert "benchmark.draft.json" in str(caught.value)
    assert "annotation_registry.csv" in str(caught.value)


def test_existing_writer_lock_rejects_lost_update_without_mutation(
    semantic_paths, tmp_path: Path
) -> None:
    lock_path = SEMANTIC._writer_lock_path(semantic_paths)
    lock_path.write_text("pid=999999\n", encoding="ascii")
    before = _artifact_bytes(semantic_paths)
    with pytest.raises(SEMANTIC.SemanticAnnotationError, match="writer lock already exists"):
        _record_current(semantic_paths, tmp_path)
    assert _artifact_bytes(semantic_paths) == before
    assert lock_path.exists()


def test_writer_lock_is_unique_and_removed_after_success(semantic_paths, tmp_path: Path) -> None:
    lock_path = SEMANTIC._writer_lock_path(semantic_paths)
    _record_current(semantic_paths, tmp_path)
    assert not lock_path.exists()


def test_crash_leftover_temp_and_rollback_files_are_never_loaded(
    semantic_paths,
) -> None:
    parent = semantic_paths.benchmark.parent
    (parent / ".benchmark.draft.json.dead.tmp").write_text("not-json", encoding="utf-8")
    (parent / ".annotation_registry.csv.dead.rollback").write_text(
        "not,csv", encoding="utf-8"
    )
    assert SEMANTIC.audit(semantic_paths)["valid"] is True


def test_equivalent_pass1_writes_are_byte_deterministic(tmp_path: Path) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    shutil.copytree(BENCHMARK_DIR, first_root)
    shutil.copytree(BENCHMARK_DIR, second_root)

    def paths(root: Path):
        return SEMANTIC.SemanticPaths(
            root / "candidate_video_manifest.csv",
            root / "annotation_plan.csv",
            root / "category_codebook.csv",
            root / "slot_assignment_manifest.csv",
            root / "candidate_review_log.csv",
            root / "benchmark.draft.json",
            root / "annotation_registry.csv",
            root / "trake_event_review.csv",
        )

    first_paths = paths(first_root)
    second_paths = paths(second_root)
    _record_current(first_paths, tmp_path)
    _record_current(second_paths, tmp_path)
    assert _artifact_bytes(first_paths) == _artifact_bytes(second_paths)


def test_pass2_next_selects_earliest_pending_and_is_split_blind(
    semantic_paths, tmp_path: Path
) -> None:
    _record_current(semantic_paths, tmp_path)
    _record_current(semantic_paths, tmp_path)
    target = SEMANTIC.pass2_next(semantic_paths)
    assert target["assignment_rank"] == 1
    assert target["query_id"] == "q1b-kis-015"
    assert "planned_split" not in json.dumps(target).casefold()


def test_pass2_rejects_same_reviewer_as_annotator(semantic_paths, tmp_path: Path) -> None:
    _record_current(semantic_paths, tmp_path)
    with pytest.raises(SEMANTIC.SemanticAnnotationError, match="must differ"):
        _record_pass2(
            semantic_paths,
            tmp_path,
            mutate=lambda payload: payload.update(reviewer_id="annotator-01"),
        )


@pytest.mark.parametrize("reviewer", [" reviewer-02", "reviewer-02 ", "reviewer-02\n"])
def test_pass2_rejects_padded_or_control_reviewer_ids(
    semantic_paths, tmp_path: Path, reviewer: str
) -> None:
    _record_current(semantic_paths, tmp_path)
    with pytest.raises(SEMANTIC.SemanticAnnotationError, match="outer whitespace|control"):
        _record_pass2(
            semantic_paths,
            tmp_path,
            mutate=lambda payload: payload.update(reviewer_id=reviewer),
        )


def test_reviewer_identity_is_case_sensitive_exact_string(
    semantic_paths, tmp_path: Path
) -> None:
    _record_current(
        semantic_paths,
        tmp_path,
        mutate=lambda payload: payload.update(annotator_id="A01"),
    )
    result = _record_pass2(
        semantic_paths,
        tmp_path,
        mutate=lambda payload: payload.update(reviewer_id="a01"),
    )
    assert result["status"] == "VERIFIED"


def test_revision_required_allows_semantic_disagreement_confirmations(
    semantic_paths, tmp_path: Path
) -> None:
    _record_current(semantic_paths, tmp_path)
    result = _record_pass2(
        semantic_paths,
        tmp_path,
        decision="REVISION_REQUIRED",
        mutate=lambda payload: payload.update(
            semantic_support_verified=False,
            video_id_verified=False,
            original_frame_coordinates_verified=False,
            intervals_verified=False,
        ),
    )
    assert result["status"] == "REVISION_REQUIRED"


@pytest.mark.parametrize("extra", ["answers_verified", "event_order_verified"])
def test_kis_pass2_rejects_irrelevant_task_confirmation(
    semantic_paths, tmp_path: Path, extra: str
) -> None:
    _record_current(semantic_paths, tmp_path)
    with pytest.raises(SEMANTIC.SemanticAnnotationError, match="fields mismatch"):
        _record_pass2(
            semantic_paths,
            tmp_path,
            mutate=lambda payload: payload.update({extra: True}),
        )


@pytest.mark.parametrize(
    "confirmation",
    [
        "raw_video_reviewed",
        "semantic_support_verified",
        "video_id_verified",
        "original_frame_coordinates_verified",
        "intervals_verified",
    ],
)
def test_kis_verified_requires_all_confirmations(
    semantic_paths, tmp_path: Path, confirmation: str
) -> None:
    _record_current(semantic_paths, tmp_path)
    with pytest.raises(SEMANTIC.SemanticAnnotationError):
        _record_pass2(
            semantic_paths,
            tmp_path,
            mutate=lambda payload: payload.update({confirmation: False}),
        )


def test_kis_verified_transition_preserves_semantics(semantic_paths, tmp_path: Path) -> None:
    _record_current(semantic_paths, tmp_path)
    before_query = SEMANTIC.load_semantic_state(semantic_paths).benchmark.queries[0]
    result = _record_pass2(semantic_paths, tmp_path)
    state = SEMANTIC.load_semantic_state(semantic_paths)
    after_query = state.benchmark.queries[0]
    assert result["status"] == "VERIFIED"
    assert after_query.annotation_status.value == "verified"
    assert after_query.label_origin.value == "human_raw_video"
    assert after_query.query_vi == before_query.query_vi
    assert after_query.ground_truth == before_query.ground_truth
    assert after_query.tags == before_query.tags
    assert state.registry[0].benchmark_included is True


def test_pass2_changes_only_workflow_metadata_and_preserves_benchmark_identity(
    semantic_paths, tmp_path: Path
) -> None:
    _record_current(semantic_paths, tmp_path)
    before_state = SEMANTIC.load_semantic_state(semantic_paths)
    before_query_payload = SEMANTIC._query_to_payload(before_state.benchmark.queries[0])
    _record_pass2(semantic_paths, tmp_path)
    after_state = SEMANTIC.load_semantic_state(semantic_paths)
    after_query_payload = SEMANTIC._query_to_payload(after_state.benchmark.queries[0])
    assert before_state.benchmark.schema_version == after_state.benchmark.schema_version == 1
    assert before_state.benchmark.benchmark_id == after_state.benchmark.benchmark_id
    assert before_state.benchmark.description == after_state.benchmark.description
    before_query_payload.pop("annotation_status")
    after_query_payload.pop("annotation_status")
    assert after_query_payload == before_query_payload


def test_pass2_one_query_does_not_mutate_other_semantic_records(
    semantic_paths, tmp_path: Path
) -> None:
    _record_current(semantic_paths, tmp_path)
    _record_current(semantic_paths, tmp_path)
    before = SEMANTIC.load_semantic_state(semantic_paths)
    second_query = SEMANTIC._query_to_payload(before.benchmark.queries[1])
    second_registry = before.registry[1]
    _record_pass2(semantic_paths, tmp_path)
    after = SEMANTIC.load_semantic_state(semantic_paths)
    assert SEMANTIC._query_to_payload(after.benchmark.queries[1]) == second_query
    assert after.registry[1] == second_registry


def test_qa_verified_requires_answers_confirmation(semantic_paths, tmp_path: Path) -> None:
    _record_through_task(semantic_paths, tmp_path, "qa")
    _advance_pass2_to_task(semantic_paths, tmp_path, "qa")
    with pytest.raises(SEMANTIC.SemanticAnnotationError, match="every relevant"):
        _record_pass2(
            semantic_paths,
            tmp_path,
            mutate=lambda payload: payload.update(answers_verified=False),
        )
    _record_pass2(semantic_paths, tmp_path)
    state = SEMANTIC.load_semantic_state(semantic_paths)
    assert state.registry[3].annotation_pass2_status == "VERIFIED"
    assert state.benchmark.queries[3].annotation_status.value == "verified"


def test_trake_verified_requires_order_and_updates_sidecar(
    semantic_paths, tmp_path: Path
) -> None:
    _record_through_task(semantic_paths, tmp_path, "trake")
    _advance_pass2_to_task(semantic_paths, tmp_path, "trake")
    with pytest.raises(SEMANTIC.SemanticAnnotationError, match="every relevant"):
        _record_pass2(
            semantic_paths,
            tmp_path,
            mutate=lambda payload: payload.update(event_order_verified=False),
        )
    _record_pass2(semantic_paths, tmp_path)
    state = SEMANTIC.load_semantic_state(semantic_paths)
    rows = [row for row in state.trake_reviews if row.query_id == "q1b-trake-001"]
    assert rows
    assert all(row.review_status == "VERIFIED" for row in rows)
    assert all(row.reviewer_id == "reviewer-02" for row in rows)


def test_revision_required_transition_and_queue(semantic_paths, tmp_path: Path) -> None:
    _record_current(semantic_paths, tmp_path)
    result = _record_pass2(semantic_paths, tmp_path, decision="REVISION_REQUIRED")
    assert result["status"] == "REVISION_REQUIRED"
    state = SEMANTIC.load_semantic_state(semantic_paths)
    assert state.benchmark.queries[0].annotation_status.value == "draft"
    assert state.registry[0].annotation_pass2_status == "REVISION_REQUIRED"
    assert state.registry[0].benchmark_included is False
    target = SEMANTIC.revision_next(semantic_paths)
    assert target["query_id"] == "q1b-kis-015"
    assert target["reviewer_id"] == "reviewer-02"
    assert target["review_notes"] == "Revise the boundary."


def test_revision_required_needs_notes(semantic_paths, tmp_path: Path) -> None:
    _record_current(semantic_paths, tmp_path)
    with pytest.raises(SEMANTIC.SemanticAnnotationError, match="non-empty"):
        _record_pass2(
            semantic_paths,
            tmp_path,
            decision="REVISION_REQUIRED",
            mutate=lambda payload: payload.update(review_notes=""),
        )


def test_pass1_revise_rejected_without_revision_state(semantic_paths, tmp_path: Path) -> None:
    _record_current(semantic_paths, tmp_path)
    state = SEMANTIC.load_semantic_state(semantic_paths)
    payload = _valid_payload(SEMANTIC._target_output(state.targets[0]))
    with pytest.raises(SEMANTIC.SemanticAnnotationError, match="no revision-required"):
        input_path = _write_json(tmp_path / "revise-too-soon.json", payload)
        SEMANTIC.pass1_revise(input_path, semantic_paths)


def test_illegal_state_transitions_are_rejected_without_mutation(
    semantic_paths, tmp_path: Path
) -> None:
    fake_pass2 = {
        "query_id": "q1b-kis-015",
        "reviewer_id": "R01",
        "decision": "VERIFIED",
        "raw_video_reviewed": True,
        "semantic_support_verified": True,
        "video_id_verified": True,
        "original_frame_coordinates_verified": True,
        "intervals_verified": True,
        "review_notes": "",
    }
    with pytest.raises(SEMANTIC.SemanticAnnotationError, match="no Pass-2"):
        SEMANTIC.pass2_record(_write_json(tmp_path / "s0-pass2.json", fake_pass2), semantic_paths)

    _record_current(semantic_paths, tmp_path)
    duplicate_payload = _valid_payload(SEMANTIC._target_output(
        SEMANTIC.load_semantic_state(semantic_paths).targets[0]
    ))
    with pytest.raises(SEMANTIC.SemanticAnnotationError, match="stale"):
        SEMANTIC.pass1_record(
            _write_json(tmp_path / "duplicate-pass1.json", duplicate_payload), semantic_paths
        )

    _record_pass2(semantic_paths, tmp_path)
    verified_bytes = _artifact_bytes(semantic_paths)
    with pytest.raises(SEMANTIC.SemanticAnnotationError, match="no revision-required"):
        SEMANTIC.pass1_revise(
            _write_json(tmp_path / "verified-revise.json", duplicate_payload), semantic_paths
        )
    with pytest.raises(SEMANTIC.SemanticAnnotationError, match="no Pass-2"):
        SEMANTIC.pass2_record(
            _write_json(tmp_path / "verified-pass2.json", fake_pass2), semantic_paths
        )
    assert _artifact_bytes(semantic_paths) == verified_bytes


def test_revision_required_cannot_be_reviewed_again_before_revision(
    semantic_paths, tmp_path: Path
) -> None:
    _record_current(semantic_paths, tmp_path)
    _record_pass2(semantic_paths, tmp_path, decision="REVISION_REQUIRED")
    before = _artifact_bytes(semantic_paths)
    payload = _pass2_payload(SEMANTIC.revision_next(semantic_paths), "VERIFIED")
    with pytest.raises(SEMANTIC.SemanticAnnotationError, match="no Pass-2"):
        SEMANTIC.pass2_record(_write_json(tmp_path / "review-again.json", payload), semantic_paths)
    assert _artifact_bytes(semantic_paths) == before


def test_successful_kis_revision_resets_review_fields_and_keeps_identity(
    semantic_paths, tmp_path: Path
) -> None:
    _record_current(semantic_paths, tmp_path)
    _record_pass2(semantic_paths, tmp_path, decision="REVISION_REQUIRED")
    target = SEMANTIC.revision_next(semantic_paths)
    payload = _valid_payload(
        SEMANTIC._target_output(SEMANTIC.load_semantic_state(semantic_paths).targets[0])
    )
    payload["query_vi"] = "Nội dung đã sửa bởi người gán nhãn."
    result = SEMANTIC.pass1_revise(
        _write_json(tmp_path / "kis-revision.json", payload), semantic_paths
    )
    state = SEMANTIC.load_semantic_state(semantic_paths)
    assert result["query_id"] == target["query_id"] == "q1b-kis-015"
    assert state.benchmark.queries[0].query_vi == "Nội dung đã sửa bởi người gán nhãn."
    assert state.benchmark.queries[0].annotation_status.value == "draft"
    assert state.registry[0].annotation_pass2_status == "REVIEW_PENDING"
    assert state.registry[0].reviewer_id == ""
    assert state.registry[0].review_notes == ""
    assert state.registry[0].benchmark_included is False


def test_trake_revision_replaces_sidecar_instead_of_appending(
    semantic_paths, tmp_path: Path
) -> None:
    _record_through_task(semantic_paths, tmp_path, "trake", event_count=2)
    _advance_pass2_to_task(semantic_paths, tmp_path, "trake")
    _record_pass2(semantic_paths, tmp_path, decision="REVISION_REQUIRED")
    state = SEMANTIC.load_semantic_state(semantic_paths)
    target = state.targets[7]
    payload = _valid_payload(SEMANTIC._target_output(target), event_count=3)
    payload["events"][0]["description"] = "Sự kiện đã sửa"
    SEMANTIC.pass1_revise(
        _write_json(tmp_path / "trake-revision.json", payload), semantic_paths
    )
    revised = SEMANTIC.load_semantic_state(semantic_paths)
    rows = [row for row in revised.trake_reviews if row.query_id == "q1b-trake-001"]
    assert len(rows) == 3
    assert [row.event_index for row in rows] == [1, 2, 3]
    assert rows[0].event_description == "Sự kiện đã sửa"
    assert all(row.review_status == "REVIEW_PENDING" for row in rows)
    assert all(row.reviewer_id == "" for row in rows)


@pytest.mark.parametrize(("original_count", "revised_count"), [(5, 2), (2, 5)])
def test_trake_revision_shrink_and_grow_remove_all_old_sidecar_rows(
    semantic_paths, tmp_path: Path, original_count: int, revised_count: int
) -> None:
    _record_through_task(semantic_paths, tmp_path, "trake", event_count=original_count)
    _advance_pass2_to_task(semantic_paths, tmp_path, "trake")
    _record_pass2(semantic_paths, tmp_path, decision="REVISION_REQUIRED")
    target = SEMANTIC.load_semantic_state(semantic_paths).targets[7]
    payload = _valid_payload(SEMANTIC._target_output(target), event_count=revised_count)
    SEMANTIC.pass1_revise(_write_json(tmp_path / "resize-events.json", payload), semantic_paths)
    state = SEMANTIC.load_semantic_state(semantic_paths)
    rows = [row for row in state.trake_reviews if row.query_id == "q1b-trake-001"]
    assert [row.event_index for row in rows] == list(range(1, revised_count + 1))
    assert all(row.review_status == "REVIEW_PENDING" and row.reviewer_id == "" for row in rows)


def _mutate_registry(paths, transform: Callable[[list[list[str]]], None]) -> None:
    rows = _csv_rows(paths.registry)
    transform(rows)
    _write_csv_rows(paths.registry, rows)


def _mutate_benchmark(paths, transform: Callable[[dict[str, Any]], None]) -> None:
    payload = _benchmark_payload(paths.benchmark)
    transform(payload)
    _write_json(paths.benchmark, payload)


def _mutate_trake(paths, transform: Callable[[list[list[str]]], None]) -> None:
    rows = _csv_rows(paths.trake_review)
    transform(rows)
    _write_csv_rows(paths.trake_review, rows)


@pytest.mark.parametrize(
    "corruption",
    [
        "benchmark_without_registry",
        "registry_without_benchmark",
        "wrong_video",
        "wrong_task",
        "wrong_slot",
        "wrong_query_id",
        "duplicate_registry_query",
        "duplicate_registry_slot",
        "invalid_source_reference",
        "kis_with_trake_row",
        "draft_included",
        "split_leakage",
        "duplicate_benchmark_query",
        "invalid_pass1_status",
        "invalid_pass2_status",
        "invalid_label_origin",
        "pending_reviewer",
    ],
)
def test_cross_artifact_audit_fails_closed_for_kis_corruptions(
    semantic_paths, tmp_path: Path, corruption: str
) -> None:
    _record_current(semantic_paths, tmp_path)
    if corruption == "benchmark_without_registry":
        _write_csv_rows(semantic_paths.registry, [_csv_rows(semantic_paths.registry)[0]])
    elif corruption == "registry_without_benchmark":
        _mutate_benchmark(semantic_paths, lambda payload: payload.update(queries=[]))
    elif corruption == "wrong_video":
        _mutate_registry(semantic_paths, lambda rows: rows[1].__setitem__(3, "L99_V999"))
    elif corruption == "wrong_task":
        _mutate_registry(semantic_paths, lambda rows: rows[1].__setitem__(2, "qa"))
    elif corruption == "wrong_slot":
        _mutate_registry(semantic_paths, lambda rows: rows[1].__setitem__(1, "KIS-017"))
    elif corruption == "wrong_query_id":
        _mutate_registry(semantic_paths, lambda rows: rows[1].__setitem__(0, "q1b-kis-999"))
    elif corruption == "duplicate_registry_query":
        _mutate_registry(semantic_paths, lambda rows: rows.append(rows[1].copy()))
    elif corruption == "duplicate_registry_slot":
        row = _csv_rows(semantic_paths.registry)[1].copy()
        row[0] = "q1b-kis-017"
        _mutate_registry(semantic_paths, lambda rows: rows.append(row))
    elif corruption == "invalid_source_reference":
        _mutate_benchmark(
            semantic_paths,
            lambda payload: payload["queries"][0].update(source_reference="D:\\video.mp4"),
        )
    elif corruption == "kis_with_trake_row":
        row = [
            "q1b-kis-015",
            "1",
            "event",
            "moment",
            "10",
            "20",
            "annotator-01",
            "",
            "REVIEW_PENDING",
            "",
        ]
        _mutate_trake(semantic_paths, lambda rows: rows.append(row))
    elif corruption == "draft_included":
        _mutate_registry(semantic_paths, lambda rows: rows[1].__setitem__(16, "true"))
    elif corruption == "split_leakage":
        _mutate_registry(semantic_paths, lambda rows: rows[1].__setitem__(13, "q1b_dev"))
    elif corruption == "duplicate_benchmark_query":
        _mutate_benchmark(
            semantic_paths,
            lambda payload: payload["queries"].append(payload["queries"][0].copy()),
        )
    elif corruption == "invalid_pass1_status":
        _mutate_registry(semantic_paths, lambda rows: rows[1].__setitem__(8, "BROKEN"))
    elif corruption == "invalid_pass2_status":
        _mutate_registry(semantic_paths, lambda rows: rows[1].__setitem__(9, "BROKEN"))
    elif corruption == "invalid_label_origin":
        _mutate_benchmark(
            semantic_paths,
            lambda payload: payload["queries"][0].update(label_origin="synthetic"),
        )
    else:
        _mutate_registry(
            semantic_paths, lambda rows: rows[1].__setitem__(11, "unexpected-reviewer")
        )
    with pytest.raises(SEMANTIC.SemanticAnnotationError):
        SEMANTIC.audit(semantic_paths)


@pytest.mark.parametrize(
    "corruption",
    [
        "missing_sidecar",
        "extra_sidecar",
        "wrong_sidecar_order",
        "wrong_event_description",
        "wrong_event_interval",
        "wrong_sidecar_status",
        "orphan_sidecar",
    ],
)
def test_cross_artifact_audit_fails_closed_for_trake_corruptions(
    semantic_paths, tmp_path: Path, corruption: str
) -> None:
    _record_through_task(semantic_paths, tmp_path, "trake", event_count=2)
    rows = _csv_rows(semantic_paths.trake_review)
    if corruption == "missing_sidecar":
        rows.pop()
    elif corruption == "extra_sidecar":
        extra = rows[-1].copy()
        extra[1] = "3"
        rows.append(extra)
    elif corruption == "wrong_sidecar_order":
        rows[-2], rows[-1] = rows[-1], rows[-2]
    elif corruption == "wrong_event_description":
        rows[-1][2] = "different"
    elif corruption == "wrong_event_interval":
        rows[-1][4] = "999"
    elif corruption == "wrong_sidecar_status":
        rows[-1][8] = "VERIFIED"
    else:
        extra = rows[-1].copy()
        extra[0] = "q1b-trake-orphan"
        rows.append(extra)
    _write_csv_rows(semantic_paths.trake_review, rows)
    with pytest.raises(SEMANTIC.SemanticAnnotationError):
        SEMANTIC.audit(semantic_paths)


def test_audit_rejects_verified_query_marked_pending(semantic_paths, tmp_path: Path) -> None:
    _record_current(semantic_paths, tmp_path)
    _record_pass2(semantic_paths, tmp_path)
    _mutate_registry(semantic_paths, lambda rows: rows[1].__setitem__(9, "REVIEW_PENDING"))
    with pytest.raises(SEMANTIC.SemanticAnnotationError):
        SEMANTIC.audit(semantic_paths)


def test_audit_rejects_verified_query_not_included(semantic_paths, tmp_path: Path) -> None:
    _record_current(semantic_paths, tmp_path)
    _record_pass2(semantic_paths, tmp_path)
    _mutate_registry(semantic_paths, lambda rows: rows[1].__setitem__(16, "false"))
    with pytest.raises(SEMANTIC.SemanticAnnotationError):
        SEMANTIC.audit(semantic_paths)


def test_audit_rejects_noncanonical_semantic_physical_order(
    semantic_paths, tmp_path: Path
) -> None:
    _record_current(semantic_paths, tmp_path)
    _record_current(semantic_paths, tmp_path)
    benchmark = _benchmark_payload(semantic_paths.benchmark)
    benchmark["queries"].reverse()
    _write_json(semantic_paths.benchmark, benchmark)
    registry_rows = _csv_rows(semantic_paths.registry)
    registry_rows[1], registry_rows[2] = registry_rows[2], registry_rows[1]
    _write_csv_rows(semantic_paths.registry, registry_rows)
    with pytest.raises(SEMANTIC.SemanticAnnotationError, match="assignment-rank prefix"):
        SEMANTIC.audit(semantic_paths)


def test_audit_success_summary_after_valid_records(semantic_paths, tmp_path: Path) -> None:
    _record_current(semantic_paths, tmp_path)
    summary = SEMANTIC.audit(semantic_paths)
    assert summary["status"] == "AUDIT_PASSED"
    assert summary["valid"] is True
    assert summary["benchmark_query_count"] == 1
    assert summary["registry_row_count"] == 1


def test_fixed_seed_randomized_single_invariant_corruptions_all_fail_closed(
    semantic_paths, tmp_path: Path
) -> None:
    _seed_pass1_prefix(semantic_paths, 8)
    baseline = _artifact_bytes(semantic_paths)
    cases = [
        "query_id",
        "slot",
        "video",
        "task",
        "status",
        "included",
        "source_reference",
        "event_count",
        "event_order",
    ]
    random.Random(20260810).shuffle(cases)
    observed: set[str] = set()
    for case in cases:
        semantic_paths.benchmark.write_bytes(baseline[0])
        semantic_paths.registry.write_bytes(baseline[1])
        semantic_paths.trake_review.write_bytes(baseline[2])
        if case == "query_id":
            _mutate_registry(semantic_paths, lambda rows: rows[1].__setitem__(0, "q1b-kis-999"))
        elif case == "slot":
            _mutate_registry(semantic_paths, lambda rows: rows[1].__setitem__(1, "KIS-017"))
        elif case == "video":
            _mutate_registry(semantic_paths, lambda rows: rows[1].__setitem__(3, "L99_V999"))
        elif case == "task":
            _mutate_registry(semantic_paths, lambda rows: rows[1].__setitem__(2, "qa"))
        elif case == "status":
            _mutate_registry(semantic_paths, lambda rows: rows[1].__setitem__(9, "BROKEN"))
        elif case == "included":
            _mutate_registry(semantic_paths, lambda rows: rows[1].__setitem__(16, "true"))
        elif case == "source_reference":
            _mutate_benchmark(
                semantic_paths,
                lambda payload: payload["queries"][0].update(
                    source_reference="raw_video:L26_V065;reviewed_frames:11-20"
                ),
            )
        elif case == "event_count":
            rows = _csv_rows(semantic_paths.trake_review)
            rows.pop()
            _write_csv_rows(semantic_paths.trake_review, rows)
        else:
            rows = _csv_rows(semantic_paths.trake_review)
            rows[-2], rows[-1] = rows[-1], rows[-2]
            _write_csv_rows(semantic_paths.trake_review, rows)
        with pytest.raises(SEMANTIC.SemanticAnnotationError):
            SEMANTIC.audit(semantic_paths)
        observed.add(case)
    assert observed == set(cases)


def test_exact_vietnamese_unicode_json_roundtrip_without_normalization(
    semantic_paths, tmp_path: Path
) -> None:
    text = "ĐỒNG BẰNG SÔNG HỒNG – tiếng Việt dựng sẵn"
    _record_current(
        semantic_paths,
        tmp_path,
        mutate=lambda payload: payload.update(query_vi=text, semantic_definition=text),
    )
    state = SEMANTIC.load_semantic_state(semantic_paths)
    assert state.benchmark.queries[0].query_vi == text
    assert state.registry[0].semantic_definition == text
    assert text.encode("utf-8") in semantic_paths.benchmark.read_bytes()
    assert b"\\u0110" not in semantic_paths.benchmark.read_bytes()


def test_pass1_and_pass2_empty_queue_statuses_after_all_targets(
    semantic_paths, tmp_path: Path
) -> None:
    for _ in range(15):
        _record_current(semantic_paths, tmp_path)
    assert SEMANTIC.pass1_next(semantic_paths) == {
        "status": "PASS1_COMPLETE_FOR_ASSIGNED_TARGETS"
    }
    for _ in range(15):
        _record_pass2(semantic_paths, tmp_path)
    assert SEMANTIC.pass2_next(semantic_paths) == {"status": "PASS2_QUEUE_EMPTY"}
    assert SEMANTIC.revision_next(semantic_paths) == {"status": "REVISION_QUEUE_EMPTY"}


def test_semantic_assignment_rank_is_independent_of_review_and_sample_rank(
    semantic_paths,
) -> None:
    _write_suitability_pattern(
        semantic_paths,
        ["SKIP_NO_SUITABLE_EVENT", "ASSIGN", "SKIP_TECHNICAL_UNREADABLE", "ASSIGN"],
    )
    targets = SEMANTIC.load_semantic_state(semantic_paths).targets
    assert [target.assignment_rank for target in targets] == [1, 2]
    assert [target.review_sequence for target in targets] == [2, 4]
    assert targets[0].video_id != "L26_V065"
    assert SEMANTIC.pass1_next(semantic_paths)["review_sequence"] == 2


def test_no_suitability_assignments_returns_explicit_terminal_objects(semantic_paths) -> None:
    _write_suitability_pattern(
        semantic_paths,
        ["SKIP_NO_SUITABLE_EVENT", "SKIP_TECHNICAL_UNREADABLE"],
    )
    assert SEMANTIC.pass1_next(semantic_paths) == {
        "status": "PASS1_COMPLETE_FOR_ASSIGNED_TARGETS"
    }
    assert SEMANTIC.pass2_next(semantic_paths) == {"status": "PASS2_QUEUE_EMPTY"}
    assert SEMANTIC.revision_next(semantic_paths) == {"status": "REVISION_QUEUE_EMPTY"}
    assert SEMANTIC.semantic_status(semantic_paths)["suitability_assign_count"] == 0


def test_pilot15_exports_first_fifteen_assignments_not_first_review_rows(
    semantic_paths, tmp_path: Path
) -> None:
    decisions = [
        "SKIP_NO_SUITABLE_EVENT" if index in {1, 4, 9} else "ASSIGN"
        for index in range(1, 19)
    ]
    _write_suitability_pattern(semantic_paths, decisions)
    output = tmp_path / "pilot-with-skips.json"
    SEMANTIC.pilot15_export(output, semantic_paths)
    packet = json.loads(output.read_text(encoding="utf-8"))
    assert len(packet["targets"]) == 15
    assert [item["assignment_rank"] for item in packet["targets"]] == list(range(1, 16))
    assert [item["review_sequence"] for item in packet["targets"]] == [
        index for index in range(1, 19) if index not in {1, 4, 9}
    ]


def test_full_sixty_slot_scale_and_terminal_pass1_states(semantic_paths) -> None:
    _write_suitability_pattern(semantic_paths, ["ASSIGN"] * 60)
    assert SEMANTIC.semantic_status(semantic_paths)["suitability_assign_count"] == 60
    _seed_pass1_prefix(semantic_paths, 59)
    target = SEMANTIC.pass1_next(semantic_paths)
    assert target["assignment_rank"] == 60
    _seed_pass1_prefix(semantic_paths, 60)
    assert SEMANTIC.pass1_next(semantic_paths) == {
        "status": "PASS1_COMPLETE_FOR_ASSIGNED_TARGETS"
    }


def test_sixty_assignments_with_skips_and_more_reviews_than_slots(semantic_paths) -> None:
    decisions: list[str] = []
    for assignment in range(60):
        if assignment % 7 == 0:
            decisions.append("SKIP_NO_SUITABLE_EVENT")
        if assignment % 11 == 0:
            decisions.append("SKIP_TECHNICAL_UNREADABLE")
        decisions.append("ASSIGN")
    _write_suitability_pattern(semantic_paths, decisions)
    state = SEMANTIC.load_semantic_state(semantic_paths)
    assert len(state.targets) == 60
    assert state.targets[-1].assignment_rank == 60
    assert state.targets[-1].review_sequence == len(decisions)
    assert [target.assignment_rank for target in state.targets] == list(range(1, 61))


def test_full_sixty_slot_pass2_completion_and_revision_selection(semantic_paths) -> None:
    _write_suitability_pattern(semantic_paths, ["ASSIGN"] * 60)
    _seed_pass1_prefix(semantic_paths, 60)
    _set_pass2_states(semantic_paths, ["VERIFIED"] * 59 + ["REVISION_REQUIRED"])
    revision = SEMANTIC.revision_next(semantic_paths)
    assert revision["assignment_rank"] == 60
    assert SEMANTIC.pass2_next(semantic_paths) == {"status": "PASS2_QUEUE_EMPTY"}
    _set_pass2_states(semantic_paths, ["VERIFIED"] * 60)
    assert SEMANTIC.pass2_next(semantic_paths) == {"status": "PASS2_QUEUE_EMPTY"}
    assert SEMANTIC.revision_next(semantic_paths) == {"status": "REVISION_QUEUE_EMPTY"}
    status = SEMANTIC.semantic_status(semantic_paths)
    assert status["verified_count"] == status["benchmark_included_count"] == 60


def test_mixed_status_counts_are_derived_without_double_counting(
    semantic_paths, tmp_path: Path
) -> None:
    for _ in range(4):
        _record_current(semantic_paths, tmp_path)
    _record_pass2(semantic_paths, tmp_path)
    _record_pass2(semantic_paths, tmp_path, decision="REVISION_REQUIRED")
    status = SEMANTIC.semantic_status(semantic_paths)
    assert status["pass1_complete_count"] == 4
    assert status["verified_count"] == 1
    assert status["revision_required_count"] == 1
    assert status["pass2_pending_count"] == 2
    assert status["benchmark_included_count"] == 1
    assert sum(status["task_counts"].values()) == 4


def test_normal_outputs_never_expose_split_identifiers(semantic_paths, tmp_path: Path) -> None:
    _record_current(semantic_paths, tmp_path)
    outputs = [
        SEMANTIC.pass1_next(semantic_paths),
        SEMANTIC.pass2_next(semantic_paths),
        SEMANTIC.revision_next(semantic_paths),
        SEMANTIC.semantic_status(semantic_paths),
        SEMANTIC.build_template("KIS-015", semantic_paths),
    ]
    text = json.dumps(outputs, ensure_ascii=False).casefold()
    for marker in ("planned_split", "q1b_dev", "q1b_holdout", "development", "holdout"):
        assert marker not in text


def test_real_suitability_and_semantic_queues_are_distinct() -> None:
    paths = SEMANTIC.default_paths()
    semantic_state = SEMANTIC.load_semantic_state(paths)
    semantic_target = SEMANTIC.pass1_next(paths)
    queue = SEMANTIC._load_queue_module()
    candidates, slots, reviews, codebook = queue.load_queue(
        SEMANTIC._queue_paths(paths)
    )
    suitability_target = queue.resolve_next_target(candidates, slots, reviews, codebook)
    assert (
        suitability_target["review_sequence"],
        suitability_target["video_id"],
        suitability_target["slot_id"],
    ) == (16, "L23_V023", "KIS-011")

    recorded_query_ids = {record.query_id for record in semantic_state.registry}
    expected_target = next(
        target
        for target in semantic_state.targets
        if target.derived_query_id not in recorded_query_ids
    )
    assert (
        semantic_target["assignment_rank"],
        semantic_target["video_id"],
        semantic_target["slot_id"],
        semantic_target["derived_query_id"],
    ) == (
        expected_target.assignment_rank,
        expected_target.video_id,
        expected_target.slot_id,
        expected_target.derived_query_id,
    )


def test_new_script_has_no_forbidden_runtime_dependencies() -> None:
    source = SCRIPT_PATH.read_text(encoding="utf-8").casefold()
    forbidden = (
        "operationalkisruntime",
        "system_tai.kis",
        "system_tai.qa.pipeline",
        "system_tai.trake.pipeline",
        "system_tai.retrieval",
        "system_tai.refinement",
        "import clip",
        "import torch",
        "requests.",
        "urllib.request",
        "subprocess.",
    )
    assert [token for token in forbidden if token in source] == []


def test_ast_import_and_call_audit_forbids_runtime_network_model_and_subprocess() -> None:
    tree = ast.parse(SCRIPT_PATH.read_text(encoding="utf-8"))
    imports: set[str] = set()
    calls: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.add(node.module or "")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                calls.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                parts: list[str] = []
                current: ast.expr = node.func
                while isinstance(current, ast.Attribute):
                    parts.append(current.attr)
                    current = current.value
                if isinstance(current, ast.Name):
                    parts.append(current.id)
                calls.add(".".join(reversed(parts)))
    forbidden_import_prefixes = (
        "system_tai.kis",
        "system_tai.qa.pipeline",
        "system_tai.trake.pipeline",
        "system_tai.retrieval",
        "system_tai.refinement",
        "torch",
        "clip",
        "requests",
        "httpx",
        "socket",
        "subprocess",
        "urllib.request",
        "openai",
    )
    assert not {
        item
        for item in imports
        if any(
            item == prefix or item.startswith(prefix + ".")
            for prefix in forbidden_import_prefixes
        )
    }
    assert not {"eval", "exec", "compile", "__import__", "subprocess.run", "os.system"} & calls


def test_cli_parser_contains_complete_operation_set() -> None:
    parser = SEMANTIC.build_parser()
    subparser_action = next(
        action for action in parser._actions if action.__class__.__name__ == "_SubParsersAction"
    )
    assert set(subparser_action.choices) == {
        "pass1-next",
        "pass1-record",
        "pass2-next",
        "pass2-record",
        "revision-next",
        "pass1-revise",
        "status",
        "audit",
        "pilot15-export",
        "template",
    }


def test_cli_all_commands_emit_one_json_stream_and_expected_exit_codes(
    semantic_paths, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    monkeypatch.setattr(SEMANTIC, "default_paths", lambda: semantic_paths)
    assert _assert_cli_json(capsys, 0, lambda: SEMANTIC.main(["pass1-next"]))["status"] == (
        "NEXT_PASS1_TARGET"
    )
    assert _assert_cli_json(capsys, 0, lambda: SEMANTIC.main(["status"]))["status"] == (
        "SEMANTIC_STATUS"
    )
    assert _assert_cli_json(capsys, 0, lambda: SEMANTIC.main(["audit"]))["valid"] is True
    assert _assert_cli_json(
        capsys, 0, lambda: SEMANTIC.main(["template", "--slot-id", "KIS-015"])
    )["status"] == "PASS1_INPUT_TEMPLATE"
    pilot = tmp_path / "cli-pilot.json"
    assert _assert_cli_json(
        capsys,
        0,
        lambda: SEMANTIC.main(["pilot15-export", "--output", str(pilot)]),
    )["status"] == "PILOT15_EXPORTED"

    pass1_input = _write_json(
        tmp_path / "cli-pass1.json", _valid_payload(_current_target(semantic_paths))
    )
    assert _assert_cli_json(
        capsys,
        0,
        lambda: SEMANTIC.main(["pass1-record", "--input", str(pass1_input)]),
    )["status"] == "PASS1_RECORDED"
    pass2_target = _assert_cli_json(capsys, 0, lambda: SEMANTIC.main(["pass2-next"]))
    pass2_input = _write_json(
        tmp_path / "cli-pass2.json", _pass2_payload(pass2_target, "REVISION_REQUIRED")
    )
    assert _assert_cli_json(
        capsys,
        0,
        lambda: SEMANTIC.main(["pass2-record", "--input", str(pass2_input)]),
    )["status"] == "REVISION_REQUIRED"
    revision_target = _assert_cli_json(capsys, 0, lambda: SEMANTIC.main(["revision-next"]))
    assert revision_target["review_notes"] == "Revise the boundary."
    revise_input = _write_json(
        tmp_path / "cli-revise.json",
        _valid_payload(SEMANTIC._target_output(SEMANTIC.load_semantic_state(semantic_paths).targets[0])),
    )
    assert _assert_cli_json(
        capsys,
        0,
        lambda: SEMANTIC.main(["pass1-revise", "--input", str(revise_input)]),
    )["status"] == "PASS1_REVISED"


@pytest.mark.parametrize("failure", ["missing_file", "malformed_json", "state_conflict"])
def test_cli_expected_errors_are_json_stderr_only_without_traceback(
    semantic_paths,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
    failure: str,
) -> None:
    monkeypatch.setattr(SEMANTIC, "default_paths", lambda: semantic_paths)
    if failure == "missing_file":
        argv = ["pass1-record", "--input", str(tmp_path / "missing.json")]
    elif failure == "malformed_json":
        malformed = tmp_path / "malformed.json"
        malformed.write_text("{", encoding="utf-8")
        argv = ["pass1-record", "--input", str(malformed)]
    else:
        payload = _write_json(tmp_path / "no-pass2.json", {"not": "a pass2 payload"})
        argv = ["pass2-record", "--input", str(payload)]
    error = _assert_cli_json(capsys, 2, lambda: SEMANTIC.main(argv))
    assert error["status"] == "ERROR"


def test_cli_audit_error_sanitizes_hidden_split_metadata(
    semantic_paths, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    plan = semantic_paths.annotation_plan.read_text(encoding="utf-8")
    semantic_paths.annotation_plan.write_text(
        plan.replace("planned_split", "planned_split_corrupt", 1), encoding="utf-8"
    )
    monkeypatch.setattr(SEMANTIC, "default_paths", lambda: semantic_paths)
    error = _assert_cli_json(capsys, 2, lambda: SEMANTIC.main(["audit"]))
    text = json.dumps(error).casefold()
    assert error["status"] == "AUDIT_FAILED"
    for marker in ("planned_split", "development", "holdout", "q1b_dev", "q1b_holdout"):
        assert marker not in text


def test_cli_argparse_invalid_arguments_use_stderr_without_traceback(capsys) -> None:
    with pytest.raises(SystemExit) as caught:
        SEMANTIC.main(["pass1-record"])
    assert caught.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Traceback" not in captured.err
