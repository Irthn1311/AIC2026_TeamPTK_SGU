from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any

import pytest

from system_tai.quality.l21_150_error_analysis import (
    analyze_l21_150_errors,
    classify_query_report,
)
from system_tai.quality.l21_150_evaluator import OFFICIAL_K, evaluate_l21_150
from system_tai.quality.l21_150_schema import (
    BENCHMARK_ID,
    BENCHMARK_ROLE,
    FRAME_GT_STATUS,
    FrameInterval,
    L21150Benchmark,
    L21150KISQuery,
    L21150QAQuery,
    L21150TRAKEEvent,
    L21150TRAKEQuery,
    serialize_l21_150_benchmark,
)

SYSTEM_ROOT = Path(__file__).parents[1]
RUNNER_PATH = SYSTEM_ROOT / "scripts" / "l21_150_run_baseline.py"
SPEC = importlib.util.spec_from_file_location("system_tai_l21_150_runner_tests", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RUNNER
SPEC.loader.exec_module(RUNNER)


def _kis(query_id: str = "KIS-X") -> L21150KISQuery:
    return L21150KISQuery(
        query_id=query_id,
        query_vi="Một người đi qua cửa.",
        video_id="L21_V001",
        reference_timestamp="00:10",
        proposed_frame_center=100,
        proposed_interval=FrameInterval(90, 110),
        branch="Visual",
        difficulty="Dễ",
        split="DEV",
    )


def _qa(query_id: str = "QA-X") -> L21150QAQuery:
    return L21150QAQuery(
        query_id=query_id,
        question_vi="Vật thể có màu gì?",
        video_id="L21_V001",
        reference_timestamp="00:10",
        proposed_frame_center=100,
        proposed_interval=FrameInterval(90, 110),
        source_answer="Đỏ / màu đỏ",
        canonical_answer="Đỏ",
        accepted_answers=("Đỏ", "màu đỏ"),
        branch="Visual",
        difficulty="Dễ",
        split="DEV",
    )


def _trake(query_id: str = "TR-X") -> L21150TRAKEQuery:
    return L21150TRAKEQuery(
        query_id=query_id,
        video_id="L21_V001",
        events=(
            L21150TRAKEEvent(1, "Sự kiện 1", "00:10", 100, FrameInterval(96, 104)),
            L21150TRAKEEvent(2, "Sự kiện 2", "00:20", 200, FrameInterval(196, 204)),
            L21150TRAKEEvent(3, "Sự kiện 3", "00:30", 300, FrameInterval(296, 304)),
        ),
        branch="Temporal + Mixed",
        difficulty="Khó",
        split="DEV",
    )


def _trake_with_overlapping_event_boundaries() -> L21150TRAKEQuery:
    return L21150TRAKEQuery(
        query_id="TR-STRICT-ORDER",
        video_id="L21_V001",
        events=(
            L21150TRAKEEvent(1, "Event 1", "00:10", 100, FrameInterval(99, 100)),
            L21150TRAKEEvent(2, "Event 2", "00:11", 101, FrameInterval(100, 101)),
            L21150TRAKEEvent(3, "Event 3", "00:20", 200, FrameInterval(199, 200)),
        ),
        branch="Temporal + Mixed",
        difficulty="Hard",
        split="DEV",
    )


def _benchmark(*queries: Any) -> L21150Benchmark:
    return L21150Benchmark(
        schema_version=1,
        benchmark_id=BENCHMARK_ID,
        benchmark_role=BENCHMARK_ROLE,
        official_ground_truth=False,
        dataset_scope="L21 16-video subset",
        frame_gt_status=FRAME_GT_STATUS,
        description="Internal diagnostic fixture, not official BTC GT.",
        queries=tuple(queries),
    )


def _coordinate_validation_report(*records: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "benchmark_id": BENCHMARK_ID,
        "validation_role": "SOURCE_PROPOSED_GT_COORDINATE_EVIDENCE",
        "gt_coordinate_validation_kind": "RAW_FRAME_STRUCTURAL",
        "semantic_gt_authority": "SOURCE_PROPOSED_INTERNAL",
        "source_gt_mutated": False,
        "automatic_frame_shift_applied": False,
        "records": list(records),
    }


def _candidate(
    query_id: str,
    task: str,
    rank: int,
    *,
    video_id: str = "L21_V001",
    frame_id: int = 100,
    answer: str = "Đỏ",
    frame_ids: list[int] | None = None,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "query_id": query_id,
        "task": task,
        "rank": rank,
        "video_id": video_id,
    }
    if task == "trake":
        value["actual_frame_ids"] = frame_ids or [100, 200, 300]
    else:
        value["actual_frame_id"] = frame_id
        if task == "qa":
            value["answer"] = answer
    return value


@pytest.mark.parametrize("frame_id", [90, 110])
def test_kis_interval_boundaries_are_inclusive(frame_id: int) -> None:
    query = _kis()
    report = evaluate_l21_150(
        _benchmark(query), [_candidate(query.query_id, "kis", 1, frame_id=frame_id)]
    )
    assert report["query_reports"][0]["r_at_k"] == {
        str(cutoff): 1.0 for cutoff in OFFICIAL_K
    }


def test_kis_wrong_video_scores_zero() -> None:
    query = _kis()
    report = evaluate_l21_150(
        _benchmark(query),
        [_candidate(query.query_id, "kis", 1, video_id="L21_V002")],
    )
    assert report["overall"]["final_score"] == 0.0


def test_kis_prefix_max_and_final_score_mean_five_values() -> None:
    query = _kis()
    predictions = [
        _candidate(query.query_id, "kis", rank, video_id="L21_V002")
        for rank in range(1, 5)
    ]
    predictions.append(_candidate(query.query_id, "kis", 5, frame_id=100))
    report = evaluate_l21_150(_benchmark(query), predictions)
    assert report["query_reports"][0]["r_at_k"] == {
        "1": 0.0,
        "5": 1.0,
        "20": 1.0,
        "50": 1.0,
        "100": 1.0,
    }
    assert report["overall"]["final_score"] == pytest.approx(0.8)
    assert report["task_metrics"]["kis"]["mrr"] == pytest.approx(0.2)


@pytest.mark.parametrize(
    ("frame_id", "expected_signed_distance"),
    [(80, -10), (120, 10)],
)
def test_kis_video_hit_frame_miss_reports_signed_interval_distance(
    frame_id: int, expected_signed_distance: int
) -> None:
    query = _kis()
    report = evaluate_l21_150(
        _benchmark(query),
        [
            _candidate(query.query_id, "kis", 1, video_id="L21_V002"),
            _candidate(query.query_id, "kis", 2, frame_id=frame_id),
        ],
    )
    query_report = report["query_reports"][0]

    assert query_report["first_video_hit_rank"] == 2
    assert query_report["first_video_hit_actual_frame_id"] == frame_id
    assert (
        query_report["first_video_hit_signed_distance_to_gt_interval_frames"]
        == expected_signed_distance
    )
    assert query_report["first_frame_hit_rank"] is None
    assert query_report["nearest_same_video_candidate_frame_id"] == frame_id
    assert query_report["nearest_same_video_candidate_rank"] == 2
    assert query_report["nearest_same_video_frame_distance_frames"] == 10


def test_kis_frame_hit_reports_first_and_nearest_same_video_candidates() -> None:
    query = _kis()
    report = evaluate_l21_150(
        _benchmark(query),
        [
            _candidate(query.query_id, "kis", 1, frame_id=80),
            _candidate(query.query_id, "kis", 2, frame_id=100),
        ],
    )
    query_report = report["query_reports"][0]

    assert query_report["first_video_hit_rank"] == 1
    assert query_report["first_video_hit_actual_frame_id"] == 80
    assert query_report["first_frame_hit_rank"] == 2
    assert query_report["first_relevant_rank"] == 2
    assert query_report["nearest_same_video_candidate_frame_id"] == 100
    assert query_report["nearest_same_video_candidate_rank"] == 2
    assert query_report["nearest_same_video_frame_distance_frames"] == 0
    assert query_report["gt_interval_start_frame_id"] == 90
    assert query_report["gt_interval_end_frame_id"] == 110


def test_kis_video_miss_has_no_same_video_frame_diagnostics() -> None:
    query = _kis()
    report = evaluate_l21_150(
        _benchmark(query),
        [_candidate(query.query_id, "kis", 1, video_id="L21_V002")],
    )
    query_report = report["query_reports"][0]

    assert query_report["first_video_hit_rank"] is None
    assert query_report["first_video_hit_actual_frame_id"] is None
    assert query_report["first_frame_hit_rank"] is None
    assert query_report["nearest_same_video_candidate_frame_id"] is None
    assert query_report["nearest_same_video_candidate_rank"] is None
    assert query_report["nearest_same_video_frame_distance_frames"] is None


@pytest.mark.parametrize(
    ("video_id", "frame_id", "answer", "expected"),
    [
        ("L21_V001", 100, "màu đỏ", 1.0),
        ("L21_V001", 120, "màu đỏ", 0.0),
        ("L21_V001", 100, "xanh", 0.0),
        ("L21_V002", 100, "màu đỏ", 0.0),
    ],
)
def test_qa_requires_full_video_frame_answer_tuple(
    video_id: str, frame_id: int, answer: str, expected: float
) -> None:
    query = _qa()
    report = evaluate_l21_150(
        _benchmark(query),
        [
            _candidate(
                query.query_id,
                "qa",
                1,
                video_id=video_id,
                frame_id=frame_id,
                answer=answer,
            )
        ],
    )
    assert report["overall"]["final_score"] == expected


def test_qa_reports_answer_right_grounding_wrong() -> None:
    query = _qa()
    report = evaluate_l21_150(
        _benchmark(query),
        [_candidate(query.query_id, "qa", 1, frame_id=120, answer="Đỏ")],
    )
    query_report = report["query_reports"][0]
    assert query_report["answer_hit"] is True
    assert query_report["frame_hit"] is False
    assert report["task_metrics"]["qa"]["answer_right_grounding_wrong_count"] == 1


@pytest.mark.parametrize(
    ("frame_ids", "expected"),
    [
        ([100, 999, 999], 1 / 3),
        ([100, 200, 999], 2 / 3),
        ([100, 200, 300], 1.0),
    ],
)
def test_trake_scores_corresponding_event_coverage(
    frame_ids: list[int], expected: float
) -> None:
    query = _trake()
    report = evaluate_l21_150(
        _benchmark(query),
        [_candidate(query.query_id, "trake", 1, frame_ids=frame_ids)],
    )
    assert report["query_reports"][0]["r_at_k"]["1"] == pytest.approx(expected)


def test_trake_wrong_video_scores_zero() -> None:
    query = _trake()
    report = evaluate_l21_150(
        _benchmark(query),
        [_candidate(query.query_id, "trake", 1, video_id="L21_V002")],
    )
    assert report["overall"]["final_score"] == 0.0


def test_trake_does_not_reorder_prediction_events() -> None:
    query = _trake()
    report = evaluate_l21_150(
        _benchmark(query),
        [_candidate(query.query_id, "trake", 1, frame_ids=[300, 200, 100])],
    )
    query_report = report["query_reports"][0]
    assert query_report["r_at_k"]["1"] == pytest.approx(1 / 3)
    assert query_report["event_order_valid"] is False


@pytest.mark.parametrize(
    ("frame_ids", "expected_order"),
    [
        ([100, 100, 200], False),
        ([100, 101, 200], True),
    ],
)
def test_trake_event_order_requires_strict_monotonic_frames(
    frame_ids: list[int], expected_order: bool
) -> None:
    query = _trake_with_overlapping_event_boundaries()
    report = evaluate_l21_150(
        _benchmark(query),
        [_candidate(query.query_id, "trake", 1, frame_ids=frame_ids)],
    )
    query_report = report["query_reports"][0]

    assert query_report["r_at_k"]["1"] == pytest.approx(1.0)
    assert query_report["event_coverage"] == pytest.approx(1.0)
    assert query_report["event_order_valid"] is expected_order


def test_trake_equal_adjacent_full_hits_fail_chain_and_report_order_error() -> None:
    query = _trake_with_overlapping_event_boundaries()
    report = evaluate_l21_150(
        _benchmark(query),
        [_candidate(query.query_id, "trake", 1, frame_ids=[100, 100, 200])],
    )
    query_report = report["query_reports"][0]

    assert query_report["r_at_k"]["1"] == pytest.approx(1.0)
    assert query_report["per_event_accuracy"] == [1.0, 1.0, 1.0]
    assert query_report["event_order_valid"] is False
    assert query_report["full_chain_accuracy"] is False
    assert "ORDER_FAIL" in classify_query_report(query_report)


def test_trake_partial_chain_is_scored_without_fabricating_missing_events() -> None:
    query = _trake()
    report = evaluate_l21_150(
        _benchmark(query),
        [_candidate(query.query_id, "trake", 1, frame_ids=[100, 200])],
    )
    query_report = report["query_reports"][0]
    assert query_report["event_coverage"] == pytest.approx(2 / 3)
    assert query_report["chain_completeness"] == pytest.approx(2 / 3)


def test_validated_only_excludes_unvalidated_queries() -> None:
    first = _kis("KIS-1")
    second = _kis("KIS-2")
    validation = _coordinate_validation_report(
        {
            "query_id": "KIS-1",
            "event_index": None,
            "status": "VALIDATED",
            "nearest_keyframe_inside_proposed_interval": False,
        },
        {"query_id": "KIS-2", "event_index": None, "status": "OUT_OF_RANGE"},
    )
    report = evaluate_l21_150(
        _benchmark(first, second),
        [_candidate("KIS-1", "kis", 1)],
        gt_policy="validated-only",
        mapping_validation_report=validation,
    )
    assert report["gt_evidence_mode"] == (
        "SOURCE_PROPOSED_RAW_FRAME_COORDINATE_VALIDATED"
    )
    assert report["gt_coordinate_validation_kind"] == "RAW_FRAME_STRUCTURAL"
    assert report["semantic_gt_authority"] == "SOURCE_PROPOSED_INTERNAL"
    assert report["selected_query_count"] == 1
    assert report["excluded_unvalidated_query_ids"] == ["KIS-2"]


def test_validated_only_trake_requires_all_coordinate_records_not_overlap() -> None:
    valid = _trake("TR-VALID")
    invalid = _trake("TR-INVALID")
    records = [
        {
            "query_id": valid.query_id,
            "event_index": event_index,
            "status": "VALIDATED",
            "nearest_keyframe_inside_proposed_interval": False,
        }
        for event_index in (1, 2, 3)
    ]
    records.extend(
        {
            "query_id": invalid.query_id,
            "event_index": event_index,
            "status": "OUT_OF_RANGE" if event_index == 2 else "VALIDATED",
            "nearest_keyframe_inside_proposed_interval": True,
        }
        for event_index in (1, 2, 3)
    )

    report = evaluate_l21_150(
        _benchmark(valid, invalid),
        [_candidate(valid.query_id, "trake", 1)],
        gt_policy="validated-only",
        mapping_validation_report=_coordinate_validation_report(*records),
    )

    assert report["selected_query_count"] == 1
    assert report["excluded_unvalidated_query_ids"] == ["TR-INVALID"]


def test_validated_only_rejects_obsolete_schema_v1_mapping_report() -> None:
    with pytest.raises(ValueError, match="obsolete keyframe-overlap semantics"):
        evaluate_l21_150(
            _benchmark(_kis()),
            [],
            gt_policy="validated-only",
            mapping_validation_report={"schema_version": 1, "records": []},
        )


def test_validated_only_requires_mapping_evidence() -> None:
    with pytest.raises(ValueError, match="requires a mapping validation report"):
        evaluate_l21_150(_benchmark(_kis()), [], gt_policy="validated-only")


def test_proposed_policy_is_prominently_marked_unverified() -> None:
    report = evaluate_l21_150(_benchmark(_kis()), [], gt_policy="proposed")
    assert report["gt_evidence_mode"] == "SOURCE_PROPOSED_GT"
    assert report["gt_coordinate_validation_kind"] == "NOT_APPLIED"
    assert report["semantic_gt_authority"] == "SOURCE_PROPOSED_INTERNAL"
    assert report["official_ground_truth"] is False
    assert report["semantic_accuracy_claim"] is False


def test_invalid_rank_and_duplicate_are_reported() -> None:
    query = _kis()
    report = evaluate_l21_150(
        _benchmark(query),
        [
            _candidate(query.query_id, "kis", 1),
            _candidate(query.query_id, "kis", 1),
            _candidate(query.query_id, "kis", 101),
        ],
    )
    query_report = report["query_reports"][0]
    assert query_report["result_valid"] is False
    assert report["overall"]["invalid_result_count"] >= 2


@pytest.mark.parametrize(
    ("report", "expected"),
    [
        (
            {
                "task": "kis",
                "result_valid": True,
                "video_hit": False,
                "frame_hit": False,
                "validation_errors": [],
            },
            ("VIDEO_MISS",),
        ),
        (
            {
                "task": "qa",
                "result_valid": True,
                "video_hit": True,
                "frame_hit": False,
                "answer_hit": True,
                "answer_hit_given_grounding": False,
            },
            ("FRAME_MISS", "ANSWER_RIGHT_GROUNDING_WRONG"),
        ),
        (
            {
                "task": "trake",
                "result_valid": True,
                "video_hit": True,
                "event_coverage": 2 / 3,
                "event_order_valid": False,
                "chain_completeness": 2 / 3,
                "full_chain_accuracy": False,
            },
            ("EVENT_MISS", "ORDER_FAIL", "PARTIAL_CHAIN"),
        ),
    ],
)
def test_error_analyzer_mechanical_categories(
    report: dict[str, Any], expected: tuple[str, ...]
) -> None:
    assert classify_query_report(report) == expected


def test_error_analyzer_aggregates_without_causal_claim() -> None:
    evaluation = evaluate_l21_150(_benchmark(_kis()), [])
    report = analyze_l21_150_errors(evaluation)
    assert report["causal_claims_made"] is False
    assert report["category_counts"] == {"VIDEO_MISS": 1}
    assert report["aggregates"]["branch"]["Visual"] == {"VIDEO_MISS": 1}


@pytest.mark.parametrize(
    ("task", "overrides", "failure_reason", "expected_category"),
    [
        ("kis", {"video_hit": True, "frame_hit": False}, None, "VIDEO_HIT_FRAME_MISS"),
        (
            "kis",
            {"video_hit": True, "frame_hit": True, "first_relevant_rank": 6},
            None,
            "FRAME_HIT_LOW_RANK",
        ),
        (
            "kis",
            {"validation_errors": ["duplicate candidate identities: 1"]},
            None,
            "DUPLICATE_PRESSURE",
        ),
        ("qa", {"video_hit": False}, None, "VIDEO_MISS"),
        (
            "qa",
            {"video_hit": True, "frame_hit": True, "answer_hit_given_grounding": False},
            None,
            "ANSWER_MISS",
        ),
        ("qa", {}, "Unsupported answer type", "UNSUPPORTED_ANSWER_TYPE"),
        ("trake", {"video_hit": False}, None, "VIDEO_MISS"),
        (
            "trake",
            {
                "video_hit": True,
                "event_coverage": 2 / 3,
                "event_order_valid": True,
                "chain_completeness": 1.0,
            },
            None,
            "EVENT_MISS",
        ),
        (
            "trake",
            {
                "video_hit": True,
                "event_coverage": 1.0,
                "event_order_valid": True,
                "chain_completeness": 1.0,
                "full_chain_accuracy": True,
                "first_relevant_rank": 6,
            },
            None,
            "FULL_CHAIN_LOW_RANK",
        ),
        ("kis", {"result_valid": False}, None, "OUTPUT_INVALID"),
        ("qa", {"result_valid": False}, None, "OUTPUT_INVALID"),
        ("trake", {"result_valid": False}, None, "OUTPUT_INVALID"),
    ],
)
def test_every_error_taxonomy_category_is_mechanically_reachable(
    task: str,
    overrides: dict[str, Any],
    failure_reason: str | None,
    expected_category: str,
) -> None:
    base: dict[str, Any] = {
        "task": task,
        "result_valid": True,
        "video_hit": True,
        "frame_hit": True,
        "answer_hit": False,
        "answer_hit_given_grounding": True,
        "event_coverage": 1.0,
        "event_order_valid": True,
        "chain_completeness": 1.0,
        "full_chain_accuracy": False,
        "first_relevant_rank": 1,
        "validation_errors": [],
    }
    base.update(overrides)
    assert expected_category in classify_query_report(base, failure_reason=failure_reason)


@dataclass
class _FakeRuntime:
    output_root: Path
    fail_query_id: str | None = None
    kis_depth_by_query: dict[str, int] | None = None
    duplicate_last_kis_identity: bool = False

    def __post_init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self.manifest = SimpleNamespace(fingerprint="fixture-corpus", schema_version=2)
        self.shared_encoder = SimpleNamespace(
            identifiers=MappingProxyType(
                {
                    "library": "clip",
                    "model": "fake-clip",
                    "tokenization": MappingProxyType({"context_length": 77}),
                }
            )
        )
        self.config = SimpleNamespace(device="cpu")

    def handle_query(self, request) -> dict[str, Any]:
        self.calls.append(("kis", request.query_id))
        if request.query_id == self.fail_query_id:
            raise RuntimeError("synthetic KIS failure")
        depth = (
            self.kis_depth_by_query.get(request.query_id, request.output_top_k)
            if self.kis_depth_by_query is not None
            else request.output_top_k
        )
        target = self.output_root / "requests" / request.query_id / "top100.jsonl"
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8", newline="") as stream:
            for rank in range(1, depth + 1):
                frame_id = 99 + rank
                if self.duplicate_last_kis_identity and rank == depth and depth > 1:
                    frame_id -= 1
                stream.write(
                    json.dumps(
                        {
                            "query_id": request.query_id,
                            "rank": rank,
                            "video_id": "L21_V001",
                            "frame_id": frame_id,
                        }
                    )
                    + "\n"
                )
        return {
            "status": "SUCCESS",
            "artifacts": {
                "top100_jsonl": target.relative_to(self.output_root).as_posix()
            },
            "timings": {"total_seconds": 0.25},
        }

    def handle_qa_query(self, request) -> dict[str, Any]:
        self.calls.append(("qa", request.query_id))
        if request.query_id == self.fail_query_id:
            raise RuntimeError("unsupported answer type fixture")
        return {
            "status": "SUCCESS",
            "predictions": [
                {
                    "query_id": request.query_id,
                    "rank": 1,
                    "video_id": "L21_V001",
                    "frame_id": 100,
                    "answer": "Đỏ",
                }
            ],
            "timings": {"total_seconds": 0.5},
        }

    def handle_trake_query(self, request) -> dict[str, Any]:
        self.calls.append(("trake", request.query_id))
        if request.query_id == self.fail_query_id:
            raise RuntimeError("synthetic TRAKE failure")
        return {
            "status": "SUCCESS",
            "predictions": [
                {
                    "query_id": request.query_id,
                    "rank": 1,
                    "video_id": "L21_V001",
                    "frame_ids": [100, 200, 300],
                }
            ],
            "timings": {"total_seconds": 0.75},
        }


def _run_fake(
    tmp_path: Path,
    *,
    fail_query_id: str | None = None,
    fail_fast: bool = False,
    top_k: int = 3,
    task: str = "all",
    queries: tuple[Any, ...] | None = None,
    kis_depth_by_query: dict[str, int] | None = None,
    duplicate_last_kis_identity: bool = False,
) -> tuple[dict[str, Any], _FakeRuntime, Path]:
    benchmark = _benchmark(*(queries or (_kis(), _qa(), _trake())))
    runtime = _FakeRuntime(
        tmp_path / "runtime",
        fail_query_id=fail_query_id,
        kis_depth_by_query=kis_depth_by_query,
        duplicate_last_kis_identity=duplicate_last_kis_identity,
    )
    output = tmp_path / "output"
    report = RUNNER.run_l21_150_baseline(
        benchmark,
        runtime,
        output,
        experiment_id="fixture-e0",
        split="all",
        task=task,
        top_k=top_k,
        refine_top_n=0,
        resume=False,
        fail_fast=fail_fast,
        benchmark_sha256="b" * 64,
        manifest_sha256="m" * 64,
        gt_policy="proposed",
    )
    return report, runtime, output


def test_runner_json_safe_converts_top_level_mapping_proxy() -> None:
    value = MappingProxyType({"model": "ViT-B/32", "device": "cuda"})

    assert RUNNER._json_safe(value) == {"model": "ViT-B/32", "device": "cuda"}


def test_runner_json_safe_converts_nested_immutable_containers() -> None:
    value = MappingProxyType(
        {
            "model": MappingProxyType({"name": "ViT-B/32"}),
            "preprocessing": ("resize", ["crop", MappingProxyType({"size": 224})]),
            "cache": Path("clip-cache/model.pt"),
        }
    )

    assert RUNNER._json_safe(value) == {
        "model": {"name": "ViT-B/32"},
        "preprocessing": ["resize", ["crop", {"size": 224}]],
        "cache": str(Path("clip-cache/model.pt")),
    }


def test_runner_json_safe_rejects_unsupported_metadata_type() -> None:
    with pytest.raises(TypeError, match="unsupported type SimpleNamespace"):
        RUNNER._json_safe({"model": SimpleNamespace(name="clip")})


def test_baseline_runner_uses_existing_runtime_contracts_and_top100_boundary(
    tmp_path: Path,
) -> None:
    report, runtime, output = _run_fake(tmp_path, top_k=100)
    assert runtime.calls == [("kis", "KIS-X"), ("qa", "QA-X"), ("trake", "TR-X")]
    predictions = [
        json.loads(line)
        for line in (output / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len([row for row in predictions if row["query_id"] == "KIS-X"]) == 100
    assert max(row["rank"] for row in predictions if row["query_id"] == "KIS-X") == 100
    assert report["production_algorithm_modified"] is False
    assert report["production_algorithm_modified_scope"] == (
        "CORE_PRODUCTION_IMPLEMENTATION"
    )
    assert report["core_production_algorithm_modified"] is False
    assert report["kis_query_policy"] == "VI_ONLY"
    assert report["query_policy_changed_from_e0"] is False
    assert "kis_query_experiment" not in report
    assert report["runtime_contract"] == "OperationalKISRuntime public task handlers"
    summary = (output / "run_summary.md").read_text(encoding="utf-8")
    assert summary.startswith("# L21-150 Experiment Run\n")
    assert "Core production retrieval/ranking implementation changed: `false`" in summary
    assert "KIS query policy: `VI_ONLY`" in summary
    assert "Query policy changed from E0: `false`" in summary


def test_baseline_depth_diagnostics_exact_requested_depth(tmp_path: Path) -> None:
    report, _, _ = _run_fake(
        tmp_path,
        top_k=100,
        task="kis",
        queries=(_kis(),),
    )

    assert report["requested_top_k"] == 100
    assert report["prediction_record_count"] == 100
    assert report["prediction_depth_min"] == 100
    assert report["prediction_depth_max"] == 100
    assert report["prediction_depth_distribution"] == {"100": 1}
    assert report["queries_below_requested_depth"] == []
    assert report["duplicate_output_identity_count"] == 0


def test_baseline_depth_diagnostics_mixed_99_and_100_are_not_failures(
    tmp_path: Path,
) -> None:
    report, _, _ = _run_fake(
        tmp_path,
        top_k=100,
        task="kis",
        queries=(_kis("KIS-1"), _kis("KIS-2")),
        kis_depth_by_query={"KIS-1": 100, "KIS-2": 99},
    )

    assert report["successful_query_count"] == 2
    assert report["failed_query_count"] == 0
    assert report["prediction_record_count"] == 199
    assert report["prediction_depth_distribution"] == {"99": 1, "100": 1}
    assert report["queries_below_requested_depth"] == ["KIS-2"]


def test_baseline_reports_duplicate_output_identity_without_padding(
    tmp_path: Path,
) -> None:
    report, _, _ = _run_fake(
        tmp_path,
        top_k=3,
        task="kis",
        queries=(_kis(),),
        duplicate_last_kis_identity=True,
    )

    assert report["prediction_record_count"] == 3
    assert report["duplicate_output_identity_count"] == 1


def test_baseline_rejects_query_output_exceeding_requested_top_k(
    tmp_path: Path,
) -> None:
    report, _, output = _run_fake(
        tmp_path,
        top_k=100,
        task="kis",
        queries=(_kis(),),
        kis_depth_by_query={"KIS-X": 101},
    )

    assert report["successful_query_count"] == 0
    assert report["failed_query_count"] == 1
    failures = [
        json.loads(line)
        for line in (output / "failures.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert "exceeding requested_top_k=100" in failures[0]["failure_reason"]


def test_baseline_runner_preserves_actual_frame_fields(tmp_path: Path) -> None:
    _, _, output = _run_fake(tmp_path)
    predictions = [
        json.loads(line)
        for line in (output / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    kis = next(row for row in predictions if row["task"] == "kis")
    trake = next(row for row in predictions if row["task"] == "trake")
    assert kis["actual_frame_id"] == 100
    assert trake["actual_frame_ids"] == [100, 200, 300]
    assert "frame_id" not in kis


def test_baseline_failure_is_captured_without_corrupting_remaining_queries(
    tmp_path: Path,
) -> None:
    report, runtime, output = _run_fake(tmp_path, fail_query_id="QA-X")
    assert report["successful_query_count"] == 2
    assert report["failed_query_count"] == 1
    assert runtime.calls == [("kis", "KIS-X"), ("qa", "QA-X"), ("trake", "TR-X")]
    failures = [
        json.loads(line)
        for line in (output / "failures.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert failures[0]["query_id"] == "QA-X"
    assert "unsupported answer type" in failures[0]["failure_reason"]


def test_baseline_fail_fast_stops_after_failure(tmp_path: Path) -> None:
    report, runtime, _ = _run_fake(tmp_path, fail_query_id="QA-X", fail_fast=True)
    assert report["failed_query_count"] == 1
    assert runtime.calls == [("kis", "KIS-X"), ("qa", "QA-X")]


def test_baseline_output_is_evaluator_compatible(tmp_path: Path) -> None:
    _, _, output = _run_fake(tmp_path)
    predictions = [
        json.loads(line)
        for line in (output / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    evaluation = evaluate_l21_150(_benchmark(_kis(), _qa(), _trake()), predictions)
    assert evaluation["selected_query_count"] == 3
    assert evaluation["overall"]["invalid_result_count"] == 0


def test_baseline_runner_does_not_duplicate_retrieval_implementation() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    assert "OperationalKISRuntime" in source
    assert "ExactNumpyRetriever" not in source
    assert "WeightedRRFRetriever" not in source
    assert "np.dot" not in source


def test_baseline_manifest_contains_reproducibility_fields(tmp_path: Path) -> None:
    report, _, output = _run_fake(tmp_path)
    assert report["benchmark_sha256"] == "b" * 64
    assert report["manifest_sha256"] == "m" * 64
    assert report["benchmark_id"] == BENCHMARK_ID
    assert report["corpus_fingerprint"] == "fixture-corpus"
    assert report["model_identity"] == {
        "library": "clip",
        "model": "fake-clip",
        "tokenization": {"context_length": 77},
    }
    assert report["gt_policy"] == "proposed"
    manifest_path = output / "experiment_manifest.json"
    assert manifest_path.is_file()
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["model_identity"] == (
        report["model_identity"]
    )
    assert (output / "run_summary.md").is_file()


def test_runner_does_not_modify_benchmark_bytes(tmp_path: Path) -> None:
    benchmark = _benchmark(_kis(), _qa(), _trake())
    before = serialize_l21_150_benchmark(benchmark)
    _run_fake(tmp_path)
    assert serialize_l21_150_benchmark(benchmark) == before
