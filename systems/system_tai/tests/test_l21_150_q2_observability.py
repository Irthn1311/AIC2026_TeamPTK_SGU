from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any

import pytest

from system_tai.quality.l21_150_kis_comparison import (
    L21150KISComparisonError,
    compare_l21_150_kis_arms,
)
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
    load_l21_150_benchmark,
)
from system_tai.quality.l21_150_stage_analysis import (
    analyze_l21_150_stages,
    compare_partial_chain_and_zero_output,
)
from system_tai.quality.l21_150_translation import (
    TRANSLATION_STATUS,
    KISTranslationSidecarError,
    load_kis_dev_translation_sidecar,
    validate_kis_dev_translation_payload,
)

SYSTEM_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_PATH = SYSTEM_ROOT / "benchmarks/l21_150_diagnostic/benchmark.json"
SIDECAR_PATH = (
    SYSTEM_ROOT
    / "benchmarks/l21_150_diagnostic/q2_kis_dev_en_translation.json"
)
RUNNER_PATH = SYSTEM_ROOT / "scripts/l21_150_run_baseline.py"


def _load_script(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RUNNER = _load_script(RUNNER_PATH, "l21_150_runner_q2_test")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )


def _synthetic_stage_benchmark() -> L21150Benchmark:
    qa = L21150QAQuery(
        query_id="QA-STAGE",
        question_vi="Chiếc xe màu gì?",
        video_id="L21_TARGET",
        reference_timestamp="00:00:01",
        proposed_frame_center=10,
        proposed_interval=FrameInterval(8, 12),
        source_answer="đỏ",
        canonical_answer="đỏ",
        accepted_answers=("đỏ",),
        branch="qa-color",
        difficulty="easy",
        split="DEV",
    )
    events = tuple(
        L21150TRAKEEvent(
            event_index=index,
            description_vi=f"Sự kiện {index}",
            reference_timestamp=f"00:00:0{index}",
            proposed_frame_center=index * 10,
            proposed_interval=FrameInterval(index * 10 - 1, index * 10 + 1),
        )
        for index in (1, 2)
    )
    trake = L21150TRAKEQuery(
        query_id="TR-STAGE",
        video_id="L21_TARGET",
        events=events,
        branch="trake",
        difficulty="medium",
        split="DEV",
    )
    return L21150Benchmark(
        schema_version=1,
        benchmark_id=BENCHMARK_ID,
        benchmark_role=BENCHMARK_ROLE,
        official_ground_truth=False,
        dataset_scope="synthetic test only",
        frame_gt_status=FRAME_GT_STATUS,
        description="Synthetic stage-analysis fixture.",
        queries=(qa, trake),
    )


def _make_stage_run(root: Path) -> tuple[Path, Path]:
    qa_dir = root / "runtime/requests/qa-stage"
    _write_json(
        qa_dir / "qa_request_manifest.json",
        {"query_id": "QA-STAGE", "question_type": "COLOR"},
    )
    _write_json(
        qa_dir / "qa_evidence.json",
        {
            "query_id": "QA-STAGE",
            "question_type": "COLOR",
            "fused_retrieval_candidates": [
                {"rank": 1, "video_id": "L21_TARGET", "frame_id": 9}
            ],
            "refined_candidates": [
                {
                    "original_rank": 1,
                    "video_id": "L21_OTHER",
                    "candidate_frame_id": 5,
                    "refined_frame_id": 6,
                    "status": "REFINED",
                }
            ],
            "usable_evidence_candidates": [
                {"rank": 1, "video_id": "L21_TARGET", "frame_id": 10}
            ],
        },
    )
    _write_jsonl(
        qa_dir / "qa_predictions.jsonl",
        [
            {
                "query_id": "QA-STAGE",
                "rank": 1,
                "video_id": "L21_TARGET",
                "frame_id": 10,
                "answer": "đỏ",
            }
        ],
    )

    trake_dir = root / "runtime/requests/tr-stage"
    _write_json(trake_dir / "trake_request_manifest.json", {"query_id": "TR-STAGE"})
    _write_json(
        trake_dir / "trake_event_candidates.json",
        {
            "query_id": "TR-STAGE",
            "event_candidates": [
                {
                    "event_index": 0,
                    "candidate_count": 1,
                    "candidates": [
                        {"rank": 1, "video_id": "L21_TARGET", "frame_id": 10}
                    ],
                },
                {
                    "event_index": 1,
                    "candidate_count": 1,
                    "candidates": [
                        {"rank": 1, "video_id": "L21_TARGET", "frame_id": 20}
                    ],
                },
            ],
        },
    )
    _write_json(
        trake_dir / "trake_refinement.json",
        {
            "query_id": "TR-STAGE",
            "path_diagnostics": [
                {
                    "c1_rank": 1,
                    "video_id": "L21_TARGET",
                    "original_frame_ids": [10, 20],
                }
            ],
        },
    )
    _write_jsonl(
        trake_dir / "trake_predictions.jsonl",
        [
            {
                "query_id": "TR-STAGE",
                "rank": 1,
                "video_id": "L21_TARGET",
                "frame_ids": [10, 20],
            }
        ],
    )
    _write_json(
        root / "experiment_manifest.json",
        {
            "queries": [
                {"query_id": "QA-STAGE", "status": "SUCCESS"},
                {"query_id": "TR-STAGE", "status": "SUCCESS"},
            ]
        },
    )
    error_path = root / "error_analysis.json"
    _write_json(
        error_path,
        {
            "query_errors": [
                {
                    "query_id": "TR-STAGE",
                    "categories": ["VIDEO_MISS"],
                }
            ]
        },
    )
    return root, error_path


def test_real_translation_sidecar_is_exactly_38_dev_queries_and_no_holdout() -> None:
    benchmark = load_l21_150_benchmark(BENCHMARK_PATH)
    sidecar = load_kis_dev_translation_sidecar(SIDECAR_PATH, benchmark, BENCHMARK_PATH)
    dev_ids = [
        query.query_id
        for query in benchmark.queries
        if isinstance(query, L21150KISQuery) and query.split == "DEV"
    ]
    holdout_ids = {
        query.query_id
        for query in benchmark.queries
        if isinstance(query, L21150KISQuery) and query.split == "HOLDOUT"
    }

    assert sidecar.query_count == 38
    assert sidecar.translation_status == TRANSLATION_STATUS
    assert [record.query_id for record in sidecar.records] == dev_ids
    assert not holdout_ids.intersection(record.query_id for record in sidecar.records)
    assert sidecar.retrieval_feedback_used is False


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.update(benchmark_sha256="0" * 64), "benchmark_sha256"),
        (lambda value: value.update(task="qa"), "task must be kis"),
        (lambda value: value.update(split="holdout"), "split must be dev"),
        (lambda value: value.update(schema_version=2), "schema_version"),
        (
            lambda value: value["records"][0].update(translation_en=""),
            "translation_en",
        ),
        (
            lambda value: value["records"][0].update(source_vi="mismatch"),
            "source_vi mismatch",
        ),
        (
            lambda value: value["records"].append(copy.deepcopy(value["records"][0])),
            "query_count does not match",
        ),
    ],
)
def test_translation_sidecar_rejects_contract_failures(mutation, message: str) -> None:
    benchmark = load_l21_150_benchmark(BENCHMARK_PATH)
    payload = json.loads(SIDECAR_PATH.read_text(encoding="utf-8"))
    mutation(payload)
    with pytest.raises(KISTranslationSidecarError, match=message):
        validate_kis_dev_translation_payload(
            payload,
            benchmark,
            benchmark_sha256=hashlib.sha256(BENCHMARK_PATH.read_bytes()).hexdigest(),
        )


def test_translation_sidecar_loader_rejects_duplicate_nested_json_key(
    tmp_path: Path,
) -> None:
    text = SIDECAR_PATH.read_text(encoding="utf-8")
    text = text.replace(
        '"query_id": "KIS-01",',
        '"query_id": "KIS-01",\n      "query_id": "KIS-01",',
        1,
    )
    path = tmp_path / "duplicate.json"
    path.write_text(text, encoding="utf-8")
    benchmark = load_l21_150_benchmark(BENCHMARK_PATH)
    with pytest.raises(KISTranslationSidecarError, match="duplicate JSON key"):
        load_kis_dev_translation_sidecar(path, benchmark, BENCHMARK_PATH)


@pytest.mark.parametrize("failure", ("duplicate", "missing", "extra", "holdout"))
def test_translation_sidecar_rejects_query_id_set_failures(failure: str) -> None:
    benchmark = load_l21_150_benchmark(BENCHMARK_PATH)
    payload = json.loads(SIDECAR_PATH.read_text(encoding="utf-8"))
    if failure == "duplicate":
        payload["records"][1] = copy.deepcopy(payload["records"][0])
        message = "duplicate query_id"
    elif failure == "missing":
        payload["records"].pop()
        payload["query_count"] -= 1
        message = "records must exactly follow"
    elif failure == "extra":
        payload["records"][0]["query_id"] = "KIS-UNKNOWN"
        message = "extra or unknown"
    else:
        holdout = next(
            query
            for query in benchmark.queries
            if isinstance(query, L21150KISQuery) and query.split == "HOLDOUT"
        )
        payload["records"][0]["query_id"] = holdout.query_id
        message = "HOLDOUT query is forbidden"
    with pytest.raises(KISTranslationSidecarError, match=message):
        validate_kis_dev_translation_payload(
            payload,
            benchmark,
            benchmark_sha256=hashlib.sha256(BENCHMARK_PATH.read_bytes()).hexdigest(),
        )


def test_runner_default_request_is_vi_only_and_augmented_request_has_two_variants() -> None:
    benchmark = load_l21_150_benchmark(BENCHMARK_PATH)
    query = next(
        query
        for query in benchmark.queries
        if isinstance(query, L21150KISQuery) and query.split == "DEV"
    )
    default = RUNNER._runtime_request(query, "e0", 100, 3)
    augmented = RUNNER._runtime_request(
        query,
        "q2",
        100,
        3,
        kis_query_policy="translation_augmented_rrf",
        kis_translations={query.query_id: "A faithful English translation."},
    )

    assert default.query_vi == query.query_vi
    assert default.query_en is None
    assert default.query_en_expansion is None
    assert len(default.variants()) == 1
    assert augmented.query_vi == query.query_vi
    assert augmented.query_en == "A faithful English translation."
    assert augmented.query_en_expansion is None
    assert [variant.weight for variant in augmented.variants()] == [1.0, 1.0]


class _Q2FakeRuntime:
    def __init__(self, root: Path) -> None:
        self.output_root = root
        self.output_root.mkdir(parents=True)
        self.manifest = SimpleNamespace(fingerprint="fixture", schema_version=2)
        self.shared_encoder = SimpleNamespace(
            identifiers=MappingProxyType({"model": "ViT-B/32", "device": "cpu"})
        )
        self.config = SimpleNamespace(device="cpu")
        self.requests = []

    def handle_query(self, request) -> dict[str, Any]:
        self.requests.append(request)
        path = self.output_root / f"{request.query_id}.jsonl"
        _write_jsonl(
            path,
            [
                {
                    "query_id": request.query_id,
                    "rank": 1,
                    "video_id": "L21_FIXTURE",
                    "frame_id": 0,
                }
            ],
        )
        return {
            "status": "SUCCESS",
            "artifacts": {"top100_jsonl": path.name},
            "timings": {"total_seconds": 0.01},
        }

    def handle_qa_query(self, request):  # pragma: no cover - task is KIS only
        raise AssertionError(request)

    def handle_trake_query(self, request):  # pragma: no cover - task is KIS only
        raise AssertionError(request)


def test_arm_b_runner_records_frozen_sidecar_provenance(tmp_path: Path) -> None:
    benchmark = load_l21_150_benchmark(BENCHMARK_PATH)
    sidecar = load_kis_dev_translation_sidecar(SIDECAR_PATH, benchmark, BENCHMARK_PATH)
    runtime = _Q2FakeRuntime(tmp_path / "runtime")
    sidecar_sha = hashlib.sha256(SIDECAR_PATH.read_bytes()).hexdigest()
    report = RUNNER.run_l21_150_baseline(
        benchmark,
        runtime,
        tmp_path / "run",
        experiment_id="q2-arm-b-fixture",
        split="dev",
        task="kis",
        top_k=100,
        refine_top_n=3,
        resume=False,
        fail_fast=True,
        benchmark_sha256=hashlib.sha256(BENCHMARK_PATH.read_bytes()).hexdigest(),
        manifest_sha256="m" * 64,
        gt_policy="proposed",
        kis_query_policy="translation_augmented_rrf",
        kis_query_sidecar=sidecar,
        kis_query_sidecar_path=SIDECAR_PATH,
        kis_query_sidecar_sha256=sidecar_sha,
    )

    assert report["successful_query_count"] == 38
    assert all(len(request.variants()) == 2 for request in runtime.requests)
    assert all(request.query_en_expansion is None for request in runtime.requests)
    experiment = report["kis_query_experiment"]
    assert experiment["query_policy"] == "TRANSLATION_AUGMENTED_RRF"
    assert experiment["sidecar_sha256"] == sidecar_sha
    assert experiment["sidecar_schema_version"] == 1
    assert experiment["translation_status"] == "REVIEWED_FROZEN"
    assert experiment["variant_count_policy"] == "2_VARIANTS_VI_PLUS_EN"


def test_stage_analyzer_reports_offline_hits_without_runtime_gt(tmp_path: Path) -> None:
    run_dir, error_path = _make_stage_run(tmp_path / "run")
    report = analyze_l21_150_stages(
        _synthetic_stage_benchmark(),
        run_dir,
        error_analysis_path=error_path,
    )

    assert report["official_competition_claim"] is False
    assert report["qa"]["stages"]["SUPPORTED_QUERY"]["stage_non_empty_query_count"] == 1
    assert report["qa"]["stages"]["RETRIEVAL_FUSED"]["target_video_hit_query_count"] == 1
    assert report["qa"]["stages"]["REFINED"]["target_video_hit_query_count"] == 0
    assert report["qa"]["stages"]["USABLE_EVIDENCE"]["target_video_hit_query_count"] == 1
    assert report["trake"]["stages"]["EVENT_1_POOL"]["target_video_hit_query_count"] == 1
    assert report["trake"]["stages"]["ALL_EVENT_POOLS_CONTAIN_TARGET_VIDEO"][
        "target_video_hit_query_count"
    ] == 1
    assert report["trake"]["stages"]["C1_PLANNER"]["target_video_hit_query_count"] == 1
    relation = report["trake"]["partial_chain_vs_zero_output"]
    assert relation["status"] == "ESTABLISHED"
    assert relation["sets_equal"] is True


def test_stage_analyzer_marks_missing_old_stage_artifacts_unavailable(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "old-run"
    _write_json(
        run_dir / "experiment_manifest.json",
        {
            "queries": [
                {"query_id": "QA-STAGE", "status": "SUCCESS"},
                {"query_id": "TR-STAGE", "status": "SUCCESS"},
            ]
        },
    )
    _write_jsonl(run_dir / "predictions.jsonl", [])
    report = analyze_l21_150_stages(_synthetic_stage_benchmark(), run_dir)

    assert report["qa"]["stages"]["RETRIEVAL_FUSED"]["status"] == "UNAVAILABLE"
    assert report["trake"]["stages"]["EVENT_1_POOL"]["status"] == "UNAVAILABLE"
    assert report["qa"]["stages"]["FINAL_OUTPUT"]["status"] == "AVAILABLE"
    assert report["trake"]["partial_chain_vs_zero_output"]["status"] == (
        "NOT_ESTABLISHED"
    )


def test_partial_chain_relation_compares_ids_not_equal_counts() -> None:
    result = compare_partial_chain_and_zero_output(
        ["TR-1", "TR-2"],
        ["TR-2", "TR-3"],
    )
    assert result["sets_equal"] is False
    assert result["partial_chain_only_query_ids"] == ["TR-1"]
    assert result["zero_output_only_query_ids"] == ["TR-3"]


def _comparison_report(
    *,
    query_a_rank: int | None,
    query_b_rank: int | None,
    duplicate_count: int,
) -> dict[str, Any]:
    def row(query_id: str, rank: int | None, depth: int) -> dict[str, Any]:
        video = {
            str(cutoff): float(rank is not None and rank <= cutoff)
            for cutoff in (1, 5, 20, 50, 100)
        }
        frame = {str(cutoff): 0.0 for cutoff in (1, 5, 20, 50, 100)}
        return {
            "query_id": query_id,
            "task": "kis",
            "split": "DEV",
            "prediction_count": depth,
            "video_recall_at_k": video,
            "frame_recall_at_k": frame,
            "first_video_hit_rank": rank,
            "final_score": 0.0,
        }

    return {
        "benchmark_id": BENCHMARK_ID,
        "query_reports": [row("KIS-A", query_a_rank, 100), row("KIS-B", query_b_rank, 99)],
        "overall": {"duplicate_count": duplicate_count},
    }


def test_paired_kis_comparator_reports_rescues_regressions_and_rank_delta() -> None:
    arm_a = _comparison_report(query_a_rank=50, query_b_rank=None, duplicate_count=0)
    arm_b = _comparison_report(query_a_rank=20, query_b_rank=5, duplicate_count=1)
    report = compare_l21_150_kis_arms(arm_a, arm_b)

    assert report["arm_b"] == "TRANSLATION_AUGMENTED_RRF"
    assert report["causal_translation_claim"] is False
    assert report["target_video_hit_query_ids"]["rescued_by_arm_b"] == ["KIS-B"]
    assert report["target_video_hit_query_ids"]["regressed_in_arm_b"] == []
    assert report["first_video_hit_rank_comparisons"][0]["delta_b_minus_a"] == -30
    assert report["output_depth_distribution"]["arm_a"] == {"99": 1, "100": 1}
    assert report["duplicate_diagnostics"]["arm_b_duplicate_count"] == 1


def test_paired_kis_comparator_rejects_unpaired_queries() -> None:
    arm_a = _comparison_report(query_a_rank=1, query_b_rank=None, duplicate_count=0)
    arm_b = copy.deepcopy(arm_a)
    arm_b["query_reports"].pop()
    with pytest.raises(L21150KISComparisonError, match="paired KIS query IDs differ"):
        compare_l21_150_kis_arms(arm_a, arm_b)


def test_runtime_instrumentation_source_contains_no_benchmark_target_dependency() -> None:
    qa_source = (SYSTEM_ROOT / "src/system_tai/qa/runtime.py").read_text(encoding="utf-8")
    trake_source = (SYSTEM_ROOT / "src/system_tai/trake/runtime.py").read_text(
        encoding="utf-8"
    )
    for source in (qa_source, trake_source):
        assert "l21_150" not in source
        assert "benchmark" not in source
        assert "target_video" not in source


def test_no_q2_module_imports_network_or_changes_production_retriever() -> None:
    paths = (
        SYSTEM_ROOT / "src/system_tai/quality/l21_150_stage_analysis.py",
        SYSTEM_ROOT / "src/system_tai/quality/l21_150_translation.py",
        SYSTEM_ROOT / "src/system_tai/quality/l21_150_kis_comparison.py",
    )
    combined = "\n".join(path.read_text(encoding="utf-8") for path in paths)
    assert "requests" not in combined
    assert "urllib" not in combined
    assert "subprocess" not in combined
    assert "ExactNumpyRetriever" not in combined
    assert "WeightedRRFRetriever" not in combined
