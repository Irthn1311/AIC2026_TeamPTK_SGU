from __future__ import annotations

import copy
import dataclasses
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any

import pytest

from system_tai.kis.session_schema import InvalidRequestError, QueryRequest, parse_session_request
from system_tai.quality.l21_150_kis_abc_comparison import (
    L21150KISABCComparisonError,
    compare_l21_150_kis_abc_arms,
)
from system_tai.quality.l21_150_schema import (
    BENCHMARK_ID,
    L21150KISQuery,
    load_l21_150_benchmark,
)
from system_tai.quality.l21_150_translation import load_kis_dev_translation_sidecar
from system_tai.retrieval.multi_query import QueryLanguage, QueryVariantType

SYSTEM_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_PATH = SYSTEM_ROOT / "benchmarks/l21_150_diagnostic/benchmark.json"
SIDECAR_PATH = (
    SYSTEM_ROOT / "benchmarks/l21_150_diagnostic/q2_kis_dev_en_translation.json"
)
RUNNER_PATH = SYSTEM_ROOT / "scripts/l21_150_run_baseline.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("l21_150_runner_arm_c_test", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


RUNNER = _load_runner()


def _first_dev_kis():
    benchmark = load_l21_150_benchmark(BENCHMARK_PATH)
    query = next(
        query
        for query in benchmark.queries
        if isinstance(query, L21150KISQuery) and query.split == "DEV"
    )
    return benchmark, query


def test_query_request_default_and_vi_en_variants_remain_backward_compatible() -> None:
    default = QueryRequest("r-a", "KIS-A", "vietnamese text")
    default_variants = default.variants()
    assert len(default_variants) == 1
    assert default_variants[0].variant_id == "KIS-A::v1_vi"
    assert default_variants[0].language is QueryLanguage.VIETNAMESE
    assert default_variants[0].variant_type is QueryVariantType.VIETNAMESE_DIRECT
    assert default_variants[0].weight == 1.0

    augmented = QueryRequest(
        "r-b",
        "KIS-B",
        "vietnamese text",
        query_en="english text",
        weight_vi=1.0,
        weight_en=1.0,
    )
    variants = augmented.variants()
    assert [variant.variant_id for variant in variants] == [
        "KIS-B::v1_vi",
        "KIS-B::v2_en",
    ]
    assert [variant.language for variant in variants] == [
        QueryLanguage.VIETNAMESE,
        QueryLanguage.ENGLISH,
    ]
    assert [variant.variant_type for variant in variants] == [
        QueryVariantType.VIETNAMESE_DIRECT,
        QueryVariantType.ENGLISH_TRANSLATION,
    ]
    assert [variant.weight for variant in variants] == [1.0, 1.0]


def test_query_request_en_only_is_one_semantically_correct_variant() -> None:
    request = QueryRequest(
        "r-c",
        "KIS-C",
        "retained Vietnamese source",
        query_en="frozen English translation",
        include_vi_variant=False,
    )
    assert request.query_vi == "retained Vietnamese source"
    assert request.query_en_expansion is None
    assert request.include_vi_variant is False
    assert len(request.variants()) == 1
    variant = request.variants()[0]
    assert variant.variant_id == "KIS-C::v2_en"
    assert variant.text == "frozen English translation"
    assert variant.language is QueryLanguage.ENGLISH
    assert variant.variant_type is QueryVariantType.ENGLISH_TRANSLATION
    assert variant.weight == 1.0


def test_query_request_en_only_without_translation_is_rejected() -> None:
    with pytest.raises(ValueError, match="query_en must be non-empty"):
        QueryRequest(
            "r-c",
            "KIS-C",
            "retained Vietnamese source",
            include_vi_variant=False,
        )
    with pytest.raises(InvalidRequestError, match="query_en must be non-empty"):
        parse_session_request(
            json.dumps(
                {
                    "type": "query",
                    "request_id": "r-c",
                    "query_id": "KIS-C",
                    "query_vi": "retained Vietnamese source",
                    "include_vi_variant": False,
                }
            )
        )


def test_session_parser_en_only_opt_in_preserves_default_behavior() -> None:
    default = parse_session_request(
        '{"type":"query","request_id":"a","query_id":"q","query_vi":"vi"}'
    )
    en_only = parse_session_request(
        '{"type":"query","request_id":"c","query_id":"q","query_vi":"vi",'
        '"query_en":"en","include_vi_variant":false}'
    )
    assert isinstance(default, QueryRequest) and default.include_vi_variant is True
    assert isinstance(en_only, QueryRequest) and en_only.include_vi_variant is False
    assert len(en_only.variants()) == 1


def test_runner_builds_distinct_arm_a_b_c_requests_without_gt_fields() -> None:
    _, query = _first_dev_kis()
    translation = "The frozen reviewed English translation."
    arm_a = RUNNER._runtime_request(query, "a", 100, 3)
    arm_b = RUNNER._runtime_request(
        query,
        "b",
        100,
        3,
        kis_query_policy="translation_augmented_rrf",
        kis_translations={query.query_id: translation},
    )
    arm_c = RUNNER._runtime_request(
        query,
        "c",
        100,
        3,
        kis_query_policy="en_only",
        kis_translations={query.query_id: translation},
    )

    assert [variant.text for variant in arm_a.variants()] == [query.query_vi]
    assert [variant.text for variant in arm_b.variants()] == [query.query_vi, translation]
    assert [variant.text for variant in arm_c.variants()] == [translation]
    assert arm_c.query_vi == query.query_vi
    assert arm_c.query_en_expansion is None
    assert arm_c.variants()[0].language is QueryLanguage.ENGLISH
    assert arm_c.variants()[0].variant_type is QueryVariantType.ENGLISH_TRANSLATION
    request_fields = {field.name for field in dataclasses.fields(arm_c)}
    assert request_fields.isdisjoint(
        {"video_id", "frame_id", "answer", "timestamp", "ground_truth"}
    )


class _FakeRuntime:
    def __init__(self, root: Path) -> None:
        self.output_root = root
        self.output_root.mkdir(parents=True)
        self.manifest = SimpleNamespace(fingerprint="fixture", schema_version=2)
        self.shared_encoder = SimpleNamespace(
            identifiers=MappingProxyType({"model": "ViT-B/32", "device": "cpu"})
        )
        self.config = SimpleNamespace(device="cpu")
        self.requests: list[QueryRequest] = []

    def handle_query(self, request: QueryRequest) -> dict[str, Any]:
        self.requests.append(request)
        path = self.output_root / f"{request.query_id}.jsonl"
        path.write_text(
            json.dumps(
                {
                    "query_id": request.query_id,
                    "rank": 1,
                    "video_id": "L21_FIXTURE",
                    "frame_id": 0,
                }
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
        return {
            "status": "SUCCESS",
            "artifacts": {"top100_jsonl": path.name},
            "timings": {"total_seconds": 0.01},
        }

    def handle_qa_query(self, request):  # pragma: no cover
        raise AssertionError(request)

    def handle_trake_query(self, request):  # pragma: no cover
        raise AssertionError(request)


@pytest.mark.parametrize(
    ("policy", "split", "task", "with_sidecar", "message"),
    (
        ("vi_only", "dev", "kis", True, "not valid with vi_only"),
        (
            "translation_augmented_rrf",
            "dev",
            "kis",
            False,
            "requires a validated",
        ),
        ("en_only", "dev", "kis", False, "requires a validated"),
        ("en_only", "holdout", "kis", True, "restricted to the KIS DEV"),
        ("en_only", "all", "kis", True, "restricted to the KIS DEV"),
        ("en_only", "dev", "qa", True, "restricted to the KIS DEV"),
        ("en_only", "dev", "trake", True, "restricted to the KIS DEV"),
        ("en_only", "dev", "all", True, "restricted to the KIS DEV"),
    ),
)
def test_runner_policy_guardrails(
    tmp_path: Path,
    policy: str,
    split: str,
    task: str,
    with_sidecar: bool,
    message: str,
) -> None:
    benchmark = load_l21_150_benchmark(BENCHMARK_PATH)
    sidecar = load_kis_dev_translation_sidecar(SIDECAR_PATH, benchmark, BENCHMARK_PATH)
    with pytest.raises(ValueError, match=message):
        RUNNER.run_l21_150_baseline(
            benchmark,
            _FakeRuntime(tmp_path / "runtime"),
            tmp_path / "run",
            experiment_id="guard",
            split=split,
            task=task,
            top_k=100,
            refine_top_n=3,
            resume=False,
            fail_fast=True,
            benchmark_sha256=hashlib.sha256(BENCHMARK_PATH.read_bytes()).hexdigest(),
            manifest_sha256=None,
            gt_policy="proposed",
            kis_query_policy=policy,
            kis_query_sidecar=sidecar if with_sidecar else None,
        )


def test_arm_c_dev_runner_uses_all_and_only_frozen_english_records(
    tmp_path: Path,
) -> None:
    benchmark = load_l21_150_benchmark(BENCHMARK_PATH)
    sidecar = load_kis_dev_translation_sidecar(SIDECAR_PATH, benchmark, BENCHMARK_PATH)
    sidecar_sha = hashlib.sha256(SIDECAR_PATH.read_bytes()).hexdigest()
    runtime = _FakeRuntime(tmp_path / "runtime")
    report = RUNNER.run_l21_150_baseline(
        benchmark,
        runtime,
        tmp_path / "run",
        experiment_id="q2-arm-c-fixture",
        split="dev",
        task="kis",
        top_k=100,
        refine_top_n=3,
        resume=False,
        fail_fast=True,
        benchmark_sha256=hashlib.sha256(BENCHMARK_PATH.read_bytes()).hexdigest(),
        manifest_sha256="m" * 64,
        gt_policy="proposed",
        kis_query_policy="en_only",
        kis_query_sidecar=sidecar,
        kis_query_sidecar_path=SIDECAR_PATH,
        kis_query_sidecar_sha256=sidecar_sha,
    )

    assert sidecar.query_count == 38
    assert report["successful_query_count"] == 38
    assert {request.query_id for request in runtime.requests} == set(sidecar.translations)
    assert all(request.include_vi_variant is False for request in runtime.requests)
    assert all(request.query_en_expansion is None for request in runtime.requests)
    assert all(len(request.variants()) == 1 for request in runtime.requests)
    assert all(
        request.variants()[0].text == sidecar.translations[request.query_id]
        for request in runtime.requests
    )
    assert all(
        request.variants()[0].language is QueryLanguage.ENGLISH
        and request.variants()[0].variant_type is QueryVariantType.ENGLISH_TRANSLATION
        for request in runtime.requests
    )
    holdout_ids = {
        query.query_id
        for query in benchmark.queries
        if isinstance(query, L21150KISQuery) and query.split == "HOLDOUT"
    }
    assert holdout_ids.isdisjoint({request.query_id for request in runtime.requests})

    assert report["production_algorithm_modified"] is False
    assert report["core_production_algorithm_modified"] is False
    assert report["production_algorithm_modified_scope"] == (
        "CORE_PRODUCTION_IMPLEMENTATION"
    )
    assert report["kis_query_policy"] == "EN_ONLY"
    assert report["query_policy_changed_from_e0"] is True
    experiment = report["kis_query_experiment"]
    assert experiment == {
        "query_policy": "EN_ONLY",
        "sidecar_path": str(SIDECAR_PATH),
        "sidecar_basename": SIDECAR_PATH.name,
        "sidecar_sha256": sidecar_sha,
        "sidecar_schema_version": 1,
        "translation_status": "REVIEWED_FROZEN",
        "query_en_expansion_used": False,
        "variant_count_policy": "1_VARIANT_EN_ONLY",
        "variant_weights": {"en": 1.0},
        "source_vi_used_for_retrieval": False,
        "translation_en_used_for_retrieval": True,
    }
    summary = (tmp_path / "run/run_summary.md").read_text(encoding="utf-8")
    assert "KIS query policy: `EN_ONLY`" in summary
    assert "Query policy changed from E0: `true`" in summary
    assert "English translation input: `REVIEWED_FROZEN`" in summary
    assert "used for retrieval: `false`" in summary
    assert "Causal or official accuracy claim: `false`" in summary


def test_arm_c_rejects_non_frozen_sidecar_sha(tmp_path: Path) -> None:
    benchmark = load_l21_150_benchmark(BENCHMARK_PATH)
    sidecar = load_kis_dev_translation_sidecar(SIDECAR_PATH, benchmark, BENCHMARK_PATH)
    with pytest.raises(ValueError, match="does not match the frozen Q2 DEV artifact"):
        RUNNER.run_l21_150_baseline(
            benchmark,
            _FakeRuntime(tmp_path / "runtime"),
            tmp_path / "run",
            experiment_id="q2-arm-c-wrong-sha",
            split="dev",
            task="kis",
            top_k=100,
            refine_top_n=3,
            resume=False,
            fail_fast=True,
            benchmark_sha256=hashlib.sha256(BENCHMARK_PATH.read_bytes()).hexdigest(),
            manifest_sha256=None,
            gt_policy="proposed",
            kis_query_policy="en_only",
            kis_query_sidecar=sidecar,
            kis_query_sidecar_path=SIDECAR_PATH,
            kis_query_sidecar_sha256="0" * 64,
        )


def _abc_report(
    ranks: dict[str, int | None],
    *,
    benchmark_id: str = BENCHMARK_ID,
    duplicate_count: int = 0,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for index in range(1, 39):
        query_id = f"KIS-{index:03d}"
        rank = ranks.get(query_id)
        video = {
            str(cutoff): float(rank is not None and rank <= cutoff)
            for cutoff in (1, 5, 20, 50, 100)
        }
        rows.append(
            {
                "query_id": query_id,
                "task": "kis",
                "split": "DEV",
                "prediction_count": 100,
                "video_recall_at_k": video,
                "frame_recall_at_k": dict(video),
                "first_video_hit_rank": rank,
                "final_score": sum(video.values()) / 5.0,
            }
        )
    return {
        "benchmark_id": benchmark_id,
        "query_reports": rows,
        "overall": {"duplicate_count": duplicate_count},
    }


def test_abc_comparator_metrics_sets_and_rank_deltas() -> None:
    arm_a = _abc_report({"KIS-001": 10, "KIS-004": 20})
    arm_b = _abc_report({"KIS-001": 2, "KIS-002": 5, "KIS-004": 5}, duplicate_count=1)
    arm_c = _abc_report({"KIS-001": 4, "KIS-003": 3, "KIS-004": 5}, duplicate_count=2)
    report = compare_l21_150_kis_abc_arms(arm_a, arm_b, arm_c)

    assert report["comparison_role"] == "PAIRED_KIS_DEV_ABC_ABLATION"
    assert report["paired_query_count"] == 38
    assert report["semantic_gt_authority"] == "SOURCE_PROPOSED_INTERNAL"
    assert report["official_competition_claim"] is False
    assert report["causal_translation_claim"] is False
    assert report["holdout_used"] is False
    recall_100 = report["video_recall_at_k"]["100"]
    assert recall_100["arm_a"] == pytest.approx(2 / 38)
    assert recall_100["arm_b"] == pytest.approx(3 / 38)
    assert recall_100["arm_c"] == pytest.approx(3 / 38)
    assert recall_100["delta_c_minus_b"] == pytest.approx(0.0)
    hit_ids = report["target_video_hit_query_ids"]
    assert hit_ids["b_rescued_vs_a"] == ["KIS-002"]
    assert hit_ids["b_regressed_vs_a"] == []
    assert hit_ids["c_rescued_vs_a"] == ["KIS-003"]
    assert hit_ids["c_regressed_vs_a"] == []
    assert hit_ids["c_unique_vs_b"] == ["KIS-003"]
    assert hit_ids["b_unique_vs_c"] == ["KIS-002"]
    assert hit_ids["shared_b_c_hits"] == ["KIS-001", "KIS-004"]
    rank_rows = report["first_target_video_hit_rank_comparisons"]
    assert rank_rows[0] == {
        "query_id": "KIS-001",
        "arm_a_first_video_hit_rank": 10,
        "arm_b_first_video_hit_rank": 2,
        "arm_c_first_video_hit_rank": 4,
        "delta_c_minus_b": 2,
    }
    assert report["duplicate_diagnostics"] == {
        "arm_a_duplicate_count": 0,
        "arm_b_duplicate_count": 1,
        "arm_c_duplicate_count": 2,
    }


@pytest.mark.parametrize(
    ("corruption", "message"),
    (
        ("benchmark", "benchmark_id mismatch"),
        ("query_ids", "paired KIS query IDs differ"),
        ("non_dev", "restricted to KIS DEV"),
        ("non_kis", "accepts KIS query reports only"),
        ("duplicate", "duplicate KIS query report"),
    ),
)
def test_abc_comparator_rejects_unpaired_or_malformed_reports(
    corruption: str,
    message: str,
) -> None:
    arm_a = _abc_report({})
    arm_b = _abc_report({})
    arm_c = _abc_report({})
    if corruption == "benchmark":
        arm_c["benchmark_id"] = "different"
    elif corruption == "query_ids":
        arm_c["query_reports"][0]["query_id"] = "KIS-DIFFERENT"
    elif corruption == "non_dev":
        arm_c["query_reports"][0]["split"] = "HOLDOUT"
    elif corruption == "non_kis":
        arm_c["query_reports"][0]["task"] = "qa"
    else:
        arm_c["query_reports"][1] = copy.deepcopy(arm_c["query_reports"][0])
    with pytest.raises(L21150KISABCComparisonError, match=message):
        compare_l21_150_kis_abc_arms(arm_a, arm_b, arm_c)
