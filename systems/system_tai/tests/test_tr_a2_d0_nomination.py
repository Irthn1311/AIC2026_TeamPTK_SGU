from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from system_tai.common.schemas import FrameMappingRecord, VideoFeatureStore
from system_tai.features.btc_clip_store import (
    FeatureStoreRegistry,
    LoadedVideoFeatureStore,
)
from system_tai.quality.l21_150_schema import (
    L21150KISQuery,
    L21150QAQuery,
    L21150TRAKEQuery,
    load_l21_150_benchmark,
)
from system_tai.quality.l21_150_trake_nomination import (
    OFFLINE_REPORT_ROLE,
    RUNTIME_ARTIFACT_ROLE,
    TRAKELanguagePolicy,
    TRAKENominationError,
    assert_runtime_gt_isolated,
    build_event_variants,
    build_nomination_inputs,
    compare_nomination_reports,
    evaluate_nomination_artifact,
    run_nomination_only,
    run_nomination_query,
)
from system_tai.quality.l21_150_trake_translation import (
    EXPECTED_EVENT_COUNT,
    EXPECTED_QUERY_COUNT,
    TRAKETranslationSidecarError,
    load_trake_dev_translation_sidecar,
    serialize_trake_dev_translation_sidecar,
    validate_trake_dev_translation_payload,
)
from system_tai.retrieval.video_evidence import (
    FullCorpusVideoMaximaOutcome,
    VideoMaximumHit,
    VideoRestrictedFeatureSearcher,
)
from system_tai.trake.video_first import (
    TRAKEVideoFirstConfig,
    build_event_video_rankings,
    nominate_videos,
)

SYSTEM_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_PATH = SYSTEM_ROOT / "benchmarks/l21_150_diagnostic/benchmark.json"
SIDECAR_PATH = (
    SYSTEM_ROOT
    / "benchmarks/l21_150_diagnostic/tr_a2_d0_trake_dev_en_translation.json"
)
RUNNER_PATH = SYSTEM_ROOT / "scripts/l21_150_run_baseline.py"
BENCHMARK_SHA = hashlib.sha256(BENCHMARK_PATH.read_bytes()).hexdigest()


def _load_runner():
    spec = importlib.util.spec_from_file_location("l21_150_runner_d0_test", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


RUNNER = _load_runner()


@pytest.fixture(scope="module")
def benchmark():
    return load_l21_150_benchmark(BENCHMARK_PATH)


@pytest.fixture(scope="module")
def sidecar(benchmark):
    return load_trake_dev_translation_sidecar(
        SIDECAR_PATH, benchmark, BENCHMARK_PATH
    )


def _sidecar_payload() -> dict[str, Any]:
    return json.loads(SIDECAR_PATH.read_text(encoding="utf-8"))


def test_sidecar_exact_schema_count_order_and_no_target_data(benchmark, sidecar) -> None:
    raw = SIDECAR_PATH.read_bytes()
    assert sidecar.query_count == EXPECTED_QUERY_COUNT == 38
    assert sidecar.event_count == EXPECTED_EVENT_COUNT == len(sidecar.records) == 114
    assert sidecar.retrieval_feedback_used is False
    assert sidecar.translation_status == "MODEL_AUTHORED_FROZEN_NOT_HUMAN_REVIEWED"
    assert not raw.startswith(b"\xef\xbb\xbf")
    assert b"\r\n" not in raw
    payload_text = raw.decode("utf-8", errors="strict")
    assert serialize_trake_dev_translation_sidecar(sidecar) == raw
    assert raw.endswith(b"\n") and not raw.endswith(b"\n\n")
    assert all(
        forbidden not in payload_text
        for forbidden in (
            "target_video_id",
            "proposed_interval",
            "proposed_frame_center",
            "reference_timestamp",
        )
    )
    assert [
        (record.query_id, record.source_event_index) for record in sidecar.records
    ] == [
        (query.query_id, event.event_index)
        for query in benchmark.queries
        if isinstance(query, L21150TRAKEQuery) and query.split == "DEV"
        for event in query.events
    ]


def test_sidecar_sha_matches_runner_frozen_constant() -> None:
    assert hashlib.sha256(SIDECAR_PATH.read_bytes()).hexdigest() == (
        RUNNER.FROZEN_TR_A2_D0_TRAKE_DEV_EN_SIDECAR_SHA256
    )


def test_sidecar_rejects_duplicate_json_key(tmp_path: Path, benchmark) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text(
        '{"schema_version":1,"schema_version":1}',
        encoding="utf-8",
    )
    with pytest.raises(TRAKETranslationSidecarError, match="duplicate JSON key"):
        load_trake_dev_translation_sidecar(path, benchmark, BENCHMARK_PATH)


def test_sidecar_rejects_duplicate_event(benchmark) -> None:
    payload = _sidecar_payload()
    payload["records"][1] = copy.deepcopy(payload["records"][0])
    with pytest.raises(TRAKETranslationSidecarError, match="duplicate query/event"):
        validate_trake_dev_translation_payload(
            payload, benchmark, benchmark_sha256=BENCHMARK_SHA
        )


def test_sidecar_rejects_missing_event(benchmark) -> None:
    payload = _sidecar_payload()
    payload["records"].pop()
    payload["event_count"] = 113
    with pytest.raises(TRAKETranslationSidecarError, match="event_count must equal"):
        validate_trake_dev_translation_payload(
            payload, benchmark, benchmark_sha256=BENCHMARK_SHA
        )


def test_sidecar_rejects_holdout_id(benchmark) -> None:
    holdout = next(
        query
        for query in benchmark.queries
        if isinstance(query, L21150TRAKEQuery) and query.split == "HOLDOUT"
    )
    payload = _sidecar_payload()
    payload["records"][0]["query_id"] = holdout.query_id
    with pytest.raises(TRAKETranslationSidecarError, match="HOLDOUT query is forbidden"):
        validate_trake_dev_translation_payload(
            payload, benchmark, benchmark_sha256=BENCHMARK_SHA
        )


def test_sidecar_rejects_source_vi_mismatch(benchmark) -> None:
    payload = _sidecar_payload()
    payload["records"][0]["source_vi"] += " changed"
    with pytest.raises(TRAKETranslationSidecarError, match="source_vi mismatch"):
        validate_trake_dev_translation_payload(
            payload, benchmark, benchmark_sha256=BENCHMARK_SHA
        )


def _first_safe_input(benchmark, sidecar, policy: TRAKELanguagePolicy):
    return build_nomination_inputs(
        benchmark,
        language_policy=policy,
        sidecar=None if policy is TRAKELanguagePolicy.VI_ONLY else sidecar,
    )[0]


def test_vi_only_variant_construction(benchmark, sidecar) -> None:
    query = _first_safe_input(benchmark, sidecar, TRAKELanguagePolicy.VI_ONLY)
    variants = build_event_variants(
        query, language_policy=TRAKELanguagePolicy.VI_ONLY
    )
    assert all(len(event_variants) == 1 for event_variants in variants.values())
    assert all(
        variant.language.value == "vi"
        for event_variants in variants.values()
        for variant in event_variants
    )


def test_en_only_variant_construction(benchmark, sidecar) -> None:
    query = _first_safe_input(benchmark, sidecar, TRAKELanguagePolicy.EN_ONLY)
    variants = build_event_variants(
        query, language_policy=TRAKELanguagePolicy.EN_ONLY
    )
    assert all(len(event_variants) == 1 for event_variants in variants.values())
    assert all(
        variant.language.value == "en"
        for event_variants in variants.values()
        for variant in event_variants
    )


def test_vi_plus_en_variant_construction_uses_equal_weights(benchmark, sidecar) -> None:
    query = _first_safe_input(
        benchmark, sidecar, TRAKELanguagePolicy.VI_PLUS_EN_WEIGHTED_RRF
    )
    variants = build_event_variants(
        query, language_policy=TRAKELanguagePolicy.VI_PLUS_EN_WEIGHTED_RRF
    )
    assert all(len(event_variants) == 2 for event_variants in variants.values())
    assert all(
        [variant.language.value for variant in event_variants] == ["vi", "en"]
        and [variant.weight for variant in event_variants] == [1.0, 1.0]
        for event_variants in variants.values()
    )


class _RegistryStub:
    stores = (object(), object())


class _RankOnlySearcher:
    registry = _RegistryStub()

    def __init__(self, cosine_scale: float = 1.0) -> None:
        self.cosine_scale = cosine_scale
        self.maxima_calls = 0
        self.selected_calls = 0

    def search_video_maxima(self, *, query_ids, query_vectors):
        self.maxima_calls += 1
        rankings = {}
        for query_id in query_ids:
            english = query_id.endswith("v2_en")
            order = ("B", "A") if english else ("A", "B")
            rankings[query_id] = tuple(
                VideoMaximumHit(
                    query_id=query_id,
                    video_id=video_id,
                    frame_id=rank,
                    clip_row=rank - 1,
                    keyframe_order=rank,
                    cosine_score=self.cosine_scale * (100.0 if rank == 2 else -100.0),
                    rank=rank,
                )
                for rank, video_id in enumerate(order, start=1)
            )
        return FullCorpusVideoMaximaOutcome(rankings, 4, 2)

    def search_selected_videos(self, *args, **kwargs):
        self.selected_calls += 1
        raise AssertionError("nomination-only must not run selected-video search")


class _ParitySearcher:
    def __init__(self, video_count: int) -> None:
        self.registry = SimpleNamespace(stores=tuple(object() for _ in range(video_count)))
        self.last_outcome: FullCorpusVideoMaximaOutcome | None = None

    def search_video_maxima(self, *, query_ids, query_vectors):
        assert len(query_ids) == len(query_vectors)
        video_ids = [f"V{index:03d}" for index in range(len(self.registry.stores))]
        rankings = {
            query_id: tuple(
                VideoMaximumHit(
                    query_id=query_id,
                    video_id=video_id,
                    frame_id=rank,
                    clip_row=rank - 1,
                    keyframe_order=rank,
                    cosine_score=1.0 / rank,
                    rank=rank,
                )
                for rank, video_id in enumerate(video_ids, start=1)
            )
            for query_id in query_ids
        }
        self.last_outcome = FullCorpusVideoMaximaOutcome(
            rankings,
            len(video_ids),
            len(video_ids),
        )
        return self.last_outcome


class _ConstantEncoder:
    identifiers = {"model": "fake", "device": "cpu"}

    def __init__(self) -> None:
        self.calls = 0

    def encode_texts(self, texts):
        self.calls += 1
        return np.asarray([[1.0, 0.0] for _ in texts], dtype=np.float32)


def test_vi_plus_en_uses_rank_fusion_not_raw_cosine(benchmark, sidecar) -> None:
    query = _first_safe_input(
        benchmark, sidecar, TRAKELanguagePolicy.VI_PLUS_EN_WEIGHTED_RRF
    )
    first = run_nomination_query(
        query,
        language_policy=TRAKELanguagePolicy.VI_PLUS_EN_WEIGHTED_RRF,
        encoder=_ConstantEncoder(),
        searcher=_RankOnlySearcher(cosine_scale=1.0),
    )
    second = run_nomination_query(
        query,
        language_policy=TRAKELanguagePolicy.VI_PLUS_EN_WEIGHTED_RRF,
        encoder=_ConstantEncoder(),
        searcher=_RankOnlySearcher(cosine_scale=-50.0),
    )
    assert [row["video_id"] for row in first["nomination_ranking"]] == ["A", "B"]
    assert first["nomination_ranking"] == second["nomination_ranking"]


def test_nomination_only_skips_restricted_search_planner_and_refiner(
    benchmark, sidecar
) -> None:
    query = _first_safe_input(benchmark, sidecar, TRAKELanguagePolicy.EN_ONLY)
    searcher = _RankOnlySearcher()
    artifact = run_nomination_query(
        query,
        language_policy=TRAKELanguagePolicy.EN_ONLY,
        encoder=_ConstantEncoder(),
        searcher=searcher,
    )
    assert searcher.maxima_calls == 1
    assert searcher.selected_calls == 0
    assert len(artifact["nomination_ranking"]) == 2
    assert all(
        term not in json.dumps(artifact)
        for term in ("planner", "refinement", "restricted_event")
    )


def test_runtime_input_and_artifact_are_target_agnostic(benchmark, sidecar) -> None:
    query = _first_safe_input(benchmark, sidecar, TRAKELanguagePolicy.EN_ONLY)
    assert set(query.__dataclass_fields__).isdisjoint(
        {"video_id", "target_video_id", "interval", "frame_center"}
    )
    artifact = run_nomination_query(
        query,
        language_policy=TRAKELanguagePolicy.EN_ONLY,
        encoder=_ConstantEncoder(),
        searcher=_RankOnlySearcher(),
    )
    assert_runtime_gt_isolated(artifact)
    assert "target_video_id" not in json.dumps(artifact)


def test_complete_video_ranking_is_contiguous_and_deterministic(benchmark, sidecar) -> None:
    query = _first_safe_input(benchmark, sidecar, TRAKELanguagePolicy.VI_ONLY)
    first = run_nomination_query(
        query,
        language_policy=TRAKELanguagePolicy.VI_ONLY,
        encoder=_ConstantEncoder(),
        searcher=_RankOnlySearcher(),
    )
    second = run_nomination_query(
        query,
        language_policy=TRAKELanguagePolicy.VI_ONLY,
        encoder=_ConstantEncoder(),
        searcher=_RankOnlySearcher(),
    )
    assert first["nomination_ranking"] == second["nomination_ranking"]
    assert [row["rank"] for row in first["nomination_ranking"]] == [1, 2]


def test_d0_vi_only_top32_exactly_matches_normal_tr_a1_and_keeps_depth_100(
    benchmark, sidecar
) -> None:
    query = _first_safe_input(benchmark, sidecar, TRAKELanguagePolicy.VI_ONLY)
    searcher = _ParitySearcher(120)
    d0 = run_nomination_query(
        query,
        language_policy=TRAKELanguagePolicy.VI_ONLY,
        encoder=_ConstantEncoder(),
        searcher=searcher,
        event_video_nomination_depth=100,
    )
    assert searcher.last_outcome is not None
    variants = build_event_variants(
        query, language_policy=TRAKELanguagePolicy.VI_ONLY
    )
    event_rankings = build_event_video_rankings(
        event_variants=variants,
        maxima=searcher.last_outcome,
        rrf_constant=60.0,
    )
    normal_top32 = nominate_videos(
        event_video_rankings=event_rankings,
        config=TRAKEVideoFirstConfig(
            enabled=True,
            selected_video_cap=32,
            event_video_nomination_depth=100,
        ),
        rrf_constant=60.0,
    )
    assert len(d0["nomination_ranking"]) == 120
    for rank, (d0_record, normal_record) in enumerate(
        zip(d0["nomination_ranking"][:32], normal_top32), start=1
    ):
        assert d0_record["rank"] == rank
        assert d0_record["video_id"] == normal_record.video_id
        assert d0_record["coverage_count"] == normal_record.coverage_count
        assert d0_record["worst_event_rank"] == normal_record.worst_event_rank
        assert d0_record["reciprocal_event_rank_sum"] == (
            normal_record.reciprocal_event_rank_sum
        )
        assert d0_record["best_event_rank"] == normal_record.best_event_rank
        assert tuple(
            event["event_video_rank"] for event in d0_record["per_event"]
        ) == normal_record.event_video_ranks
    rank_101 = next(
        record for record in d0["nomination_ranking"] if record["video_id"] == "V100"
    )
    assert rank_101["coverage_count"] == 0
    assert [event["event_video_rank"] for event in rank_101["per_event"]] == [
        101,
        101,
        101,
    ]


def _offline_artifact(benchmark, ranks: list[int], policy: str) -> dict[str, Any]:
    dev = [
        query
        for query in benchmark.queries
        if isinstance(query, L21150TRAKEQuery) and query.split == "DEV"
    ]
    queries = []
    for query, target_rank in zip(dev, ranks):
        video_ids = [f"DUMMY_{index:03d}" for index in range(1, 151)]
        video_ids[target_rank - 1] = query.video_id
        queries.append(
            {
                "query_id": query.query_id,
                "nomination_ranking": [
                    {
                        "rank": rank,
                        "video_id": video_id,
                        "per_event": [
                            {
                                "source_event_index": event_index,
                                "event_video_rank": min(target_rank + event_index, 150),
                            }
                            for event_index in (1, 2, 3)
                        ],
                    }
                    for rank, video_id in enumerate(video_ids, start=1)
                ],
            }
        )
    return {
        "artifact_role": RUNTIME_ARTIFACT_ROLE,
        "experiment_id": policy,
        "language_policy": policy,
        "queries": queries,
    }


def test_offline_recall_target_buckets_and_percentiles(benchmark) -> None:
    ranks = [10] * 10 + [40] * 10 + [75] * 10 + [150] * 8
    report = evaluate_nomination_artifact(
        _offline_artifact(benchmark, ranks, "vi_only"), benchmark
    )
    assert report["GT_USED_OFFLINE_ONLY"] is True
    assert report["GT_USED_IN_RUNTIME"] is False
    assert report["recall_at_32"] == 10 / 38
    assert report["recall_at_50"] == 20 / 38
    assert report["recall_at_100"] == 30 / 38
    assert report["target_rank_buckets"] == {
        "rank_1_32": 10,
        "rank_33_50": 10,
        "rank_51_100": 10,
        "rank_over_100": 8,
    }
    assert report["median_target_rank"] == 40.0
    assert report["p75_target_rank_nearest_rank"] == 75
    assert report["p90_target_rank_nearest_rank"] == 150
    assert report["worst_target_rank"] == 150


def test_offline_per_event_target_ranks_and_all_at_100_count(benchmark) -> None:
    ranks = [10] * 37 + [150]
    report = evaluate_nomination_artifact(
        _offline_artifact(benchmark, ranks, "en_only"), benchmark
    )
    assert report["all_event_video_ranks_at_most_100_count"] == 37
    assert report["query_reports"][0]["per_event_target_video_ranks"] == [11, 12, 13]


def test_three_arm_comparator_language_and_cap_decisions(benchmark) -> None:
    rank_sets = {
        "vi_only": [120] * 38,
        "vi_plus_en_weighted_rrf": [40] * 38,
        "en_only": [20] * 38,
    }
    reports = {
        policy: evaluate_nomination_artifact(
            _offline_artifact(benchmark, ranks, policy), benchmark
        )
        for policy, ranks in rank_sets.items()
    }
    comparison = compare_nomination_reports(reports)
    assert comparison["language_decision"] == "LANGUAGE_SIGNAL_EN_ONLY"
    assert comparison["best_arms"]["recall_at_32"] == ["en_only"]
    assert comparison["arms"]["vi_plus_en_weighted_rrf"]["cap_decision"] == (
        "CAP50_HAS_MATERIAL_OPPORTUNITY"
    )
    delta = comparison["per_query_rank_deltas"][0]
    assert delta["en_only_minus_vi_rank_delta"] == -100


def test_run_nomination_only_batches_queries_without_gt(benchmark, sidecar) -> None:
    queries = build_nomination_inputs(
        benchmark,
        language_policy=TRAKELanguagePolicy.VI_ONLY,
        sidecar=None,
    )[:2]
    encoder = _ConstantEncoder()
    searcher = _RankOnlySearcher()
    artifact = run_nomination_only(
        queries,
        benchmark_id=benchmark.benchmark_id,
        experiment_id="fixture",
        git_sha="a" * 40,
        corpus_fingerprint="f" * 64,
        model_identity=encoder.identifiers,
        language_policy=TRAKELanguagePolicy.VI_ONLY,
        translation_sidecar_sha256=None,
        translation_status=None,
        encoder=encoder,
        searcher=searcher,
    )
    assert artifact["query_count"] == 2
    assert artifact["video_count"] == 2
    assert artifact["store_scan_count"] == 4
    assert encoder.calls == 2
    assert_runtime_gt_isolated(artifact)


def _store(video_id: str, vector: tuple[float, float]) -> LoadedVideoFeatureStore:
    mapping = FrameMappingRecord(0, 1, 0, 0.0, 30.0)
    return LoadedVideoFeatureStore(
        descriptor=VideoFeatureStore(
            video_id,
            Path(f"{video_id}.csv"),
            Path(f"{video_id}.npy"),
            1,
            2,
            True,
        ),
        matrix=np.asarray([vector], dtype=np.float32),
        mappings=(mapping,),
    )


class _RunnerRuntime:
    def __init__(self) -> None:
        registry = FeatureStoreRegistry([_store("A", (1.0, 0.0)), _store("B", (0.0, 1.0))])
        self.video_restricted_searcher = VideoRestrictedFeatureSearcher(
            registry, chunk_size=1
        )
        self.shared_encoder = _ConstantEncoder()
        self.manifest = SimpleNamespace(fingerprint="fixture")

    def handle_query(self, request):
        return {"task": "kis", "request": request}

    def handle_qa_query(self, request):
        return {"task": "qa", "request": request}

    def handle_trake_query(self, request):
        raise AssertionError("normal TRAKE handler must not run in nomination-only mode")


def test_runner_nomination_only_writes_target_free_artifact(
    tmp_path: Path, benchmark
) -> None:
    runtime = _RunnerRuntime()
    artifact = RUNNER.run_trake_nomination_diagnostic(
        benchmark,
        runtime,
        tmp_path,
        experiment_id="d0-fixture",
        language_policy="vi_only",
        sidecar=None,
        sidecar_sha256=None,
    )
    assert artifact["query_count"] == 38
    assert (tmp_path / "trake_nomination_rankings.json").exists()
    assert (tmp_path / "experiment_manifest.json").exists()
    assert (tmp_path / "run_summary.md").exists()
    assert "target_video_id" not in (tmp_path / "trake_nomination_rankings.json").read_text(
        encoding="utf-8"
    )


def test_runner_defaults_preserve_tr_a1_kis_and_qa_paths(benchmark) -> None:
    args = RUNNER.build_parser().parse_args(
        [
            "--benchmark",
            str(BENCHMARK_PATH),
            "--reuse-manifest",
            "manifest.json",
            "--output-dir",
            "out",
        ]
    )
    assert args.trake_nomination_only is False
    assert args.trake_language_policy == "vi_only"
    assert args.trake_video_first_restricted_search is False

    runtime = _RunnerRuntime()
    kis = next(query for query in benchmark.queries if isinstance(query, L21150KISQuery))
    qa = next(query for query in benchmark.queries if isinstance(query, L21150QAQuery))
    kis_request = RUNNER._runtime_request(kis, "x", 100, 3)
    qa_request = RUNNER._runtime_request(qa, "x", 100, 3)
    assert RUNNER._run_request(runtime, kis, kis_request)["task"] == "kis"
    assert RUNNER._run_request(runtime, qa, qa_request)["task"] == "qa"


def test_offline_report_role_is_explicit(benchmark) -> None:
    report = evaluate_nomination_artifact(
        _offline_artifact(benchmark, [1] * 38, "vi_only"), benchmark
    )
    assert report["artifact_role"] == OFFLINE_REPORT_ROLE
    assert all("target_video_id" in row for row in report["query_reports"])


def test_runtime_gt_guard_rejects_target_field() -> None:
    with pytest.raises(TRAKENominationError, match="forbidden GT fields"):
        assert_runtime_gt_isolated({"target_video_id": "L21_V001"})
