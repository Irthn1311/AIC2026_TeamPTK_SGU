from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from system_tai.qa.grounding import QAVideoConditionedEvidenceConfig
from system_tai.quality.l21_150_qa_nomination import (
    BENCHMARK_ROLE,
    DEFAULT_QA_NOMINATION_CONFIG,
    QALanguagePolicy,
    QANominationError,
    assert_runtime_input_gt_isolated,
    build_localization_variants,
    build_nomination_inputs,
    ensure_dev_only_scope,
    evaluate_nomination_results,
    run_nomination_runtime,
)
from system_tai.quality.l21_150_qa_translation import (
    EXPECTED_QUERY_COUNT,
    QATranslationSidecarError,
    load_qa_dev_translation_sidecar,
    serialize_qa_dev_translation_sidecar,
    validate_qa_dev_translation_payload,
)
from system_tai.quality.l21_150_schema import (
    L21150QAQuery,
    load_l21_150_benchmark,
)
from system_tai.retrieval.video_evidence import (
    FullCorpusVideoMaximaOutcome,
    VideoMaximumHit,
)

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_ROOT = ROOT / "benchmarks" / "l21_150_diagnostic"
BENCHMARK_PATH = BENCHMARK_ROOT / "benchmark.json"
SIDECAR_PATH = BENCHMARK_ROOT / "qa_dev_translations_en.json"
SIDECAR_SHA256 = "45929059506de93aac574a6d2d5581691af81ae12405c18d57289485948c1f4d"


@pytest.fixture
def benchmark():
    return load_l21_150_benchmark(BENCHMARK_PATH)


@pytest.fixture
def sidecar(benchmark):
    return load_qa_dev_translation_sidecar(SIDECAR_PATH, benchmark, BENCHMARK_PATH)


def _payload() -> dict:
    return json.loads(SIDECAR_PATH.read_text(encoding="utf-8"))


def _benchmark_sha() -> str:
    return hashlib.sha256(BENCHMARK_PATH.read_bytes()).hexdigest()


def test_frozen_sidecar_is_exact_38_dev_questions_and_canonical(
    benchmark, sidecar
) -> None:
    raw = SIDECAR_PATH.read_bytes()
    dev_ids = sorted(
        query.query_id
        for query in benchmark.queries
        if isinstance(query, L21150QAQuery) and query.split == "DEV"
    )
    holdout_ids = {
        query.query_id
        for query in benchmark.queries
        if isinstance(query, L21150QAQuery) and query.split == "HOLDOUT"
    }
    assert len(sidecar.entries) == sidecar.query_count == EXPECTED_QUERY_COUNT == 38
    assert [entry.query_id for entry in sidecar.entries] == dev_ids
    assert not ({entry.query_id for entry in sidecar.entries} & holdout_ids)
    assert all(entry.question_en.strip() for entry in sidecar.entries)
    assert sidecar.official_ground_truth is False
    assert sidecar.retrieval_feedback_used is False
    assert not raw.startswith(b"\xef\xbb\xbf")
    assert serialize_qa_dev_translation_sidecar(sidecar) == raw
    assert hashlib.sha256(raw).hexdigest() == SIDECAR_SHA256


def test_sidecar_rejects_missing_query(benchmark) -> None:
    payload = _payload()
    payload["entries"].pop()
    with pytest.raises(QATranslationSidecarError, match="exactly 38"):
        validate_qa_dev_translation_payload(
            payload,
            benchmark,
            benchmark_sha256=_benchmark_sha(),
        )


@pytest.mark.parametrize("failure", ["unknown", "holdout"])
def test_sidecar_rejects_extra_or_holdout_query(benchmark, failure: str) -> None:
    payload = _payload()
    if failure == "holdout":
        replacement = next(
            query.query_id
            for query in benchmark.queries
            if isinstance(query, L21150QAQuery) and query.split == "HOLDOUT"
        )
        expected = "HOLDOUT query is forbidden"
    else:
        replacement = "QA-UNKNOWN"
        expected = "unknown DEV QA query_id"
    payload["entries"][0]["query_id"] = replacement
    with pytest.raises(QATranslationSidecarError, match=expected):
        validate_qa_dev_translation_payload(
            payload,
            benchmark,
            benchmark_sha256=_benchmark_sha(),
        )


def test_sidecar_rejects_duplicate_id(benchmark) -> None:
    payload = _payload()
    payload["entries"][1] = copy.deepcopy(payload["entries"][0])
    with pytest.raises(QATranslationSidecarError, match="duplicate query_id"):
        validate_qa_dev_translation_payload(
            payload,
            benchmark,
            benchmark_sha256=_benchmark_sha(),
        )


def test_sidecar_rejects_empty_translation(benchmark) -> None:
    payload = _payload()
    payload["entries"][0]["question_en"] = ""
    with pytest.raises(QATranslationSidecarError, match="question_en"):
        validate_qa_dev_translation_payload(
            payload,
            benchmark,
            benchmark_sha256=_benchmark_sha(),
        )


@pytest.mark.parametrize(
    ("policy", "languages", "types"),
    [
        (QALanguagePolicy.VI_ONLY, ["vi"], ["vietnamese_direct"]),
        (
            QALanguagePolicy.VI_PLUS_EN,
            ["vi", "en"],
            ["vietnamese_direct", "english_translation"],
        ),
        (QALanguagePolicy.EN_ONLY, ["en"], ["english_translation"]),
    ],
)
def test_language_policy_builds_only_true_localization_variants(
    benchmark,
    sidecar,
    policy: QALanguagePolicy,
    languages: list[str],
    types: list[str],
) -> None:
    query = build_nomination_inputs(
        benchmark,
        language_policy=policy,
        sidecar=sidecar,
    )[0]
    variants = build_localization_variants(query, language_policy=policy)
    assert [variant.language.value for variant in variants] == languages
    assert [variant.variant_type.value for variant in variants] == types
    if policy is QALanguagePolicy.EN_ONLY:
        assert query.question_en is not None
        assert len(variants) == 1
        assert variants[0].text == query.question_en
        assert "_vi" not in variants[0].variant_id


class _FakeEncoder:
    identifiers = {"library": "fake", "model": "fake", "device": "cpu"}

    def __init__(self) -> None:
        self.inputs: list[tuple[str, ...]] = []

    def encode_texts(self, texts):
        self.inputs.append(tuple(texts))
        return np.asarray([[1.0, 0.0] for _ in texts], dtype=np.float32)


class _FakeSearcher:
    def __init__(self, video_count: int = 40) -> None:
        self.registry = SimpleNamespace(stores=tuple(object() for _ in range(video_count)))
        self.calls: list[tuple[tuple[str, ...], int]] = []

    def search_video_maxima(self, *, query_ids, query_vectors):
        self.calls.append((tuple(query_ids), len(query_vectors)))
        rankings = {
            query_id: tuple(
                VideoMaximumHit(
                    query_id=query_id,
                    video_id=f"V{rank:03d}",
                    frame_id=rank * 10,
                    clip_row=rank - 1,
                    keyframe_order=rank,
                    cosine_score=1.0 / rank,
                    rank=rank,
                )
                for rank in range(1, len(self.registry.stores) + 1)
            )
            for query_id in query_ids
        }
        return FullCorpusVideoMaximaOutcome(
            rankings=rankings,
            physical_rows_scored=400,
            video_store_scan_count=len(self.registry.stores),
        )


class _DeterministicClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        self.value += 0.01
        return self.value


def _runtime_results(benchmark, sidecar, policy=QALanguagePolicy.EN_ONLY):
    inputs = build_nomination_inputs(
        benchmark,
        language_policy=policy,
        sidecar=sidecar,
    )
    encoder = _FakeEncoder()
    searcher = _FakeSearcher()
    results = run_nomination_runtime(
        inputs,
        language_policy=policy,
        encoder=encoder,
        searcher=searcher,
        clock=_DeterministicClock(),
    )
    return inputs, encoder, searcher, results


def test_nomination_runtime_is_target_agnostic_and_stops_before_evidence(
    benchmark, sidecar
) -> None:
    inputs, encoder, searcher, results = _runtime_results(benchmark, sidecar)
    assert_runtime_input_gt_isolated(inputs)
    assert set(inputs[0].__dataclass_fields__) == {
        "query_id",
        "question_vi",
        "question_en",
    }
    assert len(encoder.inputs) == len(inputs) == 38
    assert len(searcher.calls) == 38
    assert all(len(result.full_ranking) == 40 for result in results)
    assert all(len(result.capped_ranking) == 32 for result in results)
    assert DEFAULT_QA_NOMINATION_CONFIG == QAVideoConditionedEvidenceConfig(enabled=True)


def test_offline_target_metrics_do_not_change_retrieval_inputs(benchmark, sidecar) -> None:
    inputs, encoder, searcher, results = _runtime_results(benchmark, sidecar)
    encoded_before = copy.deepcopy(encoder.inputs)
    calls_before = copy.deepcopy(searcher.calls)
    targets = {query.query_id: "V001" for query in inputs}
    report = evaluate_nomination_results(
        results,
        target_video_ids=targets,
        benchmark_id=benchmark.benchmark_id,
        policy=QALanguagePolicy.EN_ONLY,
        translation_sidecar_sha256=SIDECAR_SHA256,
        manifest_sha256="m" * 64,
        corpus_fingerprint="f" * 64,
        git_sha="g" * 40,
        model_identity=_FakeEncoder.identifiers,
    )
    assert encoder.inputs == encoded_before
    assert searcher.calls == calls_before
    assert report["benchmark_role"] == BENCHMARK_ROLE
    assert report["target_video_recall_at_1"] == 1.0
    assert report["target_video_recall_at_32"] == 1.0
    assert report["mean_reciprocal_rank"] == 1.0
    assert report["top32_nomination_coverage_count"] == 38
    assert report["gt_used_in_runtime"] is False
    assert report["holdout_used"] is False
    serialized = json.dumps(report)
    assert all(
        forbidden not in serialized
        for forbidden in (
            "canonical_answer",
            "accepted_answers",
            "proposed_interval",
            "reference_timestamp",
        )
    )


def test_runtime_and_report_are_deterministic_for_equal_inputs(benchmark, sidecar) -> None:
    _, _, _, first = _runtime_results(benchmark, sidecar, QALanguagePolicy.VI_PLUS_EN)
    _, _, _, second = _runtime_results(benchmark, sidecar, QALanguagePolicy.VI_PLUS_EN)
    assert first == second


def test_dev_only_guard_rejects_holdout() -> None:
    ensure_dev_only_scope("DEV")
    with pytest.raises(QANominationError, match="HOLDOUT is forbidden"):
        ensure_dev_only_scope("HOLDOUT")
