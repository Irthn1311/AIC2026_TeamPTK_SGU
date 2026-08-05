from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest
import yaml

from system_tai.common.schemas import KISQuery
from system_tai.evaluation.annotation import (
    build_annotation_candidates,
    write_draft_annotation_review,
)
from system_tai.evaluation.benchmark_schema import (
    AnnotationStatus,
    BenchmarkLanguage,
    BenchmarkQuery,
    KISBenchmark,
    RelevantFrame,
    VariantType,
    load_benchmark,
    parse_benchmark_payload,
)
from system_tai.evaluation.benchmark_validator import BenchmarkValidator
from system_tai.evaluation.kis_benchmark import (
    KISBenchmarkEvaluator,
    NoVerifiedQueriesResult,
)
from system_tai.evaluation.reports import write_benchmark_reports
from system_tai.features.btc_clip_store import FeatureStoreRegistry
from system_tai.kis.benchmark import main as benchmark_main
from system_tai.retrieval.vector_search import ExactNumpyRetriever
from tests.phase2_helpers import make_store


class FakeBenchmarkEncoder:
    dimension = 3
    identifiers = MappingProxyType(
        {
            "library": "deterministic-test-fake",
            "model": "fixture-v1",
            "device": "cpu",
        }
    )

    def __init__(self, vectors: dict[str, tuple[float, float, float]]) -> None:
        self.vectors = vectors

    def encode(self, text: str) -> np.ndarray:
        return np.asarray(self.vectors[text], dtype=np.float32)


def _registry() -> FeatureStoreRegistry:
    return FeatureStoreRegistry(
        [
            make_store(
                "A_VIDEO",
                np.asarray(
                    [[1.0, 0.0, 0.0], [0.8, 0.6, 0.0], [0.0, 1.0, 0.0]],
                    dtype=np.float32,
                ),
                [10, 20, 30],
            ),
            make_store(
                "B_VIDEO",
                np.asarray(
                    [[0.9, 0.4358899, 0.0], [-1.0, 0.0, 0.0]],
                    dtype=np.float32,
                ),
                [40, 50],
            ),
        ]
    )


def _query(
    query_id: str,
    *,
    language: BenchmarkLanguage = BenchmarkLanguage.VIETNAMESE,
    variant_type: VariantType = VariantType.VIETNAMESE_DIRECT,
    status: AnnotationStatus = AnnotationStatus.VERIFIED,
    relevant_frames: tuple[RelevantFrame, ...] = (
        RelevantFrame("A_VIDEO", 30),
        RelevantFrame("B_VIDEO", 40),
    ),
    relevant_video_ids: tuple[str, ...] = ("A_VIDEO", "B_VIDEO"),
    group: str = "group-1",
    text: str = "vi query",
) -> BenchmarkQuery:
    return BenchmarkQuery(
        query_id=query_id,
        language=language,
        text=text,
        semantic_group_id=group,
        variant_type=variant_type,
        relevant_frames=relevant_frames,
        relevant_video_ids=relevant_video_ids,
        annotation_notes=(
            "human verified fixture"
            if status is AnnotationStatus.VERIFIED
            else "draft"
        ),
        annotation_status=status,
        source_scope=("A_VIDEO", "B_VIDEO"),
    )


def _benchmark(*queries: BenchmarkQuery) -> KISBenchmark:
    return KISBenchmark(
        schema_version=1,
        benchmark_id="synthetic-benchmark",
        description="synthetic test only",
        queries=queries,
    )


def _retriever() -> ExactNumpyRetriever:
    return ExactNumpyRetriever(
        _registry(),
        FakeBenchmarkEncoder(
            {
                "vi query": (1.0, 0.0, 0.0),
                "en query": (0.0, 1.0, 0.0),
                "en negative query": (-1.0, 0.0, 0.0),
                "draft query": (1.0, 0.0, 0.0),
                "miss query": (1.0, 0.0, 0.0),
            }
        ),
        chunk_size=2,
    )


def test_schema_parsing_yaml_and_invalid_frame_id(tmp_path: Path) -> None:
    payload = {
        "schema_version": 1,
        "benchmark_id": "schema-test",
        "description": "schema",
        "queries": [
            {
                "query_id": "q1",
                "language": "vi",
                "text": "một truy vấn",
                "semantic_group_id": "g1",
                "variant_type": "vietnamese_direct",
                "relevant_frames": [{"video_id": "A_VIDEO", "frame_id": 10}],
                "relevant_video_ids": ["A_VIDEO"],
                "annotation_notes": "verified by a human",
                "annotation_status": "verified",
                "source_scope": ["A_VIDEO"],
            }
        ],
    }
    path = tmp_path / "benchmark.yaml"
    path.write_text(yaml.safe_dump(payload, allow_unicode=True), encoding="utf-8")
    parsed = load_benchmark(path)
    assert parsed.valid
    assert parsed.benchmark is not None
    assert parsed.benchmark.queries[0].relevant_frames[0].frame_id == 10

    payload["queries"][0]["relevant_frames"][0]["frame_id"] = -1
    invalid = parse_benchmark_payload(payload)
    assert not invalid.valid
    assert {issue.code for issue in invalid.errors} == {"INVALID_RELEVANT_FRAME_ID"}


def test_validator_duplicate_ids_pairs_and_absent_mapping() -> None:
    duplicate_pairs = (
        RelevantFrame("A_VIDEO", 10),
        RelevantFrame("A_VIDEO", 10),
        RelevantFrame("A_VIDEO", 999),
    )
    first = _query(
        "duplicate",
        relevant_frames=duplicate_pairs,
        relevant_video_ids=("A_VIDEO",),
    )
    second = replace(first)
    result = BenchmarkValidator().validate(_benchmark(first, second), _registry())
    codes = [issue.code for issue in result.errors]
    assert "DUPLICATE_QUERY_ID" in codes
    assert "DUPLICATE_RELEVANT_FRAME" in codes
    assert "RELEVANT_FRAME_NOT_MAPPED" in codes


def test_validator_rejects_duplicate_variant_within_semantic_group() -> None:
    first = _query("vi-1")
    second = _query("vi-2")
    result = BenchmarkValidator().validate(_benchmark(first, second), _registry())
    assert "DUPLICATE_GROUP_VARIANT" in {issue.code for issue in result.errors}


def test_drafts_are_validated_but_excluded_from_scoring() -> None:
    verified = _query("verified")
    draft = _query(
        "draft",
        status=AnnotationStatus.DRAFT,
        relevant_frames=(),
        relevant_video_ids=(),
        group="draft-group",
        text="draft query",
    )
    validator = BenchmarkValidator()
    result = validator.validate(_benchmark(verified, draft), _registry())
    assert result.valid
    assert [query.query_id for query in result.verified_queries] == ["verified"]
    assert [query.query_id for query in result.draft_queries] == ["draft"]
    assert [query.query_id for query in result.validation_scope_queries] == ["verified"]
    assert {warning.code for warning in result.warnings} == {
        "DRAFTS_EXCLUDED_FROM_SCORING"
    }

    inclusive = validator.validate(
        _benchmark(verified, draft), _registry(), include_drafts=True
    )
    assert {query.query_id for query in inclusive.validation_scope_queries} == {
        "verified",
        "draft",
    }
    report = KISBenchmarkEvaluator().evaluate(result, _retriever(), top_ks=(1,))
    assert not isinstance(report, NoVerifiedQueriesResult)
    assert report.evaluated_query_count == 1
    assert report.excluded_draft_query_count == 1
    assert report.invalid_query_count == 0


def test_exact_metrics_aggregates_and_paired_comparison() -> None:
    vietnamese = _query("vi", text="vi query")
    english = _query(
        "en",
        language=BenchmarkLanguage.ENGLISH,
        variant_type=VariantType.ENGLISH_TRANSLATION,
        text="en query",
    )
    validation = BenchmarkValidator().validate(
        _benchmark(vietnamese, english), _registry()
    )
    report = KISBenchmarkEvaluator().evaluate(
        validation, _retriever(), top_ks=(1, 2, 5)
    )
    by_query = {metric.query_id: metric for metric in report.query_metrics}
    assert by_query["vi"].first_relevant_rank == 2
    assert by_query["vi"].reciprocal_rank == 0.5
    assert dict(by_query["vi"].hit_count_at_k) == {1: 0, 2: 1, 5: 2}
    assert dict(by_query["vi"].recall_at_k) == {1: 0.0, 2: 1.0, 5: 1.0}
    assert dict(by_query["vi"].ground_truth_coverage_at_k) == {
        1: 0.0,
        2: 0.5,
        5: 1.0,
    }
    assert dict(by_query["vi"].relevant_video_coverage_at_k or ()) == {
        1: 0.5,
        2: 1.0,
        5: 1.0,
    }
    assert by_query["en"].first_relevant_rank == 1
    all_metrics = next(
        aggregate
        for aggregate in report.aggregates
        if aggregate.group_type == "all"
    )
    assert all_metrics.query_count == 2
    assert all_metrics.mean_reciprocal_rank == 0.75
    assert dict(all_metrics.mean_recall_at_k) == {1: 0.5, 2: 1.0, 5: 1.0}
    assert dict(all_metrics.mean_ground_truth_coverage_at_k) == {
        1: 0.25,
        2: 0.5,
        5: 1.0,
    }
    pair = next(
        item
        for item in report.paired_comparisons
        if item.comparison_variant_type == "english_translation"
    )
    assert pair.status == "COMPARED"
    assert pair.first_relevant_rank_delta_english_minus_vietnamese == -1
    assert dict(pair.recall_delta_at_k) == {1: 1.0, 2: 0.0, 5: 0.0}
    assert dict(pair.recall_outcome_at_k) == {
        1: "win",
        2: "tie",
        5: "tie",
    }
    assert pair.first_relevant_rank_outcome == "win"
    summary = next(
        item
        for item in report.paired_summaries
        if item.comparison_variant_type == "english_translation"
    )
    assert summary.recall_counts_at_k == (
        (1, 1, 0, 0),
        (2, 0, 1, 0),
        (5, 0, 1, 0),
    )
    assert summary.first_relevant_rank_counts == (1, 0, 0)


def test_paired_first_rank_handles_one_sided_miss_as_english_win() -> None:
    labels = (RelevantFrame("B_VIDEO", 50),)
    vietnamese = _query(
        "vi-miss",
        text="vi query",
        relevant_frames=labels,
        relevant_video_ids=("B_VIDEO",),
    )
    english = _query(
        "en-hit",
        language=BenchmarkLanguage.ENGLISH,
        variant_type=VariantType.ENGLISH_TRANSLATION,
        text="en negative query",
        relevant_frames=labels,
        relevant_video_ids=("B_VIDEO",),
    )
    validation = BenchmarkValidator().validate(
        _benchmark(vietnamese, english), _registry()
    )
    report = KISBenchmarkEvaluator().evaluate(
        validation, _retriever(), top_ks=(1,)
    )
    assert not isinstance(report, NoVerifiedQueriesResult)
    pair = next(
        item
        for item in report.paired_comparisons
        if item.comparison_variant_type == "english_translation"
    )
    assert pair.status == "COMPARED"
    assert pair.vietnamese_first_relevant_rank is None
    assert pair.comparison_first_relevant_rank == 1
    assert pair.first_relevant_rank_delta_english_minus_vietnamese is None
    assert pair.first_relevant_rank_outcome == "win"
    assert dict(pair.recall_delta_at_k) == {1: 1.0}
    assert dict(pair.recall_outcome_at_k) == {1: "win"}


def test_rank_metrics_return_null_and_zero_when_no_positive_is_retrieved() -> None:
    missed = _query(
        "miss",
        text="miss query",
        relevant_frames=(RelevantFrame("B_VIDEO", 50),),
        relevant_video_ids=("B_VIDEO",),
    )
    validation = BenchmarkValidator().validate(_benchmark(missed), _registry())
    report = KISBenchmarkEvaluator().evaluate(
        validation, _retriever(), top_ks=(1,)
    )
    assert not isinstance(report, NoVerifiedQueriesResult)
    metric = report.query_metrics[0]
    assert metric.first_relevant_rank is None
    assert metric.reciprocal_rank == 0.0
    assert dict(metric.recall_at_k) == {1: 0.0}
    aggregate = next(item for item in report.aggregates if item.group_type == "all")
    assert aggregate.mean_reciprocal_rank == 0.0
    assert aggregate.query_count == 1


def test_zero_verified_queries_returns_explicit_state() -> None:
    draft = _query(
        "draft-only",
        status=AnnotationStatus.DRAFT,
        relevant_frames=(),
        relevant_video_ids=(),
        text="draft query",
    )
    validation = BenchmarkValidator().validate(_benchmark(draft), _registry())
    outcome = KISBenchmarkEvaluator().evaluate(validation, None, top_ks=(1, 5))
    assert isinstance(outcome, NoVerifiedQueriesResult)
    assert outcome.evaluation_state == "no_verified_queries"
    assert outcome.evaluated_query_count == 0
    assert outcome.excluded_draft_query_count == 1
    assert outcome.invalid_query_count == 0


def test_invalid_benchmark_is_never_scored() -> None:
    invalid = _query(
        "invalid",
        relevant_frames=(RelevantFrame("A_VIDEO", 999),),
        relevant_video_ids=("A_VIDEO",),
    )
    validation = BenchmarkValidator().validate(_benchmark(invalid), _registry())
    assert not validation.valid
    assert validation.invalid_query_count == 1
    with pytest.raises(ValueError, match="invalid benchmark"):
        KISBenchmarkEvaluator().evaluate(validation, _retriever(), top_ks=(1,))


def test_missing_or_unverified_pair_is_explicit() -> None:
    vietnamese = _query("vi")
    english_draft = _query(
        "en-draft",
        language=BenchmarkLanguage.ENGLISH,
        variant_type=VariantType.ENGLISH_TRANSLATION,
        status=AnnotationStatus.DRAFT,
        relevant_frames=(),
        relevant_video_ids=(),
        text="draft query",
    )
    validation = BenchmarkValidator().validate(
        _benchmark(vietnamese, english_draft), _registry()
    )
    report = KISBenchmarkEvaluator().evaluate(
        validation, _retriever(), top_ks=(1, 5)
    )
    translation = next(
        pair
        for pair in report.paired_comparisons
        if pair.comparison_variant_type == "english_translation"
    )
    expansion = next(
        pair
        for pair in report.paired_comparisons
        if pair.comparison_variant_type == "english_expansion"
    )
    assert translation.status == "ENGLISH_VARIANT_UNVERIFIED"
    assert expansion.status == "MISSING_OR_AMBIGUOUS_ENGLISH_VARIANT"


def test_evaluator_does_not_call_temporal_suppression(monkeypatch) -> None:
    def forbidden(*_args, **_kwargs):
        raise AssertionError("benchmark metrics must not use temporal suppression")

    monkeypatch.setattr("system_tai.ranking.kis_ranker.KISRanker.apply", forbidden)
    validation = BenchmarkValidator().validate(_benchmark(_query("q")), _registry())
    report = KISBenchmarkEvaluator().evaluate(
        validation, _retriever(), top_ks=(1, 5)
    )
    assert report.canonical_unsuppressed is True


def test_reports_are_deterministic_and_serializable(tmp_path: Path) -> None:
    validation = BenchmarkValidator().validate(_benchmark(_query("q")), _registry())
    report = KISBenchmarkEvaluator().evaluate(
        validation, _retriever(), top_ks=(1, 5)
    )
    first = write_benchmark_reports(report, tmp_path / "first")
    second = write_benchmark_reports(report, tmp_path / "second")
    for left, right in (
        (first.json_path, second.json_path),
        (first.csv_path, second.csv_path),
        (first.markdown_path, second.markdown_path),
    ):
        assert left.read_bytes() == right.read_bytes()
    payload = json.loads(first.json_path.read_text(encoding="utf-8"))
    assert payload["evaluation_state"] == "completed"
    assert payload["invalid_query_count"] == 0
    assert payload["canonical_unsuppressed"] is True
    csv_text = first.csv_path.read_text(encoding="utf-8")
    assert csv_text.startswith("query_id,")
    assert "ground_truth_coverage_at_1" in csv_text.splitlines()[0]
    assert "relevant_video_coverage_at_1" in csv_text.splitlines()[0]
    assert "# KIS Benchmark Report" in first.markdown_path.read_text(encoding="utf-8")


def test_annotation_helper_keeps_frame_id_and_never_marks_relevant(
    tmp_path: Path,
) -> None:
    result = _retriever().retrieve(
        KISQuery(query_id="draft", text="draft query", top_k=2)
    )
    image_root = tmp_path / "images"
    image_root.mkdir()
    (image_root / "1.jpg").write_bytes(b"not-a-real-image")
    candidates = build_annotation_candidates(
        result,
        {"A_VIDEO": image_root},
        limit=2,
    )
    assert candidates[0].frame_id == result.ranked_candidates[0].frame_id
    assert candidates[0].decision == "unreviewed"
    output = write_draft_annotation_review(
        _query(
            "draft",
            status=AnnotationStatus.DRAFT,
            relevant_frames=(),
            relevant_video_ids=(),
            text="draft query",
        ),
        candidates,
        tmp_path / "review.json",
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["annotation_status"] == "draft"
    assert payload["relevant_frames"] == []
    assert all(item["decision"] == "unreviewed" for item in payload["candidates"])


def test_cli_validation_only_does_not_load_model_or_write_reports(
    tmp_path: Path,
    capsys,
) -> None:
    mapping = tmp_path / "VIDEO.csv"
    mapping.write_text(
        "n,pts_time,fps,frame_idx\n1,0.0,30.0,0\n",
        encoding="utf-8-sig",
    )
    np.save(tmp_path / "VIDEO.npy", np.ones((1, 512), dtype=np.float32))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "videos": [
                    {
                        "video_id": "VIDEO",
                        "mapping_csv_path": str(mapping),
                        "clip_npy_path": str(tmp_path / "VIDEO.npy"),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    benchmark = tmp_path / "benchmark.yaml"
    benchmark.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "benchmark_id": "cli-validation",
                "description": "draft-only CLI fixture",
                "queries": [
                    {
                        "query_id": "draft",
                        "language": "en",
                        "text": "draft query",
                        "semantic_group_id": "draft-group",
                        "variant_type": "english_translation",
                        "relevant_frames": [],
                        "relevant_video_ids": [],
                        "annotation_notes": "not verified",
                        "annotation_status": "draft",
                        "source_scope": ["VIDEO"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "reports"
    exit_code = benchmark_main(
        [
            "--manifest",
            str(manifest),
            "--benchmark",
            str(benchmark),
            "--output-directory",
            str(output),
            "--validation-only",
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "validation-only: completed" in captured.out
    assert not output.exists()

    exit_code = benchmark_main(
        [
            "--manifest",
            str(manifest),
            "--benchmark",
            str(benchmark),
            "--output-directory",
            str(output),
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "evaluation state=no_verified_queries" in captured.out
    assert not output.exists()
