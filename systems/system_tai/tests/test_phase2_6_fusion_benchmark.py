from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest

from system_tai.common.schemas import CandidateFrame, KISQuery, KISResult
from system_tai.evaluation.benchmark_schema import (
    AnnotationStatus,
    BenchmarkLanguage,
    BenchmarkQuery,
    KISBenchmark,
    RelevantFrame,
    VariantType,
    load_benchmark,
)
from system_tai.evaluation.benchmark_validator import BenchmarkValidator
from system_tai.evaluation.fusion_benchmark import (
    FusionBenchmarkEvaluator,
    NoComparableFusionGroupsError,
    select_comparable_fusion_groups,
)
from system_tai.evaluation.fusion_reports import write_fusion_reports
from system_tai.features.btc_clip_store import FeatureStoreRegistry
from system_tai.kis.benchmark_fusion import build_parser
from system_tai.kis.benchmark_fusion import main as fusion_main
from system_tai.retrieval.multi_query import WeightedRRFRetriever
from tests.phase2_helpers import make_store, write_mapping

ROOT = Path(__file__).resolve().parents[1]
PILOT_PATH = ROOT / "config" / "kis_benchmark.pilot_three_groups.yaml"


def _pilot_registry() -> FeatureStoreRegistry:
    return FeatureStoreRegistry(
        [
            make_store(
                "L21_V001",
                np.asarray(
                    [[1.0, 0.0], [0.9, 0.1], [0.8, 0.2], [0.7, 0.3]],
                    dtype=np.float32,
                ),
                [2130, 2217, 23163, 23982],
            ),
            make_store(
                "L21_V002",
                np.asarray([[1.0, 0.0]], dtype=np.float32),
                [20574],
            ),
            make_store(
                "L22_V001",
                np.asarray([[1.0, 0.0]], dtype=np.float32),
                [19821],
            ),
        ]
    )


def _query(
    query_id: str,
    variant_type: VariantType,
    *,
    group: str = "group",
    status: AnnotationStatus = AnnotationStatus.VERIFIED,
    frames: tuple[RelevantFrame, ...] = (
        RelevantFrame("L21_V001", 2130),
        RelevantFrame("L21_V001", 2217),
    ),
    scope: tuple[str, ...] = ("L21_V001",),
) -> BenchmarkQuery:
    language = (
        BenchmarkLanguage.VIETNAMESE
        if variant_type is VariantType.VIETNAMESE_DIRECT
        else BenchmarkLanguage.ENGLISH
    )
    return BenchmarkQuery(
        query_id=query_id,
        language=language,
        text=query_id,
        semantic_group_id=group,
        variant_type=variant_type,
        relevant_frames=frames if status is AnnotationStatus.VERIFIED else (),
        relevant_video_ids=("L21_V001",) if status is AnnotationStatus.VERIFIED else (),
        annotation_notes="human verified" if status is AnnotationStatus.VERIFIED else "draft",
        annotation_status=status,
        source_scope=scope,
    )


def _benchmark(*queries: BenchmarkQuery) -> KISBenchmark:
    return KISBenchmark(1, "fusion-test", "synthetic", tuple(queries))


def _complete_group() -> tuple[BenchmarkQuery, ...]:
    return (
        _query("vi", VariantType.VIETNAMESE_DIRECT),
        _query("translation", VariantType.ENGLISH_TRANSLATION),
        _query("expansion", VariantType.ENGLISH_EXPANSION),
    )


def test_pilot_fixture_has_exact_authorized_counts_and_labels() -> None:
    parsed = load_benchmark(PILOT_PATH)
    assert parsed.valid
    assert parsed.benchmark is not None
    benchmark = parsed.benchmark
    assert len({query.semantic_group_id for query in benchmark.queries}) == 3
    assert sum(
        query.annotation_status is AnnotationStatus.VERIFIED
        for query in benchmark.queries
    ) == 9
    assert sum(
        query.annotation_status is AnnotationStatus.DRAFT
        for query in benchmark.queries
    ) == 6
    observed = {
        (label.video_id, label.frame_id)
        for query in benchmark.queries
        if query.annotation_status is AnnotationStatus.VERIFIED
        for label in query.relevant_frames
    }
    assert observed == {
        ("L21_V002", 20574),
        ("L22_V001", 19821),
        ("L21_V001", 23982),
        ("L21_V001", 2130),
        ("L21_V001", 2217),
        ("L21_V001", 23163),
    }
    validation = BenchmarkValidator().validate(benchmark, _pilot_registry())
    assert validation.valid, validation.errors
    selection = select_comparable_fusion_groups(benchmark)
    assert len(selection.groups) == 3
    assert selection.draft_query_count == 6
    assert not selection.issues


def test_group_selection_reports_missing_duplicate_draft_and_incomparable() -> None:
    draft_only = _query(
        "draft",
        VariantType.ENGLISH_EXPANSION,
        group="missing",
        status=AnnotationStatus.DRAFT,
    )
    missing = (
        _query("missing_vi", VariantType.VIETNAMESE_DIRECT, group="missing"),
        _query("missing_en", VariantType.ENGLISH_TRANSLATION, group="missing"),
        draft_only,
    )
    duplicate = (
        _query("dup_vi", VariantType.VIETNAMESE_DIRECT, group="duplicate"),
        _query("dup_en_1", VariantType.ENGLISH_TRANSLATION, group="duplicate"),
        _query("dup_en_2", VariantType.ENGLISH_TRANSLATION, group="duplicate"),
        _query("dup_exp", VariantType.ENGLISH_EXPANSION, group="duplicate"),
    )
    incomparable = list(
        replace(query, semantic_group_id="incomparable") for query in _complete_group()
    )
    incomparable[2] = replace(
        incomparable[2],
        relevant_frames=(RelevantFrame("L21_V001", 23163),),
    )
    selection = select_comparable_fusion_groups(
        _benchmark(*missing, *duplicate, *incomparable)
    )
    assert not selection.groups
    assert selection.draft_query_count == 1
    codes = {issue.code for issue in selection.issues}
    assert codes == {
        "MISSING_VERIFIED_VARIANT",
        "DUPLICATE_VERIFIED_VARIANT",
        "INCOMPARABLE_VERIFIED_VARIANTS",
    }


class RankedFakeExactRetriever:
    text_encoder = type(
        "Encoder",
        (),
        {
            "identifiers": MappingProxyType(
                {"model": "fake-openai-clip", "device": "cpu", "library": "test"}
            )
        },
    )()

    def retrieve(self, query: KISQuery) -> KISResult:
        positive_rank = {"vi": 4, "translation": 2, "expansion": 3}[query.text]
        candidates: list[CandidateFrame] = []
        for rank in range(1, max(query.top_k, positive_rank) + 1):
            if rank == positive_rank:
                frame_id = 2130
            else:
                frame_id = 10000 + rank
            candidates.append(
                CandidateFrame(
                    video_id="L21_V001",
                    frame_id=frame_id,
                    clip_row=rank - 1,
                    keyframe_order=rank,
                    score=1.0 / rank,
                    rank=rank,
                    source="clip_exact",
                )
            )
        return KISResult(query.query_id, tuple(candidates[: query.top_k]))


def test_fusion_metrics_multiple_positives_and_deterministic_reports(
    tmp_path: Path,
) -> None:
    benchmark = _benchmark(
        *_complete_group(),
        _query(
            "draft",
            VariantType.VIETNAMESE_DIRECT,
            status=AnnotationStatus.DRAFT,
        ),
    )
    validation = BenchmarkValidator().validate(benchmark, _pilot_registry())
    assert validation.valid
    report = FusionBenchmarkEvaluator().evaluate(
        validation,
        WeightedRRFRetriever(RankedFakeExactRetriever()),
        top_ks=(1, 5, 20, 100),
        top_k_per_variant=100,
        rrf_constant=60,
    )
    assert report.evaluated_group_count == 1
    assert report.evaluated_verified_query_count == 3
    assert report.excluded_draft_query_count == 1
    metric = report.group_metrics[0]
    assert metric.first_relevant_rank == 2
    assert metric.reciprocal_rank == 0.5
    assert dict(metric.recall_at_k)[1] == 0.0
    assert dict(metric.recall_at_k)[5] == 1.0
    assert dict(metric.hit_count_at_k)[5] == 1
    assert dict(metric.ground_truth_coverage_at_k)[5] == 0.5
    assert metric.contributing_variant_ids == ("vi", "translation", "expansion")
    assert metric.contributing_variant_count == 3
    first = write_fusion_reports(report, tmp_path / "first")
    second = write_fusion_reports(report, tmp_path / "second")
    assert first.json_path.read_bytes() == second.json_path.read_bytes()
    assert first.csv_path.read_bytes() == second.csv_path.read_bytes()
    assert first.markdown_path.read_bytes() == second.markdown_path.read_bytes()


def test_evaluator_fails_clearly_without_comparable_verified_group() -> None:
    benchmark = _benchmark(
        _query("vi", VariantType.VIETNAMESE_DIRECT),
        _query("draft", VariantType.ENGLISH_TRANSLATION, status=AnnotationStatus.DRAFT),
    )
    validation = BenchmarkValidator().validate(benchmark, _pilot_registry())
    assert validation.valid
    with pytest.raises(NoComparableFusionGroupsError, match="no comparable"):
        FusionBenchmarkEvaluator().evaluate(
            validation,
            WeightedRRFRetriever(RankedFakeExactRetriever()),
        )


def test_fusion_cli_parser_exposes_explicit_opt_in_configuration() -> None:
    args = build_parser().parse_args(
        [
            "--manifest",
            "manifest.json",
            "--benchmark",
            "pilot.yaml",
            "--device",
            "cpu",
            "--rrf-constant",
            "42",
            "--top-k-per-variant",
            "80",
            "--fail-on-invalid",
        ]
    )
    assert args.device == "cpu"
    assert args.rrf_constant == 42
    assert args.top_k_per_variant == 80
    assert args.fail_on_invalid


def test_fusion_cli_validation_only_avoids_model_and_reports(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    videos = {
        "L21_V001": [2130, 2217, 23163, 23982],
        "L21_V002": [20574],
        "L22_V001": [19821],
    }
    manifest_entries: list[dict[str, str]] = []
    for video_id, frame_ids in videos.items():
        mapping = tmp_path / f"{video_id}.csv"
        features = tmp_path / f"{video_id}.npy"
        write_mapping(
            mapping,
            [
                (index + 1, float(index), 30.0, frame_id)
                for index, frame_id in enumerate(frame_ids)
            ],
        )
        matrix = np.zeros((len(frame_ids), 512), dtype=np.float32)
        matrix[:, 0] = 1.0
        np.save(features, matrix)
        manifest_entries.append(
            {
                "video_id": video_id,
                "mapping_csv_path": mapping.name,
                "clip_npy_path": features.name,
            }
        )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"videos": manifest_entries}),
        encoding="utf-8",
    )
    output = tmp_path / "reports"
    exit_code = fusion_main(
        [
            "--manifest",
            str(manifest),
            "--benchmark",
            str(PILOT_PATH),
            "--output-directory",
            str(output),
            "--validation-only",
            "--fail-on-invalid",
        ]
    )
    assert exit_code == 0
    assert "comparable=3" in capsys.readouterr().out
    assert not output.exists()
