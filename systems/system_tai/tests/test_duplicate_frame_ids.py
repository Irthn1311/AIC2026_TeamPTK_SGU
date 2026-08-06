from __future__ import annotations

import json
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest

from system_tai.checkpointing.exporter import CheckpointExporter
from system_tai.common.schemas import FrameMappingRecord, KISQuery, VideoFeatureStore
from system_tai.evaluation.benchmark_schema import (
    AnnotationStatus,
    BenchmarkLanguage,
    BenchmarkQuery,
    KISBenchmark,
    RelevantFrame,
    VariantType,
)
from system_tai.evaluation.benchmark_validator import BenchmarkValidator
from system_tai.features.btc_clip_store import (
    FeatureStoreRegistry,
    LoadedVideoFeatureStore,
    VideoFeatureStoreLoader,
)
from system_tai.retrieval.multi_query import (
    QueryLanguage,
    QueryVariant,
    QueryVariantType,
    WeightedRRFRetriever,
)
from system_tai.retrieval.vector_search import ExactNumpyRetriever
from system_tai.validation.checkpoint_validator import CheckpointValidator
from tests.phase2_helpers import make_store, write_mapping


class ConstantEncoder:
    dimension = 3
    identifiers = MappingProxyType({"library": "duplicate-frame-test-fake"})

    def encode(self, text: str) -> np.ndarray:
        assert text
        return np.asarray([1.0, 0.0, 0.0], dtype=np.float32)


def _duplicate_store(tmp_path: Path) -> LoadedVideoFeatureStore:
    mapping = tmp_path / "L21_V001.csv"
    features = tmp_path / "L21_V001.npy"
    write_mapping(
        mapping,
        [
            (1, 0.0, 30.0, 0),
            (2, 0.01, 30.0, 0),
            (3, 1.0, 30.0, 10),
            (4, 2.0, 30.0, 20),
        ],
    )
    matrix = np.asarray(
        [
            [0.8, 0.6, 0.0],
            [1.0, 0.0, 0.0],
            [0.9, 0.4358899, 0.0],
            [0.7, 0.71414286, 0.0],
        ],
        dtype=np.float32,
    )
    np.save(features, matrix)
    return VideoFeatureStoreLoader(expected_dimension=3).load(
        video_id="L21_V001",
        mapping_csv_path=mapping,
        clip_npy_path=features,
    )


def test_duplicate_zero_frame_rows_are_one_to_many_not_ambiguous(tmp_path: Path) -> None:
    store = _duplicate_store(tmp_path)

    assert [record.clip_row for record in store.mappings] == [0, 1, 2, 3]
    assert [record.frame_id for record in store.mappings] == [0, 0, 10, 20]
    assert store.rows_for_frame(0) == (0, 1)
    assert store.rows_for_frame(999) == ()
    assert store.contains_frame(0)
    assert store.unique_frame_count == 3
    assert store.duplicate_frame_id_count == 1


def test_higher_scoring_duplicate_row_is_representative_and_top_k_is_unique(
    tmp_path: Path,
) -> None:
    registry = FeatureStoreRegistry([_duplicate_store(tmp_path)])
    result = ExactNumpyRetriever(registry, ConstantEncoder(), chunk_size=1).retrieve(
        KISQuery("Q", "query", 3)
    )

    assert [
        (candidate.video_id, candidate.frame_id, candidate.clip_row)
        for candidate in result.ranked_candidates
    ] == [
        ("L21_V001", 0, 1),
        ("L21_V001", 10, 2),
        ("L21_V001", 20, 3),
    ]
    assert [candidate.rank for candidate in result.ranked_candidates] == [1, 2, 3]
    assert len(
        {(candidate.video_id, candidate.frame_id) for candidate in result.ranked_candidates}
    ) == 3


def test_equal_duplicate_scores_choose_smaller_clip_row() -> None:
    store = make_store(
        "VIDEO",
        np.asarray(
            [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.9, 0.4358899, 0.0]],
            dtype=np.float32,
        ),
        [7, 7, 8],
    )
    result = ExactNumpyRetriever(
        FeatureStoreRegistry([store]), ConstantEncoder(), chunk_size=2
    ).retrieve(KISQuery("tie", "query", 2))

    assert [(candidate.frame_id, candidate.clip_row) for candidate in result.ranked_candidates] == [
        (7, 0),
        (8, 2),
    ]


def test_unique_top_k_does_not_lose_candidate_behind_duplicate_rows() -> None:
    store = make_store(
        "VIDEO",
        np.asarray(
            [[1.0, 0.0, 0.0], [0.99, 0.14106736, 0.0], [0.98, 0.19899749, 0.0]],
            dtype=np.float32,
        ),
        [0, 0, 10],
    )
    result = ExactNumpyRetriever(
        FeatureStoreRegistry([store]), ConstantEncoder(), chunk_size=1
    ).retrieve(KISQuery("unique", "query", 2))

    assert [(candidate.frame_id, candidate.clip_row) for candidate in result.ranked_candidates] == [
        (0, 0),
        (10, 2),
    ]


def test_duplicate_clip_row_or_ambiguous_physical_order_remains_rejected() -> None:
    matrix = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    matrix.setflags(write=False)
    descriptor = VideoFeatureStore(
        video_id="VIDEO",
        mapping_csv_path=Path("VIDEO.csv"),
        clip_npy_path=Path("VIDEO.npy"),
        row_count=2,
        embedding_dimension=2,
        normalized=True,
    )
    duplicate_rows = tuple(
        FrameMappingRecord(row, row + 1, row, float(row), 30.0)
        for row in (0, 0)
    )
    with pytest.raises(ValueError, match="duplicate mapping for clip_row"):
        LoadedVideoFeatureStore(descriptor, matrix, duplicate_rows)

    reversed_rows = (
        FrameMappingRecord(1, 1, 0, 0.0, 30.0),
        FrameMappingRecord(0, 2, 10, 1.0, 30.0),
    )
    with pytest.raises(ValueError, match="ambiguous physical mapping order"):
        LoadedVideoFeatureStore(descriptor, matrix, reversed_rows)


def test_checkpoint_and_benchmark_validate_duplicate_frame_corpus(tmp_path: Path) -> None:
    registry = FeatureStoreRegistry([_duplicate_store(tmp_path)])
    result = ExactNumpyRetriever(registry, ConstantEncoder()).retrieve(
        KISQuery("Q", "query", 3)
    )
    checkpoint = tmp_path / "top100.jsonl"
    CheckpointExporter().export(result, checkpoint)
    validation = CheckpointValidator().validate(checkpoint, registry)
    records = [json.loads(line) for line in checkpoint.read_text(encoding="utf-8").splitlines()]

    assert validation.valid
    assert [record["rank"] for record in records] == [1, 2, 3]
    assert [record["frame_id"] for record in records] == [0, 10, 20]
    assert all(set(record) == {"query_id", "rank", "video_id", "frame_id"} for record in records)

    query = BenchmarkQuery(
        query_id="verified",
        language=BenchmarkLanguage.VIETNAMESE,
        text="query",
        semantic_group_id="group",
        variant_type=VariantType.VIETNAMESE_DIRECT,
        relevant_frames=(RelevantFrame("L21_V001", 0),),
        relevant_video_ids=("L21_V001",),
        annotation_notes="synthetic verified duplicate-frame fixture",
        annotation_status=AnnotationStatus.VERIFIED,
        source_scope=("L21_V001",),
    )
    benchmark = KISBenchmark(1, "duplicate-frame", "synthetic", (query,))
    assert BenchmarkValidator().validate(benchmark, registry).valid


def test_weighted_rrf_preserves_unique_identity_after_exact_retrieval(tmp_path: Path) -> None:
    exact = ExactNumpyRetriever(
        FeatureStoreRegistry([_duplicate_store(tmp_path)]),
        ConstantEncoder(),
        chunk_size=1,
    )
    variants = (
        QueryVariant(
            "vi",
            "query vi",
            QueryLanguage.VIETNAMESE,
            QueryVariantType.VIETNAMESE_DIRECT,
        ),
        QueryVariant(
            "en",
            "query en",
            QueryLanguage.ENGLISH,
            QueryVariantType.ENGLISH_TRANSLATION,
        ),
    )
    result = WeightedRRFRetriever(exact).retrieve(
        query_id="Q",
        variants=variants,
        top_k_per_variant=3,
        output_top_k=3,
    )
    identities = [
        (candidate.video_id, candidate.frame_id) for candidate in result.ranked_candidates
    ]

    assert identities == [("L21_V001", 0), ("L21_V001", 10), ("L21_V001", 20)]
    assert len(identities) == len(set(identities))
    assert [candidate.rank for candidate in result.ranked_candidates] == [1, 2, 3]
