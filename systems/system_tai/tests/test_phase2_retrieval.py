from __future__ import annotations

from types import MappingProxyType

import numpy as np

from system_tai.common.schemas import CandidateFrame, KISQuery, KISResult
from system_tai.features.btc_clip_store import FeatureStoreRegistry
from system_tai.ranking.kis_ranker import KISRanker, TemporalSuppressionConfig
from system_tai.retrieval.vector_search import ExactNumpyRetriever, VectorSearch
from tests.phase2_helpers import make_store


class FakeEncoder:
    dimension = 3
    identifiers = MappingProxyType({"library": "deterministic-test-fake"})

    def __init__(self, vector=(1.0, 0.0, 0.0)) -> None:
        self.vector = np.asarray(vector, dtype=np.float32)
        self.calls = 0

    def encode(self, text: str) -> np.ndarray:
        assert text
        self.calls += 1
        return self.vector.copy()


def _registry() -> FeatureStoreRegistry:
    return FeatureStoreRegistry(
        [
            make_store(
                "A_VIDEO",
                np.asarray([[2, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float32),
                [30, 10, 20],
            ),
            make_store(
                "B_VIDEO",
                np.asarray([[1, 0, 0], [-1, 0, 0]], dtype=np.float32),
                [5, 6],
            ),
        ]
    )


def test_exact_global_ranking_normalization_and_tie_breaking() -> None:
    encoder = FakeEncoder((4, 0, 0))
    result = ExactNumpyRetriever(_registry(), encoder, chunk_size=1).retrieve(
        KISQuery(query_id="q1", text="query", top_k=10)
    )
    assert encoder.calls == 1
    assert [(c.video_id, c.frame_id, c.clip_row) for c in result.ranked_candidates] == [
        ("A_VIDEO", 30, 0),
        ("B_VIDEO", 5, 0),
        ("A_VIDEO", 10, 1),
        ("A_VIDEO", 20, 2),
        ("B_VIDEO", 6, 1),
    ]
    assert [candidate.rank for candidate in result.ranked_candidates] == [1, 2, 3, 4, 5]
    assert all(candidate.source == "clip_exact" for candidate in result.ranked_candidates)


def test_tie_break_uses_frame_then_clip_row() -> None:
    store = make_store(
        "VIDEO",
        np.asarray([[1, 0, 0], [1, 0, 0], [1, 0, 0]], dtype=np.float32),
        [20, 10, 30],
    )
    result = ExactNumpyRetriever(
        FeatureStoreRegistry([store]), FakeEncoder(), chunk_size=2
    ).search_vector(query_id="tie", query_vector=[1, 0, 0], top_k=3)
    assert [(candidate.frame_id, candidate.clip_row) for candidate in result.ranked_candidates] == [
        (10, 1),
        (20, 0),
        (30, 2),
    ]


def test_chunked_search_equals_non_chunked_reference() -> None:
    registry = _registry()
    query = np.asarray([0.3, 0.7, 0.2], dtype=np.float32)
    chunked = ExactNumpyRetriever(registry, FakeEncoder(), chunk_size=1).search_vector(
        query_id="q", query_vector=query, top_k=4
    )
    reference = ExactNumpyRetriever(registry, FakeEncoder(), chunk_size=999).search_vector(
        query_id="q", query_vector=query, top_k=4
    )
    assert [(c.video_id, c.frame_id, c.clip_row, c.score) for c in chunked.ranked_candidates] == [
        (c.video_id, c.frame_id, c.clip_row, c.score) for c in reference.ranked_candidates
    ]


def test_single_matrix_compatibility_search_rejects_zero_norm() -> None:
    hits = VectorSearch().search([1, 0], np.asarray([[2, 0], [1, 1]]), top_k=10)
    assert [hit.clip_row for hit in hits] == [0, 1]
    try:
        VectorSearch().search([1, 0], np.asarray([[0, 0]]), top_k=1)
    except ValueError as exc:
        assert "zero-norm" in str(exc)
    else:
        raise AssertionError("zero-norm row was accepted")


def _candidate(rank: int, frame_id: int, video_id: str = "v") -> CandidateFrame:
    return CandidateFrame(
        video_id=video_id,
        frame_id=frame_id,
        clip_row=rank - 1,
        keyframe_order=rank,
        score=1.0 / rank,
        rank=rank,
        source="clip_exact",
    )


def test_temporal_suppression_is_optional_and_reports_removals() -> None:
    original = KISResult(
        query_id="q",
        ranked_candidates=(
            _candidate(1, 100),
            _candidate(2, 105),
            _candidate(3, 200),
            _candidate(4, 102, "other"),
        ),
    )
    unchanged, disabled = KISRanker().apply(original)
    assert unchanged is original
    assert disabled.removed_count == 0

    filtered, report = KISRanker().apply(
        original,
        TemporalSuppressionConfig(
            enabled=True, minimum_frame_gap=10, maximum_candidates_per_video=2
        ),
    )
    assert [(c.video_id, c.frame_id, c.rank) for c in filtered.ranked_candidates] == [
        ("v", 100, 1),
        ("v", 200, 2),
        ("other", 102, 3),
    ]
    assert report.removed_count == 1
