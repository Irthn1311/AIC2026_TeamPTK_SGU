from __future__ import annotations

from pathlib import Path

import numpy as np

from system_tai.common.schemas import FrameMappingRecord, VideoFeatureStore
from system_tai.features.btc_clip_store import (
    FeatureStoreRegistry,
    LoadedVideoFeatureStore,
)
from system_tai.kis.session_schema import TRAKEQueryRequest
from system_tai.refinement.models import RefinementConfig
from system_tai.retrieval.multi_query import (
    QueryLanguage,
    QueryVariant,
    QueryVariantType,
    WeightedRRFRetriever,
)
from system_tai.retrieval.vector_search import ExactNumpyRetriever
from system_tai.retrieval.video_evidence import (
    FullCorpusVideoMaximaOutcome,
    VideoMaximumHit,
    VideoRestrictedFeatureSearcher,
    rank_store_frames,
)
from system_tai.trake.models import TRAKEEvent, TRAKEEventCandidate, TRAKEQuery
from system_tai.trake.planner import plan_trake_paths
from system_tai.trake.runtime import TRAKERuntimePipeline
from system_tai.trake.video_first import (
    TRAKE_VIDEO_FIRST_RESTRICTED_EVENT_SEARCH,
    EventVideoEvidence,
    TRAKEVideoFirstConfig,
    VariantVideoRank,
    build_event_video_rankings,
    nominate_videos,
)


def _store(
    video_id: str,
    rows: list[tuple[int, float, tuple[float, float]]],
) -> LoadedVideoFeatureStore:
    mappings = tuple(
        FrameMappingRecord(
            clip_row=index,
            keyframe_order=index + 1,
            frame_id=frame_id,
            pts_time=pts_time,
            fps=30.0,
        )
        for index, (frame_id, pts_time, _vector) in enumerate(rows)
    )
    return LoadedVideoFeatureStore(
        descriptor=VideoFeatureStore(
            video_id=video_id,
            mapping_csv_path=Path(f"{video_id}.csv"),
            clip_npy_path=Path(f"{video_id}.npy"),
            row_count=len(rows),
            embedding_dimension=2,
            normalized=False,
        ),
        matrix=np.asarray([vector for _fid, _pts, vector in rows], dtype=np.float32),
        mappings=mappings,
    )


class _TextEncoder:
    dimension = 2

    def encode(self, text: str) -> np.ndarray:
        return np.asarray([1.0, 0.0], dtype=np.float32)


class _SharedEncoder(_TextEncoder):
    identifiers = {"model": "fake"}

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        vectors = {
            "event zero": (1.0, 0.0),
            "event one": (0.0, 1.0),
        }
        return np.asarray([vectors[text] for text in texts], dtype=np.float32)


class _NoRefiner:
    def __init__(self) -> None:
        self.calls = 0

    def refine_query(self, *args, **kwargs):
        self.calls += 1
        raise AssertionError("refiner must not run when refine_top_n is zero")


def _variant(variant_id: str, weight: float = 1.0) -> QueryVariant:
    return QueryVariant(
        variant_id=variant_id,
        text=variant_id,
        language=QueryLanguage.VIETNAMESE,
        variant_type=QueryVariantType.VIETNAMESE_DIRECT,
        weight=weight,
    )


def _maximum(
    query_id: str,
    video_id: str,
    rank: int,
    cosine: float,
) -> VideoMaximumHit:
    return VideoMaximumHit(
        query_id=query_id,
        video_id=video_id,
        frame_id=rank,
        clip_row=rank - 1,
        keyframe_order=rank,
        cosine_score=cosine,
        rank=rank,
    )


def test_full_corpus_video_maxima_bypasses_global_frame_topk() -> None:
    registry = FeatureStoreRegistry(
        [
            _store("A", [(10, 0.0, (0.8, 0.6)), (20, 1.0, (0.6, 0.8))]),
            _store("B", [(5, 0.0, (1.0, 0.0))]),
            _store("C", [(30, 0.0, (0.0, 1.0))]),
        ]
    )
    exact = ExactNumpyRetriever(registry, _TextEncoder(), chunk_size=1)
    ordinary = exact.search_vectors(
        query_ids=("e0", "e1"),
        query_vectors=(np.asarray([1.0, 0.0]), np.asarray([0.0, 1.0])),
        top_k=1,
    )
    assert [item.video_id for item in ordinary["e0"].ranked_candidates] == ["B"]
    assert [item.video_id for item in ordinary["e1"].ranked_candidates] == ["C"]

    maxima = VideoRestrictedFeatureSearcher(registry, chunk_size=1).search_video_maxima(
        query_ids=("e0", "e1"),
        query_vectors=(np.asarray([1.0, 0.0]), np.asarray([0.0, 1.0])),
    )
    assert {item.video_id for item in maxima.rankings["e0"]} == {"A", "B", "C"}
    assert {item.video_id for item in maxima.rankings["e1"]} == {"A", "B", "C"}
    assert maxima.physical_rows_scored == 4
    assert maxima.video_store_scan_count == 3


def test_variant_video_rankings_use_ranks_and_not_raw_cosine() -> None:
    variants = {0: (_variant("vi"), _variant("en"))}
    maxima_a = FullCorpusVideoMaximaOutcome(
        rankings={
            "vi": (_maximum("vi", "A", 1, -0.9), _maximum("vi", "B", 2, 0.99)),
            "en": (_maximum("en", "B", 1, -0.8), _maximum("en", "A", 2, 0.98)),
        },
        physical_rows_scored=2,
        video_store_scan_count=2,
    )
    maxima_b = FullCorpusVideoMaximaOutcome(
        rankings={
            "vi": (_maximum("vi", "A", 1, 0.01), _maximum("vi", "B", 2, -0.7)),
            "en": (_maximum("en", "B", 1, 0.02), _maximum("en", "A", 2, -0.6)),
        },
        physical_rows_scored=2,
        video_store_scan_count=2,
    )
    first = build_event_video_rankings(
        event_variants=variants,
        maxima=maxima_a,
        rrf_constant=60.0,
    )
    second = build_event_video_rankings(
        event_variants=variants,
        maxima=maxima_b,
        rrf_constant=60.0,
    )
    assert [(row.video_id, row.event_video_rank) for row in first[0]] == [
        (row.video_id, row.event_video_rank) for row in second[0]
    ]
    assert first[0][0].event_video_rrf_score == second[0][0].event_video_rrf_score
    assert {item.variant_id: item.video_rank for item in first[0][0].per_variant} == {
        "en": 2,
        "vi": 1,
    }


def _event_ranking(event_index: int, rows: list[tuple[str, int]]) -> tuple[EventVideoEvidence, ...]:
    return tuple(
        EventVideoEvidence(
            event_index=event_index,
            video_id=video_id,
            event_video_rank=rank,
            event_video_rrf_score=1.0 / (60.0 + rank),
            best_variant_rank=rank,
            per_variant=(VariantVideoRank(f"v{event_index}", 1.0, rank),),
        )
        for video_id, rank in rows
    )


def test_video_nomination_exact_lexicographic_policy_and_incomplete_coverage() -> None:
    per_video = {
        "C": (1, 3, 3),
        "B": (2, 2, 4),
        "A": (1, 4, 4),
        "D": (1, 2, 5),
        "E": (2, 3, 4),
        "F": (2, 3, 4),
    }
    rankings = {
        event: _event_ranking(
            event,
            [(video_id, ranks[event]) for video_id, ranks in per_video.items()],
        )
        for event in range(3)
    }
    nominated = nominate_videos(
        event_video_rankings=rankings,
        config=TRAKEVideoFirstConfig(
            enabled=True,
            selected_video_cap=6,
            event_video_nomination_depth=4,
        ),
        rrf_constant=60.0,
    )
    assert [item.video_id for item in nominated] == ["C", "B", "A", "E", "F", "D"]
    assert nominated[-1].coverage_count == 2


def test_selected_video_cap_default_and_32_limit() -> None:
    config = TRAKEVideoFirstConfig(enabled=True)
    assert config.selected_video_cap == 32
    assert config.event_video_nomination_depth == 100
    assert config.anchors_per_event_video == 5
    rows = [(f"V{index:03d}", index + 1) for index in range(40)]
    nominated = nominate_videos(
        event_video_rankings={0: _event_ranking(0, rows)},
        config=config,
        rrf_constant=60.0,
    )
    assert len(nominated) == 32


def test_restricted_search_deduplicates_frames_and_has_no_temporal_nms() -> None:
    store = _store(
        "A",
        [
            (0, 100.0, (0.8, 0.2)),
            (0, 100.0, (1.0, 0.0)),
            (1, 100.1, (0.99, 0.01)),
            (2, 100.2, (0.98, 0.02)),
            (3, 100.3, (0.97, 0.03)),
            (4, 100.4, (0.96, 0.04)),
            (5, 100.5, (0.95, 0.05)),
        ],
    )
    ranked = rank_store_frames(
        store,
        query_ids=("q",),
        query_vectors=(np.asarray([1.0, 0.0]),),
        expected_dimension=2,
        chunk_size=2,
        per_query_cap=5,
    )["q"]
    assert [item.frame_id for item in ranked] == [0, 1, 2, 3, 4]
    assert ranked[0].clip_row == 1
    assert len({item.frame_id for item in ranked}) == 5
    assert ranked[-1].pts_time - ranked[0].pts_time < 5.0


def _pipeline() -> tuple[TRAKERuntimePipeline, _NoRefiner]:
    registry = FeatureStoreRegistry(
        [
            _store("A", [(10, 0.0, (0.8, 0.6)), (20, 1.0, (0.6, 0.8))]),
            _store("B", [(5, 0.0, (1.0, 0.0))]),
            _store("C", [(30, 0.0, (0.0, 1.0))]),
        ]
    )
    shared = _SharedEncoder()
    exact = ExactNumpyRetriever(registry, shared, chunk_size=1)
    refiner = _NoRefiner()
    return (
        TRAKERuntimePipeline(
            exact_retriever=exact,
            weighted_rrf=WeightedRRFRetriever(exact),
            refiner=refiner,
            shared_encoder=shared,
            video_restricted_searcher=VideoRestrictedFeatureSearcher(
                registry,
                chunk_size=1,
            ),
        ),
        refiner,
    )


def _request() -> TRAKEQueryRequest:
    return TRAKEQueryRequest(
        request_id="req",
        query_id="tr",
        events=(
            {"description": "event zero"},
            {"description": "event one"},
        ),
        top_k_per_variant=1,
        event_candidate_top_k=1,
        output_top_k=10,
        beam_width=100,
        refine_top_n=0,
    )


def test_tr_a1_disabled_is_identity_compatible_and_deterministic() -> None:
    pipeline, _ = _pipeline()
    first, _, first_diag = pipeline.process_trake_query(
        _request(),
        refinement_config=RefinementConfig(),
    )
    second, _, second_diag = pipeline.process_trake_query(
        _request(),
        refinement_config=RefinementConfig(),
        video_first_config=TRAKEVideoFirstConfig(enabled=False),
    )
    assert first.predictions == second.predictions == ()
    assert first_diag["event_candidate_pools"] == second_diag["event_candidate_pools"]
    assert "tr_a1" not in first_diag
    assert "tr_a1" not in second_diag


def test_tr_a1_builds_complete_pools_before_planner_and_is_repeatable() -> None:
    config = TRAKEVideoFirstConfig(
        enabled=True,
        selected_video_cap=3,
        event_video_nomination_depth=1,
        anchors_per_event_video=2,
    )
    pipeline, refiner = _pipeline()
    first, _, first_diag = pipeline.process_trake_query(
        _request(),
        refinement_config=RefinementConfig(),
        video_first_config=config,
    )
    second, _, second_diag = pipeline.process_trake_query(
        _request(),
        refinement_config=RefinementConfig(),
        video_first_config=config,
    )
    assert first.predictions == second.predictions
    assert first.predictions
    assert first_diag["tr_a1"] == second_diag["tr_a1"]
    assert first_diag["c1_diagnostics"]["complete_video_count"] == 3
    assert first_diag["tr_a1"]["candidate_pool_size_per_event"] == [4, 4]
    assert first_diag["tr_a1"]["restricted_event_video_search_count"] == 6
    assert refiner.calls == 0
    assert all(
        candidate.provenance["source"]
        == TRAKE_VIDEO_FIRST_RESTRICTED_EVENT_SEARCH
        for pool in first_diag["event_candidate_pools"]
        for candidate in pool
    )


def test_production_planner_remains_non_decreasing_and_rank_scored() -> None:
    query = TRAKEQuery(
        "tr",
        (TRAKEEvent(0, "a"), TRAKEEvent(1, "b")),
    )
    pools = (
        (TRAKEEventCandidate("tr", 0, 1, "A", 10, 0.9),),
        (TRAKEEventCandidate("tr", 1, 2, "A", 10, 0.8),),
    )
    predictions, diagnostics = plan_trake_paths(
        query,
        pools,
        rrf_constant=60.0,
    )
    assert predictions[0].frame_ids == (10, 10)
    assert diagnostics["complete_path_count_before_global_topk"] == 1
