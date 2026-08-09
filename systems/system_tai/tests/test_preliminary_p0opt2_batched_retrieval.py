import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from system_tai.common.schemas import (
    CandidateFrame,
    FrameMappingRecord,
    KISResult,
    VideoFeatureStore,
)
from system_tai.features.btc_clip_store import (
    FeatureStoreRegistry,
    LoadedVideoFeatureStore,
)
from system_tai.features.query_encoder import TextEncoder
from system_tai.kis.session_schema import parse_session_request
from system_tai.refinement.models import RefinementConfig
from system_tai.retrieval.vector_search import ExactNumpyRetriever
from system_tai.trake.runtime import TRAKERuntimePipeline

# ---------------------------------------------------------------------------
# Dummy/Mock Helper Types for Unit Tests
# ---------------------------------------------------------------------------


class DummyTextEncoder(TextEncoder):
    def __init__(self, dim: int = 4) -> None:
        self.dim = dim

    @property
    def dimension(self) -> int:
        return self.dim

    @property
    def identifiers(self) -> dict[str, str]:
        return {"model": "dummy", "device": "cpu"}

    def encode(self, text: str) -> np.ndarray:
        return np.ones(self.dim, dtype=np.float32)


class CountingMatrixProxy:
    def __init__(self, data: np.ndarray) -> None:
        self.data = data
        self.slice_count = 0

    def __getitem__(self, key: Any) -> np.ndarray:
        self.slice_count += 1
        return self.data[key]

    def __len__(self) -> int:
        return len(self.data)


def _build_test_registry(
    rows: np.ndarray,
    mappings: list[tuple[str, int, int]],
    dim: int = 4,
) -> tuple[FeatureStoreRegistry, CountingMatrixProxy]:
    video_id = mappings[0][0] if mappings else "V001"
    desc = VideoFeatureStore(
        video_id=video_id,
        mapping_csv_path=Path(f"/tmp/{video_id}.csv"),
        clip_npy_path=Path(f"/tmp/{video_id}.npy"),
        row_count=len(rows),
        embedding_dimension=dim,
        normalized=True,
    )
    chunk_mappings = tuple(
        FrameMappingRecord(
            clip_row=i,
            keyframe_order=k_ord,
            frame_id=f_id,
            pts_time=float(f_id) / 10.0,
            fps=25.0,
        )
        for i, (_, f_id, k_ord) in enumerate(mappings)
    )
    matrix_proxy = CountingMatrixProxy(np.asarray(rows, dtype=np.float32))
    store = LoadedVideoFeatureStore(
        descriptor=desc,
        matrix=matrix_proxy,  # type: ignore[arg-type]
        mappings=chunk_mappings,
    )
    registry = FeatureStoreRegistry(stores=[store])
    return registry, matrix_proxy


# ---------------------------------------------------------------------------
# Section 21: Required Retriever Unit Tests
# ---------------------------------------------------------------------------


def test_single_query_exact_equivalence() -> None:
    rows = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.5, 0.5, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    mappings = [("V1", 10, 1), ("V1", 20, 2), ("V1", 30, 3)]
    reg, _ = _build_test_registry(rows, mappings)
    retriever = ExactNumpyRetriever(reg, DummyTextEncoder(4), chunk_size=2)

    vec = [1.0, 0.2, 0.0, 0.0]
    legacy = retriever.search_vector(query_id="q1", query_vector=vec, top_k=2)
    batched = retriever.search_vectors(query_ids=["q1"], query_vectors=[vec], top_k=2)["q1"]

    assert legacy == batched


def test_multi_query_exact_equivalence() -> None:
    rows = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.7, 0.7, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    mappings = [("V1", 10, 1), ("V1", 20, 2), ("V1", 30, 3), ("V1", 40, 4)]
    reg, _ = _build_test_registry(rows, mappings)
    retriever = ExactNumpyRetriever(reg, DummyTextEncoder(4), chunk_size=2)

    q_vectors = [
        [1.0, 0.1, 0.0, 0.0],
        [0.0, 0.9, 0.1, 0.0],
        [0.2, 0.2, 0.8, 0.0],
    ]
    q_ids = ["q1", "q2", "q3"]

    batched_dict = retriever.search_vectors(query_ids=q_ids, query_vectors=q_vectors, top_k=3)

    for qid, vec in zip(q_ids, q_vectors):
        legacy = retriever.search_vector(query_id=qid, query_vector=vec, top_k=3)
        batched = batched_dict[qid]
        assert legacy == batched
        for c_leg, c_batch in zip(legacy.ranked_candidates, batched.ranked_candidates):
            assert c_leg.score == c_batch.score
            assert c_leg.video_id == c_batch.video_id
            assert c_leg.frame_id == c_batch.frame_id
            assert c_leg.clip_row == c_batch.clip_row
            assert c_leg.rank == c_batch.rank


def test_six_query_exact_equivalence() -> None:
    np.random.seed(42)
    rows = np.random.randn(50, 4).astype(np.float32)
    mappings = [("V1", i * 10, 1) for i in range(50)]
    reg, _ = _build_test_registry(rows, mappings)
    retriever = ExactNumpyRetriever(reg, DummyTextEncoder(4), chunk_size=16)

    q_ids = [f"q{i}" for i in range(6)]
    q_vecs = [np.random.randn(4).astype(np.float32) for _ in range(6)]

    batched_dict = retriever.search_vectors(query_ids=q_ids, query_vectors=q_vecs, top_k=5)

    all_results_exact = True
    all_scores_exact = True

    for i, (qid, vec) in enumerate(zip(q_ids, q_vecs)):
        legacy = retriever.search_vector(query_id=qid, query_vector=vec, top_k=5)
        batched = batched_dict[qid]
        q_exact = legacy == batched
        assert q_exact, f"Query q{i} result mismatch"
        scores_equal = [
            c_leg.score == c_batch.score
            for c_leg, c_batch in zip(legacy.ranked_candidates, batched.ranked_candidates)
        ]
        assert all(scores_equal), f"Query q{i} score mismatch"
        if not q_exact:
            all_results_exact = False
        if not all(scores_equal):
            all_scores_exact = False

    assert all_results_exact
    assert all_scores_exact


def test_non_unit_query_vector_equivalence() -> None:
    rows = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
    mappings = [("V1", 10, 1), ("V1", 20, 2)]
    reg, _ = _build_test_registry(rows, mappings)
    retriever = ExactNumpyRetriever(reg, DummyTextEncoder(4))

    non_unit_vec = [10.0, 5.0, 0.0, 0.0]
    legacy = retriever.search_vector(query_id="q1", query_vector=non_unit_vec, top_k=2)
    batched = retriever.search_vectors(
        query_ids=["q1"], query_vectors=[non_unit_vec], top_k=2
    )["q1"]

    assert legacy == batched


def test_duplicate_frame_identity() -> None:
    # Two feature rows mapping to same (V1, 10)
    rows = np.array([[1.0, 0.0, 0.0, 0.0], [0.9, 0.1, 0.0, 0.0]], dtype=np.float32)
    mappings = [("V1", 10, 1), ("V1", 10, 2)]
    reg, _ = _build_test_registry(rows, mappings)
    retriever = ExactNumpyRetriever(reg, DummyTextEncoder(4))

    vec = [1.0, 0.0, 0.0, 0.0]
    legacy = retriever.search_vector(query_id="q1", query_vector=vec, top_k=2)
    batched = retriever.search_vectors(query_ids=["q1"], query_vectors=[vec], top_k=2)["q1"]

    assert len(legacy.ranked_candidates) == 1
    assert len(batched.ranked_candidates) == 1
    assert legacy == batched
    assert legacy.ranked_candidates[0].clip_row == 0


def test_deterministic_tie() -> None:
    rows = np.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    # Distinct frame_ids (10 and 20), tied scores (1.0), different clip_rows (0 and 1)
    mappings = [("V1", 10, 1), ("V1", 20, 2)]
    reg, _ = _build_test_registry(rows, mappings)
    retriever = ExactNumpyRetriever(reg, DummyTextEncoder(4))

    vec = [1.0, 0.0, 0.0, 0.0]
    legacy = retriever.search_vector(query_id="q1", query_vector=vec, top_k=2)
    batched = retriever.search_vectors(query_ids=["q1"], query_vectors=[vec], top_k=2)["q1"]

    # Exact legacy 4-field sort key: (-score, video_id, frame_id, clip_row)
    assert legacy == batched
    assert legacy.ranked_candidates[0].frame_id == 10
    assert legacy.ranked_candidates[1].frame_id == 20
    assert legacy.ranked_candidates[0].score == legacy.ranked_candidates[1].score


def test_query_id_order() -> None:
    rows = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    mappings = [("V1", 10, 1)]
    reg, _ = _build_test_registry(rows, mappings)
    retriever = ExactNumpyRetriever(reg, DummyTextEncoder(4))

    q_ids = ["z", "a", "m"]
    q_vecs = [[1.0, 0.0, 0.0, 0.0] for _ in q_ids]
    res = retriever.search_vectors(query_ids=q_ids, query_vectors=q_vecs, top_k=1)

    assert list(res.keys()) == ["z", "a", "m"]


def test_duplicate_query_ids_rejected() -> None:
    rows = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    mappings = [("V1", 10, 1)]
    reg, _ = _build_test_registry(rows, mappings)
    retriever = ExactNumpyRetriever(reg, DummyTextEncoder(4))

    with pytest.raises(ValueError, match="query_ids must be unique"):
        retriever.search_vectors(
            query_ids=["q1", "q1"],
            query_vectors=[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
            top_k=1,
        )


def test_empty_query_list_rejected() -> None:
    rows = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    mappings = [("V1", 10, 1)]
    reg, _ = _build_test_registry(rows, mappings)
    retriever = ExactNumpyRetriever(reg, DummyTextEncoder(4))

    with pytest.raises(ValueError, match="query_ids must not be empty"):
        retriever.search_vectors(query_ids=[], query_vectors=[], top_k=1)


def test_query_count_vector_count_mismatch_rejected() -> None:
    rows = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    mappings = [("V1", 10, 1)]
    reg, _ = _build_test_registry(rows, mappings)
    retriever = ExactNumpyRetriever(reg, DummyTextEncoder(4))

    with pytest.raises(ValueError, match="length mismatch"):
        retriever.search_vectors(
            query_ids=["q1", "q2"],
            query_vectors=[[1.0, 0.0, 0.0, 0.0]],
            top_k=1,
        )


def test_empty_query_id_rejected() -> None:
    rows = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    mappings = [("V1", 10, 1)]
    reg, _ = _build_test_registry(rows, mappings)
    retriever = ExactNumpyRetriever(reg, DummyTextEncoder(4))

    with pytest.raises(ValueError, match="query_id must not be empty"):
        retriever.search_vectors(
            query_ids=["  "],
            query_vectors=[[1.0, 0.0, 0.0, 0.0]],
            top_k=1,
        )


def test_zero_norm_query_rejected() -> None:
    rows = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    mappings = [("V1", 10, 1)]
    reg, _ = _build_test_registry(rows, mappings)
    retriever = ExactNumpyRetriever(reg, DummyTextEncoder(4))

    with pytest.raises(ValueError, match="finite non-zero norm"):
        retriever.search_vectors(
            query_ids=["q1"],
            query_vectors=[[0.0, 0.0, 0.0, 0.0]],
            top_k=1,
        )


def test_nonfinite_query_rejected() -> None:
    rows = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    mappings = [("V1", 10, 1)]
    reg, _ = _build_test_registry(rows, mappings)
    retriever = ExactNumpyRetriever(reg, DummyTextEncoder(4))

    with pytest.raises(ValueError, match="NaN or Infinity"):
        retriever.search_vectors(
            query_ids=["q1"],
            query_vectors=[[np.nan, 0.0, 0.0, 0.0]],
            top_k=1,
        )


def test_wrong_dimension_query_rejected() -> None:
    rows = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    mappings = [("V1", 10, 1)]
    reg, _ = _build_test_registry(rows, mappings)
    retriever = ExactNumpyRetriever(reg, DummyTextEncoder(4))

    with pytest.raises(ValueError, match="shape mismatch"):
        retriever.search_vectors(
            query_ids=["q1"],
            query_vectors=[[1.0, 0.0, 0.0]],
            top_k=1,
        )


def test_top_k_positive_validation() -> None:
    rows = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    mappings = [("V1", 10, 1)]
    reg, _ = _build_test_registry(rows, mappings)
    retriever = ExactNumpyRetriever(reg, DummyTextEncoder(4))

    with pytest.raises(ValueError, match="top_k must be positive"):
        retriever.search_vectors(
            query_ids=["q1"],
            query_vectors=[[1.0, 0.0, 0.0, 0.0]],
            top_k=0,
        )


def test_multiple_chunk_sizes() -> None:
    np.random.seed(123)
    rows = np.random.randn(25, 4).astype(np.float32)
    mappings = [("V1", i * 10, 1) for i in range(25)]

    q_ids = ["q0", "q1", "q2"]
    q_vecs = [np.random.randn(4).astype(np.float32) for _ in range(3)]

    for chunk_size in (1, 3, 1000):
        reg, _ = _build_test_registry(rows, mappings)
        retriever = ExactNumpyRetriever(reg, DummyTextEncoder(4), chunk_size=chunk_size)
        batched_dict = retriever.search_vectors(query_ids=q_ids, query_vectors=q_vecs, top_k=4)

        for qid, vec in zip(q_ids, q_vecs):
            legacy = retriever.search_vector(query_id=qid, query_vector=vec, top_k=4)
            assert legacy == batched_dict[qid]


# ---------------------------------------------------------------------------
# Section 23: Shared Corpus Scan Evidence
# ---------------------------------------------------------------------------


def test_shared_corpus_scan_work_reduction() -> None:
    rows = np.random.randn(20, 4).astype(np.float32)
    mappings = [("V1", i * 10, 1) for i in range(20)]

    q_ids = [f"q{i}" for i in range(6)]
    q_vecs = [np.random.randn(4).astype(np.float32) for _ in range(6)]
    chunk_size = 5

    # 1. Legacy sequential search_vector calls
    reg_legacy, proxy_legacy = _build_test_registry(rows, mappings)
    retriever_legacy = ExactNumpyRetriever(
        reg_legacy, DummyTextEncoder(4), chunk_size=chunk_size
    )
    for qid, vec in zip(q_ids, q_vecs):
        retriever_legacy.search_vector(query_id=qid, query_vector=vec, top_k=3)

    legacy_slices = proxy_legacy.slice_count

    # 2. Batched search_vectors call
    reg_batched, proxy_batched = _build_test_registry(rows, mappings)
    retriever_batched = ExactNumpyRetriever(
        reg_batched, DummyTextEncoder(4), chunk_size=chunk_size
    )
    retriever_batched.search_vectors(query_ids=q_ids, query_vectors=q_vecs, top_k=3)

    batched_slices = proxy_batched.slice_count

    # 20 rows / chunk_size 5 = 4 chunks per store scan
    # 6 queries * 4 chunks = 24 chunk slices for legacy
    # 1 batched scan * 4 chunks = 4 chunk slices for batched
    assert legacy_slices == 24
    assert batched_slices == 4
    reduction = (legacy_slices - batched_slices) / legacy_slices * 100.0
    assert abs(reduction - 83.33333333333334) < 1e-4


# ---------------------------------------------------------------------------
# Section 25: TRAKE Runtime Integration Tests
# ---------------------------------------------------------------------------


class MockSharedEncoder:
    def __init__(self, dim: int = 4) -> None:
        self.dim = dim
        self.encode_texts_calls: list[list[str]] = []
        self.identifiers = {"device": "cpu", "model": "mock"}

    @property
    def dimension(self) -> int:
        return self.dim

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        self.encode_texts_calls.append(list(texts))
        vecs = []
        for i, _ in enumerate(texts):
            v = np.zeros(self.dim, dtype=np.float32)
            v[i % self.dim] = 1.0
            vecs.append(v)
        return np.array(vecs, dtype=np.float32)


class StrictMultiVectorRetriever:
    def __init__(self) -> None:
        self.search_vectors_calls: list[dict[str, Any]] = []

    def search_vector(self, query_id: str, query_vector: np.ndarray, top_k: int) -> KISResult:
        raise AssertionError("TRAKE must use batched search_vectors")

    def search_vectors(
        self,
        query_ids: Sequence[str],
        query_vectors: Sequence[np.ndarray],
        top_k: int,
    ) -> dict[str, KISResult]:
        self.search_vectors_calls.append(
            {
                "query_ids": list(query_ids),
                "query_vectors": list(query_vectors),
                "top_k": top_k,
            }
        )
        results: dict[str, KISResult] = {}
        for qid in query_ids:
            cands = (
                CandidateFrame(
                    rank=1,
                    video_id="V001",
                    frame_id=100,
                    score=0.9,
                    clip_row=1,
                    keyframe_order=1,
                    source="clip_exact",
                    diagnostic_metadata={"variant_hit_count": 1, "best_individual_rank": 1},
                ),
            )
            results[qid] = KISResult(query_id=qid, ranked_candidates=cands)
        return results


class MockWeightedRRF:
    def __init__(self) -> None:
        self.fuse_calls: list[dict[str, Any]] = []

    def fuse_rankings(
        self,
        query_id: str,
        variants: Sequence[Any],
        rankings: dict[str, KISResult],
        output_top_k: int,
        rrf_constant: float = 60.0,
    ) -> KISResult:
        self.fuse_calls.append(
            {
                "query_id": query_id,
                "variant_ids": [v.variant_id for v in variants],
                "ranking_keys": list(rankings.keys()),
            }
        )
        cands = (
            CandidateFrame(
                rank=1,
                video_id="V001",
                frame_id=100,
                score=0.9,
                clip_row=1,
                keyframe_order=1,
                source="rrf_fused",
                diagnostic_metadata={"variant_hit_count": 1, "best_individual_rank": 1},
            ),
        )
        return KISResult(query_id=query_id, ranked_candidates=cands)


class MockRefiner:
    def __init__(self) -> None:
        self.refine_query_calls: list[dict[str, Any]] = []

    def refine_query(self, **kwargs: Any) -> Any:
        self.refine_query_calls.append(kwargs)

        class MockRefinementResult:
            status = type("Status", (), {"value": "SUCCESS"})()
            refined_candidates = [
                type(
                    "RefinedCand",
                    (),
                    {
                        "original_candidate": kwargs["candidate_pool"][0],
                        "refined_rank": 1,
                        "video_id": "V001",
                        "best_frame_id": 100,
                        "best_timestamp_seconds": 10.0,
                        "best_score": 0.95,
                        "winning_window": type(
                            "Win",
                            (),
                            {
                                "start_frame_id": 90,
                                "end_frame_id": 110,
                                "sampled_frame_count": 5,
                            },
                        )(),
                    },
                )()
            ]
            diagnostic_metadata = {}

        return MockRefinementResult()


def test_trake_runtime_opt2_gates() -> None:
    encoder = MockSharedEncoder(dim=4)
    retriever = StrictMultiVectorRetriever()
    rrf = MockWeightedRRF()
    refiner = MockRefiner()
    planner = type(
        "MockPlanner",
        (),
        {
            "solve_query": lambda self, **kw: type(
                "Res", (), {"predictions": (), "diagnostics": {}, "diagnostic_metadata": {}}
            )()
        },
    )()

    pipeline = TRAKERuntimePipeline(
        shared_encoder=encoder,  # type: ignore[arg-type]
        exact_retriever=retriever,  # type: ignore[arg-type]
        weighted_rrf=rrf,  # type: ignore[arg-type]
        refiner=refiner,  # type: ignore[arg-type]
        trake_engine=planner,  # type: ignore[arg-type]
    )

    req = parse_session_request(
        json.dumps({
            "type": "trake_query",
            "request_id": "req-opt2-1",
            "query_id": "Q_OPT2",
            "events": [
                {"description": "sự kiện 0", "description_en": "event 0"},
                {"description": "sự kiện 1", "description_en": "event 1"},
                {"description": "sự kiện 2", "description_en": "event 2"},
            ],
            "top_k_per_variant": 10,
        })
    )
    ref_cfg = RefinementConfig()

    res, timings, extra_diag = pipeline.process_trake_query(req, refinement_config=ref_cfg)

    # Gate A: Exactly 1 encode call
    assert len(encoder.encode_texts_calls) == 1
    assert encoder.encode_texts_calls[0] == [
        "sự kiện 0",
        "event 0",
        "sự kiện 1",
        "event 1",
        "sự kiện 2",
        "event 2",
    ]

    # Gate B & C: Exactly 1 search_vectors call, 0 search_vector calls
    assert len(retriever.search_vectors_calls) == 1

    # Gate D: Six variant IDs in exact order
    called_ids = retriever.search_vectors_calls[0]["query_ids"]
    assert called_ids == [
        "Q_OPT2::e0::v1_vi",
        "Q_OPT2::e0::v2_en",
        "Q_OPT2::e1::v1_vi",
        "Q_OPT2::e1::v2_en",
        "Q_OPT2::e2::v1_vi",
        "Q_OPT2::e2::v2_en",
    ]

    # Gate F: Top_k propagation
    assert retriever.search_vectors_calls[0]["top_k"] == 10

    # Gate G: Event-Isolated RRF
    assert len(rrf.fuse_calls) == 3
    assert rrf.fuse_calls[0]["ranking_keys"] == ["Q_OPT2::e0::v1_vi", "Q_OPT2::e0::v2_en"]
    assert rrf.fuse_calls[1]["ranking_keys"] == ["Q_OPT2::e1::v1_vi", "Q_OPT2::e1::v2_en"]
    assert rrf.fuse_calls[2]["ranking_keys"] == ["Q_OPT2::e2::v1_vi", "Q_OPT2::e2::v2_en"]


def test_trake_runtime_vi_only_and_single_event() -> None:
    encoder = MockSharedEncoder(dim=4)
    retriever = StrictMultiVectorRetriever()
    rrf = MockWeightedRRF()
    refiner = MockRefiner()
    planner = type(
        "MockPlanner",
        (),
        {
            "solve_query": lambda self, **kw: type(
                "Res", (), {"predictions": (), "diagnostics": {}, "diagnostic_metadata": {}}
            )()
        },
    )()

    pipeline = TRAKERuntimePipeline(
        shared_encoder=encoder,  # type: ignore[arg-type]
        exact_retriever=retriever,  # type: ignore[arg-type]
        weighted_rrf=rrf,  # type: ignore[arg-type]
        refiner=refiner,  # type: ignore[arg-type]
        trake_engine=planner,  # type: ignore[arg-type]
    )

    # 3 VI-only events
    req3 = parse_session_request(
        json.dumps({
            "type": "trake_query",
            "request_id": "req-vi-3",
            "query_id": "Q_VI_3",
            "events": [
                {"description": "sự kiện 0"},
                {"description": "sự kiện 1"},
                {"description": "sự kiện 2"},
            ],
        })
    )
    pipeline.process_trake_query(req3, refinement_config=RefinementConfig())
    assert len(retriever.search_vectors_calls) == 1
    assert retriever.search_vectors_calls[0]["query_ids"] == [
        "Q_VI_3::e0::v1_vi",
        "Q_VI_3::e1::v1_vi",
        "Q_VI_3::e2::v1_vi",
    ]

    retriever.search_vectors_calls.clear()

    # 1 VI-only event
    req1 = parse_session_request(
        json.dumps({
            "type": "trake_query",
            "request_id": "req-vi-1",
            "query_id": "Q_VI_1",
            "events": [
                {"description": "sự kiện 0"},
            ],
        })
    )
    pipeline.process_trake_query(req1, refinement_config=RefinementConfig())
    assert len(retriever.search_vectors_calls) == 1
    assert retriever.search_vectors_calls[0]["query_ids"] == ["Q_VI_1::e0::v1_vi"]
