"""Unit tests for HybridFusionRetriever (Vector + BM25 Fusion)."""

from __future__ import annotations

import numpy as np
import pytest

from system_tai.common.schemas import CandidateFrame, KISQuery, KISResult
from system_tai.retrieval.bm25_search import BM25Document, BM25TextRetriever
from system_tai.retrieval.hybrid_fusion import HybridFusionRetriever


class MockVectorRetriever:
    def __init__(self, candidates: list[CandidateFrame]) -> None:
        self.candidates = candidates

    def retrieve(self, query: KISQuery) -> KISResult:
        return KISResult(
            query_id=query.query_id,
            ranked_candidates=tuple(self.candidates[:query.top_k]),
        )


def test_hybrid_fusion_rrf_scoring() -> None:
    vec_candidates = [
        CandidateFrame("V1", 100, 0, 1, 0.95, 1, "clip"),
        CandidateFrame("V2", 200, 1, 2, 0.85, 2, "clip"),
    ]
    vec_retriever = MockVectorRetriever(vec_candidates)

    text_retriever = BM25TextRetriever()
    text_retriever.add_documents([
        BM25Document("d1", "V2", 200, "người đi bộ qua cầu"),
        BM25Document("d2", "V3", 300, "người lái xe qua cầu"),
    ])

    fusion = HybridFusionRetriever(
        vector_retriever=vec_retriever,
        text_retriever=text_retriever,
        w_vector=1.0,
        w_text=1.0,
        rrf_constant=60.0,
    )

    query = KISQuery(query_id="Q1", text="người qua cầu", top_k=3)
    res = fusion.retrieve(query)

    assert res.query_id == "Q1"
    assert len(res.ranked_candidates) == 3

    # V2 (frame 200) appears in BOTH Vector rank 2 and BM25 rank 1!
    # Expected RRF for V2 = 1.0/(60+2) + 1.0/(60+1) = 1/62 + 1/61 = 0.01613 + 0.01639 = ~0.0325
    # Expected RRF for V1 = 1.0/(60+1) = 1/61 = ~0.01639 (from Vector rank 1)
    # Expected RRF for V3 = 1.0/(60+2) = 1/62 = ~0.01613 (from BM25 rank 2)
    # Therefore, V2 should rank 1st due to multimodal reinforcement!
    assert res.ranked_candidates[0].video_id == "V2"
    assert res.ranked_candidates[0].frame_id == 200
    assert res.ranked_candidates[0].rank == 1

    assert res.ranked_candidates[1].video_id == "V1"
    assert res.ranked_candidates[1].frame_id == 100
    assert res.ranked_candidates[1].rank == 2

    assert res.ranked_candidates[2].video_id == "V3"
    assert res.ranked_candidates[2].frame_id == 300
    assert res.ranked_candidates[2].rank == 3
