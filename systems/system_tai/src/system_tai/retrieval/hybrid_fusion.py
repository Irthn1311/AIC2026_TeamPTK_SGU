"""Hybrid Multimodal Fusion combining Vector Retrieval (CLIP/FAISS) and BM25 Text Search."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from system_tai.common.schemas import CandidateFrame, KISQuery, KISResult
from system_tai.features.query_encoder import TextEncoder
from system_tai.retrieval.bm25_search import BM25TextRetriever


class HybridFusionRetriever:
    """Combines Vector Search (FAISS/NumPy) and BM25 Text Search via Weighted RRF."""

    def __init__(
        self,
        vector_retriever: Any,
        text_retriever: BM25TextRetriever,
        text_encoder: TextEncoder | None = None,
        *,
        w_vector: float = 1.0,
        w_text: float = 0.5,
        rrf_constant: float = 60.0,
    ) -> None:
        if w_vector < 0 or w_text < 0:
            raise ValueError("weights must be non-negative")
        if w_vector == 0 and w_text == 0:
            raise ValueError("at least one weight must be positive")
        if rrf_constant <= 0:
            raise ValueError("rrf_constant must be positive")

        self.vector_retriever = vector_retriever
        self.text_retriever = text_retriever
        self.text_encoder = text_encoder
        self.w_vector = w_vector
        self.w_text = w_text
        self.rrf_constant = rrf_constant

    def retrieve(self, query: KISQuery) -> KISResult:
        query_vector = None
        if self.text_encoder is not None:
            query_vector = self.text_encoder.encode(query.text)
        return self.search_hybrid(
            query_id=query.query_id,
            query_text=query.text,
            query_vector=query_vector,
            top_k=query.top_k,
        )

    def search_hybrid(
        self,
        *,
        query_id: str,
        query_text: str,
        query_vector: Sequence[float] | NDArray[np.number] | None = None,
        top_k: int = 100,
    ) -> KISResult:
        if not query_id.strip():
            raise ValueError("query_id must not be empty")
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        candidate_info: dict[tuple[str, int], CandidateFrame] = {}
        rrf_scores: dict[tuple[str, int], float] = {}

        # 1. Vector Search Branch
        if self.w_vector > 0 and (query_vector is not None or hasattr(self.vector_retriever, "retrieve")):
            if query_vector is not None and hasattr(self.vector_retriever, "search_vector"):
                vec_result = self.vector_retriever.search_vector(
                    query_id=query_id,
                    query_vector=query_vector,
                    top_k=top_k * 2,
                )
            else:
                kis_q = KISQuery(query_id=query_id, text=query_text, top_k=top_k * 2)
                vec_result = self.vector_retriever.retrieve(kis_q)

            for cand in vec_result.ranked_candidates:
                key = (cand.video_id, cand.frame_id)
                candidate_info[key] = cand
                r_score = self.w_vector / (self.rrf_constant + cand.rank)
                rrf_scores[key] = rrf_scores.get(key, 0.0) + r_score

        # 2. Text BM25 Search Branch
        if self.w_text > 0 and query_text.strip():
            text_result = self.text_retriever.search_text(
                query_id=query_id,
                query_text=query_text,
                top_k=top_k * 2,
            )
            for cand in text_result.ranked_candidates:
                key = (cand.video_id, cand.frame_id)
                if key not in candidate_info:
                    candidate_info[key] = cand
                r_score = self.w_text / (self.rrf_constant + cand.rank)
                rrf_scores[key] = rrf_scores.get(key, 0.0) + r_score

        # 3. Sort fused candidates
        sorted_keys = sorted(
            rrf_scores.keys(),
            key=lambda k: (-rrf_scores[k], k[0], k[1]),
        )[:top_k]

        fused_candidates = tuple(
            CandidateFrame(
                video_id=key[0],
                frame_id=key[1],
                clip_row=candidate_info[key].clip_row,
                keyframe_order=candidate_info[key].keyframe_order,
                score=float(rrf_scores[key]),
                rank=rank,
                source="hybrid_vector_bm25",
                diagnostic_metadata={
                    "w_vector": self.w_vector,
                    "w_text": self.w_text,
                    "rrf_constant": self.rrf_constant,
                },
            )
            for rank, key in enumerate(sorted_keys, start=1)
        )

        return KISResult(
            query_id=query_id,
            ranked_candidates=fused_candidates,
        )
