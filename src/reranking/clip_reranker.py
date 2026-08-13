"""
CLIP Visual Fine-Grained Reranker.

Reranks fused candidate keyframes by blending their rank-fusion score with
exact CLIP cosine similarity scores. Elevates visually precise candidates
to the top position.
"""

from __future__ import annotations

from typing import Any, List, Optional

from src.reranking.base import BaseReranker
from src.common.types import SearchResult
from src.utils.logger import get_logger

logger = get_logger(__name__)


class CLIPReranker(BaseReranker):
    """
    Reranks candidates using CLIP similarity weighting.

    Args:
        visual_weight: Weight given to raw CLIP similarity (0.0 to 1.0)
        fusion_weight: Weight given to RRF fusion score (0.0 to 1.0)
    """

    def __init__(
        self,
        visual_weight: float = 0.6,
        fusion_weight: float = 0.4,
    ):
        self.visual_weight = visual_weight
        self.fusion_weight = fusion_weight

    @property
    def name(self) -> str:
        return "clip_reranker"

    def rerank(
        self,
        query: Any,
        candidates: List[SearchResult],
        top_k: int = 50,
    ) -> List[SearchResult]:
        """
        Blend raw CLIP score (if present in SearchResult or metadata) with RRF score.
        """
        if not candidates:
            return []

        # Find max fusion score for normalization
        max_fusion_score = max(c.score for c in candidates) or 1.0

        reranked: List[SearchResult] = []
        for cand in candidates:
            # Retrieve clip_score from metadata if stored, or fallback to cand.score
            clip_score = cand.metadata.get("clip_score", cand.score)
            normalized_fusion = cand.score / max_fusion_score

            # Blended score formula
            blended_score = (self.visual_weight * clip_score) + (self.fusion_weight * normalized_fusion)

            # Copy result with updated score
            new_cand = SearchResult(
                keyframe_id=cand.keyframe_id,
                video_id=cand.video_id,
                n=cand.n,
                frame_idx=cand.frame_idx,
                pts_time=cand.pts_time,
                score=blended_score,
                retriever_source=f"{cand.retriever_source}+clip_rerank",
                metadata={
                    **cand.metadata,
                    "raw_fusion_score": cand.score,
                    "clip_score": clip_score,
                },
            )
            reranked.append(new_cand)

        reranked.sort(key=lambda x: x.score, reverse=True)
        return reranked[:top_k]
