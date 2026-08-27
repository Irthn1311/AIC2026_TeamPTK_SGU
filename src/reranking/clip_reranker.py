"""
CLIP Visual Fine-Grained Reranker (v2 — Fixed Normalization).

Reranks fused candidate keyframes by blending their rank-fusion score with
exact CLIP cosine similarity scores. Elevates visually precise candidates
to the top position.

v2 Changes:
- Fixed score normalization: both CLIP and fusion scores are now min-max
  normalized to [0, 1] before blending, preventing fusion score from
  dominating due to different scale ranges.
- CLIP weight increased to 0.70 (from 0.60) — CLIP cosine similarity
  is the primary signal for visual retrieval accuracy.
"""

from __future__ import annotations

from typing import Any, List, Optional

from src.reranking.base import BaseReranker
from src.common.types import SearchResult
from src.utils.logger import get_logger

logger = get_logger(__name__)


class CLIPReranker(BaseReranker):
    """
    Reranks candidates using properly normalized CLIP similarity weighting.

    Args:
        visual_weight: Weight given to normalized CLIP similarity (0.0 to 1.0)
        fusion_weight: Weight given to normalized fusion score (0.0 to 1.0)
    """

    def __init__(
        self,
        visual_weight: float = 0.70,
        fusion_weight: float = 0.30,
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
        Blend normalized CLIP score with normalized fusion score.

        Both scores are min-max normalized to [0, 1] before blending,
        ensuring neither score dominates due to scale differences.
        """
        if not candidates:
            return []

        # Collect all CLIP scores and fusion scores
        clip_scores = []
        fusion_scores = []
        for cand in candidates:
            clip_s = cand.metadata.get("clip_score", cand.score)
            clip_scores.append(float(clip_s))
            fusion_scores.append(float(cand.score))

        # Min-max normalization for CLIP scores
        min_clip = min(clip_scores)
        max_clip = max(clip_scores)
        clip_range = max_clip - min_clip

        # Min-max normalization for fusion scores
        min_fusion = min(fusion_scores)
        max_fusion = max(fusion_scores)
        fusion_range = max_fusion - min_fusion

        reranked: List[SearchResult] = []
        for i, cand in enumerate(candidates):
            clip_score = clip_scores[i]
            fusion_score = fusion_scores[i]

            # Normalize both to [0, 1]
            if clip_range > 1e-8:
                norm_clip = (clip_score - min_clip) / clip_range
            else:
                norm_clip = 1.0

            if fusion_range > 1e-8:
                norm_fusion = (fusion_score - min_fusion) / fusion_range
            else:
                norm_fusion = 1.0

            # Blended score with proper normalization
            blended_score = (self.visual_weight * norm_clip) + (self.fusion_weight * norm_fusion)

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
                    "raw_fusion_score": fusion_score,
                    "clip_score": clip_score,
                    "norm_clip": round(norm_clip, 4),
                    "norm_fusion": round(norm_fusion, 4),
                },
            )
            reranked.append(new_cand)

        reranked.sort(key=lambda x: x.score, reverse=True)
        return reranked[:top_k]
