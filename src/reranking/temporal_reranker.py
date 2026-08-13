"""
Temporal Continuity Reranker.

Boosts candidate keyframes that have temporal support (adjacent keyframes
matching in the same video), signaling a stable visual event rather than a
single-frame retrieval artifact.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List

from src.reranking.base import BaseReranker
from src.common.types import SearchResult
from src.utils.logger import get_logger

logger = get_logger(__name__)


class TemporalReranker(BaseReranker):
    """
    Reranks candidates by analyzing temporal cluster density within each video.

    Args:
        temporal_window_sec: Window in seconds to consider keyframes as temporally contiguous (default: 15.0s)
        density_boost: Score multiplier per additional keyframe in the temporal neighborhood (default: 0.10)
    """

    def __init__(
        self,
        temporal_window_sec: float = 15.0,
        density_boost: float = 0.10,
    ):
        self.temporal_window_sec = temporal_window_sec
        self.density_boost = density_boost

    @property
    def name(self) -> str:
        return "temporal_reranker"

    def rerank(
        self,
        query: Any,
        candidates: List[SearchResult],
        top_k: int = 50,
    ) -> List[SearchResult]:
        if not candidates:
            return []

        # Group candidates by video_id
        video_groups: Dict[str, List[SearchResult]] = defaultdict(list)
        for cand in candidates:
            video_groups[cand.video_id].append(cand)

        reranked: List[SearchResult] = []

        for cand in candidates:
            same_video_cands = video_groups[cand.video_id]
            # Count how many keyframes in same video fall within temporal_window_sec
            neighbors = sum(
                1 for other in same_video_cands
                if other.keyframe_id != cand.keyframe_id
                and abs(other.pts_time - cand.pts_time) <= self.temporal_window_sec
            )

            # Boost score based on temporal density
            boost_factor = 1.0 + (min(neighbors, 3) * self.density_boost)
            boosted_score = cand.score * boost_factor

            new_cand = SearchResult(
                keyframe_id=cand.keyframe_id,
                video_id=cand.video_id,
                n=cand.n,
                frame_idx=cand.frame_idx,
                pts_time=cand.pts_time,
                score=boosted_score,
                retriever_source=cand.retriever_source,
                metadata={
                    **cand.metadata,
                    "temporal_neighbors": neighbors,
                    "temporal_boost": round(boost_factor, 2),
                },
            )
            reranked.append(new_cand)

        reranked.sort(key=lambda x: x.score, reverse=True)
        return reranked[:top_k]
