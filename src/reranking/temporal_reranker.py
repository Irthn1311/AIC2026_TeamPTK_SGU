"""
Temporal Continuity Reranker (v2 — Enhanced Video-Context Scoring).

Boosts candidate keyframes that have temporal support (adjacent keyframes
matching in the same video), signaling a stable visual event rather than a
single-frame retrieval artifact.

v2 Changes:
- Increased density_boost from 0.10 to 0.15 per neighbor
- Added video-context scoring: if 3+ keyframes from the same video appear
  in the candidate pool, ALL of them get a reliability bonus
- Max neighbor cap increased from 3 to 4
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List

from src.reranking.base import BaseReranker
from src.common.types import SearchResult
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Minimum keyframes in a video to trigger video-context bonus
_VIDEO_CONTEXT_MIN = 3
# Bonus for video-context (applied on top of temporal density)
_VIDEO_CONTEXT_BONUS = 1.08


class TemporalReranker(BaseReranker):
    """
    Reranks candidates by analyzing temporal cluster density within each video.

    Args:
        temporal_window_sec: Window in seconds to consider keyframes as temporally contiguous (default: 15.0s)
        density_boost: Score multiplier per additional keyframe in the temporal neighborhood (default: 0.15)
    """

    def __init__(
        self,
        temporal_window_sec: float = 15.0,
        density_boost: float = 0.15,
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

        # Identify videos with strong context (3+ keyframes in pool)
        strong_context_videos = {
            vid for vid, group in video_groups.items()
            if len(group) >= _VIDEO_CONTEXT_MIN
        }

        reranked: List[SearchResult] = []

        for cand in candidates:
            same_video_cands = video_groups[cand.video_id]
            # Count how many keyframes in same video fall within temporal_window_sec
            neighbors = sum(
                1 for other in same_video_cands
                if other.keyframe_id != cand.keyframe_id
                and abs(other.pts_time - cand.pts_time) <= self.temporal_window_sec
            )

            # Boost score based on temporal density (capped at 4 neighbors)
            boost_factor = 1.0 + (min(neighbors, 4) * self.density_boost)

            # Video-context bonus: multiple keyframes from same video = likely correct video
            if cand.video_id in strong_context_videos:
                boost_factor *= _VIDEO_CONTEXT_BONUS

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
                    "temporal_boost": round(boost_factor, 3),
                    "video_context": cand.video_id in strong_context_videos,
                },
            )
            reranked.append(new_cand)

        reranked.sort(key=lambda x: x.score, reverse=True)
        return reranked[:top_k]
