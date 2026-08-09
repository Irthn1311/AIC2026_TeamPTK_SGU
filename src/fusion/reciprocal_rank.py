"""
Reciprocal Rank Fusion (RRF) for AIC Video Retrieval System.

RRF combines ranked lists from multiple retrievers into a single unified
ranking without requiring score normalisation.

Formula:  RRF_score(d) = Σ  1 / (k + rank_i(d))
           over all retrievers i

where k=60 is the standard constant (Cormack et al., 2009).

Benefits:
- Score-agnostic: works regardless of each retriever's score scale
- Robust to outliers in any single retriever
- Simple and parameter-light
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional

from src.common.types import SearchResult
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Standard RRF constant
_DEFAULT_K = 60


class ReciprocalRankFusion:
    """
    Merge results from N retrievers using Reciprocal Rank Fusion.

    Usage:
        rrf = ReciprocalRankFusion(k=60)
        fused = rrf.fuse(
            result_lists=[visual_results, caption_results, ocr_results],
            weights=[1.0, 0.8, 0.6],   # optional per-retriever weight
            top_k=50,
        )
    """

    def __init__(self, k: int = _DEFAULT_K):
        """
        Args:
            k: RRF constant (higher k reduces the impact of top ranks).
               Default 60 works well across many retrieval tasks.
        """
        self.k = k

    def fuse(
        self,
        result_lists: List[List[SearchResult]],
        weights: Optional[List[float]] = None,
        top_k: int = 50,
        max_per_video: int = 3,
    ) -> List[SearchResult]:
        """
        Fuse multiple ranked result lists into one.

        Args:
            result_lists: Each inner list is a ranked result set from one retriever.
                          Empty lists are silently skipped.
            weights:      Per-retriever multiplier on the RRF score.
                          If None, all retrievers are weighted equally (1.0).
            top_k:        Return at most this many results.
            max_per_video: Max keyframes per video_id in fused output (default: 3).

        Returns:
            Merged list of SearchResult sorted by fused score descending.
            The retriever_source field is set to "fusion_rrf".
            Original retriever sources are stored in metadata["sources"].
        """
        if not result_lists:
            return []

        if weights is None:
            weights = [1.0] * len(result_lists)

        if len(weights) != len(result_lists):
            raise ValueError("len(weights) must equal len(result_lists)")

        # Accumulate RRF scores per keyframe_id
        rrf_scores: Dict[str, float] = defaultdict(float)
        # Keep one representative SearchResult per keyframe_id
        best_result: Dict[str, SearchResult] = {}
        # Track which retrievers contributed
        contributing_sources: Dict[str, List[str]] = defaultdict(list)

        for rank_list, weight in zip(result_lists, weights):
            if not rank_list:
                continue
            for rank, result in enumerate(rank_list, start=1):
                kid = result.keyframe_id
                rrf_score = weight / (self.k + rank)
                rrf_scores[kid] += rrf_score
                contributing_sources[kid].append(result.retriever_source)

                # Keep the result metadata from the highest-scoring individual retriever
                if kid not in best_result or result.score > best_result[kid].score:
                    best_result[kid] = result

        if not rrf_scores:
            return []

        # Build fused result list with optional video-level deduplication
        fused: List[SearchResult] = []
        video_counts: Dict[str, int] = {}

        for kid, fused_score in sorted(rrf_scores.items(), key=lambda x: -x[1]):
            ref = best_result[kid]
            if max_per_video > 0:
                cnt = video_counts.get(ref.video_id, 0)
                if cnt >= max_per_video:
                    continue
                video_counts[ref.video_id] = cnt + 1

            fused.append(SearchResult(
                keyframe_id=ref.keyframe_id,
                video_id=ref.video_id,
                n=ref.n,
                frame_idx=ref.frame_idx,
                pts_time=ref.pts_time,
                score=fused_score,
                retriever_source="fusion_rrf",
                metadata={
                    **ref.metadata,
                    "sources": contributing_sources[kid],
                    "n_sources": len(contributing_sources[kid]),
                },
            ))
            if len(fused) >= top_k:
                break

        logger.debug(
            f"[RRF] Fused {len(result_lists)} lists → {len(fused)} results from {len(video_counts)} videos "
            f"(top score={fused[0].score:.4f} if fused else 'N/A')"
        )
        return fused

    def fuse_video_level(
        self,
        result_lists: List[List[SearchResult]],
        weights: Optional[List[float]] = None,
        top_k: int = 10,
    ) -> List[SearchResult]:
        """
        Fuse at the VIDEO level instead of keyframe level.
        Used in TRAKE Phase 1 to rank videos by overall relevance.

        For each video, takes the best keyframe score as the video score.

        Returns one representative SearchResult per video.
        """
        # First do keyframe-level fusion
        kf_fused = self.fuse(result_lists, weights, top_k=10000)

        # Group by video, take highest-scoring keyframe as representative
        best_per_video: Dict[str, SearchResult] = {}
        for result in kf_fused:
            vid = result.video_id
            if vid not in best_per_video or result.score > best_per_video[vid].score:
                best_per_video[vid] = result

        video_ranked = sorted(best_per_video.values(), key=lambda r: -r.score)
        return video_ranked[:top_k]
