"""
Normalized Score Fusion (NSF) for AIC Video Retrieval System.

Unlike RRF which converts scores to 1/(k+rank) (losing magnitude information),
NSF preserves the actual score distribution from each retriever by normalizing
scores to [0, 1] range using min-max normalization within each retriever's results.

Formula:
    NSF_score(d) = Σ  weight_i × norm_score_i(d)
    where norm_score_i(d) = (score_i(d) - min_i) / (max_i - min_i)

Benefits over RRF:
- Preserves score magnitude: a cosine=0.30 candidate is distinguishable from 0.22
- Score-proportional fusion: high-confidence matches get proportionally higher fused scores
- Better for systems with 1-3 retrievers (RRF shines with many diverse retrievers)
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional

from src.common.types import SearchResult
from src.utils.logger import get_logger

logger = get_logger(__name__)


class NormalizedScoreFusion:
    """
    Merge results from N retrievers using Normalized Score Fusion.

    Usage:
        nsf = NormalizedScoreFusion()
        fused = nsf.fuse(
            result_lists=[visual_results, ocr_results],
            weights=[1.0, 0.8],
            top_k=100,
        )
    """

    def __init__(self, min_score_threshold: float = 0.0):
        """
        Args:
            min_score_threshold: Minimum normalized score to include in output.
                                 Set to 0.0 to include all candidates.
        """
        self.min_score_threshold = min_score_threshold

    def fuse(
        self,
        result_lists: List[List[SearchResult]],
        weights: Optional[List[float]] = None,
        top_k: int = 100,
        max_per_video: int = 5,
        query_topic: Optional[str] = None,
        topic_boost_weight: float = 0.15,
    ) -> List[SearchResult]:
        """
        Fuse multiple ranked result lists using normalized score combination.

        Args:
            result_lists: Each inner list is a ranked result set from one retriever.
            weights:      Per-retriever multiplier (default: all 1.0).
            top_k:        Return at most this many results.
            max_per_video: Max keyframes per video_id (default: 5).
            query_topic:   Optional classified topic for soft-scoring boost.
            topic_boost_weight: Bonus multiplier for topic-matching candidates.

        Returns:
            Merged list of SearchResult sorted by fused score descending.
        """
        if not result_lists:
            return []

        if weights is None:
            weights = [1.0] * len(result_lists)

        if len(weights) != len(result_lists):
            raise ValueError("len(weights) must equal len(result_lists)")

        # Accumulate normalized scores per keyframe_id
        fused_scores: Dict[str, float] = defaultdict(float)
        # Keep best representative SearchResult per keyframe
        best_result: Dict[str, SearchResult] = {}
        # Track contributing sources
        contributing_sources: Dict[str, List[str]] = defaultdict(list)
        # Track raw scores for debugging
        raw_scores: Dict[str, Dict[str, float]] = defaultdict(dict)

        for list_idx, (rank_list, weight) in enumerate(zip(result_lists, weights)):
            if not rank_list:
                continue

            # Compute min-max for this retriever's result set
            scores = [r.score for r in rank_list]
            min_score = min(scores)
            max_score = max(scores)
            score_range = max_score - min_score

            for result in rank_list:
                kid = result.keyframe_id

                # Min-max normalize to [0, 1]
                if score_range > 1e-8:
                    norm_score = (result.score - min_score) / score_range
                else:
                    norm_score = 1.0  # All scores identical → max normalized

                weighted_score = weight * norm_score
                fused_scores[kid] += weighted_score
                contributing_sources[kid].append(result.retriever_source)
                raw_scores[kid][f"list_{list_idx}"] = result.score

                # Keep the result with highest individual score
                if kid not in best_result or result.score > best_result[kid].score:
                    best_result[kid] = result

        if not fused_scores:
            return []

        # Apply Topic Soft-Scoring boost
        final_scores: Dict[str, float] = {}
        boosted_count = 0
        for kid, score in fused_scores.items():
            ref = best_result[kid]
            candidate_topic = ref.metadata.get("topic_category", "")
            if query_topic and candidate_topic and query_topic == candidate_topic:
                final_scores[kid] = score * (1.0 + topic_boost_weight)
                boosted_count += 1
            else:
                final_scores[kid] = score

        if query_topic and boosted_count > 0:
            logger.info(
                f"  • NSF Topic Boost: +{topic_boost_weight*100:.0f}% to "
                f"{boosted_count} candidates matching '{query_topic}'"
            )

        # Build fused result list with video-level deduplication
        fused: List[SearchResult] = []
        video_counts: Dict[str, int] = {}

        for kid, fused_score in sorted(final_scores.items(), key=lambda x: -x[1]):
            if fused_score < self.min_score_threshold:
                continue

            ref = best_result[kid]
            if max_per_video > 0:
                cnt = video_counts.get(ref.video_id, 0)
                if cnt >= max_per_video:
                    continue
                video_counts[ref.video_id] = cnt + 1

            n_sources = len(contributing_sources[kid])
            fused.append(SearchResult(
                keyframe_id=ref.keyframe_id,
                video_id=ref.video_id,
                n=ref.n,
                frame_idx=ref.frame_idx,
                pts_time=ref.pts_time,
                score=fused_score,
                retriever_source="fusion_nsf",
                metadata={
                    **ref.metadata,
                    "sources": contributing_sources[kid],
                    "n_sources": n_sources,
                    "raw_clip_score": ref.metadata.get("clip_score", ref.score),
                    # Multi-source bonus: candidates found by multiple retrievers are more reliable
                    "multi_source_bonus": n_sources > 1,
                },
            ))
            if len(fused) >= top_k:
                break

        # Log top candidates
        if fused:
            logger.info("  • Top NSF Fused Candidates:")
            for rank_idx, cand in enumerate(fused[:3], start=1):
                srcs = ",".join(dict.fromkeys(cand.metadata.get("sources", [])))
                cand_topic = cand.metadata.get("topic_category", "N/A")
                logger.info(
                    f"    Rank #{rank_idx}: {cand.video_id} "
                    f"(frame={cand.frame_idx}, pts={cand.pts_time:.1f}s) | "
                    f"Score={cand.score:.4f} | Topic={cand_topic} | Sources=[{srcs}]"
                )

        logger.debug(
            f"[NSF] Fused {len(result_lists)} lists → {len(fused)} results "
            f"from {len(video_counts)} videos"
        )
        return fused

    def fuse_video_level(
        self,
        result_lists: List[List[SearchResult]],
        weights: Optional[List[float]] = None,
        top_k: int = 10,
    ) -> List[SearchResult]:
        """
        Fuse at the VIDEO level — used in TRAKE Phase 1.

        For each video, uses a combination of:
        - Best keyframe score (peak signal)
        - Number of keyframes in top results (breadth signal)
        """
        # First do keyframe-level fusion with generous limits
        kf_fused = self.fuse(
            result_lists, weights, top_k=5000, max_per_video=50
        )

        # Aggregate per video: weighted combination of best score + breadth
        video_scores: Dict[str, float] = {}
        video_best: Dict[str, SearchResult] = {}
        video_counts: Dict[str, int] = defaultdict(int)

        for result in kf_fused:
            vid = result.video_id
            video_counts[vid] += 1
            if vid not in video_best or result.score > video_best[vid].score:
                video_best[vid] = result

        if not video_best:
            return []

        # Score = best_keyframe_score + 0.1 * log(n_keyframes_in_top)
        import math
        for vid, best in video_best.items():
            breadth_bonus = 0.1 * math.log1p(video_counts[vid])
            video_scores[vid] = best.score + breadth_bonus

        # Sort and return
        video_ranked = sorted(
            video_best.values(),
            key=lambda r: video_scores[r.video_id],
            reverse=True,
        )
        return video_ranked[:top_k]
