# ==============================================================================================================
# Multi-Variant Video-Level RRF Fusion with Full Channel Provenance
# ==============================================================================================================

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from system_tai.retrieval.query_decomposition import QueryVariants, decompose_query


@dataclass(frozen=True, slots=True)
class ChannelContribution:
    channel: str  # e.g., 'clip_b32', 'clip_l14'
    variant_name: str  # e.g., 'literal', 'entity_focused', 'action_focused'
    video_rank: int  # 1-based rank of the video in this channel
    best_frame: int
    raw_score: float


@dataclass(frozen=True, slots=True)
class VideoCandidateWithProvenance:
    video_id: str
    rrf_score: float
    contributions: tuple[ChannelContribution, ...]
    best_anchor_frame: int


@dataclass(frozen=True, slots=True)
class MultiVariantRetrievalResult:
    query_id: str
    variants: QueryVariants
    ranked_videos: tuple[VideoCandidateWithProvenance, ...]
    novel_rescue_videos: tuple[VideoCandidateWithProvenance, ...]


def fuse_multi_variant_video_ranks(
    query_id: str,
    variants: QueryVariants,
    channel_video_rankings: Mapping[str, Sequence[tuple[str, int, float]]],
    # channel_video_rankings maps (e.g. "b32:literal") -> list of (video_id, best_frame_id, raw_score) in ranked order
    baseline_top_video_ids: Sequence[str] = (),
    rrf_k: float = 60.0,
    channel_weights: Mapping[str, float] | None = None,
) -> MultiVariantRetrievalResult:
    """
    Combines video rankings from multiple query variants and channels using weighted RRF.
    Preserves exact per-channel provenance and identifies novel rescue videos not in baseline.
    """
    weights = channel_weights or {}
    rrf_scores: dict[str, float] = {}
    contributions_map: dict[str, list[ChannelContribution]] = {}
    best_frames: dict[str, tuple[int, float]] = {}  # video_id -> (best_frame, max_score)

    for channel_key, ranked_tuples in channel_video_rankings.items():
        # channel_key is format: "{channel}:{variant_name}"
        parts = channel_key.split(":", 1)
        channel_name = parts[0]
        variant_name = parts[1] if len(parts) > 1 else "literal"
        weight = float(weights.get(channel_key, weights.get(channel_name, 1.0)))

        for rank_1b, (vid, best_frame, raw_score) in enumerate(ranked_tuples, start=1):
            rrf_increment = weight / (rrf_k + rank_1b)
            rrf_scores[vid] = rrf_scores.get(vid, 0.0) + rrf_increment

            if vid not in contributions_map:
                contributions_map[vid] = []
            contributions_map[vid].append(
                ChannelContribution(
                    channel=channel_name,
                    variant_name=variant_name,
                    video_rank=rank_1b,
                    best_frame=best_frame,
                    raw_score=raw_score,
                )
            )

            if vid not in best_frames or raw_score > best_frames[vid][1]:
                best_frames[vid] = (best_frame, raw_score)

    # Sort videos deterministically by rrf_score DESC, then video_id ASC
    sorted_vids = sorted(
        rrf_scores.keys(),
        key=lambda v: (-rrf_scores[v], v),
    )

    ranked_candidates: list[VideoCandidateWithProvenance] = []
    novel_rescue_candidates: list[VideoCandidateWithProvenance] = []
    baseline_set = set(baseline_top_video_ids)

    for vid in sorted_vids:
        cand = VideoCandidateWithProvenance(
            video_id=vid,
            rrf_score=rrf_scores[vid],
            contributions=tuple(contributions_map.get(vid, ())),
            best_anchor_frame=best_frames.get(vid, (0, 0.0))[0],
        )
        ranked_candidates.append(cand)
        if vid not in baseline_set:
            novel_rescue_candidates.append(cand)

    return MultiVariantRetrievalResult(
        query_id=query_id,
        variants=variants,
        ranked_videos=tuple(ranked_candidates),
        novel_rescue_videos=tuple(novel_rescue_candidates),
    )
