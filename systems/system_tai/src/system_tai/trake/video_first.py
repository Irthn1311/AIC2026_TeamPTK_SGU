"""Opt-in TRAKE video-first nomination and restricted event-pool construction."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from system_tai.common.schemas import CandidateFrame, KISResult
from system_tai.retrieval.multi_query import QueryVariant, WeightedRRFRetriever
from system_tai.retrieval.video_evidence import (
    FullCorpusVideoMaximaOutcome,
    VideoRestrictedSearchOutcome,
)
from system_tai.trake.models import TRAKEEventCandidate

TRAKE_VIDEO_FIRST_RESTRICTED_EVENT_SEARCH = (
    "TRAKE_VIDEO_FIRST_RESTRICTED_EVENT_SEARCH"
)


@dataclass(frozen=True, slots=True)
class TRAKEVideoFirstConfig:
    enabled: bool = False
    selected_video_cap: int = 32
    event_video_nomination_depth: int = 100
    anchors_per_event_video: int = 5

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise ValueError("enabled must be a boolean")
        if type(self.selected_video_cap) is not int or not (
            1 <= self.selected_video_cap <= 1000
        ):
            raise ValueError("selected_video_cap must be an integer in [1, 1000]")
        if type(self.event_video_nomination_depth) is not int or not (
            1 <= self.event_video_nomination_depth <= 100000
        ):
            raise ValueError(
                "event_video_nomination_depth must be an integer in [1, 100000]"
            )
        if type(self.anchors_per_event_video) is not int or not (
            1 <= self.anchors_per_event_video <= 100
        ):
            raise ValueError("anchors_per_event_video must be an integer in [1, 100]")


@dataclass(frozen=True, slots=True)
class VariantVideoRank:
    variant_id: str
    weight: float
    video_rank: int


@dataclass(frozen=True, slots=True)
class EventVideoEvidence:
    event_index: int
    video_id: str
    event_video_rank: int
    event_video_rrf_score: float
    best_variant_rank: int
    per_variant: tuple[VariantVideoRank, ...]


@dataclass(frozen=True, slots=True)
class NominatedVideo:
    video_id: str
    coverage_count: int
    worst_event_rank: int
    reciprocal_event_rank_sum: float
    best_event_rank: int
    event_video_ranks: tuple[int, ...]


def build_event_video_rankings(
    *,
    event_variants: Mapping[int, Sequence[QueryVariant]],
    maxima: FullCorpusVideoMaximaOutcome,
    rrf_constant: float,
) -> dict[int, tuple[EventVideoEvidence, ...]]:
    """Fuse full-corpus variant video ranks into one rank per event."""

    if not math.isfinite(rrf_constant) or rrf_constant <= 0:
        raise ValueError("rrf_constant must be finite and positive")
    results: dict[int, tuple[EventVideoEvidence, ...]] = {}
    for event_index in sorted(event_variants):
        variants = tuple(event_variants[event_index])
        if not variants:
            raise ValueError(f"event {event_index} has no query variants")
        expected_ids = {variant.variant_id for variant in variants}
        if not expected_ids.issubset(maxima.rankings):
            missing = sorted(expected_ids - set(maxima.rankings))
            raise ValueError(f"missing video-maxima rankings: {missing}")
        ranks_by_variant = {
            variant.variant_id: {
                hit.video_id: hit.rank
                for hit in maxima.rankings[variant.variant_id]
            }
            for variant in variants
        }
        video_sets = [set(by_video) for by_video in ranks_by_variant.values()]
        if not video_sets or any(video_set != video_sets[0] for video_set in video_sets[1:]):
            raise ValueError("variant video rankings must cover the same corpus videos")

        unranked: list[EventVideoEvidence] = []
        for video_id in sorted(video_sets[0]):
            provenance = tuple(
                VariantVideoRank(
                    variant_id=variant.variant_id,
                    weight=float(variant.weight),
                    video_rank=ranks_by_variant[variant.variant_id][video_id],
                )
                for variant in sorted(variants, key=lambda item: item.variant_id)
            )
            score = sum(
                item.weight / (rrf_constant + item.video_rank)
                for item in provenance
            )
            unranked.append(
                EventVideoEvidence(
                    event_index=event_index,
                    video_id=video_id,
                    event_video_rank=0,
                    event_video_rrf_score=float(score),
                    best_variant_rank=min(item.video_rank for item in provenance),
                    per_variant=provenance,
                )
            )
        ordered = sorted(
            unranked,
            key=lambda item: (
                -item.event_video_rrf_score,
                item.best_variant_rank,
                item.video_id,
            ),
        )
        results[event_index] = tuple(
            EventVideoEvidence(
                event_index=item.event_index,
                video_id=item.video_id,
                event_video_rank=rank,
                event_video_rrf_score=item.event_video_rrf_score,
                best_variant_rank=item.best_variant_rank,
                per_variant=item.per_variant,
            )
            for rank, item in enumerate(ordered, start=1)
        )
    return results


def nominate_videos(
    *,
    event_video_rankings: Mapping[int, Sequence[EventVideoEvidence]],
    config: TRAKEVideoFirstConfig,
    rrf_constant: float,
) -> tuple[NominatedVideo, ...]:
    if not event_video_rankings:
        raise ValueError("event_video_rankings must not be empty")
    if not math.isfinite(rrf_constant) or rrf_constant <= 0:
        raise ValueError("rrf_constant must be finite and positive")
    rank_maps = {
        event_index: {item.video_id: item.event_video_rank for item in ranking}
        for event_index, ranking in sorted(event_video_rankings.items())
    }
    video_sets = [set(rank_map) for rank_map in rank_maps.values()]
    if any(video_set != video_sets[0] for video_set in video_sets[1:]):
        raise ValueError("event video rankings must cover the same corpus videos")

    candidates: list[NominatedVideo] = []
    for video_id in sorted(video_sets[0]):
        ranks = tuple(
            rank_maps[event_index][video_id] for event_index in sorted(rank_maps)
        )
        candidates.append(
            NominatedVideo(
                video_id=video_id,
                coverage_count=sum(
                    rank <= config.event_video_nomination_depth for rank in ranks
                ),
                worst_event_rank=max(ranks),
                reciprocal_event_rank_sum=float(
                    sum(1.0 / (rrf_constant + rank) for rank in ranks)
                ),
                best_event_rank=min(ranks),
                event_video_ranks=ranks,
            )
        )
    ordered = sorted(
        candidates,
        key=lambda item: (
            -item.coverage_count,
            item.worst_event_rank,
            -item.reciprocal_event_rank_sum,
            item.best_event_rank,
            item.video_id,
        ),
    )
    return tuple(ordered[: config.selected_video_cap])


def build_restricted_event_pools(
    *,
    query_id: str,
    event_variants: Mapping[int, Sequence[QueryVariant]],
    event_video_rankings: Mapping[int, Sequence[EventVideoEvidence]],
    nominated_videos: Sequence[NominatedVideo],
    restricted: VideoRestrictedSearchOutcome,
    weighted_rrf: WeightedRRFRetriever,
    anchors_per_event_video: int,
    rrf_constant: float,
) -> tuple[tuple[TRAKEEventCandidate, ...], ...]:
    """Fuse per-variant frame ranks inside nominated videos without temporal NMS."""

    event_rank_maps = {
        event_index: {item.video_id: item.event_video_rank for item in ranking}
        for event_index, ranking in event_video_rankings.items()
    }
    pools: list[tuple[TRAKEEventCandidate, ...]] = []
    for event_index in sorted(event_variants):
        variants = tuple(event_variants[event_index])
        staged: list[tuple[int, int, CandidateFrame]] = []
        for nominated in nominated_videos:
            rankings: dict[str, KISResult] = {}
            for variant in variants:
                hits = restricted.rankings[variant.variant_id][nominated.video_id]
                rankings[variant.variant_id] = KISResult(
                    query_id=variant.variant_id,
                    ranked_candidates=tuple(
                        CandidateFrame(
                            video_id=hit.video_id,
                            frame_id=hit.frame_id,
                            clip_row=hit.clip_row,
                            keyframe_order=hit.keyframe_order,
                            score=hit.cosine_score,
                            rank=hit.rank,
                            source="video_restricted_exact",
                            diagnostic_metadata={"pts_time": hit.pts_time},
                        )
                        for hit in hits
                    ),
                )
            fused = weighted_rrf.fuse_rankings(
                query_id=f"{query_id}::e{event_index}::{nominated.video_id}",
                variants=variants,
                rankings=rankings,
                output_top_k=anchors_per_event_video,
                rrf_constant=rrf_constant,
            )
            event_video_rank = event_rank_maps[event_index][nominated.video_id]
            for candidate in fused.ranked_candidates:
                staged.append((candidate.rank, event_video_rank, candidate))

        staged.sort(
            key=lambda item: (
                item[0],
                item[1],
                item[2].video_id,
                item[2].frame_id,
                item[2].clip_row,
            )
        )
        pool = tuple(
            TRAKEEventCandidate(
                query_id=query_id,
                event_index=event_index,
                rank=restricted_event_rank,
                video_id=candidate.video_id,
                frame_id=candidate.frame_id,
                retrieval_score=float(candidate.score),
                provenance={
                    "source": TRAKE_VIDEO_FIRST_RESTRICTED_EVENT_SEARCH,
                    "event_video_rank": event_video_rank,
                    "restricted_event_video_rank": local_rank,
                    "restricted_event_rank": restricted_event_rank,
                    "fusion_score": float(candidate.score),
                    "clip_row": candidate.clip_row,
                    "keyframe_order": candidate.keyframe_order,
                    "per_variant": (candidate.diagnostic_metadata or {}).get(
                        "per_variant", []
                    ),
                },
            )
            for restricted_event_rank, (local_rank, event_video_rank, candidate) in enumerate(
                staged,
                start=1,
            )
        )
        pools.append(pool)
    return tuple(pools)
