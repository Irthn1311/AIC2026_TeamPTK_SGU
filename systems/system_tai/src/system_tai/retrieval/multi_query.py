"""Opt-in deterministic multi-query fusion for KIS retrieval.

The canonical :class:`ExactNumpyRetriever` remains the only per-variant
retrieval implementation. This module combines its one-based ranks and never
uses raw cosine scores in the fusion formula.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

from system_tai.common.schemas import CandidateFrame, KISQuery, KISResult
from system_tai.retrieval.vector_search import ExactNumpyRetriever


class QueryLanguage(StrEnum):
    VIETNAMESE = "vi"
    ENGLISH = "en"


class QueryVariantType(StrEnum):
    VIETNAMESE_DIRECT = "vietnamese_direct"
    ENGLISH_TRANSLATION = "english_translation"
    ENGLISH_EXPANSION = "english_expansion"


@dataclass(frozen=True, slots=True)
class QueryVariant:
    variant_id: str
    text: str
    language: QueryLanguage
    variant_type: QueryVariantType
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.variant_id, str) or not self.variant_id.strip():
            raise ValueError("variant_id must not be empty")
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("variant text must not be empty")
        if not isinstance(self.language, QueryLanguage):
            raise ValueError("language must be a supported QueryLanguage")
        if not isinstance(self.variant_type, QueryVariantType):
            raise ValueError("variant_type must be a supported QueryVariantType")
        if (
            isinstance(self.weight, bool)
            or not isinstance(self.weight, (int, float))
            or not math.isfinite(self.weight)
            or self.weight <= 0
        ):
            raise ValueError("variant weight must be finite and positive")


@dataclass(frozen=True, slots=True)
class _VariantHit:
    variant: QueryVariant
    candidate: CandidateFrame


@dataclass(frozen=True, slots=True)
class _FusedCandidate:
    representative: CandidateFrame
    fusion_score: float
    hits: tuple[_VariantHit, ...]

    @property
    def variant_hit_count(self) -> int:
        return len(self.hits)

    @property
    def best_individual_rank(self) -> int:
        return min(hit.candidate.rank for hit in self.hits)


def _fused_sort_key(
    candidate: _FusedCandidate,
) -> tuple[float, int, int, str, int, int]:
    representative = candidate.representative
    return (
        -candidate.fusion_score,
        -candidate.variant_hit_count,
        candidate.best_individual_rank,
        representative.video_id,
        representative.frame_id,
        representative.clip_row,
    )


class WeightedRRFRetriever:
    """Fuse canonical exact rankings with weighted reciprocal-rank fusion."""

    def __init__(self, exact_retriever: ExactNumpyRetriever) -> None:
        self.exact_retriever = exact_retriever

    def retrieve(
        self,
        *,
        query_id: str,
        variants: tuple[QueryVariant, ...],
        top_k_per_variant: int = 100,
        output_top_k: int = 100,
        rrf_constant: float = 60.0,
    ) -> KISResult:
        if not query_id.strip():
            raise ValueError("query_id must not be empty")
        if not variants:
            raise ValueError("at least one query variant is required")
        variant_ids = [variant.variant_id for variant in variants]
        if len(set(variant_ids)) != len(variant_ids):
            raise ValueError("variant_id values must be unique")
        if top_k_per_variant <= 0:
            raise ValueError("top_k_per_variant must be positive")
        if output_top_k <= 0 or output_top_k > 100:
            raise ValueError("output_top_k must be between 1 and 100")
        if not math.isfinite(rrf_constant) or rrf_constant <= 0:
            raise ValueError("rrf_constant must be finite and positive")

        rankings: dict[str, KISResult] = {}
        for variant in variants:
            rankings[variant.variant_id] = self.exact_retriever.retrieve(
                KISQuery(
                    query_id=f"{query_id}::{variant.variant_id}",
                    text=variant.text,
                    top_k=top_k_per_variant,
                )
            )
        return self.fuse_rankings(
            query_id=query_id,
            variants=variants,
            rankings=rankings,
            output_top_k=output_top_k,
            rrf_constant=rrf_constant,
        )

    def fuse_rankings(
        self,
        *,
        query_id: str,
        variants: tuple[QueryVariant, ...],
        rankings: Mapping[str, KISResult],
        output_top_k: int = 100,
        rrf_constant: float = 60.0,
    ) -> KISResult:
        if not query_id.strip():
            raise ValueError("query_id must not be empty")
        if not variants:
            raise ValueError("at least one query variant is required")
        variant_ids = [variant.variant_id for variant in variants]
        if len(set(variant_ids)) != len(variant_ids):
            raise ValueError("variant_id values must be unique")
        if set(rankings) != set(variant_ids):
            raise ValueError("rankings must contain exactly one result per variant_id")
        if output_top_k <= 0 or output_top_k > 100:
            raise ValueError("output_top_k must be between 1 and 100")
        if not math.isfinite(rrf_constant) or rrf_constant <= 0:
            raise ValueError("rrf_constant must be finite and positive")

        by_identity: dict[tuple[str, int], list[_VariantHit]] = {}
        for variant in variants:
            result = rankings[variant.variant_id]
            best_for_variant: dict[tuple[str, int], CandidateFrame] = {}
            for candidate in result.ranked_candidates:
                identity = (candidate.video_id, candidate.frame_id)
                existing = best_for_variant.get(identity)
                if existing is None or (
                    candidate.rank,
                    candidate.clip_row,
                ) < (existing.rank, existing.clip_row):
                    best_for_variant[identity] = candidate
            for identity, candidate in best_for_variant.items():
                by_identity.setdefault(identity, []).append(
                    _VariantHit(variant=variant, candidate=candidate)
                )

        fused: list[_FusedCandidate] = []
        for hits in by_identity.values():
            ordered_hits = tuple(sorted(hits, key=lambda hit: hit.variant.variant_id))
            representative = min(
                (hit.candidate for hit in ordered_hits),
                key=lambda candidate: (
                    candidate.rank,
                    candidate.clip_row,
                    candidate.keyframe_order,
                ),
            )
            score = sum(
                hit.variant.weight / (rrf_constant + hit.candidate.rank)
                for hit in ordered_hits
            )
            fused.append(
                _FusedCandidate(
                    representative=representative,
                    fusion_score=float(score),
                    hits=ordered_hits,
                )
            )

        ranked_fused = sorted(fused, key=_fused_sort_key)[:output_top_k]
        candidates = tuple(
            CandidateFrame(
                video_id=item.representative.video_id,
                frame_id=item.representative.frame_id,
                clip_row=item.representative.clip_row,
                keyframe_order=item.representative.keyframe_order,
                score=item.fusion_score,
                rank=rank,
                source="weighted_rrf",
                diagnostic_metadata={
                    **dict(item.representative.diagnostic_metadata or {}),
                    "fusion_method": "weighted_reciprocal_rank_fusion",
                    "fusion_score": item.fusion_score,
                    "variant_hit_count": item.variant_hit_count,
                    "best_individual_rank": item.best_individual_rank,
                    "rrf_constant": rrf_constant,
                    "per_variant": [
                        {
                            "variant_id": hit.variant.variant_id,
                            "language": hit.variant.language.value,
                            "variant_type": hit.variant.variant_type.value,
                            "weight": hit.variant.weight,
                            "rank": hit.candidate.rank,
                            "cosine_score": hit.candidate.score,
                        }
                        for hit in item.hits
                    ],
                },
            )
            for rank, item in enumerate(ranked_fused, start=1)
        )
        return KISResult(query_id=query_id, ranked_candidates=candidates)
