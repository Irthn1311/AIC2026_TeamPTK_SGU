"""Opt-in KIS video-level RRF nomination and restricted exact frame search."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from system_tai.common.schemas import CandidateFrame, KISResult
from system_tai.retrieval.multi_query import QueryVariant, WeightedRRFRetriever
from system_tai.retrieval.video_evidence import (
    FullCorpusVideoMaximaOutcome,
    VideoRestrictedSearchOutcome,
)

KIS_SEMANTIC_VIDEO_FIRST = "KIS_SEMANTIC_VIDEO_FIRST"


@dataclass(frozen=True, slots=True)
class KISVideoFirstConfig:
    enabled: bool = False
    v2_adaptive_enabled: bool = False
    selected_video_cap: int = 32
    video_nomination_depth: int = 100
    restricted_frames_per_video_per_variant: int = 10
    full_query_weight: float = 1.0
    primary_scene_weight: float = 1.0
    supporting_attribute_weight: float = 0.35
    top_m_evidence_cap: int = 3
    top_m_min_frame_gap: int = 60
    top_m_weights: tuple[float, ...] = (0.6, 0.3, 0.1)
    adaptive_budget_base: int = 32
    adaptive_budget_medium: int = 48
    adaptive_budget_high: int = 64
    coverage_threshold: float = 0.75

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise ValueError("enabled must be a boolean")
        if type(self.v2_adaptive_enabled) is not bool:
            raise ValueError("v2_adaptive_enabled must be a boolean")
        if type(self.selected_video_cap) is not int or not (
            1 <= self.selected_video_cap <= 1000
        ):
            raise ValueError("selected_video_cap must be an integer in [1, 1000]")
        if type(self.video_nomination_depth) is not int or self.video_nomination_depth <= 0:
            raise ValueError("video_nomination_depth must be a positive integer")
        if (
            type(self.restricted_frames_per_video_per_variant) is not int
            or self.restricted_frames_per_video_per_variant <= 0
        ):
            raise ValueError(
                "restricted_frames_per_video_per_variant must be a positive integer"
            )
        for name, value in (
            ("full_query_weight", self.full_query_weight),
            ("primary_scene_weight", self.primary_scene_weight),
            ("supporting_attribute_weight", self.supporting_attribute_weight),
            ("coverage_threshold", self.coverage_threshold),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if type(self.top_m_evidence_cap) is not int or self.top_m_evidence_cap <= 0:
            raise ValueError("top_m_evidence_cap must be a positive integer")
        if type(self.top_m_min_frame_gap) is not int or self.top_m_min_frame_gap < 0:
            raise ValueError("top_m_min_frame_gap must be a non-negative integer")
        if not self.top_m_weights or any(
            not math.isfinite(w) or w <= 0 for w in self.top_m_weights
        ):
            raise ValueError("top_m_weights must contain positive finite floats")


@dataclass(frozen=True, slots=True)
class ClauseCoverageMetadata:
    must_hit: int
    must_total: int
    strong_hit: int
    strong_total: int
    coverage_ratio: float
    per_clause_hit: dict[str, bool]

    def to_dict(self) -> dict[str, object]:
        return {
            "must_hit": self.must_hit,
            "must_total": self.must_total,
            "strong_hit": self.strong_hit,
            "strong_total": self.strong_total,
            "coverage_ratio": self.coverage_ratio,
            "per_clause_hit": self.per_clause_hit,
        }


@dataclass(frozen=True, slots=True)
class AdaptiveBudgetDiagnostic:
    chosen_k: int
    complexity_k: int
    uncertainty_k: int
    normalized_entropy: float
    top1_top5_margin: float
    top1_top16_margin: float
    is_flat: bool
    adaptive_reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "chosen_k": self.chosen_k,
            "complexity_k": self.complexity_k,
            "uncertainty_k": self.uncertainty_k,
            "normalized_entropy": self.normalized_entropy,
            "top1_top5_margin": self.top1_top5_margin,
            "top1_top16_margin": self.top1_top16_margin,
            "is_flat": self.is_flat,
            "adaptive_reasons": list(self.adaptive_reasons),
        }


@dataclass(frozen=True, slots=True)
class VariantVideoEvidence:
    variant_id: str
    weight: float
    video_rank: int
    maximum_frame_id: int
    maximum_clip_row: int
    maximum_cosine_score: float
    top_m_score: float = 0.0
    normalized_clause_score: float = 0.0


@dataclass(frozen=True, slots=True)
class FusedVideoEvidence:
    video_id: str
    rank: int
    fusion_score: float
    variant_hit_count: int
    primary_coverage_count: int
    best_individual_rank: int
    per_variant: tuple[VariantVideoEvidence, ...]
    coverage_metadata: ClauseCoverageMetadata | None = None


@dataclass(frozen=True, slots=True)
class KISVideoFirstOutcome:
    result: KISResult
    selected_videos: tuple[FusedVideoEvidence, ...]
    full_corpus_rows_scored: int
    full_corpus_store_scan_count: int
    restricted_rows_scored: int
    restricted_store_scan_count: int
    adaptive_diagnostic: AdaptiveBudgetDiagnostic | None = None

    def to_trace(self) -> dict[str, object]:
        trace: dict[str, object] = {
            "policy": KIS_SEMANTIC_VIDEO_FIRST,
            "enabled": True,
            "full_corpus_rows_scored": self.full_corpus_rows_scored,
            "full_corpus_store_scan_count": self.full_corpus_store_scan_count,
            "restricted_rows_scored": self.restricted_rows_scored,
            "restricted_store_scan_count": self.restricted_store_scan_count,
            "selected_video_count": len(self.selected_videos),
            "selected_videos": [
                {
                    "rank": item.rank,
                    "video_id": item.video_id,
                    "fusion_score": item.fusion_score,
                    "variant_hit_count": item.variant_hit_count,
                    "primary_coverage_count": item.primary_coverage_count,
                    "best_individual_rank": item.best_individual_rank,
                    "coverage": (
                        item.coverage_metadata.to_dict()
                        if item.coverage_metadata
                        else None
                    ),
                    "per_variant": [
                        {
                            "variant_id": hit.variant_id,
                            "weight": hit.weight,
                            "video_rank": hit.video_rank,
                            "maximum_frame_id": hit.maximum_frame_id,
                            "maximum_clip_row_diagnostic": hit.maximum_clip_row,
                            "maximum_cosine_score_diagnostic": hit.maximum_cosine_score,
                            "top_m_score": hit.top_m_score,
                            "normalized_clause_score": hit.normalized_clause_score,
                        }
                        for hit in item.per_variant
                    ],
                }
                for item in self.selected_videos
            ],
        }
        if self.adaptive_diagnostic is not None:
            trace["adaptive_budget"] = self.adaptive_diagnostic.to_dict()
        return trace


def fuse_video_maxima(
    *,
    variants: Sequence[QueryVariant],
    maxima: FullCorpusVideoMaximaOutcome,
    primary_variant_ids: frozenset[str],
    rrf_constant: float,
    nomination_depth: int,
    selected_video_cap: int,
) -> tuple[FusedVideoEvidence, ...]:
    """Fuse one exact maximum per video and variant at `video_id` identity (Legacy)."""

    variants = tuple(variants)
    if not variants:
        raise ValueError("variants must not be empty")
    if len({variant.variant_id for variant in variants}) != len(variants):
        raise ValueError("variant_id values must be unique")
    if not math.isfinite(rrf_constant) or rrf_constant <= 0:
        raise ValueError("rrf_constant must be finite and positive")
    if nomination_depth <= 0 or selected_video_cap <= 0:
        raise ValueError("nomination_depth and selected_video_cap must be positive")
    expected_ids = {variant.variant_id for variant in variants}
    if set(maxima.rankings) != expected_ids:
        raise ValueError("maxima must contain exactly one video ranking per variant")

    by_variant_video = {
        variant.variant_id: {
            hit.video_id: hit for hit in maxima.rankings[variant.variant_id]
        }
        for variant in variants
    }
    video_sets = [set(items) for items in by_variant_video.values()]
    if not video_sets or any(items != video_sets[0] for items in video_sets[1:]):
        raise ValueError("variant video rankings must cover the same corpus videos")

    staged: list[FusedVideoEvidence] = []
    for video_id in sorted(video_sets[0]):
        provenance = tuple(
            VariantVideoEvidence(
                variant_id=variant.variant_id,
                weight=float(variant.weight),
                video_rank=by_variant_video[variant.variant_id][video_id].rank,
                maximum_frame_id=(
                    by_variant_video[variant.variant_id][video_id].frame_id
                ),
                maximum_clip_row=(
                    by_variant_video[variant.variant_id][video_id].clip_row
                ),
                maximum_cosine_score=(
                    by_variant_video[variant.variant_id][video_id].cosine_score
                ),
                top_m_score=(
                    by_variant_video[variant.variant_id][video_id].top_m_score
                ),
            )
            for variant in sorted(variants, key=lambda item: item.variant_id)
        )
        score = sum(
            hit.weight / (rrf_constant + hit.video_rank) for hit in provenance
        )
        staged.append(
            FusedVideoEvidence(
                video_id=video_id,
                rank=0,
                fusion_score=float(score),
                variant_hit_count=sum(
                    hit.video_rank <= nomination_depth for hit in provenance
                ),
                primary_coverage_count=sum(
                    hit.variant_id in primary_variant_ids
                    and hit.video_rank <= nomination_depth
                    for hit in provenance
                ),
                best_individual_rank=min(hit.video_rank for hit in provenance),
                per_variant=provenance,
            )
        )
    ordered = sorted(
        staged,
        key=lambda item: (
            -item.fusion_score,
            -item.primary_coverage_count,
            -item.variant_hit_count,
            item.best_individual_rank,
            item.video_id,
        ),
    )[:selected_video_cap]
    return tuple(
        FusedVideoEvidence(
            video_id=item.video_id,
            rank=rank,
            fusion_score=item.fusion_score,
            variant_hit_count=item.variant_hit_count,
            primary_coverage_count=item.primary_coverage_count,
            best_individual_rank=item.best_individual_rank,
            per_variant=item.per_variant,
        )
        for rank, item in enumerate(ordered, start=1)
    )


def normalize_clause_scores(
    raw_scores: Mapping[str, float],
) -> dict[str, float]:
    """Clause-local percentile/rank normalization mapping scores to (0, 1]."""
    if not raw_scores:
        return {}
    n = len(raw_scores)
    sorted_items = sorted(raw_scores.items(), key=lambda x: (-x[1], x[0]))
    return {
        vid: float((n - rank + 1) / n)
        for rank, (vid, _) in enumerate(sorted_items, start=1)
    }


def compute_adaptive_video_budget_v2(
    fused_scores: Sequence[float],
    clause_count: int,
    has_attributes: bool,
    base_k: int = 32,
    medium_k: int = 48,
    high_k: int = 64,
) -> tuple[int, AdaptiveBudgetDiagnostic]:
    """Adaptive video budget strictly clamped to {32, 48, 64} (Expand-only in V2-A)."""
    scores = sorted(fused_scores, reverse=True)[:32]
    reasons: list[str] = []
    if len(scores) < 32:
        return base_k, AdaptiveBudgetDiagnostic(
            chosen_k=base_k,
            complexity_k=base_k,
            uncertainty_k=base_k,
            normalized_entropy=0.0,
            top1_top5_margin=0.0,
            top1_top16_margin=0.0,
            is_flat=False,
            adaptive_reasons=("insufficient_candidates_for_entropy",),
        )

    # 1. Normalized entropy calculation
    max_s = scores[0]
    exp_s = [math.exp((s - max_s) / 0.1) for s in scores]
    sum_exp = sum(exp_s)
    probs = [e / sum_exp for e in exp_s]
    entropy = -sum(p * math.log(p + 1e-12) for p in probs)
    h_norm = float(entropy / math.log(32))

    # 2. Margins
    delta_1_5 = float(scores[0] - scores[4])
    delta_1_16 = float(scores[0] - scores[15])

    # Complexity rule (reachability check)
    k_complexity = base_k
    if clause_count >= 3 or has_attributes:
        k_complexity = max(k_complexity, medium_k)
        if clause_count >= 4:
            reasons.append(f"complexity: high_clause_count({clause_count}>=4)")
        elif has_attributes:
            reasons.append("complexity: has_supporting_attributes")
        else:
            reasons.append(f"complexity: multi_clause({clause_count}>=3)")

    # Uncertainty rule (Entropy + Margins)
    is_highly_flat = (h_norm >= 0.85) or (delta_1_5 <= 0.05)
    is_moderately_flat = (h_norm >= 0.70) or (delta_1_5 <= 0.10) or (delta_1_16 <= 0.20)

    if is_highly_flat:
        k_uncertainty = high_k
        reasons.append(f"uncertainty: highly_flat(entropy={h_norm:.2f}, delta1_5={delta_1_5:.3f})")
    elif is_moderately_flat:
        k_uncertainty = medium_k
        reasons.append(f"uncertainty: moderately_flat(entropy={h_norm:.2f}, delta1_5={delta_1_5:.3f})")
    else:
        k_uncertainty = base_k
        reasons.append(f"uncertainty: confident_peak(entropy={h_norm:.2f}, delta1_5={delta_1_5:.3f})")

    chosen_k = max(k_complexity, k_uncertainty)
    if chosen_k > high_k:
        chosen_k = high_k
    elif chosen_k < base_k:
        chosen_k = base_k

    diagnostic = AdaptiveBudgetDiagnostic(
        chosen_k=chosen_k,
        complexity_k=k_complexity,
        uncertainty_k=k_uncertainty,
        normalized_entropy=h_norm,
        top1_top5_margin=delta_1_5,
        top1_top16_margin=delta_1_16,
        is_flat=is_moderately_flat or is_highly_flat,
        adaptive_reasons=tuple(reasons),
    )
    return chosen_k, diagnostic


def fuse_video_maxima_v2(
    *,
    variants: Sequence[QueryVariant],
    maxima: FullCorpusVideoMaximaOutcome,
    primary_variant_ids: frozenset[str],
    rrf_constant: float,
    nomination_depth: int,
    config: KISVideoFirstConfig,
) -> tuple[tuple[FusedVideoEvidence, ...], AdaptiveBudgetDiagnostic]:
    """V2-A Fusion: Diversity-Aware Top-M + Clause Normalization + Coverage + Adaptive Budget."""
    variants = tuple(variants)
    if not variants:
        raise ValueError("variants must not be empty")
    if len({variant.variant_id for variant in variants}) != len(variants):
        raise ValueError("variant_id values must be unique")
    if not math.isfinite(rrf_constant) or rrf_constant <= 0:
        raise ValueError("rrf_constant must be finite and positive")
    expected_ids = {variant.variant_id for variant in variants}
    if set(maxima.rankings) != expected_ids:
        raise ValueError("maxima must contain exactly one video ranking per variant")

    by_variant_video = {
        variant.variant_id: {
            hit.video_id: hit for hit in maxima.rankings[variant.variant_id]
        }
        for variant in variants
    }
    video_sets = [set(items) for items in by_variant_video.values()]
    if not video_sets or any(items != video_sets[0] for items in video_sets[1:]):
        raise ValueError("variant video rankings must cover the same corpus videos")

    all_videos = sorted(video_sets[0])

    # 1. Compute clause-local normalization for each variant
    normalized_clause_scores: dict[str, dict[str, float]] = {}
    for variant in variants:
        raw_map = {
            vid: by_variant_video[variant.variant_id][vid].top_m_score
            for vid in all_videos
        }
        normalized_clause_scores[variant.variant_id] = normalize_clause_scores(raw_map)

    # 2. Stage evidence and compute fused scores
    staged: list[FusedVideoEvidence] = []
    raw_evidence_scores_list: list[float] = []

    for video_id in all_videos:
        provenance = tuple(
            VariantVideoEvidence(
                variant_id=variant.variant_id,
                weight=float(variant.weight),
                video_rank=by_variant_video[variant.variant_id][video_id].rank,
                maximum_frame_id=(
                    by_variant_video[variant.variant_id][video_id].frame_id
                ),
                maximum_clip_row=(
                    by_variant_video[variant.variant_id][video_id].clip_row
                ),
                maximum_cosine_score=(
                    by_variant_video[variant.variant_id][video_id].cosine_score
                ),
                top_m_score=(
                    by_variant_video[variant.variant_id][video_id].top_m_score
                ),
                normalized_clause_score=(
                    normalized_clause_scores[variant.variant_id][video_id]
                ),
            )
            for variant in sorted(variants, key=lambda item: item.variant_id)
        )

        # Gap-preserving raw evidence score stream for entropy/margins
        raw_evidence_score = sum(
            hit.weight * hit.top_m_score for hit in provenance
        ) / sum(hit.weight for hit in provenance)
        raw_evidence_scores_list.append(raw_evidence_score)

        # Fused score combining RRF rank and normalized evidence
        rrf_part = sum(
            hit.weight / (rrf_constant + hit.video_rank) for hit in provenance
        )
        norm_part = sum(
            hit.weight * hit.normalized_clause_score for hit in provenance
        ) / sum(hit.weight for hit in provenance)

        score = 0.5 * rrf_part + 0.5 * norm_part

        # Coverage evaluation (diagnostic only)
        clause_hits = {
            hit.variant_id: hit.normalized_clause_score >= config.coverage_threshold
            for hit in provenance
        }
        must_hits = sum(
            hit.variant_id in primary_variant_ids and clause_hits[hit.variant_id]
            for hit in provenance
        )
        must_total = len(primary_variant_ids)
        strong_hits = sum(
            hit.variant_id not in primary_variant_ids and clause_hits[hit.variant_id]
            for hit in provenance
        )
        strong_total = len(provenance) - must_total
        coverage_ratio = (
            (must_hits + strong_hits) / len(provenance) if provenance else 0.0
        )

        coverage_meta = ClauseCoverageMetadata(
            must_hit=must_hits,
            must_total=must_total,
            strong_hit=strong_hits,
            strong_total=strong_total,
            coverage_ratio=coverage_ratio,
            per_clause_hit=clause_hits,
        )

        staged.append(
            FusedVideoEvidence(
                video_id=video_id,
                rank=0,
                fusion_score=float(score),
                variant_hit_count=sum(
                    hit.video_rank <= nomination_depth for hit in provenance
                ),
                primary_coverage_count=sum(
                    hit.variant_id in primary_variant_ids
                    and hit.video_rank <= nomination_depth
                    for hit in provenance
                ),
                best_individual_rank=min(hit.video_rank for hit in provenance),
                per_variant=provenance,
                coverage_metadata=coverage_meta,
            )
        )

    # 3. Compute adaptive video budget K in {32, 48, 64} on gap-preserving raw scores
    has_attributes = len(variants) > len(primary_variant_ids)
    chosen_k, adaptive_diag = compute_adaptive_video_budget_v2(
        fused_scores=raw_evidence_scores_list,
        clause_count=len(variants),
        has_attributes=has_attributes,
        base_k=config.adaptive_budget_base,
        medium_k=config.adaptive_budget_medium,
        high_k=config.adaptive_budget_high,
    )

    ordered = sorted(
        staged,
        key=lambda item: (
            -item.fusion_score,
            -item.primary_coverage_count,
            -item.variant_hit_count,
            item.best_individual_rank,
            item.video_id,
        ),
    )[:chosen_k]

    final_selected = tuple(
        FusedVideoEvidence(
            video_id=item.video_id,
            rank=rank,
            fusion_score=item.fusion_score,
            variant_hit_count=item.variant_hit_count,
            primary_coverage_count=item.primary_coverage_count,
            best_individual_rank=item.best_individual_rank,
            per_variant=item.per_variant,
            coverage_metadata=item.coverage_metadata,
        )
        for rank, item in enumerate(ordered, start=1)
    )
    return final_selected, adaptive_diag


def fuse_restricted_frames(
    *,
    query_id: str,
    variants: Sequence[QueryVariant],
    restricted: VideoRestrictedSearchOutcome,
    selected_videos: Sequence[FusedVideoEvidence],
    weighted_rrf: WeightedRRFRetriever,
    output_top_k: int,
    rrf_constant: float,
) -> KISResult:
    """Globally rank restricted frames per variant, then apply frame-identity RRF."""

    variants = tuple(variants)
    selected_videos = tuple(selected_videos)
    selected_ids = {item.video_id for item in selected_videos}
    if not selected_ids:
        raise ValueError("selected_videos must not be empty")
    if len(selected_ids) != len(selected_videos):
        raise ValueError("selected_videos must be unique")

    rankings: dict[str, KISResult] = {}
    for variant in variants:
        per_video = restricted.rankings.get(variant.variant_id)
        if per_video is None or set(per_video) != selected_ids:
            raise ValueError(
                f"restricted ranking coverage mismatch for {variant.variant_id}"
            )
        hits = [hit for video_id in sorted(selected_ids) for hit in per_video[video_id]]
        ordered = sorted(
            hits,
            key=lambda hit: (
                -hit.cosine_score,
                hit.video_id,
                hit.frame_id,
                hit.clip_row,
            ),
        )
        rankings[variant.variant_id] = KISResult(
            query_id=variant.variant_id,
            ranked_candidates=tuple(
                CandidateFrame(
                    video_id=hit.video_id,
                    frame_id=hit.frame_id,
                    clip_row=hit.clip_row,
                    keyframe_order=hit.keyframe_order,
                    score=float(hit.cosine_score),
                    rank=rank,
                    source="video_restricted_exact",
                    diagnostic_metadata={"pts_time": hit.pts_time},
                )
                for rank, hit in enumerate(ordered, start=1)
            ),
        )

    fused = weighted_rrf.fuse_rankings(
        query_id=query_id,
        variants=variants,
        rankings=rankings,
        output_top_k=output_top_k,
        rrf_constant=rrf_constant,
    )
    video_by_id = {item.video_id: item for item in selected_videos}
    enriched = tuple(
        CandidateFrame(
            video_id=item.video_id,
            frame_id=item.frame_id,
            clip_row=item.clip_row,
            keyframe_order=item.keyframe_order,
            score=item.score,
            rank=item.rank,
            source="kis_semantic_video_first",
            diagnostic_metadata={
                **dict(item.diagnostic_metadata or {}),
                "video_nomination_rank": video_by_id[item.video_id].rank,
                "video_fusion_score": video_by_id[item.video_id].fusion_score,
                "video_primary_coverage_count": (
                    video_by_id[item.video_id].primary_coverage_count
                ),
            },
        )
        for item in fused.ranked_candidates
    )
    return KISResult(query_id=query_id, ranked_candidates=enriched)


def build_kis_video_first_outcome(
    *,
    query_id: str,
    variants: Sequence[QueryVariant],
    maxima: FullCorpusVideoMaximaOutcome,
    restricted: VideoRestrictedSearchOutcome,
    selected_videos: Sequence[FusedVideoEvidence],
    weighted_rrf: WeightedRRFRetriever,
    output_top_k: int,
    rrf_constant: float,
    adaptive_diagnostic: AdaptiveBudgetDiagnostic | None = None,
) -> KISVideoFirstOutcome:
    result = fuse_restricted_frames(
        query_id=query_id,
        variants=variants,
        restricted=restricted,
        selected_videos=selected_videos,
        weighted_rrf=weighted_rrf,
        output_top_k=output_top_k,
        rrf_constant=rrf_constant,
    )
    return KISVideoFirstOutcome(
        result=result,
        selected_videos=tuple(selected_videos),
        full_corpus_rows_scored=maxima.physical_rows_scored,
        full_corpus_store_scan_count=maxima.video_store_scan_count,
        restricted_rows_scored=restricted.physical_rows_scored,
        restricted_store_scan_count=restricted.video_store_scan_count,
        adaptive_diagnostic=adaptive_diagnostic,
    )
