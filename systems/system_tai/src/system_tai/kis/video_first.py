"""Opt-in KIS video-level RRF nomination and restricted exact frame search."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

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
    temporal_chain_frame_bonus: float = 0.05
    restricted_frame_min_gap: int = 0
    max_restricted_candidates_per_video: int | None = None
    enable_candidate_union: bool = False
    enable_score_normalization: bool = False
    enable_late_interaction: bool = False
    enable_positive_chain_bonus: bool = False
    enable_temporal_diverse_local_candidates: bool = False
    temporal_diversity_gap_seconds: float = 5.0
    enable_vi_localization_variant: bool = False
    vi_localization_weight: float = 0.5
    internal_rrf_candidate_depth: int = 100
    collect_fusion_trace: bool = False
    enable_top_video_local_anchor: bool = False
    local_anchor_top_video_count: int = 3
    local_anchor_source: str = "SEMANTIC_LOCAL"

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
        if type(self.enable_top_video_local_anchor) is not bool:
            raise ValueError("enable_top_video_local_anchor must be a boolean")
        if type(self.local_anchor_top_video_count) is not int or self.local_anchor_top_video_count <= 0:
            raise ValueError("local_anchor_top_video_count must be a positive integer")
        if self.local_anchor_source not in ("SEMANTIC_LOCAL", "VI_LOCAL"):
            raise ValueError("local_anchor_source must be 'SEMANTIC_LOCAL' or 'VI_LOCAL'")
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
        if (
            not math.isfinite(self.temporal_chain_frame_bonus)
            or self.temporal_chain_frame_bonus < 0
        ):
            raise ValueError("temporal_chain_frame_bonus must be finite and non-negative")
        if type(self.restricted_frame_min_gap) is not int or self.restricted_frame_min_gap < 0:
            raise ValueError("restricted_frame_min_gap must be a non-negative integer")
        if (
            self.max_restricted_candidates_per_video is not None
            and (
                type(self.max_restricted_candidates_per_video) is not int
                or self.max_restricted_candidates_per_video <= 0
            )
        ):
            raise ValueError("max_restricted_candidates_per_video must be a positive integer")
        if type(self.enable_temporal_diverse_local_candidates) is not bool:
            raise ValueError("enable_temporal_diverse_local_candidates must be a boolean")
        if (
            not math.isfinite(self.temporal_diversity_gap_seconds)
            or self.temporal_diversity_gap_seconds < 0
        ):
            raise ValueError("temporal_diversity_gap_seconds must be finite and non-negative")
        if type(self.enable_vi_localization_variant) is not bool:
            raise ValueError("enable_vi_localization_variant must be a boolean")
        if (
            not math.isfinite(self.vi_localization_weight)
            or self.vi_localization_weight <= 0
        ):
            raise ValueError("vi_localization_weight must be finite and positive")
        if (
            type(self.internal_rrf_candidate_depth) is not int
            or not (100 <= self.internal_rrf_candidate_depth <= 2000)
        ):
            raise ValueError("internal_rrf_candidate_depth must be an integer in [100, 2000]")
        if type(self.collect_fusion_trace) is not bool:
            raise ValueError("collect_fusion_trace must be a boolean")


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
class TemporalChainDiagnostic:
    is_temporal_compound: bool
    temporal_scene_count: int
    has_valid_chain: bool
    selected_chain_frames: tuple[int, ...]
    chain_score: float
    soft_and_score: float
    balance_ratio: float
    temporal_multiplier: float

    def to_dict(self) -> dict[str, object]:
        return {
            "is_temporal_compound": self.is_temporal_compound,
            "temporal_scene_count": self.temporal_scene_count,
            "has_valid_chain": self.has_valid_chain,
            "selected_chain_frames": list(self.selected_chain_frames),
            "chain_score": self.chain_score,
            "soft_and_score": self.soft_and_score,
            "balance_ratio": self.balance_ratio,
            "temporal_multiplier": self.temporal_multiplier,
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
    top_m_peaks: tuple[tuple[int, float], ...] = ()


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
    temporal_chain: TemporalChainDiagnostic | None = None


@dataclass(frozen=True, slots=True)
class KISVideoFirstOutcome:
    result: KISResult
    selected_videos: tuple[FusedVideoEvidence, ...]
    full_corpus_rows_scored: int
    full_corpus_store_scan_count: int
    restricted_rows_scored: int
    restricted_store_scan_count: int
    adaptive_diagnostic: AdaptiveBudgetDiagnostic | None = None
    per_scene_top128: Mapping[str, tuple[str, ...]] = MappingProxyType({})
    per_scene_ranks: Mapping[str, Mapping[str, int]] = MappingProxyType({})
    candidate_selection_telemetry: Mapping[str, Mapping[str, Mapping[str, int]]] | None = None
    fusion_trace: Mapping[tuple[str, int], dict[str, object]] | None = None

    def to_trace(self) -> dict[str, object]:
        trace: dict[str, object] = {
            "policy": KIS_SEMANTIC_VIDEO_FIRST,
            "enabled": True,
            "full_corpus_rows_scored": self.full_corpus_rows_scored,
            "full_corpus_store_scan_count": self.full_corpus_store_scan_count,
            "restricted_rows_scored": self.restricted_rows_scored,
            "restricted_store_scan_count": self.restricted_store_scan_count,
            "selected_video_count": len(self.selected_videos),
            "per_scene_top128": {k: list(v) for k, v in self.per_scene_top128.items()},
            "per_scene_ranks": {k: dict(v) for k, v in self.per_scene_ranks.items()},
            "selected_videos": [
                {
                    "rank": item.rank,
                    "video_id": item.video_id,
                    "fusion_score": item.fusion_score,
                    "variant_hit_count": item.variant_hit_count,
                    "primary_coverage_count": item.primary_coverage_count,
                    "best_individual_rank": item.best_individual_rank,
                    "temporal_chain": (
                        item.temporal_chain.to_dict()
                        if item.temporal_chain
                        else None
                    ),
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
                            "top_m_peaks": list(hit.top_m_peaks),
                        }
                        for hit in item.per_variant
                    ],
                }
                for item in self.selected_videos
            ],
        }
        if self.adaptive_diagnostic is not None:
            trace["adaptive_budget"] = self.adaptive_diagnostic.to_dict()
        if self.candidate_selection_telemetry:
            trace["candidate_selection_telemetry"] = {
                qid: {vid: dict(tel) for vid, tel in per_vid.items()}
                for qid, per_vid in self.candidate_selection_telemetry.items()
            }
        if self.fusion_trace:
            summary = self.fusion_trace.get(("__summary__", 0), {})
            trace["fusion_trace_schema_version"] = summary.get("fusion_trace_schema_version", "2.1.0")
            trace["fusion_trace"] = {
                f"{vid}::{fid}": dict(info)
                for (vid, fid), info in self.fusion_trace.items()
                if vid != "__summary__"
            }
            if ("__summary__", 0) in self.fusion_trace:
                trace["allocation_summary"] = dict(self.fusion_trace[("__summary__", 0)])
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


def solve_temporal_chain(
    peaks_by_scene: Sequence[Sequence[tuple[int, float]]],
    scene_weights: Sequence[float],
    min_gap: int = 60,
) -> tuple[bool, tuple[int, ...], float]:
    """Find the optimal chronological frame chain across N ordered scenes using DP."""
    n_scenes = len(peaks_by_scene)
    if n_scenes == 0:
        return False, (), 0.0
    if any(len(peaks) == 0 for peaks in peaks_by_scene):
        return False, (), 0.0

    if n_scenes == 1:
        best_peak = max(peaks_by_scene[0], key=lambda p: p[1])
        return True, (best_peak[0],), float(best_peak[1])

    dp: list[list[float]] = []
    parent: list[list[int]] = []

    w0 = scene_weights[0]
    dp.append([w0 * p[1] for p in peaks_by_scene[0]])
    parent.append([-1 for _ in peaks_by_scene[0]])

    for i in range(1, n_scenes):
        w_i = scene_weights[i]
        curr_peaks = peaks_by_scene[i]
        prev_peaks = peaks_by_scene[i - 1]
        prev_dp = dp[i - 1]

        curr_dp: list[float] = []
        curr_parent: list[int] = []

        for j, (curr_frame, curr_score) in enumerate(curr_peaks):
            best_val = -float("inf")
            best_k = -1
            for k, (prev_frame, _) in enumerate(prev_peaks):
                if prev_frame + min_gap <= curr_frame and prev_dp[k] > -float("inf"):
                    cand = prev_dp[k] + w_i * curr_score
                    if cand > best_val:
                        best_val = cand
                        best_k = k
            curr_dp.append(best_val)
            curr_parent.append(best_k)

        dp.append(curr_dp)
        parent.append(curr_parent)

    last_dp = dp[-1]
    best_last_score = -float("inf")
    best_last_idx = -1
    for j, score in enumerate(last_dp):
        if score > best_last_score:
            best_last_score = score
            best_last_idx = j

    if best_last_idx == -1 or best_last_score <= -float("inf"):
        return False, (), 0.0

    chain_frames: list[int] = [0] * n_scenes
    curr_idx = best_last_idx
    for i in range(n_scenes - 1, -1, -1):
        chain_frames[i] = peaks_by_scene[i][curr_idx][0]
        curr_idx = parent[i][curr_idx]

    w_sum = sum(scene_weights)
    avg_chain_score = float(best_last_score / w_sum) if w_sum > 0 else 0.0
    return True, tuple(chain_frames), avg_chain_score


def compute_soft_and_joint_score(
    scores: Sequence[float],
    weights: Sequence[float],
    epsilon: float = 1e-4,
) -> float:
    """Soft-AND geometric mean score across multiple clauses with epsilon smoothing."""
    if not scores or len(scores) != len(weights):
        return 0.0
    w_sum = sum(weights)
    if w_sum <= 0:
        return 0.0
    log_sum = sum(w * math.log(max(0.0, s) + epsilon) for w, s in zip(weights, scores))
    return float(math.exp(log_sum / w_sum))


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
    """Adaptive video budget strictly clamped to {32, 48, 64} with robust gap-preserving standardization."""
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

    # 1. Robust Median / MAD standardization
    med = float(scores[len(scores) // 2])
    abs_devs = sorted(abs(s - med) for s in scores)
    mad = float(abs_devs[len(abs_devs) // 2])
    mad_scale = max(mad * 1.4826, 0.01)

    # Standardized score difference from maximum
    max_s = scores[0]
    z_scores = [(s - max_s) / mad_scale for s in scores]
    exp_z = [math.exp(max(z, -30.0)) for z in z_scores]
    sum_exp = sum(exp_z)
    probs = [e / sum_exp for e in exp_z]
    entropy = -sum(p * math.log(p + 1e-12) for p in probs)
    h_norm = float(entropy / math.log(32))

    # 2. Margins
    delta_1_5 = float(scores[0] - scores[4])
    delta_1_16 = float(scores[0] - scores[15])

    # Complexity rule (attribute/count sensitive queries)
    k_complexity = base_k
    if has_attributes:
        k_complexity = max(k_complexity, medium_k)
        reasons.append("complexity: has_supporting_attributes")
    elif clause_count >= 4:
        k_complexity = max(k_complexity, medium_k)
        reasons.append(f"complexity: high_clause_count({clause_count}>=4)")

    # Uncertainty rule (Strong Agreement between Entropy & Margins)
    is_strongly_uncertain = (h_norm >= 0.80 and delta_1_5 <= 0.05) or (delta_1_5 <= 0.03 and delta_1_16 <= 0.08)
    is_moderately_uncertain = (h_norm >= 0.65) or (delta_1_5 <= 0.08) or (delta_1_16 <= 0.15)

    if is_strongly_uncertain:
        k_uncertainty = high_k
        reasons.append(f"uncertainty: strong_agreement(entropy={h_norm:.2f}, delta1_5={delta_1_5:.3f})")
    elif is_moderately_uncertain:
        k_uncertainty = medium_k
        reasons.append(f"uncertainty: moderate(entropy={h_norm:.2f}, delta1_5={delta_1_5:.3f})")
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
        is_flat=is_moderately_uncertain or is_strongly_uncertain,
        adaptive_reasons=tuple(reasons),
    )
    return chosen_k, diagnostic


def fuse_video_maxima_v2(
    *,
    variants: Sequence[QueryVariant],
    maxima: FullCorpusVideoMaximaOutcome,
    primary_variant_ids: frozenset[str],
    supporting_variant_ids: frozenset[str] = frozenset(),
    temporal_variants: Sequence[QueryVariant] = (),
    rrf_constant: float,
    nomination_depth: int,
    config: KISVideoFirstConfig,
) -> tuple[tuple[FusedVideoEvidence, ...], AdaptiveBudgetDiagnostic]:
    """V2-A.2 Fusion: Soft-AND Multi-Clause Coverage + DP Temporal Chain Solver + Top-M Spacing."""
    variants = tuple(variants)
    temporal_variants = tuple(temporal_variants)
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
    is_temporal_compound = len(temporal_variants) >= 2

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
                top_m_peaks=by_variant_video[variant.variant_id][video_id].top_m_peaks,
            )
            for variant in sorted(variants, key=lambda item: item.variant_id)
        )

        rrf_part = sum(
            hit.weight / (rrf_constant + hit.video_rank) for hit in provenance
        )

        # Coverage evaluation
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

        temporal_diag: TemporalChainDiagnostic | None = None

        if is_temporal_compound:
            # Multi-clause Soft-AND + DP Temporal Chain Solver
            peaks_by_scene = [
                (
                    by_variant_video[t_var.variant_id][video_id].top_m_peaks
                    if by_variant_video[t_var.variant_id][video_id].top_m_peaks
                    else [
                        (
                            by_variant_video[t_var.variant_id][video_id].frame_id,
                            by_variant_video[t_var.variant_id][video_id].cosine_score,
                        )
                    ]
                )
                for t_var in temporal_variants
            ]
            scene_weights = [float(t_var.weight) for t_var in temporal_variants]
            scene_raw_scores = [
                float(by_variant_video[t_var.variant_id][video_id].top_m_score)
                for t_var in temporal_variants
            ]

            has_valid_chain, chain_frames, chain_score = solve_temporal_chain(
                peaks_by_scene=peaks_by_scene,
                scene_weights=scene_weights,
                min_gap=config.top_m_min_frame_gap,
            )

            soft_and = compute_soft_and_joint_score(scene_raw_scores, scene_weights)
            min_s = min(scene_raw_scores)
            max_s = max(scene_raw_scores)
            balance_ratio = float(min_s / (max_s + 1e-6))

            if has_valid_chain:
                temporal_multiplier = 1.35
                score = (
                    0.65 * soft_and * temporal_multiplier
                    + 0.25 * chain_score
                    + 0.10 * rrf_part
                )
            else:
                temporal_multiplier = 0.50
                score = (
                    0.65 * soft_and * temporal_multiplier
                    + 0.10 * rrf_part
                )

            temporal_diag = TemporalChainDiagnostic(
                is_temporal_compound=True,
                temporal_scene_count=len(temporal_variants),
                has_valid_chain=has_valid_chain,
                selected_chain_frames=chain_frames,
                chain_score=chain_score,
                soft_and_score=soft_and,
                balance_ratio=balance_ratio,
                temporal_multiplier=temporal_multiplier,
            )
            raw_evidence_scores_list.append(score)
        else:
            # Single-scene predicate + attribute fusion
            raw_evidence_score = sum(
                hit.weight * hit.top_m_score for hit in provenance
            ) / sum(hit.weight for hit in provenance)
            raw_evidence_scores_list.append(raw_evidence_score)

            norm_part = sum(
                hit.weight * hit.normalized_clause_score for hit in provenance
            ) / sum(hit.weight for hit in provenance)
            score = 0.5 * rrf_part + 0.5 * norm_part

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
                temporal_chain=temporal_diag,
            )
        )

    # 3. Determine Candidate Budget K
    if is_temporal_compound:
        chosen_k = 64
        # Compute diagnostics for logging
        scores_sorted = sorted(raw_evidence_scores_list, reverse=True)
        d1_5 = float(scores_sorted[0] - scores_sorted[min(4, len(scores_sorted)-1)]) if scores_sorted else 0.0
        d1_16 = float(scores_sorted[0] - scores_sorted[min(15, len(scores_sorted)-1)]) if len(scores_sorted) > 15 else 0.0
        adaptive_diag = AdaptiveBudgetDiagnostic(
            chosen_k=64,
            complexity_k=64,
            uncertainty_k=64,
            normalized_entropy=0.85,
            top1_top5_margin=d1_5,
            top1_top16_margin=d1_16,
            is_flat=True,
            adaptive_reasons=("temporal_compound_multi_clause: fixed_k_64",),
        )
    else:
        has_attributes = len(supporting_variant_ids) > 0
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
            temporal_chain=item.temporal_chain,
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
    temporal_chain_bonus: float = 0.0,
    min_frame_gap: int = 0,
    max_candidates_per_video: int | None = None,
    internal_rrf_candidate_depth: int = 100,
    collect_fusion_trace: bool = False,
    return_trace: bool = False,
    enable_top_video_local_anchor: bool = False,
    local_anchor_top_video_count: int = 3,
    local_anchor_source: str = "SEMANTIC_LOCAL",
) -> KISResult | tuple[KISResult, dict[tuple[str, int], dict[str, object]]]:
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
        ranked = tuple(
            CandidateFrame(
                video_id=hit.video_id,
                frame_id=hit.frame_id,
                clip_row=hit.clip_row,
                keyframe_order=hit.keyframe_order,
                score=float(hit.cosine_score),
                rank=rank,
                source="video_restricted_exact",
                diagnostic_metadata={
                    "pts_time": hit.pts_time,
                    "selection_source": hit.selection_source,
                    "raw_local_rank": hit.raw_local_rank,
                },
            )
            for rank, hit in enumerate(ordered, start=1)
        )
        rankings[variant.variant_id] = KISResult(
            query_id=variant.variant_id,
            ranked_candidates=ranked,
        )

    fused_top_k = max(output_top_k, internal_rrf_candidate_depth)
    full_fused = None
    if collect_fusion_trace:
        unique_restricted_keys = {
            (hit.video_id, hit.frame_id)
            for per_video in restricted.rankings.values()
            for hits in per_video.values()
            for hit in hits
        }
        trace_depth = max(len(unique_restricted_keys), fused_top_k)
        full_fused = weighted_rrf.fuse_rankings(
            query_id=query_id,
            variants=variants,
            rankings=rankings,
            output_top_k=trace_depth,
            rrf_constant=rrf_constant,
        )
        fused_candidates = full_fused.ranked_candidates[:fused_top_k]
    else:
        fused_result = weighted_rrf.fuse_rankings(
            query_id=query_id,
            variants=variants,
            rankings=rankings,
            output_top_k=fused_top_k,
            rrf_constant=rrf_constant,
        )
        fused_candidates = fused_result.ranked_candidates

    video_by_id = {item.video_id: item for item in selected_videos}

    # Extract winning DP chain frames by video
    chain_frames_by_video: dict[str, set[int]] = {}
    for item in selected_videos:
        if (
            item.temporal_chain
            and item.temporal_chain.has_valid_chain
            and item.temporal_chain.selected_chain_frames
        ):
            chain_frames_by_video[item.video_id] = set(
                item.temporal_chain.selected_chain_frames
            )

    # Extract selection info per frame across all variants
    selection_info_per_frame: dict[tuple[str, int], dict[str, dict[str, object]]] = {}
    for var_id, r in rankings.items():
        for cand in r.ranked_candidates:
            key = (cand.video_id, cand.frame_id)
            if cand.diagnostic_metadata:
                selection_info_per_frame.setdefault(key, {})[var_id] = {
                    "source": cand.diagnostic_metadata.get("selection_source", "RAW"),
                    "raw_local_rank": cand.diagnostic_metadata.get("raw_local_rank", 0),
                    "pts_time": cand.diagnostic_metadata.get("pts_time"),
                }

    # Enrich candidates with video score & variant selection info
    enriched_list: list[CandidateFrame] = []
    for item in fused_candidates:
        is_chain_winner = item.frame_id in chain_frames_by_video.get(item.video_id, ())
        vid_ev = video_by_id[item.video_id]
        vid_rank = vid_ev.rank
        vid_rrf_boost = 1.0 / (60.0 + vid_rank)
        final_score = item.score + 0.10 * vid_rrf_boost + (temporal_chain_bonus if is_chain_winner else 0.0)

        variant_scores: dict[str, float] = {}
        for var_id, r in rankings.items():
            for hit in r.ranked_candidates:
                if hit.video_id == item.video_id and hit.frame_id == item.frame_id:
                    variant_scores[var_id] = float(hit.score)
                    break

        key = (item.video_id, item.frame_id)
        selection_by_variant = selection_info_per_frame.get(key, {})
        diag: dict[str, object] = {
            **dict(item.diagnostic_metadata or {}),
            "video_nomination_rank": vid_ev.rank,
            "video_fusion_score": vid_ev.fusion_score,
            "video_primary_coverage_count": vid_ev.primary_coverage_count,
            "is_temporal_chain_winner": is_chain_winner,
            "scores_by_variant": variant_scores,
            "rrf_score": item.score,
        }
        if selection_by_variant:
            diag["selection_by_variant"] = selection_by_variant

        enriched_list.append(
            CandidateFrame(
                video_id=item.video_id,
                frame_id=item.frame_id,
                clip_row=item.clip_row,
                keyframe_order=item.keyframe_order,
                score=float(final_score),
                rank=item.rank,
                source="kis_semantic_video_first",
                diagnostic_metadata=diag,
            )
        )

    # Sort candidates by boosted score
    enriched_sorted = sorted(
        enriched_list,
        key=lambda c: (-c.score, c.video_id, c.frame_id),
    )

    if min_frame_gap > 0:
        filtered_candidates: list[CandidateFrame] = []
        selected_frames_by_video: dict[str, list[int]] = {}
        temporal_dedup_pruned_keys: set[tuple[str, int]] = set()

        for cand in enriched_sorted:
            vid = cand.video_id
            is_chain_winner = bool(cand.diagnostic_metadata.get("is_temporal_chain_winner", False)) if cand.diagnostic_metadata else False
            existing = selected_frames_by_video.get(vid, [])
            if is_chain_winner or not any(abs(cand.frame_id - f) < min_frame_gap for f in existing):
                filtered_candidates.append(cand)
                selected_frames_by_video.setdefault(vid, []).append(cand.frame_id)
            else:
                temporal_dedup_pruned_keys.add((cand.video_id, cand.frame_id))

        primary_candidates = filtered_candidates
        secondary_candidates = []
    else:
        # Segment-Level Decoupled Frame Localization (Distinct Temporal Action Clusters)
        primary_candidates = []
        secondary_candidates = []
        selected_cluster_frames: dict[str, list[int]] = {}
        segment_gap = 75  # 3.0 seconds at 25 fps
        temporal_dedup_pruned_keys = set()

        for cand in enriched_sorted:
            vid = cand.video_id
            is_chain_winner = bool(cand.diagnostic_metadata.get("is_temporal_chain_winner", False)) if cand.diagnostic_metadata else False
            existing_frames = selected_cluster_frames.get(vid, [])
            is_new_segment = not any(abs(cand.frame_id - f) < segment_gap for f in existing_frames)

            if is_chain_winner or is_new_segment:
                primary_candidates.append(cand)
                selected_cluster_frames.setdefault(vid, []).append(cand.frame_id)
            else:
                secondary_candidates.append(cand)

    # 1. Standard pre-anchor allocation
    standard_final_candidates = (primary_candidates + secondary_candidates)[:output_top_k]
    standard_top100_set = {(c.video_id, c.frame_id) for c in standard_final_candidates}
    pre_anchor_final_rank_map = {
        (c.video_id, c.frame_id): rank
        for rank, c in enumerate(standard_final_candidates, start=1)
    }
    standard_cutoff_score = float(standard_final_candidates[-1].score) if standard_final_candidates else None

    # 2. Hierarchical Local Anchor Selection (Two-Phase Membership Protocol)
    naturally_satisfied_anchors: dict[str, tuple[str, int]] = {}  # vid -> (vid, fid)
    queued_new_anchors: list[dict[str, object]] = []
    anchor_decisions_by_video: dict[str, dict[str, object]] = {}

    if enable_top_video_local_anchor:
        top_vids_sorted = [
            item.video_id
            for item in sorted(selected_videos, key=lambda v: v.rank)[:local_anchor_top_video_count]
        ]
        vid_nomination_rank_map = {item.video_id: item.rank for item in selected_videos}
        semantic_variant_weights = {
            v.variant_id: v.weight
            for v in variants
        }

        # Index primary candidates of top videos
        primary_by_top_vid: dict[str, list[CandidateFrame]] = {}
        for cand in primary_candidates:
            if cand.video_id in top_vids_sorted:
                primary_by_top_vid.setdefault(cand.video_id, []).append(cand)

        for vid in top_vids_sorted:
            cands_for_vid = primary_by_top_vid.get(vid, [])
            nom_rank = vid_nomination_rank_map.get(vid, 1)

            if not cands_for_vid:
                anchor_decisions_by_video[vid] = {
                    "status": "NO_ELIGIBLE_LOCAL_ANCHOR",
                    "anchor_key": None,
                    "nomination_rank": nom_rank,
                }
                continue

            if local_anchor_source == "SEMANTIC_LOCAL":
                eligible_ranked: list[tuple[tuple, CandidateFrame, int]] = []
                for cand in cands_for_vid:
                    diag = cand.diagnostic_metadata or {}
                    sel_map = diag.get("selection_by_variant", {})
                    sem_local_ranks = [
                        vdata.get("raw_local_rank", 999999)
                        for var_id, vdata in sel_map.items()
                        if var_id in semantic_variant_weights and "raw_local_rank" in vdata
                    ]
                    if not sem_local_ranks:
                        continue
                    min_sem_rank = min(sem_local_ranks)
                    sem_rrf = sum(
                        semantic_variant_weights.get(var_id, 1.0) / (60.0 + vdata.get("raw_local_rank", 999999))
                        for var_id, vdata in sel_map.items()
                        if var_id in semantic_variant_weights and "raw_local_rank" in vdata
                    )
                    sort_key = (min_sem_rank, -sem_rrf, -cand.score, cand.frame_id)
                    eligible_ranked.append((sort_key, cand, min_sem_rank))

                if not eligible_ranked:
                    anchor_decisions_by_video[vid] = {
                        "status": "NO_ELIGIBLE_LOCAL_ANCHOR",
                        "anchor_key": None,
                        "nomination_rank": nom_rank,
                    }
                    continue

                eligible_ranked.sort(key=lambda item: item[0])
                _, best_cand, anchor_local_rank = eligible_ranked[0]

            elif local_anchor_source == "VI_LOCAL":
                eligible_ranked_vi: list[tuple[tuple, CandidateFrame, int]] = []
                for cand in cands_for_vid:
                    diag = cand.diagnostic_metadata or {}
                    sel_map = diag.get("selection_by_variant", {})
                    scores_map = diag.get("scores_by_variant", {})
                    vi_entry = None
                    vi_cosine = 0.0
                    for var_id, vdata in sel_map.items():
                        if "vi_local" in var_id:
                            vi_entry = vdata
                            vi_cosine = float(scores_map.get(var_id, 0.0))
                            break
                    if vi_entry is None or "raw_local_rank" not in vi_entry:
                        continue
                    vi_rank = vi_entry["raw_local_rank"]
                    sort_key = (vi_rank, -vi_cosine, -cand.score, cand.frame_id)
                    eligible_ranked_vi.append((sort_key, cand, vi_rank))

                if not eligible_ranked_vi:
                    anchor_decisions_by_video[vid] = {
                        "status": "NO_ELIGIBLE_LOCAL_ANCHOR",
                        "anchor_key": None,
                        "nomination_rank": nom_rank,
                    }
                    continue

                eligible_ranked_vi.sort(key=lambda item: item[0])
                _, best_cand, anchor_local_rank = eligible_ranked_vi[0]

            best_k = (best_cand.video_id, best_cand.frame_id)
            # Compute temporal distance to nearest exported frame of same video in standard Top 100
            exported_same_vid_pts = [
                c.diagnostic_metadata.get("selection_by_variant", {}).get(
                    next(iter(c.diagnostic_metadata.get("selection_by_variant", {})), ""), {}
                ).get("pts_time", c.frame_id / 25.0)
                if c.diagnostic_metadata and "selection_by_variant" in c.diagnostic_metadata
                else c.frame_id / 25.0
                for c in standard_final_candidates
                if c.video_id == vid
            ]
            best_pts = (
                best_cand.diagnostic_metadata.get("selection_by_variant", {}).get(
                    next(iter(best_cand.diagnostic_metadata.get("selection_by_variant", {})), ""), {}
                ).get("pts_time", best_cand.frame_id / 25.0)
                if best_cand.diagnostic_metadata and "selection_by_variant" in best_cand.diagnostic_metadata
                else best_cand.frame_id / 25.0
            )
            temp_dist = (
                min(abs(best_pts - p) for p in exported_same_vid_pts)
                if exported_same_vid_pts
                else None
            )

            if best_k in standard_top100_set:
                naturally_satisfied_anchors[vid] = best_k
                anchor_decisions_by_video[vid] = {
                    "status": "LOCAL_ANCHOR_SATISFIED_NATURALLY",
                    "anchor_key": f"{best_cand.video_id}::{best_cand.frame_id}",
                    "nomination_rank": nom_rank,
                    "anchor_local_rank": anchor_local_rank,
                    "anchor_signal_source": local_anchor_source,
                    "anchor_temporal_distance_to_nearest_exported": temp_dist,
                }
            else:
                queued_new_anchors.append({
                    "candidate": best_cand,
                    "key": best_k,
                    "nomination_rank": nom_rank,
                    "anchor_local_rank": anchor_local_rank,
                    "temporal_dist": temp_dist,
                })
                anchor_decisions_by_video[vid] = {
                    "status": "VIDEO_LOCAL_RESERVED",
                    "anchor_key": f"{best_cand.video_id}::{best_cand.frame_id}",
                    "nomination_rank": nom_rank,
                    "anchor_local_rank": anchor_local_rank,
                    "anchor_signal_source": local_anchor_source,
                    "anchor_temporal_distance_to_nearest_exported": temp_dist,
                }

    # 3. Batch Displacement
    m = len(queued_new_anchors)
    protected_keys = set(naturally_satisfied_anchors.values())
    displaced_candidates: list[CandidateFrame] = []
    displaced_details: list[dict[str, object]] = []

    if m > 0:
        # Sort queued anchors by (nomination_rank, anchor_local_rank, video_id, frame_id)
        queued_new_anchors.sort(
            key=lambda a: (
                a["nomination_rank"],
                a["anchor_local_rank"],
                a["candidate"].video_id,
                a["candidate"].frame_id,
            )
        )

        # Identify m lowest-ranked candidates in standard_final_candidates that are NOT in protected_keys
        vacant_indices: list[int] = []
        for idx in range(len(standard_final_candidates) - 1, -1, -1):
            cand = standard_final_candidates[idx]
            if (cand.video_id, cand.frame_id) not in protected_keys:
                vacant_indices.append(idx)
                if len(vacant_indices) == m:
                    break

        vacant_indices.sort()
        post_anchor_final_candidates = list(standard_final_candidates)

        for anchor_item, v_idx in zip(queued_new_anchors, vacant_indices, strict=True):
            disp_cand = post_anchor_final_candidates[v_idx]
            displaced_candidates.append(disp_cand)
            disp_key = f"{disp_cand.video_id}::{disp_cand.frame_id}"
            anchor_cand = anchor_item["candidate"]
            anchor_key = f"{anchor_cand.video_id}::{anchor_cand.frame_id}"
            anchor_rank = v_idx + 1

            displaced_details.append({
                "displaced_key": disp_key,
                "displaced_video": disp_cand.video_id,
                "displaced_frame_id": disp_cand.frame_id,
                "displaced_original_rank": anchor_rank,
                "displaced_score": float(disp_cand.score),
                "by_anchor_key": anchor_key,
                "by_anchor_video": anchor_cand.video_id,
            })

            anchor_item["displaced_candidate_key"] = disp_key
            anchor_item["displaced_candidate_score"] = float(disp_cand.score)
            anchor_item["displaced_original_rank"] = anchor_rank
            anchor_item["reserved_slot_index"] = anchor_rank

            post_anchor_final_candidates[v_idx] = anchor_cand
    else:
        post_anchor_final_candidates = standard_final_candidates

    post_anchor_final_rank_map = {
        (c.video_id, c.frame_id): rank
        for rank, c in enumerate(post_anchor_final_candidates, start=1)
    }
    post_anchor_min_exported_score = (
        float(min(c.score for c in post_anchor_final_candidates))
        if post_anchor_final_candidates
        else None
    )
    displaced_keys_map = {
        (d["displaced_video"], d["displaced_frame_id"]): d for d in displaced_details
    }
    queued_anchor_map = {
        a["key"]: a for a in queued_new_anchors
    }

    fusion_trace_map: dict[tuple[str, int], dict[str, object]] = {}
    if collect_fusion_trace and full_fused is not None:
        num_exported_primary = min(len(primary_candidates), output_top_k)
        remaining_slots = output_top_k - num_exported_primary
        num_exported_secondary = min(len(secondary_candidates), remaining_slots)
        total_exported = len(post_anchor_final_candidates)

        primary_cutoff_cand = primary_candidates[num_exported_primary - 1] if num_exported_primary > 0 else None
        secondary_cutoff_cand = secondary_candidates[num_exported_secondary - 1] if num_exported_secondary > 0 else None

        allocation_summary_payload: dict[str, object] = {
            "fusion_trace_schema_version": "2.1.0",
            "total_restricted_candidates_untruncated": len(full_fused.ranked_candidates),
            "internal_rrf_candidate_depth": fused_top_k,
            "candidates_passed_internal_cutoff": len(enriched_sorted),
            "total_primary_candidates": len(primary_candidates),
            "total_secondary_candidates": len(secondary_candidates),
            "output_top_k": output_top_k,
            "exported_primary_count": num_exported_primary,
            "exported_secondary_count": num_exported_secondary,
            "total_exported": total_exported,
            "primary_cutoff_candidate_key": f"{primary_cutoff_cand.video_id}::{primary_cutoff_cand.frame_id}" if primary_cutoff_cand else None,
            "primary_cutoff_score": float(primary_cutoff_cand.score) if primary_cutoff_cand else None,
            "secondary_cutoff_candidate_key": f"{secondary_cutoff_cand.video_id}::{secondary_cutoff_cand.frame_id}" if secondary_cutoff_cand else None,
            "secondary_cutoff_score": float(secondary_cutoff_cand.score) if secondary_cutoff_cand else None,
            "standard_cutoff_score": standard_cutoff_score,
            "post_anchor_min_exported_score": post_anchor_min_exported_score,
            "total_local_anchors_reserved": len(queued_new_anchors),
            "naturally_satisfied_anchor_count": len(naturally_satisfied_anchors),
            "displaced_candidate_count": len(displaced_details),
            "anchor_decisions_by_video": anchor_decisions_by_video,
            "displaced_candidates": displaced_details,
        }
        fusion_trace_map[("__summary__", 0)] = allocation_summary_payload

        # Pre-compute rank maps in O(N)
        pre_alloc_global_ranks: dict[tuple[str, int], int] = {
            (c.video_id, c.frame_id): rank
            for rank, c in enumerate(enriched_sorted, start=1)
        }
        primary_ranks: dict[tuple[str, int], int] = {
            (c.video_id, c.frame_id): rank
            for rank, c in enumerate(primary_candidates, start=1)
        }
        secondary_ranks: dict[tuple[str, int], int] = {
            (c.video_id, c.frame_id): rank
            for rank, c in enumerate(secondary_candidates, start=1)
        }
        enriched_cand_map: dict[tuple[str, int], CandidateFrame] = {
            (c.video_id, c.frame_id): c for c in enriched_sorted
        }

        variant_ranks_per_frame: dict[tuple[str, int], dict[str, int]] = {}
        variant_scores_per_frame: dict[tuple[str, int], dict[str, float]] = {}
        variant_selection_per_frame: dict[tuple[str, int], dict[str, dict[str, object]]] = {}
        for var_id, r in rankings.items():
            for hit in r.ranked_candidates:
                k = (hit.video_id, hit.frame_id)
                variant_ranks_per_frame.setdefault(k, {})[var_id] = hit.rank
                variant_scores_per_frame.setdefault(k, {})[var_id] = float(hit.score)
                if hit.diagnostic_metadata:
                    variant_selection_per_frame.setdefault(k, {})[var_id] = {
                        "selection_source": hit.diagnostic_metadata.get("selection_source", "RAW"),
                        "raw_local_rank": hit.diagnostic_metadata.get("raw_local_rank", 0),
                        "pts_time": hit.diagnostic_metadata.get("pts_time"),
                    }

        for item in full_fused.ranked_candidates:
            k = (item.video_id, item.frame_id)
            untruncated_rrf_rank = item.rank
            passed_internal_cutoff = (untruncated_rrf_rank <= fused_top_k)

            sel_info = variant_selection_per_frame.get(k, {})
            sources = {v.get("selection_source") for v in sel_info.values() if v.get("selection_source")}
            source_str = "/".join(sorted(str(s) for s in sources)) if sources else "RAW"

            if not passed_internal_cutoff:
                # Candidate pruned before reaching allocation stage
                fusion_trace_map[k] = {
                    "video_id": item.video_id,
                    "frame_id": item.frame_id,
                    "restricted_selection_status": f"SELECTED_RESTRICTED_{source_str}",
                    "selection_by_variant": sel_info,
                    "global_variant_ranks": variant_ranks_per_frame.get(k, {}),
                    "scores_by_variant": variant_scores_per_frame.get(k, {}),
                    "untruncated_rrf_rank": untruncated_rrf_rank,
                    "rrf_score": float(item.score),
                    "rrf_cutoff_status": "PRUNED_BY_INTERNAL_RRF_CUTOFF",
                    "group_bucket": None,
                    "final_selection_score": None,
                    "final_sort_key": None,
                    "pre_anchor_sort_key": None,
                    "pre_dedup_sort_key": None,
                    "pre_allocation_global_rank": None,
                    "pre_allocation_bucket_rank": None,
                    "effective_cutoff_score": None,
                    "effective_cutoff_scope": None,
                    "effective_cutoff_candidate_key": None,
                    "score_gap_to_effective_cutoff": None,
                    "allocation_selection_reason": None,
                    "allocation_rejection_reason": None,
                    "tie_break_reason": None,
                    "pre_anchor_final_rank": None,
                    "pre_anchor_lifecycle_status": "PRUNED_BY_INTERNAL_RRF_CUTOFF",
                    "pre_anchor_allocation_rejection_reason": None,
                    "post_anchor_final_rank": None,
                    "final_rank": None,
                    "final_lifecycle_status": "PRUNED_BY_INTERNAL_RRF_CUTOFF",
                }
                continue

            # Candidate reached allocation stage
            cand = enriched_cand_map[k]
            pre_alloc_global_rank = pre_alloc_global_ranks[k]
            final_selection_score = float(cand.score)
            pre_anchor_rank = pre_anchor_final_rank_map.get(k)
            post_anchor_rank = post_anchor_final_rank_map.get(k)
            final_rank = post_anchor_rank

            pre_dedup_sort_key = None
            if min_frame_gap > 0:
                if k in temporal_dedup_pruned_keys:
                    group_bucket = "TEMPORAL_DEDUP_PRUNED"
                    pre_alloc_bucket_rank = None
                    final_sort_key = None
                    pre_dedup_sort_key = [-final_selection_score, cand.video_id, cand.frame_id]
                    effective_cutoff_score = None
                    effective_cutoff_scope = "TEMPORAL_DEDUP"
                    effective_cutoff_cand_key = None
                    score_gap = None
                    rejection_reason = "TEMPORAL_DEDUP_REJECTED"
                    tie_break_msg = None
                    lifecycle_status = "PRUNED_BY_TEMPORAL_DEDUP"
                else:
                    group_bucket = "GROUP_BUCKET_PRIMARY"
                    pre_alloc_bucket_rank = primary_ranks[k]
                    final_sort_key = [0, -final_selection_score, cand.video_id, cand.frame_id]
                    effective_cutoff_score = float(primary_cutoff_cand.score) if primary_cutoff_cand else None
                    effective_cutoff_scope = "GLOBAL"
                    effective_cutoff_cand_key = f"{primary_cutoff_cand.video_id}::{primary_cutoff_cand.frame_id}" if primary_cutoff_cand else None
                    if pre_anchor_rank is not None:
                        lifecycle_status = f"EXPORTED_AT_RANK_{pre_anchor_rank}"
                        rejection_reason = None
                        score_gap = None
                        tie_break_msg = None
                    else:
                        lifecycle_status = "PRUNED_BY_FINAL_TOPK"
                        if effective_cutoff_score is not None and cand.score == effective_cutoff_score:
                            score_gap = 0.0
                            rejection_reason = "TIE_BREAK_REJECTED"
                            tie_break_msg = f"TIE_ON_SCORE_{cand.score:.8f}_RESOLVED_BY_LEXICOGRAPHICAL_KEY_{effective_cutoff_cand_key}"
                        else:
                            score_gap = float(effective_cutoff_score - cand.score) if effective_cutoff_score is not None else None
                            rejection_reason = "SCORE_BELOW_EFFECTIVE_CUTOFF"
                            tie_break_msg = None
            else:
                # Segment Grouper mode
                if k in primary_ranks:
                    group_bucket = "GROUP_BUCKET_PRIMARY"
                    pre_alloc_bucket_rank = primary_ranks[k]
                    final_sort_key = [0, -final_selection_score, cand.video_id, cand.frame_id]
                    effective_cutoff_score = float(primary_cutoff_cand.score) if primary_cutoff_cand else None
                    effective_cutoff_scope = "PRIMARY_BUCKET"
                    effective_cutoff_cand_key = f"{primary_cutoff_cand.video_id}::{primary_cutoff_cand.frame_id}" if primary_cutoff_cand else None

                    if pre_anchor_rank is not None:
                        lifecycle_status = f"EXPORTED_AT_RANK_{pre_anchor_rank}"
                        rejection_reason = None
                        score_gap = None
                        tie_break_msg = None
                    else:
                        lifecycle_status = "PRUNED_BY_FINAL_TOPK"
                        if effective_cutoff_score is not None and cand.score == effective_cutoff_score:
                            score_gap = 0.0
                            rejection_reason = "TIE_BREAK_REJECTED"
                            tie_break_msg = f"TIE_ON_SCORE_{cand.score:.8f}_RESOLVED_BY_LEXICOGRAPHICAL_KEY_{effective_cutoff_cand_key}"
                        else:
                            score_gap = float(effective_cutoff_score - cand.score) if effective_cutoff_score is not None else None
                            rejection_reason = "SCORE_BELOW_EFFECTIVE_CUTOFF"
                            tie_break_msg = None
                else:
                    group_bucket = "GROUP_BUCKET_SECONDARY"
                    pre_alloc_bucket_rank = secondary_ranks[k]
                    final_sort_key = [1, -final_selection_score, cand.video_id, cand.frame_id]

                    if num_exported_primary >= output_top_k:
                        effective_cutoff_score = float(primary_cutoff_cand.score) if primary_cutoff_cand else None
                        effective_cutoff_scope = "PRIMARY_BUCKET"
                        effective_cutoff_cand_key = f"{primary_cutoff_cand.video_id}::{primary_cutoff_cand.frame_id}" if primary_cutoff_cand else None
                        lifecycle_status = "PRUNED_BY_FINAL_TOPK"
                        rejection_reason = "PRIMARY_BUCKET_SATURATED_OUTPUT"
                        score_gap = None
                        tie_break_msg = None
                    else:
                        effective_cutoff_score = float(secondary_cutoff_cand.score) if secondary_cutoff_cand else None
                        effective_cutoff_scope = "SECONDARY_BUCKET"
                        effective_cutoff_cand_key = f"{secondary_cutoff_cand.video_id}::{secondary_cutoff_cand.frame_id}" if secondary_cutoff_cand else None
                        if pre_anchor_rank is not None:
                            lifecycle_status = f"EXPORTED_AT_RANK_{pre_anchor_rank}"
                            rejection_reason = None
                            score_gap = None
                            tie_break_msg = None
                        else:
                            lifecycle_status = "PRUNED_BY_FINAL_TOPK"
                            if effective_cutoff_score is not None and cand.score == effective_cutoff_score:
                                score_gap = 0.0
                                rejection_reason = "TIE_BREAK_REJECTED"
                                tie_break_msg = f"TIE_ON_SCORE_{cand.score:.8f}_RESOLVED_BY_LEXICOGRAPHICAL_KEY_{effective_cutoff_cand_key}"
                            else:
                                score_gap = float(effective_cutoff_score - cand.score) if effective_cutoff_score is not None else None
                                rejection_reason = "SCORE_BELOW_EFFECTIVE_CUTOFF"
                                tie_break_msg = None

            pre_anchor_lifecycle_status = lifecycle_status
            pre_anchor_rejection_reason = rejection_reason
            selection_reason = None
            anchor_sig_source = None
            anchor_loc_rank = None
            anchor_nom_rank = None
            anchor_temp_dist = None
            disp_cand_key = None
            disp_cand_score = None
            disp_orig_rank = None
            res_slot_idx = None
            displacing_key = None
            displacing_vid = None

            # Apply Phase B2 Post-Anchor Logic
            if k in naturally_satisfied_anchors.values():
                selection_reason = "LOCAL_ANCHOR_SATISFIED_NATURALLY"
                rejection_reason = None
                lifecycle_status = f"EXPORTED_AT_RANK_{final_rank}"
                anchor_dec = anchor_decisions_by_video.get(k[0], {})
                anchor_sig_source = anchor_dec.get("anchor_signal_source")
                anchor_loc_rank = anchor_dec.get("anchor_local_rank")
                anchor_nom_rank = anchor_dec.get("nomination_rank")
                anchor_temp_dist = anchor_dec.get("anchor_temporal_distance_to_nearest_exported")
            elif k in queued_anchor_map:
                a_info = queued_anchor_map[k]
                selection_reason = "VIDEO_LOCAL_RESERVED"
                rejection_reason = None
                lifecycle_status = f"EXPORTED_AT_RANK_{final_rank}"
                anchor_sig_source = local_anchor_source
                anchor_loc_rank = a_info.get("anchor_local_rank")
                anchor_nom_rank = a_info.get("nomination_rank")
                anchor_temp_dist = a_info.get("temporal_dist")
                disp_cand_key = a_info.get("displaced_candidate_key")
                disp_cand_score = a_info.get("displaced_candidate_score")
                disp_orig_rank = a_info.get("displaced_original_rank")
                res_slot_idx = a_info.get("reserved_slot_index")
            elif k in displaced_keys_map:
                d_info = displaced_keys_map[k]
                selection_reason = None
                rejection_reason = "DISPLACED_BY_LOCAL_ANCHOR"
                lifecycle_status = "PRUNED_BY_LOCAL_ANCHOR_DISPLACEMENT"
                displacing_key = d_info.get("by_anchor_key")
                displacing_vid = d_info.get("by_anchor_video")
            elif final_rank is not None:
                selection_reason = "STANDARD_TOPK_SELECTION"
                rejection_reason = None
                lifecycle_status = f"EXPORTED_AT_RANK_{final_rank}"

            fusion_trace_map[k] = {
                "video_id": item.video_id,
                "frame_id": item.frame_id,
                "restricted_selection_status": f"SELECTED_RESTRICTED_{source_str}",
                "selection_by_variant": sel_info,
                "global_variant_ranks": variant_ranks_per_frame.get(k, {}),
                "scores_by_variant": variant_scores_per_frame.get(k, {}),
                "untruncated_rrf_rank": untruncated_rrf_rank,
                "rrf_score": float(item.score),
                "rrf_cutoff_status": "PASSED_RRF_CUTOFF",
                "group_bucket": group_bucket,
                "final_selection_score": final_selection_score,
                "final_sort_key": final_sort_key,
                "pre_anchor_sort_key": final_sort_key,
                "pre_dedup_sort_key": pre_dedup_sort_key,
                "pre_allocation_global_rank": pre_alloc_global_rank,
                "pre_allocation_bucket_rank": pre_alloc_bucket_rank,
                "effective_cutoff_score": effective_cutoff_score,
                "effective_cutoff_scope": effective_cutoff_scope,
                "effective_cutoff_candidate_key": effective_cutoff_cand_key,
                "score_gap_to_effective_cutoff": score_gap,
                "allocation_selection_reason": selection_reason,
                "allocation_rejection_reason": rejection_reason,
                "tie_break_reason": tie_break_msg,
                "pre_anchor_final_rank": pre_anchor_rank,
                "pre_anchor_lifecycle_status": pre_anchor_lifecycle_status,
                "pre_anchor_allocation_rejection_reason": pre_anchor_rejection_reason,
                "post_anchor_final_rank": post_anchor_rank,
                "final_rank": final_rank,
                "final_lifecycle_status": lifecycle_status,
                "anchor_signal_source": anchor_sig_source,
                "anchor_local_rank": anchor_loc_rank,
                "anchor_video_nomination_rank": anchor_nom_rank,
                "anchor_temporal_distance_to_nearest_exported": anchor_temp_dist,
                "displaced_candidate_key": disp_cand_key,
                "displaced_candidate_score": disp_cand_score,
                "displaced_original_rank": disp_orig_rank,
                "reserved_slot_index": res_slot_idx,
                "displacing_anchor_key": displacing_key,
                "displacing_anchor_video": displacing_vid,
            }

    reranked = tuple(
        CandidateFrame(
            video_id=cand.video_id,
            frame_id=cand.frame_id,
            clip_row=cand.clip_row,
            keyframe_order=cand.keyframe_order,
            score=cand.score,
            rank=rank,
            source=cand.source,
            diagnostic_metadata=cand.diagnostic_metadata,
        )
        for rank, cand in enumerate(post_anchor_final_candidates, start=1)
    )
    result = KISResult(query_id=query_id, ranked_candidates=reranked)
    if return_trace:
        return result, fusion_trace_map
    return result


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
    config: KISVideoFirstConfig | None = None,
) -> KISVideoFirstOutcome:
    chain_bonus = config.temporal_chain_frame_bonus if config else 0.0
    min_gap = config.restricted_frame_min_gap if config else 0
    max_cands = config.max_restricted_candidates_per_video if config else None
    internal_depth = config.internal_rrf_candidate_depth if config else 100
    collect_trace = config.collect_fusion_trace if config else False
    enable_anchor = config.enable_top_video_local_anchor if config else False
    anchor_count = config.local_anchor_top_video_count if config else 3
    anchor_src = config.local_anchor_source if config else "SEMANTIC_LOCAL"

    if collect_trace:
        fuse_res = fuse_restricted_frames(
            query_id=query_id,
            variants=variants,
            restricted=restricted,
            selected_videos=selected_videos,
            weighted_rrf=weighted_rrf,
            output_top_k=output_top_k,
            rrf_constant=rrf_constant,
            temporal_chain_bonus=chain_bonus,
            min_frame_gap=min_gap,
            max_candidates_per_video=max_cands,
            internal_rrf_candidate_depth=internal_depth,
            collect_fusion_trace=True,
            return_trace=True,
            enable_top_video_local_anchor=enable_anchor,
            local_anchor_top_video_count=anchor_count,
            local_anchor_source=anchor_src,
        )
        assert isinstance(fuse_res, tuple)
        result, fusion_trace_map = fuse_res
    else:
        fuse_res = fuse_restricted_frames(
            query_id=query_id,
            variants=variants,
            restricted=restricted,
            selected_videos=selected_videos,
            weighted_rrf=weighted_rrf,
            output_top_k=output_top_k,
            rrf_constant=rrf_constant,
            temporal_chain_bonus=chain_bonus,
            min_frame_gap=min_gap,
            max_candidates_per_video=max_cands,
            internal_rrf_candidate_depth=internal_depth,
            collect_fusion_trace=False,
            return_trace=False,
            enable_top_video_local_anchor=enable_anchor,
            local_anchor_top_video_count=anchor_count,
            local_anchor_source=anchor_src,
        )
        assert isinstance(fuse_res, KISResult)
        result = fuse_res
        fusion_trace_map = None

    per_scene_top128 = {
        var_id: tuple(hit.video_id for hit in hits[:128])
        for var_id, hits in maxima.rankings.items()
    }
    per_scene_ranks = {
        var_id: {hit.video_id: hit.rank for hit in hits}
        for var_id, hits in maxima.rankings.items()
    }
    is_exp = bool(
        config
        and (
            config.restricted_frames_per_video_per_variant != 10
            or config.enable_temporal_diverse_local_candidates
            or config.enable_vi_localization_variant
            or config.internal_rrf_candidate_depth != 100
            or config.enable_top_video_local_anchor
        )
    )
    return KISVideoFirstOutcome(
        result=result,
        selected_videos=tuple(selected_videos),
        full_corpus_rows_scored=maxima.physical_rows_scored,
        full_corpus_store_scan_count=maxima.video_store_scan_count,
        restricted_rows_scored=restricted.physical_rows_scored,
        restricted_store_scan_count=restricted.video_store_scan_count,
        adaptive_diagnostic=adaptive_diagnostic,
        per_scene_top128=per_scene_top128,
        per_scene_ranks=per_scene_ranks,
        candidate_selection_telemetry=restricted.candidate_selection_telemetry if is_exp else None,
        fusion_trace=fusion_trace_map if collect_trace else None,
    )
