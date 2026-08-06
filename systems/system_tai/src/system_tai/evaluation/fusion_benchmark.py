"""Evaluation of opt-in Weighted RRF over comparable verified query groups."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from statistics import fmean
from typing import Any

from system_tai.evaluation.benchmark_schema import (
    AnnotationStatus,
    BenchmarkQuery,
    KISBenchmark,
    VariantType,
)
from system_tai.evaluation.benchmark_validator import BenchmarkValidationResult
from system_tai.retrieval.multi_query import (
    QueryLanguage,
    QueryVariant,
    QueryVariantType,
    WeightedRRFRetriever,
)

REQUIRED_VARIANT_TYPES = (
    VariantType.VIETNAMESE_DIRECT,
    VariantType.ENGLISH_TRANSLATION,
    VariantType.ENGLISH_EXPANSION,
)


@dataclass(frozen=True, slots=True)
class FusionGroupIssue:
    semantic_group_id: str
    code: str
    message: str


@dataclass(frozen=True, slots=True)
class ComparableFusionGroup:
    semantic_group_id: str
    variants: tuple[BenchmarkQuery, ...]
    relevant_pairs: tuple[tuple[str, int], ...]
    relevant_video_ids: tuple[str, ...]
    source_scope: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FusionGroupSelection:
    groups: tuple[ComparableFusionGroup, ...]
    issues: tuple[FusionGroupIssue, ...]
    draft_query_count: int


@dataclass(frozen=True, slots=True)
class FusionGroupMetrics:
    semantic_group_id: str
    contributing_variant_ids: tuple[str, ...]
    contributing_variant_count: int
    relevant_label_count: int
    first_relevant_rank: int | None
    reciprocal_rank: float
    recall_at_k: tuple[tuple[int, float], ...]
    ground_truth_coverage_at_k: tuple[tuple[int, float], ...]
    hit_count_at_k: tuple[tuple[int, int], ...]


@dataclass(frozen=True, slots=True)
class FusionBenchmarkReport:
    evaluation_state: str
    benchmark_id: str
    schema_version: int
    evaluated_group_count: int
    evaluated_verified_query_count: int
    excluded_draft_query_count: int
    invalid_query_count: int
    source_video_scope: tuple[str, ...]
    model_identifier: str
    model_metadata: dict[str, Any]
    device: str
    retrieval_implementation: str
    canonical_per_variant_unsuppressed: bool
    fusion_method: str
    rrf_constant: float
    top_k_per_variant: int
    output_top_k: int
    variant_weights: tuple[tuple[str, float], ...]
    top_ks: tuple[int, ...]
    metric_definitions: dict[str, str]
    group_metrics: tuple[FusionGroupMetrics, ...]
    mean_reciprocal_rank: float
    mean_recall_at_k: tuple[tuple[int, float], ...]
    mean_ground_truth_coverage_at_k: tuple[tuple[int, float], ...]
    mean_hit_count_at_k: tuple[tuple[int, float], ...]
    group_issues: tuple[FusionGroupIssue, ...]
    limitations: tuple[str, ...]


class NoComparableFusionGroupsError(ValueError):
    """Raised when no verified three-variant group can be measured."""


def select_comparable_fusion_groups(benchmark: KISBenchmark) -> FusionGroupSelection:
    grouped: dict[str, list[BenchmarkQuery]] = defaultdict(list)
    draft_count = 0
    for query in benchmark.queries:
        grouped[query.semantic_group_id].append(query)
        if query.annotation_status is AnnotationStatus.DRAFT:
            draft_count += 1

    comparable: list[ComparableFusionGroup] = []
    issues: list[FusionGroupIssue] = []
    for group_id, queries in sorted(grouped.items()):
        verified = [
            query
            for query in queries
            if query.annotation_status is AnnotationStatus.VERIFIED
        ]
        counts = Counter(query.variant_type for query in verified)
        group_failed = False
        for variant_type in REQUIRED_VARIANT_TYPES:
            count = counts[variant_type]
            if count == 0:
                issues.append(
                    FusionGroupIssue(
                        semantic_group_id=group_id,
                        code="MISSING_VERIFIED_VARIANT",
                        message=f"missing verified {variant_type.value} variant",
                    )
                )
                group_failed = True
            elif count > 1:
                issues.append(
                    FusionGroupIssue(
                        semantic_group_id=group_id,
                        code="DUPLICATE_VERIFIED_VARIANT",
                        message=f"found {count} verified {variant_type.value} variants",
                    )
                )
                group_failed = True
        if group_failed:
            continue
        ordered = tuple(
            next(query for query in verified if query.variant_type is variant_type)
            for variant_type in REQUIRED_VARIANT_TYPES
        )
        reference = ordered[0]
        reference_pairs = {
            (label.video_id, label.frame_id) for label in reference.relevant_frames
        }
        reference_videos = set(reference.relevant_video_ids)
        reference_scope = set(reference.source_scope)
        if any(
            {
                (label.video_id, label.frame_id)
                for label in query.relevant_frames
            }
            != reference_pairs
            or set(query.relevant_video_ids) != reference_videos
            or set(query.source_scope) != reference_scope
            for query in ordered[1:]
        ):
            issues.append(
                FusionGroupIssue(
                    semantic_group_id=group_id,
                    code="INCOMPARABLE_VERIFIED_VARIANTS",
                    message=(
                        "verified variants must share relevant frames, relevant videos, "
                        "and source scope"
                    ),
                )
            )
            continue
        comparable.append(
            ComparableFusionGroup(
                semantic_group_id=group_id,
                variants=ordered,
                relevant_pairs=tuple(sorted(reference_pairs)),
                relevant_video_ids=tuple(sorted(reference_videos)),
                source_scope=tuple(sorted(reference_scope)),
            )
        )
    return FusionGroupSelection(
        groups=tuple(comparable),
        issues=tuple(issues),
        draft_query_count=draft_count,
    )


class FusionBenchmarkEvaluator:
    DEFAULT_LIMITATIONS = (
        "This is a three-video, three-intent pilot and not official BTC performance.",
        "Positive labels were selected after retrieval inspection and therefore "
        "carry retrieval-selection bias.",
        "Weighted RRF is opt-in; the canonical single-query exact baseline remains unchanged.",
        "No conclusion about a dataset-wide multilingual policy is supported.",
    )

    def evaluate(
        self,
        validation: BenchmarkValidationResult,
        retriever: WeightedRRFRetriever,
        *,
        top_ks: tuple[int, ...] = (1, 5, 20, 50, 100),
        top_k_per_variant: int = 100,
        rrf_constant: float = 60.0,
        variant_weights: Mapping[VariantType | str, float] | None = None,
        limitations: tuple[str, ...] | None = None,
    ) -> FusionBenchmarkReport:
        if not validation.valid or validation.benchmark is None:
            raise ValueError("cannot evaluate an invalid benchmark")
        resolved_ks = self._validate_top_ks(top_ks)
        if top_k_per_variant <= 0:
            raise ValueError("top_k_per_variant must be positive")
        weights = self._resolve_weights(variant_weights)
        selection = select_comparable_fusion_groups(validation.benchmark)
        if not selection.groups:
            issue_codes = ", ".join(issue.code for issue in selection.issues) or "none"
            raise NoComparableFusionGroupsError(
                f"no comparable verified fusion groups; issues={issue_codes}"
            )

        output_top_k = max(resolved_ks)
        metrics = tuple(
            self._evaluate_group(
                group,
                retriever,
                resolved_ks,
                top_k_per_variant=top_k_per_variant,
                output_top_k=output_top_k,
                rrf_constant=rrf_constant,
                weights=weights,
            )
            for group in selection.groups
        )
        exact_retriever = retriever.exact_retriever
        metadata = dict(exact_retriever.text_encoder.identifiers)
        return FusionBenchmarkReport(
            evaluation_state="completed",
            benchmark_id=validation.benchmark.benchmark_id,
            schema_version=validation.benchmark.schema_version,
            evaluated_group_count=len(metrics),
            evaluated_verified_query_count=sum(
                len(group.variants) for group in selection.groups
            ),
            excluded_draft_query_count=selection.draft_query_count,
            invalid_query_count=0,
            source_video_scope=tuple(
                sorted(
                    {
                        video_id
                        for group in selection.groups
                        for video_id in group.source_scope
                    }
                )
            ),
            model_identifier=str(metadata.get("model", "unknown")),
            model_metadata=metadata,
            device=str(metadata.get("device", "unknown")),
            retrieval_implementation="exact_chunked_numpy_cosine_per_variant",
            canonical_per_variant_unsuppressed=True,
            fusion_method="weighted_reciprocal_rank_fusion",
            rrf_constant=rrf_constant,
            top_k_per_variant=top_k_per_variant,
            output_top_k=output_top_k,
            variant_weights=tuple(
                (variant_type.value, weights[variant_type])
                for variant_type in REQUIRED_VARIANT_TYPES
            ),
            top_ks=resolved_ks,
            metric_definitions={
                "weighted_rrf": (
                    "sum(weight / (rrf_constant + one_based_rank)) over variants "
                    "containing the candidate"
                ),
                "recall_at_k": (
                    "1.0 if any exact relevant (video_id, frame_id) pair appears in "
                    "fused Top-K; otherwise 0.0"
                ),
                "ground_truth_coverage_at_k": (
                    "unique relevant frame labels retrieved in fused Top-K divided by "
                    "all relevant frame labels"
                ),
                "hit_count_at_k": "number of unique relevant frame labels in fused Top-K",
                "first_relevant_rank": (
                    "minimum one-based fused rank of any relevant frame, or null on a miss"
                ),
                "reciprocal_rank": "1 / first_relevant_rank, or 0 on a miss",
            },
            group_metrics=metrics,
            mean_reciprocal_rank=fmean(metric.reciprocal_rank for metric in metrics),
            mean_recall_at_k=tuple(
                (
                    cutoff,
                    fmean(dict(metric.recall_at_k)[cutoff] for metric in metrics),
                )
                for cutoff in resolved_ks
            ),
            mean_ground_truth_coverage_at_k=tuple(
                (
                    cutoff,
                    fmean(
                        dict(metric.ground_truth_coverage_at_k)[cutoff]
                        for metric in metrics
                    ),
                )
                for cutoff in resolved_ks
            ),
            mean_hit_count_at_k=tuple(
                (
                    cutoff,
                    fmean(dict(metric.hit_count_at_k)[cutoff] for metric in metrics),
                )
                for cutoff in resolved_ks
            ),
            group_issues=selection.issues,
            limitations=limitations or self.DEFAULT_LIMITATIONS,
        )

    @staticmethod
    def _evaluate_group(
        group: ComparableFusionGroup,
        retriever: WeightedRRFRetriever,
        top_ks: tuple[int, ...],
        *,
        top_k_per_variant: int,
        output_top_k: int,
        rrf_constant: float,
        weights: dict[VariantType, float],
    ) -> FusionGroupMetrics:
        variants = tuple(
            QueryVariant(
                variant_id=query.query_id,
                text=query.text,
                language=QueryLanguage(query.language.value),
                variant_type=QueryVariantType(query.variant_type.value),
                weight=weights[query.variant_type],
            )
            for query in group.variants
        )
        result = retriever.retrieve(
            query_id=group.semantic_group_id,
            variants=variants,
            top_k_per_variant=top_k_per_variant,
            output_top_k=output_top_k,
            rrf_constant=rrf_constant,
        )
        relevant = set(group.relevant_pairs)
        ranked_pairs = [
            (candidate.video_id, candidate.frame_id)
            for candidate in result.ranked_candidates
        ]
        relevant_ranks = [
            rank
            for rank, pair in enumerate(ranked_pairs, start=1)
            if pair in relevant
        ]
        first_rank = min(relevant_ranks) if relevant_ranks else None
        recalls: list[tuple[int, float]] = []
        coverage: list[tuple[int, float]] = []
        hits: list[tuple[int, int]] = []
        for cutoff in top_ks:
            hit_count = len(relevant & set(ranked_pairs[:cutoff]))
            recalls.append((cutoff, 1.0 if hit_count else 0.0))
            coverage.append((cutoff, hit_count / len(relevant)))
            hits.append((cutoff, hit_count))
        return FusionGroupMetrics(
            semantic_group_id=group.semantic_group_id,
            contributing_variant_ids=tuple(
                variant.variant_id for variant in variants
            ),
            contributing_variant_count=len(variants),
            relevant_label_count=len(relevant),
            first_relevant_rank=first_rank,
            reciprocal_rank=0.0 if first_rank is None else 1.0 / first_rank,
            recall_at_k=tuple(recalls),
            ground_truth_coverage_at_k=tuple(coverage),
            hit_count_at_k=tuple(hits),
        )

    @staticmethod
    def _resolve_weights(
        configured: Mapping[VariantType | str, float] | None,
    ) -> dict[VariantType, float]:
        resolved = {variant_type: 1.0 for variant_type in REQUIRED_VARIANT_TYPES}
        if configured is None:
            return resolved
        for key, value in configured.items():
            variant_type = key if isinstance(key, VariantType) else VariantType(key)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"invalid positive finite weight for {variant_type.value}")
            resolved[variant_type] = float(value)
        return resolved

    @staticmethod
    def _validate_top_ks(top_ks: tuple[int, ...]) -> tuple[int, ...]:
        if not top_ks or any(type(cutoff) is not int or cutoff <= 0 for cutoff in top_ks):
            raise ValueError("top_ks must contain positive integers")
        resolved = tuple(sorted(set(top_ks)))
        if resolved[-1] > 100:
            raise ValueError("fusion benchmark Top-K cannot exceed 100")
        return resolved
