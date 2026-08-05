"""Ground-truth evaluation for canonical unsuppressed exact KIS retrieval."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from statistics import fmean
from typing import Any

from system_tai.common.schemas import KISQuery
from system_tai.evaluation.benchmark_schema import (
    BenchmarkQuery,
    VariantType,
)
from system_tai.evaluation.benchmark_validator import BenchmarkValidationResult
from system_tai.retrieval.vector_search import ExactNumpyRetriever


@dataclass(frozen=True, slots=True)
class QueryMetrics:
    query_id: str
    language: str
    variant_type: str
    semantic_group_id: str
    relevant_label_count: int
    first_relevant_rank: int | None
    reciprocal_rank: float
    recall_at_k: tuple[tuple[int, float], ...]
    ground_truth_coverage_at_k: tuple[tuple[int, float], ...]
    hit_count_at_k: tuple[tuple[int, int], ...]
    relevant_video_coverage_at_k: tuple[tuple[int, float], ...] | None


@dataclass(frozen=True, slots=True)
class AggregateMetrics:
    group_type: str
    group_value: str
    query_count: int
    mean_reciprocal_rank: float
    mean_first_relevant_rank: float | None
    queries_without_relevant_hit: int
    mean_recall_at_k: tuple[tuple[int, float], ...]
    mean_ground_truth_coverage_at_k: tuple[tuple[int, float], ...]
    mean_hit_count_at_k: tuple[tuple[int, float], ...]
    mean_relevant_video_coverage_at_k: tuple[tuple[int, float], ...] | None


@dataclass(frozen=True, slots=True)
class PairedComparison:
    semantic_group_id: str
    comparison_variant_type: str
    status: str
    vietnamese_query_id: str | None
    comparison_query_id: str | None
    vietnamese_first_relevant_rank: int | None
    comparison_first_relevant_rank: int | None
    first_relevant_rank_delta_english_minus_vietnamese: int | None
    first_relevant_rank_outcome: str | None
    recall_delta_at_k: tuple[tuple[int, float], ...]
    recall_outcome_at_k: tuple[tuple[int, str], ...]


@dataclass(frozen=True, slots=True)
class PairedSummary:
    comparison_variant_type: str
    recall_counts_at_k: tuple[tuple[int, int, int, int], ...]
    first_relevant_rank_counts: tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class KISBenchmarkReport:
    evaluation_state: str
    benchmark_id: str
    schema_version: int
    evaluated_query_count: int
    excluded_draft_query_count: int
    invalid_query_count: int
    source_video_scope: tuple[str, ...]
    model_identifier: str
    model_metadata: dict[str, Any]
    device: str
    retrieval_implementation: str
    canonical_unsuppressed: bool
    top_ks: tuple[int, ...]
    metric_definitions: dict[str, str]
    query_metrics: tuple[QueryMetrics, ...]
    aggregates: tuple[AggregateMetrics, ...]
    paired_comparisons: tuple[PairedComparison, ...]
    paired_summaries: tuple[PairedSummary, ...]
    limitations: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class NoVerifiedQueriesResult:
    evaluation_state: str
    benchmark_id: str
    schema_version: int
    evaluated_query_count: int
    excluded_draft_query_count: int
    invalid_query_count: int
    source_video_scope: tuple[str, ...]


class KISBenchmarkEvaluator:
    """Evaluate only verified labels using the unchanged exact retrieval baseline."""

    DEFAULT_LIMITATIONS = (
        "Metrics are fixture-level, not official BTC performance.",
        "Results cover only the source videos declared by human annotators.",
        "Still-frame retrieval may not fully represent actions or temporal states.",
        "Translation effects require verified paired labels and cannot be assumed.",
    )

    def evaluate(
        self,
        validation: BenchmarkValidationResult,
        retriever: ExactNumpyRetriever | None,
        *,
        top_ks: tuple[int, ...] = (1, 5, 20, 50, 100),
        limitations: tuple[str, ...] | None = None,
    ) -> KISBenchmarkReport | NoVerifiedQueriesResult:
        if not validation.valid or validation.benchmark is None:
            raise ValueError("cannot evaluate an invalid benchmark")
        resolved_ks = self._validate_top_ks(top_ks)
        verified = tuple(
            sorted(validation.verified_queries, key=lambda query: query.query_id)
        )
        if not verified:
            source_scope = tuple(
                sorted(
                    {
                        video_id
                        for query in validation.benchmark.queries
                        for video_id in query.source_scope
                    }
                )
            )
            return NoVerifiedQueriesResult(
                evaluation_state="no_verified_queries",
                benchmark_id=validation.benchmark.benchmark_id,
                schema_version=validation.benchmark.schema_version,
                evaluated_query_count=0,
                excluded_draft_query_count=len(validation.draft_queries),
                invalid_query_count=0,
                source_video_scope=source_scope,
            )
        if retriever is None:
            raise ValueError("retriever is required when verified queries are present")
        metrics = tuple(
            self._evaluate_query(query, retriever, resolved_ks) for query in verified
        )
        aggregates = self._aggregate_all(metrics, resolved_ks)
        comparisons, summaries = self._paired_comparisons(
            validation,
            metrics,
            resolved_ks,
        )
        source_scope = tuple(
            sorted({video_id for query in verified for video_id in query.source_scope})
        )
        metadata = dict(retriever.text_encoder.identifiers)
        return KISBenchmarkReport(
            evaluation_state="completed",
            benchmark_id=validation.benchmark.benchmark_id,
            schema_version=validation.benchmark.schema_version,
            evaluated_query_count=len(verified),
            excluded_draft_query_count=len(validation.draft_queries),
            invalid_query_count=0,
            source_video_scope=source_scope,
            model_identifier=str(metadata.get("model", "unknown")),
            model_metadata=metadata,
            device=str(metadata.get("device", "unknown")),
            retrieval_implementation="exact_chunked_numpy_cosine",
            canonical_unsuppressed=True,
            top_ks=resolved_ks,
            metric_definitions={
                "recall_at_k": (
                    "1.0 when at least one exact relevant (video_id, frame_id) pair is "
                    "retrieved in Top-K; otherwise 0.0"
                ),
                "ground_truth_coverage_at_k": (
                    "unique relevant frame labels retrieved in Top-K divided by the "
                    "number of relevant frame labels"
                ),
                "hit_count_at_k": "number of unique relevant frame labels in Top-K",
                "first_relevant_rank": "one-based rank of the first relevant frame",
                "reciprocal_rank": "1 / first_relevant_rank, or 0 when no hit",
                "mean_reciprocal_rank": (
                    "arithmetic mean of reciprocal_rank over all valid verified scored "
                    "queries, including zero for misses"
                ),
                "aggregate_recall_at_k": (
                    "arithmetic mean of binary per-query Recall@K over all valid "
                    "verified scored queries"
                ),
                "mean_first_relevant_rank": (
                    "arithmetic mean first relevant rank over scored queries with a "
                    "retrieved positive; queries without a hit are counted separately"
                ),
                "relevant_video_coverage_at_k": (
                    "relevant video IDs retrieved in Top-K divided by declared relevant "
                    "video IDs; aggregates include only queries declaring that field"
                ),
                "paired_delta": (
                    "English metric minus Vietnamese metric; lower first-rank delta is "
                    "better, higher Recall delta is better"
                ),
                "paired_outcome": (
                    "English wins Recall when its binary Recall is higher; it wins first "
                    "rank when it retrieves a hit and Vietnamese does not, or its rank "
                    "is numerically lower. Equal values, including two misses, are ties."
                ),
            },
            query_metrics=metrics,
            aggregates=aggregates,
            paired_comparisons=comparisons,
            paired_summaries=summaries,
            limitations=limitations or self.DEFAULT_LIMITATIONS,
        )

    def _evaluate_query(
        self,
        query: BenchmarkQuery,
        retriever: ExactNumpyRetriever,
        top_ks: tuple[int, ...],
    ) -> QueryMetrics:
        result = retriever.retrieve(
            KISQuery(query_id=query.query_id, text=query.text, top_k=max(top_ks))
        )
        relevant_pairs = {
            (label.video_id, label.frame_id) for label in query.relevant_frames
        }
        ranked_pairs = [
            (candidate.video_id, candidate.frame_id)
            for candidate in result.ranked_candidates
        ]
        relevant_ranks = [
            rank
            for rank, pair in enumerate(ranked_pairs, start=1)
            if pair in relevant_pairs
        ]
        first_rank = min(relevant_ranks) if relevant_ranks else None
        recall: list[tuple[int, float]] = []
        coverage: list[tuple[int, float]] = []
        hits: list[tuple[int, int]] = []
        video_coverage: list[tuple[int, float]] | None = (
            [] if query.relevant_video_ids else None
        )
        for cutoff in top_ks:
            retrieved_pairs = set(ranked_pairs[:cutoff])
            hit_count = len(relevant_pairs & retrieved_pairs)
            hits.append((cutoff, hit_count))
            recall.append((cutoff, 1.0 if hit_count else 0.0))
            coverage.append((cutoff, hit_count / len(relevant_pairs)))
            if video_coverage is not None:
                retrieved_videos = {video_id for video_id, _frame_id in ranked_pairs[:cutoff]}
                relevant_videos = set(query.relevant_video_ids)
                video_coverage.append(
                    (cutoff, len(relevant_videos & retrieved_videos) / len(relevant_videos))
                )
        return QueryMetrics(
            query_id=query.query_id,
            language=query.language.value,
            variant_type=query.variant_type.value,
            semantic_group_id=query.semantic_group_id,
            relevant_label_count=len(relevant_pairs),
            first_relevant_rank=first_rank,
            reciprocal_rank=0.0 if first_rank is None else 1.0 / first_rank,
            recall_at_k=tuple(recall),
            ground_truth_coverage_at_k=tuple(coverage),
            hit_count_at_k=tuple(hits),
            relevant_video_coverage_at_k=(
                tuple(video_coverage) if video_coverage is not None else None
            ),
        )

    def _aggregate_all(
        self,
        metrics: tuple[QueryMetrics, ...],
        top_ks: tuple[int, ...],
    ) -> tuple[AggregateMetrics, ...]:
        grouped: dict[tuple[str, str], list[QueryMetrics]] = defaultdict(list)
        grouped[("all", "all")].extend(metrics)
        for metric in metrics:
            grouped[("language", metric.language)].append(metric)
            grouped[("variant_type", metric.variant_type)].append(metric)
            grouped[("semantic_group_id", metric.semantic_group_id)].append(metric)
        return tuple(
            self._aggregate(group_type, group_value, values, top_ks)
            for (group_type, group_value), values in sorted(grouped.items())
        )

    def _aggregate(
        self,
        group_type: str,
        group_value: str,
        metrics: list[QueryMetrics],
        top_ks: tuple[int, ...],
    ) -> AggregateMetrics:
        first_ranks = [
            metric.first_relevant_rank
            for metric in metrics
            if metric.first_relevant_rank is not None
        ]
        recall_maps = [dict(metric.recall_at_k) for metric in metrics]
        coverage_maps = [dict(metric.ground_truth_coverage_at_k) for metric in metrics]
        hit_maps = [dict(metric.hit_count_at_k) for metric in metrics]
        video_maps = [
            dict(metric.relevant_video_coverage_at_k)
            for metric in metrics
            if metric.relevant_video_coverage_at_k is not None
        ]
        return AggregateMetrics(
            group_type=group_type,
            group_value=group_value,
            query_count=len(metrics),
            mean_reciprocal_rank=fmean(metric.reciprocal_rank for metric in metrics),
            mean_first_relevant_rank=fmean(first_ranks) if first_ranks else None,
            queries_without_relevant_hit=len(metrics) - len(first_ranks),
            mean_recall_at_k=tuple(
                (cutoff, fmean(values[cutoff] for values in recall_maps))
                for cutoff in top_ks
            ),
            mean_ground_truth_coverage_at_k=tuple(
                (cutoff, fmean(values[cutoff] for values in coverage_maps))
                for cutoff in top_ks
            ),
            mean_hit_count_at_k=tuple(
                (cutoff, fmean(values[cutoff] for values in hit_maps))
                for cutoff in top_ks
            ),
            mean_relevant_video_coverage_at_k=(
                tuple(
                    (cutoff, fmean(values[cutoff] for values in video_maps))
                    for cutoff in top_ks
                )
                if video_maps
                else None
            ),
        )

    def _paired_comparisons(
        self,
        validation: BenchmarkValidationResult,
        metrics: tuple[QueryMetrics, ...],
        top_ks: tuple[int, ...],
    ) -> tuple[tuple[PairedComparison, ...], tuple[PairedSummary, ...]]:
        metric_by_query = {metric.query_id: metric for metric in metrics}
        groups: dict[str, list[BenchmarkQuery]] = defaultdict(list)
        assert validation.benchmark is not None
        for query in validation.benchmark.queries:
            groups[query.semantic_group_id].append(query)
        comparisons: list[PairedComparison] = []
        comparison_types = (
            VariantType.ENGLISH_TRANSLATION,
            VariantType.ENGLISH_EXPANSION,
        )
        for group_id, queries in sorted(groups.items()):
            vietnamese = [
                query for query in queries if query.variant_type is VariantType.VIETNAMESE_DIRECT
            ]
            for comparison_type in comparison_types:
                english = [
                    query for query in queries if query.variant_type is comparison_type
                ]
                comparisons.append(
                    self._compare_pair(
                        group_id,
                        comparison_type,
                        vietnamese,
                        english,
                        metric_by_query,
                        top_ks,
                    )
                )
        summaries = tuple(
            self._paired_summary(comparison_type, comparisons, top_ks)
            for comparison_type in comparison_types
        )
        return tuple(comparisons), summaries

    def _compare_pair(
        self,
        group_id: str,
        comparison_type: VariantType,
        vietnamese: list[BenchmarkQuery],
        english: list[BenchmarkQuery],
        metric_by_query: dict[str, QueryMetrics],
        top_ks: tuple[int, ...],
    ) -> PairedComparison:
        vi_query = vietnamese[0] if len(vietnamese) == 1 else None
        en_query = english[0] if len(english) == 1 else None
        vi_metric = metric_by_query.get(vi_query.query_id) if vi_query else None
        en_metric = metric_by_query.get(en_query.query_id) if en_query else None
        if len(vietnamese) != 1:
            status = "MISSING_OR_AMBIGUOUS_VIETNAMESE_VARIANT"
        elif len(english) != 1:
            status = "MISSING_OR_AMBIGUOUS_ENGLISH_VARIANT"
        elif vi_metric is None:
            status = "VIETNAMESE_VARIANT_UNVERIFIED"
        elif en_metric is None:
            status = "ENGLISH_VARIANT_UNVERIFIED"
        else:
            status = "COMPARED"
        recall_delta: tuple[tuple[int, float], ...] = ()
        recall_outcomes: tuple[tuple[int, str], ...] = ()
        first_delta: int | None = None
        first_outcome: str | None = None
        if vi_metric is not None and en_metric is not None:
            vi_recall = dict(vi_metric.recall_at_k)
            en_recall = dict(en_metric.recall_at_k)
            recall_delta = tuple(
                (cutoff, en_recall[cutoff] - vi_recall[cutoff]) for cutoff in top_ks
            )
            recall_outcomes = tuple(
                (
                    cutoff,
                    "win"
                    if delta > 0
                    else "loss"
                    if delta < 0
                    else "tie",
                )
                for cutoff, delta in recall_delta
            )
            if (
                vi_metric.first_relevant_rank is not None
                and en_metric.first_relevant_rank is not None
            ):
                first_delta = (
                    en_metric.first_relevant_rank - vi_metric.first_relevant_rank
                )
            first_outcome = self._first_rank_outcome(
                vi_metric.first_relevant_rank,
                en_metric.first_relevant_rank,
            )
        return PairedComparison(
            semantic_group_id=group_id,
            comparison_variant_type=comparison_type.value,
            status=status,
            vietnamese_query_id=vi_query.query_id if vi_query else None,
            comparison_query_id=en_query.query_id if en_query else None,
            vietnamese_first_relevant_rank=(
                vi_metric.first_relevant_rank if vi_metric else None
            ),
            comparison_first_relevant_rank=(
                en_metric.first_relevant_rank if en_metric else None
            ),
            first_relevant_rank_delta_english_minus_vietnamese=first_delta,
            first_relevant_rank_outcome=first_outcome,
            recall_delta_at_k=recall_delta,
            recall_outcome_at_k=recall_outcomes,
        )

    def _paired_summary(
        self,
        comparison_type: VariantType,
        comparisons: list[PairedComparison],
        top_ks: tuple[int, ...],
    ) -> PairedSummary:
        relevant = [
            comparison
            for comparison in comparisons
            if comparison.comparison_variant_type == comparison_type.value
            and comparison.status == "COMPARED"
        ]
        counts: list[tuple[int, int, int, int]] = []
        for cutoff in top_ks:
            outcomes = [
                dict(comparison.recall_outcome_at_k)[cutoff]
                for comparison in relevant
            ]
            counts.append(
                (
                    cutoff,
                    outcomes.count("win"),
                    outcomes.count("tie"),
                    outcomes.count("loss"),
                )
            )
        return PairedSummary(
            comparison_variant_type=comparison_type.value,
            recall_counts_at_k=tuple(counts),
            first_relevant_rank_counts=(
                sum(item.first_relevant_rank_outcome == "win" for item in relevant),
                sum(item.first_relevant_rank_outcome == "tie" for item in relevant),
                sum(item.first_relevant_rank_outcome == "loss" for item in relevant),
            ),
        )

    @staticmethod
    def _first_rank_outcome(
        vietnamese_rank: int | None,
        english_rank: int | None,
    ) -> str:
        if vietnamese_rank is None and english_rank is None:
            return "tie"
        if vietnamese_rank is None:
            return "win"
        if english_rank is None:
            return "loss"
        if english_rank < vietnamese_rank:
            return "win"
        if english_rank > vietnamese_rank:
            return "loss"
        return "tie"

    @staticmethod
    def _validate_top_ks(top_ks: tuple[int, ...]) -> tuple[int, ...]:
        if not top_ks or any(type(cutoff) is not int or cutoff <= 0 for cutoff in top_ks):
            raise ValueError("top_ks must contain positive integers")
        resolved = tuple(sorted(set(top_ks)))
        if resolved[-1] > 100:
            raise ValueError("benchmark Top-K cannot exceed canonical Top-100")
        return resolved
