"""Registry-aware validation for human-authored KIS benchmark labels."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from system_tai.evaluation.benchmark_schema import (
    AnnotationStatus,
    BenchmarkIssue,
    BenchmarkQuery,
    KISBenchmark,
    VariantType,
    load_benchmark,
)
from system_tai.features.btc_clip_store import FeatureStoreRegistry


@dataclass(frozen=True, slots=True)
class BenchmarkValidationResult:
    valid: bool
    benchmark: KISBenchmark | None
    errors: tuple[BenchmarkIssue, ...]
    warnings: tuple[BenchmarkIssue, ...]
    verified_queries: tuple[BenchmarkQuery, ...]
    draft_queries: tuple[BenchmarkQuery, ...]
    validation_scope_queries: tuple[BenchmarkQuery, ...]

    @property
    def invalid_query_count(self) -> int:
        query_ids = {issue.query_id for issue in self.errors if issue.query_id}
        if query_ids:
            return len(query_ids)
        return 1 if self.errors else 0


class BenchmarkValidator:
    def validate_file(
        self,
        benchmark_path: Path,
        registry: FeatureStoreRegistry,
        *,
        include_drafts: bool = False,
    ) -> BenchmarkValidationResult:
        parsed = load_benchmark(benchmark_path)
        if not parsed.valid:
            return BenchmarkValidationResult(
                valid=False,
                benchmark=None,
                errors=parsed.errors,
                warnings=(),
                verified_queries=(),
                draft_queries=(),
                validation_scope_queries=(),
            )
        assert parsed.benchmark is not None
        return self.validate(
            parsed.benchmark,
            registry,
            include_drafts=include_drafts,
        )

    def validate(
        self,
        benchmark: KISBenchmark,
        registry: FeatureStoreRegistry,
        *,
        include_drafts: bool = False,
    ) -> BenchmarkValidationResult:
        errors: list[BenchmarkIssue] = []
        warnings: list[BenchmarkIssue] = []
        seen_query_ids: set[str] = set()
        groups: dict[str, list[BenchmarkQuery]] = defaultdict(list)
        verified: list[BenchmarkQuery] = []
        drafts: list[BenchmarkQuery] = []

        for query in benchmark.queries:
            if query.query_id in seen_query_ids:
                errors.append(
                    self._issue(
                        "DUPLICATE_QUERY_ID",
                        f"query_id is duplicated: {query.query_id}",
                        query,
                        "query_id",
                    )
                )
            seen_query_ids.add(query.query_id)
            groups[query.semantic_group_id].append(query)
            if query.annotation_status is AnnotationStatus.VERIFIED:
                verified.append(query)
            else:
                drafts.append(query)

            expected_language = (
                "vi" if query.variant_type is VariantType.VIETNAMESE_DIRECT else "en"
            )
            if query.language.value != expected_language:
                errors.append(
                    self._issue(
                        "LANGUAGE_VARIANT_MISMATCH",
                        f"{query.variant_type.value} requires language={expected_language}",
                        query,
                        "language",
                    )
                )
            if len(set(query.source_scope)) != len(query.source_scope):
                errors.append(
                    self._issue(
                        "DUPLICATE_SOURCE_VIDEO",
                        "source_scope contains duplicate video IDs",
                        query,
                        "source_scope",
                    )
                )
            for video_id in query.source_scope:
                if not self._video_exists(registry, video_id):
                    errors.append(
                        self._issue(
                            "UNKNOWN_SOURCE_VIDEO",
                            f"source_scope video is absent from registry: {video_id}",
                            query,
                            "source_scope",
                        )
                    )

            seen_pairs: set[tuple[str, int]] = set()
            for label in query.relevant_frames:
                pair = (label.video_id, label.frame_id)
                if pair in seen_pairs:
                    errors.append(
                        self._issue(
                            "DUPLICATE_RELEVANT_FRAME",
                            f"duplicate relevant frame: {pair}",
                            query,
                            "relevant_frames",
                        )
                    )
                seen_pairs.add(pair)
                if not self._video_exists(registry, label.video_id):
                    errors.append(
                        self._issue(
                            "UNKNOWN_RELEVANT_VIDEO",
                            f"relevant video is absent from registry: {label.video_id}",
                            query,
                            "relevant_frames",
                        )
                    )
                elif not registry.contains(label.video_id, label.frame_id):
                    errors.append(
                        self._issue(
                            "RELEVANT_FRAME_NOT_MAPPED",
                            "relevant frame_id does not exist in the video's mapping CSV: "
                            f"{label.video_id}/{label.frame_id}",
                            query,
                            "relevant_frames",
                        )
                    )
            if len(set(query.relevant_video_ids)) != len(query.relevant_video_ids):
                errors.append(
                    self._issue(
                        "DUPLICATE_RELEVANT_VIDEO",
                        "relevant_video_ids contains duplicates",
                        query,
                        "relevant_video_ids",
                    )
                )
            for video_id in query.relevant_video_ids:
                if not self._video_exists(registry, video_id):
                    errors.append(
                        self._issue(
                            "UNKNOWN_RELEVANT_VIDEO",
                            f"relevant video is absent from registry: {video_id}",
                            query,
                            "relevant_video_ids",
                        )
                    )
            if (
                query.annotation_status is AnnotationStatus.VERIFIED
                and not query.relevant_frames
            ):
                errors.append(
                    self._issue(
                        "VERIFIED_QUERY_WITHOUT_POSITIVES",
                        "verified queries require at least one relevant frame",
                        query,
                        "relevant_frames",
                    )
                )

        for group_id, variants in groups.items():
            self._validate_group(group_id, variants, errors)

        if drafts and not include_drafts:
            warnings.append(
                BenchmarkIssue(
                    code="DRAFTS_EXCLUDED_FROM_SCORING",
                    message=f"{len(drafts)} draft queries are excluded from scoring",
                )
            )
        validation_scope = tuple(verified + drafts) if include_drafts else tuple(verified)
        return BenchmarkValidationResult(
            valid=not errors,
            benchmark=benchmark,
            errors=tuple(errors),
            warnings=tuple(warnings),
            verified_queries=tuple(verified),
            draft_queries=tuple(drafts),
            validation_scope_queries=validation_scope,
        )

    def _validate_group(
        self,
        group_id: str,
        variants: list[BenchmarkQuery],
        errors: list[BenchmarkIssue],
    ) -> None:
        verified_variants = [
            query
            for query in variants
            if query.annotation_status is AnnotationStatus.VERIFIED
        ]
        variant_counts = Counter(query.variant_type for query in verified_variants)
        for variant_type, count in sorted(
            variant_counts.items(), key=lambda item: item[0].value
        ):
            if count > 1:
                errors.append(
                    self._issue(
                        "DUPLICATE_GROUP_VARIANT",
                        (
                            f"semantic group {group_id} contains {count} "
                            f"{variant_type.value} variants"
                        ),
                        next(
                            query
                            for query in verified_variants
                            if query.variant_type is variant_type
                        ),
                        "variant_type",
                    )
                )
        if not verified_variants:
            return
        reference = verified_variants[0]
        for variant in verified_variants[1:]:
            if set(variant.source_scope) != set(reference.source_scope):
                errors.append(
                    self._issue(
                        "INCOMPARABLE_SOURCE_SCOPE",
                        f"semantic group {group_id} variants have different source_scope",
                        variant,
                        "source_scope",
                    )
                )
            reference_frames = {
                (label.video_id, label.frame_id) for label in reference.relevant_frames
            }
            variant_frames = {
                (label.video_id, label.frame_id) for label in variant.relevant_frames
            }
            if reference_frames and variant_frames and reference_frames != variant_frames:
                errors.append(
                    self._issue(
                        "INCOMPARABLE_RELEVANT_FRAMES",
                        f"semantic group {group_id} variants have different positive frames",
                        variant,
                        "relevant_frames",
                    )
                )
            reference_videos = set(reference.relevant_video_ids)
            variant_videos = set(variant.relevant_video_ids)
            if reference_videos and variant_videos and reference_videos != variant_videos:
                errors.append(
                    self._issue(
                        "INCOMPARABLE_RELEVANT_VIDEOS",
                        f"semantic group {group_id} variants have different relevant videos",
                        variant,
                        "relevant_video_ids",
                    )
                )

    @staticmethod
    def _video_exists(registry: FeatureStoreRegistry, video_id: str) -> bool:
        try:
            registry.get(video_id)
        except KeyError:
            return False
        return True

    @staticmethod
    def _issue(
        code: str,
        message: str,
        query: BenchmarkQuery,
        field: str,
    ) -> BenchmarkIssue:
        return BenchmarkIssue(
            code=code,
            message=message,
            query_id=query.query_id,
            field=field,
        )
