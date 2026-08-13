"""GT-isolated QA-D0 language nomination runtime and offline DEV evaluator."""

from __future__ import annotations

import dataclasses
import json
import statistics
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from system_tai.qa.grounding import (
    QAVideoConditionedEvidenceConfig,
    QAVideoNomination,
    nominate_qa_videos,
)
from system_tai.retrieval.multi_query import (
    QueryLanguage,
    QueryVariant,
    QueryVariantType,
)
from system_tai.retrieval.video_evidence import VideoRestrictedFeatureSearcher

from .l21_150_qa_translation import QADevTranslationSidecar
from .l21_150_schema import L21150Benchmark, L21150QAQuery

RUNTIME_ARTIFACT_ROLE = "QA_NOMINATION_ONLY_RUNTIME"
REPORT_ROLE = "QA_D0_TARGET_VIDEO_NOMINATION_DIAGNOSTIC"
BENCHMARK_ROLE = "DIAGNOSTIC_DEVELOPMENT"
EXPECTED_DEV_QUERY_COUNT = 38
TARGET_RANK_DEPTHS = (1, 5, 20, 32, 50, 100)
DEFAULT_QA_NOMINATION_CONFIG = QAVideoConditionedEvidenceConfig(enabled=True)


class QANominationError(ValueError):
    """QA-D0 input, runtime, or offline evaluation failed closed."""


class QALanguagePolicy(StrEnum):
    VI_ONLY = "vi_only"
    VI_PLUS_EN = "vi_plus_en"
    EN_ONLY = "en_only"


class TextBatchEncoder(Protocol):
    def encode_texts(self, texts: Sequence[str]) -> Sequence[np.ndarray]: ...


@dataclass(frozen=True, slots=True)
class QANominationQueryInput:
    query_id: str
    question_vi: str
    question_en: str | None

    def __post_init__(self) -> None:
        if type(self.query_id) is not str or not self.query_id.strip():
            raise QANominationError("query_id must be a non-empty string")
        if type(self.question_vi) is not str or not self.question_vi.strip():
            raise QANominationError("question_vi must be a non-empty string")
        if self.question_en is not None and (
            type(self.question_en) is not str or not self.question_en.strip()
        ):
            raise QANominationError("question_en must be null or a non-empty string")


@dataclass(frozen=True, slots=True)
class QANominationRuntimeResult:
    query_id: str
    language_policy: QALanguagePolicy
    variants: tuple[QueryVariant, ...]
    full_ranking: tuple[QAVideoNomination, ...]
    capped_ranking: tuple[QAVideoNomination, ...]
    physical_rows_scored: int
    store_scan_count: int
    timings: Mapping[str, float]


def _policy(value: QALanguagePolicy | str) -> QALanguagePolicy:
    try:
        return QALanguagePolicy(value)
    except ValueError as exc:
        raise QANominationError(f"unsupported QA language policy: {value}") from exc


def ensure_dev_only_scope(split: str) -> None:
    if split.casefold() != "dev":
        raise QANominationError("QA-D0 is restricted to QA DEV; HOLDOUT is forbidden")


def build_nomination_inputs(
    benchmark: L21150Benchmark,
    *,
    language_policy: QALanguagePolicy | str,
    sidecar: QADevTranslationSidecar | None,
) -> tuple[QANominationQueryInput, ...]:
    """Project DEV QA to a target-free localization-text DTO."""

    policy = _policy(language_policy)
    needs_english = policy is not QALanguagePolicy.VI_ONLY
    if needs_english and sidecar is None:
        raise QANominationError(f"{policy.value} requires the frozen QA DEV sidecar")
    translations = sidecar.translations if sidecar is not None else {}

    projected: list[QANominationQueryInput] = []
    for query in benchmark.queries:
        if not isinstance(query, L21150QAQuery) or query.split != "DEV":
            continue
        question_en = translations.get(query.query_id) if needs_english else None
        if needs_english and question_en is None:
            raise QANominationError(
                f"missing English translation for DEV QA query {query.query_id}"
            )
        projected.append(
            QANominationQueryInput(
                query_id=query.query_id,
                question_vi=query.question_vi,
                question_en=question_en,
            )
        )
    if len(projected) != EXPECTED_DEV_QUERY_COUNT:
        raise QANominationError(
            f"QA-D0 requires exactly {EXPECTED_DEV_QUERY_COUNT} DEV QA queries"
        )
    return tuple(projected)


def build_localization_variants(
    query: QANominationQueryInput,
    *,
    language_policy: QALanguagePolicy | str,
) -> tuple[QueryVariant, ...]:
    policy = _policy(language_policy)
    variants: list[QueryVariant] = []
    if policy in {QALanguagePolicy.VI_ONLY, QALanguagePolicy.VI_PLUS_EN}:
        variants.append(
            QueryVariant(
                variant_id=f"{query.query_id}::qa_localization_vi",
                text=query.question_vi,
                language=QueryLanguage.VIETNAMESE,
                variant_type=QueryVariantType.VIETNAMESE_DIRECT,
                weight=1.0,
            )
        )
    if policy in {QALanguagePolicy.VI_PLUS_EN, QALanguagePolicy.EN_ONLY}:
        if query.question_en is None or not query.question_en.strip():
            raise QANominationError(
                f"English translation missing for DEV QA query {query.query_id}"
            )
        variants.append(
            QueryVariant(
                variant_id=f"{query.query_id}::qa_localization_en",
                text=query.question_en,
                language=QueryLanguage.ENGLISH,
                variant_type=QueryVariantType.ENGLISH_TRANSLATION,
                weight=1.0,
            )
        )
    return tuple(variants)


def run_nomination_query(
    query: QANominationQueryInput,
    *,
    language_policy: QALanguagePolicy | str,
    encoder: TextBatchEncoder,
    searcher: VideoRestrictedFeatureSearcher,
    clock: Callable[[], float] = time.perf_counter,
) -> QANominationRuntimeResult:
    """Run full-corpus video maxima and QA-A1 nomination, then stop."""

    policy = _policy(language_policy)
    variants = build_localization_variants(query, language_policy=policy)
    started = clock()
    encode_started = clock()
    vectors = tuple(encoder.encode_texts([variant.text for variant in variants]))
    encode_seconds = clock() - encode_started
    if len(vectors) != len(variants):
        raise QANominationError("encoded vector count does not match localization variants")

    maxima_started = clock()
    maxima = searcher.search_video_maxima(
        query_ids=[variant.variant_id for variant in variants],
        query_vectors=vectors,
    )
    maxima_seconds = clock() - maxima_started
    video_count = len(searcher.registry.stores)
    if video_count < 1:
        raise QANominationError("QA-D0 requires a non-empty feature registry")

    nomination_started = clock()
    full_ranking = nominate_qa_videos(
        variants=variants,
        maxima=maxima,
        config=dataclasses.replace(
            DEFAULT_QA_NOMINATION_CONFIG,
            selected_video_cap=video_count,
        ),
    )
    capped_ranking = nominate_qa_videos(
        variants=variants,
        maxima=maxima,
        config=DEFAULT_QA_NOMINATION_CONFIG,
    )
    nomination_seconds = clock() - nomination_started
    if len(full_ranking) != video_count:
        raise QANominationError("QA nomination ranking does not cover the full corpus")
    if capped_ranking != full_ranking[: DEFAULT_QA_NOMINATION_CONFIG.selected_video_cap]:
        raise QANominationError("QA nomination cap is not the prefix of full ranking")

    return QANominationRuntimeResult(
        query_id=query.query_id,
        language_policy=policy,
        variants=variants,
        full_ranking=full_ranking,
        capped_ranking=capped_ranking,
        physical_rows_scored=maxima.physical_rows_scored,
        store_scan_count=maxima.video_store_scan_count,
        timings={
            "text_encode_seconds": encode_seconds,
            "video_maxima_seconds": maxima_seconds,
            "qa_nomination_seconds": nomination_seconds,
            "total_seconds": clock() - started,
        },
    )


def run_nomination_runtime(
    queries: Sequence[QANominationQueryInput],
    *,
    language_policy: QALanguagePolicy | str,
    encoder: TextBatchEncoder,
    searcher: VideoRestrictedFeatureSearcher,
    clock: Callable[[], float] = time.perf_counter,
) -> tuple[QANominationRuntimeResult, ...]:
    policy = _policy(language_policy)
    return tuple(
        run_nomination_query(
            query,
            language_policy=policy,
            encoder=encoder,
            searcher=searcher,
            clock=clock,
        )
        for query in queries
    )


FORBIDDEN_RUNTIME_KEYS = {
    "target_video_id",
    "video_id",
    "canonical_answer",
    "accepted_answers",
    "source_answer",
    "proposed_interval",
    "proposed_frame_center",
    "reference_timestamp",
    "frame_gt",
}


def assert_runtime_input_gt_isolated(value: Any) -> None:
    if isinstance(value, QANominationQueryInput):
        fields = set(value.__dataclass_fields__)
        forbidden = FORBIDDEN_RUNTIME_KEYS & fields
        if forbidden:
            raise QANominationError(
                f"runtime input contains forbidden fields: {sorted(forbidden)}"
            )
        return
    if isinstance(value, Mapping):
        forbidden = FORBIDDEN_RUNTIME_KEYS & set(value)
        if forbidden:
            raise QANominationError(
                f"runtime input contains forbidden fields: {sorted(forbidden)}"
            )
        for nested in value.values():
            assert_runtime_input_gt_isolated(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            assert_runtime_input_gt_isolated(nested)


def build_dev_target_video_ids(benchmark: L21150Benchmark) -> dict[str, str]:
    """Build the offline-only query-to-target join after retrieval has completed."""

    targets = {
        query.query_id: query.video_id
        for query in benchmark.queries
        if isinstance(query, L21150QAQuery) and query.split == "DEV"
    }
    if len(targets) != EXPECTED_DEV_QUERY_COUNT:
        raise QANominationError("offline target join requires exactly 38 DEV QA queries")
    return targets


def _nomination_record(item: QAVideoNomination) -> dict[str, Any]:
    return {
        "rank": item.nomination_rank,
        "video_id": item.video_id,
        "video_rrf_score": item.video_rrf_score,
        "best_individual_variant_rank": item.best_individual_variant_rank,
        "per_variant": [
            {
                "variant_id": rank.variant_id,
                "weight": rank.weight,
                "video_rank": rank.video_rank,
            }
            for rank in item.per_variant
        ],
    }


def evaluate_nomination_results(
    results: Sequence[QANominationRuntimeResult],
    *,
    target_video_ids: Mapping[str, str],
    benchmark_id: str,
    policy: QALanguagePolicy | str,
    translation_sidecar_sha256: str,
    manifest_sha256: str,
    corpus_fingerprint: str | None,
    git_sha: str | None,
    model_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Join target videos offline and compute target-video rank diagnostics."""

    resolved_policy = _policy(policy)
    resolved_results = tuple(results)
    query_ids = [result.query_id for result in resolved_results]
    if len(query_ids) != EXPECTED_DEV_QUERY_COUNT or len(set(query_ids)) != len(query_ids):
        raise QANominationError("QA-D0 evaluation requires 38 unique DEV query results")
    if set(query_ids) != set(target_video_ids):
        raise QANominationError("offline target join IDs do not match runtime result IDs")
    if any(result.language_policy is not resolved_policy for result in resolved_results):
        raise QANominationError("runtime result language policy mismatch")

    query_reports: list[dict[str, Any]] = []
    found_ranks: list[int] = []
    reciprocal_ranks: list[float] = []
    for result in resolved_results:
        target_video_id = target_video_ids[result.query_id]
        target = next(
            (item for item in result.full_ranking if item.video_id == target_video_id),
            None,
        )
        target_rank = target.nomination_rank if target is not None else None
        if target_rank is not None:
            found_ranks.append(target_rank)
            reciprocal_ranks.append(1.0 / target_rank)
        else:
            reciprocal_ranks.append(0.0)
        nominated = target_rank is not None and target_rank <= len(result.capped_ranking)
        query_reports.append(
            {
                "query_id": result.query_id,
                "policy": resolved_policy.value,
                "target_video_rank": target_rank,
                "target_video_nominated": nominated,
                "target_video_nomination_rank": target_rank if nominated else None,
                "localization_variant_count": len(result.variants),
                "source_variant_ids": [variant.variant_id for variant in result.variants],
                "source_variant_languages": [
                    variant.language.value for variant in result.variants
                ],
                "source_variant_text_provenance": [
                    (
                        "benchmark.question_vi"
                        if variant.language is QueryLanguage.VIETNAMESE
                        else "frozen_qa_dev_translation_sidecar.question_en"
                    )
                    for variant in result.variants
                ],
                "translation_sidecar_sha256": translation_sidecar_sha256,
                "elapsed_timings": dict(result.timings),
                "top100_video_ranking": [
                    _nomination_record(item) for item in result.full_ranking[:100]
                ],
            }
        )

    count = len(query_reports)
    missing = count - len(found_ranks)
    report = {
        "schema_version": 1,
        "artifact_role": REPORT_ROLE,
        "benchmark_role": BENCHMARK_ROLE,
        "benchmark_id": benchmark_id,
        "official_ground_truth": False,
        "semantic_accuracy_claim": False,
        "question_as_localization_fallback": True,
        "gt_used_in_runtime": False,
        "holdout_used": False,
        "language_policy": resolved_policy.value,
        "translation_sidecar_sha256": translation_sidecar_sha256,
        "manifest_sha256": manifest_sha256,
        "corpus_fingerprint": corpus_fingerprint,
        "git_sha": git_sha,
        "clip_device": model_identity.get("device"),
        "model_identity": dict(model_identity),
        "query_count": count,
        "full_corpus_video_count": (
            len(resolved_results[0].full_ranking) if resolved_results else 0
        ),
        "video_store_scan_count": sum(
            result.store_scan_count for result in resolved_results
        ),
        "qa_nomination_cap": DEFAULT_QA_NOMINATION_CONFIG.selected_video_cap,
        "target_video_recall_at_1": sum(
            row["target_video_rank"] is not None and row["target_video_rank"] <= 1
            for row in query_reports
        )
        / count,
        "target_video_recall_at_5": sum(
            row["target_video_rank"] is not None and row["target_video_rank"] <= 5
            for row in query_reports
        )
        / count,
        "target_video_recall_at_20": sum(
            row["target_video_rank"] is not None and row["target_video_rank"] <= 20
            for row in query_reports
        )
        / count,
        "target_video_recall_at_32": sum(
            row["target_video_rank"] is not None and row["target_video_rank"] <= 32
            for row in query_reports
        )
        / count,
        "target_video_recall_at_50": sum(
            row["target_video_rank"] is not None and row["target_video_rank"] <= 50
            for row in query_reports
        )
        / count,
        "target_video_recall_at_100": sum(
            row["target_video_rank"] is not None and row["target_video_rank"] <= 100
            for row in query_reports
        )
        / count,
        "mean_reciprocal_rank": sum(reciprocal_ranks) / count,
        "median_target_video_rank": (
            float(statistics.median(found_ranks)) if missing == 0 else None
        ),
        "worst_target_video_rank": max(found_ranks) if missing == 0 else None,
        "target_video_miss_count_at_full_depth": missing,
        "top32_nomination_coverage_count": sum(
            bool(row["target_video_nominated"]) for row in query_reports
        ),
        "queries": query_reports,
    }
    return report


def write_json_document(path: Path, payload: Mapping[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
