"""GT-isolated TR-A2-D0 nomination runtime and offline DEV evaluator."""

from __future__ import annotations

import json
import math
import statistics
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from system_tai.retrieval.multi_query import (
    QueryLanguage,
    QueryVariant,
    QueryVariantType,
)
from system_tai.retrieval.video_evidence import VideoRestrictedFeatureSearcher
from system_tai.trake.video_first import (
    TRAKEVideoFirstConfig,
    build_event_video_rankings,
    nominate_videos,
)

from .l21_150_schema import L21150Benchmark, L21150TRAKEQuery
from .l21_150_trake_translation import TRAKEDevTranslationSidecar

RUNTIME_ARTIFACT_ROLE = "TRAKE_NOMINATION_ONLY_RUNTIME"
OFFLINE_REPORT_ROLE = "OFFLINE_DIAGNOSTIC_GT_JOIN"
GT_USED_OFFLINE_ONLY = True
GT_USED_IN_RUNTIME = False


class TRAKENominationError(ValueError):
    """D0 input or artifact validation failed closed."""


class TRAKELanguagePolicy(StrEnum):
    VI_ONLY = "vi_only"
    VI_PLUS_EN_WEIGHTED_RRF = "vi_plus_en_weighted_rrf"
    EN_ONLY = "en_only"


class TextBatchEncoder(Protocol):
    def encode_texts(self, texts: Sequence[str]) -> Sequence[np.ndarray]: ...


@dataclass(frozen=True, slots=True)
class TRAKENominationEventInput:
    source_event_index: int
    source_vi: str
    translation_en: str | None


@dataclass(frozen=True, slots=True)
class TRAKENominationQueryInput:
    query_id: str
    events: tuple[TRAKENominationEventInput, ...]

    def __post_init__(self) -> None:
        if type(self.query_id) is not str or not self.query_id.strip():
            raise TRAKENominationError("query_id must be a non-empty string")
        if tuple(event.source_event_index for event in self.events) != (1, 2, 3):
            raise TRAKENominationError("TRAKE D0 inputs require event indexes 1, 2, 3")


def _policy(value: TRAKELanguagePolicy | str) -> TRAKELanguagePolicy:
    try:
        return TRAKELanguagePolicy(value)
    except ValueError as exc:
        raise TRAKENominationError(f"unsupported TRAKE language policy: {value}") from exc


def build_nomination_inputs(
    benchmark: L21150Benchmark,
    *,
    language_policy: TRAKELanguagePolicy | str,
    sidecar: TRAKEDevTranslationSidecar | None,
) -> tuple[TRAKENominationQueryInput, ...]:
    """Project benchmark data to a runtime-safe text-only DTO."""

    policy = _policy(language_policy)
    needs_english = policy is not TRAKELanguagePolicy.VI_ONLY
    if needs_english and sidecar is None:
        raise TRAKENominationError(f"{policy.value} requires the frozen DEV sidecar")
    if not needs_english and sidecar is not None:
        raise TRAKENominationError("VI_ONLY must not receive an English sidecar")
    translations = sidecar.translations if sidecar is not None else {}

    projected: list[TRAKENominationQueryInput] = []
    for query in benchmark.queries:
        if not isinstance(query, L21150TRAKEQuery) or query.split != "DEV":
            continue
        events = tuple(
            TRAKENominationEventInput(
                source_event_index=event.event_index,
                source_vi=event.description_vi,
                translation_en=(
                    translations.get((query.query_id, event.event_index))
                    if needs_english
                    else None
                ),
            )
            for event in query.events
        )
        if needs_english and any(event.translation_en is None for event in events):
            raise TRAKENominationError(
                f"missing English translation for DEV TRAKE query {query.query_id}"
            )
        projected.append(TRAKENominationQueryInput(query.query_id, events))
    if len(projected) != 38:
        raise TRAKENominationError("D0 requires exactly 38 DEV TRAKE queries")
    return tuple(projected)


def build_event_variants(
    query: TRAKENominationQueryInput,
    *,
    language_policy: TRAKELanguagePolicy | str,
) -> dict[int, tuple[QueryVariant, ...]]:
    policy = _policy(language_policy)
    by_event: dict[int, tuple[QueryVariant, ...]] = {}
    for event in query.events:
        zero_index = event.source_event_index - 1
        variants: list[QueryVariant] = []
        if policy in {
            TRAKELanguagePolicy.VI_ONLY,
            TRAKELanguagePolicy.VI_PLUS_EN_WEIGHTED_RRF,
        }:
            variants.append(
                QueryVariant(
                    variant_id=f"{query.query_id}::e{zero_index}::v1_vi",
                    text=event.source_vi,
                    language=QueryLanguage.VIETNAMESE,
                    variant_type=QueryVariantType.VIETNAMESE_DIRECT,
                    weight=1.0,
                )
            )
        if policy in {
            TRAKELanguagePolicy.EN_ONLY,
            TRAKELanguagePolicy.VI_PLUS_EN_WEIGHTED_RRF,
        }:
            if event.translation_en is None or not event.translation_en.strip():
                raise TRAKENominationError(
                    f"English translation missing for {query.query_id}/"
                    f"{event.source_event_index}"
                )
            variants.append(
                QueryVariant(
                    variant_id=f"{query.query_id}::e{zero_index}::v2_en",
                    text=event.translation_en,
                    language=QueryLanguage.ENGLISH,
                    variant_type=QueryVariantType.ENGLISH_TRANSLATION,
                    weight=1.0,
                )
            )
        by_event[zero_index] = tuple(variants)
    return by_event


def _event_evidence_payload(
    event_index: int,
    video_id: str,
    event_video_rankings: Mapping[int, Sequence[Any]],
) -> dict[str, Any]:
    evidence = next(
        item for item in event_video_rankings[event_index] if item.video_id == video_id
    )
    return {
        "source_event_index": event_index + 1,
        "event_video_rank": evidence.event_video_rank,
        "event_video_rrf_score": evidence.event_video_rrf_score,
        "best_variant_rank": evidence.best_variant_rank,
        "per_variant": [
            {
                "variant_id": variant.variant_id,
                "weight": variant.weight,
                "video_rank": variant.video_rank,
            }
            for variant in evidence.per_variant
        ],
    }


def run_nomination_query(
    query: TRAKENominationQueryInput,
    *,
    language_policy: TRAKELanguagePolicy | str,
    encoder: TextBatchEncoder,
    searcher: VideoRestrictedFeatureSearcher,
    rrf_constant: float = 60.0,
    event_video_nomination_depth: int = 100,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    """Run text encode through complete video nomination and stop before anchors."""

    policy = _policy(language_policy)
    if not math.isfinite(rrf_constant) or rrf_constant <= 0:
        raise TRAKENominationError("rrf_constant must be finite and positive")
    event_variants = build_event_variants(query, language_policy=policy)
    flattened = [
        (event_index, variant)
        for event_index in sorted(event_variants)
        for variant in event_variants[event_index]
    ]
    texts = [variant.text for _, variant in flattened]
    started = clock()
    encode_started = clock()
    vectors = tuple(encoder.encode_texts(texts))
    encode_seconds = clock() - encode_started
    if len(vectors) != len(flattened):
        raise TRAKENominationError("encoded vector count does not match variants")

    maxima_started = clock()
    maxima = searcher.search_video_maxima(
        query_ids=[variant.variant_id for _, variant in flattened],
        query_vectors=vectors,
    )
    maxima_seconds = clock() - maxima_started
    video_count = len(searcher.registry.stores)
    if video_count < 1 or video_count > 1000:
        raise TRAKENominationError("D0 requires a corpus containing 1..1000 videos")

    nomination_started = clock()
    event_video_rankings = build_event_video_rankings(
        event_variants=event_variants,
        maxima=maxima,
        rrf_constant=rrf_constant,
    )
    complete_ranking = nominate_videos(
        event_video_rankings=event_video_rankings,
        config=TRAKEVideoFirstConfig(
            enabled=True,
            selected_video_cap=video_count,
            event_video_nomination_depth=event_video_nomination_depth,
        ),
        rrf_constant=rrf_constant,
    )
    nomination_seconds = clock() - nomination_started
    if len(complete_ranking) != video_count:
        raise TRAKENominationError("nomination ranking does not cover the full corpus")

    return {
        "query_id": query.query_id,
        "language_policy": policy.value,
        "event_variants": [
            {
                "source_event_index": event_index + 1,
                "variant_id": variant.variant_id,
                "text": variant.text,
                "language": variant.language.value,
                "variant_type": variant.variant_type.value,
                "weight": variant.weight,
            }
            for event_index, variant in flattened
        ],
        "video_count": video_count,
        "nomination_ranking": [
            {
                "rank": rank,
                "video_id": nomination.video_id,
                "coverage_count": nomination.coverage_count,
                "worst_event_rank": nomination.worst_event_rank,
                "reciprocal_event_rank_sum": nomination.reciprocal_event_rank_sum,
                "best_event_rank": nomination.best_event_rank,
                "per_event": [
                    _event_evidence_payload(
                        event_index, nomination.video_id, event_video_rankings
                    )
                    for event_index in sorted(event_video_rankings)
                ],
            }
            for rank, nomination in enumerate(complete_ranking, start=1)
        ],
        "physical_rows_scored": maxima.physical_rows_scored,
        "store_scan_count": maxima.video_store_scan_count,
        "timings": {
            "text_encode_seconds": encode_seconds,
            "video_maxima_seconds": maxima_seconds,
            "event_fusion_and_nomination_seconds": nomination_seconds,
            "total_seconds": clock() - started,
        },
        "retrieval_feedback_used": False,
    }


def run_nomination_only(
    queries: Sequence[TRAKENominationQueryInput],
    *,
    benchmark_id: str,
    experiment_id: str,
    git_sha: str | None,
    corpus_fingerprint: str | None,
    model_identity: Mapping[str, Any] | None,
    language_policy: TRAKELanguagePolicy | str,
    translation_sidecar_sha256: str | None,
    translation_status: str | None,
    encoder: TextBatchEncoder,
    searcher: VideoRestrictedFeatureSearcher,
    rrf_constant: float = 60.0,
    event_video_nomination_depth: int = 100,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    policy = _policy(language_policy)
    started = clock()
    query_reports = [
        run_nomination_query(
            query,
            language_policy=policy,
            encoder=encoder,
            searcher=searcher,
            rrf_constant=rrf_constant,
            event_video_nomination_depth=event_video_nomination_depth,
            clock=clock,
        )
        for query in queries
    ]
    artifact = {
        "schema_version": 1,
        "artifact_role": RUNTIME_ARTIFACT_ROLE,
        "benchmark_id": benchmark_id,
        "experiment_id": experiment_id,
        "git_sha": git_sha,
        "corpus_fingerprint": corpus_fingerprint,
        "language_policy": policy.value,
        "translation_sidecar_sha256": translation_sidecar_sha256,
        "translation_status": translation_status,
        "retrieval_feedback_used": False,
        "query_count": len(query_reports),
        "video_count": len(searcher.registry.stores),
        "model_identity": dict(model_identity or {}),
        "rrf_constant": rrf_constant,
        "event_video_nomination_depth": event_video_nomination_depth,
        "physical_rows_scored": sum(
            report["physical_rows_scored"] for report in query_reports
        ),
        "store_scan_count": sum(report["store_scan_count"] for report in query_reports),
        "runtime_duration_seconds": clock() - started,
        "queries": query_reports,
    }
    assert_runtime_gt_isolated(artifact)
    return artifact


FORBIDDEN_RUNTIME_KEYS = {
    "target_video_id",
    "gt_intervals",
    "gt_frame_centers",
    "proposed_interval",
    "proposed_frame_center",
    "accepted_answers",
    "canonical_answer",
}


def assert_runtime_gt_isolated(value: Any) -> None:
    if isinstance(value, Mapping):
        forbidden = FORBIDDEN_RUNTIME_KEYS & set(value)
        if forbidden:
            raise TRAKENominationError(
                f"runtime artifact contains forbidden GT fields: {sorted(forbidden)}"
            )
        for nested in value.values():
            assert_runtime_gt_isolated(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            assert_runtime_gt_isolated(nested)


def write_json_document(path: Path, payload: Mapping[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise TRAKENominationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_nomination_artifact(path: Path) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        raise TRAKENominationError("nomination artifact must not contain a UTF-8 BOM")
    try:
        payload = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicate_pairs,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TRAKENominationError(f"invalid nomination artifact: {exc}") from exc
    if type(payload) is not dict or payload.get("artifact_role") != RUNTIME_ARTIFACT_ROLE:
        raise TRAKENominationError("invalid nomination artifact role")
    if payload.get("retrieval_feedback_used") is not False:
        raise TRAKENominationError("retrieval_feedback_used must be false")
    assert_runtime_gt_isolated(payload)
    queries = payload.get("queries")
    if type(queries) is not list or len(queries) != payload.get("query_count"):
        raise TRAKENominationError("query_count does not match nomination queries")
    for query in queries:
        if type(query) is not dict:
            raise TRAKENominationError("nomination query must be an object")
        ranking = query.get("nomination_ranking")
        if type(ranking) is not list or len(ranking) != payload.get("video_count"):
            raise TRAKENominationError("nomination ranking must cover every video")
        ranks = [record.get("rank") for record in ranking]
        if ranks != list(range(1, len(ranking) + 1)):
            raise TRAKENominationError("nomination ranks must be contiguous and 1-based")
        video_ids = [record.get("video_id") for record in ranking]
        if any(type(video_id) is not str or not video_id for video_id in video_ids):
            raise TRAKENominationError("nomination video_id must be non-empty")
        if len(video_ids) != len(set(video_ids)):
            raise TRAKENominationError("nomination ranking contains duplicate videos")
    return payload


def _nearest_rank_percentile(values: Sequence[int], percentile: float) -> int:
    if not values:
        raise TRAKENominationError("target ranks must not be empty")
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def evaluate_nomination_artifact(
    artifact: Mapping[str, Any],
    benchmark: L21150Benchmark,
) -> dict[str, Any]:
    """Join DEV target videos only after runtime ranking has been completed."""

    assert_runtime_gt_isolated(artifact)
    dev_queries = [
        query
        for query in benchmark.queries
        if isinstance(query, L21150TRAKEQuery) and query.split == "DEV"
    ]
    artifact_queries = artifact.get("queries")
    if type(artifact_queries) is not list:
        raise TRAKENominationError("artifact queries must be an array")
    by_id = {record.get("query_id"): record for record in artifact_queries}
    expected_ids = [query.query_id for query in dev_queries]
    if list(by_id) != expected_ids:
        raise TRAKENominationError("artifact queries must exactly match DEV TRAKE order")

    query_reports: list[dict[str, Any]] = []
    target_ranks: list[int] = []
    all_events_at_100 = 0
    for query in dev_queries:
        runtime_query = by_id[query.query_id]
        target_record = next(
            (
                record
                for record in runtime_query["nomination_ranking"]
                if record["video_id"] == query.video_id
            ),
            None,
        )
        if target_record is None:
            raise TRAKENominationError(
                f"target video absent from complete ranking for {query.query_id}"
            )
        target_rank = int(target_record["rank"])
        event_ranks = [
            int(event["event_video_rank"]) for event in target_record["per_event"]
        ]
        target_ranks.append(target_rank)
        if all(rank <= 100 for rank in event_ranks):
            all_events_at_100 += 1
        query_reports.append(
            {
                "query_id": query.query_id,
                "target_video_id": query.video_id,
                "target_nomination_rank": target_rank,
                "per_event_target_video_ranks": event_ranks,
            }
        )

    buckets = {
        "rank_1_32": sum(rank <= 32 for rank in target_ranks),
        "rank_33_50": sum(33 <= rank <= 50 for rank in target_ranks),
        "rank_51_100": sum(51 <= rank <= 100 for rank in target_ranks),
        "rank_over_100": sum(rank > 100 for rank in target_ranks),
    }
    count = len(target_ranks)
    return {
        "schema_version": 1,
        "artifact_role": OFFLINE_REPORT_ROLE,
        "benchmark_id": benchmark.benchmark_id,
        "runtime_experiment_id": artifact.get("experiment_id"),
        "language_policy": artifact.get("language_policy"),
        "GT_USED_OFFLINE_ONLY": GT_USED_OFFLINE_ONLY,
        "GT_USED_IN_RUNTIME": GT_USED_IN_RUNTIME,
        "query_count": count,
        "recall_at_32": buckets["rank_1_32"] / count,
        "recall_at_50": (
            buckets["rank_1_32"] + buckets["rank_33_50"]
        )
        / count,
        "recall_at_100": (
            buckets["rank_1_32"]
            + buckets["rank_33_50"]
            + buckets["rank_51_100"]
        )
        / count,
        "target_rank_buckets": buckets,
        "additional_target_videos_by_cap50": buckets["rank_33_50"],
        "additional_target_videos_by_cap100": (
            buckets["rank_33_50"] + buckets["rank_51_100"]
        ),
        "median_target_rank": statistics.median(target_ranks),
        "p75_target_rank_nearest_rank": _nearest_rank_percentile(target_ranks, 0.75),
        "p90_target_rank_nearest_rank": _nearest_rank_percentile(target_ranks, 0.90),
        "worst_target_rank": max(target_ranks),
        "all_event_video_ranks_at_most_100_count": all_events_at_100,
        "query_reports": query_reports,
    }


def _cap_decision(report: Mapping[str, Any]) -> str:
    if report["additional_target_videos_by_cap50"] > 0:
        return "CAP50_HAS_MATERIAL_OPPORTUNITY"
    if report["additional_target_videos_by_cap100"] > 0:
        return "CAP100_HAS_MATERIAL_OPPORTUNITY"
    if report["target_rank_buckets"]["rank_over_100"] > 0:
        return "RANKING_WEAK_BEYOND_CAP100"
    return "CAP32_SUFFICIENT"


def compare_nomination_reports(
    reports: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    required = {policy.value for policy in TRAKELanguagePolicy}
    if set(reports) != required:
        raise TRAKENominationError(
            f"comparison requires exactly {sorted(required)}"
        )
    for name, report in reports.items():
        if report.get("artifact_role") != OFFLINE_REPORT_ROLE:
            raise TRAKENominationError(f"{name} is not an offline D0 report")
        if report.get("language_policy") != name:
            raise TRAKENominationError(f"language policy mismatch for {name}")

    metrics = ("recall_at_32", "recall_at_50", "recall_at_100")
    best_arms = {
        metric: sorted(
            name
            for name, report in reports.items()
            if report[metric] == max(item[metric] for item in reports.values())
        )
        for metric in metrics
    }
    metric_vectors = {
        name: tuple(float(report[metric]) for metric in metrics)
        for name, report in reports.items()
    }
    dominant: list[str] = []
    for name, vector in metric_vectors.items():
        if all(
            all(left >= right for left, right in zip(vector, other_vector))
            and any(left > right for left, right in zip(vector, other_vector))
            for other_name, other_vector in metric_vectors.items()
            if other_name != name
        ):
            dominant.append(name)
    if len(dominant) == 1:
        language_label = {
            TRAKELanguagePolicy.EN_ONLY.value: "LANGUAGE_SIGNAL_EN_ONLY",
            TRAKELanguagePolicy.VI_PLUS_EN_WEIGHTED_RRF.value: (
                "LANGUAGE_SIGNAL_VI_EN"
            ),
            TRAKELanguagePolicy.VI_ONLY.value: "LANGUAGE_SIGNAL_VI",
        }[dominant[0]]
    else:
        language_label = "NO_CLEAR_LANGUAGE_SIGNAL"

    vi_by_id = {
        item["query_id"]: item["target_nomination_rank"]
        for item in reports[TRAKELanguagePolicy.VI_ONLY.value]["query_reports"]
    }
    per_query_deltas = []
    for query_id in vi_by_id:
        row: dict[str, Any] = {
            "query_id": query_id,
            "vi_only_target_rank": vi_by_id[query_id],
        }
        for policy in (
            TRAKELanguagePolicy.VI_PLUS_EN_WEIGHTED_RRF.value,
            TRAKELanguagePolicy.EN_ONLY.value,
        ):
            current = next(
                item["target_nomination_rank"]
                for item in reports[policy]["query_reports"]
                if item["query_id"] == query_id
            )
            row[f"{policy}_target_rank"] = current
            row[f"{policy}_minus_vi_rank_delta"] = current - vi_by_id[query_id]
        per_query_deltas.append(row)

    return {
        "schema_version": 1,
        "artifact_role": "TR_A2_D0_THREE_ARM_COMPARISON",
        "GT_USED_OFFLINE_ONLY": True,
        "GT_USED_IN_RUNTIME": False,
        "rank_delta_definition": "arm_target_rank_minus_vi_only_target_rank; negative is better",
        "language_decision_criterion": (
            "signal only when one arm weakly dominates both other arms at "
            "Recall@32/50/100 and is strictly better on at least one metric"
        ),
        "cap_decision_criterion": (
            "any additional target in ranks 33-50 selects CAP50; otherwise any in "
            "51-100 selects CAP100; otherwise >100 indicates weak ranking"
        ),
        "best_arms": best_arms,
        "language_decision": language_label,
        "arms": {
            name: {
                "recall_at_32": report["recall_at_32"],
                "recall_at_50": report["recall_at_50"],
                "recall_at_100": report["recall_at_100"],
                "target_rank_buckets": report["target_rank_buckets"],
                "median_target_rank": report["median_target_rank"],
                "p75_target_rank_nearest_rank": report[
                    "p75_target_rank_nearest_rank"
                ],
                "p90_target_rank_nearest_rank": report[
                    "p90_target_rank_nearest_rank"
                ],
                "worst_target_rank": report["worst_target_rank"],
                "cap_decision": _cap_decision(report),
            }
            for name, report in sorted(reports.items())
        },
        "per_query_rank_deltas": per_query_deltas,
    }
