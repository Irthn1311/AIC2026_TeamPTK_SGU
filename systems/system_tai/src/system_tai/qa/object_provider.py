"""Artifact-backed open-label object/entity answer provider."""

from __future__ import annotations

import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from system_tai.evidence.object_artifacts import ObjectArtifactIndex, ObjectFrameEvidence
from system_tai.preliminary.schemas import QAPrediction

from .models import QAEvidenceCandidate, QAResult
from .question_types import QuestionType

GLOBAL_SUPPORT_RANKING = "global-support"
QUERY_CONDITIONED_FRAME_RANKING = "query-conditioned-frame"
_RANKING_POLICIES = frozenset(
    {GLOBAL_SUPPORT_RANKING, QUERY_CONDITIONED_FRAME_RANKING}
)


class ObjectTextEncoder(Protocol):
    """Minimal shared-encoder surface used by query-conditioned object ranking."""

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray: ...


def normalize_object_label(label: str) -> str:
    """Normalize an artifact label without translating or inventing content."""

    normalized = unicodedata.normalize("NFKC", label)
    return " ".join(normalized.split()).casefold()


@dataclass(frozen=True, slots=True)
class ObjectAnswerProviderConfig:
    enabled: bool = False
    allow_candidate_anchor_fallback: bool = True
    telemetry_candidate_limit: int = 10
    ranking_policy: str = GLOBAL_SUPPORT_RANKING

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise TypeError("enabled must be a boolean")
        if type(self.allow_candidate_anchor_fallback) is not bool:
            raise TypeError("allow_candidate_anchor_fallback must be a boolean")
        if not 1 <= self.telemetry_candidate_limit <= 100:
            raise ValueError("telemetry_candidate_limit must be in [1, 100]")
        if self.ranking_policy not in _RANKING_POLICIES:
            raise ValueError(
                "ranking_policy must be 'global-support' or "
                "'query-conditioned-frame'"
            )


@dataclass(frozen=True, slots=True)
class _FrameSupport:
    video_id: str
    frame_id: int
    evidence_rank: int
    confidence: float
    source_label: str
    source_artifacts: tuple[str, ...]
    lookup_kind: str
    requested_frame_id: int


@dataclass(slots=True)
class _LabelAggregate:
    normalized_label: str
    source_labels: set[str] = field(default_factory=set)
    supports: dict[tuple[str, int], _FrameSupport] = field(default_factory=dict)

    @property
    def best_support(self) -> _FrameSupport:
        return min(
            self.supports.values(),
            key=lambda item: (
                item.evidence_rank,
                -item.confidence,
                item.video_id,
                item.frame_id,
            ),
        )

    @property
    def best_confidence(self) -> float:
        return max(item.confidence for item in self.supports.values())


class ObjectEntityAnswerProvider:
    """Aggregate actual OpenImages entity labels across the QA-A1 evidence bank."""

    def __init__(
        self,
        *,
        index: ObjectArtifactIndex,
        config: ObjectAnswerProviderConfig,
        text_encoder: ObjectTextEncoder | None = None,
    ) -> None:
        self.index = index
        self.config = config
        self.text_encoder = text_encoder
        if (
            self.config.ranking_policy == QUERY_CONDITIONED_FRAME_RANKING
            and self.text_encoder is None
        ):
            raise ValueError(
                "query-conditioned-frame ranking requires a shared text encoder"
            )

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def supports(self, question_type: QuestionType) -> bool:
        return self.enabled and question_type is QuestionType.OBJECT_ENTITY

    def answer(
        self,
        *,
        query_id: str,
        question_type: QuestionType,
        evidence: Sequence[QAEvidenceCandidate],
        output_top_k: int,
        question_text: str | None = None,
        warnings: list[str] | None = None,
    ) -> tuple[QAResult, dict[str, Any]]:
        if not self.supports(question_type):
            raise ValueError(f"object provider does not support {question_type.value}")
        if not 1 <= output_top_k <= 100:
            raise ValueError("output_top_k must be in [1, 100]")
        if (
            self.config.ranking_policy == QUERY_CONDITIONED_FRAME_RANKING
            and (question_text is None or not question_text.strip())
        ):
            raise ValueError(
                "query-conditioned-frame ranking requires non-empty question_text"
            )
        emitted_warnings = warnings if warnings is not None else []
        aggregates: dict[str, _LabelAggregate] = {}
        lookup_count = 0
        exact_hits = 0
        anchor_fallbacks = 0
        detection_count = 0
        evidence_diagnostics: list[dict[str, Any]] = []

        for evidence_candidate in sorted(evidence, key=lambda item: item.rank):
            lookup_count += 1
            frame_evidence = self.index.lookup(
                evidence_candidate.video_id,
                evidence_candidate.frame_id,
            )
            if frame_evidence is not None:
                exact_hits += 1
            candidate_frame_id = evidence_candidate.provenance.get(
                "candidate_frame_id"
            )
            has_distinct_candidate_anchor = (
                type(candidate_frame_id) is int
                and candidate_frame_id >= 0
                and candidate_frame_id != evidence_candidate.frame_id
            )
            if (
                frame_evidence is None
                and self.config.allow_candidate_anchor_fallback
                and has_distinct_candidate_anchor
            ):
                lookup_count += 1
                anchor = self.index.lookup(
                    evidence_candidate.video_id,
                    candidate_frame_id,
                )
                if anchor is not None:
                    anchor_fallbacks += 1
                    frame_evidence = ObjectFrameEvidence(
                        video_id=anchor.video_id,
                        requested_frame_id=evidence_candidate.frame_id,
                        object_source_frame_id=anchor.object_source_frame_id,
                        frame_distance=abs(
                            anchor.object_source_frame_id - evidence_candidate.frame_id
                        ),
                        lookup_kind="AUTHORITATIVE_CANDIDATE_ANCHOR",
                        detections=anchor.detections,
                    )
            if frame_evidence is None:
                evidence_diagnostics.append(
                    {
                        "evidence_rank": evidence_candidate.rank,
                        "video_id": evidence_candidate.video_id,
                        "requested_frame_id": evidence_candidate.frame_id,
                        "object_source_frame_id": None,
                        "frame_distance": None,
                        "lookup_kind": "MISS",
                        "detection_count": 0,
                    }
                )
                continue

            detection_count += len(frame_evidence.detections)
            evidence_diagnostics.append(
                {
                    "evidence_rank": evidence_candidate.rank,
                    "video_id": evidence_candidate.video_id,
                    "requested_frame_id": frame_evidence.requested_frame_id,
                    "object_source_frame_id": frame_evidence.object_source_frame_id,
                    "frame_distance": frame_evidence.frame_distance,
                    "lookup_kind": frame_evidence.lookup_kind,
                    "detection_count": len(frame_evidence.detections),
                }
            )
            per_label: dict[str, tuple[float, str, set[str]]] = {}
            for detection in frame_evidence.detections:
                normalized = normalize_object_label(detection.label)
                if not normalized:
                    continue
                existing = per_label.get(normalized)
                if existing is None:
                    per_label[normalized] = (
                        detection.confidence,
                        detection.label,
                        {detection.source_artifact},
                    )
                else:
                    confidence, source_label, artifacts = existing
                    artifacts.add(detection.source_artifact)
                    if detection.confidence > confidence or (
                        detection.confidence == confidence
                        and detection.label.casefold() < source_label.casefold()
                    ):
                        confidence = detection.confidence
                        source_label = detection.label
                    per_label[normalized] = (confidence, source_label, artifacts)
            for normalized, (confidence, source_label, artifacts) in per_label.items():
                aggregate = aggregates.setdefault(
                    normalized, _LabelAggregate(normalized_label=normalized)
                )
                aggregate.source_labels.add(source_label)
                identity = (
                    frame_evidence.video_id,
                    frame_evidence.object_source_frame_id,
                )
                support = _FrameSupport(
                    video_id=frame_evidence.video_id,
                    frame_id=frame_evidence.object_source_frame_id,
                    evidence_rank=evidence_candidate.rank,
                    confidence=confidence,
                    source_label=source_label,
                    source_artifacts=tuple(sorted(artifacts)),
                    lookup_kind=frame_evidence.lookup_kind,
                    requested_frame_id=frame_evidence.requested_frame_id,
                )
                previous = aggregate.supports.get(identity)
                if previous is None or (
                    support.evidence_rank,
                    -support.confidence,
                    support.source_label.casefold(),
                ) < (
                    previous.evidence_rank,
                    -previous.confidence,
                    previous.source_label.casefold(),
                ):
                    aggregate.supports[identity] = support

        ranked = sorted(
            aggregates.values(),
            key=lambda item: (
                -len(item.supports),
                item.best_support.evidence_rank,
                -item.best_confidence,
                item.normalized_label,
            ),
        )
        prediction_candidates: list[dict[str, Any]] = []
        if self.config.ranking_policy == GLOBAL_SUPPORT_RANKING:
            for aggregate in ranked:
                best = aggregate.best_support
                prediction_candidates.append(
                    {
                        "video_id": best.video_id,
                        "frame_id": best.frame_id,
                        "answer": aggregate.normalized_label,
                        "evidence_rank": best.evidence_rank,
                        "frame_label_rank": None,
                        "query_relevance": None,
                        "confidence": best.confidence,
                    }
                )
        else:
            assert self.text_encoder is not None
            assert question_text is not None
            labels = tuple(sorted(aggregates))
            prompts = tuple(f"a photo of {label}" for label in labels)
            encoded = np.asarray(
                self.text_encoder.encode_texts((question_text, *prompts)),
                dtype=np.float32,
            )
            expected_shape = (len(labels) + 1, encoded.shape[1] if encoded.ndim == 2 else 0)
            if encoded.ndim != 2 or encoded.shape[0] != len(labels) + 1:
                raise ValueError(
                    "object label text encoder returned invalid shape: "
                    f"{encoded.shape}; expected {expected_shape}"
                )
            if encoded.shape[1] <= 0 or not np.isfinite(encoded).all():
                raise ValueError(
                    "object label text encoder must return finite non-empty vectors"
                )
            norms = np.linalg.norm(encoded, axis=1)
            if not np.isfinite(norms).all() or np.any(norms <= 0):
                raise ValueError(
                    "object label text encoder returned a zero-norm vector"
                )
            normalized_vectors = encoded / norms[:, None]
            relevance = normalized_vectors[1:] @ normalized_vectors[0]
            relevance_by_label = {
                label: float(score) for label, score in zip(labels, relevance)
            }

            by_frame: dict[tuple[str, int], list[tuple[_LabelAggregate, _FrameSupport]]] = {}
            for aggregate in aggregates.values():
                for support in aggregate.supports.values():
                    by_frame.setdefault((support.video_id, support.frame_id), []).append(
                        (aggregate, support)
                    )
            for frame_identity in sorted(by_frame):
                frame_labels = sorted(
                    by_frame[frame_identity],
                    key=lambda item: (
                        -relevance_by_label[item[0].normalized_label],
                        -item[1].confidence,
                        item[0].normalized_label,
                    ),
                )
                for frame_label_rank, (aggregate, support) in enumerate(
                    frame_labels, start=1
                ):
                    prediction_candidates.append(
                        {
                            "video_id": support.video_id,
                            "frame_id": support.frame_id,
                            "answer": aggregate.normalized_label,
                            "evidence_rank": support.evidence_rank,
                            "frame_label_rank": frame_label_rank,
                            "query_relevance": relevance_by_label[
                                aggregate.normalized_label
                            ],
                            "confidence": support.confidence,
                        }
                    )
            prediction_candidates.sort(
                key=lambda item: (
                    item["frame_label_rank"],
                    item["evidence_rank"],
                    -item["query_relevance"],
                    -item["confidence"],
                    item["video_id"],
                    item["frame_id"],
                    item["answer"],
                )
            )

        predictions = [
            QAPrediction(
                query_id=query_id,
                rank=index,
                video_id=item["video_id"],
                frame_id=item["frame_id"],
                answer=item["answer"],
            )
            for index, item in enumerate(
                prediction_candidates[:output_top_k], start=1
            )
        ]

        unsupported_reason = None
        if not predictions:
            unsupported_reason = "NO_OBJECT_ARTIFACT_EVIDENCE"
            emitted_warnings.append(
                "No mapped object artifact detections were found for the grounded evidence."
            )
        top_candidates = []
        for aggregate in ranked[: self.config.telemetry_candidate_limit]:
            best = aggregate.best_support
            top_candidates.append(
                {
                    "normalized_label": aggregate.normalized_label,
                    "source_labels": sorted(aggregate.source_labels, key=str.casefold),
                    "supporting_evidence_count": len(aggregate.supports),
                    "supporting_video_count": len(
                        {support.video_id for support in aggregate.supports.values()}
                    ),
                    "best_evidence_rank": best.evidence_rank,
                    "best_confidence": aggregate.best_confidence,
                    "source_frames": [
                        {
                            "video_id": support.video_id,
                            "frame_id": support.frame_id,
                            "requested_frame_id": support.requested_frame_id,
                            "lookup_kind": support.lookup_kind,
                            "source_artifacts": list(support.source_artifacts),
                        }
                        for support in sorted(
                            aggregate.supports.values(),
                            key=lambda item: (
                                item.evidence_rank,
                                item.video_id,
                                item.frame_id,
                            ),
                        )
                    ],
                }
            )
        telemetry = {
            "object_answer_ranking_policy": self.config.ranking_policy,
            "object_artifact_lookup_count": lookup_count,
            "exact_object_frame_hit_count": exact_hits,
            "nearest_object_frame_fallback_count": 0,
            "candidate_anchor_object_fallback_count": anchor_fallbacks,
            "object_detection_count": detection_count,
            "unique_object_label_count": len(aggregates),
            "object_answer_candidate_count": len(predictions),
            "top_object_candidates": top_candidates,
            "top_object_prediction_candidates": prediction_candidates[
                : self.config.telemetry_candidate_limit
            ],
            "object_evidence": evidence_diagnostics,
        }
        return (
            QAResult(
                query_id=query_id,
                question_type=question_type,
                predictions=predictions,
                unsupported_reason=unsupported_reason,
                warnings=emitted_warnings,
                diagnostics={
                    "confidence_level": "ARTIFACT_BACKED",
                    **telemetry,
                },
            ),
            telemetry,
        )
