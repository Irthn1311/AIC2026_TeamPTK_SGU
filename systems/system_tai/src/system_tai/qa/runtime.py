from __future__ import annotations

import dataclasses
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from system_tai.features.query_encoder import SharedOpenAIClipEncoder
from system_tai.preliminary.validation import validate_ranked_top100
from system_tai.refinement.engine import ExactFrameRefiner, QueryRefinementOutcome
from system_tai.refinement.models import (
    Phase3Candidate,
    RefinedCandidate,
    RefinementConfig,
    RefinementQuery,
    RefinementStatus,
)
from system_tai.refinement.video import (
    DecodeRequest,
    RawVideoRegistry,
    VideoDecoder,
)
from system_tai.retrieval.multi_query import WeightedRRFRetriever
from system_tai.retrieval.vector_search import ExactNumpyRetriever
from system_tai.retrieval.video_evidence import VideoRestrictedFeatureSearcher

from .answer_candidates import AnswerCandidateProvider, BaselineQuestionCandidateProvider
from .engine import QABaselineEngine
from .grounding import (
    KEYFRAME_ANCHOR,
    QA_KEYFRAME_EVIDENCE_BANK_V1,
    QA_MULTI_SEED_TEMPORAL_REFINEMENT_V1,
    QA_VIDEO_CONDITIONED_EVIDENCE_V1,
    RAW_REFINED,
    TEMPORAL_REFINED,
    QAVideoConditionedEvidenceConfig,
    build_qa_grounding_result,
    distill_qa_scene_prompt,
    nominate_qa_videos,
    nomination_diagnostics,
    select_primary_keyframe_anchors,
    select_temporal_seed_anchors,
)
from .models import QAEvidenceCandidate, QAQuery, QAResult
from .object_provider import ObjectEntityAnswerProvider
from .ocr_provider import OCRAnswerProvider
from .question_types import (
    QuestionClassification,
    QuestionType,
    classify_ocr_question,
    classify_question,
    classify_question_legacy,
)
from .visual_ontology import VisualOntologyAnswerCandidateProvider

LEGACY_PHASE_P0 = "LEGACY_PHASE_P0"
QA_A2_CAPABILITY_AWARE = "QA_A2_CAPABILITY_AWARE"
QA_A3_CAPABILITY_AWARE = "QA_A3_CAPABILITY_AWARE"


def _unattempted_refinement_record(candidate: Phase3Candidate) -> RefinedCandidate:
    """Represent a bounded evidence seed that was intentionally not raw-refined."""

    return RefinedCandidate(
        query_id=candidate.query_id,
        original_candidate_rank=candidate.rank,
        video_id=candidate.video_id,
        candidate_frame_id=candidate.frame_id,
        refined_frame_id=None,
        candidate_timestamp_seconds=None,
        refined_timestamp_seconds=None,
        fps=None,
        total_frame_count=None,
        window_start_frame=None,
        window_end_frame=None,
        coarse_frame_ids=(),
        fine_frame_ids=(),
        coarse_sample_count=0,
        fine_sample_count=0,
        decoded_frame_count=0,
        encoded_image_count=0,
        refinement_fusion_score=None,
        variant_hit_count=0,
        best_individual_rank=None,
        per_variant_provenance=(),
        decoder_backend=None,
        raw_video_path=None,
        status=RefinementStatus.NOT_REFINED,
        warnings=("temporal seed outside raw-refinement budget",),
        failure_reason=None,
        original_retrieval_provenance=candidate.retrieval_provenance,
        timings={},
    )


def _is_raw_refined_source(source_status: str) -> bool:
    return source_status in {RAW_REFINED, TEMPORAL_REFINED}


def classify_runtime_question(
    question: str,
    question_en: str | None,
    *,
    qa_a2_enabled: bool,
    qa_ocr_enabled: bool = False,
) -> tuple[QuestionClassification, str]:
    if qa_ocr_enabled:
        ocr_classification = classify_ocr_question(question, question_en)
        if ocr_classification is not None:
            return ocr_classification, QA_A3_CAPABILITY_AWARE
    if qa_a2_enabled:
        return classify_question(question, question_en), QA_A2_CAPABILITY_AWARE
    return classify_question_legacy(question, question_en), LEGACY_PHASE_P0


@dataclass
class QAPipelineTimings:
    total_seconds: float = 0.0
    text_encode_seconds: float = 0.0
    retrieval_seconds: float = 0.0
    fusion_seconds: float = 0.0
    refinement_seconds: float = 0.0
    evidence_decode_seconds: float = 0.0
    evidence_encode_seconds: float = 0.0
    answer_scoring_seconds: float = 0.0
    prompt_encode_seconds: float = 0.0
    ocr_decode_seconds: float = 0.0
    ocr_inference_seconds: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "total_seconds": self.total_seconds,
            "text_encode_seconds": self.text_encode_seconds,
            "retrieval_seconds": self.retrieval_seconds,
            "fusion_seconds": self.fusion_seconds,
            "refinement_seconds": self.refinement_seconds,
            "evidence_decode_seconds": self.evidence_decode_seconds,
            "evidence_encode_seconds": self.evidence_encode_seconds,
            "answer_scoring_seconds": self.answer_scoring_seconds,
            "prompt_encode_seconds": self.prompt_encode_seconds,
            "ocr_decode_seconds": self.ocr_decode_seconds,
            "ocr_inference_seconds": self.ocr_inference_seconds,
        }


class QARuntimePipeline:
    """Evidence-grounded Q&A runtime adapter reusing shared retrieval/refinement components."""

    def __init__(
        self,
        *,
        exact_retriever: ExactNumpyRetriever,
        weighted_rrf: WeightedRRFRetriever,
        refiner: ExactFrameRefiner,
        raw_video_registry: RawVideoRegistry,
        decoder: VideoDecoder,
        shared_encoder: SharedOpenAIClipEncoder,
        video_restricted_searcher: VideoRestrictedFeatureSearcher | None = None,
        video_conditioned_evidence_config: QAVideoConditionedEvidenceConfig | None = None,
        qa_engine: QABaselineEngine | None = None,
        candidate_provider: AnswerCandidateProvider | None = None,
        object_answer_provider: ObjectEntityAnswerProvider | None = None,
        ocr_answer_provider: OCRAnswerProvider | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.exact_retriever = exact_retriever
        self.weighted_rrf = weighted_rrf
        self.refiner = refiner
        self.raw_video_registry = raw_video_registry
        self.decoder = decoder
        self.shared_encoder = shared_encoder
        self.video_restricted_searcher = video_restricted_searcher
        self.video_conditioned_evidence_config = (
            video_conditioned_evidence_config or QAVideoConditionedEvidenceConfig()
        )
        self.candidate_provider = candidate_provider or BaselineQuestionCandidateProvider()
        self.qa_engine = qa_engine or QABaselineEngine(
            candidate_provider=self.candidate_provider
        )
        self.object_answer_provider = object_answer_provider
        self.ocr_answer_provider = ocr_answer_provider
        self.clock = clock

        self._prompt_cache: dict[str, np.ndarray] = {}

    def _answer_hypotheses(
        self,
        question_type: QuestionType,
        question_text: str,
    ) -> tuple[Any, ...]:
        query_aware = getattr(self.candidate_provider, "get_candidates_for_query", None)
        if callable(query_aware):
            return tuple(query_aware(question_type, question_text))
        return self.candidate_provider.get_candidates(question_type)

    def get_prompt_embeddings(
        self, prompts: Sequence[str]
    ) -> tuple[dict[str, np.ndarray], float]:
        t0 = self.clock()
        missing: list[str] = []
        for p in prompts:
            if p not in self._prompt_cache and p not in missing:
                missing.append(p)

        if missing:
            encoded_vecs = self.shared_encoder.encode_texts(missing)
            for p, vec in zip(missing, encoded_vecs):
                self._prompt_cache[p] = vec.astype(np.float32)

        prompt_dict = {p: self._prompt_cache[p] for p in prompts if p in self._prompt_cache}
        return prompt_dict, self.clock() - t0

    def process_qa_query(
        self,
        request: Any,  # QAQueryRequest
        refinement_config: RefinementConfig | None = None,
        rrf_constant: float = 60.0,
    ) -> tuple[QAResult, QAPipelineTimings, dict[str, Any]]:
        if refinement_config is None:
            refinement_config = RefinementConfig()
        t_start = self.clock()
        timings = QAPipelineTimings()
        diagnostics: dict[str, Any] = {
            "query_id": request.query_id,
            "request_id": request.request_id,
            "question_type": None,
            "question_classification_reason": None,
            "question_classifier_policy": None,
            "question_supported": None,
            "object_provider_enabled": bool(
                self.object_answer_provider is not None
                and self.object_answer_provider.enabled
            ),
            "object_artifact_lookup_count": 0,
            "exact_object_frame_hit_count": 0,
            "nearest_object_frame_fallback_count": 0,
            "candidate_anchor_object_fallback_count": 0,
            "object_detection_count": 0,
            "unique_object_label_count": 0,
            "object_answer_candidate_count": 0,
            "top_object_candidates": [],
            "ocr_provider_enabled": bool(
                self.ocr_answer_provider is not None
                and self.ocr_answer_provider.enabled
            ),
            "visual_ontology_provider_enabled": bool(
                isinstance(
                    self.candidate_provider,
                    VisualOntologyAnswerCandidateProvider,
                )
                and self.candidate_provider.enabled
            ),
            "visual_ontology_provider_identity": (
                dict(self.candidate_provider.identifiers)
                if isinstance(
                    self.candidate_provider,
                    VisualOntologyAnswerCandidateProvider,
                )
                and self.candidate_provider.enabled
                else None
            ),
            "visual_ontology_active_domains": [],
            "visual_ontology_candidate_count": 0,
            "ocr_backend_identity": (
                dict(self.ocr_answer_provider.identifiers)
                if self.ocr_answer_provider is not None
                and self.ocr_answer_provider.enabled
                else None
            ),
            "ocr_frames_requested": 0,
            "ocr_frames_processed": 0,
            "ocr_observation_count": 0,
            "ocr_nonempty_observation_count": 0,
            "ocr_unique_candidate_count": 0,
            "ocr_decode_seconds": 0.0,
            "ocr_inference_seconds": 0.0,
            "top_ocr_candidates": [],
            "qa_grounding_enabled": self.video_conditioned_evidence_config.enabled,
            "retrieval_candidate_count": 0,
            "refined_candidate_count": 0,
            "evidence_candidate_count": 0,
            "decoded_frame_count": 0,
            "encoded_image_count": 0,
            "fused_retrieval_candidates": [],
            "refined_candidates": [],
            "usable_evidence_candidates": [],
            "final_predictions": [],
            "warnings": [],
        }
        if self.video_conditioned_evidence_config.enabled:
            diagnostics.update(
                {
                    "question_supported_by_current_provider": None,
                    "question_capability_reason": None,
                    "qa_grounding_policy": QA_VIDEO_CONDITIONED_EVIDENCE_V1,
                    "qa_candidate_ordering_policy": (
                        self.video_conditioned_evidence_config.candidate_ordering_policy
                    ),
                    "qa_localization_policy": None,
                    "include_vi_variant": request.include_vi_variant,
                    "localization_variant_count": 0,
                    "localization_variant_languages": [],
                    "localization_variant_types": [],
                    "localization_text_provenance": [],
                    "answer_routing_question_language": "vi",
                    "full_corpus_video_count": 0,
                    "full_corpus_store_scan_count": 0,
                    "selected_video_count": 0,
                    "selected_video_ids": [],
                    "restricted_store_scan_count": 0,
                    "restricted_rows_scored": 0,
                    "grounding_candidate_count": 0,
                    "selected_video_evidence": [],
                    "grounding_candidates": [],
                    "qa_evidence_bank_policy": (
                        QA_KEYFRAME_EVIDENCE_BANK_V1
                        if self.video_conditioned_evidence_config.preserve_keyframe_evidence
                        else "LEGACY_REFINED_ONLY"
                    ),
                    "refinement_budget": request.refine_top_n,
                    "refinement_selected_count": 0,
                    "refinement_success_count": 0,
                    "keyframe_evidence_count": 0,
                    "raw_refined_evidence_count": 0,
                    "generic_evidence_bank_count": 0,
                    "provider_evidence_count": 0,
                    "refinement_selected_candidates": [],
                    "refinement_success_candidates": [],
                    "keyframe_evidence_candidates": [],
                    "raw_refined_evidence_candidates": [],
                    "provider_evidence_candidates": [],
                    "qa_temporal_refinement_policy": (
                        QA_MULTI_SEED_TEMPORAL_REFINEMENT_V1
                        if self.video_conditioned_evidence_config.temporal_refinement_enabled
                        else "DISABLED"
                    ),
                    "temporal_seed_candidate_count": 0,
                    "temporal_seed_candidates": [],
                    "temporal_refinement_video_count": 0,
                    "temporal_refinement_seed_count": 0,
                    "temporal_refinement_success_count": 0,
                    "temporal_refinement_failure_count": 0,
                    "temporal_refinement_fallback_count": 0,
                    "temporal_refined_evidence_count": 0,
                    "temporal_refined_evidence_candidates": [],
                    "temporal_evidence_candidates": [],
                    "temporal_merged_region_count": 0,
                    "temporal_decoded_frame_count": 0,
                    "temporal_encoded_image_count": 0,
                    "temporal_embedding_cache_hit_count": 0,
                    "temporal_embedding_cache_miss_count": 0,
                }
            )

        # Step 1: Question classification
        qa_a2_enabled = bool(
            self.object_answer_provider is not None
            and self.object_answer_provider.enabled
        ) or bool(
            isinstance(
                self.candidate_provider,
                VisualOntologyAnswerCandidateProvider,
            )
            and self.candidate_provider.enabled
        )
        qa_ocr_enabled = bool(
            self.ocr_answer_provider is not None
            and self.ocr_answer_provider.enabled
        )
        classification, classifier_policy = classify_runtime_question(
            request.question,
            request.question_en,
            qa_a2_enabled=qa_a2_enabled,
            qa_ocr_enabled=qa_ocr_enabled,
        )
        q_type = classification.question_type
        diagnostics["question_type"] = q_type.value
        diagnostics["question_classification_reason"] = classification.reason
        diagnostics["question_classifier_policy"] = classifier_policy
        diagnostics["question_supported"] = q_type != QuestionType.UNSUPPORTED

        answer_question_text = request.question_en or request.question
        answer_hypotheses = self._answer_hypotheses(q_type, answer_question_text)
        if isinstance(
            self.candidate_provider,
            VisualOntologyAnswerCandidateProvider,
        ) and self.candidate_provider.enabled:
            diagnostics["visual_ontology_active_domains"] = list(
                self.candidate_provider.active_domain_ids(
                    q_type,
                    answer_question_text,
                )
            )
            diagnostics["visual_ontology_candidate_count"] = len(answer_hypotheses)
        object_provider_supported = bool(
            self.video_conditioned_evidence_config.enabled
            and self.object_answer_provider is not None
            and self.object_answer_provider.supports(q_type)
        )
        ocr_provider_supported = bool(
            self.video_conditioned_evidence_config.enabled
            and self.ocr_answer_provider is not None
            and self.ocr_answer_provider.supports(q_type)
        )
        provider_supported = bool(
            answer_hypotheses or object_provider_supported or ocr_provider_supported
        )

        if (
            not provider_supported
            and not self.video_conditioned_evidence_config.enabled
        ):
            if q_type is QuestionType.OBJECT_COUNT:
                unsupported_reason = "UNSUPPORTED_OBJECT_COUNT_PROVIDER_MISSING"
            elif classification.reason.startswith("OCR_PATTERN_PROVIDER_MISSING"):
                unsupported_reason = "UNSUPPORTED_OCR_PROVIDER_MISSING"
            else:
                unsupported_reason = "UNSUPPORTED_NO_PROVIDER"
            msg = f"{unsupported_reason}; zero predictions generated without grounding."
            diagnostics["warnings"].append(msg)
            diagnostics["unsupported_reason"] = unsupported_reason
            timings.total_seconds = self.clock() - t_start
            result = QAResult(
                query_id=request.query_id,
                question_type=q_type,
                predictions=[],
                unsupported_reason=unsupported_reason,
                warnings=[msg],
            )
            return result, timings, diagnostics

        if self.video_conditioned_evidence_config.enabled:
            diagnostics["question_supported_by_current_provider"] = provider_supported
            if q_type == QuestionType.UNSUPPORTED:
                diagnostics["question_capability_reason"] = "QUESTION_PATTERN_UNSUPPORTED"
            elif not provider_supported:
                diagnostics["question_capability_reason"] = "SUPPORTED_PATTERN_NO_PROVIDER"

        # Step 2: Event-only retrieval variants
        variants = request.variants()
        if self.video_conditioned_evidence_config.enabled:
            diagnostics["qa_localization_policy"] = (
                "EN_ONLY"
                if not request.include_vi_variant
                else "VI_PLUS_EN"
                if request.event_description_en is not None
                else "LEGACY_VI"
            )
            diagnostics["localization_variant_count"] = len(variants)
            diagnostics["localization_variant_languages"] = [
                variant.language.value for variant in variants
            ]
            diagnostics["localization_variant_types"] = [
                variant.variant_type.value for variant in variants
            ]
            diagnostics["localization_text_provenance"] = [
                (
                    "explicit_event_description_en"
                    if variant.language.value == "en"
                    else "event_description"
                )
                for variant in variants
            ]

        t_text = self.clock()
        event_texts = [
            distill_qa_scene_prompt(v.text)
            if self.video_conditioned_evidence_config.enabled
            else v.text
            for v in variants
        ]
        event_vectors = self.shared_encoder.encode_texts(event_texts)
        timings.text_encode_seconds = self.clock() - t_text

        t_ret = self.clock()
        if self.video_conditioned_evidence_config.enabled:
            if self.video_restricted_searcher is None:
                raise ValueError(
                    "QA video-conditioned evidence requires a video-restricted searcher"
                )
            variant_ids = [variant.variant_id for variant in variants]
            maxima = self.video_restricted_searcher.search_video_maxima(
                query_ids=variant_ids,
                query_vectors=event_vectors,
            )
            nominations = nominate_qa_videos(
                variants=variants,
                maxima=maxima,
                config=self.video_conditioned_evidence_config,
            )
            selected_video_ids = [item.video_id for item in nominations]
            diagnostics["full_corpus_video_count"] = (
                len(maxima.rankings[variant_ids[0]]) if variant_ids else 0
            )
            diagnostics["full_corpus_store_scan_count"] = maxima.video_store_scan_count
            diagnostics["selected_video_count"] = len(selected_video_ids)
            diagnostics["selected_video_ids"] = selected_video_ids

            largest_store_rows = max(
                self.video_restricted_searcher.registry.get(video_id).descriptor.row_count
                for video_id in selected_video_ids
            )
            restricted = self.video_restricted_searcher.search_selected_videos(
                video_ids=selected_video_ids,
                query_ids=variant_ids,
                query_vectors=event_vectors,
                per_query_result_cap=largest_store_rows,
            )
            diagnostics["restricted_store_scan_count"] = restricted.video_store_scan_count
            diagnostics["restricted_rows_scored"] = restricted.physical_rows_scored
            timings.retrieval_seconds = self.clock() - t_ret

            t_fuse = self.clock()
            fused_result = build_qa_grounding_result(
                query_id=request.query_id,
                variants=variants,
                nominations=nominations,
                restricted=restricted,
                weighted_rrf=self.weighted_rrf,
                config=self.video_conditioned_evidence_config,
                output_top_k=request.output_top_k,
            )
            timings.fusion_seconds = self.clock() - t_fuse
            anchor_counts: dict[str, int] = {video_id: 0 for video_id in selected_video_ids}
            for candidate in fused_result.ranked_candidates:
                anchor_counts[candidate.video_id] += 1
            diagnostics["selected_video_evidence"] = nomination_diagnostics(
                nominations,
                anchor_counts=anchor_counts,
            )
            diagnostics["grounding_candidates"] = [
                {
                    "rank": candidate.rank,
                    "video_id": candidate.video_id,
                    "frame_id": candidate.frame_id,
                    "video_nomination_rank": (candidate.diagnostic_metadata or {}).get(
                        "video_nomination_rank"
                    ),
                    "local_anchor_rank": (candidate.diagnostic_metadata or {}).get(
                        "local_anchor_rank"
                    ),
                    "localization_score": (candidate.diagnostic_metadata or {}).get(
                        "localization_score"
                    ),
                    "localization_score_kind": (candidate.diagnostic_metadata or {}).get(
                        "localization_score_kind"
                    ),
                    "source_localization_variant_ids": (
                        candidate.diagnostic_metadata or {}
                    ).get("source_localization_variant_ids", []),
                }
                for candidate in fused_result.ranked_candidates
            ]
        else:
            rankings: dict[str, Any] = {}
            for variant, vector in zip(variants, event_vectors):
                rankings[variant.variant_id] = self.exact_retriever.search_vector(
                    query_id=f"{request.query_id}::{variant.variant_id}",
                    query_vector=vector,
                    top_k=request.top_k_per_variant,
                )
            timings.retrieval_seconds = self.clock() - t_ret

            t_fuse = self.clock()
            fused_result = self.weighted_rrf.fuse_rankings(
                query_id=request.query_id,
                variants=variants,
                rankings=rankings,
                output_top_k=request.output_top_k,
                rrf_constant=rrf_constant,
            )
            timings.fusion_seconds = self.clock() - t_fuse
        diagnostics["retrieval_candidate_count"] = len(fused_result.ranked_candidates)
        if self.video_conditioned_evidence_config.enabled:
            diagnostics["grounding_candidate_count"] = len(
                fused_result.ranked_candidates
            )
        diagnostics["fused_retrieval_candidates"] = [
            {
                "rank": candidate.rank,
                "video_id": candidate.video_id,
                "frame_id": candidate.frame_id,
            }
            for candidate in fused_result.ranked_candidates
        ]

        if not fused_result.ranked_candidates:
            timings.total_seconds = self.clock() - t_start
            result = QAResult(
                query_id=request.query_id,
                question_type=q_type,
                predictions=[],
                warnings=["Zero retrieval candidates found."],
            )
            return result, timings, diagnostics

        preserve_keyframes = (
            self.video_conditioned_evidence_config.enabled
            and self.video_conditioned_evidence_config.preserve_keyframe_evidence
        )
        temporal_refinement_enabled = (
            preserve_keyframes
            and self.video_conditioned_evidence_config.temporal_refinement_enabled
        )
        if preserve_keyframes and not provider_supported:
            if (
                temporal_refinement_enabled
                or self.video_conditioned_evidence_config.keyframe_evidence_anchors_per_video
                > 1
            ):
                anchors = select_temporal_seed_anchors(
                    fused_result.ranked_candidates,
                    anchors_per_video=(
                        self.video_conditioned_evidence_config.temporal_seed_anchors_per_video
                        if temporal_refinement_enabled
                        else self.video_conditioned_evidence_config
                        .keyframe_evidence_anchors_per_video
                    ),
                    video_cap=(
                        self.video_conditioned_evidence_config.keyframe_evidence_video_cap
                    ),
                )
            else:
                anchors = select_primary_keyframe_anchors(
                    fused_result.ranked_candidates,
                    video_cap=(
                        self.video_conditioned_evidence_config.keyframe_evidence_video_cap
                    ),
                )
            keyframe_records = [
                {
                    "rank": candidate.rank,
                    "video_id": candidate.video_id,
                    "frame_id": candidate.frame_id,
                    "video_nomination_rank": dict(
                        candidate.diagnostic_metadata or {}
                    ).get("video_nomination_rank"),
                    "local_anchor_rank": dict(
                        candidate.diagnostic_metadata or {}
                    ).get("local_anchor_rank"),
                }
                for candidate in anchors
            ]
            generic_records = [
                {
                    "query_id": request.query_id,
                    "rank": candidate.rank,
                    "video_id": candidate.video_id,
                    "candidate_frame_id": candidate.frame_id,
                    "frame_id": candidate.frame_id,
                    "evidence_frame_id": candidate.frame_id,
                    "video_nomination_rank": dict(
                        candidate.diagnostic_metadata or {}
                    ).get("video_nomination_rank"),
                    "local_anchor_rank": dict(
                        candidate.diagnostic_metadata or {}
                    ).get("local_anchor_rank"),
                    "evidence_source": KEYFRAME_ANCHOR,
                    "raw_refinement_attempted": False,
                    "raw_refinement_status": RefinementStatus.NOT_REFINED.value,
                    "raw_refinement_replaced_anchor": False,
                    "fallback_to_keyframe": False,
                    "localization_score": dict(
                        candidate.diagnostic_metadata or {}
                    ).get("localization_score"),
                    "source_localization_variant_ids": dict(
                        candidate.diagnostic_metadata or {}
                    ).get("source_localization_variant_ids", []),
                }
                for candidate in anchors
            ]
            diagnostics["keyframe_evidence_candidates"] = keyframe_records
            diagnostics["keyframe_evidence_count"] = len(keyframe_records)
            diagnostics["generic_evidence_bank_candidates"] = generic_records
            diagnostics["generic_evidence_bank_count"] = len(generic_records)
            diagnostics["evidence_candidate_count"] = len(generic_records)
            diagnostics["evidence_bank"] = generic_records
            if temporal_refinement_enabled:
                diagnostics["temporal_seed_candidate_count"] = len(anchors)
                diagnostics["temporal_seed_candidates"] = keyframe_records
                diagnostics["temporal_refinement_fallback_count"] = len(anchors)
                diagnostics["temporal_evidence_candidates"] = generic_records
            diagnostics["evidence"] = []
            unsupported_reason = "UNSUPPORTED_NO_PROVIDER"
            diagnostics["unsupported_reason"] = unsupported_reason
            msg = (
                "Evidence grounding completed, but no current answer provider supports "
                f"question type {q_type.value}; raw refinement and image decode skipped."
            )
            diagnostics["warnings"].append(msg)
            timings.total_seconds = self.clock() - t_start
            return (
                QAResult(
                    query_id=request.query_id,
                    question_type=q_type,
                    predictions=[],
                    unsupported_reason=unsupported_reason,
                    warnings=diagnostics["warnings"],
                    diagnostics={
                        "confidence_level": "UNSUPPORTED",
                        "grounding_completed": True,
                    },
                ),
                timings,
                diagnostics,
            )

        # Step 3: Exact Frame Refinement
        t_ref = self.clock()
        phase3_candidates_list: list[Phase3Candidate] = []
        for candidate in fused_result.ranked_candidates:
            metadata = dict(candidate.diagnostic_metadata or {})
            metadata.update(
                {
                    "fusion_score": candidate.score,
                    "variant_hit_count": metadata.get("variant_hit_count"),
                    "best_individual_rank": metadata.get("best_individual_rank"),
                    "clip_row_diagnostic": candidate.clip_row,
                    "keyframe_order_diagnostic": candidate.keyframe_order,
                }
            )
            phase3_candidates_list.append(
                Phase3Candidate(
                    query_id=request.query_id,
                    rank=candidate.rank,
                    video_id=candidate.video_id,
                    frame_id=candidate.frame_id,
                    retrieval_score=candidate.score,
                    retrieval_provenance=metadata,
                )
            )
        phase3_candidates = tuple(phase3_candidates_list)
        phase3_by_rank = {candidate.rank: candidate for candidate in phase3_candidates}
        temporal_evidence_anchors: tuple[Any, ...] = ()
        temporal_attempted_ranks: set[int] = set()
        if temporal_refinement_enabled:
            temporal_evidence_anchors = select_temporal_seed_anchors(
                fused_result.ranked_candidates,
                anchors_per_video=(
                    self.video_conditioned_evidence_config.temporal_seed_anchors_per_video
                ),
                video_cap=(
                    self.video_conditioned_evidence_config.keyframe_evidence_video_cap
                ),
            )
            selected_temporal_anchors = select_temporal_seed_anchors(
                fused_result.ranked_candidates,
                anchors_per_video=(
                    self.video_conditioned_evidence_config.temporal_seed_anchors_per_video
                ),
                video_cap=(
                    self.video_conditioned_evidence_config.temporal_refinement_video_cap
                ),
                total_seed_cap=(
                    self.video_conditioned_evidence_config.temporal_refinement_total_seed_cap
                ),
            )
            selected_phase3 = tuple(
                phase3_by_rank[candidate.rank] for candidate in selected_temporal_anchors
            )
            temporal_attempted_ranks = {
                candidate.rank for candidate in selected_temporal_anchors
            }
            selected_outcome = self.refiner.refine_selected_candidates(
                query_id=request.query_id,
                variants=variants,
                candidates=selected_phase3,
                config=refinement_config,
                precomputed_text_embeddings=event_vectors,
                frame_embedding_cache={},
            )
            refined_candidates = selected_outcome.candidates
            temporal_timings = dict(selected_outcome.timings)
            diagnostics["temporal_seed_candidate_count"] = len(
                temporal_evidence_anchors
            )
            diagnostics["temporal_seed_candidates"] = [
                {
                    "rank": candidate.rank,
                    "video_id": candidate.video_id,
                    "frame_id": candidate.frame_id,
                    "video_nomination_rank": dict(
                        candidate.diagnostic_metadata or {}
                    ).get("video_nomination_rank"),
                    "local_anchor_rank": dict(
                        candidate.diagnostic_metadata or {}
                    ).get("local_anchor_rank"),
                }
                for candidate in temporal_evidence_anchors
            ]
            diagnostics["temporal_refinement_video_count"] = len(
                {candidate.video_id for candidate in selected_temporal_anchors}
            )
            diagnostics["temporal_refinement_seed_count"] = len(
                selected_temporal_anchors
            )
            diagnostics["temporal_merged_region_count"] = int(
                temporal_timings.get("merged_temporal_region_count", 0)
            )
            diagnostics["temporal_decoded_frame_count"] = int(
                temporal_timings.get("decoded_frame_count", 0)
            )
            diagnostics["temporal_encoded_image_count"] = int(
                temporal_timings.get("encoded_image_count", 0)
            )
            diagnostics["temporal_embedding_cache_hit_count"] = int(
                temporal_timings.get("frame_embedding_cache_hit_count", 0)
            )
            diagnostics["temporal_embedding_cache_miss_count"] = int(
                temporal_timings.get("frame_embedding_cache_miss_count", 0)
            )
        else:
            ref_query = RefinementQuery(request.query_id, variants, phase3_candidates)
            exec_ref_config = refinement_config
            if exec_ref_config.top_candidates_to_refine != request.refine_top_n:
                exec_ref_config = dataclasses.replace(
                    exec_ref_config,
                    top_candidates_to_refine=request.refine_top_n,
                )
            ref_outcome: QueryRefinementOutcome = self.refiner.refine_query(
                ref_query,
                exec_ref_config,
                precomputed_text_embeddings=event_vectors,
            )
            refined_candidates = ref_outcome.candidates
        timings.refinement_seconds = self.clock() - t_ref
        diagnostics["refined_candidate_count"] = len(refined_candidates)
        diagnostics["refined_candidates"] = [
            {
                "original_rank": candidate.original_candidate_rank,
                "video_id": candidate.video_id,
                "candidate_frame_id": candidate.candidate_frame_id,
                "refined_frame_id": candidate.refined_frame_id,
                "status": candidate.status.value,
            }
            for candidate in refined_candidates
        ]

        refinement_selected = (
            list(refined_candidates)
            if temporal_refinement_enabled
            else [
                candidate
                for candidate in refined_candidates
                if candidate.original_candidate_rank <= request.refine_top_n
            ]
        )
        refinement_success = [
            candidate
            for candidate in refinement_selected
            if candidate.status is RefinementStatus.REFINED
            and candidate.refined_frame_id is not None
        ]
        if self.video_conditioned_evidence_config.enabled:
            diagnostics["refinement_selected_count"] = len(refinement_selected)
            diagnostics["refinement_success_count"] = len(refinement_success)
            diagnostics["refinement_selected_candidates"] = [
                {
                    "rank": candidate.original_candidate_rank,
                    "video_id": candidate.video_id,
                    "frame_id": candidate.candidate_frame_id,
                    "status": candidate.status.value,
                }
                for candidate in refinement_selected
            ]
            diagnostics["refinement_success_candidates"] = [
                {
                    "rank": candidate.original_candidate_rank,
                    "video_id": candidate.video_id,
                    "candidate_frame_id": candidate.candidate_frame_id,
                    "frame_id": candidate.refined_frame_id,
                    "status": candidate.status.value,
                }
                for candidate in refinement_success
            ]
            if temporal_refinement_enabled:
                diagnostics["temporal_refinement_success_count"] = len(
                    refinement_success
                )
                diagnostics["temporal_refinement_failure_count"] = (
                    len(refinement_selected) - len(refinement_success)
                )

        # Step 4: Build provider-neutral evidence before provider routing.
        evidence_bank: list[tuple[QAEvidenceCandidate, RefinedCandidate]] = []
        evidence_records: list[dict[str, Any]] = []

        def store_evidence_records() -> None:
            diagnostics["evidence"] = evidence_records

        def grounding_fields(refined: RefinedCandidate) -> dict[str, Any]:
            provenance = dict(refined.original_retrieval_provenance)
            return {
                "video_nomination_rank": provenance.get("video_nomination_rank"),
                "local_anchor_rank": provenance.get("local_anchor_rank"),
                "localization_score": provenance.get(
                    "localization_score", provenance.get("fusion_score")
                ),
                "localization_score_kind": provenance.get(
                    "localization_score_kind", "legacy_weighted_rrf"
                ),
                "source_localization_variant_ids": provenance.get(
                    "source_localization_variant_ids", []
                ),
                "localization_provenance": provenance.get("per_variant", []),
            }

        def grounding_artifact_fields(refined: RefinedCandidate) -> dict[str, Any]:
            if not self.video_conditioned_evidence_config.enabled:
                return {}
            return grounding_fields(refined)

        def evidence_identity_fields(
            evidence: QAEvidenceCandidate,
            refined: RefinedCandidate,
        ) -> dict[str, Any]:
            return {
                "refined_frame_id": (
                    refined.refined_frame_id
                    if _is_raw_refined_source(evidence.source_status)
                    else None
                ),
                "evidence_frame_id": evidence.frame_id,
                "evidence_source": evidence.source_status,
            }

        refined_by_rank = {
            candidate.original_candidate_rank: candidate
            for candidate in refined_candidates
        }
        if temporal_refinement_enabled:
            anchor_candidates = temporal_evidence_anchors
            source_candidates = tuple(
                (
                    candidate,
                    refined_by_rank.get(candidate.rank)
                    or _unattempted_refinement_record(
                        phase3_by_rank[candidate.rank]
                    ),
                )
                for candidate in anchor_candidates
            )
        elif preserve_keyframes:
            if (
                self.video_conditioned_evidence_config.keyframe_evidence_anchors_per_video
                > 1
            ):
                anchor_candidates = select_temporal_seed_anchors(
                    fused_result.ranked_candidates,
                    anchors_per_video=(
                        self.video_conditioned_evidence_config.keyframe_evidence_anchors_per_video
                    ),
                    video_cap=(
                        self.video_conditioned_evidence_config.keyframe_evidence_video_cap
                    ),
                )
            else:
                anchor_candidates = select_primary_keyframe_anchors(
                    fused_result.ranked_candidates,
                    video_cap=(
                        self.video_conditioned_evidence_config.keyframe_evidence_video_cap
                    ),
                )
            source_candidates = tuple(
                (candidate, refined_by_rank[candidate.rank])
                for candidate in anchor_candidates
            )
        else:
            source_candidates = tuple(
                (None, candidate) for candidate in ref_outcome.candidates
            )

        keyframe_records: list[dict[str, Any]] = []
        raw_refined_records: list[dict[str, Any]] = []
        temporal_refined_records: list[dict[str, Any]] = []
        generic_records: list[dict[str, Any]] = []
        seen_evidence_identities: set[tuple[str, int]] = set()
        for anchor_candidate, ref_cand in source_candidates:
            refinement_succeeded = (
                ref_cand.status is RefinementStatus.REFINED
                and ref_cand.refined_frame_id is not None
            )
            if not preserve_keyframes and not refinement_succeeded:
                warn_msg = (
                    f"Candidate rank {ref_cand.original_candidate_rank} ({ref_cand.video_id}) "
                    "refinement failed or incomplete."
                )
                diagnostics["warnings"].append(warn_msg)
                evidence_records.append(
                    {
                        "rank": ref_cand.original_candidate_rank,
                        "video_id": ref_cand.video_id,
                        "candidate_frame_id": ref_cand.candidate_frame_id,
                        "refined_frame_id": None,
                        "output_frame_id": None,
                        "retrieval_score": float(
                            ref_cand.original_retrieval_provenance.get("fusion_score", 0.0)
                        ),
                        "refinement_score": float(ref_cand.refinement_fusion_score)
                        if ref_cand.refinement_fusion_score is not None
                        else None,
                        "refinement_status": ref_cand.status.value,
                        "timestamp_seconds": None,
                        "answer": None,
                        "answer_score": None,
                        "answer_confidence_level": None,
                        "warning": warn_msg,
                        "skip_reason": "refinement_failed",
                    }
                )
                continue

            if not preserve_keyframes:
                try:
                    self.raw_video_registry.get(ref_cand.video_id)
                except KeyError:
                    warn_msg = (
                        f"Candidate rank {ref_cand.original_candidate_rank} "
                        f"({ref_cand.video_id}) raw video missing in registry."
                    )
                    diagnostics["warnings"].append(warn_msg)
                    evidence_records.append(
                        {
                            "rank": ref_cand.original_candidate_rank,
                            "video_id": ref_cand.video_id,
                            "candidate_frame_id": ref_cand.candidate_frame_id,
                            "refined_frame_id": ref_cand.refined_frame_id,
                            "output_frame_id": None,
                            "retrieval_score": float(
                                ref_cand.original_retrieval_provenance.get(
                                    "fusion_score", 0.0
                                )
                            ),
                            "refinement_score": (
                                float(ref_cand.refinement_fusion_score)
                                if ref_cand.refinement_fusion_score is not None
                                else None
                            ),
                            "refinement_status": ref_cand.status.value,
                            "timestamp_seconds": ref_cand.refined_timestamp_seconds,
                            "answer": None,
                            "answer_score": None,
                            "answer_confidence_level": None,
                            "warning": warn_msg,
                            "skip_reason": "raw_video_missing",
                        }
                    )
                    continue

            if preserve_keyframes:
                assert anchor_candidate is not None
                evidence_frame_id = (
                    ref_cand.refined_frame_id
                    if refinement_succeeded
                    else anchor_candidate.frame_id
                )
                evidence_source = (
                    TEMPORAL_REFINED
                    if temporal_refinement_enabled and refinement_succeeded
                    else RAW_REFINED
                    if refinement_succeeded
                    else KEYFRAME_ANCHOR
                )
                timestamp_seconds = (
                    ref_cand.refined_timestamp_seconds
                    if refinement_succeeded
                    else dict(anchor_candidate.diagnostic_metadata or {}).get(
                        "pts_time"
                    )
                )
                retrieval_score = float(anchor_candidate.score)
                attempted = (
                    ref_cand.original_candidate_rank in temporal_attempted_ranks
                    if temporal_refinement_enabled
                    else ref_cand.original_candidate_rank <= request.refine_top_n
                )
                keyframe_record = {
                    "rank": anchor_candidate.rank,
                    "video_id": anchor_candidate.video_id,
                    "frame_id": anchor_candidate.frame_id,
                    "video_nomination_rank": dict(
                        anchor_candidate.diagnostic_metadata or {}
                    ).get("video_nomination_rank"),
                    "local_anchor_rank": dict(
                        anchor_candidate.diagnostic_metadata or {}
                    ).get("local_anchor_rank"),
                }
                keyframe_records.append(keyframe_record)
            else:
                assert ref_cand.refined_frame_id is not None
                evidence_frame_id = ref_cand.refined_frame_id
                evidence_source = RAW_REFINED
                timestamp_seconds = ref_cand.refined_timestamp_seconds
                retrieval_score = float(
                    ref_cand.original_retrieval_provenance.get(
                        "fusion_score", 0.0
                    )
                )
                attempted = True

            coarse_frame = ref_cand.fine_frame_ids[0] if ref_cand.fine_frame_ids else None
            provenance = {
                "source_status": ref_cand.status.value,
                "evidence_source": evidence_source,
                "coarse_selected_frame_id": coarse_frame,
                "candidate_frame_id": ref_cand.candidate_frame_id,
                "raw_refinement_attempted": attempted,
                "raw_refinement_status": ref_cand.status.value,
                "raw_refinement_replaced_anchor": bool(
                    refinement_succeeded
                    and ref_cand.refined_frame_id != ref_cand.candidate_frame_id
                ),
                "fallback_to_keyframe": bool(
                    preserve_keyframes
                    and not refinement_succeeded
                    and (temporal_refinement_enabled or attempted)
                ),
                "refinement_policy": (
                    QA_MULTI_SEED_TEMPORAL_REFINEMENT_V1
                    if temporal_refinement_enabled
                    else "LEGACY_REFINE_TOP_N"
                ),
                **grounding_artifact_fields(ref_cand),
            }
            evidence_identity = (ref_cand.video_id, evidence_frame_id)
            if evidence_identity in seen_evidence_identities:
                continue
            seen_evidence_identities.add(evidence_identity)
            ev_cand = QAEvidenceCandidate(
                query_id=request.query_id,
                rank=(
                    len(evidence_bank) + 1
                    if temporal_refinement_enabled
                    else ref_cand.original_candidate_rank
                ),
                video_id=ref_cand.video_id,
                frame_id=evidence_frame_id,
                retrieval_score=retrieval_score,
                evidence_score=(
                    ref_cand.refinement_fusion_score
                    if refinement_succeeded
                    else None
                ),
                source_status=evidence_source,
                timestamp_seconds=timestamp_seconds,
                provenance=provenance,
            )
            evidence_bank.append((ev_cand, ref_cand))
            generic_record = {
                "query_id": request.query_id,
                "rank": ev_cand.rank,
                "video_id": ev_cand.video_id,
                "candidate_frame_id": ref_cand.candidate_frame_id,
                "frame_id": ev_cand.frame_id,
                "evidence_frame_id": ev_cand.frame_id,
                "video_nomination_rank": provenance.get("video_nomination_rank"),
                "local_anchor_rank": provenance.get("local_anchor_rank"),
                "evidence_source": evidence_source,
                "raw_refinement_attempted": attempted,
                "raw_refinement_status": ref_cand.status.value,
                "raw_refinement_replaced_anchor": provenance[
                    "raw_refinement_replaced_anchor"
                ],
                "fallback_to_keyframe": provenance["fallback_to_keyframe"],
                "localization_score": provenance.get("localization_score"),
                "source_localization_variant_ids": provenance.get(
                    "source_localization_variant_ids", []
                ),
            }
            generic_records.append(generic_record)
            if _is_raw_refined_source(evidence_source):
                raw_refined_records.append(generic_record)
            if evidence_source == TEMPORAL_REFINED:
                temporal_refined_records.append(generic_record)

        diagnostics["evidence_candidate_count"] = len(evidence_bank)
        if self.video_conditioned_evidence_config.enabled:
            diagnostics["keyframe_evidence_candidates"] = keyframe_records
            diagnostics["raw_refined_evidence_candidates"] = raw_refined_records
            diagnostics["generic_evidence_bank_candidates"] = generic_records
            diagnostics["keyframe_evidence_count"] = len(keyframe_records)
            diagnostics["raw_refined_evidence_count"] = len(raw_refined_records)
            diagnostics["generic_evidence_bank_count"] = len(generic_records)
            diagnostics["evidence_bank"] = generic_records
            if temporal_refinement_enabled:
                diagnostics["temporal_refined_evidence_candidates"] = (
                    temporal_refined_records
                )
                diagnostics["temporal_refined_evidence_count"] = len(
                    temporal_refined_records
                )
                diagnostics["temporal_refinement_fallback_count"] = sum(
                    item["evidence_source"] == KEYFRAME_ANCHOR
                    for item in generic_records
                )
                diagnostics["temporal_evidence_candidates"] = generic_records

        if not evidence_bank:
            store_evidence_records()
            timings.total_seconds = self.clock() - t_start
            result = QAResult(
                query_id=request.query_id,
                question_type=q_type,
                predictions=[],
                warnings=diagnostics["warnings"],
            )
            return result, timings, diagnostics

        # Artifact-backed object/entity answers consume the QA-A1 evidence bank without
        # pretending that detections exist on arbitrary decoded raw frames.
        if object_provider_supported:
            assert self.object_answer_provider is not None
            diagnostics["provider_evidence_candidates"] = list(generic_records)
            diagnostics["provider_evidence_count"] = len(generic_records)
            diagnostics["usable_evidence_candidates"] = list(generic_records)
            t_object = self.clock()
            qa_result, object_telemetry = self.object_answer_provider.answer(
                query_id=request.query_id,
                question_type=q_type,
                evidence=tuple(item[0] for item in evidence_bank),
                output_top_k=request.output_top_k,
                question_text=request.question_en or request.question,
                warnings=diagnostics["warnings"],
            )
            timings.answer_scoring_seconds = self.clock() - t_object
            diagnostics.update(object_telemetry)
            diagnostics["unsupported_reason"] = qa_result.unsupported_reason
            val_errors = validate_ranked_top100(
                qa_result.predictions,
                expected_task="qa",
                expected_query_id=request.query_id,
            )
            if val_errors:
                err_msgs = "; ".join(error.message for error in val_errors)
                raise ValueError(
                    f"QA object prediction validation failed for request "
                    f"{request.query_id}: {err_msgs}"
                )
            prediction_by_identity = {
                (prediction.video_id, prediction.frame_id, prediction.answer): prediction
                for prediction in qa_result.predictions
            }
            for item in object_telemetry["object_evidence"]:
                matched_evidence, matched_refined = next(
                    (candidate, refined)
                    for candidate, refined in evidence_bank
                    if candidate.rank == item["evidence_rank"]
                )
                evidence_records.append(
                    {
                        "rank": item["evidence_rank"],
                        "video_id": item["video_id"],
                        "candidate_frame_id": matched_refined.candidate_frame_id,
                        "refined_frame_id": (
                            matched_refined.refined_frame_id
                            if _is_raw_refined_source(
                                matched_evidence.source_status
                            )
                            else None
                        ),
                        "evidence_frame_id": matched_evidence.frame_id,
                        "evidence_source": matched_evidence.source_status,
                        "output_frame_id": item["object_source_frame_id"],
                        "object_source_frame_id": item["object_source_frame_id"],
                        "object_frame_distance": item["frame_distance"],
                        "object_lookup_kind": item["lookup_kind"],
                        "object_detection_count": item["detection_count"],
                        "answer": None,
                        "answer_score": None,
                        "answer_confidence_level": "ARTIFACT_BACKED",
                        "warning": None,
                        "skip_reason": (
                            None if item["object_source_frame_id"] is not None else "object_miss"
                        ),
                    }
                )
            diagnostics["object_prediction_identity_count"] = len(prediction_by_identity)
            store_evidence_records()
            diagnostics["final_predictions"] = [
                {
                    "rank": prediction.rank,
                    "video_id": prediction.video_id,
                    "frame_id": prediction.frame_id,
                    "answer": prediction.answer,
                }
                for prediction in qa_result.predictions
            ]
            timings.total_seconds = self.clock() - t_start
            return qa_result, timings, diagnostics

        # Step 5: Exact Evidence Frame Decode & Image Encode
        t_dec = self.clock()
        decoded_images: list[np.ndarray] = []
        valid_evidence_cands: list[tuple[QAEvidenceCandidate, RefinedCandidate]] = []

        cands_for_image_decode = evidence_bank
        if isinstance(
            self.candidate_provider,
            VisualOntologyAnswerCandidateProvider,
        ) and self.candidate_provider.supports(q_type):
            cands_for_image_decode = evidence_bank[
                : self.candidate_provider.config.evidence_frame_budget
            ]
        if ocr_provider_supported:
            assert self.ocr_answer_provider is not None
            cands_for_image_decode = evidence_bank[
                : self.ocr_answer_provider.config.evidence_frame_budget
            ]
            diagnostics["ocr_frames_requested"] = len(cands_for_image_decode)

        for ev_cand, ref_cand in cands_for_image_decode:
            try:
                try:
                    video_record = self.raw_video_registry.get(ev_cand.video_id)
                except KeyError as exc:
                    raise ValueError("raw video missing in registry") from exc
                if (
                    video_record.raw_video_path is None
                    or not video_record.raw_video_path.is_file()
                ):
                    warn_msg = f"Raw video path invalid for {ev_cand.video_id}."
                    diagnostics["warnings"].append(warn_msg)
                    evidence_records.append(
                        {
                            "rank": ref_cand.original_candidate_rank,
                            "video_id": ref_cand.video_id,
                            "candidate_frame_id": ref_cand.candidate_frame_id,
                            "refined_frame_id": ref_cand.refined_frame_id,
                            "output_frame_id": None,
                            "retrieval_score": float(ev_cand.retrieval_score),
                            "refinement_score": float(ev_cand.evidence_score)
                            if ev_cand.evidence_score is not None
                            else None,
                            "refinement_status": ref_cand.status.value,
                            "timestamp_seconds": ref_cand.refined_timestamp_seconds,
                            "answer": None,
                            "answer_score": None,
                            "answer_confidence_level": None,
                            "warning": warn_msg,
                            "skip_reason": "raw_video_path_invalid",
                        }
                    )
                    continue

                probe = self.decoder.probe(video_record)
                dec_req = DecodeRequest(
                    probe=probe,
                    frame_ids=(ev_cand.frame_id,),
                    max_decoded_frames=500,
                )
                dec_res = self.decoder.decode(dec_req)
                if not dec_res.frames:
                    warn_msg = (
                        f"Decode returned empty frames for {ev_cand.video_id} "
                        f"frame {ev_cand.frame_id}."
                    )
                    diagnostics["warnings"].append(warn_msg)
                    evidence_records.append(
                        {
                            "rank": ref_cand.original_candidate_rank,
                            "video_id": ref_cand.video_id,
                            "candidate_frame_id": ref_cand.candidate_frame_id,
                            "refined_frame_id": ref_cand.refined_frame_id,
                            "output_frame_id": None,
                            "retrieval_score": float(ev_cand.retrieval_score),
                            "refinement_score": float(ev_cand.evidence_score)
                            if ev_cand.evidence_score is not None
                            else None,
                            "refinement_status": ref_cand.status.value,
                            "timestamp_seconds": ref_cand.refined_timestamp_seconds,
                            "answer": None,
                            "answer_score": None,
                            "answer_confidence_level": None,
                            "warning": warn_msg,
                            "skip_reason": "decode_empty",
                        }
                    )
                    continue

                frame_obj = dec_res.frames[0]
                if frame_obj.absolute_frame_id != ev_cand.frame_id:
                    warn_msg = (
                        f"Frame ID mismatch on decode for {ev_cand.video_id}: requested "
                        f"{ev_cand.frame_id}, got {frame_obj.absolute_frame_id}."
                    )
                    diagnostics["warnings"].append(warn_msg)
                    evidence_records.append(
                        {
                            "rank": ref_cand.original_candidate_rank,
                            "video_id": ref_cand.video_id,
                            "candidate_frame_id": ref_cand.candidate_frame_id,
                            "refined_frame_id": ref_cand.refined_frame_id,
                            "output_frame_id": None,
                            "retrieval_score": float(ev_cand.retrieval_score),
                            "refinement_score": float(ev_cand.evidence_score)
                            if ev_cand.evidence_score is not None
                            else None,
                            "refinement_status": ref_cand.status.value,
                            "timestamp_seconds": ref_cand.refined_timestamp_seconds,
                            "answer": None,
                            "answer_score": None,
                            "answer_confidence_level": None,
                            "warning": warn_msg,
                            "skip_reason": "frame_id_mismatch",
                        }
                    )
                    continue

                decoded_images.append(frame_obj.image)
                valid_evidence_cands.append((ev_cand, ref_cand))
            except Exception as exc:
                warn_msg = (
                    f"Decode exception for {ev_cand.video_id} frame {ev_cand.frame_id}: {exc}"
                )
                diagnostics["warnings"].append(warn_msg)
                evidence_records.append(
                    {
                        "rank": ref_cand.original_candidate_rank,
                        "video_id": ref_cand.video_id,
                        "candidate_frame_id": ref_cand.candidate_frame_id,
                        "refined_frame_id": ref_cand.refined_frame_id,
                        "output_frame_id": None,
                        "retrieval_score": float(ev_cand.retrieval_score),
                        "refinement_score": float(ev_cand.evidence_score)
                        if ev_cand.evidence_score is not None
                        else None,
                        "refinement_status": ref_cand.status.value,
                        "timestamp_seconds": ref_cand.refined_timestamp_seconds,
                        "answer": None,
                        "answer_score": None,
                        "answer_confidence_level": None,
                        "warning": warn_msg,
                        "skip_reason": f"decode_exception: {exc}",
                    }
                )

        timings.evidence_decode_seconds = self.clock() - t_dec
        if ocr_provider_supported:
            timings.ocr_decode_seconds = timings.evidence_decode_seconds
            diagnostics["ocr_decode_seconds"] = timings.ocr_decode_seconds
        diagnostics["decoded_frame_count"] = len(decoded_images)
        diagnostics["usable_evidence_candidates"] = []
        provider_records: list[dict[str, Any]] = []
        for evidence_candidate, refined_candidate in valid_evidence_cands:
            usable_record = {
                "rank": evidence_candidate.rank,
                "video_id": evidence_candidate.video_id,
                "frame_id": evidence_candidate.frame_id,
                **(
                    {
                        "candidate_frame_id": refined_candidate.candidate_frame_id,
                        "timestamp_seconds": evidence_candidate.timestamp_seconds,
                        "refinement_score": evidence_candidate.evidence_score,
                        **(
                            {
                                "evidence_source": evidence_candidate.provenance.get(
                                    "evidence_source"
                                )
                            }
                            if preserve_keyframes
                            else {}
                        ),
                        **grounding_fields(refined_candidate),
                    }
                    if self.video_conditioned_evidence_config.enabled
                    else {}
                ),
            }
            diagnostics["usable_evidence_candidates"].append(usable_record)
            provider_records.append(dict(usable_record))
        if self.video_conditioned_evidence_config.enabled:
            diagnostics["provider_evidence_candidates"] = provider_records
            diagnostics["provider_evidence_count"] = len(provider_records)

        if not decoded_images and not ocr_provider_supported:
            store_evidence_records()
            timings.total_seconds = self.clock() - t_start
            result = QAResult(
                query_id=request.query_id,
                question_type=q_type,
                predictions=[],
                warnings=diagnostics["warnings"],
            )
            return result, timings, diagnostics

        if ocr_provider_supported:
            assert self.ocr_answer_provider is not None
            t_ocr = self.clock()
            qa_result, ocr_telemetry = self.ocr_answer_provider.answer(
                query_id=request.query_id,
                question_type=q_type,
                evidence=tuple(
                    (evidence_candidate, image)
                    for (evidence_candidate, _refined_candidate), image in zip(
                        valid_evidence_cands,
                        decoded_images,
                    )
                ),
                output_top_k=request.output_top_k,
                warnings=diagnostics["warnings"],
            )
            timings.ocr_inference_seconds = float(
                ocr_telemetry["ocr_inference_seconds"]
            )
            timings.answer_scoring_seconds = self.clock() - t_ocr
            diagnostics.update(ocr_telemetry)
            diagnostics["ocr_frames_requested"] = len(cands_for_image_decode)
            diagnostics["ocr_decode_seconds"] = timings.ocr_decode_seconds
            diagnostics["unsupported_reason"] = qa_result.unsupported_reason
            validation_errors = validate_ranked_top100(
                qa_result.predictions,
                expected_task="qa",
                expected_query_id=request.query_id,
            )
            if validation_errors:
                messages = "; ".join(error.message for error in validation_errors)
                raise ValueError(
                    f"QA OCR prediction validation failed for request "
                    f"{request.query_id}: {messages}"
                )
            prediction_by_frame = {
                (prediction.video_id, prediction.frame_id): prediction
                for prediction in qa_result.predictions
            }
            for evidence_candidate, refined_candidate in valid_evidence_cands:
                prediction = prediction_by_frame.get(
                    (evidence_candidate.video_id, evidence_candidate.frame_id)
                )
                evidence_records.append(
                    {
                        "rank": evidence_candidate.rank,
                        "video_id": evidence_candidate.video_id,
                        "candidate_frame_id": refined_candidate.candidate_frame_id,
                        **evidence_identity_fields(
                            evidence_candidate, refined_candidate
                        ),
                        "output_frame_id": (
                            prediction.frame_id if prediction is not None else None
                        ),
                        "retrieval_score": float(evidence_candidate.retrieval_score),
                        "refinement_score": (
                            float(evidence_candidate.evidence_score)
                            if evidence_candidate.evidence_score is not None
                            else None
                        ),
                        "refinement_status": refined_candidate.status.value,
                        "timestamp_seconds": evidence_candidate.timestamp_seconds,
                        **grounding_artifact_fields(refined_candidate),
                        "answer": prediction.answer if prediction is not None else None,
                        "answer_score": None,
                        "answer_confidence_level": "OCR_EVIDENCE",
                        "warning": None,
                        "skip_reason": (
                            None if prediction is not None else "no_ocr_prediction_for_frame"
                        ),
                    }
                )
            evidence_records.sort(key=lambda record: record["rank"])
            store_evidence_records()
            diagnostics["final_predictions"] = [
                {
                    "rank": prediction.rank,
                    "video_id": prediction.video_id,
                    "frame_id": prediction.frame_id,
                    "answer": prediction.answer,
                }
                for prediction in qa_result.predictions
            ]
            timings.total_seconds = self.clock() - t_start
            return qa_result, timings, diagnostics

        if not provider_supported:
            unsupported_reason = "UNSUPPORTED_NO_PROVIDER"
            diagnostics["unsupported_reason"] = unsupported_reason
            msg = (
                "Evidence grounding completed, but no current answer provider supports "
                f"question type {q_type.value}."
            )
            diagnostics["warnings"].append(msg)
            for evidence_candidate, refined_candidate in valid_evidence_cands:
                evidence_records.append(
                    {
                        "rank": evidence_candidate.rank,
                        "video_id": evidence_candidate.video_id,
                        "candidate_frame_id": refined_candidate.candidate_frame_id,
                        **evidence_identity_fields(
                            evidence_candidate, refined_candidate
                        ),
                        "output_frame_id": None,
                        "retrieval_score": float(evidence_candidate.retrieval_score),
                        "refinement_score": (
                            float(evidence_candidate.evidence_score)
                            if evidence_candidate.evidence_score is not None
                            else None
                        ),
                        "refinement_status": refined_candidate.status.value,
                        "timestamp_seconds": evidence_candidate.timestamp_seconds,
                        **grounding_artifact_fields(refined_candidate),
                        "answer": None,
                        "answer_score": None,
                        "answer_confidence_level": "UNSUPPORTED",
                        "warning": msg,
                        "skip_reason": unsupported_reason,
                    }
                )
            evidence_records.sort(key=lambda record: record["rank"])
            store_evidence_records()
            timings.total_seconds = self.clock() - t_start
            return (
                QAResult(
                    query_id=request.query_id,
                    question_type=q_type,
                    predictions=[],
                    unsupported_reason=unsupported_reason,
                    warnings=diagnostics["warnings"],
                    diagnostics={
                        "confidence_level": "UNSUPPORTED",
                        "grounding_completed": True,
                    },
                ),
                timings,
                diagnostics,
            )

        t_img_enc = self.clock()
        img_embeddings_batch = self.shared_encoder.encode_images(decoded_images)
        timings.evidence_encode_seconds = self.clock() - t_img_enc
        diagnostics["encoded_image_count"] = len(img_embeddings_batch)

        image_embeddings_map: dict[tuple[str, int], np.ndarray] = {}
        for (ev_cand, _), img_vec in zip(valid_evidence_cands, img_embeddings_batch):
            image_embeddings_map[(ev_cand.video_id, ev_cand.frame_id)] = img_vec.astype(np.float32)

        # Step 6: Visual Prompts & Answer Scoring
        if not self.video_conditioned_evidence_config.enabled:
            answer_hypotheses = self._answer_hypotheses(
                q_type,
                request.question_en or request.question,
            )
        hypotheses = answer_hypotheses
        all_prompts: list[str] = []
        for hyp in hypotheses:
            for p in hyp.visual_prompts:
                if p not in all_prompts:
                    all_prompts.append(p)

        prompt_embeddings_map, prompt_enc_time = self.get_prompt_embeddings(all_prompts)
        timings.prompt_encode_seconds = prompt_enc_time

        t_ans = self.clock()
        qa_query = QAQuery(
            query_id=request.query_id,
            event_description=request.event_description,
            question=request.question,
            event_description_en=request.event_description_en,
            question_en=request.question_en,
            question_type=q_type,
        )
        qa_result = self.qa_engine.answer(
            qa_query,
            tuple(ec for ec, _ in valid_evidence_cands),
            image_embeddings=image_embeddings_map,
            prompt_embeddings=prompt_embeddings_map,
        )
        timings.answer_scoring_seconds = self.clock() - t_ans

        val_errors = validate_ranked_top100(
            qa_result.predictions,
            expected_task="qa",
            expected_query_id=request.query_id,
        )
        if val_errors:
            err_msgs = "; ".join(e.message for e in val_errors)
            raise ValueError(
                f"QA prediction validation failed for request {request.query_id}: {err_msgs}"
            )

        pred_map = {p.rank: p for p in qa_result.predictions}
        conf_level = qa_result.diagnostics.get("confidence_level", "BASELINE")
        scores_by_rank = qa_result.diagnostics.get("scores_by_rank", {})

        for ev_cand, ref_cand in valid_evidence_cands:
            pred = pred_map.get(ev_cand.rank)
            ans_score = scores_by_rank.get(ev_cand.rank)
            if pred is not None:
                evidence_records.append(
                    {
                        "rank": ev_cand.rank,
                        "video_id": ev_cand.video_id,
                        "candidate_frame_id": ref_cand.candidate_frame_id,
                        **evidence_identity_fields(ev_cand, ref_cand),
                        "output_frame_id": pred.frame_id,
                        "retrieval_score": float(ev_cand.retrieval_score),
                        "refinement_score": float(ev_cand.evidence_score)
                        if ev_cand.evidence_score is not None
                        else None,
                        "refinement_status": ref_cand.status.value,
                        "timestamp_seconds": ev_cand.timestamp_seconds,
                        **grounding_artifact_fields(ref_cand),
                        "answer": pred.answer,
                        "answer_score": float(ans_score) if ans_score is not None else None,
                        "answer_confidence_level": conf_level,
                        "warning": None,
                        "skip_reason": None,
                    }
                )
            else:
                evidence_records.append(
                    {
                        "rank": ev_cand.rank,
                        "video_id": ev_cand.video_id,
                        "candidate_frame_id": ref_cand.candidate_frame_id,
                        **evidence_identity_fields(ev_cand, ref_cand),
                        "output_frame_id": None,
                        "retrieval_score": float(ev_cand.retrieval_score),
                        "refinement_score": float(ev_cand.evidence_score)
                        if ev_cand.evidence_score is not None
                        else None,
                        "refinement_status": ref_cand.status.value,
                        "timestamp_seconds": ev_cand.timestamp_seconds,
                        **grounding_artifact_fields(ref_cand),
                        "answer": None,
                        "answer_score": float(ans_score) if ans_score is not None else None,
                        "answer_confidence_level": conf_level,
                        "warning": "Candidate decoded but produced no valid prediction.",
                        "skip_reason": "no_prediction_generated",
                    }
                )

        evidence_records.sort(key=lambda r: r["rank"])
        store_evidence_records()
        diagnostics["final_predictions"] = [
            {
                "rank": prediction.rank,
                "video_id": prediction.video_id,
                "frame_id": prediction.frame_id,
                "answer": prediction.answer,
            }
            for prediction in qa_result.predictions
        ]

        timings.total_seconds = self.clock() - t_start
        return qa_result, timings, diagnostics
