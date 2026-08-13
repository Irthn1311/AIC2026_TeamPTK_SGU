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
    RawVideoRecord,
    RawVideoRegistry,
    VideoDecoder,
)
from system_tai.retrieval.multi_query import WeightedRRFRetriever
from system_tai.retrieval.vector_search import ExactNumpyRetriever
from system_tai.retrieval.video_evidence import VideoRestrictedFeatureSearcher

from .answer_candidates import AnswerCandidateProvider, BaselineQuestionCandidateProvider
from .engine import QABaselineEngine
from .grounding import (
    QA_VIDEO_CONDITIONED_EVIDENCE_V1,
    QAVideoConditionedEvidenceConfig,
    build_qa_grounding_result,
    nominate_qa_videos,
    nomination_diagnostics,
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

LEGACY_PHASE_P0 = "LEGACY_PHASE_P0"
QA_A2_CAPABILITY_AWARE = "QA_A2_CAPABILITY_AWARE"
QA_A3_CAPABILITY_AWARE = "QA_A3_CAPABILITY_AWARE"


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
    if qa_a2_enabled or qa_ocr_enabled:
        classification = classify_question(question, question_en)
        policy = QA_A3_CAPABILITY_AWARE if qa_ocr_enabled else QA_A2_CAPABILITY_AWARE
        return classification, policy
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
        self.qa_engine = qa_engine or QABaselineEngine()
        self.candidate_provider = candidate_provider or BaselineQuestionCandidateProvider()
        self.object_answer_provider = object_answer_provider
        self.ocr_answer_provider = ocr_answer_provider
        self.clock = clock

        self._prompt_cache: dict[str, np.ndarray] = {}

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
                    "localization_variant_count": 0,
                    "full_corpus_video_count": 0,
                    "full_corpus_store_scan_count": 0,
                    "selected_video_count": 0,
                    "selected_video_ids": [],
                    "restricted_store_scan_count": 0,
                    "restricted_rows_scored": 0,
                    "grounding_candidate_count": 0,
                    "selected_video_evidence": [],
                    "grounding_candidates": [],
                }
            )

        # Step 1: Question classification
        qa_a2_enabled = bool(
            self.object_answer_provider is not None
            and self.object_answer_provider.enabled
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

        answer_hypotheses = self.candidate_provider.get_candidates(q_type)
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
            diagnostics["localization_variant_count"] = len(variants)

        t_text = self.clock()
        event_texts = [v.text for v in variants]
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
        timings.refinement_seconds = self.clock() - t_ref
        diagnostics["refined_candidate_count"] = len(ref_outcome.candidates)
        diagnostics["refined_candidates"] = [
            {
                "original_rank": candidate.original_candidate_rank,
                "video_id": candidate.video_id,
                "candidate_frame_id": candidate.candidate_frame_id,
                "refined_frame_id": candidate.refined_frame_id,
                "status": candidate.status.value,
            }
            for candidate in ref_outcome.candidates
        ]

        # Step 4: Filter usable candidates -> QAEvidenceCandidate
        cands_to_decode: list[tuple[QAEvidenceCandidate, RefinedCandidate, RawVideoRecord]] = []
        evidence_records: list[dict[str, Any]] = []

        def store_evidence_records() -> None:
            diagnostics["evidence"] = evidence_records
            if self.video_conditioned_evidence_config.enabled:
                diagnostics["evidence_bank"] = evidence_records

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

        for ref_cand in ref_outcome.candidates:
            if ref_cand.refined_frame_id is None or ref_cand.status != RefinementStatus.REFINED:
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

            try:
                video_record = self.raw_video_registry.get(ref_cand.video_id)
            except KeyError:
                warn_msg = (
                    f"Candidate rank {ref_cand.original_candidate_rank} ({ref_cand.video_id}) "
                    "raw video missing in registry."
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
                            ref_cand.original_retrieval_provenance.get("fusion_score", 0.0)
                        ),
                        "refinement_score": float(ref_cand.refinement_fusion_score)
                        if ref_cand.refinement_fusion_score is not None
                        else None,
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

            coarse_frame = (
                ref_cand.fine_frame_ids[0] if ref_cand.fine_frame_ids else None
            )
            ev_cand = QAEvidenceCandidate(
                query_id=request.query_id,
                rank=ref_cand.original_candidate_rank,
                video_id=ref_cand.video_id,
                frame_id=ref_cand.refined_frame_id,
                retrieval_score=ref_cand.original_retrieval_provenance.get("fusion_score", 0.0),
                evidence_score=ref_cand.refinement_fusion_score,
                timestamp_seconds=ref_cand.refined_timestamp_seconds,
                provenance={
                    "source_status": ref_cand.status.value,
                    "coarse_selected_frame_id": coarse_frame,
                    "candidate_frame_id": ref_cand.candidate_frame_id,
                    **grounding_artifact_fields(ref_cand),
                },
            )
            cands_to_decode.append((ev_cand, ref_cand, video_record))

        diagnostics["evidence_candidate_count"] = len(cands_to_decode)

        if not cands_to_decode:
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
            t_object = self.clock()
            qa_result, object_telemetry = self.object_answer_provider.answer(
                query_id=request.query_id,
                question_type=q_type,
                evidence=tuple((item[0], item[1]) for item in cands_to_decode),
                output_top_k=request.output_top_k,
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
                evidence_records.append(
                    {
                        "rank": item["evidence_rank"],
                        "video_id": item["video_id"],
                        "candidate_frame_id": next(
                            refined.candidate_frame_id
                            for candidate, refined, _ in cands_to_decode
                            if candidate.rank == item["evidence_rank"]
                        ),
                        "refined_frame_id": item["requested_frame_id"],
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

        cands_for_image_decode = cands_to_decode
        if ocr_provider_supported:
            assert self.ocr_answer_provider is not None
            cands_for_image_decode = cands_to_decode[
                : self.ocr_answer_provider.config.evidence_frame_budget
            ]
            diagnostics["ocr_frames_requested"] = len(cands_for_image_decode)

        for ev_cand, ref_cand, video_record in cands_for_image_decode:
            try:
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
        for evidence_candidate, refined_candidate in valid_evidence_cands:
            diagnostics["usable_evidence_candidates"].append(
                {
                "rank": evidence_candidate.rank,
                "video_id": evidence_candidate.video_id,
                "frame_id": evidence_candidate.frame_id,
                **(
                    {
                        "candidate_frame_id": refined_candidate.candidate_frame_id,
                        "timestamp_seconds": evidence_candidate.timestamp_seconds,
                        "refinement_score": evidence_candidate.evidence_score,
                        **grounding_fields(refined_candidate),
                    }
                    if self.video_conditioned_evidence_config.enabled
                    else {}
                ),
                }
            )

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
                        "refined_frame_id": evidence_candidate.frame_id,
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
                        "refined_frame_id": evidence_candidate.frame_id,
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
            answer_hypotheses = self.candidate_provider.get_candidates(q_type)
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
                        "refined_frame_id": ev_cand.frame_id,
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
                        "refined_frame_id": ev_cand.frame_id,
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
