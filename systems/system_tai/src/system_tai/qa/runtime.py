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

from .answer_candidates import AnswerCandidateProvider, BaselineQuestionCandidateProvider
from .engine import QABaselineEngine
from .models import QAEvidenceCandidate, QAQuery, QAResult
from .question_types import QuestionType, classify_question_type


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
        qa_engine: QABaselineEngine | None = None,
        candidate_provider: AnswerCandidateProvider | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.exact_retriever = exact_retriever
        self.weighted_rrf = weighted_rrf
        self.refiner = refiner
        self.raw_video_registry = raw_video_registry
        self.decoder = decoder
        self.shared_encoder = shared_encoder
        self.qa_engine = qa_engine or QABaselineEngine()
        self.candidate_provider = candidate_provider or BaselineQuestionCandidateProvider()
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
            "retrieval_candidate_count": 0,
            "refined_candidate_count": 0,
            "evidence_candidate_count": 0,
            "decoded_frame_count": 0,
            "encoded_image_count": 0,
            "warnings": [],
        }

        # Step 1: Question classification
        q_type = classify_question_type(request.question, request.question_en)
        diagnostics["question_type"] = q_type.value

        if q_type == QuestionType.UNSUPPORTED:
            msg = (
                "Question type is UNSUPPORTED; zero predictions generated without running "
                "retrieval or refinement."
            )
            diagnostics["warnings"].append(msg)
            timings.total_seconds = self.clock() - t_start
            result = QAResult(
                query_id=request.query_id,
                question_type=q_type,
                predictions=[],
                unsupported_reason=msg,
                warnings=[msg],
            )
            return result, timings, diagnostics

        # Step 2: Event-only retrieval variants
        variants = request.variants()

        t_text = self.clock()
        event_texts = [v.text for v in variants]
        event_vectors = self.shared_encoder.encode_texts(event_texts)
        timings.text_encode_seconds = self.clock() - t_text

        t_ret = self.clock()
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
        phase3_candidates = tuple(
            Phase3Candidate(
                query_id=request.query_id,
                rank=c.rank,
                video_id=c.video_id,
                frame_id=c.frame_id,
                retrieval_score=c.score,
                retrieval_provenance={
                    "fusion_score": c.score,
                    "variant_hit_count": (c.diagnostic_metadata or {}).get("variant_hit_count"),
                    "best_individual_rank": (c.diagnostic_metadata or {}).get(
                        "best_individual_rank"
                    ),
                    "clip_row_diagnostic": c.clip_row,
                    "keyframe_order_diagnostic": c.keyframe_order,
                },
            )
            for c in fused_result.ranked_candidates
        )
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

        # Step 4: Filter usable candidates -> QAEvidenceCandidate
        cands_to_decode: list[tuple[QAEvidenceCandidate, RefinedCandidate, RawVideoRecord]] = []
        evidence_records: list[dict[str, Any]] = []

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
                },
            )
            cands_to_decode.append((ev_cand, ref_cand, video_record))

        diagnostics["evidence_candidate_count"] = len(cands_to_decode)

        if not cands_to_decode:
            diagnostics["evidence"] = evidence_records
            timings.total_seconds = self.clock() - t_start
            result = QAResult(
                query_id=request.query_id,
                question_type=q_type,
                predictions=[],
                warnings=diagnostics["warnings"],
            )
            return result, timings, diagnostics

        # Step 5: Exact Evidence Frame Decode & Image Encode
        t_dec = self.clock()
        decoded_images: list[np.ndarray] = []
        valid_evidence_cands: list[tuple[QAEvidenceCandidate, RefinedCandidate]] = []

        for ev_cand, ref_cand, video_record in cands_to_decode:
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
        diagnostics["decoded_frame_count"] = len(decoded_images)

        if not decoded_images:
            diagnostics["evidence"] = evidence_records
            timings.total_seconds = self.clock() - t_start
            result = QAResult(
                query_id=request.query_id,
                question_type=q_type,
                predictions=[],
                warnings=diagnostics["warnings"],
            )
            return result, timings, diagnostics

        t_img_enc = self.clock()
        img_embeddings_batch = self.shared_encoder.encode_images(decoded_images)
        timings.evidence_encode_seconds = self.clock() - t_img_enc
        diagnostics["encoded_image_count"] = len(img_embeddings_batch)

        image_embeddings_map: dict[tuple[str, int], np.ndarray] = {}
        for (ev_cand, _), img_vec in zip(valid_evidence_cands, img_embeddings_batch):
            image_embeddings_map[(ev_cand.video_id, ev_cand.frame_id)] = img_vec.astype(np.float32)

        # Step 6: Visual Prompts & Answer Scoring
        hypotheses = self.candidate_provider.get_candidates(q_type)
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
                        "answer": None,
                        "answer_score": float(ans_score) if ans_score is not None else None,
                        "answer_confidence_level": conf_level,
                        "warning": "Candidate decoded but produced no valid prediction.",
                        "skip_reason": "no_prediction_generated",
                    }
                )

        evidence_records.sort(key=lambda r: r["rank"])
        diagnostics["evidence"] = evidence_records

        timings.total_seconds = self.clock() - t_start
        return qa_result, timings, diagnostics
