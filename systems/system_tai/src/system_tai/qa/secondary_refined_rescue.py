# ==============================================================================================================
# Sprint 2D.1 Top-1 Secondary Refined Anchor Evidence Tail Rescue (QA Layer)
# ==============================================================================================================

from __future__ import annotations

import unicodedata
from typing import TYPE_CHECKING, Any, Sequence

from system_tai.qa.grounding import QAVideoConditionedEvidenceConfig
from system_tai.qa.models import QAEvidenceCandidate
from system_tai.qa.question_types import QuestionType
from system_tai.qa.rescue_tail import RescueCandidate, merge_rescue_tail
from system_tai.refinement.models import RefinedCandidate, RefinementStatus
from system_tai.refinement.video import DecodeRequest

if TYPE_CHECKING:
    from system_tai.kis.session_schema import QAQueryRequest
    from system_tai.qa.ocr_provider import OCRAnswerProvider
    from system_tai.qa.raw_registry import RawVideoRegistry
    from system_tai.refinement.video import VideoDecoder


def execute_top1_secondary_refined_rescue(
    *,
    request: QAQueryRequest,
    q_type: QuestionType,
    champion_selected_video_ids: Sequence[str],
    champion_refined_candidates: Sequence[RefinedCandidate],
    champion_predictions: Sequence[dict[str, Any]],
    canonical_evidence_cands: Sequence[tuple[QAEvidenceCandidate, RefinedCandidate]] = (),
    raw_video_registry: RawVideoRegistry,
    decoder: VideoDecoder,
    ocr_answer_provider: OCRAnswerProvider | None,
    ocr_provider_supported: bool,
    config: QAVideoConditionedEvidenceConfig,
) -> tuple[
    list[dict[str, Any]],
    list[RescueCandidate],
    list[dict[str, Any]],
    dict[str, Any],
]:
    """
    Executes bounded secondary refined anchor rescue for OCR queries on the Top-1 nominated video.
    Strictly preserves Champion ranks 1..95 and admits valid OCR predictions to tail slots 96..100.
    """
    rescue_candidates: list[RescueCandidate] = []
    admitted_tuples: list[dict[str, Any]] = []
    stage_telemetry: dict[str, Any] = {
        "enabled": config.top1_secondary_refined_rescue_enabled,
        "eligible": False,
        "reason": "NOT_EXECUTED",
        "top1_video": None,
        "primary_refined_anchor": None,
        "secondary_refined_anchor": None,
        "selected_physical_frame": None,
        "already_covered": False,
        "ocr_route_used": "OCR" if ocr_provider_supported else "NONE",
        "produced_answers": [],
        "rescue_candidates_count": 0,
        "admitted_tail_count": 0,
        "tail": {},
    }

    if not config.top1_secondary_refined_rescue_enabled:
        stage_telemetry["reason"] = "FEATURE_DISABLED"
        return list(champion_predictions), rescue_candidates, admitted_tuples, stage_telemetry

    if q_type != QuestionType.OCR or not ocr_provider_supported or ocr_answer_provider is None:
        stage_telemetry["reason"] = "NON_OCR_OR_UNSUPPORTED_PROVIDER"
        return list(champion_predictions), rescue_candidates, admitted_tuples, stage_telemetry

    if not champion_selected_video_ids:
        stage_telemetry["reason"] = "NO_SELECTED_VIDEOS"
        return list(champion_predictions), rescue_candidates, admitted_tuples, stage_telemetry

    top1_vid = champion_selected_video_ids[0]
    stage_telemetry["top1_video"] = top1_vid

    # Filter and sort Top-1 refined candidates using canonical anchor metadata
    top1_cands = [c for c in champion_refined_candidates if c.video_id == top1_vid]

    def _canonical_anchor_order(cand: RefinedCandidate) -> tuple[int, int]:
        prov = dict(getattr(cand, "original_retrieval_provenance", {}) or {})
        local_rank = prov.get("local_anchor_rank")
        if isinstance(local_rank, int):
            return (local_rank, getattr(cand, "original_candidate_rank", 1))
        return (getattr(cand, "original_candidate_rank", 1), 1)

    sorted_top1 = sorted(top1_cands, key=_canonical_anchor_order)

    if len(sorted_top1) < 2:
        stage_telemetry["reason"] = "NO_SECONDARY_CANDIDATE"
        return list(champion_predictions), rescue_candidates, admitted_tuples, stage_telemetry

    primary_cand = sorted_top1[0]
    secondary_cand = sorted_top1[1]

    stage_telemetry["primary_refined_anchor"] = (
        int(primary_cand.refined_frame_id)
        if primary_cand.status is RefinementStatus.REFINED and primary_cand.refined_frame_id is not None
        else int(primary_cand.candidate_frame_id)
    )

    # Contract 1: Strictly require secondary candidate to be REFINED with a valid refined_frame_id
    if secondary_cand.status is not RefinementStatus.REFINED or secondary_cand.refined_frame_id is None:
        stage_telemetry["reason"] = "NO_REFINED_SECONDARY_ANCHOR"
        return list(champion_predictions), rescue_candidates, admitted_tuples, stage_telemetry

    target_physical_frame = int(secondary_cand.refined_frame_id)
    stage_telemetry["secondary_refined_anchor"] = target_physical_frame
    stage_telemetry["selected_physical_frame"] = target_physical_frame

    # Contract 3: Deduplication / Already covered check
    covered_identities = {
        (ec.video_id, ec.frame_id) for ec, _ in canonical_evidence_cands
    }
    if (top1_vid, target_physical_frame) in covered_identities:
        stage_telemetry["already_covered"] = True
        stage_telemetry["reason"] = "ALREADY_COVERED"
        return list(champion_predictions), rescue_candidates, admitted_tuples, stage_telemetry

    stage_telemetry["eligible"] = True

    try:
        video_record = raw_video_registry.get(top1_vid)
        if not video_record or not video_record.raw_video_path or not video_record.raw_video_path.is_file():
            stage_telemetry["reason"] = "RAW_VIDEO_UNAVAILABLE"
            return list(champion_predictions), rescue_candidates, admitted_tuples, stage_telemetry

        probe = decoder.probe(video_record)
        if not (0 <= target_physical_frame < probe.total_frame_count):
            stage_telemetry["reason"] = "FRAME_OUT_OF_BOUNDS"
            return list(champion_predictions), rescue_candidates, admitted_tuples, stage_telemetry

        dec_req = DecodeRequest(
            probe=probe,
            frame_ids=(target_physical_frame,),
            max_decoded_frames=100,
        )
        dec_res = decoder.decode(dec_req)

        if not dec_res.frames:
            stage_telemetry["reason"] = "EMPTY_DECODE_FRAMES"
            return list(champion_predictions), rescue_candidates, admitted_tuples, stage_telemetry

        decoded_frame_obj = dec_res.frames[0]
        tail_budget = int(config.top1_secondary_refined_rescue_tail_budget)
        use_span_candidateizer = (
            config.top1_secondary_refined_rescue_span_candidateizer
            and q_type is QuestionType.OCR
        )

        stage_telemetry["span_candidateizer_enabled"] = use_span_candidateizer

        if use_span_candidateizer:
            from system_tai.qa.ocr_span_candidateizer import extract_and_rank_canonical_ocr_spans
            from system_tai.qa.ocr_provider import _portable_pixmap

            payload = _portable_pixmap(decoded_frame_obj.image)
            ocr_backend = ocr_answer_provider.backend
            completed = ocr_backend._invoke(
                (
                    "stdin",
                    "stdout",
                    "-l",
                    "+".join(ocr_answer_provider.config.languages),
                    "--psm",
                    str(ocr_answer_provider.config.page_segmentation_mode),
                    "tsv",
                ),
                input_bytes=payload,
            )
            all_ranked_spans = extract_and_rank_canonical_ocr_spans(
                completed.stdout,
                max_n=4,
            )
            stage_telemetry["candidate_universe_size"] = len(all_ranked_spans)
            ranked_spans = all_ranked_spans[:tail_budget]
            stage_telemetry["selected_spans_count"] = len(ranked_spans)
            for span_cand in ranked_spans:
                rescue_candidates.append(
                    RescueCandidate(
                        video_id=top1_vid,
                        frame_id=target_physical_frame,
                        answer=span_cand.raw_span,
                        rescue_score=round(span_cand.score, 6),
                        rescue_source="top1_secondary_refined_rescue_span",
                        provenance={
                            "top1_video": top1_vid,
                            "primary_anchor": stage_telemetry["primary_refined_anchor"],
                            "secondary_anchor": target_physical_frame,
                            "raw_answer": span_cand.raw_span,
                            "normalized_span": span_cand.normalized_span,
                            "n_gram": span_cand.n_gram,
                            "span_score": span_cand.score,
                            "score_components": span_cand.score_components,
                        },
                    )
                )
        else:
            # Exact Champion OCR Answer Routing (Line-Level)
            ev_cand = QAEvidenceCandidate(
                query_id=request.query_id,
                rank=1,
                video_id=top1_vid,
                frame_id=target_physical_frame,
                retrieval_score=1.0,
                source_status="TOP1_SECONDARY_REFINED_RESCUE",
            )
            ocr_res, _ = ocr_answer_provider.answer(
                query_id=request.query_id,
                question_type=q_type,
                evidence=((ev_cand, decoded_frame_obj.image),),
                output_top_k=tail_budget,
                warnings=[],
            )

            seen_keys: set[tuple[str, int, str]] = set()
            for pred in ocr_res.predictions:
                if pred.answer:
                    ans_norm = " ".join(unicodedata.normalize("NFKC", pred.answer).split()).casefold()
                    key = (top1_vid, target_physical_frame, ans_norm)
                    if key not in seen_keys:
                        seen_keys.add(key)
                        rescue_candidates.append(
                            RescueCandidate(
                                video_id=top1_vid,
                                frame_id=target_physical_frame,
                                answer=pred.answer,
                                rescue_score=0.5,
                                rescue_source="top1_secondary_refined_rescue",
                                provenance={
                                    "top1_video": top1_vid,
                                    "primary_anchor": stage_telemetry["primary_refined_anchor"],
                                    "secondary_anchor": target_physical_frame,
                                    "raw_answer": pred.answer,
                                },
                            )
                        )

        stage_telemetry["reason"] = "RESCUE_PROCESSED"
        stage_telemetry["produced_answers"] = [
            {"frame_id": r.frame_id, "answer": r.answer, "score": r.rescue_score}
            for r in rescue_candidates
        ]
        stage_telemetry["rescue_candidates_count"] = len(rescue_candidates)

    except Exception as exc:
        import sys, traceback
        print(f"[TOP1 SECONDARY REFINED RESCUE ERROR for {top1_vid}]: {exc}", file=sys.stderr)
        traceback.print_exc()
        stage_telemetry["error"] = str(exc)

    # Merge rescue candidates into tail (ranks 96..100)
    merged_predictions = merge_rescue_tail(
        champion_predictions=champion_predictions,
        rescue_candidates=rescue_candidates,
        prefix_k=95,
        max_rescue=config.top1_secondary_refined_rescue_tail_budget,
    )

    admitted_tuples = [
        p for p in merged_predictions
        if str(p.get("slot_source", "")).startswith("RESCUE_TAIL_TOP1_SECONDARY_REFINED_RESCUE")
    ]

    stage_telemetry["admitted_tail_count"] = len(admitted_tuples)
    stage_telemetry["tail"] = {
        "admitted_count": len(admitted_tuples),
        "admitted_tuples": [
            {
                "rank": p.get("rank"),
                "video_id": p.get("video_id"),
                "frame_id": p.get("frame_id"),
                "answer": p.get("answer"),
                "slot_source": p.get("slot_source"),
            }
            for p in admitted_tuples
        ],
    }

    return merged_predictions, rescue_candidates, admitted_tuples, stage_telemetry
