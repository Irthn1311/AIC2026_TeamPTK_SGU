# ==============================================================================================================
# Sprint 2C.1 Bounded Negative Temporal Tail Rescue Orchestrator (QA Layer)
# ==============================================================================================================

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

from system_tai.qa.grounding import QAVideoConditionedEvidenceConfig
from system_tai.qa.models import QAEvidenceCandidate, QAQuery
from system_tai.qa.question_types import QuestionType
from system_tai.qa.rescue_tail import RescueCandidate, merge_rescue_tail
from system_tai.refinement.models import RefinedCandidate, RefinementStatus
from system_tai.refinement.video import DecodeRequest

if TYPE_CHECKING:
    from system_tai.features.query_encoder import SharedOpenAIClipEncoder
    from system_tai.kis.session_schema import QAQueryRequest
    from system_tai.qa.engine import QAEngine
    from system_tai.qa.ocr_provider import OCRAnswerProvider
    from system_tai.qa.raw_registry import RawVideoRegistry
    from system_tai.refinement.video import VideoDecoder


def execute_sidepath_negative_temporal_rescue(
    *,
    request: QAQueryRequest,
    q_type: QuestionType,
    champion_selected_video_ids: Sequence[str],
    champion_refined_candidates: Sequence[RefinedCandidate],
    champion_predictions: Sequence[dict[str, Any]],
    raw_video_registry: RawVideoRegistry,
    decoder: VideoDecoder,
    encoder: SharedOpenAIClipEncoder,
    qa_engine: QAEngine,
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
    Executes generic bounded negative temporal rescue on the Top-1 nominated video's top refined anchor.
    Strictly preserves Champion ranks 1..95 and merges rescue candidates into tail slots 96..100.
    """
    rescue_candidates: list[RescueCandidate] = []
    admitted_tuples: list[dict[str, Any]] = []
    stage_telemetry: dict[str, Any] = {
        "enabled": config.bounded_negative_temporal_rescue_enabled,
        "source_video": None,
        "source_anchor_frame": None,
        "source_refinement_status": None,
        "offsets_evaluated": list(config.bounded_negative_temporal_rescue_offsets),
        "valid_offset_frames": [],
        "decoded_frames": [],
        "answer_route_used": "NONE",
        "produced_answers": [],
        "rescue_candidates_count": 0,
        "admitted_tail_count": 0,
        "tail": {},
    }

    if not config.bounded_negative_temporal_rescue_enabled or not champion_selected_video_ids:
        return list(champion_predictions), rescue_candidates, admitted_tuples, stage_telemetry

    top_vid = champion_selected_video_ids[0]
    stage_telemetry["source_video"] = top_vid

    # Find the top refined candidate for the Top-1 nominated video
    video_cands = [c for c in champion_refined_candidates if c.video_id == top_vid]
    if not video_cands:
        return list(champion_predictions), rescue_candidates, admitted_tuples, stage_telemetry

    sorted_cands = sorted(video_cands, key=lambda c: getattr(c, "original_candidate_rank", 1))
    top_cand = sorted_cands[0]

    source_anchor_frame = (
        int(top_cand.refined_frame_id)
        if getattr(top_cand, "status", None) is RefinementStatus.REFINED and top_cand.refined_frame_id is not None
        else int(top_cand.candidate_frame_id)
    )
    ref_status_str = top_cand.status.value if hasattr(top_cand, "status") else "KEYFRAME_ANCHOR"

    stage_telemetry["source_anchor_frame"] = source_anchor_frame
    stage_telemetry["source_refinement_status"] = ref_status_str

    try:
        video_record = raw_video_registry.get(top_vid)
        if not video_record or not video_record.raw_video_path or not video_record.raw_video_path.is_file():
            return list(champion_predictions), rescue_candidates, admitted_tuples, stage_telemetry

        probe = decoder.probe(video_record)

        # Drop out-of-bounds offset frames, never clamp
        valid_offset_frames: list[int] = []
        for offset in config.bounded_negative_temporal_rescue_offsets:
            cand_frame = source_anchor_frame + offset
            if 0 <= cand_frame < probe.total_frame_count and cand_frame not in valid_offset_frames:
                valid_offset_frames.append(cand_frame)

        stage_telemetry["valid_offset_frames"] = valid_offset_frames

        if not valid_offset_frames:
            return list(champion_predictions), rescue_candidates, admitted_tuples, stage_telemetry

        # Decode valid offset frames (sorted ascending as required by DecodeRequest)
        dec_req = DecodeRequest(
            probe=probe,
            frame_ids=tuple(sorted(valid_offset_frames)),
            max_decoded_frames=500,
        )
        dec_res = decoder.decode(dec_req)
        decoded_frames_map = {f.absolute_frame_id: f for f in dec_res.frames}
        stage_telemetry["decoded_frames"] = [f.absolute_frame_id for f in dec_res.frames]

        if not dec_res.frames:
            return list(champion_predictions), rescue_candidates, admitted_tuples, stage_telemetry

        # Canonical Champion Answer Routing
        if ocr_provider_supported and ocr_answer_provider is not None:
            stage_telemetry["answer_route_used"] = "OCR"
            evidence_pairs = tuple(
                (
                    QAEvidenceCandidate(
                        query_id=request.query_id,
                        rank=1,
                        video_id=top_vid,
                        frame_id=f.absolute_frame_id,
                        retrieval_score=1.0,
                        source_status="NEGATIVE_TEMPORAL_PROBE",
                    ),
                    f.image,
                )
                for f in dec_res.frames
            )
            ocr_res, _ = ocr_answer_provider.answer(
                query_id=request.query_id,
                question_type=q_type,
                evidence=evidence_pairs,
                output_top_k=len(evidence_pairs),
            )
            for pred in ocr_res.predictions:
                if pred.answer:
                    rescue_candidates.append(
                        RescueCandidate(
                            video_id=top_vid,
                            frame_id=pred.frame_id,
                            answer=pred.answer,
                            rescue_score=0.5,
                            rescue_source="bounded_negative_temporal_rescue",
                            provenance={
                                "source_video": top_vid,
                                "source_anchor_frame": source_anchor_frame,
                                "refinement_status": ref_status_str,
                                "probe_frame": pred.frame_id,
                                "offset_from_source": pred.frame_id - source_anchor_frame,
                                "answer_route": "OCR",
                            },
                        )
                    )
        else:
            stage_telemetry["answer_route_used"] = "QA_ENGINE"
            q_vi = request.question
            q_en = request.question_en or request.event_description_en
            qa_q = QAQuery(
                query_id=request.query_id,
                event_description=q_vi,
                question=q_vi,
                event_description_en=q_en,
                question_en=q_en,
                question_type=q_type,
            )

            for cand_f in valid_offset_frames:
                frame_obj = decoded_frames_map.get(cand_f)
                if frame_obj is None:
                    continue

                img_vec = encoder.encode_images([frame_obj.image])[0]
                ev_cand = QAEvidenceCandidate(
                    query_id=request.query_id,
                    rank=1,
                    video_id=top_vid,
                    frame_id=cand_f,
                    retrieval_score=1.0,
                    source_status="NEGATIVE_TEMPORAL_PROBE",
                )
                q_res = qa_engine.answer(
                    qa_q,
                    (ev_cand,),
                    image_embeddings={(top_vid, cand_f): img_vec},
                    output_top_k=3,
                )
                for pred in q_res.predictions:
                    if pred.answer:
                        ans_score = float(getattr(pred, "score", 0.5) or 0.5)
                        rescue_candidates.append(
                            RescueCandidate(
                                video_id=top_vid,
                                frame_id=cand_f,
                                answer=pred.answer,
                                rescue_score=ans_score,
                                rescue_source="bounded_negative_temporal_rescue",
                                provenance={
                                    "source_video": top_vid,
                                    "source_anchor_frame": source_anchor_frame,
                                    "refinement_status": ref_status_str,
                                    "probe_frame": cand_f,
                                    "offset_from_source": cand_f - source_anchor_frame,
                                    "answer_route": "QA_ENGINE",
                                },
                            )
                        )

        stage_telemetry["produced_answers"] = [
            {"frame_id": r.frame_id, "answer": r.answer, "score": r.rescue_score}
            for r in rescue_candidates
        ]
        stage_telemetry["rescue_candidates_count"] = len(rescue_candidates)

    except Exception as exc:
        import sys, traceback
        print(f"[NEGATIVE TEMPORAL RESCUE ERROR for {top_vid}]: {exc}", file=sys.stderr)
        traceback.print_exc()
        stage_telemetry["error"] = str(exc)

    # Merge rescue candidates into tail (ranks 96..100)
    merged_predictions = merge_rescue_tail(
        champion_predictions=champion_predictions,
        rescue_candidates=rescue_candidates,
        prefix_k=95,
        max_rescue=config.bounded_negative_temporal_rescue_tail_budget,
    )

    admitted_tuples = [
        p for p in merged_predictions
        if str(p.get("slot_source", "")).startswith("RESCUE_TAIL")
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
