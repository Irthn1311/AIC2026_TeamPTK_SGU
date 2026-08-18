# ==============================================================================================================
# Sprint 2B.1 Consensus Novel Video Rescue Orchestrator (QA Layer)
# ==============================================================================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

from system_tai.qa.grounding import (
    QAVideoConditionedEvidenceConfig,
    QAVideoNomination,
    build_qa_grounding_result,
    nominate_qa_videos,
    select_temporal_seed_anchors,
)
from system_tai.qa.models import (
    QAEvidenceCandidate,
    QAQuery,
)
from system_tai.qa.question_types import QuestionType
from system_tai.qa.rescue_tail import RescueCandidate, merge_rescue_tail
from system_tai.refinement.models import (
    Phase3Candidate,
    RefinedCandidate,
    RefinementConfig,
    RefinementStatus,
)
from system_tai.refinement.video import DecodeRequest
from system_tai.retrieval.multi_query import (
    QueryLanguage,
    QueryVariant,
    QueryVariantType,
    WeightedRRFRetriever,
)
from system_tai.retrieval.query_decomposition import decompose_query

if TYPE_CHECKING:
    from system_tai.features.query_encoder import SharedOpenAIClipEncoder
    from system_tai.kis.session_schema import QAQueryRequest
    from system_tai.qa.answer_provider import QAAnswerCandidateProvider
    from system_tai.qa.engine import QAEngine
    from system_tai.qa.ocr_provider import OCRAnswerProvider
    from system_tai.qa.raw_registry import RawVideoRegistry
    from system_tai.refinement.engine import ExactFrameRefiner
    from system_tai.refinement.video import VideoDecoder
    from system_tai.retrieval.video_evidence import VideoRestrictedFeatureSearcher


@dataclass(frozen=True, slots=True)
class SidepathRefinedFrame:
    candidate_frame_id: int
    refined_frame_id: int | None
    status: RefinementStatus
    refinement_fusion_score: float | None = None


@dataclass(frozen=True, slots=True)
class ConsensusRescueCandidate:
    video_id: str
    fused_rank: int
    literal_rank: int
    compact_rank: int


@dataclass(frozen=True, slots=True)
class ConsensusNovelRescueOutcome:
    eligible: bool
    reason: str
    literal_top16: tuple[str, ...] = ()
    compact_top16: tuple[str, ...] = ()
    fused_top16: tuple[str, ...] = ()
    all_consensus_candidates: tuple[ConsensusRescueCandidate, ...] = ()
    chosen_candidate: ConsensusRescueCandidate | None = None
    literal_variant_text: str = ""
    compact_variant_text: str = ""


def derive_consensus_novel_videos(
    *,
    query_id: str,
    query_text_vi: str,
    query_text_en: str | None,
    champion_selected_video_ids: Sequence[str],
    searcher: VideoRestrictedFeatureSearcher,
    encoder: SharedOpenAIClipEncoder,
    config: QAVideoConditionedEvidenceConfig,
    per_query_cap: int = 16,
) -> ConsensusNovelRescueOutcome:
    """
    Identifies novel consensus videos absent from Champion Top 16 but present
    in both literal and compact_keywords Top 16 rankings and fused Top 16.

    Reproduces the exact validated Task-2 semantics:
    - literal weight = 1.0
    - compact_keywords weight = 0.8
    - nominate_qa_videos([single_variant]) for individual pools
    - nominate_qa_videos([literal, compact]) for fused pool
    - fail closed if either required variant is missing
    """
    variants_obj = decompose_query(
        query_text_vi=query_text_vi,
        query_text_en=query_text_en,
    )
    decomp_map = dict(variants_obj.as_list())

    lit_text = decomp_map.get("literal", "").strip()
    cmp_text = decomp_map.get("compact_keywords", "").strip()

    if not lit_text or not cmp_text:
        return ConsensusNovelRescueOutcome(
            eligible=False,
            reason="REQUIRED_VARIANT_MISSING",
            literal_variant_text=lit_text,
            compact_variant_text=cmp_text,
        )

    # Build exact Task-2 query variants
    lit_lang = (
        QueryLanguage.VIETNAMESE
        if not query_text_en
        else QueryLanguage.ENGLISH
    )
    lit_type = (
        QueryVariantType.VIETNAMESE_DIRECT
        if lit_lang == QueryLanguage.VIETNAMESE
        else QueryVariantType.ENGLISH_TRANSLATION
    )
    cmp_lang = QueryLanguage.ENGLISH if query_text_en else QueryLanguage.VIETNAMESE
    cmp_type = (
        QueryVariantType.ENGLISH_TRANSLATION
        if cmp_lang == QueryLanguage.ENGLISH
        else QueryVariantType.VIETNAMESE_DIRECT
    )

    v_lit = QueryVariant(
        variant_id=f"{query_id}::literal",
        text=lit_text,
        language=lit_lang,
        variant_type=lit_type,
        weight=1.0,
    )
    v_cmp = QueryVariant(
        variant_id=f"{query_id}::compact_keywords",
        text=cmp_text,
        language=cmp_lang,
        variant_type=cmp_type,
        weight=0.8,
    )

    vec_lit = encoder.encode(lit_text)
    vec_cmp = encoder.encode(cmp_text)

    # 1. Search Individual Literal Variant
    maxima_lit = searcher.search_video_maxima(
        query_ids=[v_lit.variant_id],
        query_vectors=[vec_lit],
    )
    noms_lit = nominate_qa_videos(
        variants=[v_lit],
        maxima=maxima_lit,
        config=config,
    )
    literal_top16 = tuple(n.video_id for n in noms_lit[:per_query_cap])

    # 2. Search Individual Compact Variant
    maxima_cmp = searcher.search_video_maxima(
        query_ids=[v_cmp.variant_id],
        query_vectors=[vec_cmp],
    )
    noms_cmp = nominate_qa_videos(
        variants=[v_cmp],
        maxima=maxima_cmp,
        config=config,
    )
    compact_top16 = tuple(n.video_id for n in noms_cmp[:per_query_cap])

    # 3. Fused Multi-Variant Nomination
    maxima_fused = searcher.search_video_maxima(
        query_ids=[v_lit.variant_id, v_cmp.variant_id],
        query_vectors=[vec_lit, vec_cmp],
    )
    noms_fused = nominate_qa_videos(
        variants=[v_lit, v_cmp],
        maxima=maxima_fused,
        config=config,
    )
    fused_top16 = tuple(n.video_id for n in noms_fused[:per_query_cap])

    # 4. Consensus Novel Candidate Filtering
    champ_set = set(champion_selected_video_ids)
    lit_set = set(literal_top16)
    cmp_set = set(compact_top16)

    consensus_list: list[ConsensusRescueCandidate] = []
    for f_rank, vid in enumerate(fused_top16, start=1):
        if vid not in champ_set and vid in lit_set and vid in cmp_set:
            l_rank = literal_top16.index(vid) + 1
            c_rank = compact_top16.index(vid) + 1
            consensus_list.append(
                ConsensusRescueCandidate(
                    video_id=vid,
                    fused_rank=f_rank,
                    literal_rank=l_rank,
                    compact_rank=c_rank,
                )
            )

    # Sort deterministically by (fused_rank, video_id)
    sorted_consensus = tuple(
        sorted(consensus_list, key=lambda c: (c.fused_rank, c.video_id))
    )

    if not sorted_consensus:
        return ConsensusNovelRescueOutcome(
            eligible=False,
            reason="NO_CONSENSUS_CANDIDATE_FOUND",
            literal_top16=literal_top16,
            compact_top16=compact_top16,
            fused_top16=fused_top16,
            literal_variant_text=lit_text,
            compact_variant_text=cmp_text,
        )

    chosen = sorted_consensus[0]
    return ConsensusNovelRescueOutcome(
        eligible=True,
        reason="CONSENSUS_CANDIDATE_SELECTED",
        literal_top16=literal_top16,
        compact_top16=compact_top16,
        fused_top16=fused_top16,
        all_consensus_candidates=sorted_consensus,
        chosen_candidate=chosen,
        literal_variant_text=lit_text,
        compact_variant_text=cmp_text,
    )


def execute_sidepath_consensus_rescue(
    *,
    request: QAQueryRequest,
    q_type: QuestionType,
    variants: Sequence[QueryVariant],
    event_vectors: Sequence[np.ndarray],
    champion_selected_video_ids: Sequence[str],
    champion_predictions: Sequence[dict[str, Any]],
    searcher: VideoRestrictedFeatureSearcher,
    encoder: SharedOpenAIClipEncoder,
    decoder: VideoDecoder,
    refiner: ExactFrameRefiner,
    refinement_config: RefinementConfig,
    weighted_rrf: WeightedRRFRetriever,
    raw_video_registry: RawVideoRegistry,
    candidate_provider: QAAnswerCandidateProvider | None,
    ocr_answer_provider: OCRAnswerProvider | None,
    qa_engine: QAEngine,
    config: QAVideoConditionedEvidenceConfig,
) -> tuple[
    list[dict[str, Any]],
    ConsensusNovelRescueOutcome,
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    """
    Executes isolated side-path processing on the single chosen consensus rescued video.
    Reuses the EXACT canonical downstream helpers:
      1. search_selected_videos
      2. build_qa_grounding_result
      3. select_temporal_seed_anchors
      4. refiner.refine_selected_candidates
      5. VideoDecoder.decode & canonical QAEngine answer scoring
      6. merge_rescue_tail (preserves ranks 1..95 strictly)
    """
    q_vi = request.question
    q_en = request.question_en or request.event_description_en

    outcome = derive_consensus_novel_videos(
        query_id=request.query_id,
        query_text_vi=q_vi,
        query_text_en=q_en,
        champion_selected_video_ids=champion_selected_video_ids,
        searcher=searcher,
        encoder=encoder,
        config=config,
    )

    if not outcome.eligible or outcome.chosen_candidate is None:
        return list(champion_predictions), outcome, [], [], {}

    chosen_vid = outcome.chosen_candidate.video_id
    rescue_evidence_records: list[dict[str, Any]] = []
    rescue_candidates: list[RescueCandidate] = []
    stage_telemetry: dict[str, Any] = {
        "chosen_video_id": chosen_vid,
        "restricted_search": {},
        "grounding": {},
        "temporal_seeds": {},
        "refinement": {},
        "answers": {},
        "rescue_candidates_count": 0,
        "tail": {},
    }

    try:
        store = searcher.registry.get(chosen_vid)
        variant_ids = [v.variant_id for v in variants]
        # Canonical Restricted Frame Search
        restricted = searcher.search_selected_videos(
            video_ids=[chosen_vid],
            query_ids=variant_ids,
            query_vectors=event_vectors,
            per_query_result_cap=store.descriptor.row_count,
        )

        hits_summary = []
        for v_id in variant_ids:
            h_list = restricted.rankings.get(v_id, {}).get(chosen_vid, ())
            hits_summary.append({
                "variant_id": v_id,
                "hit_count": len(h_list),
                "top_frames": [h.frame_id for h in h_list[:3]],
                "top_scores": [float(h.cosine_score) for h in h_list[:3]],
            })
        stage_telemetry["restricted_search"] = {
            "variant_ids": variant_ids,
            "hits_summary": hits_summary,
        }

        nomination = QAVideoNomination(
            video_id=chosen_vid,
            nomination_rank=1,
            video_rrf_score=1.0,
            best_individual_variant_rank=1,
            per_variant=(),
        )

        # Canonical Grounding Candidate Fusion
        fused_grounding = build_qa_grounding_result(
            query_id=request.query_id,
            variants=variants,
            nominations=(nomination,),
            restricted=restricted,
            weighted_rrf=weighted_rrf,
            config=config,
            output_top_k=5,
        )

        stage_telemetry["grounding"] = {
            "candidate_count": len(fused_grounding.ranked_candidates),
            "candidates": [
                {
                    "rank": c.rank,
                    "frame_id": c.frame_id,
                    "score": float(c.score),
                    "local_anchor_rank": (c.diagnostic_metadata or {}).get("local_anchor_rank"),
                }
                for c in fused_grounding.ranked_candidates
            ],
        }

        if not fused_grounding.ranked_candidates:
            return list(champion_predictions), outcome, [], [], stage_telemetry

        # Canonical Temporal Seed Anchor Selection
        seed_anchors = select_temporal_seed_anchors(
            fused_grounding.ranked_candidates,
            anchors_per_video=config.temporal_seed_anchors_per_video,
            video_cap=1,
        )

        stage_telemetry["temporal_seeds"] = {
            "seed_count": len(seed_anchors),
            "seed_frames": [c.frame_id for c in seed_anchors],
        }

        phase3_list: list[Phase3Candidate] = [
            Phase3Candidate(
                query_id=request.query_id,
                rank=candidate.rank,
                video_id=candidate.video_id,
                frame_id=candidate.frame_id,
                retrieval_score=candidate.score,
                retrieval_provenance=dict(candidate.diagnostic_metadata or {}),
            )
            for candidate in seed_anchors
        ]

        # Canonical Temporal Refinement
        refined_list = []
        if config.temporal_refinement_enabled and phase3_list:
            text_vecs = np.asarray(event_vectors, dtype=np.float32)
            ref_outcome = refiner.refine_selected_candidates(
                query_id=request.query_id,
                variants=tuple(variants),
                candidates=tuple(phase3_list),
                config=refinement_config,
                precomputed_text_embeddings=text_vecs,
                frame_embedding_cache={},
            )
            refined_list = list(ref_outcome.candidates)
        else:
            refined_list = [
                SidepathRefinedFrame(
                    candidate_frame_id=c.frame_id,
                    refined_frame_id=c.frame_id,
                    status=RefinementStatus.NOT_REFINED,
                    refinement_fusion_score=c.retrieval_score,
                )
                for c in phase3_list
            ]

        stage_telemetry["refinement"] = {
            "input_count": len(phase3_list),
            "output_count": len(refined_list),
            "refined_candidates": [
                {
                    "candidate_frame_id": r.candidate_frame_id,
                    "refined_frame_id": r.refined_frame_id,
                    "status": r.status.value,
                    "score": r.refinement_fusion_score,
                }
                for r in refined_list
            ],
        }

        # Video Record Lookup for Decoding
        video_record = raw_video_registry.get(chosen_vid)
        if not video_record or not video_record.raw_video_path or not video_record.raw_video_path.is_file():
            return list(champion_predictions), outcome, [], [], stage_telemetry

        probe = decoder.probe(video_record)

        for ref_cand in refined_list:
            cand_frame_id = int(ref_cand.candidate_frame_id)
            effective_frame_id = (
                int(ref_cand.refined_frame_id)
                if ref_cand.status is RefinementStatus.REFINED and ref_cand.refined_frame_id is not None
                else cand_frame_id
            )
            source_status = (
                "TEMPORAL_REFINED"
                if ref_cand.status is RefinementStatus.REFINED
                else "KEYFRAME_ANCHOR"
            )

            dec_req = DecodeRequest(
                probe=probe,
                frame_ids=(effective_frame_id,),
                max_decoded_frames=100,
            )
            dec_res = decoder.decode(dec_req)
            if not dec_res.frames:
                continue

            frame_img = dec_res.frames[0].image
            produced_answers: list[tuple[str, float]] = []

            # Canonical Answer Provider Routing:
            # 1. Try OCR if supported
            if ocr_answer_provider is not None and ocr_answer_provider.supports(q_type):
                ev_cand = QAEvidenceCandidate(
                    query_id=request.query_id,
                    rank=1,
                    video_id=chosen_vid,
                    frame_id=effective_frame_id,
                    retrieval_score=1.0,
                    source_status=source_status,
                )
                ocr_res, _ = ocr_answer_provider.answer(
                    query_id=request.query_id,
                    question_type=q_type,
                    evidence=((ev_cand, frame_img),),
                    output_top_k=5,
                )
                for pred in ocr_res.predictions:
                    if pred.answer:
                        produced_answers.append((pred.answer, 0.5))

            # 2. Canonical QA Engine Answer Scoring (Visual Ontology / Baseline)
            if not produced_answers:
                qa_q = QAQuery(
                    query_id=request.query_id,
                    event_description=q_vi,
                    question=q_vi,
                    event_description_en=q_en,
                    question_en=q_en,
                    question_type=q_type,
                )
                ev_cand = QAEvidenceCandidate(
                    query_id=request.query_id,
                    rank=1,
                    video_id=chosen_vid,
                    frame_id=effective_frame_id,
                    retrieval_score=1.0,
                    source_status=source_status,
                )
                img_vec = encoder.encode_images([frame_img])[0]
                q_res = qa_engine.answer(
                    qa_q,
                    (ev_cand,),
                    image_embeddings={(chosen_vid, effective_frame_id): img_vec},
                    output_top_k=5,
                )
                for pred in q_res.predictions:
                    if pred.answer:
                        produced_answers.append((pred.answer, float(getattr(pred, "score", 0.5) or 0.5)))

            for ans_text, ans_score in produced_answers:
                rescue_evidence_records.append({
                    "video_id": chosen_vid,
                    "candidate_frame_id": cand_frame_id,
                    "refined_frame_id": ref_cand.refined_frame_id,
                    "evidence_frame_id": effective_frame_id,
                    "answer": ans_text,
                    "answer_score": ans_score,
                    "refinement_status": ref_cand.status.value,
                    "evidence_source": source_status,
                })
                rescue_candidates.append(
                    RescueCandidate(
                        video_id=chosen_vid,
                        frame_id=effective_frame_id,
                        answer=ans_text,
                        rescue_score=ans_score,
                        rescue_source="consensus_novel_video_rescue",
                        provenance={
                            "fused_rank": outcome.chosen_candidate.fused_rank,
                            "literal_rank": outcome.chosen_candidate.literal_rank,
                            "compact_rank": outcome.chosen_candidate.compact_rank,
                            "anchor_frame": cand_frame_id,
                            "refined_frame": ref_cand.refined_frame_id,
                            "refinement_status": ref_cand.status.value,
                        },
                    )
                )
        stage_telemetry["answers"] = {
            "produced_count": len(produced_answers),
            "answers": produced_answers,
        }
        stage_telemetry["rescue_candidates_count"] = len(rescue_candidates)
    except Exception as exc:
        import sys, traceback
        print(f"[RESCUE ERROR for {chosen_vid}]: {exc}", file=sys.stderr)
        traceback.print_exc()
        stage_telemetry["error"] = str(exc)

    # Merge into predictions tail (ranks 96..100)
    merged_predictions = merge_rescue_tail(
        champion_predictions=champion_predictions,
        rescue_candidates=rescue_candidates,
        prefix_k=95,
        max_rescue=config.consensus_novel_rescue_tail_budget,
    )

    admitted_tuples = [
        p for p in merged_predictions
        if str(p.get("slot_source", "")).startswith("RESCUE_TAIL")
    ]

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

    return merged_predictions, outcome, rescue_evidence_records, admitted_tuples, stage_telemetry
