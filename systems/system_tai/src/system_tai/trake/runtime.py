from __future__ import annotations

import dataclasses
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from system_tai.features.query_encoder import SharedOpenAIClipEncoder
from system_tai.preliminary.schemas import TRAKEPrediction
from system_tai.preliminary.validation import validate_ranked_top100
from system_tai.refinement.engine import ExactFrameRefiner
from system_tai.refinement.models import (
    Phase3Candidate,
    RefinementConfig,
    RefinementQuery,
    RefinementStatus,
)
from system_tai.retrieval.multi_query import (
    QueryLanguage,
    QueryVariant,
    QueryVariantType,
    WeightedRRFRetriever,
)
from system_tai.retrieval.vector_search import ExactNumpyRetriever

from .engine import TRAKEEngine
from .models import TRAKEEvent, TRAKEEventCandidate, TRAKEQuery, TRAKEResult


@dataclass
class TRAKERuntimeTimings:
    total_seconds: float = 0.0
    text_encode_seconds: float = 0.0
    event_retrieval_seconds: float = 0.0
    event_fusion_seconds: float = 0.0
    planning_seconds: float = 0.0
    refinement_seconds: float = 0.0
    finalization_seconds: float = 0.0
    validation_seconds: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "total_seconds": self.total_seconds,
            "text_encode_seconds": self.text_encode_seconds,
            "event_retrieval_seconds": self.event_retrieval_seconds,
            "event_fusion_seconds": self.event_fusion_seconds,
            "planning_seconds": self.planning_seconds,
            "refinement_seconds": self.refinement_seconds,
            "finalization_seconds": self.finalization_seconds,
            "validation_seconds": self.validation_seconds,
        }


class TRAKERuntimePipeline:
    """Shared-runtime TRAKE pipeline adapter reusing exact retrieval and refiner."""

    def __init__(
        self,
        *,
        exact_retriever: ExactNumpyRetriever,
        weighted_rrf: WeightedRRFRetriever,
        refiner: ExactFrameRefiner,
        shared_encoder: SharedOpenAIClipEncoder,
        trake_engine: TRAKEEngine | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.exact_retriever = exact_retriever
        self.weighted_rrf = weighted_rrf
        self.refiner = refiner
        self.shared_encoder = shared_encoder
        self.trake_engine = trake_engine or TRAKEEngine()
        self.clock = clock

    def process_trake_query(
        self,
        request: Any,
        *,
        refinement_config: RefinementConfig,
        rrf_constant: float = 60.0,
    ) -> tuple[TRAKEResult, TRAKERuntimeTimings, dict[str, Any]]:
        t_total_start = self.clock()
        timings = TRAKERuntimeTimings()

        # 1. Build domain events and query variants
        domain_events: list[TRAKEEvent] = []
        flattened_variants: list[tuple[int, QueryVariant]] = []

        for idx, ev in enumerate(request.events):
            desc = ev["description"].strip()
            desc_en = ev.get("description_en")
            if desc_en is not None:
                desc_en = desc_en.strip()

            domain_ev = TRAKEEvent(
                event_index=idx,
                description=desc,
                description_en=desc_en,
            )
            domain_events.append(domain_ev)

            # VI variant
            v_vi = QueryVariant(
                variant_id=f"{request.query_id}::e{idx}::v1_vi",
                text=desc,
                language=QueryLanguage.VIETNAMESE,
                variant_type=QueryVariantType.VIETNAMESE_DIRECT,
                weight=1.0,
            )
            flattened_variants.append((idx, v_vi))

            # Optional EN variant
            if desc_en:
                v_en = QueryVariant(
                    variant_id=f"{request.query_id}::e{idx}::v2_en",
                    text=desc_en,
                    language=QueryLanguage.ENGLISH,
                    variant_type=QueryVariantType.ENGLISH_TRANSLATION,
                    weight=1.0,
                )
                flattened_variants.append((idx, v_en))

        trake_query = TRAKEQuery(query_id=request.query_id, events=tuple(domain_events))

        # 2. Batched Text Encode ONCE across all event variants
        t0 = self.clock()
        variant_texts = [var.text for _, var in flattened_variants]
        encoded_vectors = self.shared_encoder.encode_texts(variant_texts)
        timings.text_encode_seconds = self.clock() - t0

        if len(encoded_vectors) != len(flattened_variants):
            raise ValueError(
                f"Encoded vector count ({len(encoded_vectors)}) "
                f"!= variant count ({len(flattened_variants)})"
            )

        variant_vector_map: dict[str, np.ndarray] = {}
        event_variants_map: dict[int, list[QueryVariant]] = {
            i: [] for i in range(len(domain_events))
        }
        all_variant_ids: list[str] = []
        all_variant_vectors: list[np.ndarray] = []

        for (idx, var), vec in zip(flattened_variants, encoded_vectors):
            variant_vector_map[var.variant_id] = vec
            event_variants_map[idx].append(var)
            all_variant_ids.append(var.variant_id)
            all_variant_vectors.append(vec)

        # 3. Multi-Vector Retrieval & Per-Event Fusion
        event_candidate_pools: list[list[TRAKEEventCandidate]] = []
        total_fusion_sec = 0.0

        t_r = self.clock()
        all_search_results = self.exact_retriever.search_vectors(
            query_ids=all_variant_ids,
            query_vectors=all_variant_vectors,
            top_k=request.top_k_per_variant,
        )
        timings.event_retrieval_seconds = self.clock() - t_r

        for e_idx in range(len(domain_events)):
            e_vars = event_variants_map[e_idx]
            variant_search_results: dict[str, Any] = {
                var.variant_id: all_search_results[var.variant_id] for var in e_vars
            }

            t_f = self.clock()
            fused_event_result = self.weighted_rrf.fuse_rankings(
                query_id=f"{request.query_id}::e{e_idx}",
                variants=tuple(e_vars),
                rankings=variant_search_results,
                output_top_k=request.event_candidate_top_k,
                rrf_constant=rrf_constant,
            )
            total_fusion_sec += self.clock() - t_f

            e_pool: list[TRAKEEventCandidate] = []
            for fused_cand in fused_event_result.ranked_candidates:
                meta = fused_cand.diagnostic_metadata or {}
                tc = TRAKEEventCandidate(
                    query_id=request.query_id,
                    event_index=e_idx,
                    rank=fused_cand.rank,
                    video_id=fused_cand.video_id,
                    frame_id=fused_cand.frame_id,
                    retrieval_score=fused_cand.score,
                    provenance={
                        "fusion_score": fused_cand.score,
                        "variant_hit_count": meta.get("variant_hit_count"),
                        "best_individual_rank": meta.get("best_individual_rank"),
                        "clip_row": fused_cand.clip_row,
                        "keyframe_order": fused_cand.keyframe_order,
                    },
                )
                e_pool.append(tc)
            event_candidate_pools.append(e_pool)

        timings.event_fusion_seconds = total_fusion_sec

        # 4. C1 Planner Integration
        t_p = self.clock()
        c1_result = self.trake_engine.solve_query(
            query=trake_query,
            event_candidates=tuple(event_candidate_pools),
            beam_width=request.beam_width,
            output_top_k=request.output_top_k,
            rrf_constant=rrf_constant,
        )
        timings.planning_seconds = self.clock() - t_p

        # 5. Optional Raw-Video Refinement (refine_top_n)
        t_r = self.clock()
        c1_preds = c1_result.predictions
        refine_count = min(request.refine_top_n, len(c1_preds)) if request.refine_top_n > 0 else 0

        refined_proposals: dict[tuple[int, str, int], int] = {}
        refinement_node_records: list[dict[str, Any]] = []
        frame_embedding_cache: dict[tuple[str, int], np.ndarray] = {}

        if refine_count > 0:
            top_paths = c1_preds[:refine_count]
            for e_idx in range(len(domain_events)):
                unique_nodes_map: dict[tuple[str, int], list[int]] = {}
                for p in top_paths:
                    vid = p.video_id
                    fid = p.frame_ids[e_idx]
                    key = (vid, fid)
                    if key not in unique_nodes_map:
                        unique_nodes_map[key] = []
                    unique_nodes_map[key].append(p.rank)

                sorted_unique_nodes = sorted(
                    unique_nodes_map.keys(),
                    key=lambda k: (min(unique_nodes_map[k]), k[0], k[1]),
                )
                M = len(sorted_unique_nodes)
                if M > 0:
                    p3_candidates: list[Phase3Candidate] = []
                    e_pool = event_candidate_pools[e_idx]
                    e_pool_map = {(c.video_id, c.frame_id): c for c in e_pool}

                    for loc_rank, (vid, fid) in enumerate(sorted_unique_nodes, start=1):
                        cand_match = e_pool_map.get((vid, fid))
                        score = cand_match.retrieval_score if cand_match else 0.0
                        e_rank = cand_match.rank if cand_match else None
                        p3 = Phase3Candidate(
                            query_id=f"{request.query_id}::trake_refine_e{e_idx}",
                            rank=loc_rank,
                            video_id=vid,
                            frame_id=fid,
                            retrieval_score=score,
                            retrieval_provenance={
                                "source_query_id": request.query_id,
                                "event_index": e_idx,
                                "event_candidate_rank": e_rank,
                                "source_path_ranks": unique_nodes_map[(vid, fid)],
                            },
                        )
                        p3_candidates.append(p3)

                    ref_query = RefinementQuery(
                        query_id=f"{request.query_id}::trake_refine_e{e_idx}",
                        variants=tuple(event_variants_map[e_idx]),
                        candidates=tuple(p3_candidates),
                    )

                    e_variant_vectors = np.stack(
                        [variant_vector_map[v.variant_id] for v in event_variants_map[e_idx]]
                    ).astype(np.float32)

                    exec_config = dataclasses.replace(
                        refinement_config,
                        top_candidates_to_refine=M,
                        output_top_k=M,
                    )

                    outcome = self.refiner.refine_query(
                        ref_query,
                        config=exec_config,
                        precomputed_text_embeddings=e_variant_vectors,
                        frame_embedding_cache=frame_embedding_cache,
                    )

                    for p3, ref_cand in zip(p3_candidates, outcome.candidates):
                        prop_fid = p3.frame_id
                        if (
                            ref_cand.status == RefinementStatus.REFINED
                            and ref_cand.refined_frame_id is not None
                        ):
                            prop_fid = ref_cand.refined_frame_id

                        refined_proposals[(e_idx, p3.video_id, p3.frame_id)] = prop_fid
                        refinement_node_records.append({
                            "event_index": e_idx,
                            "video_id": p3.video_id,
                            "original_frame_id": p3.frame_id,
                            "proposed_frame_id": prop_fid,
                            "refinement_status": (
                                ref_cand.status.value
                                if hasattr(ref_cand.status, "value")
                                else str(ref_cand.status)
                            ),
                            "local_refinement_rank": p3.rank,
                            "source_path_ranks": unique_nodes_map[(p3.video_id, p3.frame_id)],
                            "warnings": ref_cand.warnings,
                            "failure_reason": ref_cand.failure_reason,
                        })

        timings.refinement_seconds = self.clock() - t_r

        # 6. Temporal Safety Rule & Path Finalization
        t_fin = self.clock()
        final_predictions: list[TRAKEPrediction] = []
        seen_final_keys: set[tuple[str, tuple[int, ...]]] = set()

        path_diagnostics: list[dict[str, Any]] = []
        temporal_fallback_path_count = 0
        duplicate_fallback_count = 0
        skipped_duplicate_path_count = 0

        for path_idx, c1_pred in enumerate(c1_preds):
            vid = c1_pred.video_id
            orig_frames = c1_pred.frame_ids
            is_top_n = path_idx < refine_count

            if is_top_n:
                prop_frames = tuple(
                    refined_proposals.get((e, vid, orig_frames[e]), orig_frames[e])
                    for e in range(len(domain_events))
                )
                is_order_valid = all(
                    prop_frames[i] <= prop_frames[i + 1] for i in range(len(domain_events) - 1)
                )
                if is_order_valid:
                    option_a = prop_frames
                    option_b = orig_frames if prop_frames != orig_frames else None
                    fallback_reason = None
                else:
                    option_a = orig_frames
                    option_b = None
                    fallback_reason = "refinement_temporal_order_violation"
                    temporal_fallback_path_count += 1
            else:
                option_a = orig_frames
                option_b = None
                fallback_reason = None

            chosen_frames = None
            dupe_decision = "emitted_option_a"

            key_a = (vid, option_a)
            if key_a not in seen_final_keys:
                chosen_frames = option_a
                seen_final_keys.add(key_a)
            elif option_b is not None:
                duplicate_fallback_count += 1
                key_b = (vid, option_b)
                if key_b not in seen_final_keys:
                    chosen_frames = option_b
                    seen_final_keys.add(key_b)
                    dupe_decision = "emitted_option_b_fallback"
                else:
                    skipped_duplicate_path_count += 1
                    dupe_decision = "skipped_both_options_collided"
            else:
                skipped_duplicate_path_count += 1
                dupe_decision = "skipped_option_a_collided"

            if chosen_frames is not None:
                final_pred = TRAKEPrediction(
                    query_id=c1_pred.query_id,
                    rank=c1_pred.rank,
                    video_id=vid,
                    frame_ids=chosen_frames,
                )
                final_predictions.append(final_pred)

            path_diagnostics.append({
                "c1_rank": c1_pred.rank,
                "video_id": vid,
                "original_frame_ids": list(orig_frames),
                "proposed_frame_ids": list(option_a) if is_top_n else None,
                "chosen_frame_ids": list(chosen_frames) if chosen_frames is not None else None,
                "temporal_fallback_reason": fallback_reason,
                "duplicate_decision": dupe_decision,
            })

        timings.finalization_seconds = self.clock() - t_fin

        # 7. Final Validation Gate
        t_v = self.clock()
        for pred in final_predictions:
            if pred.query_id != request.query_id:
                raise ValueError(f"Prediction query_id mismatch: {pred.query_id}")
            if len(pred.frame_ids) != len(domain_events):
                raise ValueError("Prediction frame_ids count != event count")
            for f in pred.frame_ids:
                if type(f) is not int or f < 0:
                    raise ValueError(f"Invalid frame_id in prediction: {f}")
            for i in range(len(pred.frame_ids) - 1):
                if pred.frame_ids[i] > pred.frame_ids[i + 1]:
                    raise ValueError(
                        f"Prediction frames non-decreasing violation: {pred.frame_ids}"
                    )

        val_errors = validate_ranked_top100(
            final_predictions,
            expected_task="trake",
            expected_query_id=request.query_id,
        )
        if val_errors:
            raise ValueError(f"TRAKE prediction validation failed: {val_errors}")

        timings.validation_seconds = self.clock() - t_v
        timings.total_seconds = self.clock() - t_total_start

        # 8. Diagnostics & Result Construction
        result = TRAKEResult(
            query_id=request.query_id,
            event_count=len(domain_events),
            predictions=tuple(final_predictions),
            diagnostics={
                "query_id": request.query_id,
                "event_count": len(domain_events),
                "flattened_variant_count": len(flattened_variants),
                "event_candidate_counts": [len(p) for p in event_candidate_pools],
                "planner_prediction_count": len(c1_preds),
                "refinement_requested": request.refine_top_n > 0,
                "refine_top_n": request.refine_top_n,
                "selected_path_count": refine_count,
                "unique_refinement_node_count": len(refinement_node_records),
                "temporal_fallback_path_count": temporal_fallback_path_count,
                "duplicate_fallback_count": duplicate_fallback_count,
                "skipped_duplicate_path_count": skipped_duplicate_path_count,
                "final_prediction_count": len(final_predictions),
                "zero_output_reason": c1_result.diagnostics.get("zero_output_reason"),
            },
        )

        extra_diagnostics = {
            "c1_diagnostics": c1_result.diagnostics,
            "refinement_node_records": refinement_node_records,
            "path_diagnostics": path_diagnostics,
            "event_candidate_pools": event_candidate_pools,
            "frame_embedding_cache_entry_count": len(frame_embedding_cache),
            "flattened_variants": [
                {
                    "event_index": idx,
                    "variant_id": var.variant_id,
                    "text": var.text,
                    "language": var.language.value,
                    "variant_type": var.variant_type.value,
                    "weight": var.weight,
                }
                for idx, var in flattened_variants
            ],
        }

        return result, timings, extra_diagnostics
