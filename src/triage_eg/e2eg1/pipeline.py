"""E2E-G1 composition over frozen E2E-1 retrieval, T3, M1, and QA resources."""

from __future__ import annotations

from time import monotonic
from typing import Any

import numpy as np

from triage_eg.e2e1.contracts import QueryPlan
from triage_eg.e2e1.pipeline import (
    CanonicalTriagePipeline,
    PredictionResult,
    _renumber,
    _strictly_increasing,
)
from triage_eg.e2e1.planning import plan_query
from triage_eg.e2e1.qa import (
    VOCABULARIES,
    dynamic_object_candidates,
    numeric_tokens,
    score_answers,
)

from .contracts import VARIANTS, E2EG1Settings
from .ranking import candidate_key, coverage_order, g0_order, safe_alternative_order


def is_opaque_machine_id(value: str) -> bool:
    text = str(value).strip().casefold()
    if text.startswith("/m/") and len(text) > 3:
        return True
    return bool(text and text[0].isdigit() and len(text) >= 4 and text.replace("_", "").isalnum())


def filter_machine_ids(
    names: list[str] | tuple[str, ...],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    allowed, filtered = [], []
    for raw in names:
        text = str(raw).strip()
        target = filtered if is_opaque_machine_id(text) else allowed
        if text and text not in target:
            target.append(text)
    return tuple(allowed), tuple(filtered)


class _CountingImageEncoder:
    def __init__(self, delegate: Any, owner: SafeCoveragePipeline) -> None:
        self.delegate = delegate
        self.owner = owner

    def encode(self, frames: list[Any]) -> np.ndarray:
        self.owner.raw_decoded_frames += len(frames)
        self.owner.raw_clip_encode_count += len(frames)
        return self.delegate.encode(frames)


class SafeCoveragePipeline(CanonicalTriagePipeline):
    """Keep E2E-1 evidence intact while allocating safe alternative hypotheses."""

    def __init__(
        self,
        runtime: Any,
        dataset_root: str | Any,
        *,
        settings: E2EG1Settings | None = None,
        decoder_factory: Any = None,
        ocr: Any = None,
    ) -> None:
        kwargs: dict[str, Any] = {"settings": settings or E2EG1Settings()}
        if decoder_factory is not None:
            kwargs["decoder_factory"] = decoder_factory
        if ocr is not None:
            kwargs["ocr"] = ocr
        super().__init__(runtime, dataset_root, **kwargs)
        self.m1_call_count = 0
        self.raw_decoded_frames = 0
        self.raw_clip_encode_count = 0
        self.refined_alternative_count = 0
        self.refined_duplicate_dropped_count = 0
        self.refined_order_invalid_dropped_count = 0
        self.machine_ids_filtered = 0
        self.image_encoder = _CountingImageEncoder(self.image_encoder, self)

    def _refine(
        self, video_id: str, frame_id: int, text: str, text_embedding: np.ndarray
    ) -> dict[str, Any]:
        self.m1_call_count += 1
        return super()._refine(video_id, frame_id, text, text_embedding)

    def _decode_image(self, video_id: str, frame_id: int) -> np.ndarray:
        self.raw_decoded_frames += 1
        return super()._decode_image(video_id, frame_id)

    def _frame_embedding(self, row: dict[str, Any]) -> np.ndarray:
        key = str(row["video_id"]), int(row["frame_id"])
        needs_raw_encode = key not in self._frame_embedding_cache and int(row["frame_id"]) != int(
            row["coarse_frame_id"]
        )
        vector = super()._frame_embedding(row)
        if needs_raw_encode:
            self.raw_clip_encode_count += 1
        return vector

    @staticmethod
    def _coarse_row(source: dict[str, Any], encoding: dict[str, Any]) -> dict[str, Any]:
        frame_id = int(source["original_frame_idx"])
        return {
            **source,
            "coarse_rank": int(source["g0_rank"]),
            "coarse_frame_id": frame_id,
            "frame_id": frame_id,
            "m1_applied": False,
            "hypothesis_kind": "COARSE",
            "encoding": encoding,
        }

    def _ground_single(
        self, plan: QueryPlan, variant: str
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        vector, scores, encoding = self._scores(
            plan.grounding_text, plan.language, f"{plan.query_id}__grounding"
        )
        pool = self._single_event_pool(plan, vector, scores)
        if variant == "G0_E2E1_COARSE":
            allocation, video_ranking = g0_order(pool, self.settings)
        else:
            allocation, video_ranking = coverage_order(pool, self.settings)
        rows = [self._coarse_row(row, encoding) for row in allocation]
        diagnostics: list[dict[str, Any]] = [
            {
                "diagnostic_type": "video_hypothesis_ranking",
                "query_id": plan.query_id,
                "variant": variant,
                **row,
            }
            for row in video_ranking
        ]
        alternatives: dict[tuple[str, int], dict[str, Any]] = {}
        refinement_diagnostics: list[dict[str, Any]] = []
        if variant == "G2_SAFE_M1":
            selected = rows[: self.settings.m1_single_event_budget]
            selected_keys = {candidate_key(row) for row in selected}
            rows = [
                {**row, "was_selected_for_m1": candidate_key(row) in selected_keys} for row in rows
            ]
            for source in selected:
                source_key = candidate_key(source)
                refinement = self._refine(
                    str(source["video_id"]),
                    int(source["coarse_frame_id"]),
                    plan.grounding_text,
                    vector,
                )
                refined_frame = int(refinement["refined_frame_idx"])
                emitted = refined_frame != int(source["coarse_frame_id"])
                if emitted:
                    alternatives[source_key] = {
                        **source,
                        "original_frame_idx": refined_frame,
                        "frame_id": refined_frame,
                        "m1_applied": True,
                        "hypothesis_kind": "M1_REFINED_ALTERNATIVE",
                        "source_coarse_video_id": str(source["video_id"]),
                        "source_coarse_frame_id": int(source["coarse_frame_id"]),
                        "source_coarse_rank": int(source["coverage_rank"]),
                        "refined_frame_id": refined_frame,
                        "m1_shift_frames": refined_frame - int(source["coarse_frame_id"]),
                        "m1_score": refinement.get("refined_score"),
                    }
                refinement_diagnostics.append(
                    {
                        "diagnostic_type": "m1_alternative_provenance",
                        "query_id": plan.query_id,
                        "variant": variant,
                        "task": plan.task,
                        "source_coarse_video_id": str(source["video_id"]),
                        "source_coarse_frame_id": int(source["coarse_frame_id"]),
                        "source_coarse_rank": int(source["coverage_rank"]),
                        "refined_frame_id": refined_frame,
                        "m1_shift_frames": refined_frame - int(source["coarse_frame_id"]),
                        "m1_score": refinement.get("refined_score"),
                        "hypothesis_kind": "M1_REFINED_ALTERNATIVE",
                        "emitted_before_dedup": emitted,
                        **refinement,
                    }
                )
            rows, safety = safe_alternative_order(rows, alternatives, self.settings)
            self.refined_duplicate_dropped_count += safety["refined_duplicate_dropped_count"]
            emitted_alternative_keys = {
                (str(row["video_id"]), int(row["frame_id"]))
                for row in rows
                if row["hypothesis_kind"] == "M1_REFINED_ALTERNATIVE"
            }
            refinement_diagnostics = [
                {
                    **row,
                    "emitted_after_dedup": (
                        str(row["source_coarse_video_id"]),
                        int(row["refined_frame_id"]),
                    )
                    in emitted_alternative_keys,
                }
                for row in refinement_diagnostics
            ]
            emitted_alternatives = sum(
                row["hypothesis_kind"] == "M1_REFINED_ALTERNATIVE" for row in rows
            )
            self.refined_alternative_count += emitted_alternatives
        for final_rank, row in enumerate(rows, 1):
            row["final_hypothesis_rank"] = final_rank
        diagnostics.extend(
            {
                "diagnostic_type": "coverage_allocation",
                "query_id": plan.query_id,
                "task": plan.task,
                "variant": variant,
                **{key: value for key, value in row.items() if key != "encoding"},
            }
            for row in rows
            if row["hypothesis_kind"] == "COARSE"
        )
        diagnostics.extend(refinement_diagnostics)
        return rows, diagnostics

    def _answer_row(
        self, plan: QueryPlan, row: dict[str, Any], intent: str, rank: int
    ) -> tuple[str, dict[str, Any]]:
        language = plan.language if plan.language in {"vi", "en"} else "en"
        diagnostic: dict[str, Any] = {
            "diagnostic_type": "qa_machine_id_hygiene",
            "intent": intent,
            "answer_fallback_reason": None,
        }
        if intent in {"OCR_TEXT", "OCR_NUMERIC"} and rank <= self.settings.ocr_max_grounding_ranks:
            if self.ocr.status == "AVAILABLE":
                tokens = self.ocr.read(self._decode_image(row["video_id"], int(row["frame_id"])))
                values = [item["text"] for item in tokens]
                if intent == "OCR_NUMERIC":
                    values = [token for value in values for token in numeric_tokens(value)]
                if values:
                    diagnostic.update({"ocr_status": "AVAILABLE", "ocr_tokens": values[:10]})
                    return values[0], diagnostic
                diagnostic["answer_fallback_reason"] = "OCR_NO_TEXT"
            else:
                diagnostic["answer_fallback_reason"] = "OCR_UNAVAILABLE"
        raw_names, metadata_frame = self._object_names(row["video_id"], int(row["frame_id"]))
        names, filtered = filter_machine_ids(raw_names)
        self.machine_ids_filtered += len(filtered)
        candidates = VOCABULARIES.get(intent, ())
        if intent in {
            "OBJECT",
            "VEHICLE",
            "ANIMAL",
            "CONTAINER",
            "GARMENT",
            "GENERIC_VISUAL",
            "OCR_TEXT",
            "OCR_NUMERIC",
        }:
            dynamic = dynamic_object_candidates(list(names))
            base = candidates or VOCABULARIES["OBJECT"]
            candidates = tuple(dict.fromkeys((*dynamic, *base)))
        diagnostic.update(
            {
                "OBJECT_MACHINE_ID_FILTERED": bool(filtered),
                "filtered_machine_ids": list(filtered[:20]),
                "raw_object_class_evidence": list(raw_names[:20]),
                "human_readable_object_candidates": list(names[:20]),
                "QA_HYGIENE_DELTA": bool(filtered),
            }
        )
        if not candidates:
            answer = "không xác định" if language == "vi" else "unknown"
            diagnostic["answer_fallback_reason"] = (
                diagnostic["answer_fallback_reason"] or "NO_CANDIDATES"
            )
            return answer, diagnostic
        image = self._frame_embedding(row)
        embeddings = self._answer_embeddings(intent, candidates)
        answer, score, margin = score_answers(image, embeddings, candidates, language)
        diagnostic.update(
            {
                "answer_top1_score": score,
                "answer_top1_margin": margin,
                "object_metadata_frame_idx": metadata_frame,
                "object_class_candidates": list(names[:20]),
                "bbox_relations_used": False,
            }
        )
        return answer or ("không xác định" if language == "vi" else "unknown"), diagnostic

    def _coarse_trake_records(
        self, plan: QueryPlan
    ) -> tuple[list[dict[str, Any]], list[np.ndarray], list[dict[str, Any]]]:
        chains, embeddings, encodings = self._trake_chains(plan)
        records, seen = [], set()
        for coarse_rank, chain in enumerate(chains, 1):
            frames = tuple(int(value) for value in chain["frame_ids"])
            if len(frames) != len(plan.events) or not _strictly_increasing(frames):
                raise RuntimeError("TRAKE_OUTPUT_STRUCTURALLY_INVALID")
            key = str(chain["video_id"]), frames
            if key in seen:
                continue
            seen.add(key)
            records.append(
                {
                    "query_id": plan.query_id,
                    "video_id": str(chain["video_id"]),
                    "frame_ids": list(frames),
                    "coarse_rank": coarse_rank,
                    "hypothesis_kind": "COARSE",
                    "chain": chain,
                }
            )
            if len(records) == self.settings.max_predictions:
                break
        return records, embeddings, encodings

    def predict_trake(self, plan: QueryPlan, variant: str) -> PredictionResult:
        if variant in {"G0_E2E1_COARSE", "G1_COVERAGE_COARSE"}:
            result = super().predict_trake(plan, "P0_COARSE")
            diagnostics = tuple(
                {
                    **row,
                    "diagnostic_type": "trake_dual_hypothesis",
                    "variant": variant,
                    "hypothesis_kind": "COARSE",
                }
                for row in result.diagnostics
            )
            return PredictionResult(
                result.query_plan, result.predictions, diagnostics, result.latency_seconds
            )

        started = monotonic()
        coarse, embeddings, encodings = self._coarse_trake_records(plan)
        protected = coarse[: self.settings.m1_trake_source_chains]
        coarse_keys = {
            (str(row["video_id"]), tuple(int(value) for value in row["frame_ids"]))
            for row in coarse
        }
        alternative_keys: set[tuple[str, tuple[int, ...]]] = set()
        alternatives, diagnostics = [], []
        for source in protected:
            refinements = [
                self._refine(
                    str(source["video_id"]),
                    int(source["frame_ids"][event_index]),
                    text,
                    embeddings[event_index],
                )
                for event_index, (_, text) in enumerate(plan.events)
            ]
            refined = tuple(int(row["refined_frame_idx"]) for row in refinements)
            source_frames = tuple(int(value) for value in source["frame_ids"])
            valid = len(refined) == len(plan.events) and _strictly_increasing(refined)
            distinct = refined != source_frames
            key = str(source["video_id"]), refined
            emitted = valid and distinct and key not in coarse_keys and key not in alternative_keys
            if emitted:
                alternative_keys.add(key)
                alternatives.append(
                    {
                        "query_id": plan.query_id,
                        "video_id": str(source["video_id"]),
                        "frame_ids": list(refined),
                        "hypothesis_kind": "M1_REFINED_ALTERNATIVE",
                        "source_coarse_rank": int(source["coarse_rank"]),
                        "source_coarse_frame_ids": list(source_frames),
                    }
                )
                self.refined_alternative_count += 1
            elif not valid:
                self.refined_order_invalid_dropped_count += 1
            elif distinct:
                self.refined_duplicate_dropped_count += 1
            diagnostics.append(
                {
                    "diagnostic_type": "trake_dual_hypothesis",
                    "query_id": plan.query_id,
                    "variant": variant,
                    "video_id": str(source["video_id"]),
                    "source_coarse_rank": int(source["coarse_rank"]),
                    "source_coarse_frame_ids": list(source_frames),
                    "refined_frame_ids": list(refined),
                    "hypothesis_kind": "M1_REFINED_ALTERNATIVE",
                    "refined_order_valid": valid,
                    "refined_distinct": distinct,
                    "emitted": emitted,
                    "refinements": refinements,
                    "encodings": encodings,
                }
            )
        ordered = [*protected, *alternatives, *coarse[len(protected) :]]
        ordered = ordered[: self.settings.max_predictions]
        predictions = [
            {
                "query_id": plan.query_id,
                "video_id": row["video_id"],
                "frame_ids": list(row["frame_ids"]),
            }
            for row in ordered
        ]
        expected = [(row["video_id"], tuple(row["frame_ids"])) for row in protected]
        actual = [(row["video_id"], tuple(row["frame_ids"])) for row in ordered[: len(protected)]]
        if actual != expected or any(
            row["hypothesis_kind"] != "COARSE" for row in ordered[: len(protected)]
        ):
            raise RuntimeError("E2EG1_TRAKE_PROTECTED_PREFIX_VIOLATION")
        diagnostics.extend(
            {
                "diagnostic_type": "trake_dual_hypothesis",
                "query_id": plan.query_id,
                "variant": variant,
                "video_id": row["video_id"],
                "coarse_rank": row["coarse_rank"],
                "coarse_frame_ids": list(row["frame_ids"]),
                "hypothesis_kind": "COARSE",
                "source_coarse_retained": True,
                "t3_score": row["chain"]["score"],
                "global_rows": list(row["chain"].get("global_rows", ())),
                "event_scores": list(row["chain"].get("event_scores", ())),
                "event_region_ids": list(row["chain"].get("region_ids", ())),
            }
            for row in coarse
        )
        return PredictionResult(
            plan.as_dict(), tuple(_renumber(predictions)), tuple(diagnostics), monotonic() - started
        )

    def predict_query(self, query: dict[str, Any], variant: str = "G2_SAFE_M1") -> PredictionResult:
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}")
        plan = plan_query(query)
        if plan.task == "KIS":
            return self.predict_kis(plan, variant)
        if plan.task == "QA":
            return self.predict_qa(plan, variant)
        return self.predict_trake(plan, variant)

    def predict_queries(
        self, queries: list[dict[str, Any]], variant: str = "G2_SAFE_M1"
    ) -> list[PredictionResult]:
        return [self.predict_query(query, variant) for query in queries]

    def runtime_diagnostics(self) -> dict[str, Any]:
        return {
            **super().runtime_diagnostics(),
            "m1_call_count": self.m1_call_count,
            "raw_decoded_frames": self.raw_decoded_frames,
            "raw_clip_encode_count": self.raw_clip_encode_count,
            "refined_alternative_count": self.refined_alternative_count,
            "refined_duplicate_dropped_count": self.refined_duplicate_dropped_count,
            "refined_order_invalid_dropped_count": self.refined_order_invalid_dropped_count,
            "qa_machine_ids_filtered": self.machine_ids_filtered,
        }


__all__ = [
    "SafeCoveragePipeline",
    "filter_machine_ids",
    "is_opaque_machine_id",
]
