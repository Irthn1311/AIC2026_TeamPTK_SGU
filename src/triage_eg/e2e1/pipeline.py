"""Canonical load-once E2E-1 inference pipeline."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Any

import numpy as np

from triage_eg.data.stage0_audit.asset_resolver import discover_layout, resolve_assets
from triage_eg.experiments.moment_m1.runner import (
    M1Settings,
    VerifiedClipLocalImageEncoder,
    refine_local_event,
)
from triage_eg.experiments.reference_rt1.scoring import VideoRows, build_video_row_groups
from triage_eg.experiments.t3_diverse_temporal import (
    DiverseTemporalPath,
    build_diverse_event_pool,
    enumerate_feasible_paths,
    select_coverage_aware,
)
from triage_eg.retrieval.stage2 import OperationalRetrievalRuntime, QueryRequest
from triage_eg.video import OpenCVRawVideoDecoder

from .contracts import VARIANTS, E2E1Settings, QueryPlan
from .planning import plan_query
from .qa import (
    PROMPTS,
    VOCABULARIES,
    OptionalTesseract,
    dynamic_object_candidates,
    numeric_tokens,
    route_intent,
    score_answers,
)


@dataclass(frozen=True)
class PredictionResult:
    query_plan: dict[str, Any]
    predictions: tuple[dict[str, Any], ...]
    diagnostics: tuple[dict[str, Any], ...]
    latency_seconds: float


class _CachedM1ImageEncoder:
    """Capture the exact embeddings M1 already computed for refined raw frames."""

    def __init__(
        self, encoder: VerifiedClipLocalImageEncoder, cache: dict[tuple[str, int], np.ndarray]
    ):
        self.encoder = encoder
        self.cache = cache

    def encode(self, frames: list[Any]) -> np.ndarray:
        matrix = self.encoder.encode(frames)
        for frame, vector in zip(frames, matrix, strict=True):
            self.cache[(frame.video_id, int(frame.actual_frame_idx))] = np.asarray(
                vector, dtype=np.float32
            )
        return matrix


def _strictly_increasing(values: list[int] | tuple[int, ...]) -> bool:
    return all(left < right for left, right in zip(values, values[1:], strict=False))


def _renumber(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{**row, "rank": rank} for rank, row in enumerate(rows, 1)]


class CanonicalTriagePipeline:
    """Integrate frozen Stage2A, T3, and M1 without exposing GT to inference."""

    def __init__(
        self,
        runtime: Any,
        dataset_root: str | Path,
        *,
        settings: E2E1Settings | None = None,
        decoder_factory: Callable[[str, Path], Any] = OpenCVRawVideoDecoder,
        ocr: OptionalTesseract | None = None,
    ) -> None:
        if not getattr(runtime, "loaded", False):
            raise RuntimeError("Stage2 runtime must be loaded once before E2E inference")
        self.runtime = runtime
        self.dataset_root = Path(dataset_root).expanduser().resolve(strict=True)
        self.settings = settings or E2E1Settings()
        self.decoder_factory = decoder_factory
        self.ocr = ocr or OptionalTesseract()
        self.groups = build_video_row_groups(runtime.catalog)
        self.group_by_video = {group.video_id: group for group in self.groups}
        self.video_partitions, self.keyframe_partitions = discover_layout(self.dataset_root)
        self._encoded_text: dict[tuple[str, str], tuple[np.ndarray, dict[str, Any]]] = {}
        self._score_cache: dict[tuple[str, str], np.ndarray] = {}
        self._single_pool_cache: dict[tuple[str, str, str], tuple[dict[str, Any], ...]] = {}
        self._m1_cache: dict[tuple[str, int, str], dict[str, Any]] = {}
        self._frame_embedding_cache: dict[tuple[str, int], np.ndarray] = {}
        self._answer_embedding_cache: dict[tuple[str, tuple[str, ...]], np.ndarray] = {}
        self._object_cache: dict[tuple[str, int], tuple[str, ...]] = {}
        self.image_encoder = _CachedM1ImageEncoder(
            VerifiedClipLocalImageEncoder(runtime.encoder), self._frame_embedding_cache
        )
        self.m1_cache_hits = 0
        self.qa_frame_cache_hits = 0

    @classmethod
    def load_once(
        cls,
        stage2_config: Any,
        dataset_root: str | Path,
        *,
        settings: E2E1Settings | None = None,
        runtime_factory: Callable[[Any], Any] = OperationalRetrievalRuntime,
    ) -> CanonicalTriagePipeline:
        runtime = runtime_factory(stage2_config).load()
        return cls(runtime, dataset_root, settings=settings)

    def _encode(self, text: str, language: str, query_id: str) -> tuple[np.ndarray, dict[str, Any]]:
        key = (language, text)
        if key not in self._encoded_text:
            encoded = self.runtime.encode_requests([QueryRequest(query_id, text, language, 1)])
            self._encoded_text[key] = (
                np.asarray(encoded.embeddings[0], dtype=np.float32),
                dict(encoded.encodings[0]),
            )
        vector, provenance = self._encoded_text[key]
        return vector, dict(provenance)

    def _scores(
        self, text: str, language: str, query_id: str
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        vector, provenance = self._encode(text, language, query_id)
        key = (language, text)
        if key not in self._score_cache:
            values = np.asarray(self.runtime.backend.score_all(vector), dtype=np.float32)
            if values.shape != (self.runtime.backend.size,) or not np.isfinite(values).all():
                raise RuntimeError("E2E_STAGE1_FULL_SCORE_VECTOR_INVALID")
            self._score_cache[key] = values
        return vector, self._score_cache[key], provenance

    def _scores_many(
        self, texts: list[str], language: str, query_id: str
    ) -> tuple[list[np.ndarray], np.ndarray, list[dict[str, Any]]]:
        keys = [(language, text) for text in texts]
        missing = [index for index, key in enumerate(keys) if key not in self._encoded_text]
        if missing:
            requests = [
                QueryRequest(f"{query_id}__{index + 1}", texts[index], language, 1)
                for index in missing
            ]
            encoded = self.runtime.encode_requests(requests)
            for batch_index, source_index in enumerate(missing):
                self._encoded_text[keys[source_index]] = (
                    np.asarray(encoded.embeddings[batch_index], dtype=np.float32),
                    dict(encoded.encodings[batch_index]),
                )
        score_missing = [index for index, key in enumerate(keys) if key not in self._score_cache]
        if score_missing:
            matrix = np.asarray(
                self.runtime.backend.score_many_all(
                    np.stack([self._encoded_text[keys[index]][0] for index in score_missing])
                ),
                dtype=np.float32,
            )
            expected = (len(score_missing), self.runtime.backend.size)
            if matrix.shape != expected or not np.isfinite(matrix).all():
                raise RuntimeError("E2E_STAGE1_FULL_SCORE_MATRIX_INVALID")
            for batch_index, source_index in enumerate(score_missing):
                self._score_cache[keys[source_index]] = matrix[batch_index]
        return (
            [self._encoded_text[key][0] for key in keys],
            np.stack([self._score_cache[key] for key in keys]),
            [dict(self._encoded_text[key][1]) for key in keys],
        )

    def _fps_and_frames(self, group: VideoRows) -> tuple[float, np.ndarray]:
        fps_values = np.asarray(self.runtime.catalog.mapping_fps[group.rows], dtype=np.float64)
        valid = fps_values[np.isfinite(fps_values) & (fps_values > 0)]
        if not len(valid) or not np.allclose(valid, valid[0], rtol=0, atol=1e-6):
            raise RuntimeError(f"MAPPING_FPS_INVALID: {group.video_id}")
        frames = np.asarray(self.runtime.catalog.original_idx[group.rows], dtype=np.int64)
        return float(valid[0]), frames

    def _single_event_pool(
        self, plan: QueryPlan, vector: np.ndarray, scores: np.ndarray
    ) -> tuple[dict[str, Any], ...]:
        del vector
        cache_key = (plan.query_id, plan.language, plan.grounding_text)
        if cache_key in self._single_pool_cache:
            return self._single_pool_cache[cache_key]
        output = []
        for group in self.groups:
            fps, frames = self._fps_and_frames(group)
            local = scores[group.rows]
            pool = build_diverse_event_pool(
                f"{plan.query_id}:E1@{group.video_id}", local, frames, fps
            )
            for item in pool:
                global_row = int(group.rows[item.catalog_position])
                mapped = self.runtime.catalog.map_row(global_row)
                output.append(
                    {
                        "video_id": group.video_id,
                        "global_row": global_row,
                        "catalog_position": item.catalog_position,
                        "n": int(mapped["n"]),
                        "original_frame_idx": int(mapped["original_frame_idx"]),
                        "score": item.similarity,
                        "event_region_id": item.event_region_id,
                        "mapping_fps": fps,
                    }
                )
        ordered = tuple(sorted(output, key=lambda row: (-row["score"], row["global_row"])))
        self._single_pool_cache[cache_key] = ordered
        return ordered

    def _refine(
        self, video_id: str, frame_id: int, text: str, text_embedding: np.ndarray
    ) -> dict[str, Any]:
        fingerprint = hashlib.sha256(text.encode("utf-8")).hexdigest()
        key = (video_id, int(frame_id), fingerprint)
        if key in self._m1_cache:
            self.m1_cache_hits += 1
            return dict(self._m1_cache[key])
        assets = resolve_assets(
            self.dataset_root, video_id, self.video_partitions, self.keyframe_partitions
        )
        decoder = self.decoder_factory(video_id, assets.video)
        try:
            diagnostic, images = refine_local_event(
                decoder=decoder,
                image_encoder=self.image_encoder,
                text_embedding=text_embedding,
                anchor_frame_idx=int(frame_id),
                settings=M1Settings(),
            )
            refined = int(diagnostic["refined_frame_idx"])
            if refined not in images:
                raise RuntimeError("M1_REFINED_IMAGE_UNAVAILABLE")
            if (video_id, refined) not in self._frame_embedding_cache:
                raise RuntimeError("M1_REFINED_EMBEDDING_CACHE_MISS")
            value = {
                **diagnostic,
                "video_id": video_id,
                "coarse_frame_idx": int(frame_id),
                "event_text_fingerprint": fingerprint,
                "status": "REFINED",
            }
        except (ImportError, IndexError, OSError, RuntimeError, ValueError) as error:
            value = {
                "video_id": video_id,
                "coarse_frame_idx": int(frame_id),
                "refined_frame_idx": int(frame_id),
                "event_text_fingerprint": fingerprint,
                "status": "COARSE_FALLBACK",
                "reason": f"{type(error).__name__}: {error}",
            }
        finally:
            decoder.close()
        self._m1_cache[key] = value
        return dict(value)

    def _ground_single(
        self, plan: QueryPlan, variant: str
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        vector, scores, encoding = self._scores(
            plan.grounding_text, plan.language, f"{plan.query_id}__grounding"
        )
        pool = self._single_event_pool(plan, vector, scores)
        rows, diagnostics = [], []
        for coarse_rank, candidate in enumerate(pool, 1):
            item = dict(candidate)
            item["coarse_rank"] = coarse_rank
            item["coarse_frame_id"] = int(candidate["original_frame_idx"])
            item["frame_id"] = item["coarse_frame_id"]
            item["m1_applied"] = False
            if (
                variant == "P1_CANONICAL"
                and coarse_rank <= self.settings.m1_refine_top_single_event
            ):
                refinement = self._refine(
                    item["video_id"], item["coarse_frame_id"], plan.grounding_text, vector
                )
                item["frame_id"] = int(refinement["refined_frame_idx"])
                item["m1_applied"] = True
                diagnostics.append({"query_id": plan.query_id, "variant": variant, **refinement})
            item["encoding"] = encoding
            rows.append(item)
            if len(rows) >= self.settings.max_predictions * 2:
                break
        return rows, diagnostics

    def predict_kis(self, plan: QueryPlan, variant: str) -> PredictionResult:
        started = monotonic()
        grounded, m1_diagnostics = self._ground_single(plan, variant)
        seen, predictions, provenance = set(), [], []
        for row in grounded:
            key = (row["video_id"], row["frame_id"])
            if key in seen:
                continue
            seen.add(key)
            predictions.append(
                {
                    "query_id": plan.query_id,
                    "video_id": row["video_id"],
                    "frame_id": row["frame_id"],
                }
            )
            provenance.append(
                {
                    "query_id": plan.query_id,
                    "task": "KIS",
                    "variant": variant,
                    **{key: value for key, value in row.items() if key != "encoding"},
                }
            )
            if len(predictions) == self.settings.max_predictions:
                break
        diagnostics = provenance + m1_diagnostics
        return PredictionResult(
            plan.as_dict(), tuple(_renumber(predictions)), tuple(diagnostics), monotonic() - started
        )

    def _nearest_catalog_row(self, video_id: str, frame_id: int) -> tuple[int, dict[str, Any]]:
        group = self.group_by_video[video_id]
        frames = np.asarray(self.runtime.catalog.original_idx[group.rows], dtype=np.int64)
        position = min(
            range(len(frames)), key=lambda index: (abs(int(frames[index]) - frame_id), index)
        )
        global_row = int(group.rows[position])
        return global_row, self.runtime.catalog.map_row(global_row)

    def _object_names(self, video_id: str, frame_id: int) -> tuple[tuple[str, ...], int]:
        _, mapped = self._nearest_catalog_row(video_id, frame_id)
        n = int(mapped["n"])
        key = (video_id, n)
        if key not in self._object_cache:
            assets = resolve_assets(
                self.dataset_root, video_id, self.video_partitions, self.keyframe_partitions
            )
            path = assets.object_directory / f"{n:03d}.json"
            names: list[str] = []
            if path.is_file():
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    raw_names = payload.get("detection_class_names", [])
                    raw_scores = payload.get("detection_scores", [])
                    if isinstance(raw_names, list):
                        pairs = []
                        for index, name in enumerate(raw_names):
                            score = float(raw_scores[index]) if index < len(raw_scores) else 0.0
                            pairs.append((score, str(name)))
                        names = [
                            name for _, name in sorted(pairs, key=lambda pair: (-pair[0], pair[1]))
                        ]
                except (OSError, TypeError, ValueError, json.JSONDecodeError):
                    names = []
            self._object_cache[key] = tuple(dict.fromkeys(names))
        return self._object_cache[key], int(mapped["original_frame_idx"])

    def _frame_embedding(self, row: dict[str, Any]) -> np.ndarray:
        key = (row["video_id"], int(row["frame_id"]))
        if key in self._frame_embedding_cache:
            self.qa_frame_cache_hits += 1
            return self._frame_embedding_cache[key]
        if int(row["frame_id"]) == int(row["coarse_frame_id"]):
            vector = np.asarray(
                self.runtime.backend.vectors_at(np.asarray([row["global_row"]], dtype=np.int64))[0],
                dtype=np.float32,
            )
            norm = float(np.linalg.norm(vector))
            if norm == 0:
                raise RuntimeError("QA_COARSE_IMAGE_VECTOR_ZERO")
            vector = vector / norm
        else:
            image = self._decode_image(row["video_id"], int(row["frame_id"]))
            vector = np.asarray(
                self.runtime.encoder.encode_rgb_arrays([image])[0], dtype=np.float32
            )
        self._frame_embedding_cache[key] = vector
        return vector

    def _decode_image(self, video_id: str, frame_id: int) -> np.ndarray:
        assets = resolve_assets(
            self.dataset_root, video_id, self.video_partitions, self.keyframe_partitions
        )
        decoder = self.decoder_factory(video_id, assets.video)
        try:
            return np.asarray(decoder.decode_indices([frame_id])[0].image, dtype=np.uint8)
        finally:
            decoder.close()

    def _answer_embeddings(self, intent: str, candidates: tuple[Any, ...]) -> np.ndarray:
        ids = tuple(candidate.canonical_id for candidate in candidates)
        key = (intent, ids)
        if key not in self._answer_embedding_cache:
            prompt = PROMPTS.get(intent, PROMPTS["GENERIC_VISUAL"])
            texts = [prompt.format(answer=candidate.english_clip_text) for candidate in candidates]
            vectors = np.asarray(self.runtime.encoder.encode_text(texts), dtype=np.float32)
            self._answer_embedding_cache[key] = vectors
        return self._answer_embedding_cache[key]

    def _answer_row(
        self, plan: QueryPlan, row: dict[str, Any], intent: str, rank: int
    ) -> tuple[str, dict[str, Any]]:
        language = plan.language if plan.language in {"vi", "en"} else "en"
        diagnostic: dict[str, Any] = {"intent": intent, "answer_fallback_reason": None}
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
        names, metadata_frame = self._object_names(row["video_id"], int(row["frame_id"]))
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

    def predict_qa(self, plan: QueryPlan, variant: str) -> PredictionResult:
        started = monotonic()
        grounded, m1_diagnostics = self._ground_single(plan, variant)
        intent = route_intent(plan.question or "")
        predictions, answer_diagnostics, seen = [], [], set()
        for grounding_rank, row in enumerate(grounded, 1):
            answer, diagnostic = self._answer_row(plan, row, intent, grounding_rank)
            key = (row["video_id"], row["frame_id"], " ".join(answer.casefold().split()))
            if key in seen:
                continue
            seen.add(key)
            predictions.append(
                {
                    "query_id": plan.query_id,
                    "video_id": row["video_id"],
                    "frame_id": int(row["frame_id"]),
                    "answer": answer,
                    "global_row": int(row["global_row"]),
                    "btc_n": row.get("n"),
                    "coarse_frame_id": int(row["coarse_frame_id"]),
                    "retrieval_score": row.get("score"),
                }
            )
            answer_diagnostics.append(
                {
                    "query_id": plan.query_id,
                    "variant": variant,
                    "grounding_rank": grounding_rank,
                    "video_id": row["video_id"],
                    "frame_id": int(row["frame_id"]),
                    "answer": answer,
                    **diagnostic,
                }
            )
            if len(predictions) == self.settings.max_predictions:
                break
        return PredictionResult(
            plan.as_dict(),
            tuple(_renumber(predictions)),
            tuple(answer_diagnostics + m1_diagnostics),
            monotonic() - started,
        )

    def _trake_chains(
        self, plan: QueryPlan
    ) -> tuple[list[dict[str, Any]], list[np.ndarray], list[dict[str, Any]]]:
        event_texts = [text for _, text in plan.events]
        embeddings, matrix, encodings = self._scores_many(
            event_texts, plan.language, f"{plan.query_id}__events"
        )
        all_paths: list[DiverseTemporalPath] = []
        chain_by_positions: dict[tuple[int, ...], dict[str, Any]] = {}
        event_ids = [event_id for event_id, _ in plan.events]
        for group in self.groups:
            fps, frames = self._fps_and_frames(group)
            pools = tuple(
                build_diverse_event_pool(
                    f"{event_id}@{group.video_id}", matrix[event_index, group.rows], frames, fps
                )
                for event_index, event_id in enumerate(event_ids)
            )
            feasible, raw_count = enumerate_feasible_paths(pools)
            for local_path in feasible:
                global_rows = tuple(int(group.rows[position]) for position in local_path.positions)
                global_path = DiverseTemporalPath(
                    local_path.score, global_rows, local_path.region_ids, local_path.event_scores
                )
                frames_out = tuple(
                    int(self.runtime.catalog.map_row(global_row)["original_frame_idx"])
                    for global_row in global_rows
                )
                all_paths.append(global_path)
                chain_by_positions[global_rows] = {
                    "video_id": group.video_id,
                    "frame_ids": frames_out,
                    "global_rows": global_rows,
                    "score": global_path.score,
                    "event_scores": global_path.event_scores,
                    "region_ids": global_path.region_ids,
                    "raw_combination_count": raw_count,
                }
        frozen_top = select_coverage_aware(tuple(all_paths), self.settings.t3_selected_delta)
        selected = [chain_by_positions[path.positions] for path in frozen_top]
        selected_ids = {tuple(row["global_rows"]) for row in selected}
        tail = sorted(all_paths, key=lambda path: (-path.score, path.positions))
        for path in tail:
            if path.positions not in selected_ids:
                selected.append(chain_by_positions[path.positions])
                selected_ids.add(path.positions)
            if len(selected) == self.settings.max_predictions:
                break
        return selected, embeddings, encodings

    def predict_trake(self, plan: QueryPlan, variant: str) -> PredictionResult:
        started = monotonic()
        chains, embeddings, encodings = self._trake_chains(plan)
        predictions, diagnostics, seen = [], [], set()
        for coarse_rank, chain in enumerate(chains, 1):
            coarse = tuple(int(value) for value in chain["frame_ids"])
            output = coarse
            refinements = []
            order_fallback = False
            if variant == "P1_CANONICAL" and coarse_rank <= self.settings.m1_refine_top_chains:
                for event_index, (_, text) in enumerate(plan.events):
                    refinements.append(
                        self._refine(
                            chain["video_id"], coarse[event_index], text, embeddings[event_index]
                        )
                    )
                refined = tuple(int(value["refined_frame_idx"]) for value in refinements)
                if _strictly_increasing(refined):
                    output = refined
                else:
                    order_fallback = True
            if not _strictly_increasing(output) or len(output) != len(plan.events):
                raise RuntimeError("TRAKE_OUTPUT_STRUCTURALLY_INVALID")
            key = (chain["video_id"], output)
            if key in seen:
                continue
            seen.add(key)
            predictions.append(
                {
                    "query_id": plan.query_id,
                    "video_id": chain["video_id"],
                    "frame_ids": list(output),
                }
            )
            diagnostics.append(
                {
                    "query_id": plan.query_id,
                    "variant": variant,
                    "coarse_rank": coarse_rank,
                    "video_id": chain["video_id"],
                    "coarse_frame_ids": list(coarse),
                    "output_frame_ids": list(output),
                    "t3_score": chain["score"],
                    "global_rows": list(chain.get("global_rows", ())),
                    "event_scores": list(chain.get("event_scores", ())),
                    "event_region_ids": list(chain.get("region_ids", ())),
                    "t3_selected_delta": self.settings.t3_selected_delta,
                    "t3_top5_unchanged_before_m1": coarse_rank <= 5,
                    "m1_applied": variant == "P1_CANONICAL" and coarse_rank <= 5,
                    "M1_ORDER_FALLBACK_TO_COARSE": order_fallback,
                    "refinements": refinements,
                    "encodings": encodings,
                }
            )
            if len(predictions) == self.settings.max_predictions:
                break
        return PredictionResult(
            plan.as_dict(), tuple(_renumber(predictions)), tuple(diagnostics), monotonic() - started
        )

    def predict_query(
        self, query: dict[str, Any], variant: str = "P1_CANONICAL"
    ) -> PredictionResult:
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}")
        plan = plan_query(query)
        if plan.task == "KIS":
            return self.predict_kis(plan, variant)
        if plan.task == "QA":
            return self.predict_qa(plan, variant)
        return self.predict_trake(plan, variant)

    def predict_queries(
        self, queries: list[dict[str, Any]], variant: str = "P1_CANONICAL"
    ) -> list[PredictionResult]:
        return [self.predict_query(query, variant) for query in queries]

    def runtime_diagnostics(self) -> dict[str, Any]:
        return {
            "m1_cache_hits": self.m1_cache_hits,
            "qa_frame_embedding_cache_hits": self.qa_frame_cache_hits,
            "text_embedding_cache_size": len(self._encoded_text),
            "stage1_score_cache_size": len(self._score_cache),
            "m1_cache_size": len(self._m1_cache),
            "qa_frame_embedding_cache_size": len(self._frame_embedding_cache),
            "ocr_status": self.ocr.status,
            "assets_loaded_once": True,
        }

    def close(self) -> None:
        self.runtime.close()


__all__ = ["CanonicalTriagePipeline", "PredictionResult"]
