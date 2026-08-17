"""Diagnostic S1 grounding channel with frozen OpenAI-CLIP QA answering."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from triage_eg.e2eg1.pipeline import SafeCoveragePipeline
from triage_eg.retrieval.stage2 import QueryRequest

from .backend import Siglip2ExactBackend
from .contracts import EMBEDDING_DIMENSION, EXPECTED_OPENAI_CLIP_ID, MODEL_ID
from .encoder import Siglip2OfflineEncoder


class Siglip2GroundingPipeline(SafeCoveragePipeline):
    """Change only grounding scores; retain the inherited frozen QA answer head."""

    def __init__(
        self,
        runtime: Any,
        dataset_root: str | Any,
        *,
        grounding_encoder: Siglip2OfflineEncoder,
        grounding_backend: Siglip2ExactBackend,
        **kwargs: Any,
    ) -> None:
        super().__init__(runtime, dataset_root, **kwargs)
        if not grounding_encoder.loaded:
            raise RuntimeError("SCA1_SIGLIP2_ENCODER_NOT_LOADED")
        if grounding_backend.size != runtime.backend.size:
            raise RuntimeError("SCA1_GROUNDING_CATALOG_ROW_COUNT_MISMATCH")
        if grounding_backend.dimension != EMBEDDING_DIMENSION:
            raise RuntimeError("SCA1_GROUNDING_DIMENSION_MISMATCH")
        self.grounding_encoder = grounding_encoder
        self.grounding_backend = grounding_backend
        self.text_identity_records: dict[str, dict[str, Any]] = {}

    def _encode(self, text: str, language: str, query_id: str) -> tuple[np.ndarray, dict[str, Any]]:
        key = (language, text)
        if key not in self._encoded_text:
            canonical = self.runtime.encode_requests([QueryRequest(query_id, text, language, 1)])
            route = dict(canonical.encodings[0])
            clip_input = str(route.get("clip_input_text", ""))
            if not clip_input:
                raise RuntimeError("SCA1_CURRENT_LANGUAGE_ROUTE_EMPTY_CLIP_INPUT")
            vector = self.grounding_encoder.encode_text([clip_input])[0]
            unit_id = _request_id_to_unit_id(query_id)
            text_sha = hashlib.sha256(clip_input.encode("utf-8")).hexdigest()
            provenance = {
                **route,
                "sca1_unit_id": unit_id,
                "sca1_grounding_channel": "S1_SIGLIP2",
                "sca1_clip_input_text": clip_input,
                "sca1_clip_input_text_sha256": text_sha,
                "sca1_a0_encoder_id": EXPECTED_OPENAI_CLIP_ID,
                "sca1_s1_encoder_id": MODEL_ID,
                "sca1_direct_vietnamese": False,
                "sca1_query_rewrite": False,
            }
            self.text_identity_records[unit_id] = {
                "unit_id": unit_id,
                "source_language": language,
                "source_text": text,
                "clip_input_text": clip_input,
                "clip_input_text_sha256": text_sha,
                "a0_encoder_id": EXPECTED_OPENAI_CLIP_ID,
                "s1_encoder_id": MODEL_ID,
            }
            self._encoded_text[key] = (np.asarray(vector, dtype=np.float32), provenance)
        vector, provenance = self._encoded_text[key]
        return vector, dict(provenance)

    def _scores(
        self, text: str, language: str, query_id: str
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        vector, provenance = self._encode(text, language, query_id)
        key = (language, text)
        if key not in self._score_cache:
            scores = np.asarray(self.grounding_backend.score_all(vector), dtype=np.float32)
            if scores.shape != (self.grounding_backend.size,) or not np.isfinite(scores).all():
                raise RuntimeError("SCA1_SIGLIP2_FULL_SCORE_VECTOR_INVALID")
            self._score_cache[key] = scores
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
            canonical = self.runtime.encode_requests(requests)
            clip_inputs = [str(row.get("clip_input_text", "")) for row in canonical.encodings]
            if any(not text for text in clip_inputs):
                raise RuntimeError("SCA1_CURRENT_LANGUAGE_ROUTE_EMPTY_CLIP_INPUT")
            vectors = self.grounding_encoder.encode_text(clip_inputs)
            for batch_index, source_index in enumerate(missing):
                request = requests[batch_index]
                unit_id = _request_id_to_unit_id(request.query_id)
                clip_input = clip_inputs[batch_index]
                text_sha = hashlib.sha256(clip_input.encode("utf-8")).hexdigest()
                route = dict(canonical.encodings[batch_index])
                provenance = {
                    **route,
                    "sca1_unit_id": unit_id,
                    "sca1_grounding_channel": "S1_SIGLIP2",
                    "sca1_clip_input_text": clip_input,
                    "sca1_clip_input_text_sha256": text_sha,
                    "sca1_a0_encoder_id": EXPECTED_OPENAI_CLIP_ID,
                    "sca1_s1_encoder_id": MODEL_ID,
                    "sca1_direct_vietnamese": False,
                    "sca1_query_rewrite": False,
                }
                self.text_identity_records[unit_id] = {
                    "unit_id": unit_id,
                    "source_language": language,
                    "source_text": texts[source_index],
                    "clip_input_text": clip_input,
                    "clip_input_text_sha256": text_sha,
                    "a0_encoder_id": EXPECTED_OPENAI_CLIP_ID,
                    "s1_encoder_id": MODEL_ID,
                }
                self._encoded_text[keys[source_index]] = (
                    np.asarray(vectors[batch_index], dtype=np.float32),
                    provenance,
                )
        score_missing = [index for index, key in enumerate(keys) if key not in self._score_cache]
        if score_missing:
            matrix = np.asarray(
                self.grounding_backend.score_many_all(
                    np.stack([self._encoded_text[keys[index]][0] for index in score_missing])
                ),
                dtype=np.float32,
            )
            expected = (len(score_missing), self.grounding_backend.size)
            if matrix.shape != expected or not np.isfinite(matrix).all():
                raise RuntimeError("SCA1_SIGLIP2_FULL_SCORE_MATRIX_INVALID")
            for batch_index, source_index in enumerate(score_missing):
                self._score_cache[keys[source_index]] = matrix[batch_index]
        return (
            [self._encoded_text[key][0] for key in keys],
            np.stack([self._score_cache[key] for key in keys]),
            [dict(self._encoded_text[key][1]) for key in keys],
        )

    def runtime_diagnostics(self) -> dict[str, Any]:
        return {
            **super().runtime_diagnostics(),
            "sca1_grounding_channel": "S1_SIGLIP2",
            "sca1_grounding_encoder_id": MODEL_ID,
            "sca1_qa_frame_encoder_id": EXPECTED_OPENAI_CLIP_ID,
            "sca1_qa_answer_text_encoder_id": EXPECTED_OPENAI_CLIP_ID,
            "sca1_text_identity_count": len(self.text_identity_records),
            "production_policy_changed": False,
        }


def _request_id_to_unit_id(request_id: str) -> str:
    if request_id.endswith("__grounding"):
        return f"{request_id.removesuffix('__grounding')}:E1"
    marker = "__events__"
    if marker in request_id:
        query_id, event = request_id.rsplit(marker, 1)
        if event.isdigit() and int(event) >= 1:
            return f"{query_id}:E{int(event)}"
    raise RuntimeError(f"SCA1_UNKNOWN_SEMANTIC_REQUEST_ID: {request_id}")


__all__ = ["Siglip2GroundingPipeline"]
