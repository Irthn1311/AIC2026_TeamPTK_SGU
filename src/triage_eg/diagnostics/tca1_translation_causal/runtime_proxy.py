"""Narrow runtime proxy that overrides only frozen FAIL semantic units."""

from __future__ import annotations

from dataclasses import replace
from time import monotonic
from typing import Any

from triage_eg.retrieval.stage2.contracts import QueryRequest
from triage_eg.retrieval.stage2.runtime import EncodedQueryBatch

from .review import FrozenReview


def request_id_to_unit_id(request_id: str) -> str:
    """Map existing E2E request IDs to the frozen semantic-unit identity."""

    if request_id.endswith("__grounding"):
        query_id = request_id.removesuffix("__grounding")
        return f"{query_id}:E1"
    marker = "__events__"
    if marker in request_id:
        query_id, raw_index = request_id.rsplit(marker, 1)
        try:
            event_index = int(raw_index)
        except ValueError as error:
            raise RuntimeError(f"TCA1_UNKNOWN_REQUEST_ID: {request_id}") from error
        if event_index < 1:
            raise RuntimeError(f"TCA1_UNKNOWN_REQUEST_ID: {request_id}")
        return f"{query_id}:E{event_index}"
    raise RuntimeError(f"TCA1_UNKNOWN_REQUEST_ID: {request_id}")


class TCA1RuntimeProxy:
    """Delegate Stage 2A unchanged except the exact A1 FAIL text/language substitution."""

    def __init__(self, delegate: Any, frozen: FrozenReview, arm: str) -> None:
        if arm not in {"A0", "A1"}:
            raise ValueError("TCA-1 arm must be A0 or A1")
        self.delegate = delegate
        self.frozen = frozen
        self.arm = arm
        self.intervention_records: list[dict[str, Any]] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self.delegate, name)

    def load(self) -> TCA1RuntimeProxy:
        self.delegate.load()
        return self

    def encode_requests(self, requests: list[QueryRequest]) -> EncodedQueryBatch:
        if not requests:
            raise ValueError("encode_requests requires at least one request")
        intervention_started = monotonic()
        unit_ids = [request_id_to_unit_id(request.query_id) for request in requests]
        unknown = set(unit_ids) - set(self.frozen.rows_by_unit)
        if unknown:
            raise RuntimeError(f"TCA1_UNKNOWN_SEMANTIC_UNITS: {sorted(unknown)}")
        transformed = []
        applied = []
        for request, unit_id in zip(requests, unit_ids, strict=True):
            review = self.frozen.rows_by_unit[unit_id]
            if request.text.strip() != str(review["source_vi"]).strip() or request.language != "vi":
                raise RuntimeError(f"TCA1_SOURCE_OR_LANGUAGE_MISMATCH: {unit_id}")
            override = self.arm == "A1" and unit_id in self.frozen.fail_unit_ids
            transformed.append(
                replace(request, text=self.frozen.overrides[unit_id], language="en")
                if override
                else request
            )
            applied.append(override)
        intervention_ms = (monotonic() - intervention_started) * 1000
        batch = self.delegate.encode_requests(transformed)
        encodings = []
        latencies = []
        for request, unit_id, override, encoding, latency in zip(
            requests,
            unit_ids,
            applied,
            batch.encodings,
            batch.latencies_ms,
            strict=True,
        ):
            review = self.frozen.rows_by_unit[unit_id]
            expected_clip = (
                self.frozen.overrides[unit_id] if override else str(review["opus_en"]).strip()
            )
            if str(encoding.get("clip_input_text", "")).strip() != expected_clip:
                raise RuntimeError(f"TCA1_CLIP_INPUT_CONTRACT_MISMATCH: {unit_id}")
            patched = {
                **encoding,
                "tca1_arm": self.arm,
                "tca1_unit_id": unit_id,
                "tca1_override_applied": override,
                "tca1_original_vi": request.text.strip(),
                "tca1_baseline_opus_en": str(review["opus_en"]).strip(),
                "tca1_reference_en": self.frozen.overrides.get(unit_id),
                "tca1_translation_latency_comparable_to_a0": False,
            }
            encodings.append(patched)
            latencies.append(
                {
                    **latency,
                    "tca1_intervention_control_ms": intervention_ms / len(requests),
                }
            )
            self.intervention_records.append(
                {
                    "arm": self.arm,
                    "request_id": request.query_id,
                    "unit_id": unit_id,
                    "override_applied": override,
                    "source_vi": request.text.strip(),
                    "baseline_opus_en": str(review["opus_en"]).strip(),
                    "reference_en": self.frozen.overrides.get(unit_id),
                    "clip_input_text": encoding["clip_input_text"],
                    "translation_applied": bool(encoding.get("translation_applied")),
                }
            )
        return EncodedQueryBatch(
            embeddings=batch.embeddings,
            resolutions=batch.resolutions,
            encodings=tuple(encodings),
            latencies_ms=tuple(latencies),
            batch_latency_ms=batch.batch_latency_ms,
        )

    def runtime_manifest(self) -> dict[str, Any]:
        return {
            **self.delegate.runtime_manifest(),
            "tca1": {
                "diagnostic_only": True,
                "arm": self.arm,
                "frozen_fail_unit_count": len(self.frozen.fail_unit_ids),
                "production_policy_changed": False,
            },
        }


__all__ = ["TCA1RuntimeProxy", "request_id_to_unit_id"]
