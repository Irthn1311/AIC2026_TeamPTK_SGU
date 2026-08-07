"""Long-lived operational KIS runtime engine and lifecycle manager."""

from __future__ import annotations

import dataclasses
import csv
import hashlib
import json
import re
import sys
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from system_tai.checkpointing.exporter import CheckpointExporter
from system_tai.common.schemas import KISResult, ValidationResult
from system_tai.data.corpus_discovery import (
    CorpusManifest,
    DiscoveryMetrics,
    discover_corpus,
    load_corpus_manifest,
    load_or_build_manifest_cache,
)
from system_tai.features.btc_clip_store import FeatureStoreRegistry
from system_tai.features.query_encoder import SharedOpenAIClipEncoder
from system_tai.kis.benchmark import resolve_device
from system_tai.kis.contest_runner import _git_commit_hash, _validation_payload
from system_tai.kis.session_schema import (
    DuplicateRequestIdError,
    HealthRequest,
    QueryRequest,
    SessionConfig,
    ShutdownRequest,
)
from system_tai.refinement.engine import ExactFrameRefiner
from system_tai.refinement.models import Phase3Candidate, RefinementQuery
from system_tai.refinement.runner import _write_json, _write_refined_csv
from system_tai.refinement.video import OpenCVVideoDecoder, RawVideoRegistry
from system_tai.retrieval.multi_query import WeightedRRFRetriever
from system_tai.retrieval.vector_search import ExactNumpyRetriever
from system_tai.validation.checkpoint_validator import CheckpointValidator


def safe_request_directory_name(request_id: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", request_id).strip("._-") or "request"
    digest = hashlib.sha256(request_id.encode("utf-8")).hexdigest()[:8]
    return f"{normalized}-{digest}"


def _write_internal_csv(results: Sequence[KISResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "query_id",
        "rank",
        "video_id",
        "frame_id",
        "fusion_score",
        "variant_hit_count",
        "best_individual_rank",
        "clip_row_diagnostic",
        "keyframe_order_diagnostic",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for result in sorted(results, key=lambda item: item.query_id):
            for candidate in result.ranked_candidates:
                metadata = candidate.diagnostic_metadata or {}
                writer.writerow(
                    {
                        "query_id": result.query_id,
                        "rank": candidate.rank,
                        "video_id": candidate.video_id,
                        "frame_id": candidate.frame_id,
                        "fusion_score": candidate.score,
                        "variant_hit_count": metadata.get("variant_hit_count"),
                        "best_individual_rank": metadata.get("best_individual_rank"),
                        "clip_row_diagnostic": candidate.clip_row,
                        "keyframe_order_diagnostic": candidate.keyframe_order,
                    }
                )


class OperationalKISRuntime:
    def __init__(
        self,
        *,
        config: SessionConfig,
        manifest_path: Path,
        manifest: CorpusManifest,
        registry: FeatureStoreRegistry,
        raw_video_registry: RawVideoRegistry,
        shared_encoder: SharedOpenAIClipEncoder,
        decoder: OpenCVVideoDecoder,
        exporter: CheckpointExporter | None = None,
        validator: CheckpointValidator | None = None,
        clock: Callable[[], float] = time.perf_counter,
        bootstrap_timings: Mapping[str, Any] | None = None,
    ) -> None:
        self.config = config
        self.manifest_path = manifest_path
        self.manifest = manifest
        self.registry = registry
        self.raw_video_registry = raw_video_registry
        self.shared_encoder = shared_encoder
        self.decoder = decoder
        self.exporter = exporter or CheckpointExporter()
        self.validator = validator or CheckpointValidator()
        self.clock = clock
        self.bootstrap_timings = dict(bootstrap_timings or {})

        self.exact_retriever = ExactNumpyRetriever(
            registry=self.registry,
            text_encoder=self.shared_encoder,
            chunk_size=config.chunk_size,
        )
        self.weighted_rrf = WeightedRRFRetriever(self.exact_retriever)
        self.refiner = ExactFrameRefiner(
            raw_videos=self.raw_video_registry,
            decoder=self.decoder,
            encoder=self.shared_encoder,
            clock=self.clock,
        )

        self.session_id = config.session_id or f"session-{uuid.uuid4().hex[:8]}"
        self.start_time_utc = datetime.now(UTC).isoformat()
        self.output_root = Path(config.output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        (self.output_root / "requests").mkdir(parents=True, exist_ok=True)

        self._seen_request_ids: set[str] = set()
        self._request_count = 0
        self._successful_query_count = 0
        self._failed_query_count = 0
        self._health_request_count = 0
        self._malformed_request_count = 0

        self._save_session_manifest(shutdown_reason=None)

    @classmethod
    def bootstrap(
        cls,
        config: SessionConfig,
        *,
        clock: Callable[[], float] = time.perf_counter,
        registry_loader: Callable[[Path], FeatureStoreRegistry] | None = None,
        encoder_factory: Callable[..., SharedOpenAIClipEncoder] | None = None,
        decoder_factory: Callable[[], OpenCVVideoDecoder] | None = None,
        exporter: CheckpointExporter | None = None,
        validator: CheckpointValidator | None = None,
    ) -> OperationalKISRuntime:
        start_time = clock()
        output_root = Path(config.output_root)
        output_root.mkdir(parents=True, exist_ok=True)
        feature_manifest_path = output_root / "feature_manifest.json"

        discovery_metrics = DiscoveryMetrics()
        manifest_source_status = "UNKNOWN"
        manifest_write_seconds = 0.0

        if config.reuse_manifest is not None:
            manifest_start = clock()
            manifest = load_corpus_manifest(
                config.reuse_manifest,
                input_root=config.input_root,
            )
            manifest_seconds = clock() - manifest_start
            discovery_seconds = 0.0
            write_start = clock()
            manifest.write(feature_manifest_path, portable=False)
            manifest_write_seconds = clock() - write_start
            manifest_source_status = "REUSED"
        elif config.manifest_cache is not None:
            discovery_start = clock()
            cached = load_or_build_manifest_cache(
                config.manifest_cache,
                input_root=config.input_root,
            )
            manifest = cached.manifest
            discovery_seconds = clock() - discovery_start if cached.status != "CACHE_HIT" else 0.0
            discovery_metrics = manifest.discovery_metrics if cached.status != "CACHE_HIT" else DiscoveryMetrics()
            manifest_start = clock()
            manifest.write(feature_manifest_path, portable=False)
            manifest_seconds = clock() - manifest_start
            manifest_write_seconds = manifest_seconds
            manifest_source_status = cached.status
        else:
            discovery_start = clock()
            manifest = discover_corpus(config.input_root)
            discovery_seconds = clock() - discovery_start
            discovery_metrics = manifest.discovery_metrics
            manifest_start = clock()
            manifest.write(feature_manifest_path, portable=False)
            manifest_seconds = clock() - manifest_start
            manifest_write_seconds = manifest_seconds
            manifest_source_status = "BUILT"

        resolved_device = resolve_device(config.device)
        ref_config = config.refinement_config
        if ref_config.device not in {"cpu", "cuda"}:
            ref_config = dataclasses.replace(ref_config, device=resolved_device)
        exec_config = dataclasses.replace(config, device=resolved_device, refinement_config=ref_config)

        registry_start = clock()
        loader = registry_loader or FeatureStoreRegistry.from_manifest
        registry = loader(feature_manifest_path)
        registry_seconds = clock() - registry_start

        raw_video_registry = RawVideoRegistry.from_manifest(manifest)

        model_start = clock()
        factory = encoder_factory or SharedOpenAIClipEncoder
        shared_encoder = factory(
            device=resolved_device,
            allow_model_download=config.allow_model_download,
            cache_dir=config.clip_cache_dir,
        )
        model_seconds = clock() - model_start

        decoder_make = decoder_factory or OpenCVVideoDecoder
        decoder = decoder_make()

        bootstrap_timings = {
            "discovery_seconds": discovery_seconds,
            "manifest_load_or_build_seconds": manifest_seconds,
            "manifest_write_seconds": manifest_write_seconds,
            "registry_load_seconds": registry_seconds,
            "model_load_seconds": model_seconds,
            "total_bootstrap_seconds": clock() - start_time,
            "manifest_source_status": manifest_source_status,
            **discovery_metrics.to_payload(),
        }

        return cls(
            config=exec_config,
            manifest_path=feature_manifest_path,
            manifest=manifest,
            registry=registry,
            raw_video_registry=raw_video_registry,
            shared_encoder=shared_encoder,
            decoder=decoder,
            exporter=exporter,
            validator=validator,
            clock=clock,
            bootstrap_timings=bootstrap_timings,
        )

    def handle_health(self, request: HealthRequest) -> dict[str, Any]:
        if request.request_id in self._seen_request_ids:
            raise DuplicateRequestIdError(f"request_id '{request.request_id}' has already been processed")
        self._seen_request_ids.add(request.request_id)
        self._health_request_count += 1
        self._request_count += 1
        return {
            "type": "health",
            "request_id": request.request_id,
            "status": "READY",
            "session_id": self.session_id,
            "device": self.shared_encoder.identifiers.get("device", self.config.device),
            "manifest_fingerprint": self.manifest.fingerprint,
            "video_count": len(self.manifest.videos),
            "feature_row_count": self.registry.total_rows,
            "request_count": self._request_count,
        }

    def handle_query(self, request: QueryRequest) -> dict[str, Any]:
        if request.request_id in self._seen_request_ids:
            raise DuplicateRequestIdError(f"request_id '{request.request_id}' has already been processed")
        self._seen_request_ids.add(request.request_id)
        self._request_count += 1

        req_start = self.clock()
        query_dir = self.output_root / "requests" / safe_request_directory_name(request.request_id)
        if query_dir.exists():
            # If folder already exists, avoid overwriting
            query_dir = self.output_root / "requests" / f"{safe_request_directory_name(request.request_id)}-{uuid.uuid4().hex[:4]}"
        query_dir.mkdir(parents=True, exist_ok=True)

        validation_start = self.clock()
        variants = request.variants()
        validation_seconds = self.clock() - validation_start

        # Text encoding
        text_encode_start = self.clock()
        rankings: dict[str, KISResult] = {}
        for variant in variants:
            vector = self.shared_encoder.encode(variant.text)
            rankings[variant.variant_id] = self.exact_retriever.search_vector(
                query_id=f"{request.query_id}::{variant.variant_id}",
                query_vector=vector,
                top_k=request.top_k_per_variant,
            )
        text_encode_seconds = self.clock() - text_encode_start

        # Retrieval & Fusion
        retrieval_start = self.clock()
        fusion_start = self.clock()
        fused_result = self.weighted_rrf.fuse_rankings(
            query_id=request.query_id,
            variants=variants,
            rankings=rankings,
            output_top_k=request.output_top_k,
            rrf_constant=self.config.rrf_constant,
        )
        fusion_seconds = self.clock() - fusion_start
        retrieval_seconds = self.clock() - retrieval_start

        # Phase 3 Export & Validation
        export_start = self.clock()
        top100_jsonl = query_dir / "top100.jsonl"
        self.exporter.export(fused_result, top100_jsonl)
        _write_internal_csv((fused_result,), query_dir / "top100.csv")

        # candidate details json
        candidates_json = query_dir / "candidates.json"
        candidates_data = {
            "query_id": request.query_id,
            "request_id": request.request_id,
            "records": [
                {
                    "query_id": fused_result.query_id,
                    "rank": candidate.rank,
                    "video_id": candidate.video_id,
                    "frame_id": candidate.frame_id,
                    "fusion_score": candidate.score,
                    "variant_hit_count": (candidate.diagnostic_metadata or {}).get("variant_hit_count"),
                    "best_individual_rank": (candidate.diagnostic_metadata or {}).get("best_individual_rank"),
                    "clip_row_diagnostic": candidate.clip_row,
                    "keyframe_order_diagnostic": candidate.keyframe_order,
                }
                for candidate in fused_result.ranked_candidates
            ],
        }
        candidates_json.write_text(json.dumps(candidates_data, indent=2) + "\n", encoding="utf-8")
        retrieval_export_seconds = self.clock() - export_start

        retrieval_val_start = self.clock()
        validation = self.validator.validate(top100_jsonl, self.registry)
        retrieval_val_seconds = self.clock() - retrieval_val_start

        val_report_path = query_dir / "validation_report.json"
        val_report_path.write_text(
            json.dumps(_validation_payload(validation), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        artifacts_dict: dict[str, str] = {
            "top100_jsonl": str(top100_jsonl.relative_to(self.output_root)).replace("\\", "/"),
            "top100_csv": str((query_dir / "top100.csv").relative_to(self.output_root)).replace("\\", "/"),
            "candidates_json": str(candidates_json.relative_to(self.output_root)).replace("\\", "/"),
            "validation_report": str(val_report_path.relative_to(self.output_root)).replace("\\", "/"),
        }

        refinement_requested = request.refine_top_n > 0
        refinement_valid: bool | None = None
        refined_count = 0
        refinement_seconds = 0.0
        refinement_export_seconds = 0.0
        refinement_val_seconds = 0.0

        if refinement_requested:
            ref_start = self.clock()
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
                        "best_individual_rank": (c.diagnostic_metadata or {}).get("best_individual_rank"),
                        "clip_row_diagnostic": c.clip_row,
                        "keyframe_order_diagnostic": c.keyframe_order,
                    },
                )
                for c in fused_result.ranked_candidates
            )
            ref_query = RefinementQuery(request.query_id, variants, phase3_candidates)

            # Execution config for refinement
            exec_ref_config = self.config.refinement_config
            if exec_ref_config.top_candidates_to_refine != request.refine_top_n:
                exec_ref_config = dataclasses.replace(
                    exec_ref_config,
                    top_candidates_to_refine=request.refine_top_n,
                )

            outcome = self.refiner.refine_query(ref_query, exec_ref_config)
            refinement_seconds = self.clock() - ref_start

            ref_export_start = self.clock()
            refined_jsonl = query_dir / "refined_top100.jsonl"
            self.exporter.export(outcome.result, refined_jsonl)
            refined_csv = _write_refined_csv((outcome.result,), query_dir / "refined_top100.csv")
            ref_cand_json = _write_json(query_dir / "refinement_candidates.json", outcome.candidates)
            ref_trace_json = _write_json(
                query_dir / "refinement_trace.json",
                {
                    "query_id": request.query_id,
                    "candidates": outcome.candidates,
                    "warnings": outcome.warnings,
                },
            )
            refinement_export_seconds = self.clock() - ref_export_start

            ref_val_start = self.clock()
            ref_validation = self.validator.validate(refined_jsonl)
            refinement_val_seconds = self.clock() - ref_val_start
            ref_val_report = _write_json(
                query_dir / "refinement_validation_report.json",
                _validation_payload(ref_validation),
            )

            refinement_valid = ref_validation.valid
            refined_count = len([item for item in outcome.candidates if item.refined_frame_id is not None])

            artifacts_dict.update(
                {
                    "refined_top100_jsonl": str(refined_jsonl.relative_to(self.output_root)).replace("\\", "/"),
                    "refined_top100_csv": str(refined_csv.relative_to(self.output_root)).replace("\\", "/"),
                    "refinement_candidates_json": str(ref_cand_json.relative_to(self.output_root)).replace("\\", "/"),
                    "refinement_trace_json": str(ref_trace_json.relative_to(self.output_root)).replace("\\", "/"),
                    "refinement_validation_report": str(ref_val_report.relative_to(self.output_root)).replace("\\", "/"),
                }
            )

        total_seconds = self.clock() - req_start

        # Write request manifest & timings & summary
        req_manifest_payload = {
            "request_id": request.request_id,
            "query_id": request.query_id,
            "query_vi": request.query_vi,
            "query_en": request.query_en,
            "query_en_expansion": request.query_en_expansion,
            "weights": {
                "weight_vi": request.weight_vi,
                "weight_en": request.weight_en,
                "weight_en_expansion": request.weight_en_expansion,
            },
            "top_k_per_variant": request.top_k_per_variant,
            "output_top_k": request.output_top_k,
            "refine_top_n": request.refine_top_n,
            "retrieval_valid": validation.valid,
            "refinement_requested": refinement_requested,
            "refinement_valid": refinement_valid,
            "artifacts": artifacts_dict,
        }
        _write_json(query_dir / "request_manifest.json", req_manifest_payload)

        timings_payload = {
            "validation_seconds": validation_seconds,
            "text_encode_seconds": text_encode_seconds,
            "retrieval_seconds": retrieval_seconds,
            "fusion_seconds": fusion_seconds,
            "retrieval_export_seconds": retrieval_export_seconds,
            "retrieval_validation_seconds": retrieval_val_seconds,
            "refinement_seconds": refinement_seconds,
            "refinement_export_seconds": refinement_export_seconds,
            "refinement_validation_seconds": refinement_val_seconds,
            "total_seconds": total_seconds,
            "decoded_frame_count": self.decoder.metrics.decoded_frames if hasattr(self.decoder, "metrics") else 0,
            "encoded_image_count": 0,
        }
        _write_json(query_dir / "request_timings.json", timings_payload)

        summary_md = [
            f"# Request Summary: {request.request_id} ({request.query_id})",
            "",
            f"- Retrieval valid: `{validation.valid}`",
            f"- Refinement requested: `{refinement_requested}`",
            f"- Refinement valid: `{refinement_valid}`",
            f"- Result count: {len(fused_result.ranked_candidates)}",
            f"- Total seconds: {total_seconds:.6f}s",
            "",
        ]
        (query_dir / "request_summary.md").write_text("\n".join(summary_md) + "\n", encoding="utf-8")

        self._successful_query_count += 1

        return {
            "type": "query_result",
            "request_id": request.request_id,
            "query_id": request.query_id,
            "status": "SUCCESS",
            "retrieval_valid": validation.valid,
            "refinement_requested": refinement_requested,
            "refinement_valid": refinement_valid,
            "result_count": len(fused_result.ranked_candidates),
            "refined_count": refined_count,
            "artifacts": artifacts_dict,
            "timings": timings_payload,
        }

    def handle_shutdown(
        self,
        request: ShutdownRequest,
        shutdown_reason: str = "requested",
    ) -> dict[str, Any]:
        if request.request_id in self._seen_request_ids:
            raise DuplicateRequestIdError(f"request_id '{request.request_id}' has already been processed")
        self._seen_request_ids.add(request.request_id)
        self._request_count += 1
        self.close(shutdown_reason=shutdown_reason)
        return {
            "type": "shutdown",
            "request_id": request.request_id,
            "status": "STOPPING",
            "processed_requests": self._request_count,
            "successful_queries": self._successful_query_count,
            "failed_queries": self._failed_query_count,
        }

    def handle_error(
        self,
        request_id: str | None,
        error_code: str,
        error_type: str,
        message: str,
        session_continues: bool,
    ) -> dict[str, Any]:
        self._request_count += 1
        if error_code == "MALFORMED_JSON":
            self._malformed_request_count += 1
        else:
            self._failed_query_count += 1

        return {
            "type": "error",
            "request_id": request_id,
            "status": "ERROR",
            "error_code": error_code,
            "error_type": error_type,
            "message": message,
            "session_continues": session_continues,
        }

    def close(self, shutdown_reason: str = "normal") -> None:
        """Flushes session summary and releases resources cleanly."""
        try:
            if hasattr(self.decoder, "close"):
                self.decoder.close()
        except Exception as exc:
            print(f"warning: error closing decoder: {exc}", file=sys.stderr)

        self._save_session_manifest(shutdown_reason=shutdown_reason)

    def _save_session_manifest(self, shutdown_reason: str | None) -> None:
        manifest_payload = {
            "session_id": self.session_id,
            "git_commit_hash": _git_commit_hash(),
            "start_time_utc": self.start_time_utc,
            "end_time_utc": datetime.now(UTC).isoformat() if shutdown_reason else None,
            "device": self.shared_encoder.identifiers.get("device", self.config.device),
            "manifest_path": str(self.manifest_path),
            "manifest_schema_version": self.manifest.schema_version,
            "manifest_fingerprint": self.manifest.fingerprint,
            "video_count": len(self.manifest.videos),
            "feature_row_count": self.registry.total_rows,
            "raw_video_count": len(self.raw_video_registry.records),
            "model_identifier": self.shared_encoder.identifiers.get("model", "ViT-B/32"),
            "model_metadata": dict(self.shared_encoder.identifiers),
            "model_load_count": 1,
            "registry_load_count": 1,
            "decoder_initialization_count": 1,
            "registry_load_seconds": self.bootstrap_timings.get("registry_load_seconds", 0.0),
            "model_load_seconds": self.bootstrap_timings.get("model_load_seconds", 0.0),
            "refinement_model_load_seconds": 0.0,
            "bootstrap_timings": self.bootstrap_timings,
            "request_count": self._request_count,
            "successful_query_count": self._successful_query_count,
            "failed_query_count": self._failed_query_count,
            "health_request_count": self._health_request_count,
            "malformed_request_count": self._malformed_request_count,
            "shutdown_reason": shutdown_reason,
        }
        _write_json(self.output_root / "session_manifest.json", manifest_payload)

        summary_lines = [
            "# system_tai Operational Session Summary",
            "",
            f"- Session ID: `{self.session_id}`",
            f"- Device: `{manifest_payload['device']}`",
            f"- Video count: {manifest_payload['video_count']}",
            f"- Feature rows: {manifest_payload['feature_row_count']}",
            f"- Model load count: {manifest_payload['model_load_count']}",
            f"- Registry load count: {manifest_payload['registry_load_count']}",
            f"- Total requests: {self._request_count}",
            f"- Successful queries: {self._successful_query_count}",
            f"- Failed queries: {self._failed_query_count}",
            f"- Health requests: {self._health_request_count}",
            f"- Malformed requests: {self._malformed_request_count}",
            f"- Shutdown reason: `{shutdown_reason or 'running'}`",
            "",
        ]
        (self.output_root / "session_summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
