"""Unified offline Stage 2A operational retrieval orchestration."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic
from typing import Any

import numpy as np

from triage_eg.retrieval.stage1.contracts import SearchConfig
from triage_eg.retrieval.stage1.search import load_search_backend, rank_loaded_query
from triage_eg.retrieval.stage1b.adapters.openai_clip_official import (
    materialize_kaggle_expanded_tokenizer,
    preflight_official_openai_clip,
    resolve_official_asset_paths,
)
from triage_eg.retrieval.stage1b.assets import load_multimodal_encoder
from triage_eg.retrieval.stage1b.contracts import CandidateContract
from triage_eg.retrieval.stage1b.writers import write_json, write_jsonl
from triage_eg.retrieval.stage1d.config import settings_from_yaml
from triage_eg.retrieval.stage1d.contracts import GenerationConfig, TranslatorConfig
from triage_eg.retrieval.stage1d.inputs import validate_translator_asset
from triage_eg.retrieval.stage1d.translator import (
    OfflineViEnTranslator,
    translator_dependency_versions,
)
from triage_eg.video import HardwareConfig, resolve_hardware

from .artifacts import write_operational_query_artifacts
from .contracts import (
    STAGE2_VERSION,
    QueryRequest,
    Stage2RuntimeConfig,
    Stage2RuntimeError,
)
from .language import LanguageResolution, resolve_language
from .results import QueryResult

BackendLoader = Callable[[SearchConfig], tuple[Any, Any]]
ClipPreparer = Callable[
    [Stage2RuntimeConfig, dict[str, Any]], tuple[CandidateContract, dict[str, Any]]
]
TranslatorValidator = Callable[[str | Path], dict[str, Any]]
EncoderFactory = Callable[[CandidateContract], Any]
TranslatorFactory = Callable[[Path, TranslatorConfig, GenerationConfig], Any]


@dataclass(frozen=True)
class EncodedQueryBatch:
    """Validated Stage 2A language-path outputs without retrieval side effects."""

    embeddings: np.ndarray
    resolutions: tuple[LanguageResolution, ...]
    encodings: tuple[dict[str, Any], ...]
    latencies_ms: tuple[dict[str, float], ...]
    batch_latency_ms: float


def _json(path: Path, *, error_code: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Stage2RuntimeError(error_code, str(error)) from error
    if not isinstance(value, dict):
        raise Stage2RuntimeError(error_code, str(path))
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_contracts(config: Stage2RuntimeConfig) -> dict[str, Any]:
    try:
        stage1 = config.stage1_root.expanduser().resolve(strict=True)
    except FileNotFoundError as error:
        raise Stage2RuntimeError("STAGE1_INDEX_NOT_READY", str(error)) from error
    try:
        stage1b = config.stage1b_root.expanduser().resolve(strict=True)
    except FileNotFoundError as error:
        raise Stage2RuntimeError("STAGE1B_ENCODER_NOT_VERIFIED", str(error)) from error
    try:
        stage1e = config.stage1e_root.expanduser().resolve(strict=True)
    except FileNotFoundError as error:
        raise Stage2RuntimeError("STAGE1E_CONTRACT_INVALID", str(error)) from error
    stage1_summary = _json(stage1 / "stage1_summary.json", error_code="STAGE1_INDEX_NOT_READY")
    index_manifest = _json(
        stage1 / "index/index_manifest.json", error_code="STAGE1_INDEX_NOT_READY"
    )
    if (
        stage1_summary.get("index_status") != "COMPLETE"
        or index_manifest.get("status") != "COMPLETE"
    ):
        raise Stage2RuntimeError("STAGE1_INDEX_NOT_READY")
    stage1b_summary = _json(
        stage1b / "stage1b_summary.json", error_code="STAGE1B_ENCODER_NOT_VERIFIED"
    )
    selected = _json(
        stage1b / "encoder/selected_encoder_contract.json",
        error_code="STAGE1B_ENCODER_NOT_VERIFIED",
    )
    adapter = _json(
        stage1b / "encoder/runtime_adapter_manifest.json",
        error_code="STAGE1B_MODEL_SPACE_NOT_VERIFIED",
    )
    if (
        stage1b_summary.get("evaluation_status") != "COMPLETE"
        or selected.get("compatibility_status") != "VERIFIED"
        or not selected.get("checkpoint_sha256")
    ):
        raise Stage2RuntimeError("STAGE1B_ENCODER_NOT_VERIFIED")
    if adapter.get("model_space_status") != "MODEL_SPACE_VERIFIED":
        raise Stage2RuntimeError("STAGE1B_MODEL_SPACE_NOT_VERIFIED")
    stage1e_summary = _json(stage1e / "stage1e_summary.json", error_code="STAGE1E_CONTRACT_INVALID")
    contract_path = stage1e / "language_path_contract.json"
    language_contract = _json(contract_path, error_code="STAGE1E_CONTRACT_INVALID")
    if (
        stage1e_summary.get("stage1e_execution") != "COMPLETE"
        or stage1e_summary.get("stage2_readiness") != "READY"
        or language_contract.get("language_path_status") != "FROZEN_FOR_INTERNAL_BASELINE"
    ):
        raise Stage2RuntimeError("STAGE1E_LANGUAGE_PATH_NOT_FROZEN")
    fingerprint = stage1_summary.get("index_fingerprint")
    if (
        not fingerprint
        or stage1b_summary.get("stage1_index_fingerprint") != fingerprint
        or language_contract.get("stage1_index_fingerprint") != fingerprint
    ):
        raise Stage2RuntimeError("STAGE1_INDEX_FINGERPRINT_MISMATCH")
    candidate_id = selected.get("selected_candidate_id") or selected.get("candidate_id")
    if (
        language_contract.get("english_path", {}).get("mode") != "DIRECT"
        or language_contract.get("english_path", {}).get("text_encoder") != candidate_id
        or language_contract.get("vietnamese_path", {}).get("mode")
        != "TRANSLATE_TO_ENGLISH_THEN_CLIP"
        or language_contract.get("vietnamese_path", {}).get("text_encoder") != candidate_id
        or language_contract.get("clip_compatibility") != "VERIFIED"
        or language_contract.get("model_space_status") != "MODEL_SPACE_VERIFIED"
    ):
        raise Stage2RuntimeError("STAGE1E_CONTRACT_INVALID", "encoder path")
    translator_contract = language_contract.get("vietnamese_path", {}).get("translator", {})
    translator_config, generation, _, _ = settings_from_yaml(config.stage1d_config)
    if (
        translator_contract.get("model_id") != translator_config.model_id
        or translator_contract.get("exact_revision") != translator_config.exact_revision
    ):
        raise Stage2RuntimeError("TRANSLATOR_REVISION_MISMATCH")
    return {
        "stage1_root": stage1,
        "stage1b_root": stage1b,
        "stage1e_root": stage1e,
        "stage1_summary": stage1_summary,
        "stage1b_summary": stage1b_summary,
        "selected_contract": selected,
        "stage1b_adapter": adapter,
        "stage1e_summary": stage1e_summary,
        "language_contract": language_contract,
        "language_contract_sha256": _sha256(contract_path),
        "translator_config": replace(
            translator_config,
            device=config.translator_device,
            batch_size=config.translator_batch_size,
        ),
        "generation_config": generation,
        "generation_config_sha256": _sha256(config.stage1d_config.resolve(strict=True)),
    }


def _prepare_clip(
    config: Stage2RuntimeConfig, selected: dict[str, Any]
) -> tuple[CandidateContract, dict[str, Any]]:
    paths = resolve_official_asset_paths(config.clip_asset_root)
    runtime_source, _ = materialize_kaggle_expanded_tokenizer(
        paths.source_root,
        config.output_root / "runtime_cache/openai_clip_source",
    )
    paths = resolve_official_asset_paths(
        config.clip_asset_root,
        source_root=runtime_source,
        checkpoint_path=paths.checkpoint_path,
        asset_manifest_path=paths.asset_manifest_path,
    )
    provenance, issues, _ = preflight_official_openai_clip(
        paths, requested_device=config.clip_device
    )
    blockers = [item for item in issues if item.get("severity") == "ERROR"]
    if blockers:
        code = blockers[0].get("code", "CLIP_LOAD_FAILED")
        mapped = "CLIP_ASSET_NOT_FOUND" if "MISSING" in str(code) else "CLIP_LOAD_FAILED"
        raise Stage2RuntimeError(mapped, str(code))
    expected_sha = selected.get("checkpoint_sha256")
    if provenance.get("checkpoint_sha256") != expected_sha:
        raise Stage2RuntimeError("STAGE1B_ENCODER_NOT_VERIFIED", "checkpoint SHA mismatch")
    candidate = replace(
        CandidateContract.from_dict(selected),
        source_root=str(runtime_source),
        checkpoint_path=str(paths.checkpoint_path),
        asset_manifest_path=str(paths.asset_manifest_path),
        device=config.clip_device,
        batch_size=config.clip_batch_size,
    )
    return candidate, provenance


def _known_failures(stage1e_root: Path) -> list[dict[str, Any]]:
    path = stage1e_root / "issues.jsonl"
    if not path.is_file():
        return []
    wanted = {
        "SEMANTIC_RETRIEVAL_FAILURE_AFTER_TRANSLATION",
        "LANGUAGE_BRIDGE_INSUFFICIENT_FOR_SEMANTIC_FAILURE",
    }
    return [
        value
        for value in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines())
        if value.get("code") in wanted
    ]


def _public_preflight(inputs: dict[str, Any]) -> dict[str, Any]:
    selected, adapter = inputs["selected_contract"], inputs["stage1b_adapter"]
    return {
        "status": "READY",
        "stage1_index_status": inputs["stage1_summary"]["index_status"],
        "stage1_index_fingerprint": inputs["stage1_summary"]["index_fingerprint"],
        "stage1b_encoder_status": selected["compatibility_status"],
        "stage1b_candidate_id": selected.get("selected_candidate_id")
        or selected.get("candidate_id"),
        "stage1b_checkpoint_sha256": selected["checkpoint_sha256"],
        "model_space_status": adapter["model_space_status"],
        "stage1e_execution": inputs["stage1e_summary"]["stage1e_execution"],
        "language_path_status": inputs["language_contract"]["language_path_status"],
        "language_contract_sha256": inputs["language_contract_sha256"],
        "translator_asset_status": inputs["translator_asset"]["status"],
        "clip_asset_status": "VALID",
        "network_required": False,
    }


class OperationalRetrievalRuntime:
    """Load-once CPU-first runtime over frozen Stage 1 assets and decisions."""

    def __init__(
        self,
        config: Stage2RuntimeConfig,
        *,
        backend_loader: BackendLoader = load_search_backend,
        clip_preparer: ClipPreparer = _prepare_clip,
        translator_validator: TranslatorValidator = validate_translator_asset,
        encoder_factory: EncoderFactory | None = None,
        translator_factory: TranslatorFactory | None = None,
        dependency_probe: Callable[[], dict[str, Any]] = translator_dependency_versions,
    ) -> None:
        self.config = config
        self.backend_loader = backend_loader
        self.clip_preparer = clip_preparer
        self.translator_validator = translator_validator
        self.encoder_factory = encoder_factory
        self.translator_factory = translator_factory
        self.dependency_probe = dependency_probe
        self.backend: Any = None
        self.catalog: Any = None
        self.encoder: Any = None
        self.translator: Any = None
        self.inputs: dict[str, Any] = {}
        self.preflight: dict[str, Any] = {}
        self.load_latencies_ms: dict[str, float] = {}
        self.completed: list[dict[str, Any]] = []
        self.loaded = False
        self.effective_hardware: dict[str, Any] = {}

    def load(self) -> OperationalRetrievalRuntime:
        if self.loaded:
            return self
        total_started = monotonic()
        hardware = resolve_hardware(
            HardwareConfig(
                mode=self.config.hardware_mode,
                video_backend=self.config.video_backend,
                clip_device=self.config.clip_device,
                translator_device=self.config.translator_device,
                auto_clip_promoted=self.config.auto_clip_promoted,
                auto_translator_promoted=self.config.auto_translator_promoted,
                auto_nvdec_promoted=self.config.auto_nvdec_promoted,
            )
        )
        self.effective_hardware = hardware.as_dict()
        self.config = replace(
            self.config,
            clip_device=hardware.clip_device,
            translator_device=hardware.translator_device,
        )
        inputs = _validate_contracts(self.config)
        try:
            translator_started = monotonic()
            inputs["translator_asset"] = self.translator_validator(
                self.config.translator_asset_root
            )
            translator_contract = inputs["language_contract"]["vietnamese_path"]["translator"]
            if (
                inputs["translator_asset"].get("model_id") != translator_contract["model_id"]
                or inputs["translator_asset"].get("exact_revision")
                != translator_contract["exact_revision"]
            ):
                raise Stage2RuntimeError("TRANSLATOR_REVISION_MISMATCH")
            inputs["translator_dependencies"] = self.dependency_probe()
            self.load_latencies_ms["translator_preflight_ms"] = (
                monotonic() - translator_started
            ) * 1000
        except FileNotFoundError as error:
            raise Stage2RuntimeError("TRANSLATOR_ASSET_NOT_FOUND", str(error)) from error
        except Stage2RuntimeError:
            raise
        except (ImportError, OSError, ValueError) as error:
            code = (
                "TRANSLATOR_REVISION_MISMATCH"
                if "REVISION" in str(error)
                else "TRANSLATOR_LOAD_FAILED"
            )
            raise Stage2RuntimeError(code, str(error)) from error
        candidate, clip_provenance = self.clip_preparer(self.config, inputs["selected_contract"])
        inputs["clip_candidate"] = candidate
        inputs["clip_provenance"] = clip_provenance
        self.inputs = inputs
        if not self.config.translator_lazy_load:
            self._load_translator()
        search_config = SearchConfig(
            inputs["stage1_root"],
            "stage2_preflight",
            top_k=1,
            search_chunk_rows=self.config.search_chunk_rows,
        )
        index_started = monotonic()
        try:
            self.backend, self.catalog = self.backend_loader(search_config)
        except (FileNotFoundError, OSError, TypeError, ValueError) as error:
            raise Stage2RuntimeError("STAGE1_INDEX_NOT_READY", str(error)) from error
        self.load_latencies_ms["index_load_ms"] = (monotonic() - index_started) * 1000
        encoder_started = monotonic()
        try:
            self.encoder = (
                self.encoder_factory(candidate)
                if self.encoder_factory is not None
                else load_multimodal_encoder(candidate, provenance=clip_provenance)
            )
        except (FileNotFoundError, ImportError, OSError, RuntimeError, ValueError) as error:
            raise Stage2RuntimeError("CLIP_LOAD_FAILED", str(error)) from error
        self.load_latencies_ms["clip_load_ms"] = (monotonic() - encoder_started) * 1000
        self.load_latencies_ms["total_load_ms"] = (monotonic() - total_started) * 1000
        self.preflight = _public_preflight(inputs)
        self.loaded = True
        self.config.output_root.mkdir(parents=True, exist_ok=True)
        write_json(self.config.output_root / "preflight.json", self.preflight)
        write_json(self.config.output_root / "runtime_manifest.json", self.runtime_manifest())
        self._write_session_artifacts()
        return self

    def _load_translator(self) -> Any:
        if self.translator is not None:
            return self.translator
        started = monotonic()
        factory = self.translator_factory or (
            lambda model_root, translator_config, generation: OfflineViEnTranslator(
                model_root, translator_config, generation
            )
        )
        try:
            self.translator = factory(
                Path(self.inputs["translator_asset"]["model_root"]),
                self.inputs["translator_config"],
                self.inputs["generation_config"],
            )
            self.translator.load()
        except (FileNotFoundError, ImportError, OSError, RuntimeError, ValueError) as error:
            raise Stage2RuntimeError("TRANSLATOR_LOAD_FAILED", str(error)) from error
        self.load_latencies_ms["translator_load_ms"] = (monotonic() - started) * 1000
        return self.translator

    def search_one(self, request: QueryRequest) -> QueryResult:
        return self.search_many([request])[0]

    def search(self, request: QueryRequest | list[QueryRequest]) -> QueryResult | list[QueryResult]:
        """Search one request or an explicitly bounded in-process batch."""

        return self.search_many(request) if isinstance(request, list) else self.search_one(request)

    def encode_requests(self, requests: list[QueryRequest]) -> EncodedQueryBatch:
        """Apply the frozen Stage 2A language path and verified CLIP encoder."""

        if not self.loaded:
            raise RuntimeError("OperationalRetrievalRuntime.load() must be called first")
        if not requests:
            raise ValueError("encode_requests requires at least one request")
        batch_started = monotonic()
        resolutions: list[LanguageResolution] = []
        language_ms: list[float] = []
        for request in requests:
            route_started = monotonic()
            resolutions.append(resolve_language(request))
            language_ms.append((monotonic() - route_started) * 1000)
        translated: dict[int, dict[str, Any]] = {}
        vi_indices = [
            index for index, value in enumerate(resolutions) if value.resolved_language == "vi"
        ]
        if vi_indices:
            translator = self._load_translator()
            try:
                values = translator.translate(
                    [requests[index].text.strip() for index in vi_indices]
                )
            except ValueError as error:
                raise Stage2RuntimeError("TRANSLATION_EMPTY", str(error)) from error
            except RuntimeError as error:
                raise Stage2RuntimeError("TRANSLATION_FAILED", str(error)) from error
            translated = dict(zip(vi_indices, values, strict=True))
        clip_inputs = [
            translated[index]["translated_text_for_clip"]
            if index in translated
            else request.text.strip()
            for index, request in enumerate(requests)
        ]
        clip_started = monotonic()
        try:
            embeddings = np.asarray(self.encoder.encode_text(clip_inputs), dtype=np.float32)
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            raise Stage2RuntimeError("CLIP_ENCODING_FAILED", str(error)) from error
        clip_elapsed_ms = (monotonic() - clip_started) * 1000
        if embeddings.shape != (len(requests), 512) or not np.isfinite(embeddings).all():
            raise Stage2RuntimeError("CLIP_ENCODING_FAILED", "expected finite (n, 512)")
        norms = np.linalg.norm(embeddings, axis=1)
        if not np.allclose(norms, 1.0, rtol=0.0, atol=1e-4):
            raise Stage2RuntimeError("CLIP_ENCODING_FAILED", "embeddings are not normalized")
        encodings = []
        latencies = []
        for index, request in enumerate(requests):
            translation = translated.get(index)
            encodings.append(
                {
                    "original_query_text": request.text.strip(),
                    "resolved_language": resolutions[index].resolved_language,
                    "translation_applied": translation is not None,
                    "translated_text": (
                        translation["translated_text_for_clip"] if translation else None
                    ),
                    "clip_input_text": clip_inputs[index],
                    "clip_candidate_id": self.preflight["stage1b_candidate_id"],
                    "embedding_dimension": 512,
                    "embedding_finite": True,
                    "embedding_normalized": True,
                }
            )
            latencies.append(
                {
                    "language_resolution_ms": language_ms[index],
                    "translation_ms": float(
                        translation.get("translation_latency_ms", 0.0) if translation else 0.0
                    ),
                    "clip_encoding_ms": clip_elapsed_ms / len(requests),
                }
            )
        return EncodedQueryBatch(
            embeddings,
            tuple(resolutions),
            tuple(encodings),
            tuple(latencies),
            (monotonic() - batch_started) * 1000,
        )

    def search_many(self, requests: list[QueryRequest]) -> list[QueryResult]:
        if not self.loaded:
            raise RuntimeError("OperationalRetrievalRuntime.load() must be called first")
        if not requests:
            return []
        if len({item.query_id for item in requests}) != len(requests):
            raise ValueError("query_id values must be unique within a batch")
        if any(item.top_k > self.config.max_top_k for item in requests):
            raise ValueError(f"top_k exceeds configured maximum {self.config.max_top_k}")
        started = [monotonic() for _ in requests]
        encoded = self.encode_requests(requests)
        results = []
        for index, request in enumerate(requests):
            try:
                frames, search_seconds = rank_loaded_query(
                    encoded.embeddings[index : index + 1],
                    self.backend,
                    self.catalog,
                    top_k=request.top_k,
                    metric="cosine",
                    query_id=request.query_id,
                    encoder_status="VERIFIED",
                )
            except (IndexError, RuntimeError, TypeError, ValueError) as error:
                raise Stage2RuntimeError("SEARCH_FAILED", str(error)) from error
            encoding = encoded.encodings[index]
            latencies = {
                **encoded.latencies_ms[index],
                "search_ms": search_seconds * 1000,
                "artifact_write_ms": 0.0,
                "total_ms": (monotonic() - started[index]) * 1000,
            }
            query_root, videos, artifact_ms = write_operational_query_artifacts(
                self.config.output_root,
                request,
                encoded.resolutions[index],
                encoding,
                frames,
                latencies,
            )
            latencies["artifact_write_ms"] = artifact_ms
            latencies["total_ms"] = (monotonic() - started[index]) * 1000
            result = QueryResult(
                request.query_id,
                asdict(request),
                encoded.resolutions[index].as_dict(),
                encoding,
                frames,
                videos,
                latencies,
                query_root,
            )
            results.append(result)
            self.completed.append(
                {
                    "query_id": request.query_id,
                    "requested_language": request.language,
                    "resolved_language": encoded.resolutions[index].resolved_language,
                    "translation_applied": encoding["translation_applied"],
                    "top_k": request.top_k,
                    "results": len(frames),
                    "latencies_ms": latencies,
                }
            )
        self._write_session_artifacts()
        write_json(self.config.output_root / "runtime_manifest.json", self.runtime_manifest())
        return results

    def runtime_manifest(self) -> dict[str, Any]:
        selected = self.inputs.get("selected_contract", {})
        translator_asset = self.inputs.get("translator_asset", {})
        return {
            "stage2_version": STAGE2_VERSION,
            "status": "LOADED" if self.loaded else "NOT_LOADED",
            "build_git_commit": self.config.build_git_commit,
            "stage1_index_fingerprint": self.inputs.get("stage1_summary", {}).get(
                "index_fingerprint"
            ),
            "stage1b": {
                "candidate_id": selected.get("selected_candidate_id")
                or selected.get("candidate_id"),
                "checkpoint_sha256": selected.get("checkpoint_sha256"),
                "compatibility_status": selected.get("compatibility_status"),
                "model_space_status": self.inputs.get("stage1b_adapter", {}).get(
                    "model_space_status"
                ),
            },
            "stage1e_language_contract_sha256": self.inputs.get("language_contract_sha256"),
            "translator": {
                "model_id": translator_asset.get("model_id"),
                "exact_revision": translator_asset.get("exact_revision"),
                "asset_status": translator_asset.get("status"),
                "runtime_files": translator_asset.get("runtime_files", []),
                "generation_config": (
                    asdict(self.inputs["generation_config"])
                    if "generation_config" in self.inputs
                    else None
                ),
                "generation_config_sha256": self.inputs.get("generation_config_sha256"),
                "lazy_load": self.config.translator_lazy_load,
                "loaded": self.translator is not None,
            },
            "devices": {
                "clip": self.config.clip_device,
                "translator": self.config.translator_device,
                "index": "cpu_numpy_exact",
            },
            "hardware": dict(self.effective_hardware),
            "load_latencies_ms": dict(self.load_latencies_ms),
            "assets_loaded_once": True,
            "network_required": False,
            "ranking_policy": "FROZEN_STAGE1A_EXACT_COSINE_NO_RERANKING",
            "known_failure_modes": _known_failures(self.config.stage1e_root),
        }

    def _write_session_artifacts(self) -> None:
        known = _known_failures(self.config.stage1e_root)
        write_jsonl(self.config.output_root / "issues.jsonl", known)
        write_jsonl(self.config.output_root / "smoke_results.jsonl", self.completed)
        totals = [item["latencies_ms"]["total_ms"] for item in self.completed]
        write_json(
            self.config.output_root / "latency_summary.json",
            {
                "queries_completed": len(self.completed),
                "total_ms": {
                    "min": min(totals) if totals else None,
                    "max": max(totals) if totals else None,
                    "mean": sum(totals) / len(totals) if totals else None,
                },
                "optimization_status": "BASELINE_ONLY",
            },
        )
        write_json(
            self.config.output_root / "run_manifest.json",
            {
                "stage2_version": STAGE2_VERSION,
                "status": "COMPLETE" if self.completed else "READY",
                "build_git_commit": self.config.build_git_commit,
                "updated_at": datetime.now(UTC).isoformat(),
                "queries_completed": len(self.completed),
                "no_stage0_rerun": True,
                "no_stage1_rebuild": True,
                "no_model_download": True,
                "no_reranking": True,
                "ranking_quality_status": "UNCHANGED_FROM_FROZEN_BASELINE",
                "known_failure_modes": known,
            },
        )

    def close(self) -> None:
        for component in (self.translator, self.encoder):
            closer = getattr(component, "close", None)
            if callable(closer):
                closer()
        self.translator = None
        self.encoder = None
        self.backend = None
        self.catalog = None
        self.loaded = False


def preflight_stage2(
    config: Stage2RuntimeConfig,
    **runtime_dependencies: Any,
) -> dict[str, Any]:
    runtime = OperationalRetrievalRuntime(config, **runtime_dependencies)
    try:
        runtime.load()
        return runtime.preflight
    finally:
        runtime.close()


__all__ = ["EncodedQueryBatch", "OperationalRetrievalRuntime", "preflight_stage2"]
