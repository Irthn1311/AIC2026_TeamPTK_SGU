"""Long-lived operational KIS runtime engine and lifecycle manager."""

from __future__ import annotations

import csv
import dataclasses
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

import numpy as np

from system_tai.checkpointing.exporter import CheckpointExporter
from system_tai.common.schemas import KISResult
from system_tai.data.corpus_discovery import (
    CorpusManifest,
    DiscoveryMetrics,
    discover_corpus,
    load_corpus_manifest,
    load_or_build_manifest_cache,
)
from system_tai.evidence.object_artifacts import (
    ObjectArtifactIndex,
    resolve_object_artifact_root,
)
from system_tai.features.btc_clip_store import FeatureStoreRegistry
from system_tai.features.query_encoder import SharedOpenAIClipEncoder
from system_tai.kis.benchmark import resolve_device
from system_tai.kis.contest_runner import _git_commit_hash, _validation_payload
from system_tai.kis.session_schema import (
    DuplicateRequestIdError,
    HealthRequest,
    QAQueryRequest,
    QueryLanguage,
    QueryRequest,
    QueryVariant,
    QueryVariantType,
    SessionConfig,
    ShutdownRequest,
    TRAKEQueryRequest,
)
from system_tai.kis.video_first import (
    KIS_SEMANTIC_VIDEO_FIRST,
    build_kis_video_first_outcome,
    fuse_video_maxima,
    fuse_video_maxima_v2,
    fuse_video_maxima_v2_paraphrase_ensemble,
)
from system_tai.preliminary.runtime_bridge import (
    audit_runtime_top100_artifact,
    kis_result_to_top100_query,
    qa_predictions_to_top100_query,
    trake_predictions_to_top100_query,
)
from system_tai.qa.object_provider import (
    QUERY_CONDITIONED_FRAME_RANKING,
    ObjectEntityAnswerProvider,
)
from system_tai.qa.ocr_provider import (
    OCRAnswerProvider,
    TesseractCLIBackend,
)
from system_tai.qa.runtime import QARuntimePipeline
from system_tai.qa.visual_ontology import (
    VisualOntologyAnswerCandidateProvider,
    load_visual_answer_ontology,
)
from system_tai.refinement.engine import ExactFrameRefiner, FrameEmbeddingCache
from system_tai.refinement.models import Phase3Candidate, RefinementQuery
from system_tai.refinement.q3_anchor import (
    integrate_q3_anchor_refinements,
    select_q3_anchor_candidates,
)
from system_tai.refinement.runner import _write_json, _write_refined_csv
from system_tai.refinement.video import OpenCVVideoDecoder, RawVideoRegistry
from system_tai.refinement.visual_verifier import (
    HuggingFaceStructuredVisualVerifier,
    StructuredVisualVerifier,
)
from system_tai.retrieval.multi_query import WeightedRRFRetriever
from system_tai.retrieval.semantic_query import (
    CompiledParaphraseEnsemble,
    CompiledParaphraseGroup,
    CompiledSemanticQuery,
    CompiledSemanticVariant,
    SemanticQueryConfig,
    allocate_hierarchical_quotas,
    compile_vietnamese_semantic_query,
    compute_normalized_ensemble_weights,
    decompose_vietnamese_semantic_units,
)
from system_tai.retrieval.vector_search import ExactNumpyRetriever
from system_tai.retrieval.video_evidence import VideoRestrictedFeatureSearcher
from system_tai.retrieval.video_restricted import (
    VIDEO_CONDITIONED_KEYFRAME_DIVERSITY,
    VideoConditionedKeyframeDiversity,
)
from system_tai.trake.engine import TRAKEEngine
from system_tai.trake.runtime import TRAKERuntimePipeline
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
        visual_verifier: StructuredVisualVerifier | None = None,
        exporter: CheckpointExporter | None = None,
        validator: CheckpointValidator | None = None,
        object_answer_provider: ObjectEntityAnswerProvider | None = None,
        ocr_answer_provider: OCRAnswerProvider | None = None,
        translation_provider: Any = None,
        token_budget_guard: Any = None,
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
        self.visual_verifier = visual_verifier
        self.exporter = exporter or CheckpointExporter()
        self.validator = validator or CheckpointValidator()
        self.clock = clock
        self.bootstrap_timings = dict(bootstrap_timings or {})

        self.object_artifact_index: ObjectArtifactIndex | None = None
        if config.qa_object_answer_provider_config.enabled:
            if object_answer_provider is None:
                object_root = resolve_object_artifact_root(self.manifest.dataset_root)
                root_identity = object_root.relative_to(
                    self.manifest.dataset_root.resolve(strict=False)
                ).as_posix()
                self.object_artifact_index = ObjectArtifactIndex(
                    object_root=object_root,
                    mappings_by_video={
                        store.descriptor.video_id: store.mappings
                        for store in self.registry.stores
                    },
                    source_root_identity=root_identity,
                )
                object_answer_provider = ObjectEntityAnswerProvider(
                    index=self.object_artifact_index,
                    config=config.qa_object_answer_provider_config,
                    text_encoder=(
                        self.shared_encoder
                        if config.qa_object_answer_provider_config.ranking_policy
                        == QUERY_CONDITIONED_FRAME_RANKING
                        else None
                    ),
                )
            else:
                self.object_artifact_index = object_answer_provider.index
        self.object_answer_provider = object_answer_provider

        if config.qa_ocr_answer_provider_config.enabled and ocr_answer_provider is None:
            ocr_answer_provider = OCRAnswerProvider(
                backend=TesseractCLIBackend(config.qa_ocr_answer_provider_config),
                config=config.qa_ocr_answer_provider_config,
                clock=clock,
            )
        self.ocr_answer_provider = ocr_answer_provider

        self.visual_ontology_provider: (
            VisualOntologyAnswerCandidateProvider | None
        ) = None
        if config.qa_visual_ontology_config.enabled:
            assert config.qa_visual_ontology_config.ontology_path is not None
            self.visual_ontology_provider = VisualOntologyAnswerCandidateProvider(
                load_visual_answer_ontology(
                    config.qa_visual_ontology_config.ontology_path
                ),
                config.qa_visual_ontology_config,
            )

        self.exact_retriever = ExactNumpyRetriever(
            registry=self.registry,
            text_encoder=self.shared_encoder,
            chunk_size=config.chunk_size,
        )
        self.weighted_rrf = WeightedRRFRetriever(self.exact_retriever)
        self.video_restricted_searcher = VideoRestrictedFeatureSearcher(
            self.registry,
            chunk_size=config.chunk_size,
        )
        self.video_conditioner = VideoConditionedKeyframeDiversity(
            self.registry,
            clock=self.clock,
        )
        self.refiner = ExactFrameRefiner(
            raw_videos=self.raw_video_registry,
            decoder=self.decoder,
            encoder=self.shared_encoder,
            visual_verifier=self.visual_verifier,
            clock=self.clock,
        )

        self.qa_pipeline = QARuntimePipeline(
            exact_retriever=self.exact_retriever,
            weighted_rrf=self.weighted_rrf,
            refiner=self.refiner,
            raw_video_registry=self.raw_video_registry,
            decoder=self.decoder,
            shared_encoder=self.shared_encoder,
            video_restricted_searcher=self.video_restricted_searcher,
            video_conditioned_evidence_config=(
                config.qa_video_conditioned_evidence_config
            ),
            candidate_provider=self.visual_ontology_provider,
            object_answer_provider=self.object_answer_provider,
            ocr_answer_provider=self.ocr_answer_provider,
            allow_unsupported_provider_fallback=config.qa_unsupported_provider_fallback,
            clock=self.clock,
        )

        self.trake_engine = TRAKEEngine()
        self.trake_pipeline = TRAKERuntimePipeline(
            exact_retriever=self.exact_retriever,
            weighted_rrf=self.weighted_rrf,
            refiner=self.refiner,
            shared_encoder=self.shared_encoder,
            video_restricted_searcher=self.video_restricted_searcher,
            trake_engine=self.trake_engine,
            clock=self.clock,
        )

        self.translation_provider: Any | None = None
        self.token_budget_guard: Any | None = None
        if config.enable_dynamic_translation:
            from system_tai.translation.provider import (
                GoogleTranslateProvider,
                TokenBudgetGuard,
                VinAITranslateProvider,
            )

            cache_path = None
            if config.translation_cache_dir:
                cache_path = Path(config.translation_cache_dir) / "translation_cache.json"
            elif Path("/kaggle/working").exists():
                cache_path = Path("/kaggle/working/translation_cache.json")
            else:
                cache_path = Path(config.output_root) / "translation_cache.json"

            if translation_provider is not None:
                self.translation_provider = translation_provider
            elif (
                config.translation_model_name == "google-translate"
                or "google" in config.translation_model_name.lower()
                or getattr(config, "use_google_translation", False)
            ):
                self.translation_provider = GoogleTranslateProvider(
                    cache_path=cache_path,
                    enable_network=True,
                )
            else:
                self.translation_provider = VinAITranslateProvider(
                    model_name_or_path=config.translation_model_name,
                    device=config.translation_device,
                    cache_dir=config.translation_cache_dir,
                    allow_model_download=config.translation_allow_model_download,
                    revision=config.translation_revision,
                )

            if token_budget_guard is not None:
                self.token_budget_guard = token_budget_guard
            else:
                self.token_budget_guard = TokenBudgetGuard(
                    max_tokens=config.translation_max_clip_tokens
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
        visual_verifier_factory: Callable[..., StructuredVisualVerifier] | None = None,
        exporter: CheckpointExporter | None = None,
        validator: CheckpointValidator | None = None,
        object_answer_provider: ObjectEntityAnswerProvider | None = None,
        ocr_answer_provider: OCRAnswerProvider | None = None,
        translation_provider: Any = None,
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
            discovery_metrics = (
                manifest.discovery_metrics
                if cached.status != "CACHE_HIT"
                else DiscoveryMetrics()
            )
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
        visual_config = config.selected_video_visual_verifier_config
        if visual_config.device != resolved_device:
            visual_config = dataclasses.replace(visual_config, device=resolved_device)
        exec_config = dataclasses.replace(
            config,
            device=resolved_device,
            refinement_config=ref_config,
            selected_video_visual_verifier_config=visual_config,
        )

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

        visual_verifier = None
        visual_verifier_model_seconds = 0.0
        visual_config = exec_config.selected_video_visual_verifier_config
        if visual_config.enabled:
            visual_model_start = clock()
            visual_factory = (
                visual_verifier_factory or HuggingFaceStructuredVisualVerifier
            )
            visual_verifier = visual_factory(
                model_name=visual_config.model_name,
                revision=visual_config.model_revision,
                device=resolved_device,
                allow_model_download=visual_config.allow_model_download,
                cache_dir=visual_config.cache_dir,
                max_new_tokens=visual_config.effective_max_new_tokens,
                **(
                    {
                        "max_image_pixels": visual_config.effective_max_image_pixels,
                        "execution_profile": (
                            "cpu-fast"
                            if visual_config.cpu_fast_profile_applied
                            else "full"
                        ),
                        "progress_callback": lambda message: print(
                            message,
                            file=sys.stderr,
                            flush=True,
                        ),
                    }
                    if visual_verifier_factory is None
                    else {}
                ),
            )
            visual_verifier_model_seconds = clock() - visual_model_start

        bootstrap_timings = {
            "discovery_seconds": discovery_seconds,
            "manifest_load_or_build_seconds": manifest_seconds,
            "manifest_write_seconds": manifest_write_seconds,
            "registry_load_seconds": registry_seconds,
            "model_load_seconds": model_seconds,
            "visual_verifier_model_load_seconds": visual_verifier_model_seconds,
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
            visual_verifier=visual_verifier,
            exporter=exporter,
            validator=validator,
            object_answer_provider=object_answer_provider,
            ocr_answer_provider=ocr_answer_provider,
            translation_provider=translation_provider,
            clock=clock,
            bootstrap_timings=bootstrap_timings,
        )

    def handle_health(self, request: HealthRequest) -> dict[str, Any]:
        if request.request_id in self._seen_request_ids:
            raise DuplicateRequestIdError(
                f"request_id '{request.request_id}' has already been processed"
            )
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
            raise DuplicateRequestIdError(
                f"request_id '{request.request_id}' has already been processed"
            )
        self._seen_request_ids.add(request.request_id)
        self._request_count += 1

        req_start = self.clock()
        query_dir = self.output_root / "requests" / safe_request_directory_name(request.request_id)
        if query_dir.exists():
            query_dir = (
                self.output_root
                / "requests"
                / f"{safe_request_directory_name(request.request_id)}-{uuid.uuid4().hex[:4]}"
            )
        query_dir.mkdir(parents=True, exist_ok=True)

        validation_start = self.clock()
        translation_seconds = 0.0
        translation_metadata: dict[str, Any] = {"dynamic_translation_enabled": False}
        video_first_enabled = self.config.kis_video_first_config.enabled
        compiled_semantic_query = None
        compiled_paraphrase_ensemble: CompiledParaphraseEnsemble | None = None
        variant_quotas_mapping: dict[str, int] | None = None

        if video_first_enabled:
            if self.translation_provider is None or self.token_budget_guard is None:
                raise ValueError(
                    "KIS semantic video-first retrieval requires a translation provider"
                )
            if request.query_en or request.query_en_expansion:
                raise ValueError(
                    "KIS semantic video-first retrieval accepts Vietnamese input only; "
                    "manual English variants are not allowed"
                )
            if callable(getattr(self.translation_provider, "validate_query_vi", None)):
                self.translation_provider.validate_query_vi(request.query_id, request.query_vi)
            t_trans = self.clock()
            video_first_config = self.config.kis_video_first_config
            sem_config = SemanticQueryConfig(
                full_query_weight=video_first_config.full_query_weight,
                primary_scene_weight=video_first_config.primary_scene_weight,
                supporting_attribute_weight=(
                    video_first_config.supporting_attribute_weight
                ),
            )

            if (
                video_first_config.enable_paraphrase_ensemble
                and hasattr(self.translation_provider, "get_paraphrase_groups")
            ):
                paraphrase_groups_data = self.translation_provider.get_paraphrase_groups(request.query_id)
                n_groups = len(paraphrase_groups_data)
                if n_groups == 0:
                    raise ValueError(f"No paraphrase groups found for query '{request.query_id}'")

                compiled_groups: list[CompiledParaphraseGroup] = []
                compiled_queries_list: list[CompiledSemanticQuery] = []

                for grp in paraphrase_groups_data:
                    gid = grp["group_id"]
                    src_text = grp.get("source_text", request.query_vi)

                    units = decompose_vietnamese_semantic_units(
                        query_id=request.query_id,
                        query_vi=src_text,
                        config=sem_config,
                    )
                    translations: list[str] = []
                    for u in units:
                        en_text = self.translation_provider.translate_unit(
                            query_id=request.query_id,
                            group_id=gid,
                            vi_text=u.text,
                        )
                        translations.append(en_text)

                    compiled_vars: list[CompiledSemanticVariant] = []
                    for unit_index, (unit, raw_english) in enumerate(
                        zip(units, translations, strict=True),
                        start=1,
                    ):
                        raw_english = raw_english.strip()
                        if not raw_english:
                            raise ValueError(f"translation for {unit.unit_id} in group {gid} is empty")
                        segments = tuple(self.token_budget_guard.split_for_clip(raw_english))
                        if not segments:
                            raise ValueError(f"translation for {unit.unit_id} produced no CLIP segments")
                        segment_weight = unit.weight / len(segments)
                        for segment_index, segment in enumerate(segments, start=1):
                            var_id = (
                                f"{request.query_id}::{gid}::semantic_{unit_index:02d}_s{segment_index:02d}"
                                if n_groups > 1
                                else f"{request.query_id}::semantic_{unit_index:02d}_s{segment_index:02d}"
                            )
                            compiled_vars.append(
                                CompiledSemanticVariant(
                                    query_variant=QueryVariant(
                                        variant_id=var_id,
                                        text=segment,
                                        language=QueryLanguage.ENGLISH,
                                        variant_type=QueryVariantType.ENGLISH_TRANSLATION,
                                        weight=segment_weight,
                                    ),
                                    semantic_unit_id=unit.unit_id,
                                    semantic_role=unit.role,
                                    source_vietnamese=unit.text,
                                    raw_english=raw_english,
                                    segment_index=segment_index,
                                    segment_count=len(segments),
                                    clip_token_count=self.token_budget_guard.count_tokens(segment),
                                    temporal_index=unit.temporal_index,
                                )
                            )

                    cq = CompiledSemanticQuery(
                        query_id=request.query_id,
                        source_vietnamese=src_text,
                        units=units,
                        variants=tuple(compiled_vars),
                        provider_name=getattr(self.translation_provider, "sidecar_id", "paraphrase_ensemble_sidecar"),
                    )
                    compiled_queries_list.append(cq)
                    compiled_groups.append(
                        CompiledParaphraseGroup(
                            group_id=gid,
                            source_text=src_text,
                            compiled_query=cq,
                            group_weight_mass=1.0 / n_groups,
                        )
                    )

                w0 = sum(v.query_variant.weight for v in compiled_queries_list[0].variants)
                normalized_weights = compute_normalized_ensemble_weights(
                    compiled_queries_list,
                    baseline_weight_mass=w0,
                )

                v0_count = len(compiled_queries_list[0].variants)
                b_sem_c1 = 20 * v0_count
                quotas_c1 = allocate_hierarchical_quotas(
                    compiled_queries_list,
                    total_semantic_budget=b_sem_c1,
                )

                total_all_vars = sum(len(cq.variants) for cq in compiled_queries_list)
                b_sem_c2 = 20 * total_all_vars
                quotas_c2 = {
                    v.query_variant.variant_id: 20
                    for cq in compiled_queries_list
                    for v in cq.variants
                }

                compiled_paraphrase_ensemble = CompiledParaphraseEnsemble(
                    query_id=request.query_id,
                    source_vietnamese=request.query_vi,
                    groups=tuple(compiled_groups),
                    normalized_weights=normalized_weights,
                    hierarchical_quotas_c1=quotas_c1,
                    hierarchical_quotas_c2=quotas_c2,
                    provider_name=getattr(self.translation_provider, "sidecar_id", "paraphrase_ensemble_sidecar"),
                    baseline_weight_mass=w0,
                    total_semantic_budget_c1=b_sem_c1,
                    total_semantic_budget_c2=b_sem_c2,
                )
                variants = compiled_paraphrase_ensemble.all_variants
                compiled_semantic_query = compiled_queries_list[0]
                translation_seconds = self.clock() - t_trans
                translation_metadata = compiled_paraphrase_ensemble.to_metadata()
                translation_metadata["translation_seconds"] = translation_seconds

                if video_first_config.paraphrase_ensemble_mode == "EQUAL_BUDGET":
                    variant_quotas_mapping = dict(quotas_c1)
                    active_sem_budget = b_sem_c1
                else:
                    variant_quotas_mapping = dict(quotas_c2)
                    active_sem_budget = b_sem_c2

                # Pre-retrieval golden verification per group
                if hasattr(self.translation_provider, "expected_group_hashes"):
                    has_real_clip = False
                    try:
                        import clip
                        has_real_clip = True
                    except ImportError:
                        pass

                    if has_real_clip:
                        exp_hashes = self.translation_provider.expected_group_hashes(request.query_id)
                        exp_var_counts = self.translation_provider.expected_group_variant_counts(request.query_id)

                        for grp in compiled_groups:
                            gid = grp.group_id
                            exp_hash = exp_hashes.get(gid)
                            exp_var_count = exp_var_counts.get(gid)

                            if exp_hash is None:
                                raise RuntimeError(
                                    f"Pre-retrieval golden verification failed: missing expected hash for query '{request.query_id}', group '{gid}'"
                                )

                            semantic_payload = [
                                {
                                    "variant_id": (
                                        v.query_variant.variant_id.replace(f"::{gid}::", "::")
                                        if f"::{gid}::" in v.query_variant.variant_id
                                        else v.query_variant.variant_id
                                    ),
                                    "text": v.query_variant.text,
                                    "weight": v.query_variant.weight,
                                }
                                for v in grp.compiled_query.variants
                            ]
                            actual_sha = hashlib.sha256(
                                json.dumps(
                                    semantic_payload,
                                    ensure_ascii=False,
                                    sort_keys=True,
                                    separators=(",", ":"),
                                ).encode("utf-8")
                            ).hexdigest()[:12]

                            if actual_sha.lower() != exp_hash.lower():
                                raise RuntimeError(
                                    f"Pre-retrieval golden hash mismatch for query '{request.query_id}', group '{gid}': "
                                    f"expected {exp_hash}, got {actual_sha}"
                                )
                            if isinstance(exp_var_count, int) and not isinstance(exp_var_count, bool) and len(semantic_payload) != exp_var_count:
                                raise RuntimeError(
                                    f"Pre-retrieval variant count mismatch for query '{request.query_id}', group '{gid}': "
                                    f"expected {exp_var_count}, got {len(semantic_payload)}"
                                )

                if hasattr(self.translation_provider, "sidecar_metadata"):
                    translation_metadata["sidecar_telemetry"] = self.translation_provider.sidecar_metadata()
            else:
                compiled_semantic_query = compile_vietnamese_semantic_query(
                    query_id=request.query_id,
                    query_vi=request.query_vi,
                    provider=self.translation_provider,
                    token_budget_guard=self.token_budget_guard,
                    config=sem_config,
                )
                variants = compiled_semantic_query.query_variants
                translation_seconds = self.clock() - t_trans
                translation_metadata = compiled_semantic_query.to_metadata()
                translation_metadata["translation_seconds"] = translation_seconds
                active_sem_budget = video_first_config.restricted_frames_per_video_per_variant * len(variants)
                variant_quotas_mapping = {
                    v.variant_id: video_first_config.restricted_frames_per_video_per_variant
                    for v in variants
                }

                # Pre-retrieval golden verification when running with immutable sidecar
                if callable(getattr(self.translation_provider, "expected_semantic_hash", None)):
                    has_real_clip = False
                    try:
                        import clip
                        has_real_clip = True
                    except ImportError:
                        pass

                    if has_real_clip:
                        exp_hash = self.translation_provider.expected_semantic_hash(request.query_id)
                        exp_var_count = self.translation_provider.expected_variant_count(request.query_id)

                        u_meta = translation_metadata.get("units", [])
                        semantic_payload = [
                            {
                                "variant_id": seg.get("variant_id"),
                                "text": seg.get("text"),
                                "weight": seg.get("weight"),
                            }
                            for unit in u_meta
                            for seg in unit.get("segments", [])
                        ]
                        actual_compiled_sha = hashlib.sha256(
                            json.dumps(
                                semantic_payload,
                                ensure_ascii=False,
                                sort_keys=True,
                                separators=(",", ":"),
                            ).encode("utf-8")
                        ).hexdigest()[:12]

                        if isinstance(exp_hash, str) and actual_compiled_sha.lower() != exp_hash.lower():
                            raise RuntimeError(
                                f"Pre-retrieval golden hash mismatch for query '{request.query_id}': "
                                f"expected {exp_hash}, got {actual_compiled_sha}"
                            )
                        if isinstance(exp_var_count, int) and not isinstance(exp_var_count, bool) and len(semantic_payload) != exp_var_count:
                            raise RuntimeError(
                                f"Pre-retrieval variant count mismatch for query '{request.query_id}': "
                                f"expected {exp_var_count}, got {len(semantic_payload)}"
                            )

                if callable(getattr(self.translation_provider, "sidecar_metadata", None)):
                    sidecar_meta = self.translation_provider.sidecar_metadata()
                    if isinstance(sidecar_meta, dict):
                        translation_metadata["sidecar_telemetry"] = sidecar_meta
        elif self.config.enable_dynamic_translation and self.translation_provider is not None:
            t_trans = self.clock()
            raw_en = self.translation_provider.translate(request.query_vi)
            english_segments = self.token_budget_guard.split_for_clip(raw_en)
            clip_token_counts = tuple(
                self.token_budget_guard.count_tokens(segment)
                for segment in english_segments
            )
            translation_seconds = self.clock() - t_trans
            translation_metadata = {
                "dynamic_translation_enabled": True,
                "provider": getattr(self.translation_provider, "provider_name", "dynamic_vinai"),
                "source_vietnamese": request.query_vi,
                "raw_english": raw_en,
                "english_segments": list(english_segments),
                "clip_token_counts": list(clip_token_counts),
                "segment_count": len(english_segments),
                "lossless_segmentation": True,
                "was_truncated": False,
                "translation_seconds": translation_seconds,
            }
            # EN_ONLY variants. Long translations are losslessly segmented,
            # then fused at ranking level; Vietnamese is not mixed into CLIP.
            variants = tuple(
                QueryVariant(
                    variant_id=f"{request.query_id}::vinai_en_{index:02d}",
                    text=segment,
                    language=QueryLanguage.ENGLISH,
                    variant_type=QueryVariantType.ENGLISH_TRANSLATION,
                    weight=1.0,
                )
                for index, segment in enumerate(english_segments, start=1)
            )
        else:
            variants = request.variants()
        validation_seconds = self.clock() - validation_start

        if not variants:
            raise RuntimeError(f"No query variants produced for query_id '{request.query_id}'")

        text_encode_start = self.clock()
        texts = [variant.text for variant in variants]
        variant_embeddings = self.shared_encoder.encode_texts(texts)
        if variant_embeddings.shape[0] != len(variants):
            raise ValueError(
                "Batch text encode returned "
                f"{variant_embeddings.shape[0]} rows for {len(variants)} variants"
            )
        text_encode_seconds = self.clock() - text_encode_start

        retrieval_start = self.clock()
        video_first_trace: dict[str, Any] = {
            "policy": KIS_SEMANTIC_VIDEO_FIRST,
            "enabled": False,
        }
        full_corpus_video_search_seconds = 0.0
        video_fusion_seconds = 0.0
        restricted_frame_search_seconds = 0.0
        restricted_frame_fusion_seconds = 0.0
        video_first_full_corpus_rows_scored = 0
        video_first_full_corpus_store_scan_count = 0
        video_first_restricted_rows_scored = 0
        video_first_restricted_store_scan_count = 0
        selected_videos = ()
        video_first_outcome = None
        restricted = None
        retrieval_seconds = 0.0

        if video_first_enabled:
            # Step 1: Video-level search
            maxima_started = self.clock()
            primary_var_ids = (
                compiled_semantic_query.primary_variant_ids
                if compiled_semantic_query is not None
                else frozenset([variants[0].variant_id] if variants else [])
            )
            supporting_var_ids = (
                compiled_semantic_query.supporting_variant_ids
                if compiled_semantic_query is not None
                else frozenset(v.variant_id for v in variants[1:])
            )
            temporal_var_list = (
                tuple(item.query_variant for item in compiled_semantic_query.temporal_scene_variants)
                if compiled_semantic_query is not None
                else ()
            )

            maxima = self.video_restricted_searcher.search_video_maxima(
                query_ids=tuple(variant.variant_id for variant in variants),
                query_vectors=variant_embeddings,
                top_m_evidence_cap=self.config.kis_video_first_config.top_m_evidence_cap,
                top_m_weights=self.config.kis_video_first_config.top_m_weights,
                top_m_min_frame_gap=self.config.kis_video_first_config.top_m_min_frame_gap,
            )
            full_corpus_video_search_seconds = self.clock() - maxima_started

            # Step 2: Fuse video rankings
            video_fusion_started = self.clock()
            v2_adaptive = self.config.kis_video_first_config.v2_adaptive_enabled
            if v2_adaptive:
                if compiled_paraphrase_ensemble is not None:
                    selected_videos, adaptive_diag = fuse_video_maxima_v2_paraphrase_ensemble(
                        ensemble=compiled_paraphrase_ensemble,
                        maxima=maxima,
                        rrf_constant=self.config.rrf_constant,
                        nomination_depth=self.config.kis_video_first_config.video_nomination_depth,
                        config=self.config.kis_video_first_config,
                    )
                else:
                    selected_videos, adaptive_diag = fuse_video_maxima_v2(
                        variants=variants,
                        maxima=maxima,
                        primary_variant_ids=primary_var_ids,
                        supporting_variant_ids=supporting_var_ids,
                        temporal_variants=temporal_var_list,
                        rrf_constant=self.config.rrf_constant,
                        nomination_depth=self.config.kis_video_first_config.video_nomination_depth,
                        config=self.config.kis_video_first_config,
                    )
            else:
                selected_videos = fuse_video_maxima(
                    variants=variants,
                    maxima=maxima,
                    primary_variant_ids=primary_var_ids,
                    rrf_constant=self.config.rrf_constant,
                    nomination_depth=self.config.kis_video_first_config.video_nomination_depth,
                    selected_video_cap=self.config.kis_video_first_config.selected_video_cap,
                )
                adaptive_diag = None
            video_fusion_seconds = self.clock() - video_fusion_started

            # Step 3: Restricted frame search
            restricted_started = self.clock()
            compulsory_by_video: dict[str, list[int]] = {}
            for vid_evidence in selected_videos:
                if vid_evidence.temporal_chain and vid_evidence.temporal_chain.has_valid_chain:
                    compulsory_by_video[vid_evidence.video_id] = list(
                        vid_evidence.temporal_chain.selected_chain_frames
                    )

            video_first_config = self.config.kis_video_first_config
            if video_first_config.enable_vi_localization_variant:
                vi_embedding = self.shared_encoder.encode_texts((request.query_vi,))[0]
                vi_token_count = self.token_budget_guard.count_tokens(request.query_vi)
                vi_was_truncated = vi_token_count > (self.token_budget_guard.max_tokens + 2)
                vi_variant = QueryVariant(
                    variant_id=f"{request.query_id}::vi_local_query",
                    text=request.query_vi,
                    language=QueryLanguage.VIETNAMESE,
                    variant_type=QueryVariantType.VIETNAMESE_DIRECT,
                    weight=video_first_config.vi_localization_weight,
                )
                localization_variants = tuple(variants) + (vi_variant,)
                localization_vectors = np.vstack([variant_embeddings, vi_embedding[None, :]])
                vi_cap = (
                    20
                    if video_first_config.enable_paraphrase_ensemble
                    else video_first_config.restricted_frames_per_video_per_variant
                )
                if variant_quotas_mapping is not None:
                    variant_quotas_mapping[vi_variant.variant_id] = vi_cap
                vi_nominal_budget = vi_cap
                translation_metadata["vi_localizer_telemetry"] = {
                    "vi_localizer_enabled": True,
                    "vi_localization_weight": video_first_config.vi_localization_weight,
                    "vi_context_token_count": vi_token_count,
                    "vi_was_truncated": vi_was_truncated,
                }
            else:
                localization_variants = variants
                localization_vectors = variant_embeddings
                vi_nominal_budget = 0

            restricted = self.video_restricted_searcher.search_selected_videos(
                video_ids=tuple(item.video_id for item in selected_videos),
                query_ids=tuple(variant.variant_id for variant in localization_variants),
                query_vectors=localization_vectors,
                per_query_result_cap=(
                    variant_quotas_mapping
                    if variant_quotas_mapping is not None
                    else self.config.kis_video_first_config.restricted_frames_per_video_per_variant
                ),
                compulsory_frame_ids_by_video=compulsory_by_video if v2_adaptive else None,
                enable_temporal_diversity=(
                    self.config.kis_video_first_config
                    .enable_temporal_diverse_local_candidates
                ),
                temporal_diversity_gap_seconds=(
                    self.config.kis_video_first_config
                    .temporal_diversity_gap_seconds
                ),
                raw_top_k=10,
            )
            restricted_frame_search_seconds = self.clock() - restricted_started

            frame_fusion_started = self.clock()
            video_first_outcome = build_kis_video_first_outcome(
                query_id=request.query_id,
                variants=localization_variants,
                maxima=maxima,
                restricted=restricted,
                selected_videos=selected_videos,
                weighted_rrf=self.weighted_rrf,
                output_top_k=request.output_top_k,
                rrf_constant=self.config.rrf_constant,
                adaptive_diagnostic=adaptive_diag,
                config=self.config.kis_video_first_config,
            )
            restricted_frame_fusion_seconds = self.clock() - frame_fusion_started
            fused_result = video_first_outcome.result
            video_first_trace = video_first_outcome.to_trace()
            video_first_full_corpus_rows_scored = (
                video_first_outcome.full_corpus_rows_scored
            )
            video_first_full_corpus_store_scan_count = (
                video_first_outcome.full_corpus_store_scan_count
            )
            video_first_restricted_rows_scored = (
                video_first_outcome.restricted_rows_scored
            )
            video_first_restricted_store_scan_count = (
                video_first_outcome.restricted_store_scan_count
            )
            fusion_seconds = video_fusion_seconds + restricted_frame_fusion_seconds

            # Phase C telemetry computation
            candidate_tel = restricted.candidate_selection_telemetry or {}
            compulsory_extra_count = 0
            candidate_count_before_dedup = 0
            effective_quota_by_group_variant_video: dict[str, dict[str, int]] = {}

            for qid_key, vmap in candidate_tel.items():
                effective_quota_by_group_variant_video[qid_key] = {}
                for vid_key, tel_dict in vmap.items():
                    compulsory_extra_count += tel_dict.get("compulsory_extra_count", 0)
                    cand_cnt = tel_dict.get("effective_candidate_count", 0)
                    candidate_count_before_dedup += cand_cnt
                    effective_quota_by_group_variant_video[qid_key][vid_key] = cand_cnt

            all_restricted_frame_identities: set[tuple[str, int]] = set()
            for qid_key, per_vid_map in restricted.rankings.items():
                for vid_key, hits in per_vid_map.items():
                    for h in hits:
                        all_restricted_frame_identities.add((vid_key, h.frame_id))

            unique_candidates_after_dedup = len(all_restricted_frame_identities)
            dup_rate = (
                1.0 - (unique_candidates_after_dedup / candidate_count_before_dedup)
                if candidate_count_before_dedup > 0
                else 0.0
            )

            maxima_dot_evals = video_first_full_corpus_rows_scored * len(variants)
            restricted_dot_evals = video_first_restricted_rows_scored * len(localization_variants)
            total_logical_similarity_evaluations = maxima_dot_evals + restricted_dot_evals

            phase_c_telemetry = {
                "semantic_nominal_budget": active_sem_budget,
                "vi_nominal_budget": vi_nominal_budget,
                "total_nominal_budget": active_sem_budget + vi_nominal_budget,
                "requested_quota_by_variant": variant_quotas_mapping,
                "effective_quota_by_variant_video": effective_quota_by_group_variant_video,
                "compulsory_extra_count": compulsory_extra_count,
                "candidate_count_before_dedup": candidate_count_before_dedup,
                "effective_unique_candidate_count_after_dedup": unique_candidates_after_dedup,
                "duplication_rate": dup_rate,
                "selected_video_count": len(selected_videos),
                "compiled_group_count": len(compiled_paraphrase_ensemble.groups) if compiled_paraphrase_ensemble else 1,
                "compiled_variant_count": len(variants),
                "text_embedding_count": len(localization_variants),
                "logical_similarity_evaluations": {
                    "maxima_dot_evals": maxima_dot_evals,
                    "restricted_dot_evals": restricted_dot_evals,
                    "total_evals": total_logical_similarity_evaluations,
                },
                "normalized_weight_mass_by_group": (
                    {g.group_id: sum(compiled_paraphrase_ensemble.normalized_weights[v.query_variant.variant_id] for v in g.compiled_query.variants) for g in compiled_paraphrase_ensemble.groups}
                    if compiled_paraphrase_ensemble
                    else {"group_0": sum(v.weight for v in variants)}
                ),
                "total_normalized_weight_mass": (
                    compiled_paraphrase_ensemble.baseline_weight_mass
                    if compiled_paraphrase_ensemble
                    else sum(v.weight for v in variants)
                ),
                "timings": {
                    "translation_seconds": translation_seconds,
                    "text_encode_seconds": text_encode_seconds,
                    "full_corpus_video_search_seconds": full_corpus_video_search_seconds,
                    "video_fusion_seconds": video_fusion_seconds,
                    "restricted_frame_search_seconds": restricted_frame_search_seconds,
                    "restricted_frame_fusion_seconds": restricted_frame_fusion_seconds,
                    "total_retrieval_seconds": self.clock() - retrieval_start,
                },
            }
            video_first_trace["phase_c_telemetry"] = phase_c_telemetry
            translation_metadata["phase_c_telemetry"] = phase_c_telemetry
        else:
            rankings: dict[str, KISResult] = {}
            for variant, vector in zip(variants, variant_embeddings, strict=True):
                rankings[variant.variant_id] = self.exact_retriever.search_vector(
                    query_id=f"{request.query_id}::{variant.variant_id}",
                    query_vector=vector,
                    top_k=request.top_k_per_variant,
                )
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

        conditioned_result = fused_result
        q3_config = self.config.video_conditioned_keyframe_config
        q3_enabled = q3_config.enabled
        q3_trace: Mapping[str, Any] = {
            "policy": VIDEO_CONDITIONED_KEYFRAME_DIVERSITY,
            "enabled": False,
        }
        selected_video_count = 0
        restricted_keyframe_rows_scored = 0
        anchor_count = 0
        substitution_count = 0
        selected_videos_with_no_replacement_capacity = 0
        total_same_video_replacement_slots = 0
        restricted_search_seconds = 0.0
        conditioning_seconds = 0.0
        if q3_enabled:
            if not video_first_enabled and (
                len(variants) != 1
                or variants[0].variant_type is not QueryVariantType.ENGLISH_TRANSLATION
            ):
                raise ValueError(
                    "VIDEO_CONDITIONED_KEYFRAME_DIVERSITY requires exactly one "
                    "English translation variant"
                )
            q3_outcome = self.video_conditioner.condition(
                global_result=fused_result,
                query_vector=variant_embeddings[0],
                config=q3_config,
                protected_prefix_rank=request.refine_top_n,
                semantic_variants=(
                    variants if q3_config.semantic_variant_coverage else ()
                ),
                semantic_query_vectors=(
                    variant_embeddings
                    if q3_config.semantic_variant_coverage
                    else None
                ),
            )
            conditioned_result = q3_outcome.result
            q3_trace = q3_outcome.trace
            selected_video_count = q3_outcome.selected_video_count
            restricted_keyframe_rows_scored = q3_outcome.restricted_keyframe_rows_scored
            anchor_count = q3_outcome.anchor_count
            substitution_count = q3_outcome.substitution_count
            selected_videos_with_no_replacement_capacity = (
                q3_outcome.selected_videos_with_no_replacement_capacity
            )
            total_same_video_replacement_slots = (
                q3_outcome.total_same_video_replacement_slots
            )
            restricted_search_seconds = q3_outcome.restricted_search_seconds
            conditioning_seconds = q3_outcome.conditioning_seconds

        export_start = self.clock()
        top100_jsonl = query_dir / "top100.jsonl"
        self.exporter.export(conditioned_result, top100_jsonl)
        final_kis_result = conditioned_result
        final_prediction_artifact = top100_jsonl
        _write_internal_csv((conditioned_result,), query_dir / "top100.csv")

        global_top100_jsonl: Path | None = None
        q3_trace_json: Path | None = None
        video_first_trace_json: Path | None = None
        if video_first_enabled:
            video_first_trace_json = _write_json(
                query_dir / "kis_video_first_trace.json",
                video_first_trace,
            )
        if q3_enabled:
            global_top100_jsonl = query_dir / "global_top100.jsonl"
            self.exporter.export(fused_result, global_top100_jsonl)
            q3_trace_json = _write_json(
                query_dir / "video_conditioned_keyframe_trace.json",
                q3_trace,
            )

        top100_bytes = top100_jsonl.read_bytes()
        top100_sha256 = hashlib.sha256(top100_bytes).hexdigest()

        evidence_frame_pool: list[dict[str, object]] = []
        if video_first_outcome is not None and restricted is not None:
            for vid_evidence in video_first_outcome.selected_videos:
                vid = vid_evidence.video_id
                frames_for_video: set[int] = set()
                if vid_evidence.temporal_chain and vid_evidence.temporal_chain.selected_chain_frames:
                    for cf in vid_evidence.temporal_chain.selected_chain_frames:
                        frames_for_video.add(cf)
                for var in variants:
                    per_v = restricted.rankings.get(var.variant_id, {}).get(vid, ())
                    for rf in per_v[:6]:
                        frames_for_video.add(rf.frame_id)
                for fid in sorted(frames_for_video):
                    evidence_frame_pool.append({"video_id": vid, "frame_id": fid})

        def _get_cand_pts(c: CandidateFrame) -> float:
            diag_pts = float((c.diagnostic_metadata or {}).get("pts_time") or 0.0)
            if diag_pts > 0.0:
                return diag_pts
            try:
                st = self.registry.get_store(c.video_id)
                if st is not None:
                    for mp in st.mappings:
                        if mp.frame_id == c.frame_id:
                            return float(mp.pts_time)
            except Exception:
                pass
            return float(c.frame_id) / 25.0

        def _get_cand_order(c: CandidateFrame) -> int:
            if c.keyframe_order and c.keyframe_order > 0:
                return c.keyframe_order
            try:
                st = self.registry.get_store(c.video_id)
                if st is not None:
                    for mp in st.mappings:
                        if mp.frame_id == c.frame_id:
                            return int(mp.keyframe_order)
            except Exception:
                pass
            return 0

        candidates_json = query_dir / "candidates.json"
        is_experimental_localization = bool(
            getattr(self.config.kis_video_first_config, "restricted_frames_per_video_per_variant", 10) != 10
            or getattr(self.config.kis_video_first_config, "enable_temporal_diverse_local_candidates", False)
            or getattr(self.config.kis_video_first_config, "enable_vi_localization_variant", False)
            or getattr(self.config.kis_video_first_config, "internal_rrf_candidate_depth", 100) != 100
        )

        candidates_data = {
            "query_id": conditioned_result.query_id,
            "request_id": request.request_id,
            "top100_sha256": top100_sha256,
            "evidence_frame_pool": evidence_frame_pool,
            "translation": translation_metadata,
            "video_first": video_first_trace,
            **({"candidate_selection_telemetry": {
                qid: {vid: dict(tel) for vid, tel in per_vid.items()}
                for qid, per_vid in restricted.candidate_selection_telemetry.items()
            }} if (restricted is not None and getattr(restricted, "candidate_selection_telemetry", None)
                   and is_experimental_localization) else {}),
            "enabled_features": [
                name for name, val in [
                    ("candidate_union", getattr(self.config.kis_video_first_config, "enable_candidate_union", False)),
                    ("score_normalization", getattr(self.config.kis_video_first_config, "enable_score_normalization", False)),
                    ("late_interaction", getattr(self.config.kis_video_first_config, "enable_late_interaction", False)),
                    ("positive_chain_bonus", getattr(self.config.kis_video_first_config, "enable_positive_chain_bonus", False)),
                    ("temporal_diverse_local_candidates", getattr(self.config.kis_video_first_config, "enable_temporal_diverse_local_candidates", False)),
                    ("vi_localization_variant", getattr(self.config.kis_video_first_config, "enable_vi_localization_variant", False)),
                    ("internal_rrf_candidate_depth_500", getattr(self.config.kis_video_first_config, "internal_rrf_candidate_depth", 100) == 500),
                ] if val
            ],
            "records": [
                {
                    "query_id": conditioned_result.query_id,
                    "rank": candidate.rank,
                    "video_id": candidate.video_id,
                    "frame_id": candidate.frame_id,
                    "fusion_score": candidate.score,
                    "pts_time": _get_cand_pts(candidate),
                    "scores_by_variant": (candidate.diagnostic_metadata or {}).get("scores_by_variant", {}),
                    **({"selection_by_variant": (candidate.diagnostic_metadata or {})["selection_by_variant"]}
                       if is_experimental_localization
                          and "selection_by_variant" in (candidate.diagnostic_metadata or {})
                       else {}),
                    "is_temporal_chain_winner": (candidate.diagnostic_metadata or {}).get("is_temporal_chain_winner", False),
                    "video_nomination_rank": (candidate.diagnostic_metadata or {}).get("video_nomination_rank"),
                    "variant_hit_count": (candidate.diagnostic_metadata or {}).get(
                        "variant_hit_count"
                    ),
                    "best_individual_rank": (candidate.diagnostic_metadata or {}).get(
                        "best_individual_rank"
                    ),
                    "clip_row_diagnostic": candidate.clip_row,
                    "keyframe_order_diagnostic": _get_cand_order(candidate),
                    "keyframe_order": _get_cand_order(candidate),
                    "source": candidate.source,
                    "q3_policy": (candidate.diagnostic_metadata or {}).get("q3_policy"),
                }
                for candidate in conditioned_result.ranked_candidates
            ],
        }
        cand_bytes = (json.dumps(candidates_data, indent=2) + "\n").encode("utf-8")
        candidates_json.write_bytes(cand_bytes)
        candidates_sha256 = hashlib.sha256(cand_bytes).hexdigest()
        (query_dir / "candidates.json.sha256").write_text(candidates_sha256 + "\n", encoding="utf-8")
        (query_dir / "top100.jsonl.sha256").write_text(top100_sha256 + "\n", encoding="utf-8")
        retrieval_export_seconds = self.clock() - export_start

        retrieval_val_start = self.clock()
        validation = self.validator.validate(top100_jsonl, self.registry)
        retrieval_val_seconds = self.clock() - retrieval_val_start

        val_report_path = query_dir / "validation_report.json"
        val_report_path.write_text(
            json.dumps(
                _validation_payload(validation),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        artifacts_dict: dict[str, str] = {
            "top100_jsonl": str(top100_jsonl.relative_to(self.output_root)).replace("\\", "/"),
            "top100_csv": str(
                (query_dir / "top100.csv").relative_to(self.output_root)
            ).replace("\\", "/"),
            "candidates_json": str(
                candidates_json.relative_to(self.output_root)
            ).replace("\\", "/"),
            "validation_report": str(
                val_report_path.relative_to(self.output_root)
            ).replace("\\", "/"),
        }
        if global_top100_jsonl is not None and q3_trace_json is not None:
            artifacts_dict.update(
                {
                    "global_top100_jsonl": str(
                        global_top100_jsonl.relative_to(self.output_root)
                    ).replace("\\", "/"),
                    "video_conditioned_keyframe_trace_json": str(
                        q3_trace_json.relative_to(self.output_root)
                    ).replace("\\", "/"),
                }
            )
        if video_first_trace_json is not None:
            artifacts_dict["kis_video_first_trace_json"] = str(
                video_first_trace_json.relative_to(self.output_root)
            ).replace("\\", "/")

        refinement_requested = request.refine_top_n > 0
        refinement_valid: bool | None = None
        refined_count = 0
        refinement_seconds = 0.0
        refinement_export_seconds = 0.0
        refinement_val_seconds = 0.0
        decoded_frame_count = 0
        encoded_image_count = 0
        coarse_requested_frame_count = 0
        coarse_decoded_frame_count = 0
        fine_requested_frame_count = 0
        fine_decoded_frame_count = 0
        coarse_sparse_request_count = 0
        coarse_sparse_success_count = 0
        coarse_sparse_fallback_count = 0
        video_probe_seconds = 0.0
        video_open_seconds = 0.0
        coarse_decode_seconds = 0.0
        coarse_encode_seconds = 0.0
        coarse_score_seconds = 0.0
        coarse_fusion_seconds = 0.0
        fine_decode_seconds = 0.0
        fine_encode_seconds = 0.0
        fine_score_seconds = 0.0
        fine_fusion_seconds = 0.0
        candidate_total_seconds = 0.0
        q3_anchor_config = self.config.q3_anchor_refinement_config
        q3_anchor_enabled = q3_anchor_config.enabled
        eligible_q3_anchor_count = 0
        selected_q3_anchor_count = 0
        selected_q3_video_count = 0
        q3_anchor_refined_count = 0
        q3_anchor_kept_original_count = 0
        q3_anchor_collision_skip_count = 0
        q3_anchor_failure_count = 0
        q3_anchor_refinement_seconds = 0.0
        unique_q3_coarse_frame_count = 0
        unique_q3_fine_frame_count = 0
        frame_embedding_cache_hit_count = 0
        frame_embedding_cache_miss_count = 0
        merged_temporal_region_count = 0
        q3_anchor_trace_json: Path | None = None
        timeline_config = self.config.selected_video_timeline_scout_config
        timeline_enabled = timeline_config.enabled
        timeline_scout_seconds = 0.0
        timeline_video_count = 0
        timeline_sample_count = 0
        timeline_decoded_frame_count = 0
        timeline_encoded_image_count = 0
        timeline_region_count = 0
        timeline_visual_verified_candidate_count = 0
        timeline_visual_verifier_seconds = 0.0
        timeline_refined_count = 0
        timeline_kept_original_count = 0
        timeline_collision_skip_count = 0
        timeline_failure_count = 0
        timeline_trace_json: Path | None = None
        frame_embedding_cache: FrameEmbeddingCache = {}

        if q3_anchor_enabled and not refinement_requested:
            raise ValueError("Q3 anchor refinement requires refine_top_n > 0")
        if timeline_enabled and not refinement_requested:
            raise ValueError("selected-video timeline scout requires refine_top_n > 0")

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
                        "best_individual_rank": (c.diagnostic_metadata or {}).get(
                            "best_individual_rank"
                        ),
                        "clip_row_diagnostic": c.clip_row,
                        "keyframe_order_diagnostic": c.keyframe_order,
                        "q3_policy": (c.diagnostic_metadata or {}).get("q3_policy"),
                        "q3_original_frame_id": (c.diagnostic_metadata or {}).get(
                            "original_frame_id"
                        ),
                        "q3_restricted_cosine_score": (
                            c.diagnostic_metadata or {}
                        ).get("restricted_cosine_score"),
                        "q3_restricted_rank": (c.diagnostic_metadata or {}).get(
                            "restricted_rank"
                        ),
                        "q3_semantic_anchor_score": (
                            c.diagnostic_metadata or {}
                        ).get("semantic_anchor_score"),
                        "q3_semantic_variant_ids": (
                            c.diagnostic_metadata or {}
                        ).get("semantic_variant_ids"),
                        "candidate_source": c.source,
                    },
                )
                for c in conditioned_result.ranked_candidates
            )
            ref_query = RefinementQuery(request.query_id, variants, phase3_candidates)

            exec_ref_config = self.config.refinement_config
            if exec_ref_config.top_candidates_to_refine != request.refine_top_n:
                exec_ref_config = dataclasses.replace(
                    exec_ref_config,
                    top_candidates_to_refine=request.refine_top_n,
                )

            if q3_anchor_enabled:
                outcome = self.refiner.refine_query(
                    ref_query,
                    exec_ref_config,
                    precomputed_text_embeddings=variant_embeddings,
                    frame_embedding_cache=frame_embedding_cache,
                )
            else:
                outcome = self.refiner.refine_query(
                    ref_query,
                    exec_ref_config,
                    precomputed_text_embeddings=variant_embeddings,
                )
            refinement_seconds = self.clock() - ref_start

            final_kis_result = outcome.result
            if q3_anchor_enabled:
                raw_available_candidates = tuple(
                    candidate
                    for candidate in phase3_candidates
                    if (
                        raw_path := self.raw_video_registry.get(
                            candidate.video_id
                        ).raw_video_path
                    )
                    is not None
                    and raw_path.is_file()
                )
                selection = select_q3_anchor_candidates(
                    raw_available_candidates,
                    protected_prefix_rank=request.refine_top_n,
                    max_extra_q3_anchors=q3_anchor_config.max_extra_q3_anchors,
                )
                eligible_q3_anchor_count = len(selection.eligible)
                selected_q3_anchor_count = len(selection.selected)
                selected_q3_video_count = len(
                    {candidate.video_id for candidate in selection.selected}
                )
                selected_outcome = self.refiner.refine_selected_candidates(
                    query_id=request.query_id,
                    variants=variants,
                    candidates=selection.selected,
                    config=exec_ref_config,
                    precomputed_text_embeddings=variant_embeddings,
                    frame_embedding_cache=frame_embedding_cache,
                )
                integration = integrate_q3_anchor_refinements(
                    outcome.result,
                    selected_outcome.candidates,
                )
                final_kis_result = integration.result
                q3_anchor_refined_count = integration.refined_count
                q3_anchor_kept_original_count = integration.kept_original_count
                q3_anchor_collision_skip_count = integration.collision_skip_count
                q3_anchor_failure_count = integration.failure_count
                q3_anchor_refinement_seconds = float(
                    selected_outcome.timings["q3_anchor_refinement_seconds"]
                )
                unique_q3_coarse_frame_count = int(
                    selected_outcome.timings["unique_q3_coarse_frame_count"]
                )
                unique_q3_fine_frame_count = int(
                    selected_outcome.timings["unique_q3_fine_frame_count"]
                )
                frame_embedding_cache_hit_count = int(
                    selected_outcome.timings["frame_embedding_cache_hit_count"]
                )
                frame_embedding_cache_miss_count = int(
                    selected_outcome.timings["frame_embedding_cache_miss_count"]
                )
                merged_temporal_region_count = int(
                    selected_outcome.timings["merged_temporal_region_count"]
                )
                q3_anchor_trace_json = _write_json(
                    query_dir / "q3_anchor_refinement_trace.json",
                    {
                        "query_id": request.query_id,
                        "enabled": True,
                        "protected_prefix_rank": request.refine_top_n,
                        "max_extra_q3_anchors": q3_anchor_config.max_extra_q3_anchors,
                        "eligible_candidate_ranks": [
                            candidate.rank for candidate in selection.eligible
                        ],
                        "selected_candidate_ranks": [
                            candidate.rank for candidate in selection.selected
                        ],
                        "records": selected_outcome.candidates,
                        "warnings": selected_outcome.warnings,
                        "timings": selected_outcome.timings,
                        "integration": integration,
                    },
                )

            if timeline_enabled:
                timeline_outcome = self.refiner.scout_selected_video_timelines(
                    query_id=request.query_id,
                    query_vi=request.query_vi,
                    query_en=compiled_semantic_query.variants[0].raw_english,
                    variants=variants,
                    ranked_video_ids=tuple(item.video_id for item in selected_videos),
                    rank_slots=phase3_candidates,
                    config=timeline_config,
                    visual_verifier_config=(
                        self.config.selected_video_visual_verifier_config
                    ),
                    refinement_config=exec_ref_config,
                    precomputed_text_embeddings=variant_embeddings,
                    frame_embedding_cache=frame_embedding_cache,
                )
                timeline_selected_outcome = self.refiner.refine_selected_candidates(
                    query_id=request.query_id,
                    variants=variants,
                    candidates=timeline_outcome.candidates,
                    config=exec_ref_config,
                    precomputed_text_embeddings=variant_embeddings,
                    frame_embedding_cache=frame_embedding_cache,
                )
                timeline_integration = integrate_q3_anchor_refinements(
                    final_kis_result,
                    timeline_selected_outcome.candidates,
                )
                final_kis_result = timeline_integration.result
                timeline_scout_seconds = float(
                    timeline_outcome.timings["timeline_scout_seconds"]
                )
                timeline_video_count = int(
                    timeline_outcome.timings["timeline_video_count"]
                )
                timeline_sample_count = int(
                    timeline_outcome.timings["timeline_sample_count"]
                )
                timeline_decoded_frame_count = int(
                    timeline_outcome.timings["timeline_decoded_frame_count"]
                )
                timeline_encoded_image_count = int(
                    timeline_outcome.timings["timeline_encoded_image_count"]
                )
                timeline_region_count = int(
                    timeline_outcome.timings["timeline_region_count"]
                )
                timeline_visual_verified_candidate_count = int(
                    timeline_outcome.timings[
                        "timeline_visual_verified_candidate_count"
                    ]
                )
                timeline_visual_verifier_seconds = float(
                    timeline_outcome.timings["timeline_visual_verifier_seconds"]
                )
                timeline_refined_count = timeline_integration.refined_count
                timeline_kept_original_count = timeline_integration.kept_original_count
                timeline_collision_skip_count = timeline_integration.collision_skip_count
                timeline_failure_count = timeline_integration.failure_count
                timeline_trace_json = _write_json(
                    query_dir / "selected_video_timeline_scout_trace.json",
                    {
                        "query_id": request.query_id,
                        "config": dataclasses.asdict(timeline_config),
                        "scout": timeline_outcome.trace,
                        "scout_timings": timeline_outcome.timings,
                        "scout_warnings": timeline_outcome.warnings,
                        "refinement_records": timeline_selected_outcome.candidates,
                        "refinement_warnings": timeline_selected_outcome.warnings,
                        "refinement_timings": timeline_selected_outcome.timings,
                        "integration": timeline_integration,
                    },
                )

            ref_export_start = self.clock()
            refined_jsonl = query_dir / "refined_top100.jsonl"
            self.exporter.export(final_kis_result, refined_jsonl)
            final_prediction_artifact = refined_jsonl
            refined_csv = _write_refined_csv(
                (final_kis_result,), query_dir / "refined_top100.csv"
            )
            ref_cand_json = _write_json(
                query_dir / "refinement_candidates.json",
                outcome.candidates,
            )
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
            refined_count = int(outcome.timings["refined_candidate_count"])
            decoded_frame_count = int(outcome.timings["decoded_frame_count"])
            encoded_image_count = int(outcome.timings["encoded_image_count"])
            coarse_requested_frame_count = int(outcome.timings["coarse_requested_frame_count"])
            coarse_decoded_frame_count = int(outcome.timings["coarse_decoded_frame_count"])
            fine_requested_frame_count = int(outcome.timings["fine_requested_frame_count"])
            fine_decoded_frame_count = int(outcome.timings["fine_decoded_frame_count"])
            coarse_sparse_request_count = int(outcome.timings["coarse_sparse_request_count"])
            coarse_sparse_success_count = int(outcome.timings["coarse_sparse_success_count"])
            coarse_sparse_fallback_count = int(outcome.timings["coarse_sparse_fallback_count"])
            video_probe_seconds = float(outcome.timings["video_probe_seconds"])
            video_open_seconds = float(outcome.timings["video_open_seconds"])
            coarse_decode_seconds = float(outcome.timings["coarse_decode_seconds"])
            coarse_encode_seconds = float(outcome.timings["coarse_encode_seconds"])
            coarse_score_seconds = float(outcome.timings["coarse_score_seconds"])
            coarse_fusion_seconds = float(outcome.timings["coarse_fusion_seconds"])
            fine_decode_seconds = float(outcome.timings["fine_decode_seconds"])
            fine_encode_seconds = float(outcome.timings["fine_encode_seconds"])
            fine_score_seconds = float(outcome.timings["fine_score_seconds"])
            fine_fusion_seconds = float(outcome.timings["fine_fusion_seconds"])
            candidate_total_seconds = float(outcome.timings["candidate_total_seconds"])

            artifacts_dict.update(
                {
                    "refined_top100_jsonl": str(
                        refined_jsonl.relative_to(self.output_root)
                    ).replace("\\", "/"),
                    "refined_top100_csv": str(
                        refined_csv.relative_to(self.output_root)
                    ).replace("\\", "/"),
                    "refinement_candidates_json": str(
                        ref_cand_json.relative_to(self.output_root)
                    ).replace("\\", "/"),
                    "refinement_trace_json": str(
                        ref_trace_json.relative_to(self.output_root)
                    ).replace("\\", "/"),
                    "refinement_validation_report": str(
                        ref_val_report.relative_to(self.output_root)
                    ).replace("\\", "/"),
                }
            )
            if q3_anchor_trace_json is not None:
                artifacts_dict["q3_anchor_refinement_trace_json"] = str(
                    q3_anchor_trace_json.relative_to(self.output_root)
                ).replace("\\", "/")
            if timeline_trace_json is not None:
                artifacts_dict["selected_video_timeline_scout_trace_json"] = str(
                    timeline_trace_json.relative_to(self.output_root)
                ).replace("\\", "/")

        canonical_kis_query = kis_result_to_top100_query(final_kis_result)
        audit_runtime_top100_artifact(
            canonical_kis_query,
            final_prediction_artifact,
        )

        total_seconds = self.clock() - req_start

        req_manifest_payload = {
            "request_id": request.request_id,
            "query_id": request.query_id,
            "query_vi": request.query_vi,
            "query_en": request.query_en,
            "query_en_expansion": request.query_en_expansion,
            "include_vi_variant": request.include_vi_variant,
            "weights": {
                "weight_vi": request.weight_vi,
                "weight_en": request.weight_en,
                "weight_en_expansion": request.weight_en_expansion,
            },
            "top_k_per_variant": request.top_k_per_variant,
            "output_top_k": request.output_top_k,
            "refine_top_n": request.refine_top_n,
            "kis_video_first_enabled": video_first_enabled,
            "kis_video_first_config": dataclasses.asdict(
                self.config.kis_video_first_config
            ),
            "q3_enabled": q3_enabled,
            "q3_temporal_policy": (
                VIDEO_CONDITIONED_KEYFRAME_DIVERSITY if q3_enabled else None
            ),
            "q3_config": dataclasses.asdict(q3_config),
            "q3_anchor_refinement_config": dataclasses.asdict(q3_anchor_config),
            "selected_video_timeline_scout_config": dataclasses.asdict(
                timeline_config
            ),
            "selected_video_visual_verifier_config": dataclasses.asdict(
                self.config.selected_video_visual_verifier_config
            ),
            "retrieval_valid": validation.valid,
            "refinement_requested": refinement_requested,
            "refinement_valid": refinement_valid,
            "artifacts": artifacts_dict,
        }
        _write_json(query_dir / "request_manifest.json", req_manifest_payload)

        timings_payload = {
            "validation_seconds": validation_seconds,
            "translation_seconds": translation_seconds,
            "text_encode_seconds": text_encode_seconds,
            "retrieval_seconds": retrieval_seconds,
            "fusion_seconds": fusion_seconds,
            "kis_video_first_enabled": video_first_enabled,
            "full_corpus_video_search_seconds": full_corpus_video_search_seconds,
            "video_fusion_seconds": video_fusion_seconds,
            "restricted_frame_search_seconds": restricted_frame_search_seconds,
            "restricted_frame_fusion_seconds": restricted_frame_fusion_seconds,
            "video_first_full_corpus_rows_scored": (
                video_first_full_corpus_rows_scored
            ),
            "video_first_full_corpus_store_scan_count": (
                video_first_full_corpus_store_scan_count
            ),
            "video_first_restricted_rows_scored": (
                video_first_restricted_rows_scored
            ),
            "video_first_restricted_store_scan_count": (
                video_first_restricted_store_scan_count
            ),
            "retrieval_export_seconds": retrieval_export_seconds,
            "retrieval_validation_seconds": retrieval_val_seconds,
            "q3_enabled": q3_enabled,
            "selected_video_count": selected_video_count,
            "restricted_keyframe_rows_scored": restricted_keyframe_rows_scored,
            "anchor_count": anchor_count,
            "substitution_count": substitution_count,
            "selected_videos_with_no_replacement_capacity": (
                selected_videos_with_no_replacement_capacity
            ),
            "total_same_video_replacement_slots": total_same_video_replacement_slots,
            "restricted_search_seconds": restricted_search_seconds,
            "conditioning_seconds": conditioning_seconds,
            "q3_anchor_refinement_enabled": q3_anchor_enabled,
            "eligible_q3_anchor_count": eligible_q3_anchor_count,
            "selected_q3_anchor_count": selected_q3_anchor_count,
            "selected_q3_video_count": selected_q3_video_count,
            "q3_anchor_refined_count": q3_anchor_refined_count,
            "q3_anchor_kept_original_count": q3_anchor_kept_original_count,
            "q3_anchor_collision_skip_count": q3_anchor_collision_skip_count,
            "q3_anchor_failure_count": q3_anchor_failure_count,
            "q3_anchor_refinement_seconds": q3_anchor_refinement_seconds,
            "unique_q3_coarse_frame_count": unique_q3_coarse_frame_count,
            "unique_q3_fine_frame_count": unique_q3_fine_frame_count,
            "frame_embedding_cache_hit_count": frame_embedding_cache_hit_count,
            "frame_embedding_cache_miss_count": frame_embedding_cache_miss_count,
            "merged_temporal_region_count": merged_temporal_region_count,
            "selected_video_timeline_scout_enabled": timeline_enabled,
            "timeline_scout_seconds": timeline_scout_seconds,
            "timeline_video_count": timeline_video_count,
            "timeline_sample_count": timeline_sample_count,
            "timeline_decoded_frame_count": timeline_decoded_frame_count,
            "timeline_encoded_image_count": timeline_encoded_image_count,
            "timeline_region_count": timeline_region_count,
            "timeline_visual_verifier_enabled": (
                self.config.selected_video_visual_verifier_config.enabled
            ),
            "timeline_visual_verified_candidate_count": (
                timeline_visual_verified_candidate_count
            ),
            "timeline_visual_verifier_seconds": timeline_visual_verifier_seconds,
            "timeline_refined_count": timeline_refined_count,
            "timeline_kept_original_count": timeline_kept_original_count,
            "timeline_collision_skip_count": timeline_collision_skip_count,
            "timeline_failure_count": timeline_failure_count,
            "refinement_seconds": refinement_seconds,
            "refinement_export_seconds": refinement_export_seconds,
            "refinement_validation_seconds": refinement_val_seconds,
            "video_probe_seconds": video_probe_seconds,
            "video_open_seconds": video_open_seconds,
            "coarse_decode_seconds": coarse_decode_seconds,
            "coarse_encode_seconds": coarse_encode_seconds,
            "coarse_score_seconds": coarse_score_seconds,
            "coarse_fusion_seconds": coarse_fusion_seconds,
            "fine_decode_seconds": fine_decode_seconds,
            "fine_encode_seconds": fine_encode_seconds,
            "fine_score_seconds": fine_score_seconds,
            "fine_fusion_seconds": fine_fusion_seconds,
            "candidate_total_seconds": candidate_total_seconds,
            "total_seconds": total_seconds,
            "decoded_frame_count": decoded_frame_count,
            "encoded_image_count": encoded_image_count,
            "coarse_requested_frame_count": coarse_requested_frame_count,
            "coarse_decoded_frame_count": coarse_decoded_frame_count,
            "fine_requested_frame_count": fine_requested_frame_count,
            "fine_decoded_frame_count": fine_decoded_frame_count,
            "coarse_sparse_request_count": coarse_sparse_request_count,
            "coarse_sparse_success_count": coarse_sparse_success_count,
            "coarse_sparse_fallback_count": coarse_sparse_fallback_count,
        }
        _write_json(query_dir / "request_timings.json", timings_payload)

        summary_md = [
            f"# Request Summary: {request.request_id} ({request.query_id})",
            "",
            f"- Retrieval valid: `{validation.valid}`",
            f"- Refinement requested: `{refinement_requested}`",
            f"- Refinement valid: `{refinement_valid}`",
            f"- Q3 enabled: `{q3_enabled}`",
            f"- KIS semantic video-first enabled: `{video_first_enabled}`",
            f"- Q3 anchor refinement enabled: `{q3_anchor_enabled}`",
            f"- Selected-video timeline scout enabled: `{timeline_enabled}`",
            f"- Result count: {len(conditioned_result.ranked_candidates)}",
            f"- Total seconds: {total_seconds:.6f}s",
            "",
        ]
        (query_dir / "request_summary.md").write_text(
            "\n".join(summary_md) + "\n",
            encoding="utf-8",
        )

        self._successful_query_count += 1

        return {
            "type": "query_result",
            "request_id": request.request_id,
            "query_id": request.query_id,
            "status": "SUCCESS",
            "retrieval_valid": validation.valid,
            "refinement_requested": refinement_requested,
            "refinement_valid": refinement_valid,
            "result_count": len(conditioned_result.ranked_candidates),
            "refined_count": refined_count,
            "artifacts": artifacts_dict,
            "timings": timings_payload,
        }

    def handle_qa_query(self, request: QAQueryRequest) -> dict[str, Any]:
        if request.request_id in self._seen_request_ids:
            msg = f"request_id '{request.request_id}' has already been processed"
            raise DuplicateRequestIdError(msg)
        self._seen_request_ids.add(request.request_id)
        self._request_count += 1

        req_dir_name = safe_request_directory_name(request.request_id)
        query_dir = self.output_root / "requests" / req_dir_name
        if query_dir.exists():
            query_dir = (
                self.output_root / "requests" / f"{req_dir_name}-{uuid.uuid4().hex[:4]}"
            )
        query_dir.mkdir(parents=True, exist_ok=True)

        qa_result, timings, diagnostics = self.qa_pipeline.process_qa_query(
            request,
            refinement_config=self.config.refinement_config,
            rrf_constant=self.config.rrf_constant,
        )

        # Artifact 1: qa_predictions.jsonl
        predictions_jsonl = query_dir / "qa_predictions.jsonl"
        with predictions_jsonl.open("w", encoding="utf-8", newline="") as stream:
            for p in qa_result.predictions:
                rec = {
                    "query_id": p.query_id,
                    "rank": p.rank,
                    "video_id": p.video_id,
                    "frame_id": p.frame_id,
                    "answer": p.answer,
                }
                stream.write(json.dumps(rec, ensure_ascii=False) + "\n")

        canonical_qa_query = qa_predictions_to_top100_query(
            query_id=request.query_id,
            predictions=qa_result.predictions,
        )
        audit_runtime_top100_artifact(canonical_qa_query, predictions_jsonl)

        # Artifact 2: qa_evidence.json
        evidence_json = _write_json(query_dir / "qa_evidence.json", diagnostics)

        # Artifact 3: qa_request_manifest.json
        # Artifact 4: qa_timings.json
        _write_json(query_dir / "qa_timings.json", timings.to_dict())

        rel_pred = str(predictions_jsonl.relative_to(self.output_root)).replace("\\", "/")
        rel_ev = str(evidence_json.relative_to(self.output_root)).replace("\\", "/")
        rel_man = str(
            (query_dir / "qa_request_manifest.json").relative_to(self.output_root)
        ).replace("\\", "/")
        rel_tim = str(
            (query_dir / "qa_timings.json").relative_to(self.output_root)
        ).replace("\\", "/")

        artifacts_dict = {
            "qa_predictions_jsonl": rel_pred,
            "qa_evidence_json": rel_ev,
            "qa_request_manifest": rel_man,
            "qa_timings": rel_tim,
        }

        # Artifact 3: qa_request_manifest.json
        req_manifest_payload = {
            "request_id": request.request_id,
            "query_id": request.query_id,
            "event_description": request.event_description,
            "question": request.question,
            "event_description_en": request.event_description_en,
            "question_en": request.question_en,
            "question_type": qa_result.question_type.value,
            "event_variants": [
                {
                    "variant_id": v.variant_id,
                    "text": v.text,
                    "language": (
                        v.language.value if hasattr(v.language, "value") else str(v.language)
                    ),
                    "variant_type": (
                        v.variant_type.value
                        if hasattr(v.variant_type, "value")
                        else str(v.variant_type)
                    ),
                    "weight": v.weight,
                }
                for v in request.variants()
            ],
            "top_k_per_variant": request.top_k_per_variant,
            "output_top_k": request.output_top_k,
            "refine_top_n": request.refine_top_n,
            "prediction_count": len(qa_result.predictions),
            "warnings": qa_result.warnings,
            "artifacts": artifacts_dict,
        }
        if self.config.qa_video_conditioned_evidence_config.enabled:
            req_manifest_payload.update(
                {
                    "qa_grounding_policy": diagnostics["qa_grounding_policy"],
                    "qa_video_conditioned_evidence_config": {
                        "selected_video_cap": (
                            self.config.qa_video_conditioned_evidence_config.selected_video_cap
                        ),
                        "anchors_per_video": (
                            self.config.qa_video_conditioned_evidence_config.anchors_per_video
                        ),
                        "video_rrf_constant": (
                            self.config.qa_video_conditioned_evidence_config.video_rrf_constant
                        ),
                        "candidate_ordering_policy": (
                            self.config.qa_video_conditioned_evidence_config
                            .candidate_ordering_policy
                        ),
                    },
                }
            )
        if self.config.qa_object_answer_provider_config.enabled:
            req_manifest_payload.update(
                {
                    "qa_object_provider_enabled": True,
                    "object_artifact_schema": (
                        self.object_artifact_index.schema_identity
                        if self.object_artifact_index is not None
                        else None
                    ),
                    "object_artifact_root_identity": (
                        self.object_artifact_index.source_root_identity
                        if self.object_artifact_index is not None
                        else None
                    ),
                    "object_frame_identity_contract": (
                        "JSON filename keyframe ordinal -> mapping n -> frame_idx"
                    ),
                    "official_ground_truth": False,
                    "diagnostic_development_only": True,
                }
            )
        if self.config.qa_ocr_answer_provider_config.enabled:
            req_manifest_payload.update(
                {
                    "qa_ocr_provider_enabled": True,
                    "ocr_backend_identity": (
                        dict(self.ocr_answer_provider.identifiers)
                        if self.ocr_answer_provider is not None
                        else None
                    ),
                    "ocr_evidence_frame_budget": (
                        self.config.qa_ocr_answer_provider_config.evidence_frame_budget
                    ),
                    "ocr_frame_identity_contract": (
                        "decoded QA-A1 evidence frame -> original-video absolute frame_id"
                    ),
                    "official_ground_truth": False,
                    "diagnostic_development_only": True,
                }
            )
        if self.config.qa_visual_ontology_config.enabled:
            assert self.visual_ontology_provider is not None
            req_manifest_payload.update(
                {
                    "qa_visual_ontology_enabled": True,
                    "qa_visual_ontology_identity": dict(
                        self.visual_ontology_provider.identifiers
                    ),
                    "qa_visual_frame_identity_contract": (
                        "decoded QA-A1 evidence frame -> original-video absolute frame_id"
                    ),
                    "official_ground_truth": False,
                    "diagnostic_development_only": True,
                }
            )
        _write_json(query_dir / "qa_request_manifest.json", req_manifest_payload)

        self._successful_query_count += 1

        response = {
            "type": "qa_result",
            "request_id": request.request_id,
            "query_id": request.query_id,
            "status": "SUCCESS",
            "question_type": qa_result.question_type.value,
            "prediction_count": len(qa_result.predictions),
            "predictions": [
                {
                    "query_id": p.query_id,
                    "rank": p.rank,
                    "video_id": p.video_id,
                    "frame_id": p.frame_id,
                    "answer": p.answer,
                }
                for p in qa_result.predictions
            ],
            "warnings": qa_result.warnings,
            "timings": timings.to_dict(),
            "artifacts": artifacts_dict,
        }
        if self.config.qa_video_conditioned_evidence_config.enabled:
            response["qa_grounding_policy"] = diagnostics["qa_grounding_policy"]
            response["grounding_candidate_count"] = diagnostics[
                "grounding_candidate_count"
            ]
            response["question_supported_by_current_provider"] = diagnostics[
                "question_supported_by_current_provider"
            ]
            response["unsupported_reason"] = diagnostics.get("unsupported_reason")
        return response

    def handle_trake_query(self, request: TRAKEQueryRequest) -> dict[str, Any]:
        if request.request_id in self._seen_request_ids:
            msg = f"request_id '{request.request_id}' has already been processed"
            raise DuplicateRequestIdError(msg)
        self._seen_request_ids.add(request.request_id)
        self._request_count += 1

        req_dir_name = safe_request_directory_name(request.request_id)
        query_dir = self.output_root / "requests" / req_dir_name
        if query_dir.exists():
            query_dir = (
                self.output_root / "requests" / f"{req_dir_name}-{uuid.uuid4().hex[:4]}"
            )
        query_dir.mkdir(parents=True, exist_ok=True)

        trake_kwargs: dict[str, Any] = {
            "refinement_config": self.config.refinement_config,
            "rrf_constant": self.config.rrf_constant,
        }
        if self.config.trake_video_first_config.enabled:
            trake_kwargs["video_first_config"] = self.config.trake_video_first_config
        if self.config.trake_shared_raw_region_config.enabled:
            trake_kwargs["shared_raw_region_config"] = (
                self.config.trake_shared_raw_region_config
            )
        trake_result, timings, extra_diag = self.trake_pipeline.process_trake_query(
            request,
            **trake_kwargs,
        )

        # Artifact 1: trake_predictions.jsonl
        predictions_jsonl = query_dir / "trake_predictions.jsonl"
        with predictions_jsonl.open("w", encoding="utf-8", newline="") as stream:
            for p in trake_result.predictions:
                rec = {
                    "query_id": p.query_id,
                    "rank": p.rank,
                    "video_id": p.video_id,
                    "frame_ids": list(p.frame_ids),
                }
                stream.write(json.dumps(rec, ensure_ascii=False) + "\n")

        expected_event_count = len(request.events)
        canonical_trake_query = trake_predictions_to_top100_query(
            query_id=request.query_id,
            predictions=trake_result.predictions,
            expected_event_count=expected_event_count,
        )
        audit_runtime_top100_artifact(
            canonical_trake_query,
            predictions_jsonl,
            expected_trake_event_count=expected_event_count,
        )

        # Artifact 2: trake_event_candidates.json
        candidates_payload = {
            "query_id": request.query_id,
            "event_candidates": [
                    {
                        "event_index": idx,
                        "candidate_count": len(pool),
                        "candidates": [
                            {
                                "rank": c.rank,
                                "video_id": c.video_id,
                                "frame_id": c.frame_id,
                                "retrieval_score": c.retrieval_score,
                                "provenance": c.provenance,
                            }
                            for c in pool
                        ],
                    }
                    for idx, pool in enumerate(extra_diag["event_candidate_pools"])
            ],
        }
        if "tr_a1" in extra_diag:
            candidates_payload["tr_a1"] = extra_diag["tr_a1"]
        candidates_json = _write_json(
            query_dir / "trake_event_candidates.json",
            candidates_payload,
        )

        c1_paths = extra_diag.get("c1_paths")
        if type(c1_paths) is not list:
            c1_paths = [
                {
                    "rank": record.get("c1_rank"),
                    "video_id": record.get("video_id"),
                    "frame_ids": record.get("original_frame_ids"),
                }
                for record in extra_diag.get("path_diagnostics", [])
                if type(record) is dict
            ]

        # Artifact 3: trake_refinement.json
        refinement_payload = {
            "query_id": request.query_id,
            "c1_diagnostics": extra_diag["c1_diagnostics"],
            "c1_paths": c1_paths,
            "refinement_requested": trake_result.diagnostics["refinement_requested"],
            "refine_top_n": request.refine_top_n,
            "refinement_node_records": extra_diag["refinement_node_records"],
            "path_diagnostics": extra_diag["path_diagnostics"],
            "warnings": list(trake_result.diagnostics.get("warnings", [])),
        }
        shared_raw_telemetry = extra_diag.get("shared_raw_region_refinement")
        if type(shared_raw_telemetry) is dict:
            refinement_payload["shared_raw_region_refinement"] = shared_raw_telemetry
        refinement_json = _write_json(
            query_dir / "trake_refinement.json",
            refinement_payload,
        )

        # Artifact 5: trake_timings.json
        timings_json = _write_json(query_dir / "trake_timings.json", timings.to_dict())

        rel_pred = str(predictions_jsonl.relative_to(self.output_root)).replace("\\", "/")
        rel_cand = str(candidates_json.relative_to(self.output_root)).replace("\\", "/")
        rel_ref = str(refinement_json.relative_to(self.output_root)).replace("\\", "/")
        rel_man = str(
            (query_dir / "trake_request_manifest.json").relative_to(self.output_root)
        ).replace("\\", "/")
        rel_tim = str(timings_json.relative_to(self.output_root)).replace("\\", "/")

        artifacts_dict = {
            "trake_predictions_jsonl": rel_pred,
            "trake_event_candidates_json": rel_cand,
            "trake_refinement_json": rel_ref,
            "trake_request_manifest": rel_man,
            "trake_timings": rel_tim,
        }

        # Artifact 4: trake_request_manifest.json
        req_manifest_payload = {
            "request_id": request.request_id,
            "query_id": request.query_id,
            "events": list(request.events),
            "include_vi_variant": request.include_vi_variant,
            "source_vi_retained_for_provenance": True,
            "event_variants": extra_diag["flattened_variants"],
            "top_k_per_variant": request.top_k_per_variant,
            "event_candidate_top_k": request.event_candidate_top_k,
            "output_top_k": request.output_top_k,
            "beam_width": request.beam_width,
            "refine_top_n": request.refine_top_n,
            "prediction_count": len(trake_result.predictions),
            "artifacts": artifacts_dict,
        }
        if "tr_a1" in extra_diag:
            req_manifest_payload["tr_a1"] = extra_diag["tr_a1"]
        if self.config.trake_shared_raw_region_config.enabled:
            req_manifest_payload["trake_shared_raw_region_refinement_config"] = (
                dataclasses.asdict(self.config.trake_shared_raw_region_config)
            )
        _write_json(query_dir / "trake_request_manifest.json", req_manifest_payload)

        self._successful_query_count += 1

        return {
            "type": "trake_result",
            "request_id": request.request_id,
            "query_id": request.query_id,
            "status": "SUCCESS",
            "event_count": len(request.events),
            "prediction_count": len(trake_result.predictions),
            "predictions": [
                {
                    "query_id": p.query_id,
                    "rank": p.rank,
                    "video_id": p.video_id,
                    "frame_ids": list(p.frame_ids),
                }
                for p in trake_result.predictions
            ],
            "warnings": list(trake_result.diagnostics.get("warnings", [])),
            "timings": timings.to_dict(),
            "artifacts": artifacts_dict,
        }

    def handle_shutdown(
        self,
        request: ShutdownRequest,
        shutdown_reason: str = "requested",
    ) -> dict[str, Any]:
        if request.request_id in self._seen_request_ids:
            raise DuplicateRequestIdError(
                f"request_id '{request.request_id}' has already been processed"
            )
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
            "q3_temporal_policy": (
                VIDEO_CONDITIONED_KEYFRAME_DIVERSITY
                if self.config.video_conditioned_keyframe_config.enabled
                else None
            ),
            "q3_config": dataclasses.asdict(
                self.config.video_conditioned_keyframe_config
            ),
            "q3_anchor_refinement_config": dataclasses.asdict(
                self.config.q3_anchor_refinement_config
            ),
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
        if self.config.trake_video_first_config.enabled:
            manifest_payload["trake_video_first_config"] = dataclasses.asdict(
                self.config.trake_video_first_config
            )
        if self.config.trake_shared_raw_region_config.enabled:
            manifest_payload["trake_shared_raw_region_config"] = dataclasses.asdict(
                self.config.trake_shared_raw_region_config
            )
        if self.config.qa_object_answer_provider_config.enabled:
            manifest_payload["qa_object_answer_provider_config"] = dataclasses.asdict(
                self.config.qa_object_answer_provider_config
            )
            manifest_payload["object_artifact_schema"] = (
                self.object_artifact_index.schema_identity
                if self.object_artifact_index is not None
                else None
            )
            manifest_payload["object_artifact_root_identity"] = (
                self.object_artifact_index.source_root_identity
                if self.object_artifact_index is not None
                else None
            )
        if self.config.qa_ocr_answer_provider_config.enabled:
            manifest_payload["qa_ocr_answer_provider_config"] = dataclasses.asdict(
                self.config.qa_ocr_answer_provider_config
            )
            manifest_payload["ocr_backend_identity"] = (
                dict(self.ocr_answer_provider.identifiers)
                if self.ocr_answer_provider is not None
                else None
            )
        if self.config.qa_visual_ontology_config.enabled:
            visual_config_payload = dataclasses.asdict(
                self.config.qa_visual_ontology_config
            )
            visual_config_payload["ontology_path"] = str(
                self.config.qa_visual_ontology_config.ontology_path
            )
            manifest_payload["qa_visual_ontology_config"] = visual_config_payload
            assert self.visual_ontology_provider is not None
            manifest_payload["qa_visual_ontology_identity"] = dict(
                self.visual_ontology_provider.identifiers
            )
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
        (self.output_root / "session_summary.md").write_text(
            "\n".join(summary_lines) + "\n",
            encoding="utf-8",
        )
