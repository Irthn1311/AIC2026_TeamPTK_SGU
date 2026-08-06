"""End-to-end contest Textual KIS execution and reproducibility artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from system_tai.checkpointing.exporter import CheckpointExporter
from system_tai.common.schemas import KISResult, ValidationResult
from system_tai.data.corpus_discovery import CorpusManifest
from system_tai.features.btc_clip_store import FeatureStoreRegistry
from system_tai.features.query_encoder import OpenAIClipTextEncoder, TextEncoder
from system_tai.inspection.candidate_report import build_candidate_inspection
from system_tai.kis.contest_schema import ContestQuery
from system_tai.retrieval.multi_query import WeightedRRFRetriever
from system_tai.retrieval.vector_search import ExactNumpyRetriever
from system_tai.validation.checkpoint_validator import CheckpointValidator


@dataclass(frozen=True, slots=True)
class ContestRunConfig:
    device: str = "cpu"
    top_k_per_variant: int = 100
    output_top_k_override: int | None = None
    rrf_constant: float = 60.0
    chunk_size: int = 4096
    inspection_top_n: int = 50
    continue_on_query_error: bool = False
    create_contact_sheet: bool = False
    allow_model_download: bool = False
    clip_cache_dir: Path | None = None

    def __post_init__(self) -> None:
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("resolved device must be cpu or cuda")
        if self.top_k_per_variant <= 0:
            raise ValueError("top_k_per_variant must be positive")
        if self.output_top_k_override is not None and not 1 <= self.output_top_k_override <= 100:
            raise ValueError("output_top_k_override must be between 1 and 100")
        if self.rrf_constant <= 0:
            raise ValueError("rrf_constant must be positive")
        if self.chunk_size <= 0 or self.inspection_top_n <= 0:
            raise ValueError("chunk_size and inspection_top_n must be positive")


@dataclass(frozen=True, slots=True)
class ContestRunOutcome:
    exit_code: int
    successful_query_ids: tuple[str, ...]
    failed_queries: tuple[tuple[str, str], ...]
    validation: ValidationResult
    output_files: tuple[Path, ...]


def safe_query_directory_name(query_id: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", query_id).strip("._-") or "query"
    digest = hashlib.sha256(query_id.encode("utf-8")).hexdigest()[:8]
    return f"{normalized}-{digest}"


def _git_commit_hash() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = completed.stdout.strip()
    return value or None


def _validation_payload(validation: ValidationResult) -> dict[str, Any]:
    return {
        "valid": validation.valid,
        "errors": [asdict(issue) for issue in validation.errors],
        "warnings": [asdict(issue) for issue in validation.warnings],
    }


def _write_internal_csv(results: Sequence[KISResult], path: Path) -> None:
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


class ContestRunner:
    def __init__(
        self,
        *,
        registry_loader: Callable[[Path], FeatureStoreRegistry] | None = None,
        encoder_factory: Callable[..., TextEncoder] | None = None,
        exporter: CheckpointExporter | None = None,
        validator: CheckpointValidator | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.registry_loader = registry_loader or FeatureStoreRegistry.from_manifest
        self.encoder_factory = encoder_factory or OpenAIClipTextEncoder
        self.exporter = exporter or CheckpointExporter()
        self.validator = validator or CheckpointValidator()
        self.clock = clock

    def run(
        self,
        *,
        manifest_path: Path,
        manifest: CorpusManifest,
        queries: tuple[ContestQuery, ...],
        output_directory: Path,
        config: ContestRunConfig,
        bootstrap_timings: Mapping[str, float] | None = None,
    ) -> ContestRunOutcome:
        if not queries:
            raise ValueError("contest run requires at least one query")
        output = Path(output_directory)
        output.mkdir(parents=True, exist_ok=True)
        batch_start = self.clock()
        registry_start = self.clock()
        registry = self.registry_loader(Path(manifest_path))
        registry_time = self.clock() - registry_start
        model_start = self.clock()
        encoder = self.encoder_factory(
            device=config.device,
            allow_model_download=config.allow_model_download,
            cache_dir=config.clip_cache_dir,
        )
        model_time = self.clock() - model_start
        exact = ExactNumpyRetriever(registry, encoder, chunk_size=config.chunk_size)
        weighted_rrf = WeightedRRFRetriever(exact)

        results: list[KISResult] = []
        failures: list[tuple[str, str]] = []
        query_timings: list[dict[str, Any]] = []
        query_manifest: list[dict[str, Any]] = []
        resolved_manifest_path = Path(manifest_path).resolve(strict=False)
        resolved_output = output.resolve(strict=False)
        output_files: list[Path] = (
            [resolved_manifest_path]
            if resolved_manifest_path.is_relative_to(resolved_output)
            else []
        )
        total_export_time = 0.0
        for query in queries:
            query_start = self.clock()
            variant_timing: list[dict[str, Any]] = []
            variants = query.variants()
            try:
                rankings: dict[str, KISResult] = {}
                for variant in variants:
                    encode_start = self.clock()
                    vector = encoder.encode(variant.text)
                    encode_seconds = self.clock() - encode_start
                    retrieval_start = self.clock()
                    rankings[variant.variant_id] = exact.search_vector(
                        query_id=f"{query.query_id}::{variant.variant_id}",
                        query_vector=vector,
                        top_k=config.top_k_per_variant,
                    )
                    retrieval_seconds = self.clock() - retrieval_start
                    variant_timing.append(
                        {
                            "variant_id": variant.variant_id,
                            "variant_type": variant.variant_type.value,
                            "encode_seconds": encode_seconds,
                            "retrieval_seconds": retrieval_seconds,
                        }
                    )
                fusion_start = self.clock()
                result = weighted_rrf.fuse_rankings(
                    query_id=query.query_id,
                    variants=variants,
                    rankings=rankings,
                    output_top_k=(
                        config.output_top_k_override
                        if config.output_top_k_override is not None
                        else query.output_top_k
                    ),
                    rrf_constant=config.rrf_constant,
                )
                fusion_seconds = self.clock() - fusion_start
                isolated_dir = output / "queries" / safe_query_directory_name(query.query_id)
                isolated_dir.mkdir(parents=True, exist_ok=True)
                export_start = self.clock()
                isolated_jsonl = isolated_dir / "top100.jsonl"
                self.exporter.export(result, isolated_jsonl)
                _write_internal_csv((result,), isolated_dir / "top100.csv")
                isolated_inspection = build_candidate_inspection(
                    (result,),
                    registry,
                    manifest,
                    isolated_dir,
                    top_n=config.inspection_top_n,
                    create_contact_sheet=False,
                )
                export_seconds = self.clock() - export_start
                total_export_time += export_seconds
                results.append(result)
                output_files.extend(
                    [
                        isolated_jsonl,
                        isolated_dir / "top100.csv",
                        isolated_inspection.json_path,
                        isolated_inspection.markdown_path,
                    ]
                )
                query_timings.append(
                    {
                        "query_id": query.query_id,
                        "status": "SUCCESS",
                        "variants": variant_timing,
                        "fusion_seconds": fusion_seconds,
                        "export_seconds": export_seconds,
                        "total_seconds": self.clock() - query_start,
                    }
                )
                query_manifest.append(
                    self._query_manifest_record(
                        query,
                        variants,
                        status="SUCCESS",
                        failure_reason=None,
                    )
                )
            except Exception as exc:
                reason = f"{type(exc).__name__}: {exc}"
                failures.append((query.query_id, reason))
                query_timings.append(
                    {
                        "query_id": query.query_id,
                        "status": "FAILED",
                        "variants": variant_timing,
                        "fusion_seconds": None,
                        "export_seconds": None,
                        "total_seconds": self.clock() - query_start,
                        "failure_reason": reason,
                    }
                )
                query_manifest.append(
                    self._query_manifest_record(
                        query,
                        variants,
                        status="FAILED",
                        failure_reason=reason,
                    )
                )
                if not config.continue_on_query_error:
                    break

        combined_jsonl = output / "top100.jsonl"
        combined_csv = output / "top100.csv"
        combined_export_start = self.clock()
        if results:
            ordered_results = tuple(sorted(results, key=lambda item: item.query_id))
            self.exporter.export(ordered_results, combined_jsonl)
            _write_internal_csv(ordered_results, combined_csv)
            inspection = build_candidate_inspection(
                ordered_results,
                registry,
                manifest,
                output,
                top_n=config.inspection_top_n,
                create_contact_sheet=config.create_contact_sheet,
            )
            output_files.extend(
                [
                    combined_jsonl,
                    combined_csv,
                    inspection.json_path,
                    inspection.markdown_path,
                ]
            )
            if inspection.contact_sheet_path is not None:
                output_files.append(inspection.contact_sheet_path)
        else:
            combined_jsonl.write_text("", encoding="utf-8")
            combined_csv.write_text("", encoding="utf-8")
            (output / "candidates.json").write_text(
                json.dumps({"records": [], "warnings": ["no successful queries"]}, indent=2)
                + "\n",
                encoding="utf-8",
            )
            (output / "candidate_inspection.md").write_text(
                "# Candidate Inspection\n\n- No successful queries.\n",
                encoding="utf-8",
            )
            output_files.extend(
                [
                    combined_jsonl,
                    combined_csv,
                    output / "candidates.json",
                    output / "candidate_inspection.md",
                ]
            )
        total_export_time += self.clock() - combined_export_start
        validation_start = self.clock()
        validation = self.validator.validate(combined_jsonl, registry)
        validation_seconds = self.clock() - validation_start
        validation_path = output / "validation_report.json"
        validation_path.write_text(
            json.dumps(
                _validation_payload(validation),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        output_files.append(validation_path)

        timings = {
            "discovery_seconds": float((bootstrap_timings or {}).get("discovery_seconds", 0.0)),
            "manifest_load_or_build_seconds": float(
                (bootstrap_timings or {}).get("manifest_load_or_build_seconds", 0.0)
            ),
            "registry_load_seconds": registry_time,
            "model_load_seconds": model_time,
            "queries": query_timings,
            "export_seconds": total_export_time,
            "validation_seconds": validation_seconds,
            "total_batch_seconds": (
                float((bootstrap_timings or {}).get("pre_runner_total_seconds", 0.0))
                + self.clock()
                - batch_start
            ),
            "corpus_video_count": len(manifest.videos),
            "corpus_feature_row_count": registry.total_rows,
        }
        timings_path = output / "timings.json"
        timings_path.write_text(
            json.dumps(timings, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        output_files.append(timings_path)

        model_metadata = dict(encoder.identifiers)
        final_exit_code = 0 if validation.valid and not failures else 2
        relative_outputs = tuple(
            sorted(
                {
                    str(path.relative_to(output)).replace("\\", "/")
                    for path in output_files
                }
                | {"run_manifest.json", "run_summary.md"},
                key=str.casefold,
            )
        )
        run_manifest = {
            "schema_version": 1,
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "git_commit_hash": _git_commit_hash(),
            "model_identifier": model_metadata.get("model", "unknown"),
            "model_metadata": model_metadata,
            "device": config.device,
            "manifest_fingerprint": manifest.fingerprint,
            "feature_manifest": str(Path(manifest_path)),
            "video_count": len(manifest.videos),
            "feature_row_count": registry.total_rows,
            "queries": query_manifest,
            "rrf_constant": config.rrf_constant,
            "top_k_per_variant": config.top_k_per_variant,
            "output_top_k_override": config.output_top_k_override,
            "retrieval_backend": "exact_chunked_numpy_cosine",
            "fusion_strategy": "weighted_reciprocal_rank_fusion",
            "temporal_suppression": False,
            "exporter_mode": "proposed_shared_core_jsonl",
            "validation_result": _validation_payload(validation),
            "output_filenames": relative_outputs,
            "failures": [
                {"query_id": query_id, "failure_reason": reason}
                for query_id, reason in failures
            ],
            "exit_code": final_exit_code,
        }
        run_manifest_path = output / "run_manifest.json"
        run_manifest_path.write_text(
            json.dumps(run_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        output_files.append(run_manifest_path)
        summary_path = output / "run_summary.md"
        summary_path.write_text(
            self._summary_markdown(run_manifest, timings),
            encoding="utf-8",
        )
        output_files.append(summary_path)
        timings["total_batch_seconds"] = (
            float((bootstrap_timings or {}).get("pre_runner_total_seconds", 0.0))
            + self.clock()
            - batch_start
        )
        timings_path.write_text(
            json.dumps(timings, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        summary_path.write_text(
            self._summary_markdown(run_manifest, timings),
            encoding="utf-8",
        )
        return ContestRunOutcome(
            exit_code=final_exit_code,
            successful_query_ids=tuple(sorted(result.query_id for result in results)),
            failed_queries=tuple(failures),
            validation=validation,
            output_files=tuple(sorted(set(output_files), key=lambda path: str(path).casefold())),
        )

    @staticmethod
    def _query_manifest_record(
        query: ContestQuery,
        variants: Sequence[Any],
        *,
        status: str,
        failure_reason: str | None,
    ) -> dict[str, Any]:
        return {
            "query_id": query.query_id,
            "status": status,
            "failure_reason": failure_reason,
            "output_top_k": query.output_top_k,
            "metadata": dict(query.metadata or {}),
            "variants": [
                {
                    "variant_id": variant.variant_id,
                    "text": variant.text,
                    "language": variant.language.value,
                    "variant_type": variant.variant_type.value,
                    "weight": variant.weight,
                }
                for variant in variants
            ],
        }

    @staticmethod
    def _summary_markdown(
        run_manifest: Mapping[str, Any],
        timings: Mapping[str, Any],
    ) -> str:
        lines = [
            "# system_tai Contest KIS Run",
            "",
            f"- Exit code: `{run_manifest['exit_code']}`",
            f"- Validation valid: `{str(run_manifest['validation_result']['valid']).lower()}`",
            f"- Corpus videos: {run_manifest['video_count']}",
            f"- Corpus feature rows: {run_manifest['feature_row_count']}",
            f"- Device: `{run_manifest['device']}`",
            f"- Model: `{run_manifest['model_identifier']}`",
            f"- Backend: `{run_manifest['retrieval_backend']}`",
            f"- Fusion: `{run_manifest['fusion_strategy']}`",
            "- Temporal suppression: `false`",
            "- `top100.csv` is internal convenience output, not official BTC format.",
            "",
            "## Query status",
            "",
        ]
        lines.extend(
            f"- `{query['query_id']}`: {query['status']}"
            + (f" — {query['failure_reason']}" if query["failure_reason"] else "")
            for query in run_manifest["queries"]
        )
        lines.extend(
            [
                "",
                "## Timings",
                "",
                f"- Discovery: {timings['discovery_seconds']:.6f}s",
                f"- Manifest load/build: {timings['manifest_load_or_build_seconds']:.6f}s",
                f"- Registry load: {timings['registry_load_seconds']:.6f}s",
                f"- Model load: {timings['model_load_seconds']:.6f}s",
                f"- Export: {timings['export_seconds']:.6f}s",
                f"- Validation: {timings['validation_seconds']:.6f}s",
                f"- Total batch: {timings['total_batch_seconds']:.6f}s",
            ]
        )
        return "\n".join(lines) + "\n"
