"""Phase 3 artifact loading, Phase 4 execution, and bounded audit artifacts."""

from __future__ import annotations

import csv
import json
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from system_tai.checkpointing.exporter import CheckpointExporter
from system_tai.common.schemas import KISResult, ValidationResult
from system_tai.data.corpus_discovery import CorpusManifest, load_corpus_manifest
from system_tai.features.btc_clip_store import FeatureStoreRegistry
from system_tai.kis.contest_runner import safe_query_directory_name
from system_tai.refinement.clip_encoder import (
    OpenAIClipRefinementEncoder,
    RefinementEncoder,
)
from system_tai.refinement.engine import ExactFrameRefiner, QueryRefinementOutcome
from system_tai.refinement.models import (
    Phase3Candidate,
    RefinementConfig,
    RefinementQuery,
)
from system_tai.refinement.video import (
    DecodeRequest,
    OpenCVVideoDecoder,
    RawVideoRegistry,
    VideoDecoder,
)
from system_tai.retrieval.multi_query import (
    QueryLanguage,
    QueryVariant,
    QueryVariantType,
)
from system_tai.validation.checkpoint_validator import CheckpointValidator


@dataclass(frozen=True, slots=True)
class Phase3RunArtifacts:
    run_directory: Path
    manifest_path: Path
    manifest: CorpusManifest
    queries: tuple[RefinementQuery, ...]
    run_manifest: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class RefinementRunOutcome:
    exit_code: int
    successful_query_ids: tuple[str, ...]
    failed_queries: tuple[tuple[str, str], ...]
    validation: ValidationResult
    output_files: tuple[Path, ...]


def _json_value(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _json_value(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    return value


def _read_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"required Phase 3 artifact missing: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON artifact {path.name}: {exc}") from exc


def _phase3_core_records(path: Path) -> tuple[dict[str, Any], ...]:
    if not path.is_file():
        raise FileNotFoundError(f"required Phase 3 artifact missing: {path}")
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid top100.jsonl line {line_number}: {exc}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"top100.jsonl line {line_number} is not an object")
        records.append(record)
    if not records:
        raise ValueError("Phase 3 top100.jsonl contains no records")
    return tuple(records)


def _variant_from_payload(payload: Any) -> QueryVariant:
    if not isinstance(payload, dict):
        raise ValueError("Phase 3 query variant must be an object")
    try:
        return QueryVariant(
            variant_id=str(payload["variant_id"]),
            text=str(payload["text"]),
            language=QueryLanguage(str(payload["language"])),
            variant_type=QueryVariantType(str(payload["variant_type"])),
            weight=float(payload["weight"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Phase 3 run manifest has an invalid query variant") from exc


def load_phase3_run(run_directory: Path) -> Phase3RunArtifacts:
    run_dir = Path(run_directory)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Phase 3 run directory not found: {run_dir}")
    run_manifest = _read_json(run_dir / "run_manifest.json")
    candidates_payload = _read_json(run_dir / "candidates.json")
    manifest_path = run_dir / "feature_manifest.json"
    manifest = load_corpus_manifest(manifest_path)
    core_records = _phase3_core_records(run_dir / "top100.jsonl")
    if not isinstance(run_manifest, dict):
        raise ValueError("Phase 3 run_manifest.json root must be an object")
    if not isinstance(candidates_payload, dict) or not isinstance(
        candidates_payload.get("records"), list
    ):
        raise ValueError("Phase 3 candidates.json must contain a records list")
    registry = FeatureStoreRegistry.from_manifest(
        manifest_path,
        expected_dimension=manifest.videos[0].embedding_dimension,
    )
    core_validation = CheckpointValidator().validate(run_dir / "top100.jsonl", registry=registry)
    if not core_validation.valid:
        raise ValueError("Phase 3 top100.jsonl is invalid")

    inspections: dict[tuple[str, int, str, int], dict[str, Any]] = {}
    for item in candidates_payload["records"]:
        if not isinstance(item, dict):
            raise ValueError("Phase 3 candidate inspection record must be an object")
        try:
            key = (
                str(item["query_id"]),
                int(item["rank"]),
                str(item["video_id"]),
                int(item["frame_id"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid Phase 3 candidate inspection record") from exc
        if key in inspections:
            raise ValueError(f"duplicate Phase 3 candidate inspection identity: {key}")
        inspections[key] = item

    candidates_by_query: dict[str, list[Phase3Candidate]] = {}
    for record in core_records:
        try:
            key = (
                str(record["query_id"]),
                int(record["rank"]),
                str(record["video_id"]),
                int(record["frame_id"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid Phase 3 core checkpoint record") from exc
        inspection = inspections.get(key)
        if inspection is None:
            raise ValueError(f"Phase 3 candidate provenance missing for {key}")
        provenance = {
            "fusion_score": inspection.get("fusion_score", 0.0),
            "variant_hit_count": inspection.get("variant_hit_count"),
            "best_individual_rank": inspection.get("best_individual_rank"),
            "per_variant": inspection.get("per_variant", []),
            "clip_row_diagnostic": inspection.get("clip_row_diagnostic", 0),
            "keyframe_order_diagnostic": inspection.get("keyframe_order_diagnostic", 0),
        }
        candidates_by_query.setdefault(key[0], []).append(
            Phase3Candidate(
                query_id=key[0],
                rank=key[1],
                video_id=key[2],
                frame_id=key[3],
                retrieval_score=float(inspection.get("fusion_score", 0.0)),
                retrieval_provenance=provenance,
            )
        )

    query_entries = run_manifest.get("queries")
    if not isinstance(query_entries, list):
        raise ValueError("Phase 3 run manifest queries must be a list")
    variants_by_query: dict[str, tuple[QueryVariant, ...]] = {}
    for entry in query_entries:
        if not isinstance(entry, dict) or entry.get("status") != "SUCCESS":
            continue
        query_id = str(entry.get("query_id", ""))
        variants_payload = entry.get("variants")
        if not isinstance(variants_payload, list) or not variants_payload:
            raise ValueError(f"Phase 3 query variants unavailable for {query_id}")
        variants_by_query[query_id] = tuple(
            _variant_from_payload(item) for item in variants_payload
        )
    queries: list[RefinementQuery] = []
    for query_id in sorted(candidates_by_query):
        variants = variants_by_query.get(query_id)
        if variants is None:
            raise ValueError(f"Phase 3 run manifest has no successful variants for {query_id}")
        candidates = tuple(
            sorted(candidates_by_query[query_id], key=lambda candidate: candidate.rank)
        )
        queries.append(RefinementQuery(query_id, variants, candidates))
    if not queries:
        raise ValueError("Phase 3 artifacts contain no successful query to refine")
    return Phase3RunArtifacts(
        run_directory=run_dir,
        manifest_path=manifest_path,
        manifest=manifest,
        queries=tuple(queries),
        run_manifest=run_manifest,
    )


def _validation_payload(validation: ValidationResult) -> dict[str, Any]:
    return {
        "valid": validation.valid,
        "errors": [_json_value(item) for item in validation.errors],
        "warnings": [_json_value(item) for item in validation.warnings],
    }


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_value(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _write_refined_csv(results: Sequence[KISResult], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("query_id", "rank", "video_id", "frame_id"),
            lineterminator="\n",
        )
        writer.writeheader()
        for result in sorted(results, key=lambda item: item.query_id):
            for item in result.ranked_candidates:
                writer.writerow(
                    {
                        "query_id": result.query_id,
                        "rank": item.rank,
                        "video_id": item.video_id,
                        "frame_id": item.frame_id,
                    }
                )
    return path


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
    return completed.stdout.strip() or None


def _write_contact_sheet(
    outcomes: Sequence[QueryRefinementOutcome],
    raw_videos: RawVideoRegistry,
    decoder: VideoDecoder,
    destination: Path,
) -> tuple[Path | None, tuple[str, ...]]:
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return None, ("refined contact sheet skipped because Pillow is unavailable",)
    cells: list[tuple[Any, str]] = []
    warnings: list[str] = []
    probe_cache: dict[str, Any] = {}
    for outcome in outcomes:
        by_identity = {
            (item.video_id, item.refined_frame_id): item
            for item in outcome.candidates
            if item.refined_frame_id is not None
        }
        for candidate in outcome.result.ranked_candidates:
            refined = by_identity.get((candidate.video_id, candidate.frame_id))
            if refined is None:
                continue
            record = raw_videos.get(candidate.video_id)
            if record.raw_video_path is None:
                warnings.append(f"contact sheet raw video missing for {candidate.video_id}")
                continue
            try:
                probe = probe_cache.get(candidate.video_id)
                if probe is None:
                    probe = decoder.probe(record)
                    probe_cache[candidate.video_id] = probe
                decoded = decoder.decode(DecodeRequest(probe, (candidate.frame_id,), 1))
                array = np.asarray(decoded.frames[0].image)
                if array.ndim != 3 or array.shape[2] != 3:
                    raise ValueError("decoded image is not a three-channel frame")
                image = Image.fromarray(np.asarray(array[:, :, ::-1], dtype=np.uint8).copy())
                image.thumbnail((240, 140))
                cells.append(
                    (
                        image.copy(),
                        f"#{candidate.rank} {candidate.video_id} "
                        f"{refined.candidate_frame_id}->{candidate.frame_id} "
                        f"t={candidate.frame_id / probe.fps:.3f}s",
                    )
                )
            except Exception as exc:
                warnings.append(
                    f"contact sheet frame unavailable for {candidate.video_id}/"
                    f"{candidate.frame_id}: {type(exc).__name__}: {exc}"
                )
    if not cells:
        return None, tuple(sorted(set(warnings or ["no refined frames for contact sheet"])))
    columns, cell_width, cell_height = 4, 260, 180
    rows = (len(cells) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * cell_width, rows * cell_height), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (image, caption) in enumerate(cells):
        x = (index % columns) * cell_width
        y = (index // columns) * cell_height
        canvas.paste(image, (x + 10, y + 5))
        draw.text((x + 10, y + 150), caption, fill="black")
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, format="JPEG", quality=82, optimize=True)
    return destination, tuple(sorted(set(warnings)))


class RefinementRunner:
    def __init__(
        self,
        *,
        decoder_factory: Callable[[], VideoDecoder] | None = None,
        encoder_factory: Callable[..., RefinementEncoder] | None = None,
        exporter: CheckpointExporter | None = None,
        validator: CheckpointValidator | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.decoder_factory = decoder_factory or OpenCVVideoDecoder
        self.encoder_factory = encoder_factory or OpenAIClipRefinementEncoder
        self.exporter = exporter or CheckpointExporter()
        self.validator = validator or CheckpointValidator()
        self.clock = clock

    def run(
        self,
        *,
        run_directory: Path,
        output_directory: Path,
        config: RefinementConfig,
        continue_on_query_error: bool = False,
        create_contact_sheet: bool = False,
    ) -> RefinementRunOutcome:
        run_start = self.clock()
        input_start = self.clock()
        artifacts = load_phase3_run(run_directory)
        input_seconds = self.clock() - input_start
        output = Path(output_directory)
        output.mkdir(parents=True, exist_ok=True)
        registry_start = self.clock()
        raw_videos = RawVideoRegistry.from_manifest(artifacts.manifest)
        registry_seconds = self.clock() - registry_start
        decoder = self.decoder_factory()
        model_start = self.clock()
        encoder = self.encoder_factory(
            device=config.device,
            allow_model_download=config.allow_model_download,
            cache_dir=config.clip_cache_dir,
        )
        model_seconds = self.clock() - model_start
        refiner = ExactFrameRefiner(
            raw_videos=raw_videos,
            decoder=decoder,
            encoder=encoder,
            clock=self.clock,
        )
        results: list[KISResult] = []
        outcomes: list[QueryRefinementOutcome] = []
        failures: list[tuple[str, str]] = []
        files: list[Path] = []
        for query in artifacts.queries:
            print(f"refinement query started: {query.query_id}")
            try:
                outcome = refiner.refine_query(query, config)
                results.append(outcome.result)
                outcomes.append(outcome)
                query_dir = output / "queries" / safe_query_directory_name(query.query_id)
                query_jsonl = query_dir / "refined_top100.jsonl"
                self.exporter.export(outcome.result, query_jsonl)
                files.extend(
                    [
                        query_jsonl,
                        _write_refined_csv((outcome.result,), query_dir / "refined_top100.csv"),
                        _write_json(query_dir / "refinement_candidates.json", outcome.candidates),
                        _write_json(
                            query_dir / "refinement_trace.json",
                            {
                                "query_id": query.query_id,
                                "candidates": outcome.candidates,
                                "warnings": outcome.warnings,
                            },
                        ),
                    ]
                )
                print(
                    f"refinement query completed: {query.query_id} "
                    f"records={len(outcome.result.ranked_candidates)}"
                )
            except Exception as exc:
                reason = f"{type(exc).__name__}: {exc}"
                failures.append((query.query_id, reason))
                print(f"refinement query failed: {query.query_id}: {reason}")
                if not continue_on_query_error:
                    break

        export_start = self.clock()
        combined_jsonl = output / "refined_top100.jsonl"
        if results:
            ordered = tuple(sorted(results, key=lambda result: result.query_id))
            self.exporter.export(ordered, combined_jsonl)
        else:
            combined_jsonl.write_text("", encoding="utf-8")
            ordered = ()
        combined_csv = _write_refined_csv(ordered, output / "refined_top100.csv")
        candidates_path = _write_json(
            output / "refinement_candidates.json",
            {
                "queries": [
                    {"query_id": item.query_id, "candidates": item.candidates} for item in outcomes
                ]
            },
        )
        trace_path = _write_json(
            output / "refinement_trace.json",
            {
                "schema_version": 1,
                "coordinate_contract": "frame_id is absolute original-video frame index",
                "queries": [
                    {
                        "query_id": item.query_id,
                        "candidates": item.candidates,
                        "warnings": item.warnings,
                    }
                    for item in outcomes
                ],
                "failures": [
                    {"query_id": query_id, "failure_reason": reason}
                    for query_id, reason in failures
                ],
            },
        )
        export_seconds = self.clock() - export_start
        contact_start = self.clock()
        contact_sheet_path: Path | None = None
        contact_warnings: tuple[str, ...] = ()
        if create_contact_sheet:
            contact_sheet_path, contact_warnings = _write_contact_sheet(
                outcomes,
                raw_videos,
                decoder,
                output / "refined_contact_sheet.jpg",
            )
            if contact_sheet_path is not None:
                files.append(contact_sheet_path)
        contact_sheet_seconds = self.clock() - contact_start if create_contact_sheet else 0.0
        validation_start = self.clock()
        validation = self.validator.validate(combined_jsonl)
        validation_seconds = self.clock() - validation_start
        validation_path = _write_json(
            output / "refinement_validation_report.json",
            _validation_payload(validation),
        )
        aggregate = self._aggregate_timings(outcomes)
        timings = {
            "input_artifact_load_seconds": input_seconds,
            "raw_video_registry_seconds": registry_seconds,
            "model_load_seconds": model_seconds,
            **aggregate,
            "queries": [{"query_id": item.query_id, **dict(item.timings)} for item in outcomes],
            "export_seconds": export_seconds,
            "contact_sheet_seconds": contact_sheet_seconds,
            "validation_seconds": validation_seconds,
            "total_run_seconds": self.clock() - run_start,
        }
        timings_path = _write_json(output / "refinement_timings.json", timings)
        artifact_paths = (
            *files,
            combined_jsonl,
            combined_csv,
            candidates_path,
            trace_path,
            timings_path,
            validation_path,
        )
        relative_outputs = sorted(
            {str(path.relative_to(output)).replace("\\", "/") for path in artifact_paths}
            | {"refinement_run_manifest.json", "refinement_summary.md"},
            key=str.casefold,
        )
        exit_code = 0 if validation.valid and not failures else 2
        manifest_payload = {
            "schema_version": 1,
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "git_commit_hash": _git_commit_hash(),
            "source_phase3_run": str(Path(run_directory)),
            "source_manifest_fingerprint": artifacts.manifest.fingerprint,
            "video_count": len(artifacts.manifest.videos),
            "model": dict(encoder.identifiers),
            "device": config.device,
            "decoder_backend": getattr(decoder, "backend_identifier", "unknown"),
            "configuration": config,
            "successful_query_ids": sorted(item.query_id for item in outcomes),
            "failed_query_ids": [query_id for query_id, _reason in failures],
            "failures": [
                {"query_id": query_id, "failure_reason": reason} for query_id, reason in failures
            ],
            "validation_result": _validation_payload(validation),
            "output_filenames": relative_outputs,
            "ranking_policy": "replace frame; preserve Phase 3 rank; deduplicate",
            "shared_frame_semantics": "original-video absolute frame index",
            "contact_sheet_requested": create_contact_sheet,
            "contact_sheet_status": (
                "created"
                if contact_sheet_path is not None
                else "not-created"
                if not create_contact_sheet
                else "skipped"
            ),
            "contact_sheet_warnings": contact_warnings,
            "exit_code": exit_code,
        }
        run_manifest_path = _write_json(output / "refinement_run_manifest.json", manifest_payload)
        summary_path = output / "refinement_summary.md"
        summary_path.write_text(self._summary(manifest_payload, timings), encoding="utf-8")
        files.extend(
            [
                combined_jsonl,
                combined_csv,
                candidates_path,
                trace_path,
                timings_path,
                validation_path,
                run_manifest_path,
                summary_path,
            ]
        )
        return RefinementRunOutcome(
            exit_code=exit_code,
            successful_query_ids=tuple(sorted(item.query_id for item in outcomes)),
            failed_queries=tuple(failures),
            validation=validation,
            output_files=tuple(sorted(set(files), key=lambda path: str(path).casefold())),
        )

    @staticmethod
    def _aggregate_timings(
        outcomes: Sequence[QueryRefinementOutcome],
    ) -> dict[str, float | int]:
        fields_to_sum = (
            "video_probe_seconds",
            "video_open_seconds",
            "coarse_decode_seconds",
            "coarse_encode_seconds",
            "coarse_score_seconds",
            "coarse_fusion_seconds",
            "fine_decode_seconds",
            "fine_encode_seconds",
            "fine_score_seconds",
            "fine_fusion_seconds",
            "candidate_total_seconds",
            "query_total_seconds",
            "decoded_frame_count",
            "encoded_image_count",
            "refined_candidate_count",
            "kept_original_count",
            "skipped_candidate_count",
            "failed_candidate_count",
            "missing_raw_video_count",
        )
        return {
            key: sum(float(item.timings.get(key, 0)) for item in outcomes)
            if key.endswith("seconds")
            else sum(int(item.timings.get(key, 0)) for item in outcomes)
            for key in fields_to_sum
        }

    @staticmethod
    def _summary(manifest: Mapping[str, Any], timings: Mapping[str, Any]) -> str:
        return "\n".join(
            [
                "# system_tai Phase 4 Refinement Run",
                "",
                f"- Validation valid: `{str(manifest['validation_result']['valid']).lower()}`",
                f"- Successful queries: {len(manifest['successful_query_ids'])}",
                f"- Failed queries: {len(manifest['failed_query_ids'])}",
                f"- Device: `{manifest['device']}`",
                f"- Decoder: `{manifest['decoder_backend']}`",
                "- Ranking: original Phase 3 order preserved; frame replacement only.",
                "- CSV is internal convenience output, not official BTC format.",
                "",
                "## Timings",
                "",
                f"- Input artifacts: {timings['input_artifact_load_seconds']:.6f}s",
                f"- Raw video registry: {timings['raw_video_registry_seconds']:.6f}s",
                f"- Model load: {timings['model_load_seconds']:.6f}s",
                f"- Total run: {timings['total_run_seconds']:.6f}s",
                "",
            ]
        )
