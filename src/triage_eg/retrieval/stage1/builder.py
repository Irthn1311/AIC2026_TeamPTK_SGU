"""Atomic Stage 1 compact-catalog and global-vector index builder."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from triage_eg.data.dataset_survey import KAGGLE_INPUT_ROOT, _is_within
from triage_eg.retrieval.stage1.catalog import CatalogData, load_catalog_rows, write_compact_catalog
from triage_eg.retrieval.stage1.contracts import STAGE1_VERSION, EncoderContract
from triage_eg.retrieval.stage1.stage0_loader import Stage0Bundle, load_stage0_bundle


@dataclass(frozen=True)
class Stage1BuildConfig:
    stage0_root: Path
    dataset_root: Path
    output_root: Path
    backend: str = "numpy_exact"
    metric: str = "cosine"
    dimension: int = 512
    search_chunk_rows: int = 16_384
    expected_rows: int = 177_321
    expected_videos: int = 873
    self_queries: int = 100
    seed: int = 2026
    overwrite: bool = False
    reuse_index: bool = False
    strict_root: bool = False
    repo_root: Path | None = None
    build_git_commit: str | None = None

    def __post_init__(self) -> None:
        if self.backend not in {"numpy_exact", "faiss_flat_ip"}:
            raise ValueError("Unsupported backend")
        if self.metric not in {"cosine", "dot"}:
            raise ValueError("metric must be cosine or dot")
        if self.backend == "faiss_flat_ip":
            raise ValueError(
                "faiss_flat_ip is optional and not enabled in Stage 1 v0.1; use numpy_exact"
            )
        if self.overwrite and self.reuse_index:
            raise ValueError("overwrite and reuse_index cannot both be enabled")
        if (
            min(
                self.dimension,
                self.search_chunk_rows,
                self.expected_rows,
                self.expected_videos,
                self.self_queries,
            )
            <= 0
        ):
            raise ValueError("Build dimensions/counts must be positive")


@dataclass(frozen=True)
class BuildResult:
    output_root: Path
    summary: dict[str, Any]
    index_manifest: dict[str, Any]
    reused: bool


def resolve_git_commit(
    repo_root: Path, explicit_commit: str | None = None
) -> tuple[str, str]:
    """Resolve build provenance without invoking a shell."""

    explicit = (explicit_commit or "").strip()
    if explicit:
        return explicit, "CLI"
    environment = os.environ.get("AIC_RESOLVED_GIT_COMMIT", "").strip()
    if environment:
        return environment, "ENV"
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return "UNKNOWN", "UNKNOWN"
    commit = result.stdout.strip()
    if result.returncode == 0 and commit:
        return commit, "GIT_AUTO_DETECT"
    return "UNKNOWN", "UNKNOWN"


def corpus_readiness_for_self_status(status: str) -> str:
    """Map the self-integrity aggregate onto corpus-index readiness."""

    mapping = {
        "PASS": "READY",
        "PASS_WITH_WARNINGS": "READY_WITH_TIE_WARNINGS",
        "FAIL": "BLOCKED_SELF_RETRIEVAL_FAILED",
    }
    try:
        return mapping[status]
    except KeyError as error:
        raise ValueError(f"Unknown self-retrieval status: {status}") from error


def _hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def _source_records(bundle: Stage0Bundle, dataset_root: Path) -> tuple[list[dict[str, Any]], str]:
    records = []
    seen_video_ids: set[str] = set()
    for item in sorted(bundle.clip_records, key=lambda value: value["video_id"]):
        video_id = str(item["video_id"])
        if video_id in seen_video_ids:
            raise ValueError(f"Duplicate clip manifest video_id: {video_id}")
        seen_video_ids.add(video_id)
        path = dataset_root / item["relative_path"]
        if not path.is_file() or not _is_within(path.resolve(), dataset_root.resolve()):
            raise FileNotFoundError(f"Manifest-referenced CLIP source missing/unsafe: {path}")
        records.append(
            {
                "video_id": video_id,
                "relative_path": item["relative_path"],
                "size_bytes": path.stat().st_size,
                "shape": item["shape"],
                "dtype": item["dtype"],
                "row_count": item["row_count"],
                "dimension": item["dimension"],
            }
        )
    payload = {"stage0_config_fingerprint": bundle.summary["config_fingerprint"], "clips": records}
    return records, _hash(payload)


def _build_fingerprint(config: Stage1BuildConfig) -> str:
    payload = asdict(config)
    for name in ("overwrite", "reuse_index", "repo_root", "build_git_commit"):
        payload.pop(name)
    for name in ("stage0_root", "dataset_root", "output_root"):
        payload[name] = str(Path(payload[name]).resolve(strict=False))
    return _hash(payload)


def _validate_roots(config: Stage1BuildConfig) -> tuple[Path, Path]:
    dataset = config.dataset_root.expanduser().resolve(strict=False)
    output = config.output_root.expanduser().resolve(strict=False)
    if not dataset.is_dir():
        raise FileNotFoundError(f"Dataset root missing: {dataset}")
    if config.strict_root and not _is_within(dataset, KAGGLE_INPUT_ROOT):
        raise ValueError("Strict dataset root must be below /kaggle/input")
    if _is_within(output, dataset) or _is_within(output, KAGGLE_INPUT_ROOT.resolve(strict=False)):
        raise ValueError("Stage 1 output must not be below dataset or /kaggle/input")
    return dataset, output


def _write_encoder_files(output: Path) -> None:
    encoder_root = output / "encoder"
    encoder_root.mkdir(parents=True, exist_ok=True)
    contract = asdict(EncoderContract())
    (encoder_root / "encoder_contract.json").write_text(
        json.dumps(contract, indent=2) + "\n", encoding="utf-8"
    )
    (encoder_root / "compatibility_report.json").write_text(
        json.dumps(
            {
                "compatibility_status": "BLOCKED",
                "reason": "CLIP model identity is unverified; no encoder was auto-selected",
                "dimension_only_is_insufficient": True,
                "text_search_available": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _copy_vectors(
    config: Stage1BuildConfig,
    dataset_root: Path,
    catalog: CatalogData,
    records: list[dict[str, Any]],
    temporary_index: Path,
) -> None:
    total = len(catalog.rows)
    vector_tmp = temporary_index / "clip_vectors.f16.npy"
    norm_tmp = temporary_index / "vector_norms.f32.npy"
    target = np.lib.format.open_memmap(
        vector_tmp, mode="w+", dtype=np.float16, shape=(total, config.dimension)
    )
    norms = np.lib.format.open_memmap(norm_tmp, mode="w+", dtype=np.float32, shape=(total,))
    try:
        rows_by_video: dict[str, list[dict[str, Any]]] = {}
        for row in catalog.rows:
            rows_by_video.setdefault(row["video_id"], []).append(row)
        offset = 0
        record_by_video = {item["video_id"]: item for item in records}
        for video_id in sorted(rows_by_video):
            record = record_by_video.get(video_id)
            if record is None:
                raise ValueError(f"No clip manifest record for {video_id}")
            source = np.load(
                dataset_root / record["relative_path"], mmap_mode="r", allow_pickle=False
            )
            try:
                expected_rows = rows_by_video[video_id]
                if source.ndim != 2 or source.shape != (len(expected_rows), config.dimension):
                    raise ValueError(f"CLIP source shape mismatch for {video_id}: {source.shape}")
                manifest_shape = tuple(int(value) for value in record["shape"])
                if tuple(source.shape) != manifest_shape:
                    raise ValueError(f"CLIP manifest shape mismatch for {video_id}")
                if int(record["row_count"]) != len(expected_rows):
                    raise ValueError(f"CLIP manifest row count mismatch for {video_id}")
                if int(record["dimension"]) != config.dimension:
                    raise ValueError(f"CLIP manifest dimension mismatch for {video_id}")
                if source.dtype != np.float16 or str(record["dtype"]) != "float16":
                    raise ValueError(f"CLIP source dtype mismatch for {video_id}")
                if [row["clip_row_index"] for row in expected_rows] != list(
                    range(len(expected_rows))
                ):
                    raise ValueError(f"Catalog CLIP row order mismatch for {video_id}")
                for start in range(0, len(source), config.search_chunk_rows):
                    stop = min(start + config.search_chunk_rows, len(source))
                    chunk = np.asarray(source[start:stop])
                    if not np.isfinite(chunk).all():
                        raise ValueError(f"CLIP source finiteness mismatch for {video_id}")
                    target[offset + start : offset + stop] = chunk
                    norms[offset + start : offset + stop] = np.linalg.norm(
                        chunk.astype(np.float32), axis=1
                    )
                offset += len(source)
            finally:
                source_mmap = getattr(source, "_mmap", None)
                if source_mmap is not None:
                    source_mmap.close()
        if offset != total or np.any(norms <= 0) or not np.isfinite(norms).all():
            raise ValueError("Global vector copy/norm validation failed")
        target.flush()
        norms.flush()
    finally:
        for array in (target, norms):
            array_mmap = getattr(array, "_mmap", None)
            if array_mmap is not None:
                array_mmap.close()


def _index_manifest(
    config: Stage1BuildConfig,
    bundle: Stage0Bundle,
    source_fingerprint: str,
    build_fingerprint: str,
    started_at: str,
    build_git_commit: str,
    build_git_commit_source: str,
) -> dict[str, Any]:
    return {
        "stage1_version": STAGE1_VERSION,
        "dataset_version": "aic25-b1",
        "stage0_git_commit": bundle.summary["git_commit"],
        "stage0_config_fingerprint": bundle.summary["config_fingerprint"],
        "stage0_audit_version": bundle.summary["audit_version"],
        "vector_count": config.expected_rows,
        "dimension": config.dimension,
        "dtype": "float16",
        "catalog_order": "video_id_then_n",
        "metric_default": config.metric,
        "backend": config.backend,
        "source_clip_files": len(bundle.clip_records),
        "source_fingerprint": source_fingerprint,
        "build_git_commit": build_git_commit,
        "build_git_commit_source": build_git_commit_source,
        "build_config_fingerprint": build_fingerprint,
        "started_at": started_at,
        "completed_at": datetime.now(UTC).isoformat(),
        "status": "COMPLETE",
    }


def _publish_staged_output(staging: Path, output: Path, *, overwrite: bool) -> None:
    """Publish a complete staging tree while preserving the previous output on failure."""
    backup = output.with_name(f".{output.name}.previous")
    if backup.exists():
        raise FileExistsError(f"Stale Stage 1 backup exists: {backup}")
    if not output.exists():
        os.replace(staging, output)
        return
    if not overwrite:
        raise FileExistsError("Stage 1 output exists; use overwrite or reuse-index")
    os.replace(output, backup)
    try:
        os.replace(staging, output)
    except Exception:
        os.replace(backup, output)
        raise
    shutil.rmtree(backup)


def build_index(config: Stage1BuildConfig) -> BuildResult:
    started_at = datetime.now(UTC).isoformat()
    repo_root = (
        config.repo_root.expanduser().resolve(strict=False)
        if config.repo_root is not None
        else Path(__file__).resolve().parents[4]
    )
    build_git_commit, build_git_commit_source = resolve_git_commit(
        repo_root, config.build_git_commit
    )
    dataset, output = _validate_roots(config)
    bundle = load_stage0_bundle(config.stage0_root, require_full=config.expected_videos == 873)
    if (
        bundle.summary["mapping_rows"] != config.expected_rows
        or len(bundle.clip_records) != config.expected_videos
    ):
        raise ValueError("Stage 0 counts do not match configured Stage 1 expectations")
    records, source_fingerprint = _source_records(bundle, dataset)
    build_fingerprint = _build_fingerprint(config)
    existing_manifest_path = output / "index" / "index_manifest.json"
    if config.reuse_index:
        if not existing_manifest_path.is_file():
            raise FileNotFoundError("No complete index exists for reuse")
        existing = json.loads(existing_manifest_path.read_text(encoding="utf-8"))
        if (
            existing.get("status") != "COMPLETE"
            or existing.get("source_fingerprint") != source_fingerprint
            or existing.get("build_config_fingerprint") != build_fingerprint
        ):
            raise ValueError("Existing index fingerprint is stale or incompatible")
        summary = json.loads((output / "stage1_summary.json").read_text(encoding="utf-8"))
        return BuildResult(output, summary, existing, True)
    if output.exists() and not config.overwrite:
        raise FileExistsError("Stage 1 output exists; use overwrite or reuse-index")
    if config.overwrite and (output == Path(output.anchor) or len(output.parts) < 3):
        raise ValueError("Refusing broad overwrite target")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(f".{output.name}.building")
    if staging.exists():
        if staging == Path(staging.anchor) or len(staging.parts) < 3:
            raise ValueError("Refusing broad staging cleanup target")
        shutil.rmtree(staging)
    temporary_index = staging / "index"
    temporary_index.mkdir(parents=True)
    try:
        catalog = load_catalog_rows(bundle)
        if (
            len(catalog.rows) != config.expected_rows
            or len(catalog.video_table) != config.expected_videos
        ):
            raise ValueError("Catalog rows/videos do not match expectations")
        catalog_manifest = write_compact_catalog(catalog, temporary_index)
        _copy_vectors(config, dataset, catalog, records, temporary_index)
        manifest = _index_manifest(
            config,
            bundle,
            source_fingerprint,
            build_fingerprint,
            started_at,
            build_git_commit,
            build_git_commit_source,
        )
        (temporary_index / "index_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        _write_encoder_files(staging)
        (staging / "benchmark").mkdir(exist_ok=True)
        (staging / "queries").mkdir(exist_ok=True)
        (staging / "logs").mkdir(exist_ok=True)
        index_fingerprint = _hash(
            {"source": source_fingerprint, "build": build_fingerprint, "catalog": catalog_manifest}
        )
        summary = {
            "stage1_version": STAGE1_VERSION,
            "stage0_audit_version": bundle.summary["audit_version"],
            "dataset_version": "aic25-b1",
            "stage0_config_fingerprint": bundle.summary["config_fingerprint"],
            "index_fingerprint": index_fingerprint,
            "vector_count": config.expected_rows,
            "dimension": config.dimension,
            "dtype": "float16",
            "catalog_rows": config.expected_rows,
            "videos": config.expected_videos,
            "index_status": "COMPLETE",
            "build_git_commit": build_git_commit,
            "build_git_commit_source": build_git_commit_source,
            "self_retrieval_status": "PENDING",
            "encoder_compatibility_status": "BLOCKED",
            "text_search_available": False,
            "vector_search_available": True,
            "unknown_contracts": ["CLIP model identity"],
            "next_stage_readiness": {
                "corpus_index": "READY",
                "text_retrieval": "BLOCKED_PENDING_ENCODER_COMPATIBILITY",
            },
        }
        from triage_eg.retrieval.stage1.benchmark import run_self_retrieval

        self_report = run_self_retrieval(
            staging,
            samples=config.self_queries,
            top_k=5,
            seed=config.seed,
            chunk_rows=config.search_chunk_rows,
        )
        summary["self_retrieval_status"] = self_report["status"]
        summary["next_stage_readiness"]["corpus_index"] = (
            corpus_readiness_for_self_status(self_report["status"])
        )
        (staging / "stage1_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        (staging / "run_manifest.json").write_text(
            json.dumps(
                {
                    "stage1_version": STAGE1_VERSION,
                    "status": "COMPLETE",
                    "build_config_fingerprint": build_fingerprint,
                    "index_fingerprint": index_fingerprint,
                    "build_git_commit": build_git_commit,
                    "build_git_commit_source": build_git_commit_source,
                    "stage0_root": str(config.stage0_root.resolve(strict=False)),
                    "dataset_root": str(dataset),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        (staging / "contract_notes.json").write_text(
            json.dumps(
                {
                    "frame_source": "BTC_KEYFRAME_ONLY",
                    "original_frame_policy": bundle.contract_notes["original_frame_policy"],
                    "duplicate_frame_idx_policy": (
                        "INTERNAL_ROWS_PRESERVED_OUTPUT_PAIRS_DEDUPLICATED"
                    ),
                    "clip_model_identity": "UNVERIFIED",
                    "text_search_default": "BLOCKED",
                    "bbox_order": "NOT_USED_STAGE1",
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        (staging / "stage1_report.md").write_text(
            "# Stage 1 BTC Retrieval Baseline\n\n"
            "- Index: COMPLETE\n"
            f"- Self retrieval: {self_report['status']}\n"
            "- Vector search: available\n"
            "- Text search: BLOCKED pending verified encoder compatibility\n"
            "- Frame source: BTC keyframes only\n",
            encoding="utf-8",
        )
        _publish_staged_output(staging, output, overwrite=config.overwrite)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return BuildResult(output, summary, manifest, False)
