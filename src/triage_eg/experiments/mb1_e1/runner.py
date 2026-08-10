"""Bounded MB1-E1 interval re-evaluation of frozen M0 versus frozen M1."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

from triage_eg.data.stage0_audit.asset_resolver import discover_layout, resolve_assets
from triage_eg.experiments.mb1 import validate_annotation
from triage_eg.experiments.moment_m1 import (
    M1Settings,
    OpenCVRawVideoDecoder,
    VerifiedClipLocalImageEncoder,
    coarse_frame_indices,
    dense_frame_indices,
    select_best_frame,
)
from triage_eg.retrieval.stage1b.adapters.openai_clip_official import (
    materialize_kaggle_expanded_tokenizer,
    preflight_official_openai_clip,
    resolve_official_asset_paths,
)
from triage_eg.retrieval.stage1b.assets import load_multimodal_encoder
from triage_eg.retrieval.stage1b.contracts import CandidateContract
from triage_eg.retrieval.stage1b.writers import write_json, write_jsonl

from .metrics import build_mb1_e1_metrics, distance_to_interval
from .visuals import render_blinded_sheet

MB1_E1_VERSION = "0.1.0"
MB1_E1_METHOD_M0 = "SOURCE_ANCHOR_FRAME"
MB1_E1_METHOD_M1 = "LOCAL_RAW_CLIP_COARSE_TO_FINE"
FROZEN_M1_SETTINGS = M1Settings()
BUNDLE_FILES = (
    "mb1_e1_summary.json",
    "mb1_e1_metrics.json",
    "moment_results.jsonl",
    "run_manifest.json",
    "issues.jsonl",
    "benchmark/mb1_ai_semantic_moments.jsonl",
    "benchmark/mb1_ai_semantic_moments.sha256",
    "visuals/review_key.json",
)
HEAVY_SUFFIXES = {".pt", ".pth", ".bin", ".npy", ".npz", ".mp4", ".avi", ".mkv", ".mov"}


@dataclass(frozen=True)
class MB1E1Config:
    dataset_root: Path
    candidate_manifest_path: Path
    annotation_path: Path
    stage1b_root: Path
    clip_asset_root: Path
    output_root: Path
    seed: int = 2026
    device: str = "auto"
    batch_size: int = 16
    build_git_commit: str | None = None

    def __post_init__(self) -> None:
        if self.seed != 2026:
            raise ValueError("MB1-E1 blind randomization seed is frozen at 2026")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_benchmark_preserving_hash(source: str | Path, target: str | Path) -> str:
    source_path = Path(source).expanduser().resolve(strict=True)
    target_path = Path(target).expanduser().resolve(strict=False)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    source_hash = sha256_file(source_path)
    shutil.copyfile(source_path, target_path)
    if sha256_file(target_path) != source_hash:
        raise RuntimeError("MB1 annotation benchmark hash was not preserved")
    return source_hash


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"JSONL row {line_number} is not an object: {path}")
        rows.append(value)
    return rows


def load_interval_benchmark(
    candidate_manifest_path: str | Path, annotation_path: str | Path
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    manifest_path = Path(candidate_manifest_path).expanduser().resolve(strict=True)
    annotations_path = Path(annotation_path).expanduser().resolve(strict=True)
    candidates = _read_jsonl(manifest_path)
    by_candidate = {str(row["candidate_id"]): row for row in candidates}
    if len(by_candidate) != len(candidates):
        raise ValueError("MB1 candidate IDs must be unique")
    annotations = _read_jsonl(annotations_path)
    if len(annotations) != 20:
        raise ValueError(f"MB1-E1 requires exactly 20 annotations; found {len(annotations)}")
    moment_ids: set[str] = set()
    for annotation in annotations:
        validate_annotation(annotation)
        moment_id = str(annotation["moment_id"])
        if moment_id in moment_ids:
            raise ValueError(f"Duplicate MB1 moment_id: {moment_id}")
        moment_ids.add(moment_id)
        if annotation["annotation_confidence"] not in {"HIGH", "MEDIUM"}:
            raise ValueError("MB1-E1 accepts HIGH/MEDIUM annotations only")
        candidate = by_candidate.get(str(annotation["source_candidate_id"]))
        if candidate is None:
            raise ValueError(f"Missing source candidate for {moment_id}")
        start, end = int(candidate["window_start_frame"]), int(candidate["window_end_frame"])
        if annotation["video_id"] != candidate["video_id"]:
            raise ValueError(f"Video mismatch for {moment_id}")
        if not (
            start <= int(annotation["acceptable_start_frame"])
            <= int(annotation["acceptable_end_frame"])
            <= end
        ):
            raise ValueError(f"Annotation interval is outside candidate window for {moment_id}")
        if not start <= int(candidate["source_anchor_frame"]) <= end:
            raise ValueError(f"M0 anchor is outside candidate window for {moment_id}")
    return annotations, by_candidate


def _selected_contract(stage1b_root: Path) -> dict[str, Any]:
    path = stage1b_root / "encoder/selected_encoder_contract.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("compatibility_status") != "VERIFIED":
        raise RuntimeError("STAGE1B_ENCODER_NOT_VERIFIED")
    if value.get("implementation") != "openai_clip" or value.get("architecture") != "ViT-B/32":
        raise RuntimeError("MB1-E1 requires verified official OpenAI CLIP ViT-B/32")
    return value


def _prepare_verified_clip(config: MB1E1Config) -> tuple[Any, dict[str, Any]]:
    selected = _selected_contract(config.stage1b_root.expanduser().resolve(strict=True))
    paths = resolve_official_asset_paths(config.clip_asset_root)
    runtime_source, tokenizer_materialized = materialize_kaggle_expanded_tokenizer(
        paths.source_root, config.output_root / "_runtime_cache/openai_clip_source"
    )
    paths = resolve_official_asset_paths(
        config.clip_asset_root,
        source_root=runtime_source,
        checkpoint_path=paths.checkpoint_path,
        asset_manifest_path=paths.asset_manifest_path,
    )
    provenance, issues, _ = preflight_official_openai_clip(
        paths, requested_device=config.device
    )
    blockers = [item for item in issues if item.get("severity") == "ERROR"]
    if blockers:
        raise RuntimeError(str(blockers[0].get("code", "CLIP_LOAD_FAILED")))
    if provenance.get("checkpoint_sha256") != selected.get("checkpoint_sha256"):
        raise RuntimeError("STAGE1B_ENCODER_NOT_VERIFIED: checkpoint SHA mismatch")
    candidate = replace(
        CandidateContract.from_dict(selected),
        source_root=str(runtime_source),
        checkpoint_path=str(paths.checkpoint_path),
        asset_manifest_path=str(paths.asset_manifest_path),
        device=config.device,
        batch_size=config.batch_size,
    )
    adapter = load_multimodal_encoder(candidate, provenance=provenance)
    return adapter, {
        **provenance,
        "candidate_id": candidate.candidate_id,
        "compatibility_status": selected["compatibility_status"],
        "tokenizer_materialized_from_kaggle_expansion": tokenizer_materialized,
    }


def _cosine_scores(text_embedding: np.ndarray, image_embeddings: np.ndarray) -> np.ndarray:
    text = np.asarray(text_embedding, dtype=np.float32).reshape(-1)
    images = np.asarray(image_embeddings, dtype=np.float32)
    if text.shape != (512,) or images.ndim != 2 or images.shape[1] != 512:
        raise ValueError("MB1-E1 cosine inputs must be 512-dimensional")
    text_norm, image_norms = np.linalg.norm(text), np.linalg.norm(images, axis=1)
    if text_norm == 0 or np.any(image_norms == 0):
        raise ValueError("MB1-E1 cosine inputs must have non-zero norms")
    return ((images @ text) / (image_norms * text_norm)).astype(np.float32, copy=False)


def refine_inside_candidate_window(
    *,
    decoder: Any,
    image_encoder: Any,
    text_embedding: np.ndarray,
    window_start: int,
    window_end: int,
    source_anchor_frame: int,
) -> tuple[dict[str, Any], dict[int, np.ndarray]]:
    """Apply frozen M1 sampling/argmax, bounded by the supplied MB1 window."""

    if not 0 <= window_start <= source_anchor_frame <= window_end < decoder.info.total_frames:
        raise ValueError("candidate window is invalid for the raw video")
    started = monotonic()
    coarse_ids = coarse_frame_indices(
        window_start,
        window_end,
        source_anchor_frame,
        FROZEN_M1_SETTINGS.coarse_stride_frames,
    )
    coarse_frames = decoder.decode_indices(coarse_ids)
    coarse_embeddings = image_encoder.encode(coarse_frames)
    coarse_scores = _cosine_scores(text_embedding, coarse_embeddings)
    peak, peak_score = select_best_frame(coarse_ids, coarse_scores)
    anchor_position = coarse_ids.index(source_anchor_frame)

    dense_ids = dense_frame_indices(
        window_start, window_end, peak, FROZEN_M1_SETTINGS.dense_radius_frames
    )
    dense_frames = decoder.decode_indices(dense_ids)
    dense_scores = _cosine_scores(text_embedding, image_encoder.encode(dense_frames))
    refined, refined_score = select_best_frame(dense_ids, dense_scores)
    if not window_start <= refined <= window_end:
        raise RuntimeError("M1 prediction escaped the MB1 candidate window")
    images = {frame.actual_frame_idx: frame.image for frame in coarse_frames + dense_frames}
    return {
        "local_window_start": window_start,
        "local_window_end": window_end,
        "coarse_stride_frames": FROZEN_M1_SETTINGS.coarse_stride_frames,
        "dense_radius_frames": FROZEN_M1_SETTINGS.dense_radius_frames,
        "coarse_sample_count": len(coarse_ids),
        "coarse_peak_frame": peak,
        "coarse_peak_score": peak_score,
        "dense_candidate_count": len(dense_ids),
        "m0_score": float(coarse_scores[anchor_position]),
        "m1_frame": refined,
        "m1_score": refined_score,
        "elapsed_ms": (monotonic() - started) * 1000,
    }, images


def build_moment_result(
    annotation: dict[str, Any], candidate: dict[str, Any], search: dict[str, Any]
) -> dict[str, Any]:
    """Build one result while keeping interval GT primary and the manifest anchor as M0."""

    start = int(annotation["acceptable_start_frame"])
    end = int(annotation["acceptable_end_frame"])
    preferred = int(annotation["preferred_frame"])
    m0_frame = int(candidate["source_anchor_frame"])
    m1_frame = int(search["m1_frame"])
    m0_distance = distance_to_interval(m0_frame, start, end)
    m1_distance = distance_to_interval(m1_frame, start, end)
    outcome = (
        "M1_WINS"
        if m1_distance < m0_distance
        else "M0_WINS"
        if m0_distance < m1_distance
        else "TIES"
    )
    return {
        **annotation,
        "candidate_window_start": int(candidate["window_start_frame"]),
        "candidate_window_end": int(candidate["window_end_frame"]),
        "m0_method": MB1_E1_METHOD_M0,
        "m0_frame": m0_frame,
        "m0_score": search["m0_score"],
        "m0_interval_hit": m0_distance == 0,
        "m0_distance_to_interval": m0_distance,
        "m0_preferred_frame_error": abs(m0_frame - preferred),
        "m1_method": MB1_E1_METHOD_M1,
        "m1_frame": m1_frame,
        "m1_score": search["m1_score"],
        "m1_interval_hit": m1_distance == 0,
        "m1_distance_to_interval": m1_distance,
        "m1_preferred_frame_error": abs(m1_frame - preferred),
        "score_gain": float(search["m1_score"] - search["m0_score"]),
        "interval_error_improvement": m0_distance - m1_distance,
        "pairwise_outcome": outcome,
        "m1_diagnostics": search,
    }


def preflight_mb1_e1(config: MB1E1Config) -> dict[str, Any]:
    annotations, candidates = load_interval_benchmark(
        config.candidate_manifest_path, config.annotation_path
    )
    dataset = config.dataset_root.expanduser().resolve(strict=True)
    stage1b = config.stage1b_root.expanduser().resolve(strict=True)
    selected = _selected_contract(stage1b)
    if config.output_root.exists():
        raise FileExistsError(f"MB1-E1 output already exists: {config.output_root}")
    video_partitions, keyframe_partitions = discover_layout(dataset)
    missing = []
    for video_id in sorted({str(row["video_id"]) for row in annotations}):
        assets = resolve_assets(dataset, video_id, video_partitions, keyframe_partitions)
        if not assets.video.is_file():
            missing.append(video_id)
    if missing:
        raise FileNotFoundError(f"Missing MB1-E1 raw videos: {missing}")
    paths = resolve_official_asset_paths(config.clip_asset_root)
    if not paths.checkpoint_path.is_file() or not paths.source_root.is_dir():
        raise FileNotFoundError("Offline OpenAI CLIP asset is incomplete")
    return {
        "status": "READY",
        "annotation_count": len(annotations),
        "candidate_count_referenced": len({row["source_candidate_id"] for row in annotations}),
        "candidate_manifest_count": len(candidates),
        "annotation_sha256": sha256_file(config.annotation_path),
        "encoder_status": selected["compatibility_status"],
        "checkpoint_sha256": selected["checkpoint_sha256"],
        "network_required": False,
        "m0": MB1_E1_METHOD_M0,
        "m1": MB1_E1_METHOD_M1,
        "m1_search_window": "EXACT_MB1_CANDIDATE_WINDOW",
    }


DecoderFactory = Callable[[str, Path], Any]


def run_mb1_e1(
    config: MB1E1Config,
    *,
    adapter: Any | None = None,
    decoder_factory: DecoderFactory = OpenCVRawVideoDecoder,
    render_visuals: bool = True,
) -> dict[str, Any]:
    preflight = preflight_mb1_e1(config)
    output = config.output_root.expanduser().resolve(strict=False)
    output.mkdir(parents=True)
    annotations, candidates = load_interval_benchmark(
        config.candidate_manifest_path, config.annotation_path
    )
    annotation_copy = output / "benchmark/mb1_ai_semantic_moments.jsonl"
    source_hash = copy_benchmark_preserving_hash(config.annotation_path, annotation_copy)
    (output / "benchmark/mb1_ai_semantic_moments.sha256").write_text(
        f"{source_hash}  mb1_ai_semantic_moments.jsonl\n", encoding="utf-8"
    )

    owned_adapter = adapter is None
    active_adapter, clip_provenance = (
        _prepare_verified_clip(config)
        if adapter is None
        else (adapter, {"candidate_id": "TEST_ADAPTER", "compatibility_status": "INJECTED"})
    )
    image_encoder = VerifiedClipLocalImageEncoder(active_adapter)
    text_embeddings = np.asarray(
        active_adapter.encode_text([str(row["query_text"]) for row in annotations]),
        dtype=np.float32,
    )
    if text_embeddings.shape != (len(annotations), 512):
        raise RuntimeError("MB1-E1 text encoding returned invalid embeddings")
    dataset = config.dataset_root.expanduser().resolve(strict=True)
    video_partitions, keyframe_partitions = discover_layout(dataset)
    results: list[dict[str, Any]] = []
    review_key: list[dict[str, Any]] = []
    started = monotonic()
    try:
        for annotation, text_embedding in zip(annotations, text_embeddings, strict=True):
            candidate = candidates[str(annotation["source_candidate_id"])]
            video_id = str(annotation["video_id"])
            assets = resolve_assets(dataset, video_id, video_partitions, keyframe_partitions)
            decoder = decoder_factory(video_id, assets.video)
            try:
                search, images = refine_inside_candidate_window(
                    decoder=decoder,
                    image_encoder=image_encoder,
                    text_embedding=text_embedding,
                    window_start=int(candidate["window_start_frame"]),
                    window_end=int(candidate["window_end_frame"]),
                    source_anchor_frame=int(candidate["source_anchor_frame"]),
                )
            finally:
                decoder.close()
            row = build_moment_result(annotation, candidate, search)
            m0_frame, m1_frame = int(row["m0_frame"]), int(row["m1_frame"])
            results.append(row)
            if render_visuals and m0_frame != m1_frame:
                relative = Path("visuals") / f"{annotation['moment_id']}_ab.jpg"
                review_key.append(
                    {
                        **render_blinded_sheet(
                            output / relative,
                            moment_id=str(annotation["moment_id"]),
                            query_text=str(annotation["query_text"]),
                            m0_frame=m0_frame,
                            m1_frame=m1_frame,
                            images=images,
                            seed=config.seed,
                        ),
                        "visual_path": relative.as_posix(),
                    }
                )
    finally:
        if owned_adapter and hasattr(active_adapter, "close"):
            active_adapter.close()

    metrics = build_mb1_e1_metrics(results)
    issues: list[dict[str, Any]] = []
    for name, value in metrics["BY_MOMENT_TYPE"].items():
        if value.get("small_slice_warning"):
            issues.append(
                {
                    "severity": "WARNING",
                    "code": "SMALL_MOMENT_TYPE_SLICE",
                    "message": "Do not over-interpret this moment-type slice.",
                    "evidence": {"moment_type": name, "event_count": value["event_count"]},
                }
            )
    summary = {
        "experiment": "MB1-E1",
        "version": MB1_E1_VERSION,
        "MB1_E1_IMPLEMENTATION_STATUS": "COMPLETE",
        "MB1_E1_REAL_STATUS": "COMPLETE",
        "INTERVAL_BENCHMARK_STATUS": "VALID",
        "M1_INTERVAL_QUALITY_DECISION": "NOT_EVALUATED",
        "decision_note": (
            "Metrics are complete, but the AI interval benchmark remains human_reviewed=false."
        ),
        "annotation_count": len(results),
        "visual_count": len(review_key),
        "primary_decision_slice": metrics["ALL_HIGH_MEDIUM"],
    }
    manifest = {
        "experiment": "MB1-E1",
        "version": MB1_E1_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "build_git_commit": config.build_git_commit,
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(config).items()
        },
        "preflight": preflight,
        "annotation_sha256": source_hash,
        "candidate_manifest_sha256": sha256_file(config.candidate_manifest_path),
        "clip_provenance": clip_provenance,
        "m0_policy": "EXACT_SOURCE_ANCHOR_FRAME_FROM_MB1_CANDIDATE_MANIFEST",
        "m1_policy": {
            "method": MB1_E1_METHOD_M1,
            "window": "EXACT_MB1_CANDIDATE_WINDOW_NO_EXPANSION",
            "coarse_stride_frames": FROZEN_M1_SETTINGS.coarse_stride_frames,
            "dense_radius_frames": FROZEN_M1_SETTINGS.dense_radius_frames,
            "ranking": "RAW_COSINE_ARGMAX_EARLIEST_FRAME_TIEBREAK",
            "query_expansion": False,
            "temporal_smoothing": False,
            "fallback_or_gate": False,
        },
        "network_required": False,
        "elapsed_seconds": monotonic() - started,
    }
    write_json(output / "mb1_e1_summary.json", summary)
    write_json(output / "mb1_e1_metrics.json", metrics)
    write_jsonl(output / "moment_results.jsonl", results)
    write_json(output / "run_manifest.json", manifest)
    write_jsonl(output / "issues.jsonl", issues)
    write_json(output / "visuals/review_key.json", review_key)
    return {"summary": summary, "metrics": metrics, "manifest": manifest, "issues": issues}


def create_mb1_e1_bundle(output_root: str | Path, zip_path: str | Path) -> Path:
    output = Path(output_root).expanduser().resolve(strict=True)
    target = Path(zip_path).expanduser().resolve(strict=False)
    members = [output / name for name in BUNDLE_FILES]
    members.extend(sorted((output / "visuals").glob("*_ab.jpg")))
    missing = [str(path.relative_to(output)) for path in members if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"MB1-E1 bundle members missing: {missing}")
    for path in members:
        if path.suffix.lower() in HEAVY_SUFFIXES:
            raise RuntimeError(f"Heavy asset blocked from MB1-E1 bundle: {path}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(target, "w", compression=ZIP_DEFLATED) as archive:
        for path in members:
            archive.write(path, path.relative_to(output).as_posix())
    return target
