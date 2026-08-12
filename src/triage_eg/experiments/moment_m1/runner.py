"""M1 local raw-video coarse-to-fine semantic frame refinement."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from time import monotonic
from typing import Any, Protocol
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import yaml

from triage_eg.data.stage0_audit.asset_resolver import discover_layout, resolve_assets
from triage_eg.experiments.reference_rt1.dante import dante_monotonic_dp
from triage_eg.experiments.reference_rt1.scoring import build_video_row_groups
from triage_eg.experiments.reference_rt2 import (
    BENCHMARK_TYPE,
    RT2BenchmarkQuery,
    resolve_benchmark_identities,
)
from triage_eg.retrieval.stage1b.writers import write_json, write_jsonl
from triage_eg.retrieval.stage2 import (
    OperationalRetrievalRuntime,
    QueryRequest,
    Stage2RuntimeConfig,
)
from triage_eg.video import (
    DecodedFrame,
    OpenCVRawVideoDecoder,
    RawVideoDecoder,
    VideoInfo,
)

from .metrics import build_m1_metrics, failure_diagnostics
from .visuals import (
    M0_METHOD,
    M1_METHOD,
    blinded_mapping,
    render_blinded_event_sheet,
    render_debug_strip,
)

M1_VERSION = "0.1.0"
REFERENCE_SOLVER = "ORDER_ONLY_MONOTONIC_DP"
FORBIDDEN_BUNDLE_SUFFIXES = {".pt", ".pth", ".bin", ".npy", ".npz", ".mp4", ".avi", ".mkv", ".mov"}


@dataclass(frozen=True)
class M1Settings:
    seed: int = 2026
    distance_lambda: float = 0.0
    local_window_seconds: float = 6.0
    coarse_stride_frames: int = 12
    dense_radius_frames: int = 15
    debug_strip_min_abs_delta_frames: int = 30

    def __post_init__(self) -> None:
        if self.seed != 2026 or self.distance_lambda != 0.0:
            raise ValueError("M1 freezes seed=2026 and ORDER_ONLY_MONOTONIC_DP lambda=0")
        if self.local_window_seconds != 6.0:
            raise ValueError("M1 local_window_seconds is frozen at 6.0")
        if self.coarse_stride_frames != 12 or self.dense_radius_frames != 15:
            raise ValueError("M1 coarse stride and dense radius are frozen at 12 and 15")
        if self.debug_strip_min_abs_delta_frames < 0:
            raise ValueError("debug strip threshold must be non-negative")


@dataclass(frozen=True)
class M1RunnerConfig:
    stage2: Stage2RuntimeConfig
    dataset_root: Path
    benchmark_path: Path
    output_root: Path
    settings: M1Settings


class LocalImageEncoder(Protocol):
    def encode(self, frames: list[DecodedFrame]) -> np.ndarray: ...


class VerifiedClipLocalImageEncoder:
    """Feed decoded RGB frames through the already-loaded verified CLIP adapter."""

    def __init__(self, adapter: Any) -> None:
        self.adapter = adapter

    def encode(self, frames: list[DecodedFrame]) -> np.ndarray:
        if not frames:
            return np.empty((0, 512), dtype=np.float32)
        encoder = getattr(self.adapter, "encode_rgb_arrays", None)
        if not callable(encoder):
            raise RuntimeError("M1_IN_MEMORY_CLIP_API_UNAVAILABLE")
        matrix = np.asarray(
            encoder([np.asarray(frame.image, dtype=np.uint8) for frame in frames]),
            dtype=np.float32,
        )
        if matrix.shape != (len(frames), 512) or not np.isfinite(matrix).all():
            raise RuntimeError("M1 local CLIP image encoding returned invalid embeddings")
        return matrix


def load_m1_settings(path: str | Path) -> M1Settings:
    source = Path(path).expanduser().resolve(strict=True)
    value = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("experiment") != "MOMENT_M1":
        raise ValueError("Invalid M1 experiment configuration")
    if value.get("reference_temporal_solver") != REFERENCE_SOLVER:
        raise ValueError(f"M1 requires {REFERENCE_SOLVER}")
    local = value.get("local_refinement", {})
    return M1Settings(
        seed=int(value.get("seed", 2026)),
        distance_lambda=float(value.get("distance_lambda", 0.0)),
        local_window_seconds=float(local.get("window_seconds", 6.0)),
        coarse_stride_frames=int(local.get("coarse_stride_frames", 12)),
        dense_radius_frames=int(local.get("dense_radius_frames", 15)),
        debug_strip_min_abs_delta_frames=int(local.get("debug_strip_min_abs_delta_frames", 30)),
    )


def clipped_local_window(
    anchor_frame_idx: int, *, fps: float, total_frames: int, seconds: float = 6.0
) -> tuple[int, int]:
    if total_frames <= 0 or not math.isfinite(fps) or fps <= 0:
        raise ValueError("video FPS and total_frames must be valid")
    if not 0 <= anchor_frame_idx < total_frames:
        raise IndexError("anchor frame is outside the video")
    radius = int(round(seconds * fps))
    return max(0, anchor_frame_idx - radius), min(total_frames - 1, anchor_frame_idx + radius)


def coarse_frame_indices(start: int, end: int, anchor: int, stride: int = 12) -> list[int]:
    if stride <= 0 or not start <= anchor <= end:
        raise ValueError("invalid coarse sampling bounds")
    return sorted(set(range(start, end + 1, stride)) | {start, anchor, end})


def dense_frame_indices(start: int, end: int, peak: int, radius: int = 15) -> list[int]:
    if radius < 0 or not start <= peak <= end:
        raise ValueError("invalid dense refinement bounds")
    lower, upper = max(start, peak - radius), min(end, peak + radius)
    return list(range(lower, upper + 1))


def reference_is_reachable(reference_frame_idx: int, start: int, end: int) -> bool:
    return start <= reference_frame_idx <= end


def select_best_frame(frame_indices: list[int], scores: np.ndarray) -> tuple[int, float]:
    values = np.asarray(scores, dtype=np.float32)
    if len(frame_indices) == 0 or values.shape != (len(frame_indices),):
        raise ValueError("frame IDs and scores must be aligned non-empty vectors")
    if not np.isfinite(values).all():
        raise ValueError("local frame scores must be finite")
    winner = min(
        range(len(frame_indices)), key=lambda index: (-float(values[index]), frame_indices[index])
    )
    return int(frame_indices[winner]), float(values[winner])


def _cosine_scores(text_embedding: np.ndarray, image_embeddings: np.ndarray) -> np.ndarray:
    text = np.asarray(text_embedding, dtype=np.float32).reshape(-1)
    images = np.asarray(image_embeddings, dtype=np.float32)
    if text.shape != (512,) or images.ndim != 2 or images.shape[1] != 512:
        raise ValueError("M1 cosine inputs must use 512-dimensional CLIP embeddings")
    text_norm = np.linalg.norm(text)
    image_norms = np.linalg.norm(images, axis=1)
    if text_norm == 0 or np.any(image_norms == 0):
        raise ValueError("M1 cosine inputs must have non-zero norms")
    return ((images @ text) / (image_norms * text_norm)).astype(np.float32, copy=False)


def order_only_source_chain(
    *,
    backend: Any,
    catalog: Any,
    source_rows: np.ndarray,
    event_ids: list[str],
    text_embeddings: np.ndarray,
) -> list[dict[str, Any]]:
    """Run the frozen RT1 DP with lambda=0 over only the known source video."""

    rows = np.asarray(source_rows, dtype=np.int64)
    vectors = np.asarray(backend.vectors_at(rows), dtype=np.float32)
    queries = np.asarray(text_embeddings, dtype=np.float32)
    vector_norms = np.linalg.norm(vectors, axis=1)
    query_norms = np.linalg.norm(queries, axis=1)
    if np.any(vector_norms == 0) or np.any(query_norms == 0):
        raise ValueError("M1 source-video cosine inputs must have non-zero norms")
    local_scores = (queries @ vectors.T) / (query_norms[:, None] * vector_norms[None, :])
    if not np.isfinite(local_scores).all():
        raise ValueError("M1 source-video score matrix must be finite")
    alignment = dante_monotonic_dp(local_scores, distance_lambda=0.0)
    if alignment is None:
        raise RuntimeError("ORDER_ONLY_MONOTONIC_DP has no valid source-video alignment")
    if any(
        left >= right
        for left, right in zip(alignment.positions[:-1], alignment.positions[1:], strict=True)
    ):
        raise RuntimeError("M1 source-video chain is not strictly increasing")
    chain = []
    for event_index, position in enumerate(alignment.positions):
        global_row = int(rows[position])
        mapped = catalog.map_row(global_row)
        chain.append(
            {
                "event_id": event_ids[event_index],
                "catalog_position": int(position),
                "global_row": global_row,
                "n": int(mapped["n"]),
                "original_frame_idx": int(mapped["original_frame_idx"]),
                "coarse_score": float(local_scores[event_index, position]),
            }
        )
    return chain


def refine_local_event(
    *,
    decoder: RawVideoDecoder,
    image_encoder: LocalImageEncoder,
    text_embedding: np.ndarray,
    anchor_frame_idx: int,
    settings: M1Settings,
) -> tuple[dict[str, Any], dict[int, np.ndarray]]:
    event_started = monotonic()
    start, end = clipped_local_window(
        anchor_frame_idx,
        fps=decoder.info.fps,
        total_frames=decoder.info.total_frames,
        seconds=settings.local_window_seconds,
    )
    coarse_ids = coarse_frame_indices(start, end, anchor_frame_idx, settings.coarse_stride_frames)
    coarse_started = monotonic()
    decode_started = monotonic()
    coarse_frames = decoder.decode_indices(coarse_ids)
    coarse_decode_ms = (monotonic() - decode_started) * 1000
    encode_started = monotonic()
    coarse_embeddings = image_encoder.encode(coarse_frames)
    coarse_encode_ms = (monotonic() - encode_started) * 1000
    coarse_scores = _cosine_scores(text_embedding, coarse_embeddings)
    peak_idx, peak_score = select_best_frame(coarse_ids, coarse_scores)
    coarse_search_ms = (monotonic() - coarse_started) * 1000

    dense_started = monotonic()
    dense_ids = dense_frame_indices(start, end, peak_idx, settings.dense_radius_frames)
    decode_started = monotonic()
    dense_frames = decoder.decode_indices(dense_ids)
    dense_decode_ms = (monotonic() - decode_started) * 1000
    encode_started = monotonic()
    dense_embeddings = image_encoder.encode(dense_frames)
    dense_encode_ms = (monotonic() - encode_started) * 1000
    dense_scores = _cosine_scores(text_embedding, dense_embeddings)
    refined_idx, refined_score = select_best_frame(dense_ids, dense_scores)
    dense_refinement_ms = (monotonic() - dense_started) * 1000

    images = {frame.actual_frame_idx: frame.image for frame in coarse_frames + dense_frames}
    return (
        {
            "local_window_start": start,
            "local_window_end": end,
            "coarse_sample_count": len(coarse_ids),
            "coarse_peak_frame_idx": peak_idx,
            "coarse_peak_score": peak_score,
            "dense_candidate_count": len(dense_ids),
            "refined_frame_idx": refined_idx,
            "refined_score": refined_score,
            "timings_ms": {
                "raw_decode_ms": coarse_decode_ms + dense_decode_ms,
                "image_encode_ms": coarse_encode_ms + dense_encode_ms,
                "coarse_local_search_ms": coarse_search_ms,
                "dense_refinement_ms": dense_refinement_ms,
                "total_event_ms": (monotonic() - event_started) * 1000,
            },
        },
        images,
    )


def preflight_moment_m1(config: M1RunnerConfig, queries: list[RT2BenchmarkQuery]) -> dict[str, Any]:
    dataset = config.dataset_root.expanduser().resolve(strict=True)
    benchmark = config.benchmark_path.expanduser().resolve(strict=True)
    if config.output_root.exists():
        raise FileExistsError(f"M1 output already exists: {config.output_root}")
    try:
        import cv2
    except ImportError as error:
        raise ImportError("M1 requires OpenCV (cv2)") from error
    video_partitions, keyframe_partitions = discover_layout(dataset)
    source_ids = sorted({query.source_video_id for query in queries})
    missing = [
        video_id
        for video_id in source_ids
        if not resolve_assets(
            dataset, video_id, video_partitions, keyframe_partitions
        ).video.is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Missing M1 raw source videos: {missing}")
    return {
        "status": "READY",
        "benchmark_path": str(benchmark),
        "benchmark_query_count": len(queries),
        "benchmark_event_count": sum(len(query.events) for query in queries),
        "source_video_count": len(source_ids),
        "raw_video_decoder": "OpenCVRawVideoDecoder",
        "opencv_version": str(cv2.__version__),
        "reference_temporal_solver": REFERENCE_SOLVER,
        "distance_lambda": 0.0,
    }


DecoderFactory = Callable[[str, Path], RawVideoDecoder]


def run_moment_m1(
    config: M1RunnerConfig,
    queries: list[RT2BenchmarkQuery],
    *,
    runtime: OperationalRetrievalRuntime | None = None,
    decoder_factory: DecoderFactory = OpenCVRawVideoDecoder,
    local_image_encoder: LocalImageEncoder | None = None,
    render_visuals: bool = True,
) -> dict[str, Any]:
    output = config.output_root.expanduser().resolve(strict=False)
    dataset = config.dataset_root.expanduser().resolve(strict=True)
    if output.exists():
        existing = {path.name for path in output.iterdir()}
        if runtime is None or existing - {"_stage2_control"}:
            raise FileExistsError(f"M1 output already exists: {output}")
    else:
        output.mkdir(parents=True)
    write_jsonl(output / "benchmark/rt2_ai_benchmark.jsonl", [query.as_dict() for query in queries])
    stage2_config = replace(config.stage2, output_root=output / "_stage2_control")
    active_runtime = runtime or OperationalRetrievalRuntime(stage2_config)
    owns_runtime = runtime is None
    issues: list[dict[str, Any]] = []
    experiment_started = monotonic()
    try:
        active_runtime.load()
        resolve_benchmark_identities(queries, active_runtime.catalog)
        groups = {group.video_id: group for group in build_video_row_groups(active_runtime.catalog)}
        requests = [
            QueryRequest(f"{query.query_id}__{event.event_id}", event.text, query.language, 1)
            for query in queries
            for event in query.events
        ]
        text_encode_started = monotonic()
        encoded = active_runtime.encode_requests(requests)
        text_encode_ms = (monotonic() - text_encode_started) * 1000
        if len(encoded.embeddings) != len(requests):
            raise RuntimeError("M1 text embedding count mismatch")
        for request, encoding in zip(requests, encoded.encodings, strict=True):
            if request.language == "en" and bool(encoding.get("translation_applied")):
                raise RuntimeError("M1 English event unexpectedly invoked translator")
        frame_encoder = local_image_encoder or VerifiedClipLocalImageEncoder(active_runtime.encoder)
        video_partitions, keyframe_partitions = discover_layout(dataset)
        event_results: list[dict[str, Any]] = []
        review_key = []
        embedding_offset = 0
        for query in queries:
            event_count = len(query.events)
            query_embeddings = np.asarray(
                encoded.embeddings[embedding_offset : embedding_offset + event_count],
                dtype=np.float32,
            )
            embedding_offset += event_count
            group = groups.get(query.source_video_id)
            if group is None:
                raise RuntimeError(f"M1 source video missing from Stage1: {query.source_video_id}")
            chain = order_only_source_chain(
                backend=active_runtime.backend,
                catalog=active_runtime.catalog,
                source_rows=group.rows,
                event_ids=[event.event_id for event in query.events],
                text_embeddings=query_embeddings,
            )
            assets = resolve_assets(
                dataset,
                query.source_video_id,
                video_partitions,
                keyframe_partitions,
            )
            decoder = decoder_factory(query.source_video_id, assets.video)
            try:
                for event_index, (event, coarse) in enumerate(
                    zip(query.events, chain, strict=True)
                ):
                    local, images = refine_local_event(
                        decoder=decoder,
                        image_encoder=frame_encoder,
                        text_embedding=query_embeddings[event_index],
                        anchor_frame_idx=int(coarse["original_frame_idx"]),
                        settings=config.settings,
                    )
                    reference = int(event.reference_original_frame_idx)
                    coarse_idx = int(coarse["original_frame_idx"])
                    if not 0 <= reference < decoder.info.total_frames:
                        raise ValueError(
                            f"M1 reference frame is outside raw video: "
                            f"{query.query_id}/{event.event_id} frame={reference}"
                        )
                    refined_idx = int(local["refined_frame_idx"])
                    coarse_error = abs(coarse_idx - reference)
                    refined_error = abs(refined_idx - reference)
                    reachable = reference_is_reachable(
                        reference,
                        int(local["local_window_start"]),
                        int(local["local_window_end"]),
                    )
                    result = {
                        "query_id": query.query_id,
                        "event_id": event.event_id,
                        "source_video_id": query.source_video_id,
                        "event_text": event.text,
                        "reference_frame_idx": reference,
                        "reference_catalog_position": event.reference_catalog_position,
                        "coarse_anchor_catalog_position": int(coarse["catalog_position"]),
                        "coarse_anchor_frame_idx": coarse_idx,
                        "coarse_error_frames": coarse_error,
                        "coarse_score": float(coarse["coarse_score"]),
                        "local_window_start": int(local["local_window_start"]),
                        "local_window_end": int(local["local_window_end"]),
                        "reference_reachable": reachable,
                        "coarse_peak_frame_idx": int(local["coarse_peak_frame_idx"]),
                        "coarse_peak_score": float(local["coarse_peak_score"]),
                        "refined_frame_idx": refined_idx,
                        "refined_score": float(local["refined_score"]),
                        "refined_error_frames": refined_error,
                        "error_delta": coarse_error - refined_error,
                        "diagnostics": failure_diagnostics(
                            reference_reachable=reachable,
                            coarse_error_frames=coarse_error,
                            refined_error_frames=refined_error,
                        ),
                        "coarse_sample_count": int(local["coarse_sample_count"]),
                        "dense_candidate_count": int(local["dense_candidate_count"]),
                        "timings_ms": local["timings_ms"],
                    }
                    event_results.append(result)
                    mapping = blinded_mapping(query.query_id, event.event_id, config.settings.seed)
                    review_key.append(
                        {"query_id": query.query_id, "event_id": event.event_id, **mapping}
                    )
                    if render_visuals:
                        render_blinded_event_sheet(
                            output / "visuals" / f"{query.query_id}_{event.event_id}_ab.jpg",
                            query_id=query.query_id,
                            event_id=event.event_id,
                            event_text=event.text,
                            video_id=query.source_video_id,
                            coarse_frame_idx=coarse_idx,
                            coarse_image=images[coarse_idx],
                            refined_frame_idx=refined_idx,
                            refined_image=images[refined_idx],
                            mapping=mapping,
                        )
                        if (
                            abs(int(result["error_delta"]))
                            >= config.settings.debug_strip_min_abs_delta_frames
                        ):
                            peak_idx = int(local["coarse_peak_frame_idx"])
                            render_debug_strip(
                                output
                                / "diagnostics"
                                / f"{query.query_id}_{event.event_id}_local_strip.jpg",
                                video_id=query.source_video_id,
                                frames=[
                                    ("COARSE_ANCHOR", coarse_idx, images[coarse_idx]),
                                    ("COARSE_LOCAL_PEAK", peak_idx, images[peak_idx]),
                                    ("REFINED", refined_idx, images[refined_idx]),
                                ],
                            )
            finally:
                decoder.close()
        if embedding_offset != len(encoded.embeddings):
            raise RuntimeError("M1 did not consume every event text embedding exactly once")
        metrics = {
            "BENCHMARK_TYPE": BENCHMARK_TYPE,
            "HUMAN_REVIEW_STATUS": "NOT_PERFORMED",
            "OFFICIAL_GT_STATUS": "NOT_AVAILABLE",
            "reference_temporal_solver": REFERENCE_SOLVER,
            "distance_lambda": 0.0,
            **build_m1_metrics(event_results),
        }
        reachable_count = sum(bool(item["reference_reachable"]) for item in event_results)
        summary = {
            "experiment": "MOMENT_M1",
            "status": "COMPLETE",
            "benchmark_query_count": len(queries),
            "benchmark_event_count": len(event_results),
            "reference_reachable_event_count": reachable_count,
            "reference_reachable_event_rate": reachable_count / len(event_results),
            "BENCHMARK_TYPE": BENCHMARK_TYPE,
            "HUMAN_REVIEW_STATUS": "NOT_PERFORMED",
            "OFFICIAL_GT_STATUS": "NOT_AVAILABLE",
            "M1_REAL_EXPERIMENT_STATUS": "COMPLETE",
            "M1_QUALITY_DECISION": "NOT_EVALUATED",
            "issues": len(issues),
        }
        timings = [item["timings_ms"] for item in event_results]
        manifest = {
            **summary,
            "m1_version": M1_VERSION,
            "completed_at": datetime.now(UTC).isoformat(),
            "build_git_commit": config.stage2.build_git_commit,
            "stage1_index_fingerprint": active_runtime.preflight["stage1_index_fingerprint"],
            "stage2a_runtime_manifest": active_runtime.runtime_manifest(),
            "settings": asdict(config.settings),
            "arms": [M0_METHOD, M1_METHOD],
            "reference_temporal_solver": REFERENCE_SOLVER,
            "distance_lambda": 0.0,
            "source_video_only_evaluation": True,
            "text_embedding_batches": 1,
            "text_embedding_count": len(requests),
            "text_embeddings_reused_for_coarse_and_local": True,
            "verified_clip_adapter_reused_from_stage2a": True,
            "raw_frames_are_query_local": True,
            "raw_frames_added_to_global_index": False,
            "network_required": False,
            "timings": {
                "text_encode_ms": text_encode_ms,
                "event_count": len(timings),
                "mean_total_event_ms": mean(float(item["total_event_ms"]) for item in timings),
                "events": [
                    {
                        "query_id": result["query_id"],
                        "event_id": result["event_id"],
                        **result["timings_ms"],
                    }
                    for result in event_results
                ],
                "experiment_total_ms": (monotonic() - experiment_started) * 1000,
            },
        }
        write_json(output / "m1_summary.json", summary)
        write_json(output / "m1_metrics.json", metrics)
        write_jsonl(output / "event_results.jsonl", event_results)
        write_jsonl(output / "issues.jsonl", issues)
        write_json(output / "run_manifest.json", manifest)
        write_json(
            output / "visuals/review_key.json",
            {"seed": config.settings.seed, "events": review_key},
        )
        return summary
    finally:
        if owns_runtime:
            active_runtime.close()


def create_m1_bundle(root: str | Path, zip_path: str | Path) -> Path:
    source = Path(root).expanduser().resolve(strict=True)
    destination = Path(zip_path).expanduser().resolve(strict=False)
    required = {
        "m1_summary.json",
        "m1_metrics.json",
        "event_results.jsonl",
        "issues.jsonl",
        "run_manifest.json",
        "visuals/review_key.json",
    }
    missing = [name for name in sorted(required) if not (source / name).is_file()]
    if missing:
        raise FileNotFoundError(f"M1 bundle missing required files: {missing}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    allowed_roots = {"visuals", "diagnostics"}
    with ZipFile(destination, "w", compression=ZIP_DEFLATED) as stream:
        for path in sorted(source.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(source)
            if (
                relative.parts[0] == "_stage2_control"
                or path.suffix.lower() in FORBIDDEN_BUNDLE_SUFFIXES
            ):
                continue
            if len(relative.parts) > 1 and relative.parts[0] not in allowed_roots:
                continue
            stream.write(path, relative.as_posix())
    return destination


__all__ = [
    "DecodedFrame",
    "M1RunnerConfig",
    "M1Settings",
    "M1_VERSION",
    "OpenCVRawVideoDecoder",
    "REFERENCE_SOLVER",
    "VerifiedClipLocalImageEncoder",
    "VideoInfo",
    "clipped_local_window",
    "coarse_frame_indices",
    "create_m1_bundle",
    "dense_frame_indices",
    "load_m1_settings",
    "order_only_source_chain",
    "preflight_moment_m1",
    "reference_is_reachable",
    "refine_local_event",
    "run_moment_m1",
    "select_best_frame",
]
