"""Bounded M2 moment-type-aware temporal plateau experiment."""

from __future__ import annotations

import hashlib
import shutil
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, median
from time import monotonic
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

from triage_eg.data.stage0_audit.asset_resolver import discover_layout, resolve_assets
from triage_eg.experiments.mb1_e1.metrics import distance_to_interval
from triage_eg.experiments.mb1_e1.runner import (
    FROZEN_M1_SETTINGS,
    _cosine_scores,
    _prepare_verified_clip,
    _selected_contract,
    copy_benchmark_preserving_hash,
    load_interval_benchmark,
    refine_inside_candidate_window,
    sha256_file,
)
from triage_eg.experiments.moment_m1 import (
    OpenCVRawVideoDecoder,
    VerifiedClipLocalImageEncoder,
)
from triage_eg.retrieval.stage1b.adapters.openai_clip_official import (
    resolve_official_asset_paths,
)
from triage_eg.retrieval.stage1b.writers import write_json, write_jsonl

from .plateau import (
    EXTREMUM_TYPE,
    HALF_PROMINENCE,
    LOW_CONTRAST_EPSILON,
    SMOOTHING_SECONDS,
    centered_moving_average,
    select_dense_peak,
    smoothing_width_frames,
    solve_plateau_from_smoothed,
)

M2_VERSION = "0.1.0"
EXPECTED_ANNOTATION_SHA256 = (
    "4d3f47b5fcc727a8eea893e93420c8db51b4301e966eac6d69508f530515cbbe"
)
BENCHMARK_SEMANTICS = "AI_CURATED_INTERNAL_INTERVAL_PSEUDO_GT"
A0_METHOD = "FROZEN_M1_COARSE_TO_FINE"
A1_METHOD = "DENSE_CLIP_PEAK"
A2_METHOD = "MOMENT_TYPE_PLATEAU"
BOUNDARY_LIKE_TYPES = frozenset(
    {
        "FIRST_OCCURRENCE",
        "TRANSITION_ONSET",
        "TRANSITION_OFFSET",
        "CONTACT",
        "SEPARATION",
        "LAST_OCCURRENCE",
    }
)
TOLERANCES = (1, 5, 10, 30)
ALLOWED_DIAGNOSTICS = frozenset(
    {
        "A2_INTERVAL_HIT",
        "A2_IMPROVED_OVER_DENSE_PEAK",
        "A2_TIED_DENSE_PEAK",
        "A2_REGRESSED_FROM_DENSE_PEAK",
        "LOW_CONTRAST_CURVE",
        "PLATEAU_TOUCHES_WINDOW_START",
        "PLATEAU_TOUCHES_WINDOW_END",
        "EXTREMUM_TEMPORAL_SOLVER_NOT_IMPLEMENTED",
    }
)
BUNDLE_FILES = (
    "m2_summary.json",
    "m2_metrics.json",
    "moment_results.jsonl",
    "moment_score_curves.jsonl",
    "run_manifest.json",
    "issues.jsonl",
    "benchmark/mb1_ai_semantic_moments.jsonl",
    "benchmark/mb1_ai_semantic_moments.sha256",
    "benchmark/mb1_candidate_manifest.jsonl",
    "benchmark/mb1_candidate_manifest.sha256",
    "visuals/review_key.json",
)
HEAVY_SUFFIXES = {
    ".pt",
    ".pth",
    ".bin",
    ".npy",
    ".npz",
    ".mp4",
    ".avi",
    ".mkv",
    ".mov",
}


@dataclass(frozen=True)
class M2Config:
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
            raise ValueError("M2 blinded-review seed is frozen at 2026")
        if self.batch_size <= 0:
            raise ValueError("M2 batch_size must be positive")


def score_dense_window(
    *,
    decoder: Any,
    image_encoder: Any,
    text_embedding: np.ndarray,
    window_start: int,
    window_end: int,
) -> tuple[np.ndarray, np.ndarray, dict[int, np.ndarray], dict[str, float | int]]:
    """Decode and score every raw frame exactly once for shared A1/A2 use."""

    if not 0 <= window_start <= window_end < decoder.info.total_frames:
        raise ValueError("M2 candidate window is invalid for the raw video")
    frame_indices = np.arange(window_start, window_end + 1, dtype=np.int64)
    decode_started = monotonic()
    frames = decoder.decode_indices(frame_indices.tolist())
    decode_ms = (monotonic() - decode_started) * 1000
    decoded_ids = [int(frame.actual_frame_idx) for frame in frames]
    if decoded_ids != frame_indices.tolist():
        raise RuntimeError("M2_DENSE_FRAME_DECODE_INCOMPLETE")
    encoding_started = monotonic()
    embeddings = image_encoder.encode(frames)
    scores = _cosine_scores(text_embedding, embeddings)
    encoding_ms = (monotonic() - encoding_started) * 1000
    if scores.shape != frame_indices.shape or not np.isfinite(scores).all():
        raise RuntimeError("M2_DENSE_SCORE_SEQUENCE_INVALID")
    images = {int(frame.actual_frame_idx): frame.image for frame in frames}
    return frame_indices, scores, images, {
        "raw_decode_ms": decode_ms,
        "dense_image_encoding_ms": encoding_ms,
        "dense_frame_count": len(frame_indices),
        "dense_image_encoding_calls": 1,
    }


def _blinded_mapping(moment_id: str, seed: int) -> dict[str, str]:
    digest = hashlib.sha256(f"M2:{seed}:{moment_id}".encode()).digest()
    if digest[0] % 2:
        return {"METHOD_A": A2_METHOD, "METHOD_B": A1_METHOD}
    return {"METHOD_A": A1_METHOD, "METHOD_B": A2_METHOD}


def _font(size: int):
    from PIL import ImageFont

    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def render_blinded_m2_sheet(
    path: Path,
    *,
    moment_id: str,
    query_text: str,
    moment_type: str,
    video_id: str,
    a1_frame: int,
    a2_frame: int,
    images: dict[int, np.ndarray],
    seed: int,
) -> dict[str, Any]:
    """Render A1/A2 without exposing methods, GT, scores, or errors."""

    from PIL import Image, ImageDraw, ImageOps

    mapping = _blinded_mapping(moment_id, seed)
    method_frames = {A1_METHOD: a1_frame, A2_METHOD: a2_frame}
    width, tile_height, header_height, label_height = 640, 360, 122, 48
    sheet = Image.new("RGB", (width * 2, header_height + tile_height + label_height), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((14, 12), query_text, fill="black", font=_font(21))
    draw.text((14, 54), f"moment_type={moment_type}", fill="black", font=_font(18))
    for index, blind_label in enumerate(("METHOD_A", "METHOD_B")):
        frame_idx = method_frames[mapping[blind_label]]
        image = Image.fromarray(np.asarray(images[frame_idx], dtype=np.uint8), mode="RGB")
        fitted = ImageOps.fit(image, (width, tile_height), method=Image.Resampling.LANCZOS)
        x = index * width
        sheet.paste(fitted, (x, header_height))
        draw.text(
            (x + 14, header_height + tile_height + 10),
            f"{blind_label}  video_id={video_id}  actual_frame_idx={frame_idx}",
            fill="black",
            font=_font(18),
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, format="JPEG", quality=90, optimize=True)
    return {
        "moment_id": moment_id,
        "seed": seed,
        "mapping": mapping,
        "frames": {
            side: method_frames[mapping[side]] for side in ("METHOD_A", "METHOD_B")
        },
    }


def _arm_metrics(rows: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    if not rows:
        return {"moment_count": 0, "status": "EMPTY_SLICE"}
    distances = [int(row[f"{arm}_distance_to_interval"]) for row in rows]
    preferred_errors = [int(row[f"{arm}_preferred_frame_error"]) for row in rows]
    return {
        "moment_count": len(rows),
        "INTERVAL_HIT_RATE": mean(value == 0 for value in distances),
        "MEAN_DISTANCE_TO_INTERVAL": mean(distances),
        "MEDIAN_DISTANCE_TO_INTERVAL": median(distances),
        **{
            f"WITHIN_{tolerance}_FRAMES_RATE": mean(
                value <= tolerance for value in distances
            )
            for tolerance in TOLERANCES
        },
        "preferred_frame_MAE": mean(preferred_errors),
        "preferred_frame_metric_role": "SECONDARY_DIAGNOSTIC_ONLY",
    }


def _pairwise(rows: list[dict[str, Any]], baseline: str) -> dict[str, int]:
    wins = sum(
        int(row["a2_distance_to_interval"])
        < int(row[f"{baseline}_distance_to_interval"])
        for row in rows
    )
    losses = sum(
        int(row["a2_distance_to_interval"])
        > int(row[f"{baseline}_distance_to_interval"])
        for row in rows
    )
    ties = len(rows) - wins - losses
    return {
        "A2_WINS": wins,
        f"{baseline.upper()}_WINS": losses,
        "TIES": ties,
        "NEW_HITS": sum(
            bool(row["a2_interval_hit"]) and not bool(row[f"{baseline}_interval_hit"])
            for row in rows
        ),
        "LOST_HITS": sum(
            not bool(row["a2_interval_hit"]) and bool(row[f"{baseline}_interval_hit"])
            for row in rows
        ),
    }


def aggregate_m2_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"moment_count": 0, "status": "EMPTY_SLICE", "small_slice_warning": True}
    return {
        "moment_count": len(rows),
        "A0_FROZEN_M1_COARSE_TO_FINE": _arm_metrics(rows, "a0"),
        "A1_DENSE_CLIP_PEAK": _arm_metrics(rows, "a1"),
        "A2_MOMENT_TYPE_PLATEAU": _arm_metrics(rows, "a2"),
        "A2_VS_A1": _pairwise(rows, "a1"),
        "A2_VS_A0": _pairwise(rows, "a0"),
        "small_slice_warning": len(rows) < 5,
    }


def build_m2_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Build mandatory interval-first M2 slices and routing-safety guards."""

    action = [row for row in rows if row["moment_type"] == "ACTION_VISIBILITY"]
    slices = {
        "ALL_MOMENTS": rows,
        "HIGH_CONFIDENCE_ONLY": [
            row for row in rows if row["annotation_confidence"] == "HIGH"
        ],
        "ACTION_VISIBILITY": action,
        "BOUNDARY_LIKE": [row for row in rows if row["moment_type"] in BOUNDARY_LIKE_TYPES],
        "EXTREMUM": [row for row in rows if row["moment_type"] == EXTREMUM_TYPE],
    }
    return {
        "benchmark_semantics": BENCHMARK_SEMANTICS,
        "primary_metric_semantics": "DISTANCE_TO_ACCEPTABLE_INTERVAL",
        "preferred_frame_semantics": "SECONDARY_DIAGNOSTIC_ONLY",
        "SLICES": {name: aggregate_m2_metrics(value) for name, value in slices.items()},
        "ROUTING_SAFETY": {
            "ACTION_VISIBILITY_REGRESSION_COUNT": sum(
                int(row["a2_distance_to_interval"]) > int(row["a1_distance_to_interval"])
                for row in action
            ),
            "ACTION_VISIBILITY_LOST_INTERVAL_HITS": sum(
                bool(row["a1_interval_hit"]) and not bool(row["a2_interval_hit"])
                for row in action
            ),
        },
        "DIAGNOSTIC_COUNTS": dict(
            sorted(Counter(code for row in rows for code in row["diagnostics"]).items())
        ),
    }


def _result_row(
    annotation: dict[str, Any],
    candidate: dict[str, Any],
    a0_search: dict[str, Any],
    a1_frame: int,
    a1_score: float,
    solution: Any,
) -> dict[str, Any]:
    start = int(annotation["acceptable_start_frame"])
    end = int(annotation["acceptable_end_frame"])
    preferred = int(annotation["preferred_frame"])
    frames = {
        "a0": int(a0_search["m1_frame"]),
        "a1": int(a1_frame),
        "a2": int(solution.prediction),
    }
    distances = {
        arm: distance_to_interval(frame, start, end) for arm, frame in frames.items()
    }
    diagnostics = list(solution.diagnostics)
    if distances["a2"] == 0:
        diagnostics.append("A2_INTERVAL_HIT")
    diagnostics.append(
        "A2_IMPROVED_OVER_DENSE_PEAK"
        if distances["a2"] < distances["a1"]
        else "A2_REGRESSED_FROM_DENSE_PEAK"
        if distances["a2"] > distances["a1"]
        else "A2_TIED_DENSE_PEAK"
    )
    if not set(diagnostics).issubset(ALLOWED_DIAGNOSTICS):
        raise RuntimeError("M2 emitted a diagnostic outside the frozen taxonomy")
    return {
        **annotation,
        "candidate_window_start": int(candidate["window_start_frame"]),
        "candidate_window_end": int(candidate["window_end_frame"]),
        "candidate_fps": float(candidate["fps"]),
        "source_anchor_frame": int(candidate["source_anchor_frame"]),
        "a0_method": A0_METHOD,
        "a0_frame": frames["a0"],
        "a0_interval_hit": distances["a0"] == 0,
        "a0_distance_to_interval": distances["a0"],
        "a0_preferred_frame_error": abs(frames["a0"] - preferred),
        "a1_method": A1_METHOD,
        "a1_frame": frames["a1"],
        "a1_raw_clip_score": float(a1_score),
        "a1_interval_hit": distances["a1"] == 0,
        "a1_distance_to_interval": distances["a1"],
        "a1_preferred_frame_error": abs(frames["a1"] - preferred),
        "a2_method": A2_METHOD,
        "a2_frame": frames["a2"],
        "a2_interval_hit": distances["a2"] == 0,
        "a2_distance_to_interval": distances["a2"],
        "a2_preferred_frame_error": abs(frames["a2"] - preferred),
        "a0_diagnostics": a0_search,
        "diagnostics": diagnostics,
    }


def preflight_m2(config: M2Config) -> dict[str, Any]:
    annotations, candidates = load_interval_benchmark(
        config.candidate_manifest_path, config.annotation_path
    )
    annotation_hash = sha256_file(config.annotation_path)
    if annotation_hash != EXPECTED_ANNOTATION_SHA256:
        raise RuntimeError("M2_ANNOTATION_SHA256_MISMATCH")
    if config.output_root.exists():
        raise FileExistsError(f"M2 output already exists: {config.output_root}")
    selected = _selected_contract(config.stage1b_root.expanduser().resolve(strict=True))
    clip_paths = resolve_official_asset_paths(config.clip_asset_root)
    if not clip_paths.source_root.is_dir() or not clip_paths.checkpoint_path.is_file():
        raise FileNotFoundError("Offline OpenAI CLIP asset is incomplete")
    dataset = config.dataset_root.expanduser().resolve(strict=True)
    video_partitions, keyframe_partitions = discover_layout(dataset)
    missing_videos = []
    for annotation in annotations:
        candidate = candidates[str(annotation["source_candidate_id"])]
        required = {
            "candidate_id",
            "video_id",
            "window_start_frame",
            "window_end_frame",
            "fps",
            "source_anchor_frame",
        }
        if not required.issubset(candidate):
            raise ValueError(f"M2 candidate identity fields missing: {annotation['moment_id']}")
        if str(annotation["video_id"]) != str(candidate["video_id"]):
            raise RuntimeError(f"M2_CANDIDATE_IDENTITY_MISMATCH: {annotation['moment_id']}")
        assets = resolve_assets(
            dataset, str(annotation["video_id"]), video_partitions, keyframe_partitions
        )
        if not assets.video.is_file():
            missing_videos.append(str(annotation["video_id"]))
    if missing_videos:
        raise FileNotFoundError(f"Missing M2 raw videos: {sorted(set(missing_videos))}")
    return {
        "status": "READY",
        "benchmark_semantics": BENCHMARK_SEMANTICS,
        "annotation_sha256": annotation_hash,
        "annotation_count": len(annotations),
        "human_reviewed": False,
        "encoder_status": selected["compatibility_status"],
        "encoder": "openai_clip_vit_b32_openai_official",
        "window_scope": "EXACT_MB1_CANDIDATE_WINDOW_INCLUSIVE",
        "smoothing_seconds": SMOOTHING_SECONDS,
        "half_prominence": HALF_PROMINENCE,
        "network_required": False,
        "model_download_required": False,
    }


DecoderFactory = Callable[[str, Path], Any]


def run_m2(
    config: M2Config,
    *,
    adapter: Any | None = None,
    decoder_factory: DecoderFactory = OpenCVRawVideoDecoder,
    render_visuals: bool = True,
) -> dict[str, Any]:
    preflight = preflight_m2(config)
    output = config.output_root.expanduser().resolve(strict=False)
    output.mkdir(parents=True)
    annotations, candidates = load_interval_benchmark(
        config.candidate_manifest_path, config.annotation_path
    )
    annotation_copy = output / "benchmark/mb1_ai_semantic_moments.jsonl"
    annotation_hash = copy_benchmark_preserving_hash(config.annotation_path, annotation_copy)
    (output / "benchmark/mb1_ai_semantic_moments.sha256").write_text(
        f"{annotation_hash}  mb1_ai_semantic_moments.jsonl\n", encoding="utf-8"
    )
    candidate_copy = output / "benchmark/mb1_candidate_manifest.jsonl"
    candidate_hash = copy_benchmark_preserving_hash(
        config.candidate_manifest_path, candidate_copy
    )
    (output / "benchmark/mb1_candidate_manifest.sha256").write_text(
        f"{candidate_hash}  mb1_candidate_manifest.jsonl\n", encoding="utf-8"
    )

    owned_adapter = adapter is None
    active_adapter, clip_provenance = (
        _prepare_verified_clip(config)
        if adapter is None
        else (adapter, {"candidate_id": "TEST_ADAPTER", "compatibility_status": "INJECTED"})
    )
    image_encoder = VerifiedClipLocalImageEncoder(active_adapter)
    text_started = monotonic()
    clip_texts = [str(row["query_text"]).strip() for row in annotations]
    if any(not text for text in clip_texts):
        raise ValueError("M2 query text is empty")
    text_embeddings = np.asarray(
        active_adapter.encode_text(clip_texts),
        dtype=np.float32,
    )
    text_encoding_ms = (monotonic() - text_started) * 1000
    if text_embeddings.shape != (20, 512) or not np.isfinite(text_embeddings).all():
        raise RuntimeError("M2 text encoding returned invalid embeddings")

    dataset = config.dataset_root.expanduser().resolve(strict=True)
    video_partitions, keyframe_partitions = discover_layout(dataset)
    results: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    review_key: list[dict[str, Any]] = []
    timings: list[dict[str, Any]] = []
    experiment_started = monotonic()
    try:
        for annotation, text_embedding in zip(annotations, text_embeddings, strict=True):
            moment_started = monotonic()
            candidate = candidates[str(annotation["source_candidate_id"])]
            video_id = str(annotation["video_id"])
            assets = resolve_assets(dataset, video_id, video_partitions, keyframe_partitions)
            decoder = decoder_factory(video_id, assets.video)
            try:
                candidate_fps = float(candidate["fps"])
                if not np.isclose(decoder.info.fps, candidate_fps, rtol=0.0, atol=1e-3):
                    raise RuntimeError(
                        f"M2_CANDIDATE_FPS_MISMATCH: {annotation['moment_id']}"
                    )
                window_start = int(candidate["window_start_frame"])
                window_end = int(candidate["window_end_frame"])
                a0_search, _ = refine_inside_candidate_window(
                    decoder=decoder,
                    image_encoder=image_encoder,
                    text_embedding=text_embedding,
                    window_start=window_start,
                    window_end=window_end,
                    source_anchor_frame=int(candidate["source_anchor_frame"]),
                )
                frames, raw_scores, images, dense_timing = score_dense_window(
                    decoder=decoder,
                    image_encoder=image_encoder,
                    text_embedding=text_embedding,
                    window_start=window_start,
                    window_end=window_end,
                )
                a1_frame, a1_score = select_dense_peak(frames, raw_scores)
                smoothing_started = monotonic()
                width = smoothing_width_frames(candidate_fps)
                smoothed = centered_moving_average(raw_scores, width)
                smoothing_ms = (monotonic() - smoothing_started) * 1000
                plateau_started = monotonic()
                solution = solve_plateau_from_smoothed(
                    frames,
                    raw_scores,
                    smoothed,
                    candidate_fps,
                    str(annotation["moment_type"]),
                    width,
                )
                plateau_ms = (monotonic() - plateau_started) * 1000
                row = _result_row(
                    annotation, candidate, a0_search, a1_frame, a1_score, solution
                )
                results.append(row)
                curves.append(
                    {
                        "moment_id": annotation["moment_id"],
                        "query_text": annotation["query_text"],
                        "moment_type": annotation["moment_type"],
                        "video_id": video_id,
                        "frame_indices": frames.tolist(),
                        "raw_clip_scores": [float(value) for value in raw_scores],
                        "smoothed_clip_scores": list(solution.smoothed_clip_scores),
                        "smoothing_width_frames": solution.smoothing_width_frames,
                        "raw_dense_peak_frame": solution.raw_dense_peak_frame,
                        "smoothed_peak_frame": solution.smoothed_peak_frame,
                        "baseline_score": solution.baseline_score,
                        "peak_score": solution.peak_score,
                        "prominence": solution.prominence,
                        "plateau_threshold": solution.plateau_threshold,
                        "plateau_start_frame": solution.plateau_start_frame,
                        "plateau_end_frame": solution.plateau_end_frame,
                        "plateau_duration_frames": solution.plateau_duration_frames,
                        "plateau_duration_seconds": solution.plateau_duration_seconds,
                        "A0_prediction": row["a0_frame"],
                        "A1_prediction": row["a1_frame"],
                        "A2_prediction": row["a2_frame"],
                    }
                )
                should_review = (
                    a1_frame != solution.prediction
                    or str(annotation["moment_type"]) in BOUNDARY_LIKE_TYPES
                )
                if render_visuals and should_review:
                    relative = Path("visuals") / f"{annotation['moment_id']}_ab.jpg"
                    review_key.append(
                        {
                            **render_blinded_m2_sheet(
                                output / relative,
                                moment_id=str(annotation["moment_id"]),
                                query_text=str(annotation["query_text"]),
                                moment_type=str(annotation["moment_type"]),
                                video_id=video_id,
                                a1_frame=a1_frame,
                                a2_frame=solution.prediction,
                                images=images,
                                seed=config.seed,
                            ),
                            "visual_path": relative.as_posix(),
                        }
                    )
                timings.append(
                    {
                        "moment_id": annotation["moment_id"],
                        **dense_timing,
                        "curve_smoothing_ms": smoothing_ms,
                        "plateau_solving_ms": plateau_ms,
                        "a0_frozen_m1_ms": float(a0_search["elapsed_ms"]),
                        "total_moment_ms": (monotonic() - moment_started) * 1000,
                    }
                )
            finally:
                decoder.close()
    finally:
        if owned_adapter and hasattr(active_adapter, "close"):
            active_adapter.close()

    if sum(bool(row["a0_interval_hit"]) for row in results) != 12:
        raise RuntimeError("M2_A0_REPRODUCTION_FAILED: expected 12/20 interval hits")
    metrics = build_m2_metrics(results)
    diagnostic_counts = metrics["DIAGNOSTIC_COUNTS"]
    issues = [
        {
            "severity": (
                "WARNING"
                if code
                in {
                    "LOW_CONTRAST_CURVE",
                    "A2_REGRESSED_FROM_DENSE_PEAK",
                    "EXTREMUM_TEMPORAL_SOLVER_NOT_IMPLEMENTED",
                }
                else "INFO"
            ),
            "code": code,
            "evidence": {"count": count},
        }
        for code, count in diagnostic_counts.items()
    ]
    summary = {
        "experiment": "M2",
        "version": M2_VERSION,
        "M2_IMPLEMENTATION_STATUS": "COMPLETE",
        "M2_REAL_STATUS": "COMPLETE",
        "M2_DIAGNOSTIC_STATUS": "COMPLETE",
        "M2_QUALITY_DECISION": "NOT_EVALUATED",
        "benchmark_semantics": BENCHMARK_SEMANTICS,
        "annotation_count": len(results),
        "annotation_sha256": annotation_hash,
        "dense_frames_encoded": sum(int(row["dense_frame_count"]) for row in timings),
        "blinded_visual_count": len(review_key),
        "primary_comparison": metrics["SLICES"]["ALL_MOMENTS"],
        "boundary_like": metrics["SLICES"]["BOUNDARY_LIKE"],
        "action_visibility_guard": metrics["ROUTING_SAFETY"],
    }
    manifest = {
        "experiment": "M2",
        "version": M2_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "build_git_commit": config.build_git_commit,
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(config).items()
        },
        "preflight": preflight,
        "annotation_sha256": annotation_hash,
        "candidate_manifest_sha256": candidate_hash,
        "clip_provenance": clip_provenance,
        "arms": [A0_METHOD, A1_METHOD, A2_METHOD],
        "a0_reuses": "mb1_e1.refine_inside_candidate_window",
        "a0_frozen_m1_settings": asdict(FROZEN_M1_SETTINGS),
        "text_embedding_batches": 1,
        "text_embedding_count": len(text_embeddings),
        "text_embedding_ms": text_encoding_ms,
        "language_path": "FROZEN_STAGE2A_ENGLISH_OPENAI_CLIP_NO_TRANSLATION",
        "dense_scores_computed_once_per_moment_and_shared_by_a1_a2": True,
        "smoothing_seconds": SMOOTHING_SECONDS,
        "half_prominence": HALF_PROMINENCE,
        "low_contrast_epsilon": LOW_CONTRAST_EPSILON,
        "gt_used_for_prediction": False,
        "preferred_frame_role": "SECONDARY_DIAGNOSTIC_ONLY",
        "raw_frames_added_to_stage1_or_framemap": False,
        "network_required": False,
        "model_download_required": False,
        "timings": {
            "moments": timings,
            "experiment_total_ms": (monotonic() - experiment_started) * 1000,
        },
    }
    write_json(output / "m2_summary.json", summary)
    write_json(output / "m2_metrics.json", metrics)
    write_jsonl(output / "moment_results.jsonl", results)
    write_jsonl(output / "moment_score_curves.jsonl", curves)
    write_json(output / "run_manifest.json", manifest)
    write_jsonl(output / "issues.jsonl", issues)
    write_json(
        output / "visuals/review_key.json",
        {"seed": config.seed, "comparison": "A1_VS_A2", "moments": review_key},
    )
    return {"summary": summary, "metrics": metrics, "manifest": manifest}


def create_m2_bundle(output_root: str | Path, zip_path: str | Path) -> Path:
    source = Path(output_root).expanduser().resolve(strict=True)
    target = Path(zip_path).expanduser().resolve(strict=False)
    if source in target.parents:
        raise ValueError("M2 ZIP must be outside output root")
    members = [source / name for name in BUNDLE_FILES]
    members.extend(sorted((source / "visuals").glob("*_ab.jpg")))
    missing = [str(path.relative_to(source)) for path in members if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"M2 bundle members missing: {missing}")
    if any(path.suffix.lower() in HEAVY_SUFFIXES for path in members):
        raise RuntimeError("M2 bundle contains a forbidden heavy artifact")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.with_suffix(target.suffix + ".building")
    staging.unlink(missing_ok=True)
    try:
        with ZipFile(staging, "w", compression=ZIP_DEFLATED) as archive:
            for path in members:
                archive.write(path, path.relative_to(source).as_posix())
        shutil.move(staging, target)
    finally:
        staging.unlink(missing_ok=True)
    return target


__all__ = [
    "A0_METHOD",
    "A1_METHOD",
    "A2_METHOD",
    "BENCHMARK_SEMANTICS",
    "BOUNDARY_LIKE_TYPES",
    "EXPECTED_ANNOTATION_SHA256",
    "M2Config",
    "M2_VERSION",
    "aggregate_m2_metrics",
    "build_m2_metrics",
    "create_m2_bundle",
    "preflight_m2",
    "render_blinded_m2_sheet",
    "run_m2",
    "score_dense_window",
]
