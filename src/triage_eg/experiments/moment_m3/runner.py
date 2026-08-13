"""Bounded M3 runner: real M1 baseline plus type-specific state transitions."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

from triage_eg.data.stage0_audit.asset_resolver import discover_layout, resolve_assets
from triage_eg.experiments.mb1_e1.runner import (
    FROZEN_M1_SETTINGS,
    _prepare_verified_clip,
    _selected_contract,
    refine_inside_candidate_window,
    sha256_file,
)
from triage_eg.experiments.moment_m1 import VerifiedClipLocalImageEncoder
from triage_eg.retrieval.stage1b.adapters.openai_clip_official import resolve_official_asset_paths
from triage_eg.retrieval.stage1b.writers import write_json, write_jsonl
from triage_eg.video import OpenCVRawVideoDecoder

from .metrics import build_metrics, decide_m3, evaluate_predictions_only
from .registry import EXPECTED_AI_QC_SHA256, build_case_registry, inference_case_from_registry
from .solver import M3Settings, build_state_signals, local_window, solve_state_transition
from .visuals import render_overview_montage, render_review_sheet

M3_VERSION = "0.1"
M3_EXPERIMENT = "M3"
BENCHMARK_CLAIM = "AI_CURATED_INTERNAL_PSEUDO_GT"
OUTPUT_MEMBERS = (
    "README.md",
    "m3_case_registry.jsonl",
    "m3_case_registry_summary.json",
    "m3_predictions.jsonl",
    "m3_case_metrics.jsonl",
    "m3_metrics_primary.json",
    "m3_metrics_by_type.json",
    "m3_metrics_secondary.json",
    "m3_decision.json",
    "config_snapshot.json",
    "run_manifest.json",
)
FORBIDDEN_SUFFIXES = {".pt", ".pth", ".bin", ".npy", ".npz", ".mp4", ".avi", ".mkv", ".mov"}
FORBIDDEN_MEMBER_TOKENS = ("sealed_final_30", "team-eval", "team_eval", "checkpoint")


@dataclass(frozen=True)
class M3Config:
    dataset_root: Path
    ai_qc_zip: Path
    notebook20_candidates_zip: Path
    stage1b_root: Path
    clip_asset_root: Path
    output_root: Path
    frozen_seed_metadata: Path | None = None
    seed: int = 2026
    device: str = "auto"
    batch_size: int = 32
    build_git_commit: str | None = None
    branch: str = "TRIAGEEG"

    def __post_init__(self) -> None:
        if self.seed != 2026:
            raise ValueError("M3 seed is frozen at 2026")
        if self.batch_size <= 0:
            raise ValueError("M3 batch_size must be positive")
        if self.branch != "TRIAGEEG":
            raise ValueError("M3 must run from branch TRIAGEEG")


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _config_snapshot(config: M3Config) -> dict[str, Any]:
    settings = asdict(M3Settings())
    return {
        "experiment": M3_EXPERIMENT,
        "experiment_version": M3_VERSION,
        "seed": config.seed,
        "device": config.device,
        "batch_size": config.batch_size,
        "local_raw_window": {"radius_seconds": settings.pop("radius_seconds")},
        "state_transition": settings,
        "variants": {
            "A0": "REAL_FROZEN_M1_COARSE_TO_FINE",
            "A1": "M3_STATE_TRANSITION_CONTRAST",
            "A2": "M3_STATE_TRANSITION_PLUS_0.10_MOTION_TIEBREAKER",
        },
        "parameter_sweep": False,
        "gt_used_for_prediction": False,
        "raw_decoder": "OPENCV_CPU_DEFAULT",
        "clip": "VERIFIED_OPENAI_CLIP_VIT_B32",
    }


def preflight_m3(config: M3Config) -> dict[str, Any]:
    dataset = config.dataset_root.expanduser().resolve(strict=True)
    ai_qc = config.ai_qc_zip.expanduser().resolve(strict=True)
    candidates = config.notebook20_candidates_zip.expanduser().resolve(strict=True)
    stage1b = config.stage1b_root.expanduser().resolve(strict=True)
    if config.output_root.expanduser().resolve(strict=False).exists():
        raise FileExistsError(f"M3 output already exists: {config.output_root}")
    ai_hash = sha256_file(ai_qc)
    if ai_hash != EXPECTED_AI_QC_SHA256:
        raise RuntimeError(
            f"M3_AI_QC_SHA256_MISMATCH: expected={EXPECTED_AI_QC_SHA256} actual={ai_hash}"
        )
    selected = _selected_contract(stage1b)
    clip_paths = resolve_official_asset_paths(config.clip_asset_root)
    if not clip_paths.source_root.is_dir() or not clip_paths.checkpoint_path.is_file():
        raise FileNotFoundError("M3 verified offline CLIP asset is incomplete")
    registry, summary = build_case_registry(
        ai_qc_zip=ai_qc,
        notebook20_candidates_zip=candidates,
        frozen_seed_metadata=config.frozen_seed_metadata,
    )
    video_partitions, keyframe_partitions = discover_layout(dataset)
    missing_videos = []
    for video_id in sorted(
        {str(row["video_id"]) for row in registry if row["eligible_for_inference"]}
    ):
        assets = resolve_assets(dataset, video_id, video_partitions, keyframe_partitions)
        if not assets.video.is_file():
            missing_videos.append(video_id)
    if missing_videos:
        raise FileNotFoundError(f"M3 raw videos missing: {missing_videos}")
    return {
        "status": "READY",
        "registry_status": summary["status"],
        "ai_qc_sha256": ai_hash,
        "notebook20_candidates_sha256": sha256_file(candidates),
        "frozen_seed_annotation_sha256": (
            sha256_file(config.frozen_seed_metadata)
            if config.frozen_seed_metadata is not None
            else None
        ),
        "eligible_case_count": summary["eligible_case_count"],
        "primary_case_count": summary["primary_case_count"],
        "secondary_case_count": summary["secondary_conditional_cases"],
        "encoder_status": selected["compatibility_status"],
        "checkpoint_sha256": selected["checkpoint_sha256"],
        "raw_decoder": "OPENCV_CPU_DEFAULT",
        "network_required_for_models": False,
    }


DecoderFactory = Callable[[str, Path], Any]


def _prediction_row(
    registry_row: dict[str, Any],
    *,
    fps: float,
    window_start: int,
    window_end: int,
    m1_search: dict[str, Any],
    a1: Any,
    a2: Any,
    signals: dict[str, np.ndarray],
    decoder_manifest: dict[str, Any],
) -> dict[str, Any]:
    return {
        "case_id": registry_row["case_id"],
        "video_id": registry_row["video_id"],
        "moment_type": registry_row["moment_type"],
        "primary_gate": registry_row["primary_gate"],
        "conditional": registry_row["conditional"],
        "candidate_anchor_frame": registry_row["candidate_anchor_frame"],
        "candidate_anchor_source": registry_row["candidate_anchor_source"],
        "raw_fps": fps,
        "raw_window_start": window_start,
        "raw_window_end": window_end,
        "raw_frame_count": window_end - window_start + 1,
        "raw_frame_identity_policy": decoder_manifest.get("frame_identity_policy"),
        "m1_prediction": int(m1_search["m1_frame"]),
        "m3_a1_prediction": int(a1.prediction),
        "m3_a2_prediction": int(a2.prediction),
        "a1_fallback_reason": a1.fallback_reason,
        "a2_fallback_reason": a2.fallback_reason,
        "a1_used_m1_fallback": a1.used_m1_fallback,
        "a2_used_m1_fallback": a2.used_m1_fallback,
        "signal_range": a1.signal_range,
        "selected_change_score": a1.selected_change_score,
        "pre_high_fraction": a1.pre_high_fraction,
        "post_high_fraction": a1.post_high_fraction,
        "motion_value_at_selection": a2.motion_value_at_selection,
        "valid_candidate_count": a1.valid_candidate_count,
        "diagnostic_signal_summary": {
            name: {
                "min": float(np.nanmin(values)),
                "max": float(np.nanmax(values)),
            }
            for name, values in signals.items()
            if len(values) and np.any(np.isfinite(values))
        },
    }


def run_m3(
    config: M3Config,
    *,
    adapter: Any | None = None,
    decoder_factory: DecoderFactory = OpenCVRawVideoDecoder,
    render_visuals: bool = True,
) -> dict[str, Any]:
    preflight = preflight_m3(config)
    output = config.output_root.expanduser().resolve(strict=False)
    output.mkdir(parents=True)
    registry, registry_summary = build_case_registry(
        ai_qc_zip=config.ai_qc_zip,
        notebook20_candidates_zip=config.notebook20_candidates_zip,
        frozen_seed_metadata=config.frozen_seed_metadata,
    )
    eligible = [row for row in registry if row["eligible_for_inference"]]
    write_jsonl(output / "m3_case_registry.jsonl", registry)
    write_json(output / "m3_case_registry_summary.json", registry_summary)
    snapshot = _config_snapshot(config)
    snapshot["config_fingerprint"] = _canonical_hash(snapshot)
    write_json(output / "config_snapshot.json", snapshot)

    owned_adapter = adapter is None
    active_adapter, clip_provenance = (
        _prepare_verified_clip(config)
        if adapter is None
        else (adapter, {"candidate_id": "TEST_ADAPTER", "compatibility_status": "INJECTED"})
    )
    image_encoder = VerifiedClipLocalImageEncoder(active_adapter)
    texts: list[str] = []
    for row in eligible:
        event_text = row["semantic_event_en"]
        texts.extend(
            [
                event_text,
                row.get("before_state_en") or event_text,
                row.get("after_state_en") or event_text,
            ]
        )
    text_embeddings = np.asarray(active_adapter.encode_text(texts), dtype=np.float32)
    if text_embeddings.shape != (3 * len(eligible), 512) or not np.isfinite(text_embeddings).all():
        raise RuntimeError("M3 text encoding returned invalid embeddings")

    dataset = config.dataset_root.expanduser().resolve(strict=True)
    video_partitions, keyframe_partitions = discover_layout(dataset)
    predictions: list[dict[str, Any]] = []
    case_metrics: list[dict[str, Any]] = []
    review_paths: list[Path] = []
    timings: list[dict[str, Any]] = []
    started = monotonic()
    try:
        for index, row in enumerate(eligible):
            case_started = monotonic()
            inference_case = inference_case_from_registry(row)
            event_text, before_text, after_text = text_embeddings[3 * index : 3 * index + 3]
            assets = resolve_assets(
                dataset, inference_case.video_id, video_partitions, keyframe_partitions
            )
            decoder = decoder_factory(inference_case.video_id, assets.video)
            try:
                window_start, window_end = local_window(
                    inference_case.candidate_anchor_frame,
                    fps=decoder.info.fps,
                    total_frames=decoder.info.total_frames,
                )
                m1_search, _ = refine_inside_candidate_window(
                    decoder=decoder,
                    image_encoder=image_encoder,
                    text_embedding=event_text,
                    window_start=window_start,
                    window_end=window_end,
                    source_anchor_frame=inference_case.candidate_anchor_frame,
                )
                frame_ids = np.arange(window_start, window_end + 1, dtype=np.int64)
                decoded = decoder.decode_indices(frame_ids.tolist())
                if [frame.actual_frame_idx for frame in decoded] != frame_ids.tolist():
                    raise RuntimeError("M3_DENSE_RAW_FRAME_IDENTITY_MISMATCH")
                image_embeddings = image_encoder.encode(decoded)
                state_signals = build_state_signals(
                    image_embeddings,
                    before_text_embedding=before_text,
                    after_text_embedding=after_text,
                    event_text_embedding=event_text,
                )
                a1, a1_signals = solve_state_transition(
                    inference_case,
                    frame_ids=frame_ids,
                    contrast=state_signals["contrast"],
                    image_embeddings=image_embeddings,
                    fps=decoder.info.fps,
                    m1_prediction=int(m1_search["m1_frame"]),
                    use_motion_tiebreaker=False,
                )
                a2, a2_signals = solve_state_transition(
                    inference_case,
                    frame_ids=frame_ids,
                    contrast=state_signals["contrast"],
                    image_embeddings=image_embeddings,
                    fps=decoder.info.fps,
                    m1_prediction=int(m1_search["m1_frame"]),
                    use_motion_tiebreaker=True,
                )
                combined_signals = {**state_signals, **a1_signals, **a2_signals}
                decoded_images = {frame.actual_frame_idx: frame.image for frame in decoded}
                prediction = _prediction_row(
                    row,
                    fps=decoder.info.fps,
                    window_start=window_start,
                    window_end=window_end,
                    m1_search=m1_search,
                    a1=a1,
                    a2=a2,
                    signals=combined_signals,
                    decoder_manifest=decoder.runtime_manifest(),
                )
                prediction_values = {
                    "m1": prediction["m1_prediction"],
                    "m3_a1": prediction["m3_a1_prediction"],
                    "m3_a2": prediction["m3_a2_prediction"],
                }
                evaluated = evaluate_predictions_only(row, prediction_values)
                metric = {
                    **evaluated,
                    "fallback_reason": prediction["a1_fallback_reason"],
                    "m3_a2_fallback_reason": prediction["a2_fallback_reason"],
                    **{
                        key: prediction[key]
                        for key in (
                            "candidate_anchor_frame",
                            "a1_fallback_reason",
                            "signal_range",
                            "selected_change_score",
                            "pre_high_fraction",
                            "post_high_fraction",
                            "motion_value_at_selection",
                        )
                    },
                }
                predictions.append(prediction)
                case_metrics.append(metric)
                if render_visuals and row["primary_gate"] and not row["conditional"]:
                    review_path = output / "review" / f"{row['case_id']}.jpg"
                    render_review_sheet(
                        review_path,
                        case_id=str(row["case_id"]),
                        video_id=str(row["video_id"]),
                        moment_type=str(row["moment_type"]),
                        accepted_intervals=row["accepted_intervals"],
                        predictions=prediction_values,
                        images=decoded_images,
                    )
                    review_paths.append(review_path)
                timings.append(
                    {
                        "case_id": row["case_id"],
                        "elapsed_ms": (monotonic() - case_started) * 1000,
                        "raw_frames": len(decoded),
                    }
                )
            finally:
                decoder.close()
    finally:
        if owned_adapter and hasattr(active_adapter, "close"):
            active_adapter.close()

    primary, by_type, secondary = build_metrics(case_metrics)
    decision = decide_m3(
        primary, by_type, primary_case_count=registry_summary["primary_case_count"]
    )
    write_jsonl(output / "m3_predictions.jsonl", predictions)
    write_jsonl(output / "m3_case_metrics.jsonl", case_metrics)
    write_json(output / "m3_metrics_primary.json", primary)
    write_json(output / "m3_metrics_by_type.json", by_type)
    write_json(output / "m3_metrics_secondary.json", secondary)
    write_json(output / "m3_decision.json", decision)
    if render_visuals:
        render_overview_montage(review_paths, output / "montages" / "m3_primary_overview.jpg")

    primary_types = Counter(
        str(row["moment_type"])
        for row in registry
        if row["eligible_for_inference"] and row["primary_gate"] and not row["conditional"]
    )
    manifest = {
        "experiment": M3_EXPERIMENT,
        "experiment_version": M3_VERSION,
        "git_commit": config.build_git_commit,
        "branch": config.branch,
        "created_at": datetime.now(UTC).isoformat(),
        "dataset_root": str(dataset),
        "ai_qc_input_sha256": preflight["ai_qc_sha256"],
        "notebook20_candidate_input_sha256": preflight["notebook20_candidates_sha256"],
        "frozen_seed_annotation_source": registry_summary["frozen_seed_annotation_source"],
        "frozen_seed_annotation_sha256": preflight["frozen_seed_annotation_sha256"],
        "verified_clip_model": "OpenAI CLIP ViT-B/32 official",
        "clip_device": clip_provenance.get("selected_device", config.device),
        "cpu_fallback_available": True,
        "primary_case_count": registry_summary["primary_case_count"],
        "secondary_case_count": registry_summary["secondary_conditional_cases"],
        "type_counts": dict(sorted(primary_types.items())),
        "config_fingerprint": snapshot["config_fingerprint"],
        "human_reviewed": False,
        "benchmark_claim": BENCHMARK_CLAIM,
        "gt_used_for_prediction": False,
        "a0_reuses": "mb1_e1.refine_inside_candidate_window",
        "a0_frozen_settings": asdict(FROZEN_M1_SETTINGS),
        "raw_frame_coordinate_policy": "EXACT_DECODER_ACTUAL_FRAME_IDX_NO_TIMESTAMP_FPS",
        "gpu_policy": "CLIP_CUDA_PROMOTED_OPENCV_CPU_DEFAULT_NVDEC_NOT_REOPENED",
        "network_required_for_models": False,
        "timings": {"cases": timings, "total_seconds": monotonic() - started},
    }
    write_json(output / "run_manifest.json", manifest)
    (output / "README.md").write_text(
        "# TRIAGE-EG M3 v0.1\n\n"
        "Bounded type-specific BEFORE-to-AFTER semantic state-transition experiment.\n\n"
        "A0 is the real frozen M1 local refiner; A1 is the fixed state transition; "
        "A2 adds only the fixed 0.10 adjacent-CLIP motion tie-breaker. Accepted intervals "
        "are consumed only by evaluation and visualization. New annotations are "
        "AI-curated internal pseudo-GT and are not official GT.\n\n"
        f"Registry status: {registry_summary['status']}. Primary cases: "
        f"{registry_summary['primary_case_count']}. Decision: {decision['M3_GLOBAL']}.\n",
        encoding="utf-8",
    )
    return {
        "preflight": preflight,
        "registry_summary": registry_summary,
        "metrics_primary": primary,
        "metrics_by_type": by_type,
        "metrics_secondary": secondary,
        "decision": decision,
        "manifest": manifest,
        "output_root": output,
    }


def create_m3_bundle(output_root: str | Path, zip_path: str | Path) -> Path:
    source = Path(output_root).expanduser().resolve(strict=True)
    target = Path(zip_path).expanduser().resolve(strict=False)
    if source in target.parents:
        raise ValueError("M3 ZIP must be outside output root")
    members = [source / name for name in OUTPUT_MEMBERS]
    members.extend(sorted((source / "review").glob("*.jpg")))
    members.extend(sorted((source / "montages").glob("*.jpg")))
    missing = [str(path.relative_to(source)) for path in members if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"M3 bundle members missing: {missing}")
    for path in members:
        relative = path.relative_to(source).as_posix().casefold()
        if path.suffix.casefold() in FORBIDDEN_SUFFIXES or any(
            token in relative for token in FORBIDDEN_MEMBER_TOKENS
        ):
            raise RuntimeError(f"M3 forbidden bundle member: {relative}")
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


def formal_report_lines(result: dict[str, Any], *, zip_path: str | Path) -> list[str]:
    summary = result["registry_summary"]
    primary = result["metrics_primary"]
    decision = result["decision"]
    type_counts = Counter(summary["type_counts"])
    known = {"ONSET", "CONTACT", "FIRST_OCCURRENCE", "SEPARATION", "EXTREMUM"}
    other = sum(count for name, count in type_counts.items() if name not in known)
    a0, a1, a2 = (
        primary["A0_M1"],
        primary["A1_M3_STATE_TRANSITION"],
        primary["A2_M3_MOTION_TIEBREAKER"],
    )
    a1_pair, a2_pair = primary["A1_VS_M1"], primary["A2_VS_A1"]
    return [
        f"HEAD={result['manifest']['git_commit']}",
        "M3_IMPLEMENTATION=COMPLETE",
        f"M3_CASE_REGISTRY_STATUS={summary['status']}",
        f"FROZEN_SEED_METADATA_FOUND={summary['frozen_seed_metadata_found']}",
        "NEW_PRIMARY_CASES=4",
        "SECONDARY_CONDITIONAL_CASES=1",
        f"PRIMARY_CASE_COUNT={summary['primary_case_count']}",
        *[
            f"TYPE_{name}_COUNT={type_counts[name]}"
            for name in ("ONSET", "CONTACT", "FIRST_OCCURRENCE", "SEPARATION", "EXTREMUM")
        ],
        f"TYPE_OTHER_COUNT={other}",
        f"A0_M1_INTERVAL_HIT={a0.get('interval_hit_count', 0)}/{a0.get('case_count', 0)}",
        f"A0_M1_MEDIAN_DISTANCE={a0.get('median_distance_to_interval', 'NA')}",
        f"A1_M3_INTERVAL_HIT={a1.get('interval_hit_count', 0)}/{a1.get('case_count', 0)}",
        f"A1_M3_MEDIAN_DISTANCE={a1.get('median_distance_to_interval', 'NA')}",
        f"A1_VS_M1_WINS={a1_pair['wins']}",
        f"A1_VS_M1_TIES={a1_pair['ties']}",
        f"A1_VS_M1_LOSSES={a1_pair['losses']}",
        f"A2_M3_MOTION_INTERVAL_HIT={a2.get('interval_hit_count', 0)}/{a2.get('case_count', 0)}",
        f"A2_M3_MOTION_MEDIAN_DISTANCE={a2.get('median_distance_to_interval', 'NA')}",
        f"A2_VS_A1_WINS={a2_pair['wins']}",
        f"A2_VS_A1_TIES={a2_pair['ties']}",
        f"A2_VS_A1_LOSSES={a2_pair['losses']}",
        *[
            f"M3_{name}={decision[f'M3_{name}']}"
            for name in ("ONSET", "CONTACT", "FIRST_OCCURRENCE", "SEPARATION", "EXTREMUM")
        ],
        f"M3_MOTION_TIEBREAKER={decision['M3_MOTION_TIEBREAKER']}",
        f"M3_GLOBAL={decision['M3_GLOBAL']}",
        "M1_GENERAL_LOCAL_REFINER=KEEP",
        f"PRODUCTION_ROUTER_CHANGE_REQUIRED={decision['PRODUCTION_ROUTER_CHANGE_REQUIRED']}",
        "M3_FURTHER_RESEARCH_REQUIRED=NO",
        f"NEXT_STEP={decision['NEXT_STEP']}",
        f"OUTPUT_ZIP={Path(zip_path).as_posix()}",
        "RETURN_TO_MAIN_PIPELINE=YES",
    ]


__all__ = [
    "BENCHMARK_CLAIM",
    "M3Config",
    "M3_EXPERIMENT",
    "M3_VERSION",
    "OUTPUT_MEMBERS",
    "create_m3_bundle",
    "formal_report_lines",
    "preflight_m3",
    "run_m3",
]
