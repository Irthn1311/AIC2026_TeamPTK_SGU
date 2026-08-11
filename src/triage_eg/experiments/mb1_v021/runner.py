"""MB1 v0.2.1 supplementary boundary-rich candidate mining runner."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

from triage_eg.data.stage0_audit.asset_resolver import discover_layout, resolve_assets
from triage_eg.experiments.mb1_v02 import render_contact_sheet
from triage_eg.experiments.mb1_v02.signals import (
    ShotSegment,
    continuous_shot_segments,
)
from triage_eg.experiments.moment_m1 import OpenCVRawVideoDecoder
from triage_eg.retrieval.stage1b.writers import write_json, write_jsonl

from .signals import (
    COARSE_SAMPLES_PER_SECOND,
    LOCAL_EXPANSION_SECONDS,
    LOCAL_SAMPLES_PER_SECOND,
    MAX_LOCAL_PROPOSALS_PER_VIDEO,
    MAX_NEW_CANDIDATES_PER_VIDEO,
    MINIMUM_WINDOW_SECONDS,
    NMS_SECONDS,
    PREFERRED_WINDOW_SECONDS,
    SCAN_SIZE,
    SEED_CENTER_EXCLUSION_SECONDS,
    SEED_IOU_EXCLUSION,
    LocalContinuityResult,
    SignalSeries,
    WindowAdjustment,
    adaptive_window,
    dense_displayed_frames,
    dense_focus_frame,
    empirical_percentiles,
    hard_cut_mask,
    is_seed_near_duplicate,
    overview_displayed_frames,
    scan_coarse_video,
    scan_local_video,
    temporal_iou,
)

MB1_V021_VERSION = "0.2.1"
MB1_V021_MODE = "SUPPLEMENTARY_BOUNDARY_RICH_CANDIDATE_MINING_MODE_A"
EXPECTED_QC_SHA256 = "7094bc9f21d1ecf3b2b90af3a6a97f0e9c4e976e7823de180fb75d59ab097899"
EXPECTED_SEED_COUNT = 13
TARGET_SOURCE_MIN = 24
TARGET_SOURCE_MAX = 32
TARGET_NEW_CANDIDATES = 68
ACCEPTABLE_NEW_MIN = 56
MAX_NEW_CANDIDATES = 72
ALLOWED_QC_STATUS = frozenset({"USABLE", "CONDITIONAL", "REJECT"})
BUNDLE_BASE_FILES = (
    "mb1_v021_seed_manifest.jsonl",
    "mb1_v021_candidate_manifest.jsonl",
    "mb1_v021_candidate_diagnostics.jsonl",
    "candidate_selection_v021.json",
    "cut_guard_regression_audit.json",
    "README_AI_QC_V021.md",
    "run_manifest.json",
    "issues.jsonl",
)
FORBIDDEN_SUFFIXES = {
    ".mp4",
    ".avi",
    ".mkv",
    ".mov",
    ".pt",
    ".pth",
    ".bin",
    ".npy",
    ".npz",
}


@dataclass(frozen=True)
class MB1V021Config:
    dataset_root: Path
    old_candidate_manifest_path: Path
    old_candidate_diagnostics_path: Path
    old_candidate_selection_path: Path
    ai_qc_path: Path
    ai_qc_summary_path: Path
    rt2_benchmark_path: Path
    output_root: Path
    prior_rt2_selection_path: Path | None = None
    seed: int = 2026
    jpeg_quality: int = 88
    build_git_commit: str | None = None

    def __post_init__(self) -> None:
        if self.seed != 2026:
            raise ValueError("MB1 v0.2.1 seed is frozen at 2026")
        if not 70 <= self.jpeg_quality <= 95:
            raise ValueError("JPEG quality must be between 70 and 95")


@dataclass
class Proposal:
    video_id: str
    fps: float
    center_frame: int
    window: WindowAdjustment
    coarse_shot: ShotSegment
    pre_activity: dict[str, float]
    center_activity: dict[str, float]
    post_activity: dict[str, float]
    transition_strength: float
    before_after_visual_difference: float
    spatial_activity_concentration: float
    overall_activity: float
    source_pool_origin: str
    proposal_score: float = 0.0
    local_result: LocalContinuityResult | None = None
    dense_focus_frame: int | None = None
    candidate_origin: str = "NEW_SUPPLEMENTARY_REGION"
    overlapping_prior_candidate_id: str | None = None
    prior_qc_status: str | None = None
    prior_reason_code: str | None = None
    diagnostics_extra: dict[str, Any] = field(default_factory=dict)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).expanduser().resolve(strict=True).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path).expanduser().resolve(strict=True)
    rows = []
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"JSONL row is not an object: {source}:{line_number}")
        rows.append(value)
    return rows


def load_and_validate_qc(path: str | Path) -> list[dict[str, Any]]:
    actual = sha256_file(path)
    if actual != EXPECTED_QC_SHA256:
        raise ValueError(
            "MB1_V021_AI_QC_HASH_MISMATCH: "
            f"expected={EXPECTED_QC_SHA256} actual={actual}"
        )
    rows = _read_jsonl(path)
    if len(rows) != 41:
        raise ValueError(f"Expected 41 MB1 v0.2 QC rows, found {len(rows)}")
    if len({str(row.get("candidate_id")) for row in rows}) != len(rows):
        raise ValueError("MB1 v0.2 QC candidate IDs must be unique")
    for row in rows:
        if str(row.get("qc_status")) not in ALLOWED_QC_STATUS:
            raise ValueError("MB1 v0.2 QC status is invalid")
    return rows


def resolve_frozen_seeds(
    old_manifest_rows: list[dict[str, Any]], qc_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    candidates = {str(row["candidate_id"]): row for row in old_manifest_rows}
    qc_by_id = {str(row["candidate_id"]): row for row in qc_rows}
    if set(candidates) != set(qc_by_id):
        raise ValueError("MB1 v0.2 manifest/QC candidate identity sets differ")
    seeds = []
    for candidate_id in sorted(qc_by_id):
        qc = qc_by_id[candidate_id]
        if qc["qc_status"] != "USABLE":
            continue
        candidate = candidates[candidate_id]
        if str(candidate["video_id"]) != str(qc["video_id"]):
            raise ValueError("MB1 v0.2 seed candidate/video identity mismatch")
        seeds.append(
            {
                "seed_class": "MB1_V02_USABLE_SEED",
                "candidate_id": candidate_id,
                "video_id": str(candidate["video_id"]),
                "fps": float(candidate["fps"]),
                "window_start_frame": int(candidate["window_start_frame"]),
                "window_end_frame": int(candidate["window_end_frame"]),
                "proposal_center_frame": int(candidate["proposal_center_frame"]),
                "overview_sheet_path": str(candidate["overview_sheet_path"]),
                "dense_sheet_path": str(candidate["dense_sheet_path"]),
                "qc_status": "USABLE",
                "boundary_richness": qc.get("boundary_richness"),
            }
        )
    if len(seeds) != EXPECTED_SEED_COUNT:
        raise ValueError(f"Expected 13 frozen USABLE seeds, found {len(seeds)}")
    return seeds


def build_source_video_pool(
    rt2_rows: list[dict[str, Any]],
    prior_rt2_selection: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Build 24 RT2-query sources then up to 8 prior Mode-A sources."""

    pool: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in sorted(rt2_rows, key=lambda item: str(item["query_id"])):
        video_id = str(row["source_video_id"])
        if video_id in seen:
            continue
        seen.add(video_id)
        pool.append(
            {
                "video_id": video_id,
                "source_pool_origin": "RT2_BENCHMARK_QUERY_VIDEO",
                "existing_tags": [str(value) for value in row.get("difficulty_tags", [])],
            }
        )
    if prior_rt2_selection is not None:
        for row in prior_rt2_selection.get("selected_videos", []):
            video_id = str(row["video_id"])
            if video_id in seen:
                continue
            seen.add(video_id)
            pool.append(
                {
                    "video_id": video_id,
                    "source_pool_origin": "PRIOR_RT2_MODE_A_SELECTION",
                    "sampling_bucket": row.get("sampling_bucket"),
                }
            )
            if len(pool) == TARGET_SOURCE_MAX:
                break
    return pool


def _summary(values: np.ndarray) -> dict[str, float]:
    vector = np.asarray(values, dtype=np.float64)
    if not len(vector):
        return {"mean": 0.0, "std": 0.0, "peak": 0.0, "median": 0.0}
    return {
        "mean": float(np.mean(vector)),
        "std": float(np.std(vector)),
        "peak": float(np.max(vector)),
        "median": float(np.median(vector)),
    }


def _proposal_features(
    series: SignalSeries, window: WindowAdjustment
) -> tuple[dict[str, float], dict[str, float], dict[str, float], float, float, float, float]:
    start, end = window.start_frame, window.end_frame
    duration = end - start
    pre_end = start + int(round(0.3125 * duration))
    center_end = start + int(round(0.6875 * duration))
    pre_positions = np.flatnonzero(
        (series.frame_indices >= start) & (series.frame_indices < pre_end)
    )
    center_positions = np.flatnonzero(
        (series.frame_indices >= pre_end) & (series.frame_indices <= center_end)
    )
    post_positions = np.flatnonzero(
        (series.frame_indices > center_end) & (series.frame_indices <= end)
    )
    if min(len(pre_positions), len(center_positions), len(post_positions)) < 2:
        raise ValueError("candidate does not contain enough coarse samples in PRE/CENTER/POST")
    pre = _summary(series.pixel_differences[pre_positions])
    center = _summary(series.pixel_differences[center_positions])
    post = _summary(series.pixel_differences[post_positions])
    transition_strength = center["peak"] - max(pre["median"], post["median"])
    pre_histogram = np.mean(
        series.histograms[pre_positions[: max(1, len(pre_positions) // 2)]],
        axis=0,
    )
    post_histogram = np.mean(
        series.histograms[post_positions[-max(1, len(post_positions) // 2) :]], axis=0
    )
    before_after = float(0.5 * np.abs(pre_histogram - post_histogram).sum())
    all_positions = np.concatenate((pre_positions, center_positions, post_positions))
    spatial = float(np.mean(series.spatial_concentrations[center_positions]))
    overall = float(np.mean(series.pixel_differences[all_positions]))
    return pre, center, post, transition_strength, before_after, spatial, overall


def _assign_proposal_scores(proposals: list[Proposal]) -> None:
    components = (
        ("transition_strength", 0.40),
        ("before_after_visual_difference", 0.25),
        ("spatial_activity_concentration", 0.20),
        ("overall_activity", 0.15),
    )
    for proposal in proposals:
        proposal.proposal_score = 0.0
    for field_name, weight in components:
        values = np.asarray([getattr(item, field_name) for item in proposals])
        normalized = empirical_percentiles(values)
        for item, value in zip(proposals, normalized, strict=True):
            item.proposal_score += weight * float(value)


def _initial_proposals(
    *,
    video_id: str,
    fps: float,
    total_frames: int,
    series: SignalSeries,
    segments: tuple[ShotSegment, ...],
    coarse_cut_frames: tuple[int, ...],
    source_pool_origin: str,
    seeds: list[dict[str, Any]],
) -> tuple[list[Proposal], Counter[str]]:
    counts: Counter[str] = Counter()
    proposals: list[Proposal] = []
    center_step = max(1, int(round((0.5 * fps) / series.stride_frames)))
    activity_floor = max(0.004, 0.5 * series.pixel_baseline.median)
    for segment in segments:
        positions = np.flatnonzero(
            (series.frame_indices >= segment.start_frame)
            & (series.frame_indices <= segment.end_frame)
        )[::center_step]
        for position in positions:
            counts["raw_proposals_generated"] += 1
            center_frame = int(series.frame_indices[position])
            window = adaptive_window(
                center_frame=center_frame,
                fps=fps,
                shot_start_frame=segment.start_frame,
                shot_end_frame=segment.end_frame,
            )
            if window is None:
                counts["rejected_insufficient_clean_window"] += 1
                continue
            if any(
                window.start_frame < cut <= window.end_frame
                for cut in coarse_cut_frames
            ):
                counts["rejected_coarse_cut_overlap"] += 1
                continue
            try:
                features = _proposal_features(series, window)
            except ValueError:
                counts["rejected_insufficient_clean_window"] += 1
                continue
            pre, middle, post, transition, before_after, spatial, overall = features
            if overall < activity_floor:
                counts["rejected_activity_floor"] += 1
                continue
            if is_seed_near_duplicate(
                video_id=video_id,
                start_frame=window.start_frame,
                end_frame=window.end_frame,
                center_frame=center_frame,
                fps=fps,
                seeds=seeds,
            ):
                counts["rejected_seed_duplicate"] += 1
                continue
            proposals.append(
                Proposal(
                    video_id=video_id,
                    fps=fps,
                    center_frame=center_frame,
                    window=window,
                    coarse_shot=segment,
                    pre_activity=pre,
                    center_activity=middle,
                    post_activity=post,
                    transition_strength=transition,
                    before_after_visual_difference=before_after,
                    spatial_activity_concentration=spatial,
                    overall_activity=overall,
                    source_pool_origin=source_pool_origin,
                )
            )
    if proposals:
        _assign_proposal_scores(proposals)
    ordered = sorted(proposals, key=lambda item: (-item.proposal_score, item.center_frame))
    shortlist: list[Proposal] = []
    for proposal in ordered:
        if any(
            abs(proposal.center_frame - existing.center_frame) / fps < 1.0
            for existing in shortlist
        ):
            counts["rejected_pre_local_peak_dedup"] += 1
            continue
        if len(shortlist) >= MAX_LOCAL_PROPOSALS_PER_VIDEO:
            counts["rejected_pre_local_budget"] += 1
            continue
        shortlist.append(proposal)
    return shortlist, counts


def _overlapping_old_region(
    proposal: Proposal,
    old_rows: list[dict[str, Any]],
    qc_by_id: dict[str, dict[str, Any]],
) -> None:
    overlaps = []
    for row in old_rows:
        if str(row["video_id"]) != proposal.video_id:
            continue
        qc = qc_by_id[str(row["candidate_id"])]
        if qc["qc_status"] == "USABLE":
            continue
        overlap = temporal_iou(
            proposal.window.start_frame,
            proposal.window.end_frame,
            int(row["window_start_frame"]),
            int(row["window_end_frame"]),
        )
        if overlap > 0:
            overlaps.append((overlap, str(row["candidate_id"]), qc))
    if not overlaps:
        return
    _, candidate_id, qc = max(overlaps, key=lambda item: (item[0], item[1]))
    proposal.candidate_origin = "RESCUED_PRIOR_REGION"
    proposal.overlapping_prior_candidate_id = candidate_id
    proposal.prior_qc_status = str(qc["qc_status"])
    proposal.prior_reason_code = str(qc.get("reason_code"))


def _local_verify_proposals(
    *,
    decoder: Any,
    series: SignalSeries,
    proposals: list[Proposal],
    old_rows: list[dict[str, Any]],
    qc_by_id: dict[str, dict[str, Any]],
) -> tuple[list[Proposal], Counter[str], dict[str, float | int]]:
    counts: Counter[str] = Counter()
    retained = []
    local_frames = 0
    local_ms = 0.0
    adaptive_ms = 0.0
    for proposal in proposals:
        expansion = int(round(LOCAL_EXPANSION_SECONDS * proposal.fps))
        local_start = max(0, proposal.window.start_frame - expansion)
        local_end = min(
            int(decoder.info.total_frames) - 1, proposal.window.end_frame + expansion
        )
        local_series, result = scan_local_video(decoder, local_start, local_end, series)
        local_frames += len(result.frame_indices)
        local_ms += result.decode_ms + result.signal_ms
        hard_inside = tuple(
            value
            for value in result.hard_cut_frames
            if proposal.window.start_frame < value <= proposal.window.end_frame
        )
        soft_inside = tuple(
            value
            for value in result.soft_cut_frames
            if proposal.window.start_frame < value <= proposal.window.end_frame
        )
        adjustment_started = monotonic()
        if hard_inside or soft_inside:
            adjusted = adaptive_window(
                center_frame=proposal.center_frame,
                fps=proposal.fps,
                shot_start_frame=proposal.coarse_shot.start_frame,
                shot_end_frame=proposal.coarse_shot.end_frame,
                veto_frames=hard_inside + soft_inside,
            )
            adaptive_ms += (monotonic() - adjustment_started) * 1000
            if adjusted is None:
                if hard_inside:
                    counts["rejected_candidate_local_hard_cut"] += 1
                else:
                    counts["rejected_candidate_local_soft_cut"] += 1
                continue
            proposal.window = adjusted
            local_start = max(0, adjusted.start_frame - expansion)
            local_end = min(int(decoder.info.total_frames) - 1, adjusted.end_frame + expansion)
            local_series, result = scan_local_video(decoder, local_start, local_end, series)
            local_frames += len(result.frame_indices)
            local_ms += result.decode_ms + result.signal_ms
            if any(
                adjusted.start_frame < value <= adjusted.end_frame
                for value in result.hard_cut_frames
            ):
                counts["rejected_candidate_local_hard_cut"] += 1
                continue
            if any(
                adjusted.start_frame < value <= adjusted.end_frame
                for value in result.soft_cut_frames
            ):
                counts["rejected_candidate_local_soft_cut"] += 1
                continue
        else:
            adaptive_ms += (monotonic() - adjustment_started) * 1000
        proposal.local_result = result
        (
            proposal.pre_activity,
            proposal.center_activity,
            proposal.post_activity,
            proposal.transition_strength,
            proposal.before_after_visual_difference,
            proposal.spatial_activity_concentration,
            proposal.overall_activity,
        ) = _proposal_features(series, proposal.window)
        proposal.dense_focus_frame = dense_focus_frame(
            local_series, proposal.window.start_frame, proposal.window.end_frame
        )
        _overlapping_old_region(proposal, old_rows, qc_by_id)
        retained.append(proposal)
    return retained, counts, {
        "local_verification_frames": local_frames,
        "candidate_local_verification_ms": local_ms,
        "adaptive_window_ms": adaptive_ms,
    }


def _nms_and_cap(
    proposals: list[Proposal], fps: float
) -> tuple[list[Proposal], Counter[str]]:
    selected: list[Proposal] = []
    counts: Counter[str] = Counter()
    for proposal in sorted(
        proposals, key=lambda item: (-item.proposal_score, item.center_frame)
    ):
        if any(
            abs(proposal.center_frame - existing.center_frame) / fps < NMS_SECONDS
            for existing in selected
        ):
            counts["rejected_temporal_nms"] += 1
            continue
        if len(selected) >= MAX_NEW_CANDIDATES_PER_VIDEO:
            counts["rejected_per_video_cap"] += 1
            continue
        selected.append(proposal)
    return selected, counts


def cut_guard_regression_audit(
    *,
    dataset_root: Path,
    old_manifest_rows: list[dict[str, Any]],
    qc_rows: list[dict[str, Any]],
    coarse_by_video: dict[str, SignalSeries],
    video_assets: dict[str, Path],
    decoder_factory: Callable[[str, Path], Any] = OpenCVRawVideoDecoder,
) -> dict[str, Any]:
    qc_by_id = {str(row["candidate_id"]): row for row in qc_rows}
    rows = []
    for old in sorted(old_manifest_rows, key=lambda item: str(item["candidate_id"])):
        video_id = str(old["video_id"])
        decoder = decoder_factory(video_id, video_assets[video_id])
        try:
            expansion = int(round(LOCAL_EXPANSION_SECONDS * float(old["fps"])))
            start = max(0, int(old["window_start_frame"]) - expansion)
            end = min(
                int(decoder.info.total_frames) - 1,
                int(old["window_end_frame"]) + expansion,
            )
            _, result = scan_local_video(decoder, start, end, coarse_by_video[video_id])
        finally:
            decoder.close()
        hard = any(
            int(old["window_start_frame"]) < value <= int(old["window_end_frame"])
            for value in result.hard_cut_frames
        )
        soft = any(
            int(old["window_start_frame"]) < value <= int(old["window_end_frame"])
            for value in result.soft_cut_frames
        )
        qc = qc_by_id[str(old["candidate_id"])]
        rows.append(
            {
                "candidate_id": old["candidate_id"],
                "video_id": video_id,
                "qc_status": qc["qc_status"],
                "reason_code": qc.get("reason_code"),
                "hard_cut_veto": hard,
                "soft_cut_veto": soft,
                "either_veto": hard or soft,
            }
        )
    old_hard = [row for row in rows if row["reason_code"] == "HARD_CUT"]
    old_usable = [row for row in rows if row["qc_status"] == "USABLE"]
    hard_caught = sum(bool(row["either_veto"]) for row in old_hard)
    usable_vetoed = sum(bool(row["either_veto"]) for row in old_usable)
    return {
        "audit_role": "POST_HOC_DIAGNOSTIC_NOT_OFFICIAL_GT",
        "threshold_tuning_from_audit": False,
        "old_candidate_count": len(rows),
        "old_ai_qc_hard_cut_count": len(old_hard),
        "old_hard_cut_vetoed_by_hard_cut": sum(
            bool(row["hard_cut_veto"]) for row in old_hard
        ),
        "old_hard_cut_vetoed_by_soft_cut": sum(
            bool(row["soft_cut_veto"]) for row in old_hard
        ),
        "old_hard_cut_vetoed_by_either": hard_caught,
        "OLD_HARD_CUT_RECALL": hard_caught / len(old_hard) if old_hard else None,
        "old_usable_count": len(old_usable),
        "old_usable_falsely_vetoed": usable_vetoed,
        "OLD_USABLE_FALSE_VETO_RATE": (
            usable_vetoed / len(old_usable) if old_usable else None
        ),
        "candidate_results": rows,
    }


def preflight_mb1_v021(config: MB1V021Config) -> dict[str, Any]:
    dataset = config.dataset_root.expanduser().resolve(strict=True)
    if config.output_root.exists():
        raise FileExistsError(f"MB1 v0.2.1 output already exists: {config.output_root}")
    old_rows = _read_jsonl(config.old_candidate_manifest_path)
    _read_jsonl(config.old_candidate_diagnostics_path)
    _read_json(config.old_candidate_selection_path)
    qc_rows = load_and_validate_qc(config.ai_qc_path)
    qc_summary = _read_json(config.ai_qc_summary_path)
    if qc_summary.get("qc_sha256") != EXPECTED_QC_SHA256:
        raise ValueError("MB1 v0.2.1 QC summary does not bind the expected QC hash")
    seeds = resolve_frozen_seeds(old_rows, qc_rows)
    rt2_rows = _read_jsonl(config.rt2_benchmark_path)
    prior_selection = (
        _read_json(config.prior_rt2_selection_path)
        if config.prior_rt2_selection_path is not None
        else None
    )
    pool = build_source_video_pool(rt2_rows, prior_selection)
    video_partitions, keyframe_partitions = discover_layout(dataset)
    missing = []
    for source in pool:
        assets = resolve_assets(
            dataset, source["video_id"], video_partitions, keyframe_partitions
        )
        if not assets.video.is_file():
            missing.append(source["video_id"])
    if missing:
        raise FileNotFoundError(f"MB1 v0.2.1 raw videos missing: {missing}")
    return {
        "status": "READY",
        "mode": MB1_V021_MODE,
        "ai_qc_sha256": EXPECTED_QC_SHA256,
        "frozen_seed_count": len(seeds),
        "source_video_count": len(pool),
        "source_videos": pool,
        "target_source_range": [TARGET_SOURCE_MIN, TARGET_SOURCE_MAX],
        "target_new_candidate_count": TARGET_NEW_CANDIDATES,
        "acceptable_new_candidate_range": [ACCEPTABLE_NEW_MIN, MAX_NEW_CANDIDATES],
        "maximum_with_source_pool_and_cap": len(pool) * MAX_NEW_CANDIDATES_PER_VIDEO,
        "model_inference_required": False,
        "semantic_interval_gt_required": False,
        "network_required": False,
    }


def _readme() -> str:
    return """# TRIAGE-EG MB1 v0.2.1 AI QC

This supplementary pack contains model-free visual evidence only. It assigns no semantic
moment type, query, preferred frame, or acceptable interval. The 13 MB1 v0.2 USABLE seeds
are frozen separately and are not regenerated here.

Review every NEW candidate using overview and dense sheets, then assign exactly one:
`USABLE`, `CONDITIONAL`, or `REJECT`.

Suggested reason codes: `HARD_CUT`, `SOFT_CUT`, `STATIC_STATE`, `CAMERA_MOTION`,
`GRAPHIC_OVERLAY`, `NO_DEFENSIBLE_TEMPORAL_CHANGE`, `REPETITIVE_AMBIGUOUS_ACTION`,
`OCCLUDED`, `INSUFFICIENT_BEFORE_AFTER_EVIDENCE`, or `OTHER`.

After QC, merge only new USABLE candidates with the frozen 13 seeds. Semantic interval
annotation begins only after the merged pool is sufficiently rich. Use the printed
`actual_frame_idx` values; do not reconstruct raw coordinates from timestamps.
"""


def _render_montages(
    output: Path, manifest_rows: list[dict[str, Any]], view: str
) -> list[Path]:
    from PIL import Image, ImageDraw

    key = f"{view}_sheet_path"
    paths = []
    for page, start in enumerate(range(0, len(manifest_rows), 4), 1):
        subset = manifest_rows[start : start + 4]
        canvas = Image.new("RGB", (1280, 1000), "white")
        draw = ImageDraw.Draw(canvas)
        for slot, row in enumerate(subset):
            source = Image.open(output / row[key]).convert("RGB")
            source.thumbnail((620, 450))
            x = 10 + (slot % 2) * 640
            y = 35 + (slot // 2) * 480
            canvas.paste(source, (x, y))
            draw.text((x, y - 22), f"{row['candidate_id']} | {row['video_id']}", fill="black")
        target = output / "montages" / f"{view}_montage_{page:03d}.jpg"
        target.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(target, format="JPEG", quality=82, optimize=True)
        paths.append(target)
    return paths


def prepare_mb1_v021_candidates(
    config: MB1V021Config,
    *,
    decoder_factory: Callable[[str, Path], Any] = OpenCVRawVideoDecoder,
) -> dict[str, Any]:
    overall_started = monotonic()
    source_started = monotonic()
    preflight = preflight_mb1_v021(config)
    source_pool_ms = (monotonic() - source_started) * 1000
    dataset = config.dataset_root.expanduser().resolve(strict=True)
    output = config.output_root.expanduser().resolve(strict=False)
    output.mkdir(parents=True)
    old_rows = _read_jsonl(config.old_candidate_manifest_path)
    qc_rows = load_and_validate_qc(config.ai_qc_path)
    qc_by_id = {str(row["candidate_id"]): row for row in qc_rows}
    seeds = resolve_frozen_seeds(old_rows, qc_rows)
    write_jsonl(output / "mb1_v021_seed_manifest.jsonl", seeds)
    video_partitions, keyframe_partitions = discover_layout(dataset)
    video_assets: dict[str, Path] = {}
    coarse_by_video: dict[str, SignalSeries] = {}
    coarse_cuts_by_video: dict[str, tuple[int, ...]] = {}
    shots_by_video: dict[str, tuple[ShotSegment, ...]] = {}
    performance_by_video: dict[str, dict[str, Any]] = {}

    for source in preflight["source_videos"]:
        video_id = str(source["video_id"])
        assets = resolve_assets(dataset, video_id, video_partitions, keyframe_partitions)
        video_assets[video_id] = assets.video
        decoder = decoder_factory(video_id, assets.video)
        try:
            coarse = scan_coarse_video(decoder)
            cut_mask = hard_cut_mask(
                coarse.pixel_percentiles, coarse.histogram_percentiles
            )
            cuts = tuple(
                int(value)
                for value in coarse.frame_indices[np.flatnonzero(cut_mask)]
                if 0 < int(value) < int(decoder.info.total_frames)
            )
            shots = continuous_shot_segments(int(decoder.info.total_frames), cuts)
            coarse_by_video[video_id] = coarse
            coarse_cuts_by_video[video_id] = cuts
            shots_by_video[video_id] = shots
            performance_by_video[video_id] = {
                "video_id": video_id,
                "fps": float(decoder.info.fps),
                "total_frames": int(decoder.info.total_frames),
                "coarse_scan_frames": len(coarse.frame_indices),
                "coarse_cut_count": len(cuts),
                "coarse_scan_decode_ms": coarse.decode_ms,
                "signal_computation_ms": coarse.signal_ms,
                "proposal_construction_ms": 0.0,
                "candidate_local_verification_ms": 0.0,
                "adaptive_window_ms": 0.0,
                "sheet_rendering_ms": 0.0,
                "local_verification_frames": 0,
                "normalization": {
                    "pixel": coarse.pixel_baseline.as_dict(),
                    "histogram": coarse.histogram_baseline.as_dict(),
                },
            }
        finally:
            decoder.close()

    audit = cut_guard_regression_audit(
        dataset_root=dataset,
        old_manifest_rows=old_rows,
        qc_rows=qc_rows,
        coarse_by_video=coarse_by_video,
        video_assets=video_assets,
        decoder_factory=decoder_factory,
    )
    write_json(output / "cut_guard_regression_audit.json", audit)

    all_selected: list[Proposal] = []
    global_counts: Counter[str] = Counter()
    source_by_video = {
        str(row["video_id"]): str(row["source_pool_origin"])
        for row in preflight["source_videos"]
    }
    for source in preflight["source_videos"]:
        video_id = str(source["video_id"])
        decoder = decoder_factory(video_id, video_assets[video_id])
        proposal_started = monotonic()
        try:
            proposals, counts = _initial_proposals(
                video_id=video_id,
                fps=float(decoder.info.fps),
                total_frames=int(decoder.info.total_frames),
                series=coarse_by_video[video_id],
                segments=shots_by_video[video_id],
                coarse_cut_frames=coarse_cuts_by_video[video_id],
                source_pool_origin=source_by_video[video_id],
                seeds=seeds,
            )
            performance_by_video[video_id]["proposal_construction_ms"] = (
                monotonic() - proposal_started
            ) * 1000
            verified, local_counts, local_performance = _local_verify_proposals(
                decoder=decoder,
                series=coarse_by_video[video_id],
                proposals=proposals,
                old_rows=old_rows,
                qc_by_id=qc_by_id,
            )
            if verified:
                _assign_proposal_scores(verified)
            selected, selection_counts = _nms_and_cap(
                verified, float(decoder.info.fps)
            )
            all_selected.extend(selected)
            counts.update(local_counts)
            counts.update(selection_counts)
            global_counts.update(counts)
            performance_by_video[video_id].update(local_performance)
            performance_by_video[video_id]["selection_counts"] = dict(counts)
            performance_by_video[video_id]["retained"] = len(selected)
        finally:
            decoder.close()

    globally_ranked = sorted(
        all_selected,
        key=lambda item: (-item.proposal_score, item.video_id, item.center_frame),
    )
    bounded = globally_ranked[: min(TARGET_NEW_CANDIDATES, MAX_NEW_CANDIDATES)]
    global_counts["rejected_global_target"] = len(globally_ranked) - len(bounded)
    ordered = sorted(
        bounded,
        key=lambda item: (
            item.video_id,
            item.window.start_frame,
            item.center_frame,
        ),
    )
    manifest_rows: list[dict[str, Any]] = []
    diagnostics_rows: list[dict[str, Any]] = []
    rendered_frame_count = 0
    for number, proposal in enumerate(ordered, 1):
        if proposal.local_result is None or proposal.dense_focus_frame is None:
            raise RuntimeError("MB1_V021_RETAINED_PROPOSAL_NOT_LOCALLY_VERIFIED")
        candidate_id = f"mb1v021_c{number:03d}"
        overview_ids = overview_displayed_frames(
            proposal.window.start_frame, proposal.window.end_frame
        )
        dense_ids = dense_displayed_frames(
            proposal.window.start_frame,
            proposal.window.end_frame,
            proposal.dense_focus_frame,
            proposal.fps,
        )
        requested = sorted(set(overview_ids + dense_ids))
        decoder = decoder_factory(proposal.video_id, video_assets[proposal.video_id])
        render_started = monotonic()
        try:
            frames = decoder.decode_indices(requested)
        finally:
            decoder.close()
        if [int(frame.actual_frame_idx) for frame in frames] != requested:
            raise RuntimeError("MB1_V021_RENDER_FRAME_IDENTITY_MISMATCH")
        by_id = {int(frame.actual_frame_idx): frame for frame in frames}
        overview_path = Path("overview") / f"{candidate_id}.jpg"
        dense_path = Path("dense") / f"{candidate_id}.jpg"
        overview_labels = render_contact_sheet(
            output / overview_path,
            candidate_id=candidate_id,
            video_id=proposal.video_id,
            view_name="OVERVIEW",
            fps=proposal.fps,
            frames=[by_id[value] for value in overview_ids],
            quality=config.jpeg_quality,
        )
        dense_labels = render_contact_sheet(
            output / dense_path,
            candidate_id=candidate_id,
            video_id=proposal.video_id,
            view_name="DENSE_LOCAL_CHANGE",
            fps=proposal.fps,
            frames=[by_id[value] for value in dense_ids],
            quality=config.jpeg_quality,
        )
        if overview_labels != overview_ids or dense_labels != dense_ids:
            raise RuntimeError("MB1_V021_MANIFEST_SHEET_LABEL_MISMATCH")
        performance_by_video[proposal.video_id]["sheet_rendering_ms"] += (
            monotonic() - render_started
        ) * 1000
        rendered_frame_count += len(requested)
        manifest_rows.append(
            {
                "candidate_id": candidate_id,
                "video_id": proposal.video_id,
                "fps": proposal.fps,
                "window_start_frame": proposal.window.start_frame,
                "window_end_frame": proposal.window.end_frame,
                "final_window_duration_seconds": proposal.window.final_duration_seconds,
                "proposal_center_frame": proposal.center_frame,
                "dense_focus_frame": proposal.dense_focus_frame,
                "coarse_shot_start_frame": proposal.coarse_shot.start_frame,
                "coarse_shot_end_frame": proposal.coarse_shot.end_frame,
                "overview_displayed_frames": overview_ids,
                "dense_displayed_frames": dense_ids,
                "overview_sheet_path": overview_path.as_posix(),
                "dense_sheet_path": dense_path.as_posix(),
                "proposal_score": proposal.proposal_score,
                "transition_strength": proposal.transition_strength,
                "before_after_visual_difference": (
                    proposal.before_after_visual_difference
                ),
                "spatial_activity_concentration": (
                    proposal.spatial_activity_concentration
                ),
                "overall_activity": proposal.overall_activity,
                "continuity_status": "PASS_LOCAL_HARD_AND_SOFT_CUT_GUARD",
                "window_adjustment_reason": proposal.window.reason,
                "candidate_origin": proposal.candidate_origin,
                "overlapping_prior_candidate_id": (
                    proposal.overlapping_prior_candidate_id
                ),
                "prior_qc_status": proposal.prior_qc_status,
                "prior_reason_code": proposal.prior_reason_code,
                "source_pool_origin": proposal.source_pool_origin,
            }
        )
        all_cut_frames = tuple(
            sorted(
                set(
                    coarse_cuts_by_video[proposal.video_id]
                    + proposal.local_result.veto_frames
                )
            )
        )
        distances = [
            min(
                abs(proposal.window.start_frame - cut),
                abs(proposal.window.end_frame - cut),
            )
            / proposal.fps
            for cut in all_cut_frames
        ]
        diagnostics_rows.append(
            {
                "candidate_id": candidate_id,
                "video_id": proposal.video_id,
                "coarse_scan_signal_summaries": {
                    "pixel": coarse_by_video[proposal.video_id].pixel_baseline.as_dict(),
                    "histogram": coarse_by_video[
                        proposal.video_id
                    ].histogram_baseline.as_dict(),
                },
                "max_pixel_percentile": proposal.local_result.max_pixel_percentile,
                "max_hist_percentile": proposal.local_result.max_hist_percentile,
                "max_pixel_robust_z": proposal.local_result.max_pixel_robust_z,
                "max_hist_robust_z": proposal.local_result.max_hist_robust_z,
                "hard_cut_veto": False,
                "soft_cut_veto": False,
                "minimum_distance_to_detected_cut_seconds": (
                    min(distances) if distances else None
                ),
                "pre_activity": proposal.pre_activity,
                "center_activity": proposal.center_activity,
                "post_activity": proposal.post_activity,
                "transition_strength": proposal.transition_strength,
                "before_after_visual_difference": (
                    proposal.before_after_visual_difference
                ),
                "spatial_activity_concentration": (
                    proposal.spatial_activity_concentration
                ),
                "requested_window_duration_seconds": (
                    proposal.window.requested_duration_seconds
                ),
                "final_window_duration_seconds": (
                    proposal.window.final_duration_seconds
                ),
                "window_adjustment_reason": proposal.window.reason,
            }
        )

    per_video_counts = dict(
        sorted(Counter(row["video_id"] for row in manifest_rows).items())
    )
    if any(value > MAX_NEW_CANDIDATES_PER_VIDEO for value in per_video_counts.values()):
        raise RuntimeError("MB1 v0.2.1 per-video cap was exceeded")
    if any(
        not MINIMUM_WINDOW_SECONDS
        <= row["final_window_duration_seconds"]
        <= PREFERRED_WINDOW_SECONDS + 0.05
        for row in manifest_rows
    ):
        raise RuntimeError("MB1 v0.2.1 retained an invalid window duration")
    if any(
        is_seed_near_duplicate(
            video_id=row["video_id"],
            start_frame=row["window_start_frame"],
            end_frame=row["window_end_frame"],
            center_frame=row["proposal_center_frame"],
            fps=row["fps"],
            seeds=seeds,
        )
        for row in manifest_rows
    ):
        raise RuntimeError("MB1 v0.2.1 retained a frozen-seed near duplicate")
    overview_montages = _render_montages(output, manifest_rows, "overview")
    dense_montages = _render_montages(output, manifest_rows, "dense")
    issues = []
    if len(manifest_rows) < ACCEPTABLE_NEW_MIN:
        issues.append(
            {
                "severity": "WARNING",
                "code": "FEWER_THAN_56_NEW_CANDIDATES",
                "message": "Quality and continuity gates were not relaxed to force count.",
                "evidence": {"retained_new": len(manifest_rows)},
            }
        )
    selection = {
        "experiment": "MB1_V021",
        "version": MB1_V021_VERSION,
        "mode": MB1_V021_MODE,
        "semantic_labels_assigned": False,
        "source_videos_available": [
            row["video_id"] for row in preflight["source_videos"]
        ],
        "source_video_count_available": len(preflight["source_videos"]),
        "source_videos_considered": [
            row["video_id"] for row in preflight["source_videos"]
        ],
        "source_video_count_considered": len(preflight["source_videos"]),
        "source_videos_with_retained_candidates": sorted(per_video_counts),
        "source_video_count_with_retained_candidates": len(per_video_counts),
        "existing_seed_count": len(seeds),
        "raw_proposals_generated": global_counts["raw_proposals_generated"],
        "rejected": {
            "activity_floor": global_counts["rejected_activity_floor"],
            "coarse_cut_overlap": global_counts["rejected_coarse_cut_overlap"],
            "candidate_local_HARD_CUT": global_counts[
                "rejected_candidate_local_hard_cut"
            ],
            "candidate_local_SOFT_CUT": global_counts[
                "rejected_candidate_local_soft_cut"
            ],
            "insufficient_clean_3_second_window": global_counts[
                "rejected_insufficient_clean_window"
            ],
            "seed_duplicate": global_counts["rejected_seed_duplicate"],
            "temporal_NMS": global_counts["rejected_temporal_nms"],
            "per_video_cap": global_counts["rejected_per_video_cap"],
            "pre_local_peak_dedup": global_counts[
                "rejected_pre_local_peak_dedup"
            ],
            "pre_local_verification_budget": global_counts[
                "rejected_pre_local_budget"
            ],
            "global_preferred_target": global_counts["rejected_global_target"],
        },
        "retained_NEW_candidate_count": len(manifest_rows),
        "combined_potential_pool_count": len(seeds) + len(manifest_rows),
        "per_video_retained_counts": per_video_counts,
        "rescued_prior_region_count": sum(
            row["candidate_origin"] == "RESCUED_PRIOR_REGION"
            for row in manifest_rows
        ),
        "candidate_ordering_rule": (
            "video_id ASC, window_start_frame ASC, proposal_center_frame ASC"
        ),
        "scan_parameters": {
            "coarse_samples_per_second": COARSE_SAMPLES_PER_SECOND,
            "coarse_stride_rule": "max(2, round(fps / 10))",
            "local_samples_per_second": LOCAL_SAMPLES_PER_SECOND,
            "local_stride_rule": "max(1, round(fps / 15))",
            "local_expansion_seconds_each_side": LOCAL_EXPANSION_SECONDS,
            "low_resolution_size": list(SCAN_SIZE),
        },
        "normalization_rules": {
            "robust_z": "(x - median) / max(1.4826 * MAD, 1e-9)",
            "empirical_percentile": "count(reference <= x) / len(reference)",
            "proposal_components": "within-video empirical percentile ranks",
        },
        "hard_cut_rule": {
            "strong_dual": "(pixel>=.99 AND hist>=.95) OR reverse",
            "extreme_supported": "(pixel>=.998 AND hist>=.80) OR reverse",
        },
        "soft_cut_rule": {
            "both_robust_z_minimum": 2.5,
            "run_duration_seconds": [0.20, 0.45],
            "before_after_histogram_minimum": "video coarse histogram p95",
        },
        "transition_score_formula": (
            "0.40*rank(transition_strength) + 0.25*rank(before_after_difference) "
            "+ 0.20*rank(spatial_concentration) + 0.15*rank(overall_activity)"
        ),
        "adaptive_window_rule": {
            "preferred_seconds": PREFERRED_WINDOW_SECONDS,
            "minimum_seconds": MINIMUM_WINDOW_SECONDS,
            "preserve_proposal_center": True,
            "reject_cut_within_center_guard_seconds": 0.5,
        },
        "seed_exclusion_rule": {
            "window_iou_greater_than": SEED_IOU_EXCLUSION,
            "center_distance_less_than_seconds": SEED_CENTER_EXCLUSION_SECONDS,
        },
        "nms_rule": {
            "minimum_center_separation_seconds": NMS_SECONDS,
            "per_video_new_cap": MAX_NEW_CANDIDATES_PER_VIDEO,
            "tie_break": "lower actual frame index first",
        },
        "overview_sheet_count": len(manifest_rows),
        "dense_sheet_count": len(manifest_rows),
        "overview_montage_count": len(overview_montages),
        "dense_montage_count": len(dense_montages),
    }
    run_manifest = {
        "experiment": "MB1_V021",
        "version": MB1_V021_VERSION,
        "mode": MB1_V021_MODE,
        "created_at": datetime.now(UTC).isoformat(),
        "build_git_commit": config.build_git_commit,
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(config).items()
        },
        "preflight": preflight,
        "prior_artifact_hashes": {
            "old_candidate_manifest": sha256_file(
                config.old_candidate_manifest_path
            ),
            "old_candidate_diagnostics": sha256_file(
                config.old_candidate_diagnostics_path
            ),
            "old_candidate_selection": sha256_file(
                config.old_candidate_selection_path
            ),
            "ai_qc_pass1": sha256_file(config.ai_qc_path),
            "ai_qc_summary": sha256_file(config.ai_qc_summary_path),
            "rt2_benchmark": sha256_file(config.rt2_benchmark_path),
            "prior_rt2_selection": (
                sha256_file(config.prior_rt2_selection_path)
                if config.prior_rt2_selection_path is not None
                else None
            ),
        },
        "old_qc_used_only_for_post_hoc_audit_and_seed_identity": True,
        "previous_semantic_interval_fields_read_by_generation": False,
        "model_inference": False,
        "raw_frame_coordinate_source": "M1_OPENCV_RAW_VIDEO_DECODER_ACTUAL_FRAME_IDX",
        "raw_frames_added_to_stage1_or_framemap": False,
        "network_required": False,
        "performance": {
            "source_pool_resolution_ms": source_pool_ms,
            "videos": [
                performance_by_video[row["video_id"]]
                for row in preflight["source_videos"]
            ],
            "overall_runtime_ms": (monotonic() - overall_started) * 1000,
            "coarse_scan_frames": sum(
                int(row["coarse_scan_frames"])
                for row in performance_by_video.values()
            ),
            "local_verification_frames": sum(
                int(row["local_verification_frames"])
                for row in performance_by_video.values()
            ),
            "full_resolution_rendered_frames": rendered_frame_count,
        },
        "quality_gates": {
            "all_final_candidates_pass_local_continuity": True,
            "both_visual_scales_present": True,
            "manifest_sheet_frame_mapping_exact": True,
            "raw_frame_ids_strict_and_unique": True,
            "window_duration_valid": True,
            "per_video_new_cap_valid": True,
            "zero_frozen_seed_near_duplicates": True,
            "previous_semantic_interval_used": False,
            "candidate_count_quality_target_met": len(manifest_rows)
            >= ACCEPTABLE_NEW_MIN,
        },
        "MB1_V021_REAL_STATUS": "COMPLETE",
        "MB1_V021_AI_QC_STATUS": "WAITING_FOR_AI",
        "M3_IMPLEMENTATION_STATUS": "NOT_STARTED",
    }
    write_jsonl(output / "mb1_v021_candidate_manifest.jsonl", manifest_rows)
    write_jsonl(output / "mb1_v021_candidate_diagnostics.jsonl", diagnostics_rows)
    write_json(output / "candidate_selection_v021.json", selection)
    (output / "README_AI_QC_V021.md").write_text(_readme(), encoding="utf-8")
    write_json(output / "run_manifest.json", run_manifest)
    write_jsonl(output / "issues.jsonl", issues)
    return {
        "selection": selection,
        "audit": audit,
        "run_manifest": run_manifest,
        "issues": issues,
    }


def create_mb1_v021_bundle(output_root: str | Path, zip_path: str | Path) -> Path:
    source = Path(output_root).expanduser().resolve(strict=True)
    target = Path(zip_path).expanduser().resolve(strict=False)
    if source in target.parents:
        raise ValueError("MB1 v0.2.1 ZIP must be outside output root")
    manifest = _read_jsonl(source / "mb1_v021_candidate_manifest.jsonl")
    members = [source / name for name in BUNDLE_BASE_FILES]
    members.extend(source / row["overview_sheet_path"] for row in manifest)
    members.extend(source / row["dense_sheet_path"] for row in manifest)
    members.extend(sorted((source / "montages").glob("overview_montage_*.jpg")))
    members.extend(sorted((source / "montages").glob("dense_montage_*.jpg")))
    if len(members) != len(set(members)):
        raise ValueError("MB1 v0.2.1 bundle members are not unique")
    missing = [str(path.relative_to(source)) for path in members if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"MB1 v0.2.1 bundle members missing: {missing}")
    if any(path.suffix.lower() in FORBIDDEN_SUFFIXES for path in members):
        raise ValueError("MB1 v0.2.1 bundle contains a forbidden heavy artifact")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.with_suffix(target.suffix + ".building")
    staging.unlink(missing_ok=True)
    try:
        with ZipFile(staging, "w", compression=ZIP_DEFLATED) as archive:
            for path in sorted(members, key=lambda item: item.relative_to(source).as_posix()):
                archive.write(path, path.relative_to(source).as_posix())
        shutil.move(staging, target)
    finally:
        staging.unlink(missing_ok=True)
    return target


__all__ = [
    "EXPECTED_QC_SHA256",
    "MB1V021Config",
    "MB1_V021_MODE",
    "MB1_V021_VERSION",
    "build_source_video_pool",
    "create_mb1_v021_bundle",
    "cut_guard_regression_audit",
    "load_and_validate_qc",
    "preflight_mb1_v021",
    "prepare_mb1_v021_candidates",
    "resolve_frozen_seeds",
    "sha256_file",
]
