"""Targeted trusted-source MB1 v0.2.2 candidate preparation."""

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
from triage_eg.experiments.mb1_v021.runner import _decode_resized_frames, _render_montages
from triage_eg.experiments.mb1_v021.signals import (
    SignalSeries,
    empirical_percentiles,
    hard_cut_mask,
    is_seed_near_duplicate,
    overview_displayed_frames,
    scan_coarse_video,
    temporal_iou,
)
from triage_eg.retrieval.stage1b.writers import write_json, write_jsonl
from triage_eg.video import OpenCVRawVideoDecoder

from .signals import (
    CUT_SAFETY_MARGIN_SECONDS,
    DENSE_EDGE_CONTEXT_SECONDS,
    MIN_POST_CONTEXT_SECONDS,
    MIN_PRE_CONTEXT_SECONDS,
    ContextWindow,
    FinalContinuityAudit,
    center_relative_features,
    context_window,
    dense_displayed_frames,
    final_continuity_audit,
    safe_dense_focus,
)

MB1_V022_VERSION = "0.2.2"
MB1_V022_MODE = "TRUSTED_SOURCE_CONTINUITY_REPAIR_TARGETED_MODE_A"
EXPECTED_QC_SHA256 = "7094bc9f21d1ecf3b2b90af3a6a97f0e9c4e976e7823de180fb75d59ab097899"
EXPECTED_SEED_COUNT = 13
TARGET_NEW = 42
MIN_TARGET_NEW = 36
MAX_TARGET_NEW = 48
MAX_PER_VIDEO = 4
NMS_SECONDS = 5.0
MAX_LOCAL_PER_VIDEO = 24
SECONDARY_PREQUALIFIED = ("L26_V065", "L26_V094", "L26_V156", "L30_V043")
FORBIDDEN_SEMANTIC_FIELDS = frozenset(
    {"acceptable_start_frame", "acceptable_end_frame", "preferred_frame"}
)
BUNDLE_BASE_FILES = (
    "mb1_v022_candidate_manifest.jsonl",
    "mb1_v022_candidate_diagnostics.jsonl",
    "candidate_selection_v022.json",
    "old_qc_continuity_audit_v022.json",
    "v021_geometry_audit.json",
    "README_AI_QC_V022.md",
    "run_manifest.json",
    "issues.jsonl",
)
FORBIDDEN_SUFFIXES = frozenset(
    {".mp4", ".avi", ".mkv", ".mov", ".pt", ".pth", ".bin", ".npy", ".npz"}
)


@dataclass(frozen=True)
class MB1V022Config:
    dataset_root: Path
    v021_seed_manifest_path: Path
    v021_candidate_manifest_path: Path
    v021_candidate_diagnostics_path: Path
    v021_selection_path: Path
    v021_cut_audit_path: Path
    old_v02_candidate_manifest_path: Path
    ai_qc_path: Path
    rt2_benchmark_path: Path
    output_root: Path
    jpeg_quality: int = 88
    seed: int = 2026
    build_git_commit: str | None = None

    def __post_init__(self) -> None:
        if self.seed != 2026:
            raise ValueError("MB1 v0.2.2 seed is frozen at 2026")
        if not 70 <= self.jpeg_quality <= 95:
            raise ValueError("JPEG quality must be between 70 and 95")


@dataclass
class Proposal:
    video_id: str
    fps: float
    center_frame: int
    context: ContextWindow
    source_pool_origin: str
    features: dict[str, Any]
    base_transition_score: float = 0.0
    proposal_score: float = 0.0
    audit: FinalContinuityAudit | None = None
    dense_focus_frame: int | None = None
    overlapping_seed_id: str | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.expanduser().resolve(strict=True).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.expanduser().resolve(strict=True).read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row must be object: {path}")
            rows.append(value)
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_seeds(path: Path) -> list[dict[str, Any]]:
    seeds = _read_jsonl(path)
    if len(seeds) != EXPECTED_SEED_COUNT or any(row.get("qc_status") != "USABLE" for row in seeds):
        raise ValueError("MB1 v0.2.2 requires exactly 13 frozen USABLE seeds")
    return seeds


def build_trusted_source_pool(
    seeds: list[dict[str, Any]], v021_manifest: list[dict[str, Any]]
) -> list[dict[str, str]]:
    pool = []
    seen = set()
    for video_id in sorted({str(row["video_id"]) for row in seeds}):
        pool.append({"video_id": video_id, "source_pool_origin": "FROZEN_USABLE_SEED_VIDEO"})
        seen.add(video_id)
    available = {str(row["video_id"]) for row in v021_manifest}
    for video_id in SECONDARY_PREQUALIFIED:
        if video_id in available and video_id not in seen:
            pool.append(
                {"video_id": video_id, "source_pool_origin": "PROMPT_PREQUALIFIED_RESOLVED_V021"}
            )
            seen.add(video_id)
    if not 10 <= len(pool) <= 16:
        raise ValueError(f"Trusted source pool must contain 10-16 videos, found {len(pool)}")
    return pool


def v021_geometry_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    results = []
    for row in rows:
        fps = float(row["fps"])
        pre = (int(row["proposal_center_frame"]) - int(row["window_start_frame"])) / fps
        post = (int(row["window_end_frame"]) - int(row["proposal_center_frame"])) / fps
        reasons = []
        if pre < MIN_PRE_CONTEXT_SECONDS:
            reasons.append("INSUFFICIENT_PRE_CONTEXT")
        if post < MIN_POST_CONTEXT_SECONDS:
            reasons.append("INSUFFICIENT_POST_CONTEXT")
        results.append(
            {
                "candidate_id": row["candidate_id"],
                "video_id": row["video_id"],
                "pre_context_seconds": pre,
                "post_context_seconds": post,
                "geometry_reject": bool(reasons),
                "reasons": reasons,
            }
        )
    rejected = sum(row["geometry_reject"] for row in results)
    edge = sum(
        int(row["proposal_center_frame"])
        in {int(row["window_start_frame"]), int(row["window_end_frame"])}
        for row in rows
    )
    return {
        "audit_role": "STRUCTURAL_GEOMETRY_ONLY_NO_SEMANTIC_LABELS",
        "candidate_count": len(rows),
        "proposal_center_at_window_edge": edge,
        "V021_REJECTED_BY_CONTEXT_GEOMETRY": rejected,
        "candidate_results": results,
    }


def _old_qc_audit(
    *,
    dataset: Path,
    old_rows: list[dict[str, Any]],
    qc_rows: list[dict[str, Any]],
    decoder_factory: Callable[[str, Path], Any],
    video_partitions: Any,
    keyframe_partitions: Any,
    coarse_by_video: dict[str, SignalSeries] | None = None,
) -> dict[str, Any]:
    qc = {str(row["candidate_id"]): row for row in qc_rows}
    results = []
    audit_performance = {
        "additional_coarse_decode_ms": 0.0,
        "additional_coarse_signal_ms": 0.0,
        "local_decode_and_signal_ms": 0.0,
        "orb_ms": 0.0,
        "local_frames": 0,
    }
    baselines: dict[str, tuple[Any, Any]] = {
        video_id: (series.pixel_baseline, series.histogram_baseline)
        for video_id, series in (coarse_by_video or {}).items()
    }
    for row in sorted(old_rows, key=lambda item: str(item["candidate_id"])):
        video_id = str(row["video_id"])
        asset = resolve_assets(dataset, video_id, video_partitions, keyframe_partitions).video
        if video_id not in baselines:
            decoder = decoder_factory(video_id, asset)
            try:
                coarse = scan_coarse_video(decoder)
            finally:
                decoder.close()
            audit_performance["additional_coarse_decode_ms"] += coarse.decode_ms
            audit_performance["additional_coarse_signal_ms"] += coarse.signal_ms
            baselines[video_id] = (coarse.pixel_baseline, coarse.histogram_baseline)
        decoder = decoder_factory(video_id, asset)
        try:
            audit = final_continuity_audit(
                decoder,
                int(row["window_start_frame"]),
                int(row["window_end_frame"]),
                video_pixel_baseline=baselines[video_id][0],
                video_histogram_baseline=baselines[video_id][1],
            )
        finally:
            decoder.close()
        audit_performance["local_decode_and_signal_ms"] += audit.decode_ms + audit.signal_ms
        audit_performance["orb_ms"] += audit.orb_ms
        audit_performance["local_frames"] += len(audit.frame_indices)
        pre = (int(row["proposal_center_frame"]) - int(row["window_start_frame"])) / float(
            row["fps"]
        )
        post = (int(row["window_end_frame"]) - int(row["proposal_center_frame"])) / float(
            row["fps"]
        )
        item = qc[str(row["candidate_id"])]
        results.append(
            {
                "candidate_id": row["candidate_id"],
                "video_id": video_id,
                "qc_status": item["qc_status"],
                "reason_code": item.get("reason_code"),
                "auto_continuity_status": audit.status,
                "abrupt_global_replacement": bool(audit.abrupt_transition_frames),
                "soft_editorial_transition": bool(audit.soft_transition_frames),
                "geometry_violation": pre < MIN_PRE_CONTEXT_SECONDS
                or post < MIN_POST_CONTEXT_SECONDS,
            }
        )
    hard = [row for row in results if row["reason_code"] == "HARD_CUT"]
    usable = [row for row in results if row["qc_status"] == "USABLE"]
    caught = sum(row["auto_continuity_status"] == "AUTO_CONTINUITY_SCREEN_REJECT" for row in hard)
    false = sum(row["auto_continuity_status"] == "AUTO_CONTINUITY_SCREEN_REJECT" for row in usable)
    return {
        "audit_role": "POST_HOC_DIAGNOSTIC_NOT_OFFICIAL_GT",
        "threshold_tuning_from_audit": False,
        "old_candidate_count": len(results),
        "old_hard_cut_count": len(hard),
        "old_hard_cut_rejected": caught,
        "OLD_HARD_CUT_RECALL": caught / len(hard) if hard else None,
        "old_usable_count": len(usable),
        "old_usable_falsely_rejected": false,
        "OLD_USABLE_FALSE_VETO_RATE": false / len(usable) if usable else None,
        "EDGE_BUG_AFFECTED_OLD_CASES": sum(row["geometry_violation"] for row in results),
        "performance": audit_performance,
        "candidate_results": results,
    }


def _assign_base_scores(proposals: list[Proposal]) -> None:
    components = (
        ("transition_strength", 0.40),
        ("before_after_visual_difference", 0.25),
        ("spatial_activity_concentration", 0.20),
        ("overall_activity", 0.15),
    )
    for proposal in proposals:
        proposal.base_transition_score = 0.0
    for field_name, weight in components:
        ranks = empirical_percentiles(np.asarray([p.features[field_name] for p in proposals]))
        for proposal, rank in zip(proposals, ranks, strict=True):
            proposal.base_transition_score += weight * float(rank)


def _initial_proposals(
    *,
    video_id: str,
    fps: float,
    total_frames: int,
    coarse: SignalSeries,
    cuts: tuple[int, ...],
    source_origin: str,
    seeds: list[dict[str, Any]],
) -> tuple[list[Proposal], Counter[str]]:
    counts: Counter[str] = Counter()
    proposals = []
    center_step = max(1, int(round((0.5 * fps) / coarse.stride_frames)))
    activity_floor = max(0.004, 0.5 * coarse.pixel_baseline.median)
    for position in range(1, len(coarse.frame_indices) - 1, center_step):
        counts["raw_proposals"] += 1
        center = int(coarse.frame_indices[position])
        context, reason = context_window(
            center_frame=center, fps=fps, total_frames=total_frames, cut_frames=cuts
        )
        if context is None:
            counts[f"rejected_{reason.lower()}"] += 1
            continue
        try:
            features = center_relative_features(coarse, center, fps)
        except ValueError:
            counts["rejected_context_profile"] += 1
            continue
        if features["overall_activity"] < activity_floor:
            counts["rejected_activity_floor"] += 1
            continue
        duplicate_id = next(
            (
                str(seed["candidate_id"])
                for seed in seeds
                if is_seed_near_duplicate(
                    video_id=video_id,
                    start_frame=context.window.start_frame,
                    end_frame=context.window.end_frame,
                    center_frame=center,
                    fps=fps,
                    seeds=[seed],
                )
            ),
            None,
        )
        if duplicate_id is not None:
            counts["rejected_seed_duplicate"] += 1
            continue
        seed_overlaps = [
            (
                temporal_iou(
                    context.window.start_frame,
                    context.window.end_frame,
                    int(seed["window_start_frame"]),
                    int(seed["window_end_frame"]),
                ),
                str(seed["candidate_id"]),
            )
            for seed in seeds
            if str(seed["video_id"]) == video_id
        ]
        best_seed_overlap = max(seed_overlaps, default=(0.0, None))
        proposals.append(
            Proposal(
                video_id,
                fps,
                center,
                context,
                source_origin,
                features,
                overlapping_seed_id=(best_seed_overlap[1] if best_seed_overlap[0] > 0 else None),
            )
        )
    if proposals:
        _assign_base_scores(proposals)
    ordered = sorted(proposals, key=lambda p: (-p.base_transition_score, p.center_frame))
    shortlisted = []
    for proposal in ordered:
        if any(
            abs(proposal.center_frame - other.center_frame) / fps < 1.0 for other in shortlisted
        ):
            counts["rejected_pre_local_dedup"] += 1
        elif len(shortlisted) >= MAX_LOCAL_PER_VIDEO:
            counts["rejected_pre_local_budget"] += 1
        else:
            shortlisted.append(proposal)
    return shortlisted, counts


def _verify_and_select(
    decoder: Any, coarse: SignalSeries, proposals: list[Proposal]
) -> tuple[list[Proposal], Counter[str], dict[str, float | int]]:
    counts: Counter[str] = Counter()
    retained = []
    local_ms = orb_ms = 0.0
    local_frames = 0
    for proposal in proposals:
        audit = final_continuity_audit(
            decoder,
            proposal.context.window.start_frame,
            proposal.context.window.end_frame,
            video_pixel_baseline=coarse.pixel_baseline,
            video_histogram_baseline=coarse.histogram_baseline,
        )
        local_ms += audit.decode_ms + audit.signal_ms
        orb_ms += audit.orb_ms
        local_frames += len(audit.frame_indices)
        if audit.abrupt_transition_frames:
            counts["rejected_abrupt_global_replacement"] += 1
            continue
        if audit.soft_transition_frames:
            counts["rejected_soft_transition"] += 1
            continue
        proposal.audit = audit
        capped_before_after = proposal.features["before_after_visual_difference"] * (
            0.25 + 0.75 * audit.continuity_quality
        )
        proposal.diagnostics["continuity_capped_before_after"] = capped_before_after
        proposal.dense_focus_frame = safe_dense_focus(
            coarse,
            proposal.context.window.start_frame,
            proposal.context.window.end_frame,
            proposal.center_frame,
            proposal.fps,
        )
        retained.append(proposal)
    if retained:
        components = (
            ("transition_strength", 0.40),
            ("continuity_capped_before_after", 0.25),
            ("spatial_activity_concentration", 0.20),
            ("overall_activity", 0.15),
        )
        for proposal in retained:
            proposal.base_transition_score = 0.0
        for field_name, weight in components:
            values = np.asarray(
                [
                    proposal.diagnostics[field_name]
                    if field_name == "continuity_capped_before_after"
                    else proposal.features[field_name]
                    for proposal in retained
                ]
            )
            ranks = empirical_percentiles(values)
            for proposal, rank in zip(retained, ranks, strict=True):
                proposal.base_transition_score += weight * float(rank)
        for proposal in retained:
            assert proposal.audit is not None
            proposal.proposal_score = (
                proposal.base_transition_score * proposal.audit.continuity_quality
            )
    selected = []
    for proposal in sorted(retained, key=lambda p: (-p.proposal_score, p.center_frame)):
        if any(
            abs(proposal.center_frame - other.center_frame) / proposal.fps < NMS_SECONDS
            for other in selected
        ):
            counts["rejected_temporal_nms"] += 1
        elif len(selected) >= MAX_PER_VIDEO:
            counts["rejected_per_video_cap"] += 1
        else:
            selected.append(proposal)
    return (
        selected,
        counts,
        {
            "local_verification_ms": local_ms,
            "orb_ms": orb_ms,
            "local_verification_frames": local_frames,
        },
    )


def preflight_mb1_v022(config: MB1V022Config) -> dict[str, Any]:
    dataset = config.dataset_root.expanduser().resolve(strict=True)
    if config.output_root.exists():
        raise FileExistsError(f"MB1 v0.2.2 output already exists: {config.output_root}")
    seeds = resolve_seeds(config.v021_seed_manifest_path)
    v021 = _read_jsonl(config.v021_candidate_manifest_path)
    _read_jsonl(config.v021_candidate_diagnostics_path)
    _read_json(config.v021_selection_path)
    _read_json(config.v021_cut_audit_path)
    _read_jsonl(config.old_v02_candidate_manifest_path)
    if sha256_file(config.ai_qc_path) != EXPECTED_QC_SHA256:
        raise ValueError("MB1_V022_AI_QC_HASH_MISMATCH")
    _read_jsonl(config.rt2_benchmark_path)
    pool = build_trusted_source_pool(seeds, v021)
    video_partitions, keyframe_partitions = discover_layout(dataset)
    for row in pool:
        resolve_assets(dataset, row["video_id"], video_partitions, keyframe_partitions)
    return {
        "status": "READY",
        "mode": MB1_V022_MODE,
        "frozen_seed_count": len(seeds),
        "trusted_source_count": len(pool),
        "trusted_sources": pool,
        "target_new_range": [MIN_TARGET_NEW, MAX_TARGET_NEW],
        "preferred_new": TARGET_NEW,
        "model_inference_required": False,
        "semantic_interval_gt_required": False,
        "network_required": False,
    }


def _readme() -> str:
    return """# TRIAGE-EG MB1 v0.2.2 AI QC

This pack contains model-free candidate evidence. `AUTO_CONTINUITY_SCREEN_PASS` does
NOT imply verified absence of editorial cuts. Automatic continuity screening is only
a prefilter; final candidate quality requires GPT-5.6 Sol AI QC.

Review every retained candidate and assign `USABLE`, `CONDITIONAL`, or `REJECT`.
Potential rejection reasons: `HARD_CUT`, `SOFT_CUT`, `CAMERA_MOTION`, `STATIC_STATE`,
`GRAPHIC_TRANSITION`, `NO_DEFENSIBLE_TEMPORAL_CHANGE`, `REPETITIVE_AMBIGUOUS_ACTION`,
`OCCLUSION`, `INSUFFICIENT_BEFORE_AFTER`, or `OTHER`.

No semantic moment type, query, preferred frame, or acceptable interval was assigned.
"""


def prepare_mb1_v022_candidates(
    config: MB1V022Config, *, decoder_factory: Callable[[str, Path], Any] = OpenCVRawVideoDecoder
) -> dict[str, Any]:
    started = monotonic()
    preflight = preflight_mb1_v022(config)
    dataset = config.dataset_root.expanduser().resolve(strict=True)
    output = config.output_root.expanduser().resolve(strict=False)
    output.mkdir(parents=True)
    seeds = resolve_seeds(config.v021_seed_manifest_path)
    v021_rows = _read_jsonl(config.v021_candidate_manifest_path)
    old_rows = _read_jsonl(config.old_v02_candidate_manifest_path)
    qc_rows = _read_jsonl(config.ai_qc_path)
    video_partitions, keyframe_partitions = discover_layout(dataset)
    geometry = v021_geometry_audit(v021_rows)
    write_json(output / "v021_geometry_audit.json", geometry)
    all_selected = []
    counts: Counter[str] = Counter()
    performance = []
    assets = {}
    coarse_by_video = {}
    for source in preflight["trusted_sources"]:
        video_id = source["video_id"]
        asset = resolve_assets(dataset, video_id, video_partitions, keyframe_partitions).video
        assets[video_id] = asset
        decoder = decoder_factory(video_id, asset)
        try:
            coarse = scan_coarse_video(decoder)
        finally:
            decoder.close()
        coarse_by_video[video_id] = coarse
        performance.append(
            {
                "video_id": video_id,
                "coarse_decode_ms": coarse.decode_ms,
                "signal_ms": coarse.signal_ms,
                "coarse_frames": len(coarse.frame_indices),
                "retained_before_global": 0,
            }
        )
    old_audit = _old_qc_audit(
        dataset=dataset,
        old_rows=old_rows,
        qc_rows=qc_rows,
        decoder_factory=decoder_factory,
        video_partitions=video_partitions,
        keyframe_partitions=keyframe_partitions,
        coarse_by_video=coarse_by_video,
    )
    write_json(output / "old_qc_continuity_audit_v022.json", old_audit)
    for source in preflight["trusted_sources"]:
        video_id = source["video_id"]
        decoder = decoder_factory(video_id, assets[video_id])
        try:
            coarse = coarse_by_video[video_id]
            cut_mask = hard_cut_mask(coarse.pixel_percentiles, coarse.histogram_percentiles)
            cuts = tuple(
                int(value)
                for value in coarse.frame_indices[np.flatnonzero(cut_mask)]
                if 0 < int(value) < decoder.info.total_frames
            )
            proposals, initial_counts = _initial_proposals(
                video_id=video_id,
                fps=float(decoder.info.fps),
                total_frames=int(decoder.info.total_frames),
                coarse=coarse,
                cuts=cuts,
                source_origin=source["source_pool_origin"],
                seeds=seeds,
            )
            selected, verify_counts, timing = _verify_and_select(decoder, coarse, proposals)
            all_selected.extend(selected)
            initial_counts.update(verify_counts)
            counts.update(initial_counts)
            performance_row = next(row for row in performance if row["video_id"] == video_id)
            performance_row.update(timing)
            performance_row["retained_before_global"] = len(selected)
        finally:
            decoder.close()
    ranked = sorted(all_selected, key=lambda p: (-p.proposal_score, p.video_id, p.center_frame))
    bounded = ranked[:MAX_TARGET_NEW]
    if len(bounded) > TARGET_NEW:
        counts["rejected_global_preferred_target"] += len(bounded) - TARGET_NEW
        bounded = bounded[:TARGET_NEW]
    ordered = sorted(
        bounded, key=lambda p: (p.video_id, p.context.window.start_frame, p.center_frame)
    )
    manifest = []
    diagnostics = []
    rendered = 0
    for number, proposal in enumerate(ordered, 1):
        assert proposal.audit is not None and proposal.dense_focus_frame is not None
        cid = f"mb1v022_c{number:03d}"
        window = proposal.context.window
        overview_ids = overview_displayed_frames(window.start_frame, window.end_frame)
        dense_ids = dense_displayed_frames(
            window.start_frame, window.end_frame, proposal.dense_focus_frame, proposal.fps
        )
        requested = sorted(set(overview_ids + dense_ids))
        decoder = decoder_factory(proposal.video_id, assets[proposal.video_id])
        render_started = monotonic()
        try:
            frames = _decode_resized_frames(decoder, requested)
        finally:
            decoder.close()
        by_id = {int(frame.actual_frame_idx): frame for frame in frames}
        overview_path = Path("overview") / f"{cid}.jpg"
        dense_path = Path("dense") / f"{cid}.jpg"
        if (
            render_contact_sheet(
                output / overview_path,
                candidate_id=cid,
                video_id=proposal.video_id,
                view_name="OVERVIEW",
                fps=proposal.fps,
                frames=[by_id[x] for x in overview_ids],
                quality=config.jpeg_quality,
            )
            != overview_ids
        ):
            raise RuntimeError("MB1_V022_OVERVIEW_MAPPING_MISMATCH")
        if (
            render_contact_sheet(
                output / dense_path,
                candidate_id=cid,
                video_id=proposal.video_id,
                view_name="DENSE",
                fps=proposal.fps,
                frames=[by_id[x] for x in dense_ids],
                quality=config.jpeg_quality,
            )
            != dense_ids
        ):
            raise RuntimeError("MB1_V022_DENSE_MAPPING_MISMATCH")
        render_ms = (monotonic() - render_started) * 1000
        next(row for row in performance if row["video_id"] == proposal.video_id).setdefault(
            "render_ms", 0.0
        )
        next(row for row in performance if row["video_id"] == proposal.video_id)["render_ms"] += (
            render_ms
        )
        rendered += len(requested)
        c = proposal.context
        manifest.append(
            {
                "candidate_id": cid,
                "video_id": proposal.video_id,
                "fps": proposal.fps,
                "window_start_frame": window.start_frame,
                "window_end_frame": window.end_frame,
                "final_window_duration_seconds": window.final_duration_seconds,
                "proposal_center_frame": proposal.center_frame,
                "dense_focus_frame": proposal.dense_focus_frame,
                "pre_context_seconds": c.pre_context_seconds,
                "post_context_seconds": c.post_context_seconds,
                "previous_cut_frame": c.previous_cut_frame,
                "next_cut_frame": c.next_cut_frame,
                "distance_to_previous_cut_seconds": (
                    (proposal.center_frame - c.previous_cut_frame) / proposal.fps
                    if c.previous_cut_frame is not None
                    else None
                ),
                "distance_to_next_cut_seconds": (
                    (c.next_cut_frame - proposal.center_frame) / proposal.fps
                    if c.next_cut_frame is not None
                    else None
                ),
                "proposal_score": proposal.proposal_score,
                "base_transition_score": proposal.base_transition_score,
                "continuity_quality": proposal.audit.continuity_quality,
                "orb_continuity_mean": proposal.audit.orb_continuity_mean,
                "orb_continuity_min": proposal.audit.orb_continuity_min,
                "orb_available_fraction": proposal.audit.orb_available_fraction,
                "maximum_local_pixel_jump": proposal.audit.maximum_local_pixel_jump,
                "maximum_local_histogram_jump": proposal.audit.maximum_local_histogram_jump,
                "auto_continuity_status": proposal.audit.status,
                "overview_displayed_frames": overview_ids,
                "dense_displayed_frames": dense_ids,
                "overview_sheet_path": overview_path.as_posix(),
                "dense_sheet_path": dense_path.as_posix(),
                "source_pool_origin": proposal.source_pool_origin,
                "overlapping_seed_id": proposal.overlapping_seed_id,
            }
        )
        diagnostics.append(
            {
                "candidate_id": cid,
                "video_id": proposal.video_id,
                **proposal.features,
                "context_geometry": asdict(c),
                "base_transition_score": proposal.base_transition_score,
                "continuity_quality": proposal.audit.continuity_quality,
                "final_proposal_score": proposal.proposal_score,
                "before_after_continuity_cap": proposal.diagnostics[
                    "continuity_capped_before_after"
                ],
                "local_change": {
                    "pixel_z": proposal.audit.pixel_local_z,
                    "histogram_z": proposal.audit.histogram_local_z,
                    "video_pixel_z": proposal.audit.pixel_video_z,
                    "video_histogram_z": proposal.audit.histogram_video_z,
                    "maximum_pixel_jump": proposal.audit.maximum_local_pixel_jump,
                    "maximum_histogram_jump": proposal.audit.maximum_local_histogram_jump,
                },
                "orb": {
                    "available_fraction": proposal.audit.orb_available_fraction,
                    "continuity_mean": proposal.audit.orb_continuity_mean,
                    "continuity_min": proposal.audit.orb_continuity_min,
                    "transitions": [asdict(row) for row in proposal.audit.orb_transitions],
                },
                "hard_cut_evidence": list(proposal.audit.abrupt_transition_frames),
                "soft_transition_evidence": list(proposal.audit.soft_transition_frames),
                "auto_continuity_status": proposal.audit.status,
            }
        )
    per_video = dict(sorted(Counter(row["video_id"] for row in manifest).items()))
    structural_errors = []
    seed_duplicate_ids = []
    for row in manifest:
        if row["proposal_center_frame"] in {
            row["window_start_frame"],
            row["window_end_frame"],
        }:
            structural_errors.append(f"{row['candidate_id']}:center_at_edge")
        if row["pre_context_seconds"] < MIN_PRE_CONTEXT_SECONDS:
            structural_errors.append(f"{row['candidate_id']}:pre_context")
        if row["post_context_seconds"] < MIN_POST_CONTEXT_SECONDS:
            structural_errors.append(f"{row['candidate_id']}:post_context")
        if not 3.0 <= row["final_window_duration_seconds"] <= 4.05:
            structural_errors.append(f"{row['candidate_id']}:window_duration")
        if row["auto_continuity_status"] != "AUTO_CONTINUITY_SCREEN_PASS":
            structural_errors.append(f"{row['candidate_id']}:continuity_status")
        if is_seed_near_duplicate(
            video_id=row["video_id"],
            start_frame=row["window_start_frame"],
            end_frame=row["window_end_frame"],
            center_frame=row["proposal_center_frame"],
            fps=row["fps"],
            seeds=seeds,
        ):
            seed_duplicate_ids.append(row["candidate_id"])
        dense_margin = min(
            (row["dense_focus_frame"] - row["window_start_frame"]) / row["fps"],
            (row["window_end_frame"] - row["dense_focus_frame"]) / row["fps"],
        )
        if dense_margin < DENSE_EDGE_CONTEXT_SECONDS:
            structural_errors.append(f"{row['candidate_id']}:dense_focus")
        for key in ("overview_displayed_frames", "dense_displayed_frames"):
            values = row[key]
            if values != sorted(set(values)) or not all(
                row["window_start_frame"] <= value <= row["window_end_frame"] for value in values
            ):
                structural_errors.append(f"{row['candidate_id']}:{key}")
    if any(value > MAX_PER_VIDEO for value in per_video.values()):
        structural_errors.append("per_video_cap")
    if seed_duplicate_ids:
        structural_errors.append(f"seed_duplicates:{seed_duplicate_ids}")
    if structural_errors:
        raise RuntimeError(f"MB1_V022_STRUCTURAL_GATE_FAILED: {structural_errors}")
    issues = []
    if len(manifest) < MIN_TARGET_NEW:
        issues.append(
            {
                "severity": "WARNING",
                "code": "BELOW_PREFERRED_CANDIDATE_RANGE",
                "message": "Continuity/context gates were not relaxed.",
                "retained": len(manifest),
            }
        )
    selection = {
        "experiment": "MB1_V022",
        "version": MB1_V022_VERSION,
        "mode": MB1_V022_MODE,
        "trusted_seed_videos": [
            row["video_id"]
            for row in preflight["trusted_sources"]
            if row["source_pool_origin"] == "FROZEN_USABLE_SEED_VIDEO"
        ],
        "additional_prequalified_action_rich_videos": [
            row["video_id"]
            for row in preflight["trusted_sources"]
            if row["source_pool_origin"] != "FROZEN_USABLE_SEED_VIDEO"
        ],
        "trusted_source_count": len(preflight["trusted_sources"]),
        "existing_seed_count": len(seeds),
        "raw_proposals": counts["raw_proposals"],
        "rejected": {
            "insufficient_PRE_context": counts["rejected_insufficient_pre_context"],
            "insufficient_POST_context": counts["rejected_insufficient_post_context"],
            "known_cut_safety_margin": counts["rejected_known_cut_safety_margin"],
            "abrupt_global_replacement": counts["rejected_abrupt_global_replacement"],
            "soft_transition": counts["rejected_soft_transition"],
            "seed_duplicate": counts["rejected_seed_duplicate"],
            "temporal_NMS": counts["rejected_temporal_nms"],
            "per_video_cap": counts["rejected_per_video_cap"],
            "activity_floor": counts["rejected_activity_floor"],
            "context_profile": counts["rejected_context_profile"],
            "center_at_boundary": counts["rejected_center_at_boundary"],
            "pre_local_dedup": counts["rejected_pre_local_dedup"],
            "pre_local_budget": counts["rejected_pre_local_budget"],
            "global_preferred_target": counts["rejected_global_preferred_target"],
        },
        "retained_NEW_candidates": len(manifest),
        "combined_potential_pool": len(seeds) + len(manifest),
        "per_video_retained_counts": per_video,
        "context_rule": {
            "minimum_pre_seconds": MIN_PRE_CONTEXT_SECONDS,
            "minimum_post_seconds": MIN_POST_CONTEXT_SECONDS,
            "allowed_adjustments": [
                "NONE",
                "SYMMETRIC_SHRINK",
                "SMALL_SHIFT_WITH_CONTEXT_PRESERVED",
            ],
        },
        "cut_semantics": {
            "coordinate": "c means transition c-1 -> c",
            "safety_margin_seconds": CUT_SAFETY_MARGIN_SECONDS,
        },
        "profile_semantics": "PRE/CENTER/POST are proposal-center-relative",
        "score_formula": (
            "final_proposal_score = base_transition_score * continuity_quality; "
            "before/after contribution is continuity-capped"
        ),
        "orb_settings": {
            "nfeatures": 500,
            "fast_threshold": 12,
            "cross_check": True,
            "max_hamming_distance": 48,
            "minimum_descriptors": 12,
        },
        "nms_seconds": NMS_SECONDS,
        "per_video_cap": MAX_PER_VIDEO,
        "automatic_status_disclaimer": (
            "AUTO_CONTINUITY_SCREEN_PASS does not imply verified absence of editorial cuts."
        ),
    }
    run_manifest = {
        "experiment": "MB1_V022",
        "version": MB1_V022_VERSION,
        "mode": MB1_V022_MODE,
        "created_at": datetime.now(UTC).isoformat(),
        "build_git_commit": config.build_git_commit,
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(config).items()},
        "preflight": preflight,
        "input_hashes": {
            "v021_seed_manifest": sha256_file(config.v021_seed_manifest_path),
            "v021_candidate_manifest": sha256_file(config.v021_candidate_manifest_path),
            "v021_candidate_diagnostics": sha256_file(config.v021_candidate_diagnostics_path),
            "v021_selection": sha256_file(config.v021_selection_path),
            "v021_cut_audit": sha256_file(config.v021_cut_audit_path),
            "old_v02_manifest": sha256_file(config.old_v02_candidate_manifest_path),
            "ai_qc": sha256_file(config.ai_qc_path),
            "rt2_benchmark": sha256_file(config.rt2_benchmark_path),
        },
        "model_inference": False,
        "semantic_interval_fields_consumed": False,
        "raw_frame_coordinate_source": "M1_OPENCV_RAW_VIDEO_DECODER_ACTUAL_FRAME_IDX",
        "raw_frames_added_to_stage1_or_framemap": False,
        "network_required": False,
        "performance": {
            "videos": performance,
            "old_qc_audit": old_audit["performance"],
            "coarse_decode_ms": sum(x["coarse_decode_ms"] for x in performance),
            "signal_ms": sum(x["signal_ms"] for x in performance),
            "local_verification_ms": sum(x["local_verification_ms"] for x in performance),
            "orb_ms": sum(x["orb_ms"] for x in performance),
            "render_ms": sum(x.get("render_ms", 0) for x in performance),
            "old_qc_audit_ms": sum(
                float(value)
                for key, value in old_audit["performance"].items()
                if key.endswith("_ms")
            ),
            "total_ms": (monotonic() - started) * 1000,
            "full_resolution_rendered_frames": rendered,
        },
        "quality_gates": {
            "proposal_centers_not_at_edges": all(
                r["proposal_center_frame"] not in {r["window_start_frame"], r["window_end_frame"]}
                for r in manifest
            ),
            "minimum_pre_context": all(
                r["pre_context_seconds"] >= MIN_PRE_CONTEXT_SECONDS for r in manifest
            ),
            "minimum_post_context": all(
                r["post_context_seconds"] >= MIN_POST_CONTEXT_SECONDS for r in manifest
            ),
            "dense_focus_safe": all(
                min(
                    (r["dense_focus_frame"] - r["window_start_frame"]) / r["fps"],
                    (r["window_end_frame"] - r["dense_focus_frame"]) / r["fps"],
                )
                >= DENSE_EDGE_CONTEXT_SECONDS
                for r in manifest
            ),
            "display_frames_valid": True,
            "seed_duplicate_count": len(seed_duplicate_ids),
            "per_video_cap_respected": all(v <= MAX_PER_VIDEO for v in per_video.values()),
            "semantic_gt_leakage": False,
        },
        "MB1_V022_REAL_STATUS": "COMPLETE",
        "MB1_V022_AI_QC_STATUS": "WAITING_FOR_AI",
        "M3_IMPLEMENTATION_STATUS": "NOT_STARTED",
    }
    write_jsonl(output / "mb1_v022_candidate_manifest.jsonl", manifest)
    write_jsonl(output / "mb1_v022_candidate_diagnostics.jsonl", diagnostics)
    write_json(output / "candidate_selection_v022.json", selection)
    (output / "README_AI_QC_V022.md").write_text(_readme(), encoding="utf-8")
    write_json(output / "run_manifest.json", run_manifest)
    write_jsonl(output / "issues.jsonl", issues)
    _render_montages(output, manifest, "overview")
    _render_montages(output, manifest, "dense")
    return {
        "selection": selection,
        "old_qc_audit": old_audit,
        "v021_geometry_audit": geometry,
        "run_manifest": run_manifest,
        "issues": issues,
    }


def create_mb1_v022_bundle(output_root: Path, zip_path: Path) -> Path:
    source = output_root.expanduser().resolve(strict=True)
    target = zip_path.expanduser().resolve(strict=False)
    manifest = _read_jsonl(source / "mb1_v022_candidate_manifest.jsonl")
    members = [source / name for name in BUNDLE_BASE_FILES]
    members += [
        source / row[key] for row in manifest for key in ("overview_sheet_path", "dense_sheet_path")
    ]
    members += sorted((source / "montages").glob("*.jpg"))
    if any(not path.is_file() for path in members) or any(
        path.suffix.lower() in FORBIDDEN_SUFFIXES for path in members
    ):
        raise ValueError("MB1 v0.2.2 bundle allowlist validation failed")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.with_suffix(target.suffix + ".building")
    staging.unlink(missing_ok=True)
    with ZipFile(staging, "w", ZIP_DEFLATED) as archive:
        for path in sorted(set(members), key=lambda x: x.relative_to(source).as_posix()):
            archive.write(path, path.relative_to(source).as_posix())
    shutil.move(staging, target)
    return target
