"""MB1 v0.2 boundary-rich, model-free semantic-moment candidate preparation."""

from __future__ import annotations

import hashlib
import json
import math
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
from triage_eg.experiments.moment_m1 import DecodedFrame, OpenCVRawVideoDecoder
from triage_eg.retrieval.stage1b.writers import write_json, write_jsonl

from .signals import (
    DENSE_FRAME_COUNT,
    DENSE_SECONDS,
    MAX_CANDIDATES_PER_VIDEO,
    NMS_SECONDS,
    OVERVIEW_FRAME_COUNT,
    SCAN_SAMPLES_PER_SECOND,
    SCAN_SIZE,
    SHOT_MARGIN_SECONDS,
    WINDOW_SECONDS,
    CandidateProposal,
    continuous_shot_segments,
    dense_displayed_frames,
    detect_hard_cuts,
    overview_displayed_frames,
    propose_active_windows,
    scan_low_resolution_video,
    window_contains_detected_cut,
)

MB1_V02_VERSION = "0.2.0"
MB1_V02_MODE = "BOUNDARY_RICH_SEMANTIC_MOMENT_CANDIDATE_PREPARATION_MODE_A"
TARGET_CANDIDATES = 60
MIN_ACCEPTABLE_CANDIDATES = 48
MAX_CANDIDATES = 72
TARGET_SOURCE_MIN = 16
TARGET_SOURCE_MAX = 24
SECONDARY_TAGS = frozenset(
    {"PROCEDURAL", "SPORTS", "CONTINUOUS_SEQUENCE", "VISUALLY_REPETITIVE"}
)
ALLOWED_USABILITY = frozenset({"USABLE", "CONDITIONAL", "REJECT"})
BUNDLE_FILES = (
    "mb1_v02_candidate_manifest.jsonl",
    "mb1_v02_candidate_diagnostics.jsonl",
    "candidate_selection.json",
    "annotation_schema_v02.json",
    "README_AI_QC_ANNOTATION.md",
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
class MB1V02Config:
    dataset_root: Path
    prior_candidate_manifest_path: Path
    prior_candidate_qc_path: Path
    output_root: Path
    rt2_benchmark_path: Path | None = None
    prior_selection_path: Path | None = None
    seed: int = 2026
    jpeg_quality: int = 88
    build_git_commit: str | None = None

    def __post_init__(self) -> None:
        if self.seed != 2026:
            raise ValueError("MB1 v0.2 seed is frozen at 2026")
        if not 70 <= self.jpeg_quality <= 95:
            raise ValueError("MB1 v0.2 JPEG quality must be between 70 and 95")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def build_source_video_pool(
    prior_manifest_rows: list[dict[str, Any]],
    qc_rows: list[dict[str, Any]],
    rt2_rows: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Use USABLE MB1 videos first, then only existing eligible RT2 tags."""

    candidates = {str(row["candidate_id"]): row for row in prior_manifest_rows}
    if len(candidates) != len(prior_manifest_rows):
        raise ValueError("Prior MB1 candidate IDs must be unique")
    primary: set[str] = set()
    for row in qc_rows:
        usability = str(row.get("usability"))
        if usability not in ALLOWED_USABILITY:
            raise ValueError("Prior MB1 QC usability is invalid")
        candidate = candidates.get(str(row.get("candidate_id")))
        if candidate is None:
            raise ValueError("Prior MB1 QC references an unknown candidate")
        if str(row.get("video_id")) != str(candidate.get("video_id")):
            raise ValueError("Prior MB1 QC candidate/video identity mismatch")
        if usability == "USABLE":
            primary.add(str(row["video_id"]))
    pool = [
        {"video_id": video_id, "source_pool_origin": "MB1_V01_USABLE_VIDEO"}
        for video_id in sorted(primary)
    ]
    if rt2_rows:
        secondary = []
        for row in rt2_rows:
            video_id = str(row["source_video_id"])
            tags = tuple(str(value) for value in row.get("difficulty_tags", []))
            if video_id not in primary and SECONDARY_TAGS.intersection(tags):
                secondary.append(
                    {
                        "video_id": video_id,
                        "source_pool_origin": "RT2_EXISTING_TAGGED_VIDEO",
                        "existing_tags": list(tags),
                    }
                )
        pool.extend(sorted(secondary, key=lambda item: item["video_id"]))
    unique = {row["video_id"]: row for row in pool}
    if len(unique) != len(pool):
        raise RuntimeError("MB1 v0.2 source pool is not unique")
    return pool


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


def render_contact_sheet(
    path: Path,
    *,
    candidate_id: str,
    video_id: str,
    view_name: str,
    fps: float,
    frames: list[DecodedFrame],
    quality: int,
) -> list[int]:
    """Render chronological visual evidence with only raw identities and timestamps."""

    from PIL import Image, ImageDraw, ImageOps

    frame_ids = [int(frame.actual_frame_idx) for frame in frames]
    if frame_ids != sorted(set(frame_ids)):
        raise ValueError("MB1 v0.2 rendered frames must be unique and chronological")
    columns = 4
    rows = int(math.ceil(len(frames) / columns))
    tile_width, image_height, label_height, header_height = 320, 180, 38, 64
    sheet = Image.new(
        "RGB",
        (columns * tile_width, header_height + rows * (image_height + label_height)),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    draw.text(
        (12, 10),
        f"{candidate_id} | {view_name} | video_id={video_id} | chronological raw frames",
        fill="black",
        font=_font(19),
    )
    for slot, frame in enumerate(frames):
        x = (slot % columns) * tile_width
        y = header_height + (slot // columns) * (image_height + label_height)
        image = Image.fromarray(np.asarray(frame.image, dtype=np.uint8), mode="RGB")
        fitted = ImageOps.fit(image, (tile_width, image_height), method=Image.Resampling.LANCZOS)
        sheet.paste(fitted, (x, y))
        draw.text(
            (x + 5, y + image_height + 5),
            f"actual_frame_idx={frame.actual_frame_idx}  t={frame.actual_frame_idx / fps:.3f}s",
            fill="black",
            font=_font(15),
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, format="JPEG", quality=quality, optimize=True)
    return frame_ids


def annotation_schema_v02() -> dict[str, Any]:
    moment_types = [
        "FIRST_OCCURRENCE",
        "LAST_OCCURRENCE",
        "TRANSITION_ONSET",
        "TRANSITION_OFFSET",
        "CONTACT",
        "SEPARATION",
        "EXTREMUM",
        "ACTION_VISIBILITY",
        "STATE",
    ]
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "TRIAGE-EG MB1 v0.2 semantic moment annotation",
        "type": "object",
        "additionalProperties": False,
        "required": [
            "moment_id",
            "video_id",
            "source_candidate_id",
            "query_text",
            "moment_definition",
            "moment_family",
            "moment_type",
            "acceptable_start_frame",
            "acceptable_end_frame",
            "annotation_confidence",
            "generator",
            "human_reviewed",
        ],
        "properties": {
            "moment_id": {"type": "string", "minLength": 1},
            "video_id": {"type": "string", "pattern": "^L[0-9]+_V[0-9]+$"},
            "source_candidate_id": {
                "type": "string",
                "pattern": "^mb1v02_c[0-9]{3}$",
            },
            "query_text": {"type": "string", "minLength": 1},
            "moment_definition": {"type": "string", "minLength": 1},
            "moment_family": {
                "enum": [
                    "SEMANTIC_ONSET",
                    "RELATIONAL_TRANSITION",
                    "TRAJECTORY_EXTREMUM",
                    "ACTION_VISIBILITY_CONTROL",
                ]
            },
            "moment_type": {"enum": moment_types},
            "acceptable_start_frame": {"type": "integer", "minimum": 0},
            "acceptable_end_frame": {"type": "integer", "minimum": 0},
            "preferred_frame": {"type": ["integer", "null"], "minimum": 0},
            "before_evidence_frame": {"type": ["integer", "null"], "minimum": 0},
            "after_evidence_frame": {"type": ["integer", "null"], "minimum": 0},
            "annotation_confidence": {"enum": ["HIGH", "MEDIUM", "LOW"]},
            "generator": {"const": "GPT-5.6 Sol"},
            "human_reviewed": {"const": False},
        },
        "x-invariants": [
            "acceptable_start_frame <= acceptable_end_frame",
            (
                "preferred_frame is null or acceptable_start_frame <= preferred_frame "
                "<= acceptable_end_frame"
            ),
            "before_evidence_frame should normally precede the interval when applicable",
            "after_evidence_frame should normally follow the interval when applicable",
            "before/after evidence is optional when not semantically applicable",
        ],
    }


def ai_qc_readme() -> str:
    return """# TRIAGE-EG MB1 v0.2 AI QC and annotation instructions

This pack contains model-free, chronological visual evidence. It contains no semantic labels,
ground-truth intervals, CLIP scores, or preferred heuristic frames. Each candidate has an
overview sheet for before/during/after context and a dense sheet for frame-level inspection.

## PASS 1 - Candidate usability QC

Assign exactly one: `USABLE`, `CONDITIONAL`, or `REJECT`.

Allowed rejection reasons: `HARD_CUT`, `NO_DEFENSIBLE_TEMPORAL_CHANGE`, `STATIC_STATE`,
`TOO_AMBIGUOUS`, `OCCLUDED`, `INSUFFICIENT_BEFORE_AFTER_EVIDENCE`, or `OTHER`.

Reject editorial cuts as semantic transitions. Use only the printed `actual_frame_idx` values.

## PASS 2 - Semantic moment annotation

Only annotate visually defensible candidates using `annotation_schema_v02.json`. Prefer
boundary-rich moments, but do not fabricate a type to meet a quota. Intervals must satisfy
`acceptable_start_frame <= acceptable_end_frame`; any preferred frame must lie inside it.
Before/after evidence frames are optional when not semantically applicable. Keep
`generator="GPT-5.6 Sol"` and `human_reviewed=false`.

Soft coverage goals are 8-10 onset, 6-8 first occurrence, 8-10 contact, 8-10 separation,
6-8 offset/last, 6-8 extremum, plus only a small action/state control slice. Evidence quality
takes priority over counts.
"""


def preflight_mb1_v02(config: MB1V02Config) -> dict[str, Any]:
    dataset = config.dataset_root.expanduser().resolve(strict=True)
    manifest_rows = _read_jsonl(config.prior_candidate_manifest_path)
    qc_rows = _read_jsonl(config.prior_candidate_qc_path)
    rt2_rows = (
        _read_jsonl(config.rt2_benchmark_path)
        if config.rt2_benchmark_path is not None
        else None
    )
    pool = build_source_video_pool(manifest_rows, qc_rows, rt2_rows)
    if config.output_root.exists():
        raise FileExistsError(f"MB1 v0.2 output already exists: {config.output_root}")
    video_partitions, keyframe_partitions = discover_layout(dataset)
    missing = []
    for row in pool:
        assets = resolve_assets(
            dataset, row["video_id"], video_partitions, keyframe_partitions
        )
        if not assets.video.is_file():
            missing.append(row["video_id"])
    if missing:
        raise FileNotFoundError(f"MB1 v0.2 raw videos missing: {missing}")
    return {
        "status": "READY",
        "mode": MB1_V02_MODE,
        "source_video_count": len(pool),
        "source_videos": pool,
        "target_source_range": [TARGET_SOURCE_MIN, TARGET_SOURCE_MAX],
        "target_candidate_count": TARGET_CANDIDATES,
        "acceptable_candidate_range": [MIN_ACCEPTABLE_CANDIDATES, MAX_CANDIDATES],
        "maximum_possible_with_pool_and_cap": len(pool) * MAX_CANDIDATES_PER_VIDEO,
        "model_inference_required": False,
        "network_required": False,
        "previous_semantic_intervals_required": False,
    }


DecoderFactory = Callable[[str, Path], Any]


def prepare_mb1_v02_candidates(
    config: MB1V02Config,
    *,
    decoder_factory: DecoderFactory = OpenCVRawVideoDecoder,
) -> dict[str, Any]:
    overall_started = monotonic()
    source_started = monotonic()
    source_pool_for_timing = build_source_video_pool(
        _read_jsonl(config.prior_candidate_manifest_path),
        _read_jsonl(config.prior_candidate_qc_path),
        _read_jsonl(config.rt2_benchmark_path)
        if config.rt2_benchmark_path is not None
        else None,
    )
    source_pool_ms = (monotonic() - source_started) * 1000
    preflight = preflight_mb1_v02(config)
    if source_pool_for_timing != preflight["source_videos"]:
        raise RuntimeError("MB1_V02_SOURCE_POOL_NOT_DETERMINISTIC")
    dataset = config.dataset_root.expanduser().resolve(strict=True)
    output = config.output_root.expanduser().resolve(strict=False)
    output.mkdir(parents=True)
    video_partitions, keyframe_partitions = discover_layout(dataset)
    source_origin = {
        row["video_id"]: row["source_pool_origin"] for row in preflight["source_videos"]
    }
    all_proposals: list[CandidateProposal] = []
    per_video_scan: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []

    for source in preflight["source_videos"]:
        video_id = str(source["video_id"])
        assets = resolve_assets(dataset, video_id, video_partitions, keyframe_partitions)
        decoder = decoder_factory(video_id, assets.video)
        try:
            series = scan_low_resolution_video(decoder)
            segmentation_started = monotonic()
            cut_frames, cut_rule = detect_hard_cuts(series)
            segments = continuous_shot_segments(decoder.info.total_frames, cut_frames)
            segmentation_ms = (monotonic() - segmentation_started) * 1000
            selection_started = monotonic()
            proposals, selection_counts = propose_active_windows(
                video_id,
                float(decoder.info.fps),
                int(decoder.info.total_frames),
                series,
                cut_frames,
                segments,
            )
            selection_ms = (monotonic() - selection_started) * 1000
            all_proposals.extend(proposals)
            per_video_scan.append(
                {
                    "video_id": video_id,
                    "source_pool_origin": source["source_pool_origin"],
                    "fps": float(decoder.info.fps),
                    "total_frames": int(decoder.info.total_frames),
                    "scan_stride_frames": series.scan_stride_frames,
                    "scan_frame_count": len(series.frame_indices),
                    "detected_cut_count": len(cut_frames),
                    "detected_cut_frames": list(cut_frames),
                    "continuous_shot_count": len(segments),
                    "shot_boundary_rule": cut_rule,
                    "selection_counts": selection_counts,
                    "timings_ms": {
                        "scan_decode_ms": series.scan_decode_ms,
                        "activity_computation_ms": series.activity_computation_ms,
                        "shot_segmentation_ms": segmentation_ms,
                        "candidate_selection_ms": selection_ms,
                        "sheet_rendering_ms": 0.0,
                    },
                }
            )
        finally:
            decoder.close()

    ordered = sorted(
        all_proposals,
        key=lambda item: (item.video_id, item.window_start_frame, item.proposal_center_frame),
    )[:MAX_CANDIDATES]
    scan_by_video = {row["video_id"]: row for row in per_video_scan}
    manifest_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    rendered_full_resolution_frames = 0
    for number, proposal in enumerate(ordered, 1):
        candidate_id = f"mb1v02_c{number:03d}"
        overview_ids = overview_displayed_frames(proposal)
        dense_ids = dense_displayed_frames(proposal)
        requested = sorted(set(overview_ids + dense_ids))
        assets = resolve_assets(
            dataset, proposal.video_id, video_partitions, keyframe_partitions
        )
        decoder = decoder_factory(proposal.video_id, assets.video)
        render_started = monotonic()
        try:
            frames = decoder.decode_indices(requested)
        finally:
            decoder.close()
        if [int(frame.actual_frame_idx) for frame in frames] != requested:
            raise RuntimeError("MB1_V02_RENDER_FRAME_IDENTITY_MISMATCH")
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
            view_name="DENSE_CENTER",
            fps=proposal.fps,
            frames=[by_id[value] for value in dense_ids],
            quality=config.jpeg_quality,
        )
        if overview_labels != overview_ids or dense_labels != dense_ids:
            raise RuntimeError("MB1_V02_MANIFEST_SHEET_LABEL_MISMATCH")
        render_ms = (monotonic() - render_started) * 1000
        scan_by_video[proposal.video_id]["timings_ms"]["sheet_rendering_ms"] += render_ms
        rendered_full_resolution_frames += len(requested)
        manifest_rows.append(
            {
                "candidate_id": candidate_id,
                "video_id": proposal.video_id,
                "fps": proposal.fps,
                "window_start_frame": proposal.window_start_frame,
                "window_end_frame": proposal.window_end_frame,
                "proposal_center_frame": proposal.proposal_center_frame,
                "shot_start_frame": proposal.shot_start_frame,
                "shot_end_frame": proposal.shot_end_frame,
                "scan_stride_frames": proposal.scan_stride_frames,
                "overview_displayed_frames": overview_ids,
                "dense_displayed_frames": dense_ids,
                "overview_sheet_path": overview_path.as_posix(),
                "dense_sheet_path": dense_path.as_posix(),
                "proposal_activity_score": proposal.proposal_activity_score,
                "source_pool_origin": source_origin[proposal.video_id],
            }
        )
        diagnostic_rows.append(
            {
                "candidate_id": candidate_id,
                "video_id": proposal.video_id,
                "activity_mean": proposal.activity_mean,
                "activity_std": proposal.activity_std,
                "activity_peak": proposal.activity_peak,
                "proposal_activity_score": proposal.proposal_activity_score,
                "minimum_distance_to_detected_cut_seconds": (
                    proposal.minimum_distance_to_detected_cut_seconds
                ),
                "window_contains_detected_cut": False,
                "shot_duration_seconds": (
                    proposal.shot_end_frame - proposal.shot_start_frame + 1
                )
                / proposal.fps,
                "candidate_window_duration_seconds": (
                    proposal.window_end_frame - proposal.window_start_frame
                )
                / proposal.fps,
            }
        )

    if len(manifest_rows) != len(diagnostic_rows) or len(manifest_rows) > MAX_CANDIDATES:
        raise RuntimeError("MB1 v0.2 final candidate count is invalid")
    per_video_counts = dict(sorted(Counter(row["video_id"] for row in manifest_rows).items()))
    if any(value > MAX_CANDIDATES_PER_VIDEO for value in per_video_counts.values()):
        raise RuntimeError("MB1 v0.2 per-video cap was exceeded")
    hard_cut_overlap_count = 0
    for row in manifest_rows:
        cuts = tuple(scan_by_video[row["video_id"]]["detected_cut_frames"])
        if window_contains_detected_cut(
            row["window_start_frame"], row["window_end_frame"], cuts
        ):
            hard_cut_overlap_count += 1
        if not (
            row["shot_start_frame"]
            <= row["window_start_frame"]
            <= row["window_end_frame"]
            <= row["shot_end_frame"]
        ):
            raise RuntimeError("MB1 v0.2 candidate escaped its shot segment")
    if hard_cut_overlap_count:
        raise RuntimeError("MB1_V02_FINAL_CANDIDATE_CROSSES_HARD_CUT")

    counts = [row["selection_counts"] for row in per_video_scan]
    selected_video_ids = sorted(per_video_counts)
    if len(manifest_rows) < MIN_ACCEPTABLE_CANDIDATES:
        issues.append(
            {
                "severity": "WARNING",
                "code": "FEWER_THAN_48_VALID_CONTINUOUS_ACTIVE_WINDOWS",
                "message": "Continuity/activity gates were not relaxed to fabricate candidates.",
                "evidence": {"retained": len(manifest_rows)},
            }
        )
    selection = {
        "experiment": "MB1_V02",
        "version": MB1_V02_VERSION,
        "mode": MB1_V02_MODE,
        "semantic_labels_assigned": False,
        "source_videos_considered": [row["video_id"] for row in preflight["source_videos"]],
        "source_videos_selected": selected_video_ids,
        "source_video_count_considered": len(preflight["source_videos"]),
        "source_video_count_selected": len(selected_video_ids),
        "candidate_proposals_before_filtering": sum(
            int(row["candidate_proposals_before_filtering"]) for row in counts
        ),
        "candidates_rejected_for_hard_cut_overlap": sum(
            int(row["rejected_for_hard_cut_overlap"]) for row in counts
        ),
        "candidates_rejected_for_insufficient_activity": sum(
            int(row["rejected_for_insufficient_activity"]) for row in counts
        ),
        "candidates_rejected_for_shot_or_margin": sum(
            int(row["rejected_for_shot_or_margin"]) for row in counts
        ),
        "candidates_rejected_by_temporal_nms": sum(
            int(row["rejected_by_temporal_nms"]) for row in counts
        ),
        "candidates_rejected_by_per_video_cap": sum(
            int(row["rejected_by_per_video_cap"]) for row in counts
        ),
        "final_candidate_count": len(manifest_rows),
        "per_video_candidate_counts": per_video_counts,
        "candidate_ordering_rule": "video_id ASC, window_start_frame ASC",
        "scan_parameters": {
            "target_samples_per_second": SCAN_SAMPLES_PER_SECOND,
            "stride_rule": "max(2, round(fps / 10))",
            "low_resolution_size": list(SCAN_SIZE),
            "signals": ["NORMALIZED_GRAYSCALE_FRAME_DIFFERENCE", "GRAYSCALE_HISTOGRAM_L1"],
        },
        "shot_boundary_rule": {
            "method": "ADAPTIVE_DUAL_SIGNAL_EXTREME",
            "pixel_threshold": "max(0.12, median + 8 * 1.4826 * MAD)",
            "histogram_threshold": "max(0.20, median + 8 * 1.4826 * MAD)",
            "decision": "both signals must exceed their per-video thresholds",
            "candidate_must_remain_inside_one_shot": True,
            "preferred_shot_margin_seconds": SHOT_MARGIN_SECONDS,
        },
        "activity_score_rule": "mean_pixel_diff + 0.5*std + 0.25*peak",
        "insufficient_activity_rule": (
            "activity_mean >= max(0.008, per-video median rolling activity)"
        ),
        "nms_rule": {
            "minimum_center_separation_seconds": NMS_SECONDS,
            "ranking": "proposal_activity_score DESC, center_frame ASC",
            "per_video_cap": MAX_CANDIDATES_PER_VIDEO,
        },
        "overview_evidence": {
            "target_frames": OVERVIEW_FRAME_COUNT,
            "window_seconds": WINDOW_SECONDS,
        },
        "dense_evidence": {"target_frames": DENSE_FRAME_COUNT, "window_seconds": DENSE_SECONDS},
        "hard_cut_overlap_count": hard_cut_overlap_count,
    }
    run_manifest = {
        "experiment": "MB1_V02",
        "version": MB1_V02_VERSION,
        "mode": MB1_V02_MODE,
        "created_at": datetime.now(UTC).isoformat(),
        "build_git_commit": config.build_git_commit,
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(config).items()
        },
        "preflight": preflight,
        "prior_artifacts": {
            "mb1_candidate_manifest_sha256": sha256_file(
                config.prior_candidate_manifest_path
            ),
            "mb1_candidate_qc_sha256": sha256_file(config.prior_candidate_qc_path),
            "rt2_ai_benchmark_sha256": (
                sha256_file(config.rt2_benchmark_path)
                if config.rt2_benchmark_path is not None
                else None
            ),
            "candidate_selection_sha256": (
                sha256_file(config.prior_selection_path)
                if config.prior_selection_path is not None
                else None
            ),
        },
        "previous_semantic_interval_fields_read_by_generation": False,
        "model_inference": False,
        "raw_frame_coordinate_source": "M1_OPENCV_RAW_VIDEO_DECODER_ACTUAL_FRAME_IDX",
        "raw_frames_added_to_stage1_or_framemap": False,
        "network_required": False,
        "performance": {
            "source_pool_construction_ms": source_pool_ms,
            "videos": per_video_scan,
            "overall_runtime_ms": (monotonic() - overall_started) * 1000,
            "low_resolution_scan_frames_processed": sum(
                int(row["scan_frame_count"]) for row in per_video_scan
            ),
            "full_resolution_frames_rendered": rendered_full_resolution_frames,
        },
        "quality_gates": {
            "minimum_48_or_report_shortage_without_relaxation": len(manifest_rows)
            >= MIN_ACCEPTABLE_CANDIDATES,
            "zero_hard_cut_overlaps": hard_cut_overlap_count == 0,
            "both_visual_scales_present": True,
            "manifest_sheet_frame_mapping_exact": True,
            "all_retained_raw_videos_resolved": True,
            "previous_semantic_interval_used": False,
        },
        "MB1_V02_REAL_STATUS": "COMPLETE",
        "MB1_V02_AI_QC_STATUS": "WAITING_FOR_AI",
        "M3_IMPLEMENTATION_STATUS": "NOT_STARTED",
    }
    write_jsonl(output / "mb1_v02_candidate_manifest.jsonl", manifest_rows)
    write_jsonl(output / "mb1_v02_candidate_diagnostics.jsonl", diagnostic_rows)
    write_json(output / "candidate_selection.json", selection)
    write_json(output / "annotation_schema_v02.json", annotation_schema_v02())
    (output / "README_AI_QC_ANNOTATION.md").write_text(
        ai_qc_readme(), encoding="utf-8"
    )
    write_json(output / "run_manifest.json", run_manifest)
    write_jsonl(output / "issues.jsonl", issues)
    return {"selection": selection, "manifest": run_manifest, "issues": issues}


def create_mb1_v02_bundle(output_root: str | Path, zip_path: str | Path) -> Path:
    source = Path(output_root).expanduser().resolve(strict=True)
    target = Path(zip_path).expanduser().resolve(strict=False)
    if source in target.parents:
        raise ValueError("MB1 v0.2 ZIP must be outside output root")
    manifest = _read_jsonl(source / "mb1_v02_candidate_manifest.jsonl")
    members = [source / name for name in BUNDLE_FILES]
    members.extend(source / row["overview_sheet_path"] for row in manifest)
    members.extend(source / row["dense_sheet_path"] for row in manifest)
    if len(members) != len(set(members)):
        raise ValueError("MB1 v0.2 bundle member paths are not unique")
    missing = [str(path.relative_to(source)) for path in members if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"MB1 v0.2 bundle members missing: {missing}")
    if any(path.suffix.lower() in FORBIDDEN_SUFFIXES for path in members):
        raise ValueError("MB1 v0.2 bundle contains a forbidden heavy artifact")
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
    "MB1V02Config",
    "MB1_V02_MODE",
    "MB1_V02_VERSION",
    "annotation_schema_v02",
    "ai_qc_readme",
    "build_source_video_pool",
    "create_mb1_v02_bundle",
    "preflight_mb1_v02",
    "prepare_mb1_v02_candidates",
    "render_contact_sheet",
    "sha256_file",
]
