"""Model-free temporal scan, shot continuity, and active-window proposal for MB1 v0.2."""

from __future__ import annotations

import math
from dataclasses import dataclass
from time import monotonic
from typing import Any

import numpy as np

SCAN_SAMPLES_PER_SECOND = 10.0
SCAN_SIZE = (160, 90)
WINDOW_SECONDS = 4.0
SHOT_MARGIN_SECONDS = 0.25
NMS_SECONDS = 6.0
MAX_CANDIDATES_PER_VIDEO = 4
OVERVIEW_FRAME_COUNT = 32
DENSE_SECONDS = 2.0
DENSE_FRAME_COUNT = 32
SCAN_CHUNK_SAMPLES = 256


@dataclass(frozen=True)
class ScanSeries:
    frame_indices: np.ndarray
    pixel_differences: np.ndarray
    histogram_differences: np.ndarray
    scan_stride_frames: int
    scan_decode_ms: float
    activity_computation_ms: float


@dataclass(frozen=True)
class ShotSegment:
    start_frame: int
    end_frame: int


@dataclass(frozen=True)
class CandidateProposal:
    video_id: str
    fps: float
    window_start_frame: int
    window_end_frame: int
    proposal_center_frame: int
    shot_start_frame: int
    shot_end_frame: int
    scan_stride_frames: int
    activity_mean: float
    activity_std: float
    activity_peak: float
    proposal_activity_score: float
    minimum_distance_to_detected_cut_seconds: float | None


def scan_stride_frames(fps: float) -> int:
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("MB1 v0.2 scan requires positive finite FPS")
    return max(2, int(round(fps / SCAN_SAMPLES_PER_SECOND)))


def _small_gray_and_histogram(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    import cv2

    rgb = np.asarray(image, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("MB1 v0.2 scan frame must be RGB HxWx3")
    small = cv2.resize(rgb, SCAN_SIZE, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    histogram = np.histogram(gray, bins=32, range=(0, 256))[0].astype(np.float64)
    histogram /= max(float(histogram.sum()), 1.0)
    return gray, histogram


def scan_low_resolution_video(decoder: Any) -> ScanSeries:
    """Decode chronological raw coordinates, retaining only low-resolution diagnostics."""

    stride = scan_stride_frames(float(decoder.info.fps))
    indices = list(range(0, int(decoder.info.total_frames), stride))
    if indices[-1] != int(decoder.info.total_frames) - 1:
        indices.append(int(decoder.info.total_frames) - 1)
    pixel = np.zeros(len(indices), dtype=np.float64)
    histogram = np.zeros(len(indices), dtype=np.float64)
    previous_gray: np.ndarray | None = None
    previous_histogram: np.ndarray | None = None
    decode_ms = 0.0
    activity_ms = 0.0
    offset = 0
    for start in range(0, len(indices), SCAN_CHUNK_SAMPLES):
        requested = indices[start : start + SCAN_CHUNK_SAMPLES]
        decode_started = monotonic()
        frames = decoder.decode_indices(requested)
        decode_ms += (monotonic() - decode_started) * 1000
        if [int(frame.actual_frame_idx) for frame in frames] != requested:
            raise RuntimeError("MB1_V02_SCAN_FRAME_IDENTITY_MISMATCH")
        activity_started = monotonic()
        for frame in frames:
            gray, current_histogram = _small_gray_and_histogram(frame.image)
            if previous_gray is not None and previous_histogram is not None:
                pixel[offset] = float(
                    np.mean(
                        np.abs(gray.astype(np.float32) - previous_gray.astype(np.float32))
                    )
                    / 255.0
                )
                histogram[offset] = float(
                    0.5 * np.sum(np.abs(current_histogram - previous_histogram))
                )
            previous_gray = gray
            previous_histogram = current_histogram
            offset += 1
        activity_ms += (monotonic() - activity_started) * 1000
    frames_array = np.asarray(indices, dtype=np.int64)
    if not np.all(np.diff(frames_array) > 0):
        raise RuntimeError("MB1_V02_SCAN_NOT_CHRONOLOGICAL")
    return ScanSeries(frames_array, pixel, histogram, stride, decode_ms, activity_ms)


def _robust_threshold(values: np.ndarray, absolute_floor: float) -> dict[str, float]:
    usable = np.asarray(values, dtype=np.float64)[1:]
    median = float(np.median(usable))
    mad = float(np.median(np.abs(usable - median)))
    robust_scale = 1.4826 * mad
    return {
        "median": median,
        "mad": mad,
        "robust_scale": robust_scale,
        "threshold": max(absolute_floor, median + 8.0 * robust_scale),
    }


def detect_hard_cuts(series: ScanSeries) -> tuple[tuple[int, ...], dict[str, Any]]:
    """Require an adaptive extreme in both pixel and histogram change."""

    pixel_rule = _robust_threshold(series.pixel_differences, 0.12)
    histogram_rule = _robust_threshold(series.histogram_differences, 0.20)
    mask = (series.pixel_differences > pixel_rule["threshold"]) & (
        series.histogram_differences > histogram_rule["threshold"]
    )
    cut_frames = tuple(int(value) for value in series.frame_indices[np.flatnonzero(mask)])
    return cut_frames, {
        "method": "ADAPTIVE_DUAL_SIGNAL_EXTREME",
        "pixel_difference": pixel_rule,
        "histogram_difference": histogram_rule,
        "decision": "pixel > threshold AND histogram > threshold",
    }


def continuous_shot_segments(
    total_frames: int, cut_frames: tuple[int, ...]
) -> tuple[ShotSegment, ...]:
    if total_frames <= 0:
        raise ValueError("total_frames must be positive")
    cuts = tuple(sorted(set(int(value) for value in cut_frames)))
    if any(value <= 0 or value >= total_frames for value in cuts):
        raise ValueError("detected cut lies outside valid internal frame boundaries")
    segments = []
    start = 0
    for cut in cuts:
        segments.append(ShotSegment(start, cut - 1))
        start = cut
    segments.append(ShotSegment(start, total_frames - 1))
    if any(segment.start_frame > segment.end_frame for segment in segments):
        raise RuntimeError("MB1 v0.2 generated an empty shot segment")
    for left, right in zip(segments[:-1], segments[1:], strict=True):
        if left.end_frame + 1 != right.start_frame:
            raise RuntimeError("MB1 v0.2 shot segments overlap or leave gaps")
    return tuple(segments)


def window_bounds(center: int, fps: float) -> tuple[int, int]:
    span = max(1, int(round(WINDOW_SECONDS * fps)))
    start = int(center) - span // 2
    return start, start + span


def window_contains_detected_cut(
    start: int, end: int, cut_frames: tuple[int, ...]
) -> bool:
    return any(start <= cut <= end for cut in cut_frames)


def _containing_shot(
    start: int, end: int, segments: tuple[ShotSegment, ...]
) -> ShotSegment | None:
    for segment in segments:
        if segment.start_frame <= start and end <= segment.end_frame:
            return segment
    return None


def _rolling_activity(series: ScanSeries, fps: float) -> np.ndarray:
    radius = max(1, int(round((WINDOW_SECONDS * fps / series.scan_stride_frames) / 2)))
    values = series.pixel_differences
    prefix = np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))
    output = np.empty_like(values)
    for index in range(len(values)):
        start = max(1, index - radius)
        stop = min(len(values), index + radius + 1)
        output[index] = (prefix[stop] - prefix[start]) / max(stop - start, 1)
    return output


def propose_active_windows(
    video_id: str,
    fps: float,
    total_frames: int,
    series: ScanSeries,
    cut_frames: tuple[int, ...],
    segments: tuple[ShotSegment, ...],
) -> tuple[tuple[CandidateProposal, ...], dict[str, int | float]]:
    """Select deterministic local activity maxima with cut guard, margin, and 6s NMS."""

    rolling = _rolling_activity(series, fps)
    valid_rolling = rolling[1:]
    activity_floor = max(0.008, float(np.median(valid_rolling)))
    local_radius = max(1, int(round(fps / series.scan_stride_frames)))
    local_maxima = []
    for index in range(1, len(series.frame_indices)):
        left = max(1, index - local_radius)
        right = min(len(rolling), index + local_radius + 1)
        if rolling[index] >= float(np.max(rolling[left:right])):
            local_maxima.append(index)

    rejected_cut = 0
    rejected_activity = 0
    rejected_shot_or_margin = 0
    raw_proposals: list[CandidateProposal] = []
    margin = int(round(SHOT_MARGIN_SECONDS * fps))
    window_radius_samples = max(
        1, int(round((WINDOW_SECONDS * fps / series.scan_stride_frames) / 2))
    )
    for index in local_maxima:
        center = int(series.frame_indices[index])
        start, end = window_bounds(center, fps)
        if start < 0 or end >= total_frames:
            rejected_shot_or_margin += 1
            continue
        if window_contains_detected_cut(start, end, cut_frames):
            rejected_cut += 1
            continue
        shot = _containing_shot(start, end, segments)
        if (
            shot is None
            or start < shot.start_frame + margin
            or end > shot.end_frame - margin
        ):
            rejected_shot_or_margin += 1
            continue
        values = series.pixel_differences[
            max(1, index - window_radius_samples) : min(
                len(series.pixel_differences), index + window_radius_samples + 1
            )
        ]
        activity_mean = float(np.mean(values))
        if activity_mean < activity_floor:
            rejected_activity += 1
            continue
        activity_std = float(np.std(values))
        activity_peak = float(np.max(values))
        score = activity_mean + 0.5 * activity_std + 0.25 * activity_peak
        distances = [min(abs(start - cut), abs(end - cut)) / fps for cut in cut_frames]
        raw_proposals.append(
            CandidateProposal(
                video_id=video_id,
                fps=fps,
                window_start_frame=start,
                window_end_frame=end,
                proposal_center_frame=center,
                shot_start_frame=shot.start_frame,
                shot_end_frame=shot.end_frame,
                scan_stride_frames=series.scan_stride_frames,
                activity_mean=activity_mean,
                activity_std=activity_std,
                activity_peak=activity_peak,
                proposal_activity_score=score,
                minimum_distance_to_detected_cut_seconds=(
                    min(distances) if distances else None
                ),
            )
        )

    ordered = sorted(
        raw_proposals,
        key=lambda item: (-item.proposal_activity_score, item.proposal_center_frame),
    )
    selected: list[CandidateProposal] = []
    rejected_nms = 0
    rejected_cap = 0
    for proposal in ordered:
        if any(
            abs(proposal.proposal_center_frame - existing.proposal_center_frame) / fps
            < NMS_SECONDS
            for existing in selected
        ):
            rejected_nms += 1
            continue
        if len(selected) >= MAX_CANDIDATES_PER_VIDEO:
            rejected_cap += 1
            continue
        selected.append(proposal)
    return tuple(selected), {
        "local_activity_maxima": len(local_maxima),
        "candidate_proposals_before_filtering": len(local_maxima),
        "rejected_for_hard_cut_overlap": rejected_cut,
        "rejected_for_insufficient_activity": rejected_activity,
        "rejected_for_shot_or_margin": rejected_shot_or_margin,
        "rejected_by_temporal_nms": rejected_nms,
        "rejected_by_per_video_cap": rejected_cap,
        "activity_floor": activity_floor,
        "retained": len(selected),
    }


def uniform_frame_indices(start: int, end: int, count: int) -> list[int]:
    if start < 0 or start > end or count < 2:
        raise ValueError("invalid uniform frame sampling request")
    actual_count = min(count, end - start + 1)
    values = np.rint(np.linspace(start, end, actual_count)).astype(np.int64)
    output = sorted(set(int(value) for value in values))
    if len(output) != actual_count or not all(start <= value <= end for value in output):
        raise RuntimeError("MB1 v0.2 displayed frames are not exact and bounded")
    return output


def overview_displayed_frames(proposal: CandidateProposal) -> list[int]:
    return uniform_frame_indices(
        proposal.window_start_frame, proposal.window_end_frame, OVERVIEW_FRAME_COUNT
    )


def dense_displayed_frames(proposal: CandidateProposal) -> list[int]:
    half_span = max(1, int(round(DENSE_SECONDS * proposal.fps)) // 2)
    start = max(proposal.window_start_frame, proposal.proposal_center_frame - half_span)
    end = min(proposal.window_end_frame, start + 2 * half_span)
    start = max(proposal.window_start_frame, end - 2 * half_span)
    return uniform_frame_indices(start, end, DENSE_FRAME_COUNT)


__all__ = [
    "CandidateProposal",
    "DENSE_FRAME_COUNT",
    "DENSE_SECONDS",
    "MAX_CANDIDATES_PER_VIDEO",
    "NMS_SECONDS",
    "OVERVIEW_FRAME_COUNT",
    "SCAN_SAMPLES_PER_SECOND",
    "SCAN_SIZE",
    "SHOT_MARGIN_SECONDS",
    "ScanSeries",
    "ShotSegment",
    "WINDOW_SECONDS",
    "continuous_shot_segments",
    "dense_displayed_frames",
    "detect_hard_cuts",
    "overview_displayed_frames",
    "propose_active_windows",
    "scan_low_resolution_video",
    "scan_stride_frames",
    "uniform_frame_indices",
    "window_bounds",
    "window_contains_detected_cut",
]
