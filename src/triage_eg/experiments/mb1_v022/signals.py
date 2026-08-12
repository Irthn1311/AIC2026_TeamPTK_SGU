"""Deterministic context geometry and ORB continuity for MB1 v0.2.2."""

from __future__ import annotations

import math
from dataclasses import dataclass
from time import monotonic
from typing import Any

import numpy as np

from triage_eg.experiments.mb1_v021.signals import (
    DENSE_FRAME_COUNT,
    DENSE_SECONDS,
    EPSILON,
    SignalSeries,
    WindowAdjustment,
    robust_baseline,
    uniform_frame_indices,
)

MIN_PRE_CONTEXT_SECONDS = 1.25
MIN_POST_CONTEXT_SECONDS = 1.25
CUT_SAFETY_MARGIN_SECONDS = 0.50
PREFERRED_WINDOW_SECONDS = 4.0
MINIMUM_WINDOW_SECONDS = 3.0
DENSE_EDGE_CONTEXT_SECONDS = 0.75
LOCAL_SAMPLES_PER_SECOND = 12.0
LOCAL_SIZE = (320, 180)
DECODE_BATCH_SIZE = 4
ORB_NFEATURES = 500
ORB_FAST_THRESHOLD = 12
ORB_MIN_DESCRIPTORS = 12
ORB_MAX_HAMMING_DISTANCE = 48
ABRUPT_LOCAL_Z = 4.0
EXTREME_HISTOGRAM_Z = 6.0
LOW_ORB_CONTINUITY = 0.08
SOFT_ORB_CONTINUITY = 0.12


@dataclass(frozen=True)
class ContextWindow:
    window: WindowAdjustment
    clean_left_frame: int
    clean_right_frame: int
    previous_cut_frame: int | None
    next_cut_frame: int | None
    pre_context_seconds: float
    post_context_seconds: float


@dataclass(frozen=True)
class ORBTransition:
    available: bool
    continuity: float | None
    keypoints_a: int
    keypoints_b: int
    good_matches: int


@dataclass(frozen=True)
class FinalContinuityAudit:
    frame_indices: tuple[int, ...]
    pixel_differences: tuple[float, ...]
    histogram_differences: tuple[float, ...]
    pixel_local_z: tuple[float, ...]
    histogram_local_z: tuple[float, ...]
    pixel_video_z: tuple[float, ...]
    histogram_video_z: tuple[float, ...]
    orb_transitions: tuple[ORBTransition, ...]
    abrupt_transition_frames: tuple[int, ...]
    soft_transition_frames: tuple[int, ...]
    continuity_quality: float
    orb_continuity_mean: float | None
    orb_continuity_min: float | None
    orb_available_fraction: float
    maximum_local_pixel_jump: float
    maximum_local_histogram_jump: float
    decode_ms: float
    signal_ms: float
    orb_ms: float

    @property
    def status(self) -> str:
        return (
            "AUTO_CONTINUITY_SCREEN_REJECT"
            if self.abrupt_transition_frames or self.soft_transition_frames
            else "AUTO_CONTINUITY_SCREEN_PASS"
        )


def cut_inside_window(cut_frame: int, start_frame: int, end_frame: int) -> bool:
    """A cut c represents transition c-1 -> c."""

    return start_frame < int(cut_frame) <= end_frame


def context_window(
    *,
    center_frame: int,
    fps: float,
    total_frames: int,
    cut_frames: tuple[int, ...],
) -> tuple[ContextWindow | None, str | None]:
    if not math.isfinite(fps) or fps <= 0 or total_frames <= 0:
        raise ValueError("MB1 v0.2.2 requires valid video metadata")
    cuts = tuple(sorted(set(int(value) for value in cut_frames)))
    previous = max((cut for cut in cuts if cut <= center_frame), default=None)
    following = min((cut for cut in cuts if cut > center_frame), default=None)
    margin = int(math.ceil(CUT_SAFETY_MARGIN_SECONDS * fps))
    clean_left = (previous + margin) if previous is not None else 0
    clean_right = (following - 1 - margin) if following is not None else total_frames - 1
    required_pre = int(math.ceil(MIN_PRE_CONTEXT_SECONDS * fps))
    required_post = int(math.ceil(MIN_POST_CONTEXT_SECONDS * fps))
    if center_frame - clean_left < required_pre:
        return None, "INSUFFICIENT_PRE_CONTEXT"
    if clean_right - center_frame < required_post:
        return None, "INSUFFICIENT_POST_CONTEXT"
    preferred_span = int(round(PREFERRED_WINDOW_SECONDS * fps))
    minimum_span = int(math.ceil(MINIMUM_WINDOW_SECONDS * fps))
    start = center_frame - preferred_span // 2
    end = start + preferred_span
    reason = "NONE"
    if start < clean_left or end > clean_right:
        available_left = center_frame - clean_left
        available_right = clean_right - center_frame
        half = min(preferred_span // 2, available_left, available_right)
        if 2 * half >= minimum_span:
            start, end, reason = center_frame - half, center_frame + half, "SYMMETRIC_SHRINK"
        else:
            target_span = min(preferred_span, clean_right - clean_left)
            if target_span < minimum_span:
                return None, "KNOWN_CUT_SAFETY_MARGIN"
            start = max(clean_left, min(start, clean_right - target_span))
            end = start + target_span
            if not (
                start <= center_frame - required_pre
                and center_frame + required_post <= end
                and clean_left <= start < end <= clean_right
            ):
                return None, "KNOWN_CUT_SAFETY_MARGIN"
            reason = "SMALL_SHIFT_WITH_CONTEXT_PRESERVED"
    if end - start < minimum_span:
        return None, "KNOWN_CUT_SAFETY_MARGIN"
    if center_frame in {start, end, clean_left, clean_right}:
        return None, "CENTER_AT_BOUNDARY"
    return (
        ContextWindow(
            window=WindowAdjustment(
                int(start),
                int(end),
                PREFERRED_WINDOW_SECONDS,
                (end - start) / fps,
                reason,
            ),
            clean_left_frame=int(clean_left),
            clean_right_frame=int(clean_right),
            previous_cut_frame=previous,
            next_cut_frame=following,
            pre_context_seconds=(center_frame - start) / fps,
            post_context_seconds=(end - center_frame) / fps,
        ),
        None,
    )


def center_relative_features(series: SignalSeries, center_frame: int, fps: float) -> dict[str, Any]:
    zones = {
        "pre": (center_frame - 1.25 * fps, center_frame - 0.25 * fps),
        "center": (center_frame - 0.25 * fps, center_frame + 0.25 * fps),
        "post": (center_frame + 0.25 * fps, center_frame + 1.25 * fps),
    }
    summaries: dict[str, dict[str, float]] = {}
    positions: dict[str, np.ndarray] = {}
    for name, (left, right) in zones.items():
        mask = (series.frame_indices >= math.ceil(left)) & (
            series.frame_indices <= math.floor(right)
        )
        pos = np.flatnonzero(mask)
        if len(pos) < 2:
            raise ValueError(f"insufficient center-relative {name.upper()} samples")
        values = series.pixel_differences[pos]
        summaries[name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "peak": float(np.max(values)),
            "median": float(np.median(values)),
        }
        positions[name] = pos
    transition = summaries["center"]["peak"] - max(
        summaries["pre"]["median"], summaries["post"]["median"]
    )
    pre_hist = np.mean(series.histograms[positions["pre"]], axis=0)
    post_hist = np.mean(series.histograms[positions["post"]], axis=0)
    before_after = float(0.5 * np.abs(pre_hist - post_hist).sum())
    spatial = float(np.mean(series.spatial_concentrations[positions["center"]]))
    all_positions = np.concatenate(tuple(positions.values()))
    overall = float(np.mean(series.pixel_differences[all_positions]))
    return {
        "pre_activity": summaries["pre"],
        "center_activity": summaries["center"],
        "post_activity": summaries["post"],
        "transition_strength": transition,
        "before_after_visual_difference": before_after,
        "spatial_activity_concentration": spatial,
        "overall_activity": overall,
    }


def orb_transition(first_gray: np.ndarray, second_gray: np.ndarray) -> ORBTransition:
    import cv2

    orb = cv2.ORB_create(nfeatures=ORB_NFEATURES, fastThreshold=ORB_FAST_THRESHOLD)
    keypoints_a, descriptors_a = orb.detectAndCompute(first_gray, None)
    keypoints_b, descriptors_b = orb.detectAndCompute(second_gray, None)
    count_a, count_b = len(keypoints_a), len(keypoints_b)
    if (
        descriptors_a is None
        or descriptors_b is None
        or min(count_a, count_b) < ORB_MIN_DESCRIPTORS
    ):
        return ORBTransition(False, None, count_a, count_b, 0)
    matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(descriptors_a, descriptors_b)
    good = sum(match.distance <= ORB_MAX_HAMMING_DISTANCE for match in matches)
    return ORBTransition(True, good / max(min(count_a, count_b), 1), count_a, count_b, good)


def _local_frames(decoder: Any, start: int, end: int) -> tuple[list[int], list[np.ndarray], float]:
    import cv2

    stride = max(1, int(round(float(decoder.info.fps) / LOCAL_SAMPLES_PER_SECOND)))
    indices = list(range(start, end + 1, stride))
    if indices[-1] != end:
        indices.append(end)
    grays: list[np.ndarray] = []
    decode_ms = 0.0
    for offset in range(0, len(indices), DECODE_BATCH_SIZE):
        batch = indices[offset : offset + DECODE_BATCH_SIZE]
        started = monotonic()
        frames = decoder.decode_indices(batch)
        decode_ms += (monotonic() - started) * 1000
        if [int(frame.actual_frame_idx) for frame in frames] != batch:
            raise RuntimeError("MB1_V022_LOCAL_FRAME_IDENTITY_MISMATCH")
        for frame in frames:
            image = cv2.resize(np.asarray(frame.image, dtype=np.uint8), LOCAL_SIZE)
            grays.append(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY))
    return indices, grays, decode_ms


def final_continuity_audit(
    decoder: Any,
    start: int,
    end: int,
    *,
    video_pixel_baseline: Any | None = None,
    video_histogram_baseline: Any | None = None,
) -> FinalContinuityAudit:
    indices, grays, decode_ms = _local_frames(decoder, start, end)
    signal_started = monotonic()
    pixels = np.zeros(len(grays), dtype=np.float64)
    hist_diffs = np.zeros(len(grays), dtype=np.float64)
    histograms = []
    for gray in grays:
        hist = np.histogram(gray, bins=32, range=(0, 256))[0].astype(np.float64)
        histograms.append(hist / max(hist.sum(), 1.0))
    for index in range(1, len(grays)):
        pixels[index] = float(
            np.mean(np.abs(grays[index].astype(np.float32) - grays[index - 1].astype(np.float32)))
            / 255.0
        )
        hist_diffs[index] = float(0.5 * np.abs(histograms[index] - histograms[index - 1]).sum())
    pixel_base = robust_baseline(pixels[1:])
    hist_base = robust_baseline(hist_diffs[1:])
    pixel_z = (pixels - pixel_base.median) / max(pixel_base.robust_scale, EPSILON)
    hist_z = (hist_diffs - hist_base.median) / max(hist_base.robust_scale, EPSILON)
    video_pixel = video_pixel_baseline or pixel_base
    video_histogram = video_histogram_baseline or hist_base
    pixel_video_z = (pixels - video_pixel.median) / max(video_pixel.robust_scale, EPSILON)
    histogram_video_z = (hist_diffs - video_histogram.median) / max(
        video_histogram.robust_scale, EPSILON
    )
    signal_ms = (monotonic() - signal_started) * 1000
    orb_started = monotonic()
    orb_rows = [ORBTransition(False, None, 0, 0, 0)]
    orb_rows.extend(orb_transition(grays[i - 1], grays[i]) for i in range(1, len(grays)))
    orb_ms = (monotonic() - orb_started) * 1000
    abrupt = []
    for i in range(1, len(indices)):
        collapse = orb_rows[i].available and float(orb_rows[i].continuity) <= LOW_ORB_CONTINUITY
        strongest_change_z = max(pixel_z[i], hist_z[i], pixel_video_z[i], histogram_video_z[i])
        if strongest_change_z >= ABRUPT_LOCAL_Z and (collapse or hist_z[i] >= EXTREME_HISTOGRAM_Z):
            abrupt.append(indices[i])
    elevated = (pixel_z >= 2.5) & (hist_z >= 2.5)
    soft: list[int] = []
    run_start: int | None = None
    for i in range(1, len(indices) + 1):
        active = i < len(indices) and bool(elevated[i])
        if active and run_start is None:
            run_start = i
        if not active and run_start is not None:
            run_end = i - 1
            if 3 <= run_end - run_start + 1 <= 7:
                available = [
                    row.continuity for row in orb_rows[run_start : run_end + 1] if row.available
                ]
                if (available and float(np.mean(available)) <= SOFT_ORB_CONTINUITY) or (
                    not available
                    and float(np.max(hist_z[run_start : run_end + 1])) >= EXTREME_HISTOGRAM_Z
                ):
                    soft.append(indices[(run_start + run_end) // 2])
            run_start = None
    available_values = [float(row.continuity) for row in orb_rows[1:] if row.available]
    availability = len(available_values) / max(len(orb_rows) - 1, 1)
    orb_mean = float(np.mean(available_values)) if available_values else None
    orb_min = float(np.min(available_values)) if available_values else None
    orb_component = orb_mean if orb_mean is not None else 0.5
    global_consistency = 1.0 / (
        1.0 + max(0.0, float(np.max(pixel_z[1:])), float(np.max(hist_z[1:]))) / 8.0
    )
    quality = float(np.clip(0.65 * orb_component + 0.35 * global_consistency, 0.0, 1.0))
    if abrupt or soft:
        quality = min(quality, 0.05)
    return FinalContinuityAudit(
        tuple(indices),
        tuple(map(float, pixels)),
        tuple(map(float, hist_diffs)),
        tuple(map(float, pixel_z)),
        tuple(map(float, hist_z)),
        tuple(map(float, pixel_video_z)),
        tuple(map(float, histogram_video_z)),
        tuple(orb_rows),
        tuple(abrupt),
        tuple(soft),
        quality,
        orb_mean,
        orb_min,
        availability,
        float(np.max(pixels[1:])),
        float(np.max(hist_diffs[1:])),
        decode_ms,
        signal_ms,
        orb_ms,
    )


def safe_dense_focus(series: SignalSeries, start: int, end: int, center: int, fps: float) -> int:
    margin = int(math.ceil(DENSE_EDGE_CONTEXT_SECONDS * fps))
    eligible = np.flatnonzero(
        (series.frame_indices >= start + margin) & (series.frame_indices <= end - margin)
    )
    if not len(eligible):
        return center
    score = (
        np.maximum(series.pixel_robust_z[eligible], 0)
        + np.maximum(series.histogram_robust_z[eligible], 0)
        + series.spatial_concentrations[eligible]
    )
    return int(series.frame_indices[eligible[int(np.argmax(score))]])


def dense_displayed_frames(start: int, end: int, focus: int, fps: float) -> list[int]:
    span = min(end - start, int(round(DENSE_SECONDS * fps)))
    dense_start = max(start, min(focus - span // 2, end - span))
    return uniform_frame_indices(dense_start, dense_start + span, DENSE_FRAME_COUNT)
