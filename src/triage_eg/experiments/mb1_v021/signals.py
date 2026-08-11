"""Deterministic two-pass visual continuity signals for MB1 v0.2.1."""

from __future__ import annotations

import math
from dataclasses import dataclass
from time import monotonic
from typing import Any

import numpy as np

COARSE_SAMPLES_PER_SECOND = 10.0
LOCAL_SAMPLES_PER_SECOND = 15.0
SCAN_SIZE = (160, 90)
GRID_SHAPE = (4, 4)
PREFERRED_WINDOW_SECONDS = 4.0
MINIMUM_WINDOW_SECONDS = 3.0
LOCAL_EXPANSION_SECONDS = 0.5
NMS_SECONDS = 6.0
SEED_CENTER_EXCLUSION_SECONDS = 3.0
SEED_IOU_EXCLUSION = 0.50
MAX_NEW_CANDIDATES_PER_VIDEO = 3
MAX_LOCAL_PROPOSALS_PER_VIDEO = 18
OVERVIEW_FRAME_COUNT = 32
DENSE_FRAME_COUNT = 32
DENSE_SECONDS = 2.0
EPSILON = 1e-9


@dataclass(frozen=True)
class SignalBaseline:
    median: float
    mad: float
    robust_scale: float
    percentile_95: float

    def as_dict(self) -> dict[str, float]:
        return {
            "median": self.median,
            "mad": self.mad,
            "robust_scale": self.robust_scale,
            "percentile_95": self.percentile_95,
        }


@dataclass(frozen=True)
class SignalSeries:
    frame_indices: np.ndarray
    pixel_differences: np.ndarray
    histogram_differences: np.ndarray
    histograms: np.ndarray
    spatial_concentrations: np.ndarray
    pixel_robust_z: np.ndarray
    histogram_robust_z: np.ndarray
    pixel_percentiles: np.ndarray
    histogram_percentiles: np.ndarray
    pixel_baseline: SignalBaseline
    histogram_baseline: SignalBaseline
    stride_frames: int
    decode_ms: float
    signal_ms: float


@dataclass(frozen=True)
class WindowAdjustment:
    start_frame: int
    end_frame: int
    requested_duration_seconds: float
    final_duration_seconds: float
    reason: str


@dataclass(frozen=True)
class LocalContinuityResult:
    frame_indices: tuple[int, ...]
    hard_cut_frames: tuple[int, ...]
    soft_cut_frames: tuple[int, ...]
    max_pixel_percentile: float
    max_hist_percentile: float
    max_pixel_robust_z: float
    max_hist_robust_z: float
    decode_ms: float
    signal_ms: float

    @property
    def veto_frames(self) -> tuple[int, ...]:
        return tuple(sorted(set(self.hard_cut_frames + self.soft_cut_frames)))


def scan_stride_frames(fps: float) -> int:
    _validate_fps(fps)
    return max(2, int(round(fps / COARSE_SAMPLES_PER_SECOND)))


def local_stride_frames(fps: float) -> int:
    _validate_fps(fps)
    return max(1, int(round(fps / LOCAL_SAMPLES_PER_SECOND)))


def _validate_fps(fps: float) -> None:
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("MB1 v0.2.1 requires positive finite FPS")


def robust_baseline(values: np.ndarray) -> SignalBaseline:
    usable = np.asarray(values, dtype=np.float64)
    if usable.ndim != 1 or not len(usable):
        raise ValueError("robust baseline requires a non-empty one-dimensional signal")
    median = float(np.median(usable))
    mad = float(np.median(np.abs(usable - median)))
    return SignalBaseline(
        median=median,
        mad=mad,
        robust_scale=max(1.4826 * mad, EPSILON),
        percentile_95=float(np.quantile(usable, 0.95)),
    )


def robust_z(values: np.ndarray, baseline: SignalBaseline) -> np.ndarray:
    return (np.asarray(values, dtype=np.float64) - baseline.median) / max(
        baseline.robust_scale, EPSILON
    )


def empirical_percentiles(
    values: np.ndarray, reference: np.ndarray | None = None
) -> np.ndarray:
    source = np.asarray(values, dtype=np.float64)
    population = np.sort(
        np.asarray(reference if reference is not None else values, dtype=np.float64)
    )
    if source.ndim != 1 or population.ndim != 1 or not len(population):
        raise ValueError("empirical percentile inputs must be non-empty vectors")
    ranks = np.searchsorted(population, source, side="right")
    return ranks.astype(np.float64) / float(len(population))


def hard_cut_mask(
    pixel_percentiles: np.ndarray, histogram_percentiles: np.ndarray
) -> np.ndarray:
    pixel = np.asarray(pixel_percentiles, dtype=np.float64)
    histogram = np.asarray(histogram_percentiles, dtype=np.float64)
    if pixel.shape != histogram.shape:
        raise ValueError("hard-cut percentile signals must have matching shapes")
    strong_dual = ((pixel >= 0.99) & (histogram >= 0.95)) | (
        (histogram >= 0.99) & (pixel >= 0.95)
    )
    extreme_supported = ((pixel >= 0.998) & (histogram >= 0.80)) | (
        (histogram >= 0.998) & (pixel >= 0.80)
    )
    return strong_dual | extreme_supported


def _small_frame_features(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    import cv2

    rgb = np.asarray(image, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("MB1 v0.2.1 scan frame must be RGB HxWx3")
    small = cv2.resize(rgb, SCAN_SIZE, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    histogram = np.histogram(gray, bins=32, range=(0, 256))[0].astype(np.float64)
    histogram /= max(float(histogram.sum()), 1.0)
    return gray, histogram


def spatial_activity_concentration(
    current_gray: np.ndarray, previous_gray: np.ndarray
) -> float:
    difference = np.abs(
        current_gray.astype(np.float32) - previous_gray.astype(np.float32)
    )
    rows = np.array_split(difference, GRID_SHAPE[0], axis=0)
    cell_values = np.asarray(
        [float(cell.mean()) for row in rows for cell in np.array_split(row, GRID_SHAPE[1], axis=1)],
        dtype=np.float64,
    )
    total = float(cell_values.sum())
    if total <= EPSILON:
        return 0.0
    top_count = max(1, int(round(len(cell_values) * 0.25)))
    return float(np.sort(cell_values)[-top_count:].sum() / total)


def _decode_series(decoder: Any, indices: list[int], stride: int) -> SignalSeries:
    if not indices:
        raise ValueError("MB1 v0.2.1 cannot scan an empty frame range")
    decode_started = monotonic()
    frames = decoder.decode_indices(indices)
    decode_ms = (monotonic() - decode_started) * 1000
    identities = [int(frame.actual_frame_idx) for frame in frames]
    if identities != indices:
        raise RuntimeError("MB1_V021_SCAN_FRAME_IDENTITY_MISMATCH")
    pixel = np.zeros(len(frames), dtype=np.float64)
    histogram_difference = np.zeros(len(frames), dtype=np.float64)
    concentration = np.zeros(len(frames), dtype=np.float64)
    histograms = np.empty((len(frames), 32), dtype=np.float64)
    previous_gray: np.ndarray | None = None
    previous_histogram: np.ndarray | None = None
    signal_started = monotonic()
    for index, frame in enumerate(frames):
        gray, histogram = _small_frame_features(frame.image)
        histograms[index] = histogram
        if previous_gray is not None and previous_histogram is not None:
            pixel[index] = float(
                np.mean(
                    np.abs(gray.astype(np.float32) - previous_gray.astype(np.float32))
                )
                / 255.0
            )
            histogram_difference[index] = float(
                0.5 * np.abs(histogram - previous_histogram).sum()
            )
            concentration[index] = spatial_activity_concentration(gray, previous_gray)
        previous_gray = gray
        previous_histogram = histogram
    signal_ms = (monotonic() - signal_started) * 1000
    pixel_baseline = robust_baseline(pixel[1:] if len(pixel) > 1 else pixel)
    histogram_baseline = robust_baseline(
        histogram_difference[1:] if len(histogram_difference) > 1 else histogram_difference
    )
    return SignalSeries(
        frame_indices=np.asarray(indices, dtype=np.int64),
        pixel_differences=pixel,
        histogram_differences=histogram_difference,
        histograms=histograms,
        spatial_concentrations=concentration,
        pixel_robust_z=robust_z(pixel, pixel_baseline),
        histogram_robust_z=robust_z(histogram_difference, histogram_baseline),
        pixel_percentiles=empirical_percentiles(pixel),
        histogram_percentiles=empirical_percentiles(histogram_difference),
        pixel_baseline=pixel_baseline,
        histogram_baseline=histogram_baseline,
        stride_frames=stride,
        decode_ms=decode_ms,
        signal_ms=signal_ms,
    )


def scan_coarse_video(decoder: Any) -> SignalSeries:
    total_frames = int(decoder.info.total_frames)
    if total_frames <= 0:
        raise ValueError("MB1 v0.2.1 video contains no frames")
    stride = scan_stride_frames(float(decoder.info.fps))
    indices = list(range(0, total_frames, stride))
    if indices[-1] != total_frames - 1:
        indices.append(total_frames - 1)
    return _decode_series(decoder, indices, stride)


def scan_local_video(
    decoder: Any,
    start_frame: int,
    end_frame: int,
    coarse: SignalSeries,
) -> tuple[SignalSeries, LocalContinuityResult]:
    if start_frame < 0 or start_frame > end_frame:
        raise ValueError("invalid MB1 v0.2.1 local scan range")
    stride = local_stride_frames(float(decoder.info.fps))
    indices = list(range(start_frame, end_frame + 1, stride))
    if indices[-1] != end_frame:
        indices.append(end_frame)
    local = _decode_series(decoder, indices, stride)
    pixel_z = robust_z(local.pixel_differences, coarse.pixel_baseline)
    histogram_z = robust_z(
        local.histogram_differences, coarse.histogram_baseline
    )
    pixel_percentile = empirical_percentiles(
        local.pixel_differences, coarse.pixel_differences[1:]
    )
    histogram_percentile = empirical_percentiles(
        local.histogram_differences, coarse.histogram_differences[1:]
    )
    local = SignalSeries(
        frame_indices=local.frame_indices,
        pixel_differences=local.pixel_differences,
        histogram_differences=local.histogram_differences,
        histograms=local.histograms,
        spatial_concentrations=local.spatial_concentrations,
        pixel_robust_z=pixel_z,
        histogram_robust_z=histogram_z,
        pixel_percentiles=pixel_percentile,
        histogram_percentiles=histogram_percentile,
        pixel_baseline=coarse.pixel_baseline,
        histogram_baseline=coarse.histogram_baseline,
        stride_frames=stride,
        decode_ms=local.decode_ms,
        signal_ms=local.signal_ms,
    )
    hard_indices = np.flatnonzero(hard_cut_mask(pixel_percentile, histogram_percentile))
    hard_frames = tuple(int(local.frame_indices[index]) for index in hard_indices)
    soft_runs = detect_soft_transition_runs(
        local,
        fps=float(decoder.info.fps),
        reference_histogram_p95=coarse.histogram_baseline.percentile_95,
    )
    soft_frames = tuple(
        int(local.frame_indices[(start + end) // 2]) for start, end in soft_runs
    )
    usable = slice(1, None) if len(local.frame_indices) > 1 else slice(None)
    result = LocalContinuityResult(
        frame_indices=tuple(int(value) for value in local.frame_indices),
        hard_cut_frames=hard_frames,
        soft_cut_frames=soft_frames,
        max_pixel_percentile=float(np.max(pixel_percentile[usable])),
        max_hist_percentile=float(np.max(histogram_percentile[usable])),
        max_pixel_robust_z=float(np.max(pixel_z[usable])),
        max_hist_robust_z=float(np.max(histogram_z[usable])),
        decode_ms=local.decode_ms,
        signal_ms=local.signal_ms,
    )
    return local, result


def detect_soft_transition_runs(
    series: SignalSeries,
    *,
    fps: float,
    reference_histogram_p95: float,
) -> tuple[tuple[int, int], ...]:
    rate = fps / series.stride_frames
    minimum_steps = max(3, int(math.ceil(0.20 * rate)))
    maximum_steps = max(minimum_steps, int(math.ceil(0.45 * rate)))
    elevated = (series.pixel_robust_z >= 2.5) & (
        series.histogram_robust_z >= 2.5
    )
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index in range(1, len(elevated) + 1):
        active = index < len(elevated) and bool(elevated[index])
        if active and start is None:
            start = index
        if not active and start is not None:
            end = index - 1
            length = end - start + 1
            before = max(0, start - 1)
            after = min(len(series.histograms) - 1, end + 1)
            appearance_change = float(
                0.5
                * np.abs(series.histograms[after] - series.histograms[before]).sum()
            )
            if (
                minimum_steps <= length <= maximum_steps
                and appearance_change >= reference_histogram_p95
            ):
                runs.append((start, end))
            start = None
    return tuple(runs)


def adaptive_window(
    *,
    center_frame: int,
    fps: float,
    shot_start_frame: int,
    shot_end_frame: int,
    veto_frames: tuple[int, ...] = (),
) -> WindowAdjustment | None:
    _validate_fps(fps)
    if not shot_start_frame <= center_frame <= shot_end_frame:
        raise ValueError("proposal center is outside coarse shot")
    minimum_span = int(math.ceil(MINIMUM_WINDOW_SECONDS * fps))
    preferred_span = int(round(PREFERRED_WINDOW_SECONDS * fps))
    clean_start = int(shot_start_frame)
    clean_end = int(shot_end_frame)
    relevant = tuple(
        sorted(
            cut
            for cut in set(int(value) for value in veto_frames)
            if clean_start <= cut <= clean_end
        )
    )
    central_guard = int(round(0.5 * fps))
    if any(abs(cut - center_frame) <= central_guard for cut in relevant):
        return None
    for cut in relevant:
        if cut < center_frame:
            clean_start = max(clean_start, cut)
        elif cut > center_frame:
            clean_end = min(clean_end, cut - 1)
    if clean_end - clean_start < minimum_span:
        return None
    nominal_start = center_frame - preferred_span // 2
    nominal_end = nominal_start + preferred_span
    target_span = min(preferred_span, clean_end - clean_start)
    if target_span < minimum_span:
        return None
    start = nominal_start
    end = start + target_span
    if start < clean_start:
        start = clean_start
        end = start + target_span
    if end > clean_end:
        end = clean_end
        start = end - target_span
    if not (clean_start <= start <= center_frame <= end <= clean_end):
        return None
    trimmed_left = clean_start > shot_start_frame
    trimmed_right = clean_end < shot_end_frame
    if target_span < preferred_span:
        if trimmed_left and trimmed_right:
            reason = "TRIMMED_BOTH"
        elif trimmed_left:
            reason = "TRIMMED_LEFT"
        elif trimmed_right:
            reason = "TRIMMED_RIGHT"
        else:
            reason = "TRIMMED_BOTH"
    elif start != nominal_start or end != nominal_end:
        reason = "RECENTERED"
    else:
        reason = "NONE"
    return WindowAdjustment(
        start_frame=int(start),
        end_frame=int(end),
        requested_duration_seconds=PREFERRED_WINDOW_SECONDS,
        final_duration_seconds=(end - start) / fps,
        reason=reason,
    )


def temporal_iou(
    first_start: int, first_end: int, second_start: int, second_end: int
) -> float:
    intersection = max(0, min(first_end, second_end) - max(first_start, second_start) + 1)
    union = max(first_end, second_end) - min(first_start, second_start) + 1
    return float(intersection / union) if union else 0.0


def is_seed_near_duplicate(
    *,
    video_id: str,
    start_frame: int,
    end_frame: int,
    center_frame: int,
    fps: float,
    seeds: list[dict[str, Any]],
) -> bool:
    for seed in seeds:
        if str(seed["video_id"]) != video_id:
            continue
        if (
            temporal_iou(
                start_frame,
                end_frame,
                int(seed["window_start_frame"]),
                int(seed["window_end_frame"]),
            )
            > SEED_IOU_EXCLUSION
            or abs(center_frame - int(seed["proposal_center_frame"])) / fps
            < SEED_CENTER_EXCLUSION_SECONDS
        ):
            return True
    return False


def dense_focus_frame(
    series: SignalSeries, start_frame: int, end_frame: int
) -> int:
    mask = (series.frame_indices >= start_frame) & (series.frame_indices <= end_frame)
    positions = np.flatnonzero(mask)
    if not len(positions):
        raise ValueError("dense-focus window does not intersect signal series")
    score = (
        np.maximum(series.pixel_robust_z[positions], 0.0)
        + np.maximum(series.histogram_robust_z[positions], 0.0)
        + series.spatial_concentrations[positions]
    )
    best = positions[int(np.argmax(score))]
    return int(series.frame_indices[best])


def uniform_frame_indices(start: int, end: int, count: int) -> list[int]:
    if start < 0 or start > end or count < 2:
        raise ValueError("invalid frame sampling request")
    actual_count = min(count, end - start + 1)
    values = np.rint(np.linspace(start, end, actual_count)).astype(np.int64)
    output = sorted(set(int(value) for value in values))
    if len(output) != actual_count or not all(start <= value <= end for value in output):
        raise RuntimeError("MB1 v0.2.1 displayed frame sampling is not exact")
    return output


def overview_displayed_frames(start: int, end: int) -> list[int]:
    return uniform_frame_indices(start, end, OVERVIEW_FRAME_COUNT)


def dense_displayed_frames(
    start: int, end: int, focus_frame: int, fps: float
) -> list[int]:
    half_span = max(1, int(round(DENSE_SECONDS * fps)) // 2)
    dense_start = max(start, focus_frame - half_span)
    dense_end = min(end, dense_start + 2 * half_span)
    dense_start = max(start, dense_end - 2 * half_span)
    return uniform_frame_indices(dense_start, dense_end, DENSE_FRAME_COUNT)


__all__ = [
    "COARSE_SAMPLES_PER_SECOND",
    "DENSE_FRAME_COUNT",
    "DENSE_SECONDS",
    "EPSILON",
    "GRID_SHAPE",
    "LOCAL_EXPANSION_SECONDS",
    "LOCAL_SAMPLES_PER_SECOND",
    "LocalContinuityResult",
    "MAX_LOCAL_PROPOSALS_PER_VIDEO",
    "MAX_NEW_CANDIDATES_PER_VIDEO",
    "MINIMUM_WINDOW_SECONDS",
    "NMS_SECONDS",
    "OVERVIEW_FRAME_COUNT",
    "PREFERRED_WINDOW_SECONDS",
    "SCAN_SIZE",
    "SEED_CENTER_EXCLUSION_SECONDS",
    "SEED_IOU_EXCLUSION",
    "SignalBaseline",
    "SignalSeries",
    "WindowAdjustment",
    "adaptive_window",
    "dense_displayed_frames",
    "dense_focus_frame",
    "detect_soft_transition_runs",
    "empirical_percentiles",
    "hard_cut_mask",
    "is_seed_near_duplicate",
    "local_stride_frames",
    "overview_displayed_frames",
    "robust_baseline",
    "robust_z",
    "scan_coarse_video",
    "scan_local_video",
    "scan_stride_frames",
    "spatial_activity_concentration",
    "temporal_iou",
    "uniform_frame_indices",
]
