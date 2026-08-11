"""Ground-truth-free temporal plateau solver for bounded M2 moment localization."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

SMOOTHING_SECONDS = 0.16
HALF_PROMINENCE = 0.5
LOW_CONTRAST_EPSILON = 1e-8

PEAK_TYPES = frozenset(
    {"ACTION_VISIBILITY", "STATE", "RELATION_SATISFIED", "COUNT_STABLE"}
)
LEFT_EDGE_TYPES = frozenset(
    {"FIRST_OCCURRENCE", "TRANSITION_ONSET", "CONTACT", "SEPARATION"}
)
RIGHT_EDGE_TYPES = frozenset({"LAST_OCCURRENCE", "TRANSITION_OFFSET"})
EXTREMUM_TYPE = "EXTREMUM"
SUPPORTED_MOMENT_TYPES = PEAK_TYPES | LEFT_EDGE_TYPES | RIGHT_EDGE_TYPES | {
    EXTREMUM_TYPE
}


@dataclass(frozen=True)
class PlateauSolution:
    """Complete deterministic M2 solution over one dense raw-frame score curve."""

    smoothing_width_frames: int
    smoothed_clip_scores: tuple[float, ...]
    raw_dense_peak_frame: int
    smoothed_peak_frame: int
    baseline_score: float
    peak_score: float
    prominence: float
    plateau_threshold: float | None
    plateau_start_frame: int
    plateau_end_frame: int
    plateau_duration_frames: int
    plateau_duration_seconds: float
    prediction: int
    diagnostics: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def smoothing_width_frames(fps: float, seconds: float = SMOOTHING_SECONDS) -> int:
    """Convert seconds to the nearest practical odd frame width, with minimum three."""

    if not math.isfinite(fps) or fps <= 0 or seconds != SMOOTHING_SECONDS:
        raise ValueError("M2 freezes smoothing_seconds=0.16 and requires positive FPS")
    rounded = max(3, int(math.floor(fps * seconds + 0.5)))
    if rounded % 2 == 0:
        rounded += 1
    return rounded


def centered_moving_average(scores: np.ndarray, width: int) -> np.ndarray:
    """Average only available samples at sequence edges; never synthesize padding."""

    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("M2 smoothing requires a finite non-empty score vector")
    if width < 3 or width % 2 == 0:
        raise ValueError("M2 smoothing width must be odd and at least three")
    radius = width // 2
    prefix = np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))
    output = np.empty_like(values)
    for index in range(len(values)):
        start = max(0, index - radius)
        stop = min(len(values), index + radius + 1)
        output[index] = (prefix[stop] - prefix[start]) / (stop - start)
    return output


def select_dense_peak(frame_indices: np.ndarray, scores: np.ndarray) -> tuple[int, float]:
    """Select maximum score with lower actual frame index as the deterministic tie-break."""

    frames = np.asarray(frame_indices, dtype=np.int64)
    values = np.asarray(scores, dtype=np.float64)
    if (
        frames.ndim != 1
        or values.shape != frames.shape
        or not len(frames)
        or not np.isfinite(values).all()
        or len(set(int(value) for value in frames)) != len(frames)
    ):
        raise ValueError("M2 peak inputs are invalid")
    order = np.lexsort((frames, -values))
    index = int(order[0])
    return int(frames[index]), float(values[index])


def solve_moment_plateau(
    frame_indices: np.ndarray,
    raw_clip_scores: np.ndarray,
    fps: float,
    moment_type: str,
) -> PlateauSolution:
    """Solve one M2 plateau without accepting annotation/evaluation fields."""

    frames = np.asarray(frame_indices, dtype=np.int64)
    raw = np.asarray(raw_clip_scores, dtype=np.float64)
    if (
        frames.ndim != 1
        or raw.shape != frames.shape
        or not len(frames)
        or not np.isfinite(raw).all()
        or not np.all(np.diff(frames) == 1)
    ):
        raise ValueError("M2 requires a finite, consecutive dense raw-frame curve")
    width = smoothing_width_frames(fps)
    smoothed = centered_moving_average(raw, width)
    return solve_plateau_from_smoothed(frames, raw, smoothed, fps, moment_type, width)


def solve_plateau_from_smoothed(
    frame_indices: np.ndarray,
    raw_clip_scores: np.ndarray,
    smoothed_clip_scores: np.ndarray,
    fps: float,
    moment_type: str,
    smoothing_width: int,
) -> PlateauSolution:
    """Apply half-prominence/routing to an already computed deterministic curve."""

    frames = np.asarray(frame_indices, dtype=np.int64)
    raw = np.asarray(raw_clip_scores, dtype=np.float64)
    smoothed = np.asarray(smoothed_clip_scores, dtype=np.float64)
    if (
        frames.ndim != 1
        or raw.shape != frames.shape
        or smoothed.shape != frames.shape
        or not len(frames)
        or not np.isfinite(raw).all()
        or not np.isfinite(smoothed).all()
        or not np.all(np.diff(frames) == 1)
        or smoothing_width < 3
        or smoothing_width % 2 == 0
    ):
        raise ValueError("M2 plateau inputs are invalid")
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("M2 plateau requires positive FPS")
    if moment_type not in SUPPORTED_MOMENT_TYPES:
        raise ValueError(f"M2_UNKNOWN_MOMENT_TYPE: {moment_type}")

    raw_peak, _ = select_dense_peak(frames, raw)
    smooth_peak, peak_score = select_dense_peak(frames, smoothed)
    peak_index = int(np.flatnonzero(frames == smooth_peak)[0])
    baseline = float(np.median(smoothed))
    prominence = float(peak_score - baseline)
    diagnostics: list[str] = []

    if prominence <= LOW_CONTRAST_EPSILON:
        diagnostics.append("LOW_CONTRAST_CURVE")
        if moment_type == EXTREMUM_TYPE:
            diagnostics.append("EXTREMUM_TEMPORAL_SOLVER_NOT_IMPLEMENTED")
        plateau_threshold = None
        plateau_start = smooth_peak
        plateau_end = smooth_peak
        prediction = raw_peak
    else:
        plateau_threshold = float(baseline + HALF_PROMINENCE * prominence)
        above = smoothed >= plateau_threshold
        start_index = peak_index
        while start_index > 0 and bool(above[start_index - 1]):
            start_index -= 1
        end_index = peak_index
        while end_index + 1 < len(frames) and bool(above[end_index + 1]):
            end_index += 1
        plateau_start = int(frames[start_index])
        plateau_end = int(frames[end_index])
        if plateau_start == int(frames[0]):
            diagnostics.append("PLATEAU_TOUCHES_WINDOW_START")
        if plateau_end == int(frames[-1]):
            diagnostics.append("PLATEAU_TOUCHES_WINDOW_END")

        if moment_type in PEAK_TYPES:
            prediction, _ = select_dense_peak(
                frames[start_index : end_index + 1], raw[start_index : end_index + 1]
            )
        elif moment_type in LEFT_EDGE_TYPES:
            prediction = plateau_start
        elif moment_type in RIGHT_EDGE_TYPES:
            prediction = plateau_end
        else:
            prediction = raw_peak
            diagnostics.append("EXTREMUM_TEMPORAL_SOLVER_NOT_IMPLEMENTED")

    duration_frames = plateau_end - plateau_start + 1
    return PlateauSolution(
        smoothing_width_frames=smoothing_width,
        smoothed_clip_scores=tuple(float(value) for value in smoothed),
        raw_dense_peak_frame=raw_peak,
        smoothed_peak_frame=smooth_peak,
        baseline_score=baseline,
        peak_score=peak_score,
        prominence=prominence,
        plateau_threshold=plateau_threshold,
        plateau_start_frame=plateau_start,
        plateau_end_frame=plateau_end,
        plateau_duration_frames=duration_frames,
        plateau_duration_seconds=duration_frames / fps,
        prediction=prediction,
        diagnostics=tuple(diagnostics),
    )


__all__ = [
    "EXTREMUM_TYPE",
    "HALF_PROMINENCE",
    "LEFT_EDGE_TYPES",
    "LOW_CONTRAST_EPSILON",
    "PEAK_TYPES",
    "PlateauSolution",
    "RIGHT_EDGE_TYPES",
    "SMOOTHING_SECONDS",
    "SUPPORTED_MOMENT_TYPES",
    "centered_moving_average",
    "select_dense_peak",
    "smoothing_width_frames",
    "solve_plateau_from_smoothed",
    "solve_moment_plateau",
]
