"""Type-specific semantic state-transition solver for bounded M3."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

RADIUS_SECONDS = 1.5
SMOOTH_SECONDS = 0.12
CONTEXT_SECONDS = 0.20
PERSISTENCE_SECONDS = 0.20
PRE_HIGH_FRACTION_MAX = 0.40
POST_HIGH_FRACTION_MIN = 0.60
FIRST_OCCURRENCE_RELATIVE_CHANGE = 0.75
MOTION_WEIGHT = 0.10
LOW_INFORMATION_EPSILON = 1e-6

BOUNDARY_TYPES = frozenset({"ONSET", "CONTACT", "FIRST_OCCURRENCE", "SEPARATION"})
M1_ROUTED_TYPES = frozenset({"ACTION_VISIBILITY", "STATE", "STATE_CONTROL", "CONTROL"})


@dataclass(frozen=True)
class M3Settings:
    radius_seconds: float = RADIUS_SECONDS
    smooth_seconds: float = SMOOTH_SECONDS
    context_seconds: float = CONTEXT_SECONDS
    persistence_seconds: float = PERSISTENCE_SECONDS
    pre_high_fraction_max: float = PRE_HIGH_FRACTION_MAX
    post_high_fraction_min: float = POST_HIGH_FRACTION_MIN
    first_occurrence_relative_change: float = FIRST_OCCURRENCE_RELATIVE_CHANGE
    motion_weight: float = MOTION_WEIGHT
    low_information_epsilon: float = LOW_INFORMATION_EPSILON

    def __post_init__(self) -> None:
        frozen = {
            "radius_seconds": RADIUS_SECONDS,
            "smooth_seconds": SMOOTH_SECONDS,
            "context_seconds": CONTEXT_SECONDS,
            "persistence_seconds": PERSISTENCE_SECONDS,
            "pre_high_fraction_max": PRE_HIGH_FRACTION_MAX,
            "post_high_fraction_min": POST_HIGH_FRACTION_MIN,
            "first_occurrence_relative_change": FIRST_OCCURRENCE_RELATIVE_CHANGE,
            "motion_weight": MOTION_WEIGHT,
            "low_information_epsilon": LOW_INFORMATION_EPSILON,
        }
        if asdict(self) != frozen:
            raise ValueError("M3 parameters are frozen; parameter sweeps are forbidden")


@dataclass(frozen=True)
class M3InferenceCase:
    """GT-free inference contract; accepted intervals cannot enter this API."""

    case_id: str
    video_id: str
    moment_type: str
    semantic_event_en: str
    before_state_en: str | None
    after_state_en: str | None
    candidate_anchor_frame: int


@dataclass(frozen=True)
class TransitionSolution:
    prediction: int
    used_m1_fallback: bool
    fallback_reason: str | None
    signal_range: float
    selected_change_score: float | None
    pre_high_fraction: float | None
    post_high_fraction: float | None
    motion_value_at_selection: float | None
    selected_index: int | None
    valid_candidate_count: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def local_window(
    anchor_frame: int, *, fps: float, total_frames: int, radius_seconds: float = RADIUS_SECONDS
) -> tuple[int, int]:
    if radius_seconds != RADIUS_SECONDS:
        raise ValueError("M3 radius_seconds is frozen at 1.5")
    if total_frames <= 0 or not math.isfinite(fps) or fps <= 0:
        raise ValueError("M3 requires valid raw-video metadata")
    if not 0 <= anchor_frame < total_frames:
        raise IndexError("M3 candidate anchor is outside the raw video")
    radius = max(1, int(round(radius_seconds * fps)))
    return max(0, anchor_frame - radius), min(total_frames - 1, anchor_frame + radius)


def _cosine_scores(embeddings: np.ndarray, text_embedding: np.ndarray) -> np.ndarray:
    images = np.asarray(embeddings, dtype=np.float64)
    text = np.asarray(text_embedding, dtype=np.float64).reshape(-1)
    if images.ndim != 2 or images.shape[1:] != text.shape or not np.isfinite(images).all():
        raise ValueError("M3 image/text embeddings are invalid")
    image_norms = np.linalg.norm(images, axis=1)
    text_norm = float(np.linalg.norm(text))
    if text_norm == 0 or np.any(image_norms == 0):
        raise ValueError("M3 cosine inputs must have non-zero norms")
    return (images @ text) / (image_norms * text_norm)


def build_state_signals(
    image_embeddings: np.ndarray,
    *,
    before_text_embedding: np.ndarray,
    after_text_embedding: np.ndarray,
    event_text_embedding: np.ndarray,
) -> dict[str, np.ndarray]:
    before = _cosine_scores(image_embeddings, before_text_embedding)
    after = _cosine_scores(image_embeddings, after_text_embedding)
    event = _cosine_scores(image_embeddings, event_text_embedding)
    return {
        "before": before,
        "after": after,
        "contrast": after - before,
        "event": event,
    }


def odd_window_frames(fps: float, seconds: float = SMOOTH_SECONDS) -> int:
    if not math.isfinite(fps) or fps <= 0 or seconds != SMOOTH_SECONDS:
        raise ValueError("M3 smoothing is frozen at 0.12 seconds")
    width = max(1, int(round(fps * seconds)))
    return width if width % 2 else width + 1


def moving_median(values: np.ndarray, width: int) -> np.ndarray:
    signal = np.asarray(values, dtype=np.float64)
    if signal.ndim != 1 or not len(signal) or not np.isfinite(signal).all():
        raise ValueError("M3 smoothing requires a finite one-dimensional signal")
    if width <= 0 or width % 2 == 0:
        raise ValueError("M3 median width must be positive and odd")
    radius = width // 2
    return np.asarray(
        [
            np.median(signal[max(0, index - radius) : min(len(signal), index + radius + 1)])
            for index in range(len(signal))
        ],
        dtype=np.float64,
    )


def adjacent_embedding_motion(image_embeddings: np.ndarray) -> np.ndarray:
    values = np.asarray(image_embeddings, dtype=np.float64)
    if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
        raise ValueError("M3 motion requires finite image embeddings")
    norms = np.linalg.norm(values, axis=1)
    if np.any(norms == 0):
        raise ValueError("M3 motion embeddings must have non-zero norms")
    normalized = values / norms[:, None]
    motion = np.zeros(len(values), dtype=np.float64)
    if len(values) > 1:
        motion[1:] = 1.0 - np.sum(normalized[1:] * normalized[:-1], axis=1)
    return motion


def _robust_unit(values: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    source = np.asarray(values, dtype=np.float64)
    basis = source if valid is None else source[valid]
    if not len(basis):
        return np.zeros_like(source)
    low, high = np.percentile(basis, [10, 90])
    if high - low <= LOW_INFORMATION_EPSILON:
        return np.zeros_like(source)
    return np.clip((source - low) / (high - low), 0.0, 1.0)


def solve_state_transition(
    case: M3InferenceCase,
    *,
    frame_ids: np.ndarray,
    contrast: np.ndarray,
    image_embeddings: np.ndarray,
    fps: float,
    m1_prediction: int,
    use_motion_tiebreaker: bool,
    settings: M3Settings | None = None,
) -> tuple[TransitionSolution, dict[str, np.ndarray]]:
    """Predict from GT-free signals; accepted intervals are not an argument."""
    active = settings or M3Settings()
    frames = np.asarray(frame_ids, dtype=np.int64)
    raw = np.asarray(contrast, dtype=np.float64)
    images = np.asarray(image_embeddings, dtype=np.float64)
    if (
        frames.ndim != 1
        or raw.shape != frames.shape
        or images.shape[0] != len(frames)
        or not len(frames)
        or not np.all(np.diff(frames) == 1)
    ):
        raise ValueError("M3 requires consecutive exact raw-frame coordinates")
    if case.moment_type == "EXTREMUM":
        solution = TransitionSolution(
            m1_prediction, True, "M3_EXTREMUM_UNSUPPORTED", 0.0, None, None, None, None, None, 0
        )
        return solution, {"raw_contrast": raw, "smoothed_contrast": raw.copy()}
    if case.moment_type in M1_ROUTED_TYPES or case.moment_type not in BOUNDARY_TYPES:
        solution = TransitionSolution(
            m1_prediction, True, "M3_TYPE_ROUTED_TO_M1", 0.0, None, None, None, None, None, 0
        )
        return solution, {"raw_contrast": raw, "smoothed_contrast": raw.copy()}

    smoothed = moving_median(raw, odd_window_frames(fps, active.smooth_seconds))
    low, high = (float(value) for value in np.percentile(smoothed, [10, 90]))
    signal_range = high - low
    if signal_range <= active.low_information_epsilon:
        solution = TransitionSolution(
            m1_prediction,
            True,
            "LOW_INFORMATION_SIGNAL",
            signal_range,
            None,
            None,
            None,
            None,
            None,
            0,
        )
        return solution, {"raw_contrast": raw, "smoothed_contrast": smoothed}

    context = max(1, int(round(fps * active.context_seconds)))
    persistence = max(1, int(round(fps * active.persistence_seconds)))
    midpoint = low + 0.5 * signal_range
    change = np.full(len(frames), np.nan, dtype=np.float64)
    pre_fraction = np.full(len(frames), np.nan, dtype=np.float64)
    post_fraction = np.full(len(frames), np.nan, dtype=np.float64)
    eligible = np.zeros(len(frames), dtype=bool)
    margin = max(context, persistence)
    for index in range(margin, len(frames) - margin + 1):
        change[index] = float(
            np.mean(smoothed[index : index + context]) - np.mean(smoothed[index - context : index])
        )
        pre_fraction[index] = float(np.mean(smoothed[index - persistence : index] >= midpoint))
        post_fraction[index] = float(np.mean(smoothed[index : index + persistence] >= midpoint))
        eligible[index] = (
            pre_fraction[index] <= active.pre_high_fraction_max
            and post_fraction[index] >= active.post_high_fraction_min
        )
    positive = np.isfinite(change) & (change > 0)
    valid = eligible & positive
    motion = adjacent_embedding_motion(images)
    finite_change = np.isfinite(change)
    state_score = _robust_unit(np.nan_to_num(change, nan=0.0), finite_change)
    motion_score = _robust_unit(motion)
    final_score = state_score + active.motion_weight * motion_score
    selection_score = final_score if use_motion_tiebreaker else change

    fallback_reason = None
    used_m1 = False
    if case.moment_type == "FIRST_OCCURRENCE":
        if np.any(valid):
            maximum = float(np.max(change[valid]))
            candidates = np.flatnonzero(
                valid & (change >= active.first_occurrence_relative_change * maximum)
            )
            selected = int(candidates[0])
        else:
            selected = None
            fallback_reason = "NO_PERSISTENT_FIRST_OCCURRENCE"
            used_m1 = True
    elif np.any(valid):
        candidates = np.flatnonzero(valid)
        selected = min(
            candidates, key=lambda index: (-float(selection_score[index]), int(frames[index]))
        )
    elif np.any(positive):
        candidates = np.flatnonzero(positive)
        selected = min(
            candidates, key=lambda index: (-float(selection_score[index]), int(frames[index]))
        )
        fallback_reason = "NO_PERSISTENCE_VALID_CANDIDATE_USED_HIGHEST_CHANGE"
    else:
        selected = None
        fallback_reason = "NO_POSITIVE_STATE_TRANSITION"
        used_m1 = True

    prediction = m1_prediction if selected is None else int(frames[selected])
    solution = TransitionSolution(
        prediction=prediction,
        used_m1_fallback=used_m1,
        fallback_reason=fallback_reason,
        signal_range=signal_range,
        selected_change_score=float(change[selected]) if selected is not None else None,
        pre_high_fraction=float(pre_fraction[selected]) if selected is not None else None,
        post_high_fraction=float(post_fraction[selected]) if selected is not None else None,
        motion_value_at_selection=float(motion[selected]) if selected is not None else None,
        selected_index=selected,
        valid_candidate_count=int(np.sum(valid)),
    )
    return solution, {
        "raw_contrast": raw,
        "smoothed_contrast": smoothed,
        "change": change,
        "pre_high_fraction": pre_fraction,
        "post_high_fraction": post_fraction,
        "motion": motion,
        "normalized_state_change": state_score,
        "normalized_motion": motion_score,
        "motion_tiebreak_score": final_score,
    }


__all__ = [
    "BOUNDARY_TYPES",
    "CONTEXT_SECONDS",
    "FIRST_OCCURRENCE_RELATIVE_CHANGE",
    "LOW_INFORMATION_EPSILON",
    "M1_ROUTED_TYPES",
    "M3InferenceCase",
    "M3Settings",
    "MOTION_WEIGHT",
    "PERSISTENCE_SECONDS",
    "POST_HIGH_FRACTION_MIN",
    "PRE_HIGH_FRACTION_MAX",
    "RADIUS_SECONDS",
    "SMOOTH_SECONDS",
    "TransitionSolution",
    "adjacent_embedding_motion",
    "build_state_signals",
    "local_window",
    "moving_median",
    "odd_window_frames",
    "solve_state_transition",
]
