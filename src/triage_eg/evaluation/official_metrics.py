"""Transparent v0.1 simulations of task R-Score contracts."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any

AnswerMatcher = Callable[[str, Sequence[str]], bool]


def normalize_answer(answer: str) -> str:
    """Strip, lowercase, and collapse whitespace for template Q&A matching."""

    return re.sub(r"\s+", " ", answer.strip().lower())


def exact_normalized_answer_match(answer: str, accepted_answers: Sequence[str]) -> bool:
    """Match exact normalized text; this is replaceable and not a semantic evaluator."""

    normalized = normalize_answer(answer)
    return normalized in {normalize_answer(item) for item in accepted_answers}


def kis_rscore(prediction: Mapping[str, Any], ground_truth: Mapping[str, Any]) -> float:
    """Return one for the correct video and inclusive frame interval."""

    correct = (
        prediction.get("video_id") == ground_truth.get("video_id")
        and ground_truth["start_frame"]
        <= prediction.get("frame_id", -1)
        <= ground_truth["end_frame"]
    )
    return float(correct)


def qa_rscore(
    prediction: Mapping[str, Any],
    ground_truth: Mapping[str, Any],
    answer_matcher: AnswerMatcher = exact_normalized_answer_match,
) -> float:
    """Score video, frame interval, and replaceable normalized answer match."""

    location_score = kis_rscore(prediction, ground_truth)
    answer = prediction.get("answer")
    accepted = ground_truth.get("accepted_answers", [])
    if not isinstance(answer, str) or not isinstance(accepted, list):
        return 0.0
    return float(location_score == 1.0 and answer_matcher(answer, accepted))


def trake_rscore(prediction: Mapping[str, Any], ground_truth: Mapping[str, Any]) -> float:
    """Score corresponding event frames against inclusive ground-truth intervals."""

    intervals = ground_truth.get("intervals")
    frame_ids = prediction.get("frame_ids")
    if not isinstance(intervals, list) or not intervals:
        raise ValueError("TRAKE ground truth must contain at least one interval")
    if not isinstance(frame_ids, list) or len(frame_ids) != len(intervals):
        raise ValueError("TRAKE prediction frame count must equal interval count")
    if prediction.get("video_id") != ground_truth.get("video_id"):
        return 0.0
    correct = 0
    for frame_id, interval in zip(frame_ids, intervals, strict=True):
        if not isinstance(interval, list | tuple) or len(interval) != 2:
            raise ValueError("Every TRAKE interval must contain [start_frame, end_frame]")
        start_frame, end_frame = interval
        correct += int(start_frame <= frame_id <= end_frame)
    return correct / len(intervals)


def best_rscore_at_k(scores: Sequence[float], k: int) -> float:
    """Return the highest R-Score among the first ``k`` ranked predictions."""

    if k <= 0:
        raise ValueError("k must be greater than zero")
    return max(scores[:k], default=0.0)


def final_score(
    scores: Sequence[float] | Mapping[int, float], ks: Sequence[int] = (1, 5, 20, 50, 100)
) -> float:
    """Average best R-Score at the five qualification cutoffs."""

    values = (
        [float(scores[k]) for k in ks]
        if isinstance(scores, Mapping)
        else [best_rscore_at_k(scores, k) for k in ks]
    )
    return sum(values) / len(values)
