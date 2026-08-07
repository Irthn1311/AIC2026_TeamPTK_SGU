from dataclasses import dataclass
from typing import Any

from .matching import NormalizedAliasAnswerMatcher
from .schemas import (
    KISGroundTruth,
    KISPrediction,
    QAGroundTruth,
    QAPrediction,
    TRAKEGroundTruth,
    TRAKEPrediction,
)


@dataclass(frozen=True, slots=True)
class ValidationError:
    message: str


TASK_PREDICTION_TYPES = {
    "kis": KISPrediction,
    "qa": QAPrediction,
    "trake": TRAKEPrediction,
}

TASK_GROUND_TRUTH_TYPES = {
    "kis": KISGroundTruth,
    "qa": QAGroundTruth,
    "trake": TRAKEGroundTruth,
}


def validate_ranked_top100(
    predictions: list[Any],
    expected_task: str,
    gt: Any = None,
    expected_query_id: str | None = None,
) -> list[ValidationError]:
    errors = []

    # 1. Supported task type check
    if expected_task not in TASK_PREDICTION_TYPES:
        errors.append(ValidationError(f"Unknown task type: {expected_task}"))
        return errors

    # 2. Ground truth concrete schema matches expected task
    if gt is not None:
        expected_gt_cls = TASK_GROUND_TRUTH_TYPES[expected_task]
        if not isinstance(gt, expected_gt_cls):
            errors.append(
                ValidationError(
                    f"Ground truth type mismatch for task '{expected_task}': "
                    f"expected {expected_gt_cls.__name__}, got {type(gt).__name__}"
                )
            )

    # 3. Ground truth query_id matches expected_query_id
    if gt is not None and expected_query_id is not None:
        if gt.query_id != expected_query_id:
            errors.append(
                ValidationError(
                    f"Ground truth query_id mismatch: expected {expected_query_id}, "
                    f"got {gt.query_id}"
                )
            )

    # 4. Max 100 predictions check
    if len(predictions) > 100:
        errors.append(ValidationError("Cannot exceed 100 predictions"))

    # 5. Return accumulated errors if predictions list is empty
    if not predictions:
        return errors

    query_ids = set()
    ranks = set()
    keys = set()
    matcher = NormalizedAliasAnswerMatcher(strip_punctuation=True)

    for p in predictions:
        if expected_query_id is not None and p.query_id != expected_query_id:
            errors.append(
                ValidationError(
                    f"Prediction query_id mismatch: expected {expected_query_id}, got {p.query_id}"
                )
            )

        if p.rank <= 0:
            errors.append(ValidationError(f"Rank {p.rank} is <= 0"))
        if p.rank in ranks:
            errors.append(ValidationError(f"Duplicate rank: {p.rank}"))
        ranks.add(p.rank)

        query_ids.add(p.query_id)

        expected_pred_cls = TASK_PREDICTION_TYPES[expected_task]
        if not isinstance(p, expected_pred_cls):
            errors.append(
                ValidationError(f"Wrong prediction type for {expected_task.upper()}")
            )
            continue

        if expected_task == "kis":
            key = (p.video_id, p.frame_id)
            if key in keys:
                errors.append(ValidationError(f"Duplicate KIS key: {key}"))
            keys.add(key)

        elif expected_task == "qa":
            if not p.answer.strip():
                errors.append(ValidationError("Empty answer for QA"))
            normalized = matcher.normalize(p.answer)
            key = (p.video_id, p.frame_id, normalized)
            if key in keys:
                errors.append(ValidationError(f"Duplicate QA key: {key}"))
            keys.add(key)

        elif expected_task == "trake":
            if not p.frame_ids:
                errors.append(ValidationError("Empty TRAKE frame_ids"))
            if gt is not None and isinstance(gt, TRAKEGroundTruth):
                if len(p.frame_ids) != len(gt.event_intervals):
                    errors.append(ValidationError("TRAKE event-count mismatch"))
            key = (p.video_id, p.frame_ids)
            if key in keys:
                errors.append(ValidationError(f"Duplicate TRAKE key: {key}"))
            keys.add(key)

    if len(query_ids) > 1:
        errors.append(ValidationError("Mixed query_ids found"))

    return errors
