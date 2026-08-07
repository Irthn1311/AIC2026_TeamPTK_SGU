from dataclasses import dataclass
from typing import Any

from .matching import NormalizedAliasAnswerMatcher
from .schemas import KISPrediction, QAPrediction, TRAKEGroundTruth, TRAKEPrediction


@dataclass(frozen=True, slots=True)
class ValidationError:
    message: str


def validate_ranked_top100(
    predictions: list[Any],
    expected_task: str,
    gt: Any = None,
    expected_query_id: str | None = None,
) -> list[ValidationError]:
    errors = []

    if len(predictions) > 100:
        errors.append(ValidationError("Cannot exceed 100 predictions"))

    if gt is not None and expected_query_id is not None:
        if gt.query_id != expected_query_id:
            errors.append(
                ValidationError(
                    f"Ground truth query_id mismatch: expected {expected_query_id}, "
                    f"got {gt.query_id}"
                )
            )

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

        if expected_task == "kis":
            if not isinstance(p, KISPrediction):
                errors.append(ValidationError("Wrong prediction type for KIS"))
            else:
                key = (p.video_id, p.frame_id)
                if key in keys:
                    errors.append(ValidationError(f"Duplicate KIS key: {key}"))
                keys.add(key)

        elif expected_task == "qa":
            if not isinstance(p, QAPrediction):
                errors.append(ValidationError("Wrong prediction type for QA"))
            else:
                if not p.answer.strip():
                    errors.append(ValidationError("Empty answer for QA"))
                normalized = matcher.normalize(p.answer)
                key = (p.video_id, p.frame_id, normalized)
                if key in keys:
                    errors.append(ValidationError(f"Duplicate QA key: {key}"))
                keys.add(key)

        elif expected_task == "trake":
            if not isinstance(p, TRAKEPrediction):
                errors.append(ValidationError("Wrong prediction type for TRAKE"))
            else:
                if not p.frame_ids:
                    errors.append(ValidationError("Empty TRAKE frame_ids"))
                if gt is not None and isinstance(gt, TRAKEGroundTruth):
                    if len(p.frame_ids) != len(gt.event_intervals):
                        errors.append(ValidationError("TRAKE event-count mismatch"))
                key = (p.video_id, p.frame_ids)
                if key in keys:
                    errors.append(ValidationError(f"Duplicate TRAKE key: {key}"))
                keys.add(key)
        else:
            errors.append(ValidationError(f"Unknown task type: {expected_task}"))

    if len(query_ids) > 1:
        errors.append(ValidationError("Mixed query_ids found"))

    return errors
