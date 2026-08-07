from dataclasses import dataclass
from typing import Any

from .schemas import KISPrediction, QAPrediction, TRAKEGroundTruth, TRAKEPrediction


@dataclass(frozen=True, slots=True)
class ValidationError:
    message: str


def validate_ranked_top100(
    predictions: list[Any], expected_task: str, gt: Any = None
) -> list[ValidationError]:
    errors = []

    if len(predictions) > 100:
        errors.append(ValidationError("Cannot exceed 100 predictions"))

    if not predictions:
        return errors

    query_ids = set()
    ranks = set()
    keys = set()

    for p in predictions:
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
                from .matching import NormalizedAliasAnswerMatcher

                matcher = NormalizedAliasAnswerMatcher(strip_punctuation=True)
                normalized = matcher._normalize(p.answer)
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
