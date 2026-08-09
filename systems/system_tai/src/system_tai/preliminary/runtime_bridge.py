"""Fail-closed bridge from runtime results to the canonical P0-D boundary."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from system_tai.common.schemas import KISResult

from .schemas import KISPrediction, QAPrediction, TRAKEPrediction
from .top100 import (
    RankedTop100Dataset,
    RankedTop100Query,
    TaskType,
    load_top100_jsonl,
    validate_top100_query,
)

RuntimeArtifactRoundtripStatus = Literal["EXACT", "EMPTY_QUERY_UNREPRESENTABLE"]


class RuntimeTop100MismatchError(ValueError):
    """Runtime memory and its existing prediction artifact do not match exactly."""


@dataclass(frozen=True, slots=True)
class RuntimeTop100Audit:
    task_type: TaskType
    query_id: str
    prediction_count: int
    artifact_path: Path
    roundtrip_status: RuntimeArtifactRoundtripStatus
    loaded_dataset: RankedTop100Dataset | None = None


def _bridge_error(
    message: str,
    *,
    task_type: str,
    query_id: str,
    artifact_path: Path | None = None,
) -> RuntimeTop100MismatchError:
    context = f"task_type={task_type!r}, query_id={query_id!r}"
    if artifact_path is not None:
        context += f", artifact_path={str(artifact_path)!r}"
    return RuntimeTop100MismatchError(f"{message} ({context})")


def kis_result_to_top100_query(result: KISResult) -> RankedTop100Query:
    """Convert a runtime KIS result without leaking retrieval-only fields."""

    if type(result) is not KISResult:
        raise _bridge_error(
            f"result must be KISResult, got {type(result).__name__}",
            task_type="kis",
            query_id=getattr(result, "query_id", "<unknown>"),
        )
    predictions = tuple(
        KISPrediction(
            query_id=result.query_id,
            rank=candidate.rank,
            video_id=candidate.video_id,
            frame_id=candidate.frame_id,
        )
        for candidate in result.ranked_candidates
    )
    try:
        return RankedTop100Query("kis", result.query_id, predictions)
    except (TypeError, ValueError) as exc:
        raise _bridge_error(
            f"invalid runtime KIS predictions: {exc}",
            task_type="kis",
            query_id=result.query_id,
        ) from exc


def qa_predictions_to_top100_query(
    *,
    query_id: str,
    predictions: Sequence[QAPrediction],
) -> RankedTop100Query:
    """Wrap exact runtime Q&A predictions in their physical sequence order."""

    resolved = tuple(predictions)
    for index, prediction in enumerate(resolved):
        if type(prediction) is not QAPrediction:
            raise _bridge_error(
                f"prediction {index} must be QAPrediction, got {type(prediction).__name__}",
                task_type="qa",
                query_id=query_id,
            )
        if prediction.query_id != query_id:
            raise _bridge_error(
                f"prediction {index} query_id mismatch: {prediction.query_id!r}",
                task_type="qa",
                query_id=query_id,
            )
    try:
        return RankedTop100Query("qa", query_id, resolved)
    except (TypeError, ValueError) as exc:
        raise _bridge_error(
            f"invalid runtime Q&A predictions: {exc}",
            task_type="qa",
            query_id=query_id,
        ) from exc


def trake_predictions_to_top100_query(
    *,
    query_id: str,
    predictions: Sequence[TRAKEPrediction],
    expected_event_count: int,
) -> RankedTop100Query:
    """Wrap exact runtime TRAKE predictions without reordering event frames."""

    resolved = tuple(predictions)
    for index, prediction in enumerate(resolved):
        if type(prediction) is not TRAKEPrediction:
            raise _bridge_error(
                f"prediction {index} must be TRAKEPrediction, got {type(prediction).__name__}",
                task_type="trake",
                query_id=query_id,
            )
        if prediction.query_id != query_id:
            raise _bridge_error(
                f"prediction {index} query_id mismatch: {prediction.query_id!r}",
                task_type="trake",
                query_id=query_id,
            )
    try:
        query = RankedTop100Query("trake", query_id, resolved)
        validate_top100_query(query, expected_trake_event_count=expected_event_count)
    except (TypeError, ValueError) as exc:
        raise _bridge_error(
            f"invalid runtime TRAKE predictions: {exc}",
            task_type="trake",
            query_id=query_id,
        ) from exc
    return query


def audit_runtime_top100_artifact(
    query: RankedTop100Query,
    artifact_path: Path,
    *,
    expected_trake_event_count: int | None = None,
) -> RuntimeTop100Audit:
    """Require exact equality between canonical runtime memory and existing JSONL."""

    path = Path(artifact_path)
    try:
        validate_top100_query(
            query,
            expected_trake_event_count=expected_trake_event_count,
        )
    except (TypeError, ValueError) as exc:
        raise _bridge_error(
            f"invalid canonical runtime query: {exc}",
            task_type=query.task_type,
            query_id=query.query_id,
            artifact_path=path,
        ) from exc

    if not query.predictions:
        try:
            payload = path.read_bytes()
        except OSError as exc:
            raise _bridge_error(
                f"cannot read runtime prediction artifact: {exc}",
                task_type=query.task_type,
                query_id=query.query_id,
                artifact_path=path,
            ) from exc
        try:
            text = payload.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise _bridge_error(
                f"runtime prediction artifact is not valid UTF-8: {exc}",
                task_type=query.task_type,
                query_id=query.query_id,
                artifact_path=path,
            ) from exc
        if text.startswith("\ufeff"):
            raise _bridge_error(
                "UTF-8 BOM is not permitted",
                task_type=query.task_type,
                query_id=query.query_id,
                artifact_path=path,
            )
        if any(line.strip() for line in text.splitlines()):
            raise _bridge_error(
                "zero-prediction query artifact contains a non-empty record",
                task_type=query.task_type,
                query_id=query.query_id,
                artifact_path=path,
            )
        return RuntimeTop100Audit(
            task_type=query.task_type,
            query_id=query.query_id,
            prediction_count=0,
            artifact_path=path,
            roundtrip_status="EMPTY_QUERY_UNREPRESENTABLE",
        )

    expected_dataset = RankedTop100Dataset(query.task_type, (query,))
    expected_event_counts = None
    if expected_trake_event_count is not None:
        expected_event_counts = {query.query_id: expected_trake_event_count}
    try:
        loaded_dataset = load_top100_jsonl(
            path,
            task_type=query.task_type,
            expected_query_ids=(query.query_id,),
            expected_trake_event_counts=expected_event_counts,
        )
    except (TypeError, ValueError) as exc:
        raise _bridge_error(
            f"strict P0-D artifact load failed: {exc}",
            task_type=query.task_type,
            query_id=query.query_id,
            artifact_path=path,
        ) from exc
    if loaded_dataset != expected_dataset:
        raise _bridge_error(
            "runtime prediction artifact differs from canonical in-memory predictions",
            task_type=query.task_type,
            query_id=query.query_id,
            artifact_path=path,
        )
    return RuntimeTop100Audit(
        task_type=query.task_type,
        query_id=query.query_id,
        prediction_count=len(query.predictions),
        artifact_path=path,
        roundtrip_status="EXACT",
        loaded_dataset=loaded_dataset,
    )
