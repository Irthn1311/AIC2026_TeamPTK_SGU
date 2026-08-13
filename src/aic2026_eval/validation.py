"""Strict validation for team-neutral final prediction JSONL."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from .contracts import TASKS, VIDEO_ID_PATTERN, validate_query


def _issue(code: str, message: str, **context: Any) -> dict[str, Any]:
    return {"severity": "ERROR", "code": code, "message": message, **context}


def _integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def validate_predictions(
    queries: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    *,
    inventory: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    issues = []
    normalized_queries: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(queries, 1):
        try:
            query = validate_query(raw)
        except ValueError as error:
            issues.append(_issue("QUERY_INVALID", str(error), row=index))
            continue
        if query["query_id"] in normalized_queries:
            issues.append(
                _issue("QUERY_ID_DUPLICATE", "query_id must be unique", query_id=query["query_id"])
            )
        normalized_queries[query["query_id"]] = query
    bounds = (
        {row["video_id"]: int(row["total_frames"]) for row in inventory}
        if inventory is not None
        else None
    )
    counts: Counter[str] = Counter()
    ranks: dict[str, list[int]] = defaultdict(list)
    for row_number, row in enumerate(predictions, 1):
        if not isinstance(row, dict):
            issues.append(
                _issue("PREDICTION_NOT_OBJECT", "prediction must be an object", row=row_number)
            )
            continue
        query_id = row.get("query_id")
        query = normalized_queries.get(query_id)
        if query is None:
            issues.append(
                _issue("UNKNOWN_QUERY_ID", "prediction query_id is unknown", row=row_number)
            )
            continue
        task = query["task"]
        rank = row.get("rank")
        video_id = row.get("video_id")
        if not _integer(rank) or not 1 <= rank <= 100:
            issues.append(
                _issue("RANK_INVALID", "rank must be an integer from 1 to 100", row=row_number)
            )
        else:
            ranks[query_id].append(rank)
        counts[query_id] += 1
        if not isinstance(video_id, str) or not VIDEO_ID_PATTERN.fullmatch(video_id):
            issues.append(_issue("VIDEO_ID_INVALID", "video_id is invalid", row=row_number))
            continue
        if bounds is not None and video_id not in bounds:
            issues.append(
                _issue("UNKNOWN_VIDEO_ID", "video_id is not in inventory", row=row_number)
            )
            continue
        if task in {"KIS", "QA"}:
            frame_id = row.get("frame_id")
            if not _integer(frame_id) or frame_id < 0:
                issues.append(
                    _issue(
                        "FRAME_ID_INVALID",
                        "frame_id must be a non-negative integer",
                        row=row_number,
                    )
                )
            elif bounds is not None and frame_id >= bounds[video_id]:
                issues.append(
                    _issue("FRAME_OUT_OF_BOUNDS", "frame_id is outside raw video", row=row_number)
                )
            if task == "QA" and (
                not isinstance(row.get("answer"), str) or not row["answer"].strip()
            ):
                issues.append(
                    _issue("QA_ANSWER_INVALID", "QA answer must be non-empty", row=row_number)
                )
        elif task == "TRAKE":
            frame_ids = row.get("frame_ids")
            if not isinstance(frame_ids, list) or len(frame_ids) != query["event_count"]:
                issues.append(
                    _issue(
                        "TRAKE_EVENT_COUNT_MISMATCH",
                        "len(frame_ids) must equal event_count",
                        row=row_number,
                    )
                )
                continue
            for frame_id in frame_ids:
                if not _integer(frame_id) or frame_id < 0:
                    issues.append(
                        _issue(
                            "FRAME_ID_INVALID", "TRAKE frame_ids must be integers", row=row_number
                        )
                    )
                elif bounds is not None and frame_id >= bounds[video_id]:
                    issues.append(
                        _issue(
                            "FRAME_OUT_OF_BOUNDS",
                            "TRAKE frame is outside raw video",
                            row=row_number,
                        )
                    )
        else:  # pragma: no cover - guarded by query validation
            issues.append(_issue("TASK_INVALID", f"task must be one of {TASKS}", row=row_number))
    for query_id, values in ranks.items():
        if len(values) != len(set(values)):
            issues.append(_issue("RANK_DUPLICATE", "ranks must be unique", query_id=query_id))
        if values != sorted(values):
            issues.append(_issue("RANK_NOT_SORTED", "ranks must be ascending", query_id=query_id))
        if counts[query_id] > 100:
            issues.append(
                _issue("TOO_MANY_PREDICTIONS", "maximum is 100 per query", query_id=query_id)
            )
    summary = {
        "status": "PASS" if not issues else "FAIL",
        "query_count": len(normalized_queries),
        "prediction_count": len(predictions),
        "validated_prediction_count": len(predictions)
        - len({issue.get("row") for issue in issues if issue.get("row")}),
        "issue_count": len(issues),
        "inventory_validation_enabled": inventory is not None,
    }
    return summary, issues


__all__ = ["validate_predictions"]
