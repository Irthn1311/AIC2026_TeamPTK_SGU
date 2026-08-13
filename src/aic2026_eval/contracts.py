"""System-neutral query, prediction, and ground-truth contracts."""

from __future__ import annotations

import re
from typing import Any

TASKS = ("KIS", "QA", "TRAKE")
VIDEO_ID_PATTERN = re.compile(r"^L\d+_V\d+$", re.ASCII)
QUERY_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$", re.ASCII)
SLICE_NAMES = (
    "visual",
    "object",
    "small_object",
    "OCR",
    "OCR_numeric",
    "action",
    "relation",
    "count",
    "attribute",
    "state",
    "transition",
    "first_occurrence",
    "contact",
    "separation",
    "extremum",
    "long_temporal_gap",
    "repeated_action",
    "camera_cut",
)


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def validate_query(value: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("query must be an object")
    query_id = _text(value.get("query_id"), "query_id")
    if not QUERY_ID_PATTERN.fullmatch(query_id):
        raise ValueError("query_id is not a safe identifier")
    task = str(value.get("task", "")).upper()
    if task not in TASKS:
        raise ValueError(f"task must be one of {TASKS}")
    if task in {"KIS", "QA", "TRAKE"}:
        _text(value.get("query"), "query")
    if task == "QA":
        _text(value.get("question"), "question")
    if task == "TRAKE":
        event_count = value.get("event_count")
        if not isinstance(event_count, int) or isinstance(event_count, bool) or event_count <= 0:
            raise ValueError("TRAKE event_count must be a positive integer")
    return {**value, "query_id": query_id, "task": task}


def accepted_intervals(value: Any) -> list[tuple[int, int]]:
    if not isinstance(value, list) or not value:
        raise ValueError("accepted intervals must be a non-empty list")
    intervals = []
    for item in value:
        if isinstance(item, dict):
            start = item.get("start_frame", item.get("start"))
            end = item.get("end_frame", item.get("end"))
        elif isinstance(item, list | tuple) and len(item) == 2:
            start, end = item
        else:
            raise ValueError("accepted interval must be [start, end] or an object")
        if (
            not isinstance(start, int)
            or isinstance(start, bool)
            or not isinstance(end, int)
            or isinstance(end, bool)
            or start < 0
            or end < start
        ):
            raise ValueError("accepted interval has invalid frame bounds")
        intervals.append((start, end))
    return intervals


def contract_document() -> dict[str, Any]:
    return {
        "contract_version": "TEAM_EVAL_E0_E1_v1",
        "team_neutral": True,
        "frame_id_semantics": "original_frame_idx",
        "mapping_authority": "BTC mapping CSV frame_idx",
        "duplicate_original_frame_idx_policy": "PRESERVE",
        "raw_video_source_of_truth": True,
        "tasks": list(TASKS),
        "maximum_predictions_per_query": 100,
        "rank_policy": "ONE_BASED_UNIQUE_ASCENDING_PER_QUERY",
        "trake_frame_count_policy": "len(frame_ids) == event_count",
        "qa_alias_matching_scope": "INTERNAL_DETERMINISTIC_NOT_OFFICIAL_BTC_SEMANTICS",
        "recall_cutoffs": [1, 5, 20, 50, 100],
        "final_score": "mean(R@1,R@5,R@20,R@50,R@100)",
        "optional_slices": list(SLICE_NAMES),
        "forbidden_system_assumptions": [
            "TRIAGE Event Graph",
            "TRIAGE Stage1",
            "CLIP",
            "OpenCV",
            "NVDEC",
            "OCR",
            "VLM",
            "Agent",
            "specific frame bank",
            "specific feature dimension",
        ],
    }


__all__ = [
    "QUERY_ID_PATTERN",
    "SLICE_NAMES",
    "TASKS",
    "VIDEO_ID_PATTERN",
    "accepted_intervals",
    "contract_document",
    "validate_query",
]
