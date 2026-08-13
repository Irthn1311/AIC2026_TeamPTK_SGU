"""Deterministic shared KIS, QA, and TRAKE evaluator."""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from statistics import mean
from typing import Any

from .contracts import SLICE_NAMES, accepted_intervals, validate_query
from .validation import validate_predictions

CUTOFFS = (1, 5, 20, 50, 100)


def normalize_answer(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold().strip()
    normalized = re.sub(r"[^\w\s]", " ", normalized, flags=re.UNICODE)
    return " ".join(normalized.split())


def _inside(frame_id: int, intervals: list[tuple[int, int]]) -> bool:
    return any(start <= frame_id <= end for start, end in intervals)


def score_prediction(
    query: dict[str, Any], prediction: dict[str, Any], ground_truth: dict[str, Any]
) -> tuple[float, dict[str, Any]]:
    task = query["task"]
    correct_video = ground_truth.get("correct_video", ground_truth.get("video_id"))
    video_correct = prediction.get("video_id") == correct_video
    diagnostics: dict[str, Any] = {"video_correct": video_correct}
    if task in {"KIS", "QA"}:
        intervals = accepted_intervals(
            ground_truth.get("acceptable_intervals", ground_truth.get("intervals"))
        )
        grounding = video_correct and _inside(prediction["frame_id"], intervals)
        diagnostics["grounding_correct"] = grounding
        if task == "KIS":
            diagnostics["full_tuple_correct"] = grounding
            return float(grounding), diagnostics
        aliases = ground_truth.get("accepted_answers", ground_truth.get("aliases", []))
        if not isinstance(aliases, list) or not aliases:
            raise ValueError("QA ground truth requires accepted_answers or aliases")
        accepted = {normalize_answer(str(alias)) for alias in aliases}
        alias_correct = normalize_answer(prediction["answer"]) in accepted
        full = grounding and alias_correct
        diagnostics.update(
            {
                "answer_alias_correct": alias_correct,
                "full_tuple_correct": full,
                "semantic_review_required": bool(
                    ground_truth.get("semantic_review_required", False)
                ),
                "alias_matching_is_official_btc_semantics": False,
            }
        )
        return float(full), diagnostics
    intervals = []
    for item in ground_truth.get("event_intervals", []):
        if (
            isinstance(item, list)
            and len(item) == 2
            and all(isinstance(value, int) and not isinstance(value, bool) for value in item)
        ):
            item = [item]
        intervals.append(accepted_intervals(item))
    if len(intervals) != query["event_count"]:
        raise ValueError("TRAKE event_intervals must match event_count")
    frames = prediction["frame_ids"]
    hits = [
        video_correct and _inside(frame_id, event_intervals)
        for frame_id, event_intervals in zip(frames, intervals, strict=True)
    ]
    order_valid = all(left < right for left, right in zip(frames, frames[1:], strict=False))
    score = sum(hits) / query["event_count"] if video_correct else 0.0
    diagnostics.update(
        {
            "per_event_hit": hits,
            "events_hit": sum(hits),
            "event_count": query["event_count"],
            "full_chain_correct": all(hits),
            "event_order_structural_validity": order_valid,
        }
    )
    return score, diagnostics


def _query_slices(query: dict[str, Any], ground_truth: dict[str, Any]) -> set[str]:
    values = {f"task:{query['task']}"}
    difficulty = ground_truth.get("difficulty", query.get("difficulty"))
    if difficulty:
        values.add(f"difficulty:{difficulty}")
    for source in (query.get("tags", []), ground_truth.get("tags", [])):
        if isinstance(source, list):
            values.update(str(tag) for tag in source if str(tag) in SLICE_NAMES)
    values.update(name for name in SLICE_NAMES if ground_truth.get(name) is True)
    return values


def evaluate(
    queries: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    ground_truth: list[dict[str, Any]],
    *,
    metadata: dict[str, Any] | None = None,
    inventory: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    validation_summary, validation_issues = validate_predictions(
        queries,
        predictions,
        inventory=inventory,
    )
    if validation_summary["status"] != "PASS":
        codes = sorted({issue["code"] for issue in validation_issues})
        raise ValueError(f"predictions failed strict validation: {codes}")
    query_by_id = {row["query_id"]: validate_query(row) for row in queries}
    gt_by_id = {row["query_id"]: row for row in ground_truth}
    if len(gt_by_id) != len(ground_truth):
        raise ValueError("ground truth query_id values must be unique")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for prediction in predictions:
        grouped[prediction["query_id"]].append(prediction)
    per_query, issues = [], []
    for query_id, query in query_by_id.items():
        if query_id not in gt_by_id:
            raise ValueError(f"ground truth missing query_id={query_id}")
        gt = gt_by_id[query_id]
        ranked = sorted(grouped.get(query_id, []), key=lambda row: row["rank"])
        scored = []
        for prediction in ranked:
            score, diagnostics = score_prediction(query, prediction, gt)
            scored.append(
                {
                    "rank": prediction["rank"],
                    "r_score": score,
                    "diagnostics": diagnostics,
                }
            )
        recalls = {
            f"R@{cutoff}": max(
                (item["r_score"] for item in scored if item["rank"] <= cutoff),
                default=0.0,
            )
            for cutoff in CUTOFFS
        }
        final = mean(recalls.values())
        semantic_review = bool(gt.get("semantic_review_required", False))
        if semantic_review:
            issues.append(
                {
                    "severity": "INFO",
                    "code": "QA_SEMANTIC_REVIEW_REQUIRED",
                    "query_id": query_id,
                }
            )
        per_query.append(
            {
                "query_id": query_id,
                "task": query["task"],
                **recalls,
                "final_score": final,
                "slices": sorted(_query_slices(query, gt)),
                "prediction_diagnostics": scored,
            }
        )
    aggregate = {
        f"R@{cutoff}": mean(row[f"R@{cutoff}"] for row in per_query) if per_query else 0.0
        for cutoff in CUTOFFS
    }
    final_score = mean(aggregate.values())
    slices: dict[str, Any] = {}
    for name in sorted({value for row in per_query for value in row["slices"]}):
        members = [row for row in per_query if name in row["slices"]]
        slices[name] = {
            "query_count": len(members),
            **{f"R@{cutoff}": mean(row[f"R@{cutoff}"] for row in members) for cutoff in CUTOFFS},
            "final_score": mean(row["final_score"] for row in members),
        }
    summary = {
        "evaluator_version": "TEAM_EVAL_E0_E1_v1",
        "query_count": len(per_query),
        **aggregate,
        "final_score": final_score,
        "score_computed_by_shared_evaluator": True,
        "qa_alias_matching_is_official_btc_semantics": False,
        **(metadata or {}),
    }
    return summary, per_query, slices, issues


__all__ = ["CUTOFFS", "evaluate", "normalize_answer", "score_prediction"]
