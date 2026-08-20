"""Deterministic TEAM-EVAL query planning with a hard no-GT boundary."""

from __future__ import annotations

from typing import Any

from aic2026_eval.contracts import validate_query

from .contracts import FORBIDDEN_INFERENCE_FIELDS, QueryPlan


def plan_query(raw: dict[str, Any]) -> QueryPlan:
    if not isinstance(raw, dict):
        raise ValueError("query must be an object")
    leaked = sorted(FORBIDDEN_INFERENCE_FIELDS & set(raw))
    if leaked:
        raise ValueError(f"GT_FIELDS_FORBIDDEN_AT_INFERENCE: {leaked}")
    query = validate_query(raw)
    task = query["task"]
    language = str(query.get("language", "auto")).lower()
    if task == "TRAKE":
        event_count = int(query["event_count"])
        if event_count > 4:
            raise ValueError("UNSUPPORTED_EVENT_COUNT")
        descriptions = query.get("event_descriptions")
        if not isinstance(descriptions, list) or len(descriptions) != event_count:
            raise ValueError("TRAKE_EVENT_COUNT_MISMATCH")
        events = []
        for index, event in enumerate(descriptions, 1):
            if not isinstance(event, dict):
                raise ValueError("TRAKE event description must be an object")
            event_id = str(event.get("event_id", f"E{index}")).strip()
            text = str(event.get("description", "")).strip()
            if not event_id or not text:
                raise ValueError("TRAKE event_id and description must be non-empty")
            events.append((event_id, text))
        return QueryPlan(
            query["query_id"], task, language, query["query"].strip(), None, tuple(events)
        )
    question = query["question"].strip() if task == "QA" else None
    compiled = query.get("compiled_routing", [])
    if compiled is None:
        compiled = []
    if not isinstance(compiled, list) or any(not isinstance(value, str) for value in compiled):
        raise ValueError("compiled_routing must be a list of modality names")
    provenance = query.get("evidence_provenance", [])
    if provenance is None:
        provenance = []
    if not isinstance(provenance, list) or any(not isinstance(value, str) for value in provenance):
        raise ValueError("evidence_provenance must be a list of strings")
    return QueryPlan(
        query["query_id"],
        task,
        language,
        query["query"].strip(),
        question,
        (("E1", query["query"].strip()),),
        str(query.get("answer_type") or "").upper() or None,
        tuple(value.casefold() for value in compiled),
        str(query.get("answer_policy") or "").upper() or None,
        tuple(provenance),
    )


def plan_queries(rows: list[dict[str, Any]]) -> list[QueryPlan]:
    plans = [plan_query(row) for row in rows]
    ids = [plan.query_id for plan in plans]
    if len(ids) != len(set(ids)):
        raise ValueError("query_id values must be unique")
    return plans


__all__ = ["plan_queries", "plan_query"]
