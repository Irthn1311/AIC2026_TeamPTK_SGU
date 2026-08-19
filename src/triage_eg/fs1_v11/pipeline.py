"""Completion v1.1 prediction construction with executable graph activation."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Callable
from typing import Any

from triage_eg.fs1.fusion import default_key, reciprocal_rank_fusion
from triage_eg.fs1.router import route_events, route_query

from .events import CompiledEvent, compile_query_events
from .graph_runtime import (
    Candidate,
    ExecutableEventGraph,
    build_graph_chains,
    solve_event_candidates,
)

RevisionProvider = Callable[[dict[str, Any], CompiledEvent, str], list[dict[str, Any]]]


def grouped(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        output[str(row["query_id"])].append(dict(row))
    for values in output.values():
        values.sort(key=lambda item: int(item["rank"]))
    return dict(output)


def _rank(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{**row, "rank": rank} for rank, row in enumerate(rows[:100], 1)]


def _safe(task: str, b0: list[dict[str, Any]], full: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if task not in {"KIS", "TRAKE"}:
        return full
    prefix, blocked = b0[:5], {default_key(row) for row in b0[:5]}
    return _rank([*prefix, *(row for row in full if default_key(row) not in blocked)])


def _candidate(event_index: int, row: dict[str, Any], source: str) -> Candidate:
    frame = row.get("frame_id")
    if frame is None and row.get("frame_ids"):
        frames = row["frame_ids"]
        frame = frames[min(event_index, len(frames) - 1)]
    return Candidate(
        event_index,
        str(row["video_id"]),
        int(frame),
        int(row.get("rank", 100)),
        source,
        {"source": source, "source_rank": int(row.get("rank", 100))},
    )


def build_completion_arm(
    arm: str,
    queries: list[dict[str, Any]],
    b0_rows: list[dict[str, Any]],
    evidence: dict[str, dict[str, list[dict[str, Any]]]],
    available: set[str],
    *,
    revision_provider: RevisionProvider | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if arm not in {"M0_v11", "M1_v11"}:
        raise ValueError("completion arm must be M0_v11 or M1_v11")
    b0 = grouped(b0_rows)
    full, safe, diagnostics = [], [], []
    for query in queries:
        query_id, task = str(query["query_id"]), str(query["task"]).upper()
        baseline = b0[query_id]
        events = compile_query_events(query, baseline)
        routes = (
            route_events(task, [event.event_text for event in events], available=available)
            if task == "TRAKE"
            else (route_query(task, events[0].event_text, available=available, event_index=0),)
        )
        modality_lists = [
            evidence.get(modality, {}).get(query_id, [])
            for route in routes
            for modality in route.modalities
            if modality != "b0_visual"
        ]
        if task == "QA":
            answers = {
                (str(row["video_id"]), int(row["frame_id"])): str(row["answer"])
                for row in evidence.get("qwen", {}).get(query_id, [])
                if row.get("evidence_sufficient") and row.get("answer")
            }
            baseline = [
                {
                    **row,
                    **(
                        {"answer": answers[(str(row["video_id"]), int(row["frame_id"]))]}
                        if (str(row["video_id"]), int(row["frame_id"])) in answers
                        else {}
                    ),
                }
                for row in baseline
            ]
        m0 = _rank(reciprocal_rank_fusion([baseline, *modality_lists]))
        event_pools: dict[int, list[Candidate]] = {event.event_index: [] for event in events}
        if task == "TRAKE":
            for event in events:
                for row in baseline[:20]:
                    event_pools[event.event_index].append(
                        _candidate(event.event_index, row, "b0_t3")
                    )
                for modality in routes[event.event_index].modalities:
                    if modality == "b0_visual":
                        continue
                    for row in evidence.get(modality, {}).get(query_id, [])[:20]:
                        if row.get("event_index") in {None, event.event_index}:
                            event_pools[event.event_index].append(
                                _candidate(event.event_index, row, modality)
                            )
            m0 = solve_event_candidates(query_id, events, event_pools)
            if not m0:
                raise RuntimeError("M0_T3_SOLVER_RETURNED_NO_CHAINS")
        graph_diagnostics = None
        selected = m0
        if arm == "M1_v11" and task == "TRAKE":
            graph = ExecutableEventGraph(query_id, events)
            for event in events:
                for candidate in event_pools[event.event_index]:
                    if candidate.source == "b0_t3":
                        graph.add(candidate)
            weak_event = min(graph.by_event, key=lambda index: len(graph.by_event[index]))
            if revision_provider is None:
                raise RuntimeError("GRAPH_REVISION_PROVIDER_REQUIRED")
            revision_rows = revision_provider(query, events[weak_event], "EXPLOIT")
            added = [
                _candidate(weak_event, row, str(row.get("source", "revision")))
                for row in revision_rows
            ]
            graph.revise_once("EXPLOIT", weak_event, added)
            graph_chains = build_graph_chains(graph)
            if not graph_chains:
                raise RuntimeError("GRAPH_T3_SOLVER_RETURNED_NO_CHAINS")
            selected = _rank(reciprocal_rank_fusion([graph_chains, m0]))
            graph_diagnostics = graph.diagnostics() | {"chain_candidates_added": len(graph_chains)}
        full.extend({**row, "system_variant": arm} for row in selected)
        safe.extend(
            {**row, "system_variant": f"{arm}_SAFE"} for row in _safe(task, baseline, selected)
        )
        diagnostics.append(
            {
                "query_id": query_id,
                "task": task,
                "events": [event.as_dict() for event in events],
                "routing": [route.__dict__ for route in routes],
                "graph": graph_diagnostics,
            }
        )
    return full, safe, diagnostics


def semantic_content_hash(rows: list[dict[str, Any]]) -> str:
    keys = ("query_id", "video_id", "frame_id", "frame_ids", "answer", "rank")
    values = [{key: row[key] for key in keys if key in row} for row in rows]
    payload = json.dumps(values, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def assert_graph_exercised(
    m0_by_benchmark: dict[str, list[dict[str, Any]]],
    m1_by_benchmark: dict[str, list[dict[str, Any]]],
    diagnostics: list[dict[str, Any]],
) -> None:
    changed = any(
        semantic_content_hash(m0_by_benchmark[name]) != semantic_content_hash(m1_by_benchmark[name])
        for name in m0_by_benchmark
    )
    graph_rows = [row["graph"] for row in diagnostics if row.get("graph")]
    active = any(
        row["query_event_count"] > 1
        and row["edge_count"] > 0
        and "PRECEDES" in row.get("edge_types", [])
        and row["revision_count"] == 1
        and row["revision"]["evidence_added"] > 0
        and row["chain_candidates_added"] > 0
        for row in graph_rows
    )
    if not changed or not active:
        raise RuntimeError("GRAPH_NOT_EXERCISED")
