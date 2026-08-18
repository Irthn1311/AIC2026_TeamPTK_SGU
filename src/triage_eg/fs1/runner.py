"""Prediction-list integration for B0, M0 and M1.

Heavy encoders and Qwen live behind notebook-produced evidence records. This
module never imports model frameworks and never receives ground truth.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from .event_graph import Edge, EventGraph, Node
from .fusion import assert_protected_prefix, fuse_tail
from .router import route_query


def group_predictions(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        output[str(row["query_id"])].append(dict(row))
    for values in output.values():
        values.sort(key=lambda row: int(row["rank"]))
    return dict(output)


def build_arm(
    arm: str,
    queries: list[dict[str, Any]],
    b0_rows: list[dict[str, Any]],
    evidence: dict[str, dict[str, list[dict[str, Any]]]],
    plugin_status: dict[str, bool],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if arm not in {"M0", "M1"}:
        raise ValueError("arm must be M0 or M1")
    grouped = group_predictions(b0_rows)
    output, diagnostics = [], []
    available = {name for name, enabled in plugin_status.items() if enabled}
    for query in queries:
        query_id, task = str(query["query_id"]), str(query["task"]).upper()
        text = " ".join(str(query.get(key, "")) for key in ("query", "question", "description"))
        route = route_query(task, text, available=available)
        lists = [
            evidence.get(modality, {}).get(query_id, [])
            for modality in route.modalities
            if modality != "b0_visual"
        ]
        if task == "QA":
            lists.append(evidence.get("qwen", {}).get(query_id, []))
        graph = None
        if arm == "M1":
            graph = EventGraph(query_id)
            graph.add_node(
                Node(f"event:{query_id}:0", "QueryEvent", {"query_id": query_id}, {"text": text})
            )
            for index, row in enumerate([item for values in lists for item in values[:5]]):
                node_id = f"candidate:{query_id}:{index}"
                graph.add_node(
                    Node(
                        node_id,
                        "EventCandidate",
                        {"modality": row.get("source", "UNKNOWN"), "rank": row.get("rank")},
                        row,
                    )
                )
                graph.add_edge(
                    Edge(
                        node_id, f"event:{query_id}:0", "SUPPORTS", {"source_rank": row.get("rank")}
                    )
                )
            missing = graph.missing_events()
            if missing:
                graph.revise_once("EXPLORE", missing[0], lambda current, event: None)
        candidate = fuse_tail(task, grouped[query_id], lists)
        assert_protected_prefix(task, grouped[query_id], candidate)
        output.extend({**row, "system_variant": arm} for row in candidate)
        diagnostics.append(
            {
                "query_id": query_id,
                "arm": arm,
                "route": route.__dict__,
                "graph": graph.diagnostics() if graph else None,
            }
        )
    return output, diagnostics
