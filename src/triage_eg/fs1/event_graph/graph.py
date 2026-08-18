"""Compact provenance-mandatory graph with one revision pass."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

NODE_TYPES = {
    "VideoHypothesis",
    "QueryEvent",
    "EventCandidate",
    "EvidenceRef",
    "SemanticMomentHypothesis",
    "EntityHypothesis",
}
EDGE_TYPES = {
    "SUPPORTS",
    "CONTRADICTS",
    "PRECEDES",
    "OVERLAPS",
    "ADJACENT",
    "ANCHORS",
    "POSSIBLE_SAME_ENTITY",
}


@dataclass(frozen=True)
class Node:
    node_id: str
    node_type: str
    provenance: dict[str, Any]
    payload: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.node_type not in NODE_TYPES or not self.provenance:
            raise ValueError("FS1 graph nodes require a supported type and provenance")


@dataclass(frozen=True)
class Edge:
    source: str
    target: str
    edge_type: str
    provenance: dict[str, Any]

    def __post_init__(self) -> None:
        if self.edge_type not in EDGE_TYPES or not self.provenance:
            raise ValueError("FS1 graph edges require a supported type and provenance")


class EventGraph:
    def __init__(self, query_id: str) -> None:
        self.query_id = query_id
        self.nodes: dict[str, Node] = {}
        self.edges: list[Edge] = []
        self.revision_count = 0
        self.revision_actions: list[dict[str, Any]] = []

    def add_node(self, node: Node) -> None:
        if node.node_id in self.nodes and self.nodes[node.node_id] != node:
            raise ValueError("conflicting graph node")
        self.nodes[node.node_id] = node

    def add_edge(self, edge: Edge) -> None:
        if edge.source not in self.nodes or edge.target not in self.nodes:
            raise ValueError("edge endpoints must exist")
        self.edges.append(edge)

    def missing_events(self) -> tuple[str, ...]:
        supported = {edge.target for edge in self.edges if edge.edge_type == "SUPPORTS"}
        return tuple(
            node.node_id
            for node in self.nodes.values()
            if node.node_type == "QueryEvent" and node.node_id not in supported
        )

    def revise_once(
        self, action: str, event_id: str, callback: Callable[[EventGraph, str], None]
    ) -> None:
        if self.revision_count >= 1:
            raise RuntimeError("FS1_GRAPH_REVISION_LIMIT_EXCEEDED")
        if action not in {"EXPLOIT", "EXPLORE"}:
            raise ValueError("unsupported graph revision action")
        self.revision_count += 1
        callback(self, event_id)
        self.revision_actions.append({"action": action, "event_id": event_id})

    def diagnostics(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
            "missing_events": list(self.missing_events()),
            "revision_count": self.revision_count,
            "revision_actions": self.revision_actions,
        }
