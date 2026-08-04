"""Tests that Page 04 renders an actual evidence-grounded event graph."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from generate_architecture_assets import (  # noqa: E402
    GRAPH_NODE_TYPES,
    REQUIRED_GRAPH_RELATIONS,
    load_spec,
)

SPEC_PATH = PROJECT_ROOT / "docs" / "architecture" / "architecture-spec.yaml"


def test_event_graph_has_required_nodes_relations_and_matching() -> None:
    page = load_spec(SPEC_PATH)["pages"][4]
    assert {node.get("graph_node_type") for node in page["nodes"]} >= GRAPH_NODE_TYPES
    relations = {edge["label"].split()[0] for edge in page["edges"]}
    assert relations >= REQUIRED_GRAPH_RELATIONS
    assert sum(edge["flow_type"] == "match" for edge in page["edges"]) >= 3
    assert {"Q1", "Q2", "Q3", "E7", "E9", "E12"} <= {node["id"] for node in page["nodes"]}


def test_evidence_refs_are_external_and_solver_returns_top_m() -> None:
    page = load_spec(SPEC_PATH)["pages"][4]
    evidence = [node for node in page["nodes"] if node.get("graph_node_type") == "EvidenceRef"]
    assert evidence and all(node["type"] == "artifact" for node in evidence)
    text = " ".join(
        page["notes"] + [str(value) for node in page["nodes"] for value in node["responsibility"]]
    )
    assert "external" in text.lower() and "full-corpus" in text.lower()
    titles = {node["title"] for node in page["nodes"]}
    assert {
        "Event Match Matrix",
        "Temporal / Entity Solver",
        "Top-M Event Chains",
        "Gap Detector",
    } <= titles
