"""Tests that architecture nodes contain specific, edge-aligned contracts."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from generate_architecture_assets import (  # noqa: E402
    PLACEHOLDER_CONTRACT_PHRASES,
    load_spec,
)

SPEC_PATH = PROJECT_ROOT / "docs" / "architecture" / "architecture-spec.yaml"


def test_every_node_contract_is_specific_and_edge_aligned() -> None:
    spec = load_spec(SPEC_PATH)
    for page in spec["pages"]:
        nodes = {node["id"]: node for node in page["nodes"]}
        for node_id, node in nodes.items():
            incoming = [edge for edge in page["edges"] if edge["target"] == node_id]
            outgoing = [edge for edge in page["edges"] if edge["source"] == node_id]
            assert set(node["dependencies"]) == {edge["source"] for edge in incoming}
            assert set(node["next_modules"]) == {edge["target"] for edge in outgoing}
            assert all(
                edge["label"].lower() in " ".join(node["inputs"]).lower()
                for edge in incoming
            )
            assert all(
                edge["label"].lower() in " ".join(node["outputs"]).lower()
                for edge in outgoing
            )
            text = json.dumps(node, ensure_ascii=False).lower()
            assert not any(phrase in text for phrase in PLACEHOLDER_CONTRACT_PHRASES)
            assert not re.search(r"execute bounded .+ policy", text)


def test_critical_implementation_boundaries_are_explicit() -> None:
    spec = load_spec(SPEC_PATH)
    by_title = {
        node["title"]: node
        for page in spec["pages"]
        for node in page["nodes"]
    }
    assert "TransNetV2" in " ".join(by_title["Shot Detection"]["implementations"])
    assert "unselected" in " ".join(by_title["Team Visual Encoder"]["implementations"])
    assert "no GNN" in " ".join(by_title["Temporal / Entity Solver"]["implementations"])
    assert "raw-video" in " ".join(by_title["Semantic Moment Localizer"]["implementations"])
    assert "not microservices" in " ".join(
        by_title["Application Backend / Orchestrator"]["implementations"]
    )
