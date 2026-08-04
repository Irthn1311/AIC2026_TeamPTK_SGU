"""Tests the Page 07 application interaction surface."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from generate_architecture_assets import load_spec  # noqa: E402

SPEC_PATH = PROJECT_ROOT / "docs" / "architecture" / "architecture-spec.yaml"


def test_page07_contains_backend_ui_automatic_mode_and_control_bus() -> None:
    page = load_spec(SPEC_PATH)["pages"][7]
    assert page["title"] == "07 — Agent, Interaction, Output, Evaluation & Deployment"
    titles = {node["title"] for node in page["nodes"]}
    assert {
        "Application Backend / Orchestrator",
        "Interactive UI / Operator Console",
        "Automatic Mode",
        "Bounded Agent Planner",
        "Fallback Controller",
    } <= titles
    ui = next(
        node for node in page["nodes"] if node["title"] == "Interactive UI / Operator Console"
    )
    assert ui["implementation_owner"] == ["TAI"]
    assert "result grid" in ui["responsibility"][0].lower()
    bus_targets = {edge["target"] for edge in page["edges"] if edge["source"] == "P07_PVALID"}
    assert {"P07_POLICY", "P07_GRAPH", "P07_VERIFY", "P07_FORMAT"} <= bus_targets
