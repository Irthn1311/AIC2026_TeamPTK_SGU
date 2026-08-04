"""Tests for task-scoped criticality decisions."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from generate_architecture_assets import load_spec  # noqa: E402

SPEC_PATH = PROJECT_ROOT / "docs" / "architecture" / "architecture-spec.yaml"


def _nodes(title: str) -> list[dict]:
    spec = load_spec(SPEC_PATH)
    return [node for page in spec["pages"] for node in page["nodes"] if node["title"] == title]


def test_sml_and_answer_extractor_are_task_core() -> None:
    assert any(
        node["criticality"] == "CORE"
        and node["criticality_scope"] == "TRAKE"
        and node["maturity"] == "EXPERIMENTAL"
        for node in _nodes("Semantic Moment Localizer")
    )
    assert any(
        node["criticality"] == "CORE"
        and node["criticality_scope"] == "Q&A"
        and node["maturity"] == "EXPERIMENTAL"
        for node in _nodes("Q&A Answer Extractor")
    )


def test_agent_and_team_encoder_are_not_promoted() -> None:
    planner = _nodes("Bounded Agent Planner")[0]
    encoder = _nodes("Team Visual Encoder")[0]
    assert (planner["criticality"], planner["maturity"]) == ("OPTIONAL", "EXPERIMENTAL")
    assert (encoder["criticality"], encoder["maturity"]) == ("CONDITIONAL", "EXPERIMENTAL")
