"""Integration tests for deterministic architecture asset generation."""

from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from generate_architecture_assets import generate_assets, load_spec  # noqa: E402
from validate_architecture_assets import validate_assets  # noqa: E402

SPEC_PATH = PROJECT_ROOT / "docs" / "architecture" / "architecture-spec.yaml"


def test_generator_creates_parseable_complete_assets(tmp_path: Path) -> None:
    summary = generate_assets(SPEC_PATH, tmp_path)
    drawio_path = tmp_path / "TRIAGE_EG_Complete_System.drawio"
    root = ET.parse(drawio_path).getroot()
    assert root.tag == "mxfile" and root.get("compressed") == "false"
    assert len(root.findall("diagram")) == 9
    assert summary["page_count"] == 9
    assert summary["node_count"] == 203
    assert summary["edge_count"] == 211
    assert summary["validation_status"] == "PASS"
    assert summary["contract_quality"] == {
        "specific_node_contracts": 203,
        "placeholder_node_contracts": 0,
        "edge_aligned_interfaces": 203,
    }
    spec = load_spec(SPEC_PATH)
    for page in spec["pages"]:
        content = (tmp_path / "mermaid" / page["mermaid_file"]).read_text(encoding="utf-8")
        assert "flowchart " in content and "subgraph " in content
    saved = json.loads(
        (tmp_path / "generated" / "architecture_summary.json").read_text(encoding="utf-8")
    )
    assert saved["pages"][1]["aspect_ratio"] <= 3.0
    assert (tmp_path / "generated" / "architecture_quality_report.md").is_file()


def test_generation_is_deterministic_and_validator_accepts_it(tmp_path: Path) -> None:
    generate_assets(SPEC_PATH, tmp_path)
    files = [
        tmp_path / "TRIAGE_EG_Complete_System.drawio",
        *sorted((tmp_path / "mermaid").glob("*.mmd")),
        tmp_path / "generated" / "architecture_summary.json",
        tmp_path / "generated" / "architecture_quality_report.md",
    ]
    first = {path.relative_to(tmp_path): path.read_bytes() for path in files}
    generate_assets(SPEC_PATH, tmp_path)
    assert {path.relative_to(tmp_path): path.read_bytes() for path in files} == first
    counts = validate_assets(SPEC_PATH, tmp_path / "TRIAGE_EG_Complete_System.drawio")
    assert counts == {"pages": 9, "nodes": 203, "edges": 211, "warnings": []}


def test_validator_rejects_generated_asset_drift(tmp_path: Path) -> None:
    generate_assets(SPEC_PATH, tmp_path)
    mermaid = tmp_path / "mermaid" / "04_event_graph_internals.mmd"
    mermaid.write_text(mermaid.read_text(encoding="utf-8") + "%% manual drift\n", encoding="utf-8")
    with pytest.raises(ValueError, match="stale|drifted"):
        validate_assets(SPEC_PATH, tmp_path / "TRIAGE_EG_Complete_System.drawio")
