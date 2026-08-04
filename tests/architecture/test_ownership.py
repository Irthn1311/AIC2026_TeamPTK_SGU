"""Tests the split architecture, implementation and review ownership model."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from generate_architecture_assets import load_spec  # noqa: E402

SPEC_PATH = PROJECT_ROOT / "docs" / "architecture" / "architecture-spec.yaml"


def test_every_node_has_three_ownership_fields() -> None:
    spec = load_spec(SPEC_PATH)
    for page in spec["pages"]:
        for node in page["nodes"]:
            assert node["architecture_owner"]
            assert node["implementation_owner"]
            assert isinstance(node["reviewers"], list)
            assert "owner" not in node


def test_key_grounding_ownership_and_distribution() -> None:
    spec = load_spec(SPEC_PATH)
    sml = next(
        node for node in spec["pages"][5]["nodes"] if node["title"] == "Semantic Moment Localizer"
    )
    extractor = next(
        node for node in spec["pages"][5]["nodes"] if node["title"] == "Q&A Answer Extractor"
    )
    verifier = next(
        node for node in spec["pages"][5]["nodes"] if node["title"] == "Evidence Verifier"
    )
    assert (
        sml["architecture_owner"] == ["TRI"]
        and sml["implementation_owner"] == ["TAI", "PHAT"]
        and sml["reviewers"] == ["PHUC"]
    )
    assert extractor["implementation_owner"] == ["TAI"] and extractor["reviewers"] == ["PHUC"]
    assert verifier["implementation_owner"] == ["TAI"] and verifier["reviewers"] == ["PHUC"]
    implementation = [
        owner
        for page in spec["pages"]
        for node in page["nodes"]
        for owner in node["implementation_owner"]
    ]
    assert implementation.count("TRI") < len(implementation) / 2


def test_run_manifest_links_every_other_contract_artifact() -> None:
    page = load_spec(SPEC_PATH)["pages"][8]
    artifacts = {
        node["id"]
        for node in page["nodes"]
        if node["type"] == "artifact" and node["id"] != "P08_RUN"
    }
    run_targets = {edge["target"] for edge in page["edges"] if edge["source"] == "P08_RUN"}
    assert artifacts <= run_targets
