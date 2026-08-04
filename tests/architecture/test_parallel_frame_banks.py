"""Tests for parallel BTC and Team frame-bank branches."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from generate_architecture_assets import load_spec  # noqa: E402

SPEC_PATH = PROJECT_ROOT / "docs" / "architecture" / "architecture-spec.yaml"


def test_btc_and_team_branches_are_parallel_with_separate_indexes() -> None:
    page = load_spec(SPEC_PATH)["pages"][2]
    nodes = {node["id"]: node for node in page["nodes"]}
    edges = {(edge["source"], edge["target"]) for edge in page["edges"]}
    assert {"P02_BTC", "P02_TEAM", "P02_BTCIDX", "P02_TEAMIDX"} <= nodes.keys()
    assert ("P02_MAP", "P02_BTC") in edges and ("P02_MAP", "P02_TEAM") in edges
    assert nodes["P02_BTCIDX"]["title"] == "BTC Frame Vector Index"
    assert nodes["P02_TEAMIDX"]["title"] == "Team Frame Vector Index"
    assert all("Hybrid Frame Bank Router" not in node["title"] for node in page["nodes"])


def test_team_frame_pipeline_is_explicit_and_ordered() -> None:
    page = load_spec(SPEC_PATH)["pages"][2]
    edges = {(edge["source"], edge["target"]) for edge in page["edges"]}
    ordered = [
        "P02_TEAM",
        "P02_SHOT",
        "P02_DURATION",
        "P02_SELECT",
        "P02_LONG",
        "P02_QUALITY",
        "P02_DEDUP",
        "P02_FMAP",
        "P02_ENCODER",
        "P02_TEAMIDX",
    ]
    assert all((left, right) in edges for left, right in zip(ordered, ordered[1:], strict=False))
