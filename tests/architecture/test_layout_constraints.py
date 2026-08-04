"""Tests for page bounds, spacing, node count, shapes and font minimums."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from generate_architecture_assets import analyze_page_layout, load_spec, validate_spec  # noqa: E402

SPEC_PATH = PROJECT_ROOT / "docs" / "architecture" / "architecture-spec.yaml"


def test_all_pages_meet_bounds_overlap_and_gap_contracts() -> None:
    spec = load_spec(SPEC_PATH)
    for page in spec["pages"]:
        layout = analyze_page_layout(page)
        assert layout["aspect_ratio"] <= 3.5
        assert not layout["overlapping_node_pairs"]
        assert not layout["minimum_gap_warnings"]
    overview = spec["pages"][1]
    assert 2.5 <= analyze_page_layout(overview)["aspect_ratio"] <= 3.0
    assert analyze_page_layout(overview)["processing_node_count"] <= 24


def test_font_minimums_and_shape_semantics() -> None:
    spec = load_spec(SPEC_PATH)
    assert min(spec["styles"]["fonts"].values()) >= 10
    prohibited_diamonds = {
        "Hybrid Frame Bank",
        "Evidence Verifier",
        "Q&A Answer Type Router",
        "Task Policy",
    }
    for page in spec["pages"]:
        for node in page["nodes"]:
            assert not (node["title"] in prohibited_diamonds and node["type"] == "decision")
            if node["type"] == "graph_node":
                assert page["id"] == "PAGE_04"


def test_overlap_and_large_aspect_are_rejected() -> None:
    spec = copy.deepcopy(load_spec(SPEC_PATH))
    page = spec["pages"][0]
    page["nodes"][1]["geometry"] = copy.deepcopy(page["nodes"][0]["geometry"])
    with pytest.raises(ValueError, match="overlapping"):
        validate_spec(spec)
    spec = copy.deepcopy(load_spec(SPEC_PATH))
    spec["pages"][0]["groups"][0]["geometry"]["width"] = 10_000
    with pytest.raises(ValueError, match="aspect ratio|width"):
        validate_spec(spec)
