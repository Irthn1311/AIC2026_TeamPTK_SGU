"""Tests for independent criticality, maturity and source-basis axes."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from generate_architecture_assets import (  # noqa: E402
    CRITICALITIES,
    MATURITIES,
    SOURCE_BASES,
    load_spec,
    validate_spec,
)

SPEC_PATH = PROJECT_ROOT / "docs" / "architecture" / "architecture-spec.yaml"


def test_taxonomies_and_legacy_migration() -> None:
    spec = load_spec(SPEC_PATH)
    assert set(spec["criticalities"]) == CRITICALITIES
    assert set(spec["maturities"]) == MATURITIES
    assert set(spec["source_bases"]) == SOURCE_BASES
    assert "statuses" not in spec
    for page in spec["pages"]:
        for node in page["nodes"]:
            assert "status" not in node and "owner" not in node
            assert node["criticality"] in CRITICALITIES
            assert node["maturity"] in MATURITIES
            assert node["source_basis"] in SOURCE_BASES


@pytest.mark.parametrize(
    ("field", "bad"),
    [("criticality", "MANDATORY"), ("maturity", "DONE"), ("source_basis", "ASSUMED")],
)
def test_invalid_taxonomy_values_are_rejected(field: str, bad: str) -> None:
    spec = copy.deepcopy(load_spec(SPEC_PATH))
    spec["pages"][0]["nodes"][0][field] = bad
    with pytest.raises(ValueError, match=field):
        validate_spec(spec)


def test_legacy_status_field_is_rejected() -> None:
    spec = copy.deepcopy(load_spec(SPEC_PATH))
    spec["pages"][0]["nodes"][0]["status"] = "BASELINE"
    with pytest.raises(ValueError, match="legacy"):
        validate_spec(spec)
