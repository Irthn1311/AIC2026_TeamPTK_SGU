"""Core contract tests for the TRIAGE-EG v1.1 source of truth."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from generate_architecture_assets import load_spec, validate_spec  # noqa: E402

SPEC_PATH = PROJECT_ROOT / "docs" / "architecture" / "architecture-spec.yaml"


def test_architecture_has_exact_v11_page_set_and_valid_graphs() -> None:
    spec = load_spec(SPEC_PATH)
    validate_spec(spec)
    assert spec["project"]["version"] == "1.1"
    assert [page["id"] for page in spec["pages"]] == [f"PAGE_{index:02d}" for index in range(9)]
    assert all(page["nodes"] and page["edges"] for page in spec["pages"])


def test_model_registry_is_evidence_safe() -> None:
    models = load_spec(SPEC_PATH)["models"]
    assert models["btc_clip"]["selected"] == "CLIP ViT-B/32"
    assert models["btc_object_detector"]["selected"] == "Faster R-CNN Inception-ResNet-v2"
    assert models["btc_object_detector"]["training_dataset"] == "Open Images V4"
    for model in models.values():
        if model["validation_status"] == "UNSELECTED":
            assert model["selected"] is None
    assert models["GNN"]["criticality"] == "DEFERRED"
    assert models["numpy_flat_cosine"]["source_basis"] == "SOFTWARE_TEMPLATE"


def test_yaml_is_safe_loadable_without_custom_types() -> None:
    with SPEC_PATH.open(encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    assert payload["project"]["system_name"] == "TRIAGE-EG"


def test_overview_is_macro_level_and_complete() -> None:
    spec = load_spec(SPEC_PATH)
    overview = next(page for page in spec["pages"] if page["id"] == "PAGE_01")
    titles = {node["title"] for node in overview["nodes"]}
    assert len(overview["nodes"]) == 24
    assert {
        "Raw Data",
        "Multimodal Retrieval",
        "Fast Lane or Graph Lane?",
        "Candidate Local Event Graph",
        "Semantic Moment Localizer",
        "Evidence Packet",
        "Application Backend / Orchestrator",
        "Interactive UI / Operator Console",
        "Automatic Batch Mode",
        "Agent Control Bus",
    } <= titles
