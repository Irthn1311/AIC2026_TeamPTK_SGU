from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest

from triage_eg.retrieval.stage1.encoder import compatibility_gate, validate_encoder_output
from triage_eg.retrieval.stage1.writers import STAGE1B_INPUT_MEMBERS
from triage_eg.retrieval.stage1b.assets import (
    asset_source,
    inventory_fingerprint,
    load_multimodal_encoder,
    preflight_candidate,
)
from triage_eg.retrieval.stage1b.contracts import CandidateContract, CompatibilityGate
from triage_eg.retrieval.stage1b.inputs import STAGE1_REQUIRED, resolve_stage1_root
from triage_eg.retrieval.stage1b.pipeline import _load_smoke_queries
from triage_eg.retrieval.stage1b.probe import decide_candidate
from triage_eg.retrieval.stage1b.smoke import stage1_contract


def candidate(tmp_path: Path, **changes) -> CandidateContract:
    checkpoint = tmp_path / "checkpoint.bin"
    checkpoint.write_bytes(b"local-only")
    values = {
        "candidate_id": "candidate",
        "enabled": True,
        "implementation": "custom",
        "architecture": "ViT-B/32",
        "pretrained": "local",
        "checkpoint_path": str(checkpoint),
        "tokenizer": "local-tokenizer",
        "context_length": 77,
        "image_preprocessing": {
            "resize": 224,
            "crop": 224,
            "interpolation": "bicubic",
            "convert_rgb": True,
            "mean": [0.1, 0.2, 0.3],
            "std": [0.4, 0.5, 0.6],
        },
        "text_preprocessing": {
            "strip": False,
            "lowercase": False,
            "unicode_normalization": None,
        },
    }
    values.update(changes)
    return CandidateContract(**values)


def passing_summary(**changes) -> dict:
    value = {
        "samples_completed": 20,
        "dimension_match": True,
        "finite_all": True,
        "cosine": {"min": 0.999, "mean": 0.9999},
        "stored_vector_equivalence_alignment": {"top1_rate": 1.0, "top5_rate": 1.0},
    }
    value.update(changes)
    return value


def test_valid_contract_is_reproducible(tmp_path: Path) -> None:
    assert candidate(tmp_path).reproducible()


def test_missing_required_constructor_field_fails() -> None:
    with pytest.raises(TypeError):
        CandidateContract(candidate_id="x")  # type: ignore[call-arg]


@pytest.mark.parametrize("status", ["USER_ASSERTED", "UNVERIFIED", "BLOCKED"])
def test_nonverified_status_never_passes_stage1_text_gate(tmp_path: Path, status: str) -> None:
    contract = stage1_contract(candidate(tmp_path, compatibility_status=status))
    with pytest.raises(PermissionError):
        compatibility_gate(contract, allow_unverified=False)


def test_override_remains_unverified_override(tmp_path: Path) -> None:
    contract = stage1_contract(candidate(tmp_path, compatibility_status="UNVERIFIED"))
    assert compatibility_gate(contract, allow_unverified=True) == "UNVERIFIED_OVERRIDE"


@pytest.mark.parametrize(
    "values",
    [
        np.ones((1, 511), dtype=np.float32),
        np.full((1, 512), np.inf, dtype=np.float32),
        np.zeros((1, 512), dtype=np.float32),
    ],
)
def test_stage1_text_output_validation_rejects_invalid(values: np.ndarray) -> None:
    with pytest.raises(ValueError):
        validate_encoder_output(values, 1)


def test_stage1_text_output_validation_accepts_finite_512() -> None:
    assert validate_encoder_output(np.ones((2, 512)), 2).shape == (2, 512)


def test_dimension_mismatch_decision_is_rejected(tmp_path: Path) -> None:
    decision, reasons = decide_candidate(
        candidate(tmp_path),
        passing_summary(dimension_match=False),
        CompatibilityGate(),
        True,
    )
    assert decision == "REJECTED" and reasons == ["ENCODER_OUTPUT_DIMENSION_MISMATCH"]


def test_nonfinite_decision_is_rejected(tmp_path: Path) -> None:
    decision, reasons = decide_candidate(
        candidate(tmp_path), passing_summary(finite_all=False), CompatibilityGate(), True
    )
    assert decision == "REJECTED" and reasons == ["ENCODER_OUTPUT_NON_FINITE"]


def test_near_perfect_matching_passes_configured_gate(tmp_path: Path) -> None:
    gate = CompatibilityGate(pairwise_cosine_mean_min=0.999, pairwise_cosine_min_min=0.998)
    assert decide_candidate(candidate(tmp_path), passing_summary(), gate, True)[0] == "VERIFIED"


def test_threshold_change_changes_registry_fingerprint(tmp_path: Path) -> None:
    from triage_eg.retrieval.stage1b.registry import load_candidate_registry

    template = (
        "stage1b_version: '0.1.1'\ncompatibility_gate:\n"
        "  pairwise_cosine_mean_min: {threshold}\ncandidates: []\n"
    )
    first, second = tmp_path / "first.yaml", tmp_path / "second.yaml"
    first.write_text(template.format(threshold="0.995"), encoding="utf-8")
    second.write_text(template.format(threshold="0.996"), encoding="utf-8")
    assert load_candidate_registry(first)[3] != load_candidate_registry(second)[3]


def test_missing_checkpoint_is_blocked_before_adapter(tmp_path: Path) -> None:
    item = candidate(tmp_path, checkpoint_path=str(tmp_path / "missing.bin"))
    provenance, issues = preflight_candidate(item, tmp_path, tmp_path / "dataset")
    assert not provenance["asset_available"]
    assert [issue["code"] for issue in issues] == ["ENCODER_ASSET_NOT_FOUND"]


def test_declared_non512_candidate_is_not_runnable(tmp_path: Path) -> None:
    _, issues = preflight_candidate(
        candidate(tmp_path, output_dimension=511), tmp_path, tmp_path / "dataset"
    )
    assert "ENCODER_OUTPUT_DIMENSION_MISMATCH" in {item["code"] for item in issues}


def test_missing_dependency_is_reported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "triage_eg.retrieval.stage1b.assets.importlib.util.find_spec", lambda _: None
    )
    _, issues = preflight_candidate(
        candidate(tmp_path, implementation="open_clip", tokenizer="open_clip_simple"),
        tmp_path,
        tmp_path / "dataset",
    )
    assert "ENCODER_DEPENDENCY_NOT_AVAILABLE" in {item["code"] for item in issues}


def test_adapter_factory_is_lazy_and_explicit(tmp_path: Path) -> None:
    sentinel = object()
    result = load_multimodal_encoder(candidate(tmp_path), lambda _: sentinel)  # type: ignore[arg-type]
    assert result is sentinel


@pytest.mark.parametrize(
    ("location", "expected"),
    [("repo/model.bin", "REPOSITORY"), ("dataset/model.bin", "KAGGLE_INPUT")],
)
def test_asset_source_classification(tmp_path: Path, location: str, expected: str) -> None:
    repo, dataset = tmp_path / "repo", tmp_path / "dataset"
    path = tmp_path / location
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")
    assert asset_source(path, repo, dataset) == expected


def test_directory_fingerprint_is_bounded(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    for index in range(3):
        (model / f"{index}.bin").write_bytes(bytes([index]))
    with pytest.raises(ValueError, match="bounded"):
        inventory_fingerprint(model, max_files=2)


def test_directory_fingerprint_changes_with_content(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    file = model / "weights.bin"
    file.write_bytes(b"one")
    before = inventory_fingerprint(model)
    file.write_bytes(b"two")
    assert inventory_fingerprint(model) != before


def test_smoke_query_jsonl_is_config_driven(tmp_path: Path) -> None:
    path = tmp_path / "queries.jsonl"
    expected = {"query_id": "q", "text": "hello", "language": "en", "type": "object"}
    path.write_text(json.dumps(expected) + "\n", encoding="utf-8")
    assert _load_smoke_queries(path) == [expected]


@pytest.mark.parametrize("record", [{}, {"query_id": "q"}, []])
def test_invalid_smoke_query_jsonl_fails(tmp_path: Path, record) -> None:
    path = tmp_path / "queries.jsonl"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        _load_smoke_queries(path)


def test_saved_stage1_root_is_found_at_kaggle_style_depth(tmp_path: Path) -> None:
    attached = tmp_path / "datasets" / "owner" / "saved-stage1"
    for name in STAGE1_REQUIRED:
        path = attached / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
    resolved = resolve_stage1_root(tmp_path / "working-missing", search_root=tmp_path)
    assert resolved == attached.resolve()


def test_existing_kaggle_mount_discovers_nested_complete_stage1_root(tmp_path: Path) -> None:
    mount = tmp_path / "datasets/owner/triage-eg-stage1-baseline"
    complete = mount / "triage_eg_stage1_baseline"
    for name in STAGE1_REQUIRED:
        path = complete / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
    resolved = resolve_stage1_root(mount, search_root=tmp_path)
    assert resolved == complete.resolve()


def test_report_only_stage1_mount_has_actionable_error(tmp_path: Path) -> None:
    mount = tmp_path / "datasets/owner/triage-eg-stage1-baseline"
    (mount / "index").mkdir(parents=True)
    (mount / "stage1_summary.json").write_text("{}", encoding="utf-8")
    (mount / "index/index_manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="report-only bundle is insufficient"):
        resolve_stage1_root(mount, search_root=tmp_path)


def test_stage1b_input_zip_is_materialized_for_notebook07(tmp_path: Path) -> None:
    mount = tmp_path / "datasets/owner/triage-eg-stage1-baseline"
    mount.mkdir(parents=True)
    bundle = mount / "triage_eg_stage1b_input_bundle.zip"
    with ZipFile(bundle, "w") as archive:
        for name in STAGE1B_INPUT_MEMBERS:
            archive.writestr(name, b"x")
    materialized = tmp_path / "working/triage_eg_stage1b_saved_input"
    resolved = resolve_stage1_root(
        mount,
        search_root=tmp_path,
        materialize_root=materialized,
    )
    assert resolved == materialized.resolve()
    assert all((resolved / name).is_file() for name in STAGE1_REQUIRED)
    assert resolve_stage1_root(
        mount,
        search_root=tmp_path,
        materialize_root=materialized,
    ) == materialized.resolve()


def test_stage1b_input_zip_rejects_unexpected_member(tmp_path: Path) -> None:
    mount = tmp_path / "mount"
    mount.mkdir()
    bundle = mount / "triage_eg_stage1b_input_bundle.zip"
    with ZipFile(bundle, "w") as archive:
        for name in STAGE1B_INPUT_MEMBERS:
            archive.writestr(name, b"x")
        archive.writestr("../escape.txt", b"unsafe")
    with pytest.raises(ValueError, match="unexpected or duplicate"):
        resolve_stage1_root(
            mount,
            search_root=mount,
            materialize_root=tmp_path / "materialized",
        )
    assert not (tmp_path / "escape.txt").exists()


def test_saved_stage1_root_discovery_fails_on_ambiguity(tmp_path: Path) -> None:
    for root_name in ("one", "two"):
        for name in STAGE1_REQUIRED:
            path = tmp_path / root_name / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"x")
    with pytest.raises(FileNotFoundError, match="exactly one"):
        resolve_stage1_root(tmp_path / "missing", search_root=tmp_path)
