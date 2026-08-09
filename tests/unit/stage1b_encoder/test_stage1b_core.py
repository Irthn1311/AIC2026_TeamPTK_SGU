from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest

from triage_eg.retrieval.stage1b.assets import inventory_fingerprint, sha256_file
from triage_eg.retrieval.stage1b.contracts import CandidateContract, CompatibilityGate
from triage_eg.retrieval.stage1b.evidence import EvidenceLimits, discover_encoder_evidence
from triage_eg.retrieval.stage1b.probe import (
    ALIGNMENT_BASIS,
    decide_candidate,
    exact_stored_vector_alignment,
    validate_embedding_matrix,
)
from triage_eg.retrieval.stage1b.registry import load_candidate_registry
from triage_eg.retrieval.stage1b.writers import REPORT_MEMBERS, create_stage1b_report_bundle


def contract(**changes) -> CandidateContract:
    values = {
        "candidate_id": "candidate",
        "enabled": True,
        "implementation": "custom",
        "architecture": "ViT-B/32",
        "pretrained": "local",
        "checkpoint_path": "checkpoint.bin",
        "tokenizer": "tokenizer",
        "context_length": 77,
        "image_preprocessing": {
            "resize": 224,
            "crop": 224,
            "interpolation": "bicubic",
            "convert_rgb": True,
            "mean": [1, 1, 1],
            "std": [1, 1, 1],
        },
        "text_preprocessing": {
            "strip": False,
            "lowercase": False,
            "unicode_normalization": None,
        },
        "evidence_source": "HYPOTHESIS",
    }
    values.update(changes)
    return CandidateContract(**values)


def summary(**changes) -> dict:
    value = {
        "samples_completed": 20,
        "cosine": {"mean": 1.0, "min": 1.0},
        "stored_vector_equivalence_alignment": {"top1_rate": 1.0, "top5_rate": 1.0},
        "dimension_match": True,
        "finite_all": True,
    }
    value.update(changes)
    return value


def test_repository_and_dataset_evidence_are_bounded(tmp_path: Path) -> None:
    repo, data = tmp_path / "repo", tmp_path / "data"
    repo.mkdir()
    data.mkdir()
    (repo / "README.md").write_text("OpenCLIP ViT-B-32 tokenizer preprocess", encoding="utf-8")
    (data / "model_metadata.json").write_text('{"model":"CLIP"}', encoding="utf-8")
    object_root = data / "objects"
    object_root.mkdir()
    (object_root / "ignored.json").write_text('{"checkpoint":"x"}', encoding="utf-8")
    records, result = discover_encoder_evidence(
        repo, data, EvidenceLimits(repository_files=2, dataset_files=2, excerpt_chars=20)
    )
    assert {item["source_type"] for item in records} == {
        "REPOSITORY_FILE",
        "DATASET_METADATA",
    }
    assert all(len(item["excerpt"]) <= 20 for item in records)
    assert not result["authoritative_metadata_found"] and result["bounded"]
    assert all("ignored.json" not in item["path"] for item in records)


def test_evidence_ids_are_reproducible(tmp_path: Path) -> None:
    repo, data = tmp_path / "repo", tmp_path / "data"
    repo.mkdir()
    data.mkdir()
    (repo / "README.md").write_text("CLIP checkpoint", encoding="utf-8")
    first = discover_encoder_evidence(repo, data)[0]
    second = discover_encoder_evidence(repo, data)[0]
    assert first == second


@pytest.mark.parametrize(
    "changes",
    [
        {"candidate_id": "../bad"},
        {"implementation": "bad"},
        {"output_dimension": 0},
        {"runtime_dtype": "float16"},
        {"evidence_source": "DIMENSION_ONLY"},
    ],
)
def test_invalid_candidate_contract(changes: dict) -> None:
    with pytest.raises(ValueError):
        contract(**changes)


def test_dimension_only_and_user_assertion_cannot_verify() -> None:
    incomplete = contract(checkpoint_path=None, evidence_source="USER_ASSERTED")
    decision, reasons = decide_candidate(
        incomplete, summary(), CompatibilityGate(), provenance_complete=False
    )
    assert decision == "UNVERIFIED" and "ENCODER_PROVENANCE_INCOMPLETE" in reasons


def test_project_gate_verifies_reproducible_candidate() -> None:
    assert decide_candidate(contract(), summary(), CompatibilityGate(), True)[0] == "VERIFIED"


def test_gate_rejects_wrong_encoder() -> None:
    bad = summary(
        cosine={"mean": 0.1, "min": -0.1},
        stored_vector_equivalence_alignment={"top1_rate": 0.0, "top5_rate": 0.0},
    )
    decision, reasons = decide_candidate(contract(), bad, CompatibilityGate(), True)
    assert decision == "REJECTED"
    assert "PAIRWISE_COSINE_BELOW_GATE" in reasons


def test_locked_gate_thresholds_are_unchanged() -> None:
    gate = CompatibilityGate()
    assert gate.pairwise_cosine_mean_min == 0.995
    assert gate.pairwise_cosine_min_min == 0.98
    assert gate.target_top1_rate_min == 0.95
    assert gate.target_top5_rate_min == 1.0


def test_unique_target_row_passes_literal_and_equivalence_alignment() -> None:
    stored = np.eye(3, dtype=np.float32)
    diagnostics, literal, equivalent = exact_stored_vector_alignment(
        np.asarray([[1, 0, 2]]),
        np.asarray([1]),
        stored[[[1, 0, 2]]],
        stored[[1]],
    )
    assert diagnostics[0]["exact_target_row_in_top1"]
    assert diagnostics[0]["equivalent_vector_in_top1"]
    assert literal["top1_rate"] == equivalent["top1_rate"] == 1.0


def test_exact_duplicate_displacement_passes_equivalence_not_literal() -> None:
    vector = np.asarray([1.0, 0.0], dtype=np.float32)
    other = np.asarray([0.0, 1.0], dtype=np.float32)
    diagnostics, literal, equivalent = exact_stored_vector_alignment(
        np.asarray([[5, 10, 20]]),
        np.asarray([10]),
        np.asarray([[vector, vector, other]]),
        np.asarray([vector]),
    )
    item = diagnostics[0]
    assert not item["exact_target_row_in_top1"]
    assert item["exact_target_row_rank"] == 2
    assert item["equivalent_vector_in_top1"]
    assert item["literal_target_row_tie_displacement"]
    assert literal["top1_rate"] == 0.0 and equivalent["top1_rate"] == 1.0


def test_large_exact_group_passes_when_literal_target_is_outside_top20() -> None:
    vector = np.asarray([1.0, 0.0], dtype=np.float32)
    returned = np.repeat(vector[None, None, :], 20, axis=1)
    diagnostics, literal, equivalent = exact_stored_vector_alignment(
        np.arange(20, dtype=np.int64)[None, :],
        np.asarray([29]),
        returned,
        np.asarray([vector]),
    )
    assert diagnostics[0]["exact_target_row_rank"] is None
    assert diagnostics[0]["equivalent_vector_rank"] == 1
    assert diagnostics[0]["equivalent_vector_count_in_returned_topk"] == 20
    assert literal["top20_rate"] == 0.0
    assert equivalent["top1_rate"] == equivalent["top5_rate"] == 1.0


def test_near_duplicate_is_not_exact_stored_vector_equivalent() -> None:
    target = np.asarray([1.0, 0.0], dtype=np.float32)
    near = np.nextafter(target, np.asarray([2.0, 1.0], dtype=np.float32))
    diagnostics, _, equivalent = exact_stored_vector_alignment(
        np.asarray([[5, 10]]),
        np.asarray([10]),
        np.asarray([[near, target]]),
        np.asarray([target]),
    )
    assert not np.array_equal(near, target)
    assert not diagnostics[0]["equivalent_vector_in_top1"]
    assert diagnostics[0]["equivalent_vector_rank"] == 2
    assert equivalent["alignment_basis"] == ALIGNMENT_BASIS


def test_gate_keeps_insufficient_sample_unverified() -> None:
    decision, reasons = decide_candidate(
        contract(), summary(samples_completed=5), CompatibilityGate(), True
    )
    assert decision == "UNVERIFIED" and "INSUFFICIENT_COMPLETED_SAMPLES" in reasons


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (np.ones((1, 511)), "DIMENSION"),
        (np.full((1, 512), np.nan), "NON_FINITE"),
        (np.zeros((1, 512)), "ZERO_NORM"),
    ],
)
def test_embedding_validation(value: np.ndarray, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        validate_embedding_matrix(value, 1)


def test_embedding_validation_preserves_float32() -> None:
    result = validate_embedding_matrix(np.ones((2, 512), dtype=np.float16), 2)
    assert result.dtype == np.float32


def test_checkpoint_and_inventory_fingerprint(tmp_path: Path) -> None:
    first = tmp_path / "a.bin"
    first.write_bytes(b"abc")
    directory = tmp_path / "model"
    directory.mkdir()
    (directory / "a.bin").write_bytes(b"abc")
    assert len(sha256_file(first)) == 64
    assert inventory_fingerprint(directory) == inventory_fingerprint(directory)


def test_registry_disabled_candidate_is_not_enabled(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        "stage1b_version: '0.1.1'\ncompatibility_gate: {}\ncandidates:\n"
        "  - candidate_id: c\n    enabled: false\n    implementation: unknown\n"
        "    architecture: unknown\n    pretrained: unknown\n    checkpoint_path: null\n",
        encoding="utf-8",
    )
    candidates, _, _, fingerprint = load_candidate_registry(path)
    assert not candidates[0].enabled and len(fingerprint) == 64


def test_zip_has_only_allowlisted_existing_members(tmp_path: Path) -> None:
    root = tmp_path / "out"
    for name in REPORT_MEMBERS[:3]:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}" if path.suffix == ".json" else "report", encoding="utf-8")
    (root / "model.bin").write_bytes(b"secret")
    (root / "logs").mkdir()
    (root / "logs/log.txt").write_text("log", encoding="utf-8")
    target = create_stage1b_report_bundle(root, tmp_path / "reports.zip")
    with ZipFile(target) as archive:
        assert archive.namelist() == list(REPORT_MEMBERS[:3])
        assert "model.bin" not in archive.namelist()


def test_no_download_calls_in_stage1b_source() -> None:
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in Path("src/triage_eg/retrieval/stage1b").glob("*.py")
    )
    for forbidden in ("requests.get", "urlretrieve", "hf_hub_download", "snapshot_download"):
        assert forbidden not in source
