from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from zipfile import ZipFile

import numpy as np
import pytest

from triage_eg.retrieval.stage1.builder import Stage1BuildConfig, build_index
from triage_eg.retrieval.stage1b import Stage1BConfig, run_stage1b
from triage_eg.retrieval.stage1b.adapters.openai_clip_official import (
    OfficialOpenAIClipAdapter,
    resolve_official_asset_paths,
)
from triage_eg.retrieval.stage1b.sampling import select_probe_samples
from triage_eg.retrieval.stage1b.writers import REPORT_MEMBERS, create_stage1b_report_bundle


class FakeTensor:
    def __init__(self, values):
        self.values = np.asarray(values)
        self.dtype = self.values.dtype

    def __len__(self):
        return len(self.values)

    def to(self, device):
        return self

    def detach(self):
        return self

    def float(self):
        return FakeTensor(self.values.astype(np.float32))

    def cpu(self):
        return self

    def numpy(self):
        return self.values


def fake_torch_module() -> ModuleType:
    torch = ModuleType("torch")
    torch.__version__ = "fake"
    torch.cuda = SimpleNamespace(
        is_available=lambda: False,
        get_device_name=lambda index: None,
        empty_cache=lambda: None,
    )
    torch.stack = lambda tensors: FakeTensor(np.stack([tensor.values for tensor in tensors]))

    class NoGrad:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc_value, traceback):
            return False

    torch.no_grad = NoGrad
    return torch


@pytest.fixture(autouse=True)
def isolated_official_clip_modules():
    saved_clip = {
        name: module
        for name, module in sys.modules.items()
        if name == "clip" or name.startswith("clip.")
    }
    for name in saved_clip:
        sys.modules.pop(name, None)
    original_torch = sys.modules.get("torch")
    if original_torch is None:
        sys.modules["torch"] = fake_torch_module()
    yield
    for name in list(sys.modules):
        if name == "clip" or name.startswith("clip."):
            sys.modules.pop(name, None)
    sys.modules.update(saved_clip)
    if original_torch is None:
        sys.modules.pop("torch", None)
    else:
        sys.modules["torch"] = original_torch


def fixture(root: Path, *, exact_duplicate_vectors: bool = False) -> tuple[Path, Path, Path]:
    stage0, data = root / "stage0", root / "data"
    stage0.mkdir()
    data.mkdir()
    videos = ["L01_V001", "L01_V002", "L01_V003"]
    total = len(videos) * 3
    (stage0 / "audit_summary.json").write_text(
        json.dumps(
            {
                "audit_version": "0.1.0",
                "mode": "full",
                "videos_discovered": 3,
                "videos_completed": 3,
                "mapping_rows": total,
                "clip_rows": total,
                "config_fingerprint": "stage0-fp",
                "git_commit": "stage0-commit",
                "gates": {"btc_baseline": "PASS_WITH_WARNINGS"},
                "unknown_contracts": ["CLIP model compatibility"],
            }
        ),
        encoding="utf-8",
    )
    (stage0 / "run_manifest.json").write_text('{"status":"COMPLETE"}', encoding="utf-8")
    (stage0 / "contract_notes.json").write_text(
        json.dumps(
            {
                "original_frame_policy": (
                    "CSV frame_idx is authoritative; never reconstruct from pts_time*fps"
                )
            }
        ),
        encoding="utf-8",
    )
    frames, clips = [], []
    for video_index, video_id in enumerate(videos):
        matrix = np.zeros((3, 512), dtype=np.float16)
        for row in range(3):
            matrix[row, 0 if exact_duplicate_vectors else video_index * 3 + row] = 1
        relative = f"clips/{video_id}.npy"
        (data / "clips").mkdir(exist_ok=True)
        np.save(data / relative, matrix)
        clips.append(
            {
                "video_id": video_id,
                "relative_path": relative,
                "shape": [3, 512],
                "row_count": 3,
                "dimension": 512,
                "dtype": "float16",
            }
        )
        for n in range(1, 4):
            keyframe = f"keyframes/{video_id}/{n:03d}.jpg"
            target = data / keyframe
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"mock-jpg")
            frames.append(
                {
                    "video_id": video_id,
                    "n": n,
                    "clip_row_index": n - 1,
                    "pts_time": n / 25,
                    "mapping_fps": 25.0,
                    "original_frame_idx": video_index * 10 + n,
                    "keyframe_relative_path": keyframe,
                    "duplicate_frame_idx_group_size": 1,
                }
            )
    (stage0 / "btc_frame_manifest.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in frames), encoding="utf-8"
    )
    (stage0 / "clip_manifest.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in clips), encoding="utf-8"
    )
    stage1 = root / "stage1"
    build_index(
        Stage1BuildConfig(
            stage0_root=stage0,
            dataset_root=data,
            output_root=stage1,
            expected_rows=total,
            expected_videos=3,
            self_queries=3,
            overwrite=True,
            build_git_commit="stage1-commit",
        )
    )
    return stage0, data, stage1


def candidate_config(root: Path, checkpoint: Path, enabled: bool = True) -> Path:
    path = root / "candidates.yaml"
    path.write_text(
        "stage1b_version: '0.1.1'\n"
        "compatibility_gate:\n  minimum_completed_samples: 3\n"
        "  pairwise_cosine_mean_min: 0.995\n  pairwise_cosine_min_min: 0.98\n"
        "  target_top1_rate_min: 0.95\n  target_top5_rate_min: 1.0\n"
        "  require_dimension_512: true\n  require_all_finite: true\n"
        "  implementation_equivalence_cosine_min: 0.999999\n"
        "candidates:\n  - candidate_id: mock_candidate\n"
        f"    enabled: {str(enabled).lower()}\n    implementation: custom\n"
        "    architecture: ViT-B/32\n    pretrained: local\n"
        f"    checkpoint_path: '{checkpoint.as_posix()}'\n    tokenizer: mock\n"
        "    context_length: 77\n"
        "    output_dimension: 512\n    runtime_dtype: float32\n"
        "    image_preprocessing: {resize: 224, crop: 224, interpolation: bicubic, "
        "convert_rgb: true, mean: [1,1,1], std: [1,1,1]}\n"
        "    text_preprocessing: {strip: false, lowercase: false, unicode_normalization: null}\n"
        "    image_embedding_normalization: true\n"
        "    text_embedding_normalization: true\n    evidence_source: HYPOTHESIS\n",
        encoding="utf-8",
    )
    return path


def two_candidate_config(root: Path, checkpoint: Path) -> Path:
    base = candidate_config(root, checkpoint).read_text(encoding="utf-8")
    second = (
        "  - candidate_id: second_candidate\n    enabled: true\n"
        "    implementation: custom\n    architecture: ViT-B/32\n"
        "    pretrained: local\n"
        f"    checkpoint_path: '{checkpoint.as_posix()}'\n"
        "    tokenizer: mock\n    context_length: 77\n"
        "    output_dimension: 512\n    runtime_dtype: float32\n"
        "    image_preprocessing: {resize: 224, crop: 224, interpolation: bicubic, "
        "convert_rgb: true, mean: [1,1,1], std: [1,1,1]}\n"
        "    text_preprocessing: {strip: false, lowercase: false, "
        "unicode_normalization: null}\n"
        "    image_embedding_normalization: true\n"
        "    text_embedding_normalization: true\n    evidence_source: HYPOTHESIS\n"
        "    runtime_priority: 5\n"
    )
    path = root / "two_candidates.yaml"
    path.write_text(base + second, encoding="utf-8")
    return path


def official_asset(root: Path, *, checkpoint_exists: bool = True) -> tuple[Path, Path, Path]:
    import hashlib

    asset = root / "official_asset"
    source = asset / "source/openai_clip"
    package = source / "clip"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(
        "def load(*args, **kwargs): return None\ndef tokenize(*args, **kwargs): return None\n",
        encoding="utf-8",
    )
    (package / "bpe_simple_vocab_16e6.txt.gz").write_bytes(b"tokenizer")
    checkpoint = asset / "checkpoint/ViT-B-32.pt"
    checkpoint.parent.mkdir(parents=True)
    if checkpoint_exists:
        checkpoint.write_bytes(b"checkpoint")
    digest = hashlib.sha256(b"checkpoint").hexdigest()
    manifests = asset / "manifests"
    manifests.mkdir()
    (manifests / "checkpoint.sha256").write_text(digest, encoding="utf-8")
    (manifests / "SOURCE_COMMIT.txt").write_text("fake-source-commit", encoding="utf-8")
    (manifests / "asset_manifest.json").write_text(
        json.dumps(
            {
                "asset_bundle_version": "0.1.0",
                "implementation": "openai_clip",
                "source_repository": "OpenAI/CLIP",
                "source_commit": "fake-source-commit",
                "architecture": "ViT-B/32",
                "pretrained": "openai",
                "checkpoint_relative_path": "checkpoint/ViT-B-32.pt",
                "checkpoint_sha256": digest,
                "internet_required_at_runtime": False,
            }
        ),
        encoding="utf-8",
    )
    return asset, source, checkpoint


def official_candidate_config(root: Path, asset: Path, source: Path, checkpoint: Path) -> Path:
    path = root / "official_candidate.yaml"
    path.write_text(
        "stage1b_version: '0.1.1'\n"
        "compatibility_gate:\n  minimum_completed_samples: 3\n"
        "  pairwise_cosine_mean_min: 0.995\n  pairwise_cosine_min_min: 0.98\n"
        "  target_top1_rate_min: 0.95\n  target_top5_rate_min: 1.0\n"
        "  require_dimension_512: true\n  require_all_finite: true\n"
        "  implementation_equivalence_cosine_min: 0.999999\n"
        "candidates:\n  - candidate_id: openai_clip_vit_b32_openai_official\n"
        "    enabled: true\n    implementation: openai_clip\n"
        "    architecture: ViT-B/32\n    pretrained: openai\n"
        f"    source_root: '{source.as_posix()}'\n"
        f"    checkpoint_path: '{checkpoint.as_posix()}'\n"
        f"    asset_manifest_path: '{(asset / 'manifests/asset_manifest.json').as_posix()}'\n"
        "    tokenizer: official clip.tokenize\n    context_length: 77\n"
        "    image_preprocessing: {source: official_clip_load_return_value, "
        "manual_preprocess_override: false}\n"
        "    text_preprocessing: {strip: false, lowercase: false, "
        "unicode_normalization: null}\n"
        "    output_dimension: 512\n    runtime_dtype: float32\n"
        "    image_embedding_normalization: true\n"
        "    text_embedding_normalization: true\n    text_truncate: false\n"
        "    device: cpu\n    batch_size: 2\n"
        "    evidence_source: HYPOTHESIS\n    runtime_priority: 10\n",
        encoding="utf-8",
    )
    return path


class MockEncoder:
    def __init__(self, wrong: bool = False, constant: bool = False) -> None:
        self.wrong = wrong
        self.constant = constant

    def encode_images(self, paths: list[Path]) -> np.ndarray:
        output = np.zeros((len(paths), 512), dtype=np.float32)
        for index, path in enumerate(paths):
            video = int(path.parent.name[-3:]) - 1
            n = int(path.stem) - 1
            row = 0 if self.constant else video * 3 + n
            output[index, row + 100 if self.wrong else row] = 1
        return output

    def encode_text(self, texts: list[str]) -> np.ndarray:
        output = np.zeros((len(texts), 512), dtype=np.float32)
        output[:, 0] = 1
        return output


class NearMatchingEncoder(MockEncoder):
    def encode_images(self, paths: list[Path]) -> np.ndarray:
        output = super().encode_images(paths)
        for index in range(len(output)):
            target = int(np.flatnonzero(output[index])[0])
            alternate = (target + 50) % 512
            output[index, target] = 0.999
            output[index, alternate] = np.sqrt(1 - 0.999**2)
        return output


def run_case(
    tmp_path: Path,
    *,
    wrong: bool = False,
    enabled: bool = True,
    exact_duplicate_vectors: bool = False,
):
    stage0, data, stage1 = fixture(
        tmp_path, exact_duplicate_vectors=exact_duplicate_vectors
    )
    checkpoint = tmp_path / "checkpoint.bin"
    checkpoint.write_bytes(b"checkpoint")
    output = tmp_path / "stage1b"
    result = run_stage1b(
        Stage1BConfig(
            repo_root=Path.cwd(),
            dataset_root=data,
            stage0_root=stage0,
            stage1_root=stage1,
            output_root=output,
            candidate_config=candidate_config(tmp_path, checkpoint, enabled),
            sample_size=3,
            overwrite=True,
            build_git_commit="stage1b-commit",
        ),
        adapter_factory=lambda candidate: MockEncoder(wrong, exact_duplicate_vectors),
    )
    return result


def test_matching_encoder_verifies_and_enables_smoke(tmp_path: Path) -> None:
    result = run_case(tmp_path)
    assert result.summary["readiness"] == {
        "encoder_readiness": "VERIFIED",
        "encoder_compatibility": "VERIFIED",
        "text_retrieval": "READY_FOR_QUALITATIVE_TESTING",
    }
    assert result.summary["probe"]["candidates_verified"] == 1
    assert result.summary["text_smoke_status"] == "PASS"
    assert result.selected_contract["checkpoint_fingerprint"]
    candidates = [
        json.loads(line)
        for line in (result.output_root / "probe/candidate_results.jsonl").read_text().splitlines()
    ]
    assert len(candidates) == 3 and all(item["cosine_to_stored"] == 1 for item in candidates)
    assert all(item["matched_row_in_top1"] for item in candidates)
    assert all(item["equivalent_vector_in_top1"] for item in candidates)
    archive = create_stage1b_report_bundle(result.output_root, tmp_path / "report.zip")
    with ZipFile(archive) as stream:
        assert stream.namelist() == list(REPORT_MEMBERS)
        assert not any(name.endswith((".bin", ".npy", ".jpg")) for name in stream.namelist())


def test_wrong_encoder_is_rejected_and_text_stays_blocked(tmp_path: Path) -> None:
    result = run_case(tmp_path, wrong=True)
    assert result.summary["probe"]["candidates_rejected"] == 1
    assert result.summary["readiness"]["encoder_compatibility"] == "BLOCKED"
    assert result.summary["text_smoke_status"] == "NOT_RUN"


def test_exact_duplicate_ties_verify_by_stored_vector_equivalence(tmp_path: Path) -> None:
    result = run_case(tmp_path, exact_duplicate_vectors=True)
    candidate_summary = json.loads(
        (result.output_root / "probe/candidate_summaries.jsonl").read_text().splitlines()[0]
    )
    assert candidate_summary["literal_target_alignment"]["top1_rate"] < 1.0
    assert candidate_summary["stored_vector_equivalence_alignment"]["top1_rate"] == 1.0
    assert candidate_summary["stored_vector_equivalence_alignment"]["top5_rate"] == 1.0
    assert candidate_summary["gate_alignment_basis"] == (
        "EXACT_STORED_VECTOR_EQUIVALENCE_CLASS"
    )
    assert candidate_summary["decision"] == "VERIFIED"
    assert result.selected_contract["selected_candidate_id"] == "mock_candidate"
    assert result.summary["selected_candidate"] == {
        "candidate_id": "mock_candidate",
        "candidate_decision": "VERIFIED",
    }
    assert result.summary["text_smoke_status"] == "PASS"
    runtime = json.loads(
        (result.output_root / "encoder/runtime_adapter_manifest.json").read_text()
    )
    assert runtime["selected_candidate_id"] == "mock_candidate"
    per_sample = [
        json.loads(line)
        for line in (result.output_root / "probe/candidate_results.jsonl").read_text().splitlines()
    ]
    displaced = [item for item in per_sample if item["literal_target_row_tie_displacement"]]
    assert displaced
    assert all("LITERAL_TARGET_ROW_TIE_DISPLACEMENT" in item["issue_codes"] for item in displaced)


def test_disabled_candidate_is_quietly_blocked_without_execution(tmp_path: Path) -> None:
    result = run_case(tmp_path, enabled=False)
    assert result.summary["probe"]["candidates_executed"] == 0
    assert result.summary["readiness"]["encoder_compatibility"] == "BLOCKED"
    issues = (result.output_root / "issues.jsonl").read_text()
    assert "ENCODER_ASSET_NOT_FOUND" not in issues


def test_missing_checkpoint_blocks_without_adapter_load(tmp_path: Path) -> None:
    stage0, data, stage1 = fixture(tmp_path)
    missing = tmp_path / "missing.bin"
    loaded = False

    def factory(candidate):
        nonlocal loaded
        loaded = True
        return MockEncoder()

    result = run_stage1b(
        Stage1BConfig(
            repo_root=Path.cwd(),
            dataset_root=data,
            stage0_root=stage0,
            stage1_root=stage1,
            output_root=tmp_path / "stage1b",
            candidate_config=candidate_config(tmp_path, missing),
            sample_size=3,
            overwrite=True,
        ),
        adapter_factory=factory,
    )
    assert not loaded
    assert result.summary["probe"]["candidates_blocked"] == 1


def test_source_has_no_stage0_stage1_build_or_future_features() -> None:
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in Path("src/triage_eg/retrieval/stage1b").glob("*.py")
    )
    for forbidden in (
        "build_index(",
        "run_audit(",
        "VideoCapture",
        "FastLine",
        "EventGraph",
        "AgentRunner",
    ):
        assert forbidden not in source


def test_blocked_selected_contract_has_required_form(tmp_path: Path) -> None:
    result = run_case(tmp_path, enabled=False)
    assert result.selected_contract["compatibility_status"] == "BLOCKED"
    assert result.selected_contract["selected_candidate_id"] is None
    assert result.selected_contract["reason"]


def test_manifest_proves_locked_stage_reuse_and_no_download(tmp_path: Path) -> None:
    result = run_case(tmp_path)
    manifest = json.loads((result.output_root / "run_manifest.json").read_text())
    assert manifest["no_stage0_rerun"]
    assert manifest["no_stage1_rebuild"]
    assert manifest["no_model_download"]
    assert manifest["started_at"] <= manifest["completed_at"]


def test_report_states_quality_and_vietnamese_nonclaims(tmp_path: Path) -> None:
    result = run_case(tmp_path)
    summary = result.summary
    assert "Compatibility does not prove retrieval quality" in summary["non_claims"]
    assert "Compatibility does not prove Vietnamese query quality" in summary["non_claims"]
    assert "Recall@K" in (result.output_root / "stage1b_report.md").read_text()


def test_rejected_candidate_report_retains_evaluation_evidence(tmp_path: Path) -> None:
    result = run_case(tmp_path, wrong=True)
    report = (result.output_root / "stage1b_report.md").read_text(encoding="utf-8")
    assert "## Evaluated Candidates" in report
    assert "Candidate decision: REJECTED" in report
    assert "Checkpoint SHA-256:" in report
    assert "Cosine:" in report
    assert "Literal target-row alignment:" in report
    assert "Stored-vector equivalence alignment:" in report
    assert "did not satisfy the configured compatibility gate" in report
    assert "Provide or fix the required local asset" not in report
    runtime = json.loads(
        (result.output_root / "encoder/runtime_adapter_manifest.json").read_text()
    )
    assert len(runtime["evaluated_candidates"]) == 1
    assert runtime["evaluated_candidates"][0]["candidate_decision"] == "REJECTED"
    assert runtime["selected_candidate_id"] is None
    assert result.summary["evaluation_status"] == "COMPLETE"
    assert result.summary["selected_candidate"] is None
    assert result.summary["evaluated_candidates"] == [
        {
            "candidate_id": "mock_candidate",
            "candidate_decision": "REJECTED",
            "decision_reasons": [
                "PAIRWISE_COSINE_BELOW_GATE",
                "STORED_VECTOR_EQUIVALENCE_ALIGNMENT_BELOW_GATE",
            ],
            "selected": False,
        }
    ]


def test_output_does_not_modify_dataset_or_stage1(tmp_path: Path) -> None:
    stage0, data, stage1 = fixture(tmp_path)
    before_data = {path.relative_to(data): path.stat().st_mtime_ns for path in data.rglob("*")}
    before_stage1 = {
        path.relative_to(stage1): path.stat().st_mtime_ns for path in stage1.rglob("*")
    }
    checkpoint = tmp_path / "checkpoint.bin"
    checkpoint.write_bytes(b"checkpoint")
    run_stage1b(
        Stage1BConfig(
            repo_root=Path.cwd(),
            dataset_root=data,
            stage0_root=stage0,
            stage1_root=stage1,
            output_root=tmp_path / "stage1b",
            candidate_config=candidate_config(tmp_path, checkpoint),
            sample_size=3,
            build_git_commit="stage1b-commit",
        ),
        adapter_factory=lambda _: MockEncoder(),
    )
    assert before_data == {
        path.relative_to(data): path.stat().st_mtime_ns for path in data.rglob("*")
    }
    assert before_stage1 == {
        path.relative_to(stage1): path.stat().st_mtime_ns for path in stage1.rglob("*")
    }


def test_equivalent_adapters_select_configured_canonical_runtime(tmp_path: Path) -> None:
    stage0, data, stage1 = fixture(tmp_path)
    checkpoint = tmp_path / "checkpoint.bin"
    checkpoint.write_bytes(b"checkpoint")
    result = run_stage1b(
        Stage1BConfig(
            repo_root=Path.cwd(),
            dataset_root=data,
            stage0_root=stage0,
            stage1_root=stage1,
            output_root=tmp_path / "stage1b",
            candidate_config=two_candidate_config(tmp_path, checkpoint),
            sample_size=3,
            build_git_commit="stage1b-commit",
        ),
        adapter_factory=lambda _: MockEncoder(),
    )
    assert result.selected_contract["candidate_id"] == "second_candidate"
    assert result.summary["probe"]["candidates_verified"] == 2


def test_distinguishable_passing_adapters_remain_ambiguous(tmp_path: Path) -> None:
    stage0, data, stage1 = fixture(tmp_path)
    checkpoint = tmp_path / "checkpoint.bin"
    checkpoint.write_bytes(b"checkpoint")
    result = run_stage1b(
        Stage1BConfig(
            repo_root=Path.cwd(),
            dataset_root=data,
            stage0_root=stage0,
            stage1_root=stage1,
            output_root=tmp_path / "stage1b",
            candidate_config=two_candidate_config(tmp_path, checkpoint),
            sample_size=3,
            build_git_commit="stage1b-commit",
        ),
        adapter_factory=lambda item: NearMatchingEncoder()
        if item.candidate_id == "second_candidate"
        else MockEncoder(),
    )
    assert result.selected_contract["compatibility_status"] == "BLOCKED"
    assert result.summary["issues"]["by_code"]["ENCODER_CANDIDATE_AMBIGUOUS"] == 1


def test_probe_sampling_is_deterministic(tmp_path: Path) -> None:
    _, data, stage1 = fixture(tmp_path)
    first = select_probe_samples(stage1, data, sample_size=3, seed=2026)[0]
    second = select_probe_samples(stage1, data, sample_size=3, seed=2026)[0]
    assert first == second


def test_probe_sampling_covers_catalog_and_temporal_positions(tmp_path: Path) -> None:
    _, data, stage1 = fixture(tmp_path)
    samples, issues = select_probe_samples(stage1, data, sample_size=3, seed=2026)
    assert not issues
    assert [item["selection_reason"] for item in samples] == ["EARLY", "MIDDLE", "LATE"]
    assert [item["global_row"] for item in samples] == [0, 4, 8]
    assert [item["n"] for item in samples] == [1, 2, 3]


def test_probe_sampling_reports_missing_keyframe(tmp_path: Path) -> None:
    _, data, stage1 = fixture(tmp_path)
    samples, _ = select_probe_samples(stage1, data, sample_size=3, seed=2026)
    Path(samples[0]["keyframe_path"]).unlink()
    _, issues = select_probe_samples(stage1, data, sample_size=3, seed=2026)
    assert issues[0]["code"] == "IMAGE_LOAD_FAILED"
    assert issues[0]["global_row"] == samples[0]["global_row"]


@pytest.mark.parametrize(
    ("wrong", "expected"),
    [(False, "VERIFIED"), (True, "REJECTED")],
)
def test_official_candidate_synthetic_probe_decision(
    tmp_path: Path, wrong: bool, expected: str
) -> None:
    stage0, data, stage1 = fixture(tmp_path)
    asset, source, checkpoint = official_asset(tmp_path)
    output = tmp_path / "stage1b_official"
    result = run_stage1b(
        Stage1BConfig(
            repo_root=Path.cwd(),
            dataset_root=data,
            stage0_root=stage0,
            stage1_root=stage1,
            output_root=output,
            candidate_config=official_candidate_config(tmp_path, asset, source, checkpoint),
            sample_size=3,
            overwrite=True,
            build_git_commit="stage1b-official-synthetic",
        ),
        adapter_factory=lambda _: MockEncoder(wrong),
    )
    summaries = [
        json.loads(line)
        for line in (output / "probe/candidate_summaries.jsonl").read_text().splitlines()
    ]
    assert summaries[0]["decision"] == expected
    if expected == "VERIFIED":
        assert result.summary["text_smoke_status"] == "PASS"
        assert result.selected_contract["evidence_source"] == "EMPIRICAL_PROBE"
        assert result.selected_contract["asset_provenance"]["module_origin_valid"]
        assert (output / "smoke/query_artifacts/q_en_001/ranked_frames.jsonl").is_file()
    else:
        assert result.summary["text_smoke_status"] == "NOT_RUN"
        assert result.summary["readiness"]["text_retrieval"] == "BLOCKED"


def test_official_candidate_missing_asset_emits_blocked_bundle(tmp_path: Path) -> None:
    stage0, data, stage1 = fixture(tmp_path)
    asset, source, checkpoint = official_asset(tmp_path, checkpoint_exists=False)
    output = tmp_path / "stage1b_official_blocked"
    result = run_stage1b(
        Stage1BConfig(
            repo_root=Path.cwd(),
            dataset_root=data,
            stage0_root=stage0,
            stage1_root=stage1,
            output_root=output,
            candidate_config=official_candidate_config(tmp_path, asset, source, checkpoint),
            sample_size=3,
            overwrite=True,
            build_git_commit="stage1b-official-synthetic",
        ),
        adapter_factory=lambda _: pytest.fail("blocked candidate must not load adapter"),
    )
    assert result.summary["probe"]["candidates_blocked"] == 1
    assert result.selected_contract["compatibility_status"] == "BLOCKED"
    assert "ENCODER_ASSET_NOT_FOUND" in (output / "issues.jsonl").read_text()
    report = (output / "stage1b_report.md").read_text(encoding="utf-8")
    assert "## Evaluated Candidates" in report
    assert "Candidate decision: BLOCKED" in report
    assert "Provide or fix the required local asset or dependency" in report
    archive = create_stage1b_report_bundle(output, tmp_path / "blocked-report.zip")
    with ZipFile(archive) as stream:
        assert stream.namelist() == list(REPORT_MEMBERS)
        assert not any(name.endswith((".pt", ".npy", ".jpg")) for name in stream.namelist())


def test_fake_official_adapter_runs_image_and_text_towers_in_pipeline(
    tmp_path: Path,
) -> None:
    from PIL import Image

    stage0, data, stage1 = fixture(tmp_path)
    for global_row, path in enumerate(sorted((data / "keyframes").rglob("*.jpg"))):
        Image.new("RGB", (2, 2), (global_row, 0, 0)).save(path, format="PNG")
    asset, source, checkpoint = official_asset(tmp_path)
    paths = resolve_official_asset_paths(asset, source, checkpoint)
    calls = SimpleNamespace(load_path=None, image=0, text=0)

    class Model:
        def eval(self):
            return self

        def parameters(self):
            return iter([SimpleNamespace(dtype="float32")])

        def encode_image(self, batch):
            calls.image += len(batch)
            output = np.zeros((len(batch), 512), dtype=np.float32)
            for index, row in enumerate(batch.values[:, 0].astype(int)):
                output[index, row] = 1
            return FakeTensor(output)

        def encode_text(self, tokens):
            calls.text += len(tokens)
            output = np.zeros((len(tokens), 512), dtype=np.float32)
            output[:, 0] = 1
            return FakeTensor(output)

    module = ModuleType("clip")
    module._download = lambda *args, **kwargs: pytest.fail("download invoked")

    def load(path, device, jit):
        calls.load_path = path
        return Model(), lambda image: FakeTensor(
            np.asarray([image.getpixel((0, 0))[0]], dtype=np.float32)
        )

    module.load = load
    module.tokenize = lambda texts, truncate=False: FakeTensor(
        np.zeros((len(texts), 77), dtype=np.int64)
    )
    provenance = {
        "selected_device": "cpu",
        "checkpoint_sha256": "synthetic",
        "module_file": str(source / "clip/__init__.py"),
    }

    def adapter_factory(candidate):
        return OfficialOpenAIClipAdapter(candidate, paths, module, provenance)

    output = tmp_path / "stage1b_official_adapter"
    result = run_stage1b(
        Stage1BConfig(
            repo_root=Path.cwd(),
            dataset_root=data,
            stage0_root=stage0,
            stage1_root=stage1,
            output_root=output,
            candidate_config=official_candidate_config(tmp_path, asset, source, checkpoint),
            sample_size=3,
            overwrite=True,
            build_git_commit="stage1b-official-adapter",
        ),
        adapter_factory=adapter_factory,
    )
    assert result.summary["readiness"]["encoder_compatibility"] == "VERIFIED"
    assert result.summary["text_smoke_status"] == "PASS"
    assert calls.image == 3 and calls.text == 6
    assert Path(calls.load_path).is_absolute()
    runtime = json.loads((output / "encoder/runtime_adapter_manifest.json").read_text())
    assert runtime["runtime"]["preprocess_source"] == "official_clip_load_return_value"
    assert runtime["runtime"]["image_execution"]["output_dtype"] == "float32"
    assert runtime["runtime"]["text_execution"]["output_dtype"] == "float32"
    report = (output / "stage1b_report.md").read_text(encoding="utf-8")
    for heading in (
        "# Official OpenAI CLIP Candidate",
        "# Asset Provenance",
        "# Module Origin",
        "# Checkpoint Integrity",
        "# Image Preprocessing Source",
        "# Empirical Compatibility Metrics",
        "# Retrieval Alignment",
        "# Compatibility Decision",
        "# Text Smoke Status",
        "# Offline Runtime Guarantee",
        "# Non-Claims",
    ):
        assert heading in report
