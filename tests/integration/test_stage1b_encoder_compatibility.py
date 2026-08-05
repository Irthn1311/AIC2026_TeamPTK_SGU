from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from triage_eg.retrieval.stage1.builder import Stage1BuildConfig, build_index
from triage_eg.retrieval.stage1b import Stage1BConfig, run_stage1b
from triage_eg.retrieval.stage1b.sampling import select_probe_samples
from triage_eg.retrieval.stage1b.writers import REPORT_MEMBERS, create_stage1b_report_bundle


def fixture(root: Path) -> tuple[Path, Path, Path]:
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
            matrix[row, video_index * 3 + row] = 1
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
        "stage1b_version: '0.1.0'\n"
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


class MockEncoder:
    def __init__(self, wrong: bool = False) -> None:
        self.wrong = wrong

    def encode_images(self, paths: list[Path]) -> np.ndarray:
        output = np.zeros((len(paths), 512), dtype=np.float32)
        for index, path in enumerate(paths):
            video = int(path.parent.name[-3:]) - 1
            n = int(path.stem) - 1
            row = video * 3 + n
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


def run_case(tmp_path: Path, *, wrong: bool = False, enabled: bool = True):
    stage0, data, stage1 = fixture(tmp_path)
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
        adapter_factory=lambda candidate: MockEncoder(wrong),
    )
    return result


def test_matching_encoder_verifies_and_enables_smoke(tmp_path: Path) -> None:
    result = run_case(tmp_path)
    assert result.summary["readiness"] == {
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
    archive = create_stage1b_report_bundle(result.output_root, tmp_path / "report.zip")
    with ZipFile(archive) as stream:
        assert stream.namelist() == list(REPORT_MEMBERS)
        assert not any(name.endswith((".bin", ".npy", ".jpg")) for name in stream.namelist())


def test_wrong_encoder_is_rejected_and_text_stays_blocked(tmp_path: Path) -> None:
    result = run_case(tmp_path, wrong=True)
    assert result.summary["probe"]["candidates_rejected"] == 1
    assert result.summary["readiness"]["encoder_compatibility"] == "BLOCKED"
    assert result.summary["text_smoke_status"] == "NOT_RUN"


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
