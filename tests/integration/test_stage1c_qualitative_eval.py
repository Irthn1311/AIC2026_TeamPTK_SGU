from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest
from PIL import Image

from triage_eg.retrieval.stage1c import (
    Stage1CConfig,
    create_stage1c_bundle,
    run_stage1c,
    score_human_review,
)


class MockVerifiedTextEncoder:
    def encode_text(self, texts: list[str]) -> np.ndarray:
        output = np.zeros((len(texts), 512), dtype=np.float32)
        for index in range(len(texts)):
            output[index, index % 2] = 4.0
        return output


def synthetic_inputs(root: Path) -> tuple[Path, Path, Path, Path, Path]:
    stage0, dataset, stage1, stage1b = (
        root / "stage0",
        root / "dataset",
        root / "stage1",
        root / "stage1b",
    )
    for path in (stage0, dataset, stage1 / "index", stage1b / "encoder"):
        path.mkdir(parents=True)
    (stage0 / "audit_summary.json").write_text('{"status":"COMPLETE"}\n')
    (stage0 / "run_manifest.json").write_text('{"status":"COMPLETE"}\n')

    count, dimension = 120, 512
    vectors = np.zeros((count, dimension), dtype=np.float16)
    for row in range(count):
        vectors[row, row % dimension] = 1
    vectors[1] = vectors[0]
    norms = np.linalg.norm(vectors.astype(np.float32), axis=1).astype(np.float32)
    video_ids = [f"L01_V{index + 1:03d}" for index in range(12)]
    video_table = []
    video_indices, frame_n, originals = [], [], []
    pts, fps, duplicate_sizes = [], [], []
    for video_index, video_id in enumerate(video_ids):
        prefix = f"keyframes/{video_id}"
        video_table.append(
            {"video_index": video_index, "video_id": video_id, "keyframe_prefix": prefix}
        )
        for local in range(10):
            n = local + 1
            original = 0 if n == 1 else video_index * 100 + n * 3
            video_indices.append(video_index)
            frame_n.append(n)
            originals.append(original)
            pts.append(local / 25)
            fps.append(25)
            duplicate_sizes.append(1)
            target = dataset / prefix / f"{n:03d}.jpg"
            target.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (48, 32), (video_index * 10, local * 10, 30)).save(target)
    index = stage1 / "index"
    np.save(index / "clip_vectors.f16.npy", vectors)
    np.save(index / "vector_norms.f32.npy", norms)
    np.save(index / "frame_video_index.npy", np.asarray(video_indices, dtype=np.int32))
    np.save(index / "frame_n.npy", np.asarray(frame_n, dtype=np.int32))
    np.save(index / "frame_original_idx.npy", np.asarray(originals, dtype=np.int64))
    np.save(index / "frame_pts_time.npy", np.asarray(pts, dtype=np.float64))
    np.save(index / "frame_mapping_fps.npy", np.asarray(fps, dtype=np.float32))
    np.save(index / "duplicate_group_size.npy", np.asarray(duplicate_sizes, dtype=np.int16))
    (index / "video_table.json").write_text(json.dumps(video_table))
    (index / "index_manifest.json").write_text(
        json.dumps(
            {
                "status": "COMPLETE",
                "vector_count": count,
                "dimension": dimension,
                "build_git_commit": "stage1-synthetic",
            }
        )
    )
    fingerprint = "synthetic-index-fingerprint"
    (stage1 / "stage1_summary.json").write_text(
        json.dumps(
            {
                "index_status": "COMPLETE",
                "index_fingerprint": fingerprint,
                "next_stage_readiness": {"corpus_index": "READY_WITH_TIE_WARNINGS"},
            }
        )
    )
    (stage1 / "run_manifest.json").write_text(
        json.dumps({"status": "COMPLETE", "index_fingerprint": fingerprint})
    )
    checkpoint = root / "checkpoint.bin"
    checkpoint.write_bytes(b"checkpoint")
    checkpoint_sha = hashlib.sha256(b"checkpoint").hexdigest()
    selected = {
        "candidate_id": "mock_verified",
        "enabled": True,
        "implementation": "mock",
        "architecture": "ViT-B/32",
        "pretrained": "synthetic",
        "checkpoint_path": str(checkpoint),
        "output_dimension": 512,
        "tokenizer": "mock",
        "context_length": 77,
        "image_preprocessing": {},
        "text_preprocessing": {},
        "runtime_dtype": "float32",
        "evidence_source": "EMPIRICAL_PROBE",
        "compatibility_status": "VERIFIED",
        "checkpoint_sha256": checkpoint_sha,
        "selected_candidate_id": "mock_verified",
    }
    (stage1b / "encoder/selected_encoder_contract.json").write_text(json.dumps(selected))
    (stage1b / "encoder/runtime_adapter_manifest.json").write_text(
        json.dumps(
            {
                "selected_candidate_id": "mock_verified",
                "model_space_status": "MODEL_SPACE_VERIFIED",
                "adapter": "mock",
            }
        )
    )
    (stage1b / "stage1b_summary.json").write_text(
        json.dumps(
            {
                "evaluation_status": "COMPLETE",
                "stage1_index_fingerprint": fingerprint,
                "readiness": {
                    "encoder_readiness": "VERIFIED",
                    "text_retrieval": "READY_FOR_QUALITATIVE_TESTING",
                },
            }
        )
    )
    (stage1b / "run_manifest.json").write_text('{"status":"COMPLETE"}\n')
    suite = root / "queries.jsonl"
    suite.write_text(
        json.dumps(
            {
                "query_id": "obj_en",
                "pair_id": "obj",
                "language": "en",
                "category": "OBJECT",
                "difficulty": "EASY",
                "text": "object english",
                "notes": "",
            }
        )
        + "\n"
        + json.dumps(
            {
                "query_id": "obj_vi",
                "pair_id": "obj",
                "language": "vi",
                "category": "OBJECT",
                "difficulty": "EASY",
                "text": "đồ vật tiếng Việt",
                "notes": "",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return stage0, dataset, stage1, stage1b, suite


def config(root: Path, *, overwrite: bool = True) -> Stage1CConfig:
    stage0, dataset, stage1, stage1b, suite = synthetic_inputs(root)
    return Stage1CConfig(
        repo_root=Path.cwd(),
        dataset_root=dataset,
        stage0_root=stage0,
        stage1_root=stage1,
        stage1b_root=stage1b,
        encoder_asset_root=root / "unused-asset",
        query_suite=suite,
        output_root=root / "stage1c",
        frame_top_k=50,
        kis_top_k=100,
        review_top_k=10,
        contact_sheet_top_k=20,
        overwrite=overwrite,
        build_git_commit="stage1c-synthetic",
    )


def test_synthetic_stage1c_pipeline_review_and_bundle(tmp_path: Path) -> None:
    stage_config = config(tmp_path)
    stage0_mtime = (stage_config.stage0_root / "run_manifest.json").stat().st_mtime_ns
    stage1_mtime = (stage_config.stage1_root / "stage1_summary.json").stat().st_mtime_ns
    result = run_stage1c(
        stage_config,
        adapter_factory=lambda candidate: MockVerifiedTextEncoder(),
    )
    assert result.summary["retrieval_quality_status"] == "NOT_REVIEWED"
    assert result.summary["retrieval"]["queries_completed"] == 2
    assert result.summary["paired_language_diagnostics"]["pairs_completed"] == 1
    assert result.summary["human_review"]["judgments_expected"] == 20
    assert result.summary["stage1b_encoder"]["compatibility_status"] == "VERIFIED"
    assert (stage_config.stage0_root / "run_manifest.json").stat().st_mtime_ns == stage0_mtime
    assert (stage_config.stage1_root / "stage1_summary.json").stat().st_mtime_ns == stage1_mtime

    query_root = result.output_root / "queries/obj_en"
    frames = [
        json.loads(line)
        for line in (query_root / "ranked_frames.jsonl").read_text().splitlines()
    ]
    videos = [
        json.loads(line)
        for line in (query_root / "ranked_videos.jsonl").read_text().splitlines()
    ]
    assert len(frames) == 50
    assert frames[0]["global_row"] == 0
    assert frames[0]["original_frame_idx"] == 0
    assert frames[0]["n"] == 1
    assert videos[0]["best_frame_rank"] == 1
    assert videos[0]["frames_in_raw_top50"] >= 1
    with (query_root / "kis_candidates.csv").open(newline="") as stream:
        kis = list(csv.DictReader(stream))
    assert len(kis) == 100
    assert kis[0] == {"video_id": "L01_V001", "frame_id": "0"}
    assert kis[1]["frame_id"] != str(frames[1]["n"])
    diagnostics = json.loads((query_root / "retrieval_diagnostics.json").read_text())
    assert diagnostics["unique_exact_vectors_top5"] == 4
    assert diagnostics["exact_duplicate_rows_top20"] == 1
    assert (query_root / "contact_sheet_top20.jpg").is_file()
    assert (query_root / "contact_sheet_top12_videos.jpg").is_file()
    query_artifact = json.loads((query_root / "query.json").read_text())
    assert query_artifact["encoding"]["embedding_dimension"] == 512
    assert query_artifact["encoding"]["embedding_finite"]
    assert query_artifact["encoding"]["tokenization_status"] == "SUCCESS"

    report = (result.output_root / "stage1c_report.md").read_text(encoding="utf-8")
    for section in (
        "## Provenance",
        "## Stage 1A Index",
        "## Stage 1B Verified Encoder",
        "## Query Suite",
        "## Retrieval Execution",
        "## Structural Diagnostics",
        "## English/Vietnamese Pair Diagnostics",
        "## Human Review Status",
        "## Query-by-Query Summary",
        "## Non-Claims",
        "## Next Decision Gate",
    ):
        assert section in report
    assert "RETRIEVAL_QUALITY_STATUS = NOT_REVIEWED" in report
    manifest = json.loads((result.output_root / "run_manifest.json").read_text())
    assert manifest["no_stage0_rerun"]
    assert manifest["no_stage1_rebuild"]
    assert manifest["stage1b_compatibility_logic_unchanged"]
    assert manifest["no_model_download"]
    assert manifest["no_translation"]
    assert manifest["no_reranking"]

    archive = create_stage1c_bundle(result.output_root, tmp_path / "stage1c_bundle.zip")
    with ZipFile(archive) as stream:
        names = stream.namelist()
    assert "queries/obj_en/ranked_frames.jsonl" in names
    assert "queries/obj_en/ranked_videos.jsonl" in names
    assert "queries/obj_en/contact_sheet_top20.jpg" in names
    assert "review/review_template.csv" in names
    assert not any(name.endswith((".npy", ".pt", ".bin", ".mp4")) for name in names)
    assert not any(name.startswith("logs/") for name in names)

    template = result.output_root / "review/review_template.csv"
    filled = tmp_path / "filled.csv"
    with template.open(encoding="utf-8-sig", newline="") as stream:
        review_rows = list(csv.DictReader(stream))
    for index, row in enumerate(review_rows):
        row["review_label"] = ("RELEVANT", "PARTIAL", "IRRELEVANT", "UNCERTAIN")[
            index % 4
        ]
    with filled.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(review_rows[0]))
        writer.writeheader()
        writer.writerows(review_rows)
    review = score_human_review(result.output_root, filled)
    assert review["human_review_status"] == "COMPLETE"
    assert review["retrieval_quality_status"] == "QUALITATIVELY_EVALUATED"
    assert (result.output_root / "review/review_metrics.json").is_file()
    assert (result.output_root / "review/review_metrics.md").is_file()

    reused = run_stage1c(
        replace(stage_config, overwrite=False, reuse_results=True),
        adapter_factory=lambda candidate: pytest.fail("reuse must not reload encoder"),
    )
    assert reused.reused


def test_reuse_rejects_changed_retrieval_limits(tmp_path: Path) -> None:
    stage_config = config(tmp_path)
    run_stage1c(stage_config, adapter_factory=lambda candidate: MockVerifiedTextEncoder())
    changed = replace(
        stage_config,
        frame_top_k=40,
        review_top_k=5,
        overwrite=False,
        reuse_results=True,
    )
    with pytest.raises(ValueError, match="does not match"):
        run_stage1c(changed, adapter_factory=lambda candidate: MockVerifiedTextEncoder())


def test_unverified_stage1b_blocks_stage1c(tmp_path: Path) -> None:
    stage_config = config(tmp_path)
    selected_path = stage_config.stage1b_root / "encoder/selected_encoder_contract.json"
    selected = json.loads(selected_path.read_text())
    selected["compatibility_status"] = "REJECTED"
    selected_path.write_text(json.dumps(selected))
    with pytest.raises(ValueError, match="STAGE1B_ENCODER_NOT_VERIFIED"):
        run_stage1c(stage_config, adapter_factory=lambda candidate: MockVerifiedTextEncoder())


def test_stage1c_source_has_no_optimization_cli_or_pipeline_features() -> None:
    cli = Path("scripts/run_stage1c_qualitative_eval.py").read_text(encoding="utf-8")
    for forbidden_flag in (
        "--translate",
        "--query-expand",
        "--rerank",
        "--multilingual-model",
    ):
        assert forbidden_flag not in cli
    sources = "\n".join(
        path.read_text(encoding="utf-8")
        for path in Path("src/triage_eg/retrieval/stage1c").glob("*.py")
    )
    for forbidden_call in (
        "build_index(",
        "run_stage1b(",
        "run_audit(",
        "VideoCapture(",
        "OpenClipMultimodalEncoder(",
    ):
        assert forbidden_call not in sources
