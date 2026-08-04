from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from triage_eg.retrieval.stage1.benchmark import run_benchmark
from triage_eg.retrieval.stage1.builder import Stage1BuildConfig, build_index
from triage_eg.retrieval.stage1.contracts import SearchConfig
from triage_eg.retrieval.stage1.runner import search_vector


def fixture(root: Path) -> tuple[Path, Path]:
    stage0, data = root / "stage0", root / "data"
    stage0.mkdir()
    data.mkdir()
    videos = ["L01_V001", "L01_V002"]
    (stage0 / "audit_summary.json").write_text(
        json.dumps(
            {
                "audit_version": "0.1.0",
                "mode": "full",
                "videos_discovered": 2,
                "videos_completed": 2,
                "mapping_rows": 6,
                "clip_rows": 6,
                "config_fingerprint": "fp",
                "git_commit": "abc",
                "gates": {"btc_baseline": "PASS_WITH_WARNINGS"},
                "unknown_contracts": ["CLIP model compatibility"],
            }
        )
    )
    (stage0 / "run_manifest.json").write_text(json.dumps({"status": "COMPLETE"}))
    (stage0 / "contract_notes.json").write_text(
        json.dumps(
            {
                "original_frame_policy": (
                    "CSV frame_idx is authoritative; never reconstruct from pts_time*fps"
                )
            }
        )
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
            frames.append(
                {
                    "video_id": video_id,
                    "n": n,
                    "clip_row_index": n - 1,
                    "pts_time": n / 25,
                    "mapping_fps": 25.0,
                    "original_frame_idx": 7 if video_index == 0 and n < 3 else n + 10 * video_index,
                    "keyframe_relative_path": f"keyframes/{video_id}/{n:03d}.jpg",
                    "duplicate_frame_idx_group_size": 2 if video_index == 0 and n < 3 else 1,
                }
            )
    (stage0 / "btc_frame_manifest.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in frames)
    )
    (stage0 / "clip_manifest.jsonl").write_text("".join(json.dumps(item) + "\n" for item in clips))
    return stage0, data


def build_config(root: Path, stage0: Path, data: Path, **kwargs) -> Stage1BuildConfig:
    values = {
        "stage0_root": stage0,
        "dataset_root": data,
        "output_root": root / "stage1",
        "expected_rows": 6,
        "expected_videos": 2,
        "self_queries": 3,
    }
    values.update(kwargs)
    return Stage1BuildConfig(**values)


def test_api_build_search_export_benchmark_and_reuse(tmp_path: Path) -> None:
    stage0, data = fixture(tmp_path)
    built = build_index(build_config(tmp_path, stage0, data, overwrite=True))
    candidates, paths = search_vector(
        np.eye(1, 512, dtype=np.float32), SearchConfig(built.output_root, "vector", top_k=6)
    )
    assert candidates[0]["global_row"] == 0
    assert len(paths["kis_candidates"].read_text().splitlines()) == 6  # header + 5 unique pairs
    benchmark = run_benchmark(built.output_root, random_queries=2, self_queries=2, top_k=3)
    assert benchmark["queries"] == 4 and not benchmark["ground_truth_metrics_reported"]
    assert "stored_vector_load" in benchmark["query_prepare_seconds"]
    assert "p95" in benchmark["candidate_formatting_latency_seconds"]
    reused = build_index(build_config(tmp_path, stage0, data, reuse_index=True))
    assert reused.reused


def test_cli_build_vector_search_text_block_and_benchmark(tmp_path: Path) -> None:
    stage0, data = fixture(tmp_path)
    output = tmp_path / "stage1"
    build = subprocess.run(
        [
            sys.executable,
            "scripts/build_stage1_index.py",
            "--stage0-root",
            str(stage0),
            "--dataset-root",
            str(data),
            "--output-root",
            str(output),
            "--expected-rows",
            "6",
            "--expected-videos",
            "2",
            "--self-queries",
            "3",
            "--overwrite",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert build.returncode == 0, build.stderr
    query = tmp_path / "query.npy"
    np.save(query, np.eye(1, 512, dtype=np.float32))
    search = subprocess.run(
        [
            sys.executable,
            "scripts/search_stage1.py",
            "--stage1-root",
            str(output),
            "--query-vector",
            str(query),
            "--query-id",
            "cli",
            "--top-k",
            "6",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert search.returncode == 0, search.stderr
    blocked = subprocess.run(
        [
            sys.executable,
            "scripts/search_stage1.py",
            "--stage1-root",
            str(output),
            "--query-text",
            "hello",
            "--query-id",
            "blocked",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert blocked.returncode == 2 and "BLOCKED" in blocked.stderr
    benchmark = subprocess.run(
        [
            sys.executable,
            "scripts/benchmark_stage1.py",
            "--stage1-root",
            str(output),
            "--random-queries",
            "1",
            "--self-queries",
            "2",
            "--top-k",
            "3",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert benchmark.returncode == 0, benchmark.stderr
    reuse = subprocess.run(
        [
            sys.executable,
            "scripts/build_stage1_index.py",
            "--stage0-root",
            str(stage0),
            "--dataset-root",
            str(data),
            "--output-root",
            str(output),
            "--expected-rows",
            "6",
            "--expected-videos",
            "2",
            "--self-queries",
            "3",
            "--reuse-index",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert reuse.returncode == 0 and "reused: True" in reuse.stdout
