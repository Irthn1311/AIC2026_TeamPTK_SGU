from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np

from triage_eg.experiments.mb1_e1 import (
    aggregate_interval_metrics,
    build_moment_result,
    copy_benchmark_preserving_hash,
    create_mb1_e1_bundle,
    distance_to_interval,
    interval_hit,
    refine_inside_candidate_window,
    sha256_file,
)
from triage_eg.experiments.moment_m1 import DecodedFrame


def test_interval_hit_and_distance_to_interval() -> None:
    assert interval_hit(10, 10, 20)
    assert interval_hit(15, 10, 20)
    assert interval_hit(20, 10, 20)
    assert not interval_hit(9, 10, 20)
    assert distance_to_interval(15, 10, 20) == 0
    assert distance_to_interval(7, 10, 20) == 3
    assert distance_to_interval(24, 10, 20) == 4


def test_preferred_frame_is_secondary_and_m0_is_exact_manifest_anchor() -> None:
    annotation = {
        "moment_id": "mb1_001",
        "acceptable_start_frame": 100,
        "acceptable_end_frame": 120,
        "preferred_frame": 119,
        "annotation_confidence": "HIGH",
        "moment_type": "CONTACT",
    }
    candidate = {
        "source_anchor_frame": 101,
        "window_start_frame": 90,
        "window_end_frame": 130,
    }
    search = {"m0_score": 0.2, "m1_frame": 118, "m1_score": 0.3}
    row = build_moment_result(annotation, candidate, search)
    assert row["m0_frame"] == 101
    assert row["m0_distance_to_interval"] == 0
    assert row["m0_preferred_frame_error"] == 18
    metrics = aggregate_interval_metrics([row])
    assert metrics["M0_SOURCE_ANCHOR_FRAME"]["INTERVAL_HIT_RATE"] == 1
    assert metrics["M0_SOURCE_ANCHOR_FRAME"]["preferred_frame_metric_role"] == (
        "SECONDARY_DIAGNOSTIC_ONLY"
    )


class _FakeDecoder:
    info = SimpleNamespace(fps=25.0, total_frames=100)

    def decode_indices(self, frame_indices: list[int]) -> list[DecodedFrame]:
        return [
            DecodedFrame("L01_V001", value, np.full((2, 2, 3), value, dtype=np.uint8))
            for value in frame_indices
        ]


class _FakeImageEncoder:
    def encode(self, frames: list[DecodedFrame]) -> np.ndarray:
        matrix = np.zeros((len(frames), 512), dtype=np.float32)
        for row, frame in enumerate(frames):
            score = 0.1 + frame.actual_frame_idx / 100.0
            score = min(score, 0.99)
            matrix[row, 0] = score
            matrix[row, 1] = np.sqrt(1.0 - score**2)
        return matrix


def test_m1_search_stays_inside_exact_candidate_window() -> None:
    text = np.zeros(512, dtype=np.float32)
    text[0] = 1.0
    result, images = refine_inside_candidate_window(
        decoder=_FakeDecoder(),
        image_encoder=_FakeImageEncoder(),
        text_embedding=text,
        window_start=10,
        window_end=30,
        source_anchor_frame=20,
    )
    assert result["local_window_start"] == 10
    assert result["local_window_end"] == 30
    assert 10 <= result["m1_frame"] <= 30
    assert min(images) >= 10 and max(images) <= 30


def test_benchmark_copy_preserves_hash(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    target = tmp_path / "nested" / "copy.jsonl"
    source.write_bytes(b'{"moment_id":"mb1_001"}\n')
    declared = copy_benchmark_preserving_hash(source, target)
    assert declared == sha256_file(source) == sha256_file(target)
    assert target.read_bytes() == source.read_bytes()


def test_bundle_excludes_heavy_assets(tmp_path: Path) -> None:
    output = tmp_path / "output"
    (output / "benchmark").mkdir(parents=True)
    (output / "visuals").mkdir()
    for name in (
        "mb1_e1_summary.json",
        "mb1_e1_metrics.json",
        "run_manifest.json",
        "visuals/review_key.json",
    ):
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")
    for name in ("moment_results.jsonl", "issues.jsonl"):
        (output / name).write_text("", encoding="utf-8")
    annotation = output / "benchmark/mb1_ai_semantic_moments.jsonl"
    annotation.write_text(json.dumps({"moment_id": "mb1_001"}) + "\n", encoding="utf-8")
    (output / "benchmark/mb1_ai_semantic_moments.sha256").write_text(
        sha256_file(annotation), encoding="utf-8"
    )
    (output / "model.pt").write_bytes(b"heavy")
    bundle = create_mb1_e1_bundle(output, tmp_path / "bundle.zip")
    with ZipFile(bundle) as archive:
        names = set(archive.namelist())
    assert "model.pt" not in names
    assert "benchmark/mb1_ai_semantic_moments.jsonl" in names
