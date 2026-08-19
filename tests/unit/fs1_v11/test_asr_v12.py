from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from triage_eg.fs1_v11.asr_v12 import (
    aggregate_worker_reports,
    atomic_write_jsonl,
    initialize_cuda_worker,
    lpt_partition,
    material_consistency,
    merge_shards,
    normalize_segments,
    probe_audio,
    representative_sample,
    select_lowest_rtf,
    timestamps_monotonic,
)


def inventory(count: int = 20) -> list[dict]:
    return [
        {
            "video_id": f"V{index:03d}",
            "has_audio": True,
            "probe_status": "PASS",
            "duration_seconds": float(index + 1),
        }
        for index in range(count)
    ]


def test_probe_audio_uses_ffprobe_and_format_duration(monkeypatch, tmp_path: Path) -> None:
    def fake_run(command, **kwargs):
        assert command[0] == "ffprobe" and any("format=duration" in item for item in command)
        return subprocess.CompletedProcess(
            command,
            0,
            stdout='{"streams":[{"codec_name":"aac","sample_rate":"48000"}],"format":{"duration":"12.5"}}',
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    row = probe_audio("V1", tmp_path / "video.mp4")
    assert row["probe_status"] == "PASS" and row["duration_seconds"] == 12.5


def test_lpt_partition_balances_by_duration_not_count() -> None:
    shards = lpt_partition(inventory(20), 2)
    assert sum(shard["video_count"] for shard in shards) == 20
    assert shards[0]["total_audio_seconds"] == shards[1]["total_audio_seconds"]
    assert shards[0]["imbalance_percent"] == 0


def test_representative_sample_spans_duration_range() -> None:
    rows = representative_sample(inventory(100), 20)
    assert len(rows) == 20
    assert rows[0]["duration_seconds"] == 1
    assert rows[-1]["duration_seconds"] == 100


def test_timestamp_recovery_uses_next_start_and_audio_end() -> None:
    chunks = [
        {"timestamp": (0.0, None), "text": "a"},
        {"timestamp": (2.0, None), "text": "b"},
    ]
    segments, recovered = normalize_segments(chunks, 5.0)
    assert recovered == 2
    assert [(row["start_seconds"], row["end_seconds"]) for row in segments] == [
        (0.0, 2.0),
        (2.0, 5.0),
    ]
    assert timestamps_monotonic(segments)


def test_material_consistency_is_token_jaccard() -> None:
    left = {"segments": [{"normalized_text": "một xe đỏ"}]}
    right = {"segments": [{"normalized_text": "xe màu đỏ"}]}
    assert material_consistency(left, right) == pytest.approx(0.5)


def worker_report(batch: int, wall: float, empty: int = 0) -> dict:
    return {
        "shard_id": 0,
        "batch_size": batch,
        "video_count": 20,
        "completed_count": 20,
        "failed_count": 0,
        "nonempty_count": 20 - empty,
        "timestamp_valid_count": 20,
        "total_audio_seconds": 1000.0,
        "wall_seconds": wall,
        "ffmpeg_wall_seconds": 10.0,
        "whisper_wall_seconds": wall - 10,
        "peak_allocated_vram_bytes": 1,
        "peak_reserved_vram_bytes": 2,
        "device_name": "T4",
    }


def test_selection_uses_lowest_stable_end_to_end_rtf() -> None:
    reports = [
        aggregate_worker_reports([worker_report(4, 100)]),
        aggregate_worker_reports([worker_report(8, 70)]),
        aggregate_worker_reports([worker_report(16, 60, empty=1)]),
    ]
    selected = select_lowest_rtf(reports)
    assert selected["batch_size"] == 8


def test_merge_rejects_duplicate_video_ownership(tmp_path: Path) -> None:
    rows = inventory(2)
    first, second = tmp_path / "first.jsonl", tmp_path / "second.jsonl"
    atomic_write_jsonl(first, [{"video_id": "V000", "status": "PASS"}])
    atomic_write_jsonl(
        second,
        [
            {"video_id": "V000", "status": "PASS"},
            {"video_id": "V001", "status": "PASS"},
        ],
    )
    with pytest.raises(RuntimeError, match="ASR_SHARD_MERGE_GATE_FAILED"):
        merge_shards(rows, [first, second])


def test_cuda_worker_avoids_early_peak_reset_on_fresh_process() -> None:
    calls = []

    class FakeCuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def device_count():
            return 1

        @staticmethod
        def get_device_name(index):
            calls.append(("get_device_name", index))
            return "Tesla T4"

    class FakeTorch:
        cuda = FakeCuda()

    assert initialize_cuda_worker(FakeTorch()) == "Tesla T4"
    assert calls == [("get_device_name", 0)]
