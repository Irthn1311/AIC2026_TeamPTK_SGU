from __future__ import annotations

import inspect
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from triage_eg.experiments.mb1 import (
    MB1Settings,
    clipped_window,
    create_mb1_bundle,
    displayed_frame_indices,
    prepare_mb1_candidates,
    select_mb1_sources,
    validate_annotation,
)
from triage_eg.experiments.mb1 import benchmark as mb1
from triage_eg.experiments.moment_m1 import DecodedFrame, VideoInfo
from triage_eg.experiments.reference_rt2 import (
    BENCHMARK_TYPE,
    RT2BenchmarkQuery,
    RT2ReferenceEvent,
)


def query(number: int) -> RT2BenchmarkQuery:
    video_id = f"L01_V{number:03d}"
    return RT2BenchmarkQuery(
        query_id=f"rt2_{number:03d}",
        benchmark_type=BENCHMARK_TYPE,
        source_video_id=video_id,
        language="en",
        events=(
            RT2ReferenceEvent("E1", "unused source description", "S01", 0, 0, 1, 20),
            RT2ReferenceEvent("E2", "unused source description", "S02", 1, 1, 2, 80),
        ),
        difficulty_tags=("MULTI_EVENT",),
        generator="GPT-5.6 Sol",
        human_reviewed=False,
    )


def test_selection_is_preferred_then_seeded_and_deterministic() -> None:
    queries = [query(index) for index in range(1, 15)]
    preferred = tuple(item.source_video_id for item in queries[:12])
    settings = MB1Settings(
        selected_video_count=12,
        max_candidate_windows=24,
        preferred_video_ids=preferred,
    )
    available = {item.source_video_id for item in queries}
    first = select_mb1_sources(queries, available, settings)
    second = select_mb1_sources(list(reversed(queries)), available, settings)
    assert [item.source_video_id for item in first] == list(preferred)
    assert [item.source_video_id for item in second] == list(preferred)


def test_displayed_frames_are_valid_chronological_and_include_anchor() -> None:
    values = displayed_frame_indices(10, 130, 67, target_count=32)
    assert values == sorted(set(values))
    assert values[0] == 10 and values[-1] == 130 and 67 in values
    assert len(values) == 32
    assert max(right - left for left, right in zip(values, values[1:], strict=False)) <= 5


def test_window_clipping_preserves_requested_span_at_video_edges() -> None:
    assert clipped_window(2, fps=10.0, total_frames=100, seconds=4.0) == (0, 40)
    assert clipped_window(98, fps=10.0, total_frames=100, seconds=4.0) == (59, 99)


class FakeDecoder:
    info = VideoInfo(fps=10.0, total_frames=100)

    def __init__(self, video_id: str, video_path: Path) -> None:
        self.video_id = video_id
        assert video_path.is_file()

    def decode_indices(self, frame_indices: list[int]) -> list[DecodedFrame]:
        return [
            DecodedFrame(
                self.video_id,
                index,
                np.full((48, 64, 3), index % 255, dtype=np.uint8),
            )
            for index in frame_indices
        ]

    def close(self) -> None:
        return None


def _write_inputs(tmp_path: Path) -> tuple[Path, Path, tuple[str, ...]]:
    dataset = tmp_path / "dataset"
    queries = [query(index) for index in range(1, 13)]
    preferred = tuple(item.source_video_id for item in queries)
    for item in queries:
        path = dataset / "Videos_L01" / "video" / f"{item.source_video_id}.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fake")
    benchmark = tmp_path / "rt2_ai_benchmark.jsonl"
    benchmark.write_text(
        "".join(json.dumps(item.as_dict()) + "\n" for item in queries), encoding="utf-8"
    )
    return dataset, benchmark, preferred


def test_candidate_pack_manifest_sheet_consistency_and_heavy_exclusion(tmp_path: Path) -> None:
    dataset, benchmark, preferred = _write_inputs(tmp_path)
    output = tmp_path / "mb1"
    settings = MB1Settings(
        selected_video_count=12,
        max_candidate_windows=24,
        preferred_video_ids=preferred,
    )
    result = prepare_mb1_candidates(
        dataset,
        benchmark,
        output,
        settings=settings,
        build_git_commit="test",
        decoder_factory=FakeDecoder,
    )
    assert result["candidate_window_count"] == 24
    rows = [
        json.loads(line)
        for line in (output / "mb1_candidate_manifest.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 24
    assert all(row["displayed_frames"] == sorted(row["displayed_frames"]) for row in rows)
    assert all(
        row["window_start_frame"] <= frame <= row["window_end_frame"]
        for row in rows
        for frame in row["displayed_frames"]
    )
    assert all((output / path).is_file() for row in rows for path in row["image_sheet_paths"])
    assert all(len(row["image_sheet_paths"]) == 2 for row in rows)

    (output / "raw_video.mp4").write_bytes(b"heavy")
    (output / "vectors.npy").write_bytes(b"heavy")
    archive = create_mb1_bundle(output, tmp_path / "mb1.zip")
    with ZipFile(archive) as stream:
        names = stream.namelist()
    assert "mb1_candidate_manifest.jsonl" in names
    assert "annotation_schema.json" in names
    assert len([name for name in names if name.endswith(".jpg")]) == 48
    assert not any(name.endswith((".mp4", ".npy", ".bin", ".pt")) for name in names)


def test_annotation_interval_contract() -> None:
    value = {
        "moment_id": "mb1_001",
        "video_id": "L01_V001",
        "source_candidate_id": "mb1_c001",
        "query_text": "object begins being cut",
        "moment_definition": "first visible cutting contact",
        "moment_type": "TRANSITION_ONSET",
        "acceptable_start_frame": 10,
        "acceptable_end_frame": 14,
        "preferred_frame": 12,
        "annotation_confidence": "HIGH",
        "generator": "GPT-5.6 Sol",
        "human_reviewed": False,
    }
    validate_annotation(value)
    without_preferred = dict(value)
    without_preferred.pop("preferred_frame")
    validate_annotation(without_preferred)
    value["preferred_frame"] = 15
    try:
        validate_annotation(value)
    except ValueError as error:
        assert "preferred_frame" in str(error)
    else:
        raise AssertionError("out-of-interval preferred frame was accepted")


def test_mb1_source_has_no_model_inference_or_network_downloads() -> None:
    source = Path(mb1.__file__).read_text(encoding="utf-8")
    for forbidden in (
        "requests.get",
        "urlopen(",
        "hf_hub_download",
        "snapshot_download",
        "encode_images",
        "encode_text",
        "OperationalRetrievalRuntime",
    ):
        assert forbidden not in source
    parameters = inspect.signature(prepare_mb1_candidates).parameters
    assert "stage1_root" not in parameters and "model_root" not in parameters
