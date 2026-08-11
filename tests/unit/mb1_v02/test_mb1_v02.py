from __future__ import annotations

import inspect
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from triage_eg.experiments.mb1_v02 import runner, signals
from triage_eg.experiments.mb1_v02.runner import (
    BUNDLE_FILES,
    MB1V02Config,
    annotation_schema_v02,
    build_source_video_pool,
    create_mb1_v02_bundle,
    prepare_mb1_v02_candidates,
    render_contact_sheet,
)
from triage_eg.experiments.mb1_v02.signals import (
    CandidateProposal,
    ScanSeries,
    continuous_shot_segments,
    dense_displayed_frames,
    detect_hard_cuts,
    overview_displayed_frames,
    propose_active_windows,
    scan_low_resolution_video,
    window_contains_detected_cut,
)
from triage_eg.experiments.moment_m1 import DecodedFrame, VideoInfo


def _series(peaks: tuple[int, ...] = ()) -> ScanSeries:
    indices = np.arange(0, 1001, 2, dtype=np.int64)
    pixel = np.full(len(indices), 0.01, dtype=np.float64)
    for frame in peaks:
        pixel[frame // 2] = 0.20
    return ScanSeries(indices, pixel, pixel.copy(), 2, 0.0, 0.0)


def _proposal() -> CandidateProposal:
    return CandidateProposal(
        video_id="L01_V001",
        fps=30.0,
        window_start_frame=100,
        window_end_frame=220,
        proposal_center_frame=160,
        shot_start_frame=50,
        shot_end_frame=270,
        scan_stride_frames=3,
        activity_mean=0.02,
        activity_std=0.01,
        activity_peak=0.2,
        proposal_activity_score=0.075,
        minimum_distance_to_detected_cut_seconds=5.0,
    )


def test_source_pool_is_primary_first_secondary_eligible_and_deterministic() -> None:
    manifest = [
        {"candidate_id": "c2", "video_id": "L01_V002"},
        {"candidate_id": "c1", "video_id": "L01_V001"},
    ]
    qc = [
        {"candidate_id": "c2", "video_id": "L01_V002", "usability": "REJECT"},
        {"candidate_id": "c1", "video_id": "L01_V001", "usability": "USABLE"},
    ]
    rt2 = [
        {"source_video_id": "L01_V004", "difficulty_tags": ["SPORTS"]},
        {"source_video_id": "L01_V003", "difficulty_tags": ["MULTI_EVENT"]},
        {"source_video_id": "L01_V001", "difficulty_tags": ["PROCEDURAL"]},
    ]
    expected = [
        {"video_id": "L01_V001", "source_pool_origin": "MB1_V01_USABLE_VIDEO"},
        {
            "video_id": "L01_V004",
            "source_pool_origin": "RT2_EXISTING_TAGGED_VIDEO",
            "existing_tags": ["SPORTS"],
        },
    ]
    assert build_source_video_pool(manifest, qc, rt2) == expected
    assert build_source_video_pool(manifest[::-1], qc[::-1], rt2[::-1]) == expected


class _ScanDecoder:
    info = VideoInfo(fps=10.0, total_frames=6)

    def decode_indices(self, frame_indices: list[int]) -> list[DecodedFrame]:
        return [
            DecodedFrame("L01_V001", value, np.full((4, 4, 3), value, dtype=np.uint8))
            for value in frame_indices
        ]


def test_low_resolution_scan_is_chronological_and_preserves_actual_ids(monkeypatch) -> None:
    def compact(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        value = int(image[0, 0, 0])
        histogram = np.zeros(32, dtype=np.float64)
        histogram[value] = 1.0
        return np.full((2, 2), value, dtype=np.uint8), histogram

    monkeypatch.setattr(signals, "_small_gray_and_histogram", compact)
    result = scan_low_resolution_video(_ScanDecoder())
    assert result.frame_indices.tolist() == [0, 2, 4, 5]
    assert np.all(np.diff(result.frame_indices) > 0)
    assert result.scan_stride_frames == 2


def test_adaptive_dual_signal_cut_detection_requires_both_signals() -> None:
    series = _series()
    series.pixel_differences[100] = 0.8
    series.histogram_differences[100] = 0.8
    series.pixel_differences[200] = 0.8
    cuts, rule = detect_hard_cuts(series)
    assert cuts == (200,)
    assert rule["decision"] == "pixel > threshold AND histogram > threshold"


def test_continuous_shots_cover_video_without_overlap_or_gaps() -> None:
    segments = continuous_shot_segments(100, (25, 70))
    assert [(item.start_frame, item.end_frame) for item in segments] == [
        (0, 24),
        (25, 69),
        (70, 99),
    ]


def test_retained_windows_stay_in_one_shot_and_do_not_cross_cuts() -> None:
    series = _series((120, 300, 480, 660, 840))
    cuts = (400,)
    shots = continuous_shot_segments(1001, cuts)
    proposals, _ = propose_active_windows("L01_V001", 10.0, 1001, series, cuts, shots)
    assert proposals
    assert all(
        not window_contains_detected_cut(p.window_start_frame, p.window_end_frame, cuts)
        for p in proposals
    )
    assert all(
        p.shot_start_frame
        <= p.window_start_frame
        <= p.window_end_frame
        <= p.shot_end_frame
        for p in proposals
    )


def test_temporal_nms_enforces_six_second_center_separation() -> None:
    series = _series(tuple(range(100, 901, 40)))
    proposals, counts = propose_active_windows(
        "L01_V001", 10.0, 1001, series, (), continuous_shot_segments(1001, ())
    )
    centers = sorted(item.proposal_center_frame for item in proposals)
    assert all(right - left >= 60 for left, right in zip(centers, centers[1:], strict=False))
    assert counts["rejected_by_temporal_nms"] > 0


def test_per_video_candidate_cap_is_four_and_rejections_are_counted() -> None:
    series = _series((100, 200, 300, 400, 500, 600, 700, 800, 900))
    proposals, counts = propose_active_windows(
        "L01_V001", 10.0, 1001, series, (), continuous_shot_segments(1001, ())
    )
    assert len(proposals) == 4
    assert counts["rejected_by_per_video_cap"] > 0


def test_overview_and_dense_evidence_have_exact_raw_bounds() -> None:
    proposal = _proposal()
    overview = overview_displayed_frames(proposal)
    dense = dense_displayed_frames(proposal)
    assert len(overview) == len(dense) == 32
    assert overview[0] == 100 and overview[-1] == 220
    assert min(dense) >= 130 and max(dense) <= 190


def test_rendered_sheet_returns_exact_actual_frame_labels(tmp_path: Path) -> None:
    frames = [
        DecodedFrame("L01_V001", value, np.full((24, 32, 3), value, dtype=np.uint8))
        for value in (10, 20, 30)
    ]
    target = tmp_path / "sheet.jpg"
    labels = render_contact_sheet(
        target,
        candidate_id="mb1v02_c001",
        video_id="L01_V001",
        view_name="OVERVIEW",
        fps=10.0,
        frames=frames,
        quality=80,
    )
    assert labels == [10, 20, 30]
    assert target.is_file() and target.stat().st_size > 0


def test_annotation_schema_has_required_moment_contract() -> None:
    schema = annotation_schema_v02()
    required = set(schema["required"])
    assert {
        "moment_family",
        "moment_type",
        "acceptable_start_frame",
        "acceptable_end_frame",
    } <= required
    assert schema["properties"]["generator"] == {"const": "GPT-5.6 Sol"}
    assert schema["properties"]["human_reviewed"] == {"const": False}


class _PreparationDecoder:
    info = VideoInfo(fps=30.0, total_frames=900)

    def __init__(self, video_id: str, video_path: Path) -> None:
        self.video_id = video_id
        assert video_path.is_file()

    def decode_indices(self, frame_indices: list[int]) -> list[DecodedFrame]:
        return [
            DecodedFrame(
                self.video_id,
                value,
                np.full(
                    (48, 64, 3),
                    int(100 + 80 * np.sin(value / 30)),
                    dtype=np.uint8,
                ),
            )
            for value in frame_indices
        ]

    def close(self) -> None:
        return None


def test_end_to_end_pack_has_exact_manifest_sheet_mapping(
    tmp_path: Path, monkeypatch
) -> None:
    def compact(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        value = int(image[0, 0, 0])
        histogram = np.zeros(32, dtype=np.float64)
        histogram[min(value // 8, 31)] = 1.0
        return np.full((8, 8), value, dtype=np.uint8), histogram

    monkeypatch.setattr(signals, "_small_gray_and_histogram", compact)
    dataset = tmp_path / "dataset"
    video = dataset / "Videos_L01" / "video" / "L01_V001.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"fake")
    manifest = tmp_path / "mb1_candidate_manifest.jsonl"
    manifest.write_text(
        json.dumps({"candidate_id": "mb1_c001", "video_id": "L01_V001"}) + "\n",
        encoding="utf-8",
    )
    qc = tmp_path / "mb1_candidate_qc.jsonl"
    qc.write_text(
        json.dumps(
            {
                "candidate_id": "mb1_c001",
                "video_id": "L01_V001",
                "usability": "USABLE",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "output"
    result = prepare_mb1_v02_candidates(
        MB1V02Config(dataset, manifest, qc, output),
        decoder_factory=_PreparationDecoder,
    )
    rows = [
        json.loads(line)
        for line in (output / "mb1_v02_candidate_manifest.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert rows and result["selection"]["hard_cut_overlap_count"] == 0
    for row in rows:
        assert (output / row["overview_sheet_path"]).is_file()
        assert (output / row["dense_sheet_path"]).is_file()
        assert row["overview_displayed_frames"] == sorted(
            set(row["overview_displayed_frames"])
        )
        assert row["dense_displayed_frames"] == sorted(
            set(row["dense_displayed_frames"])
        )


def test_generation_functions_do_not_read_prior_semantic_ground_truth() -> None:
    source = inspect.getsource(build_source_video_pool) + inspect.getsource(
        runner.prepare_mb1_v02_candidates
    )
    for forbidden in (
        "acceptable_start_frame",
        "acceptable_end_frame",
        "preferred_frame",
        "moment_type",
        "query_text",
    ):
        assert forbidden not in source
    parameters = inspect.signature(runner.prepare_mb1_v02_candidates).parameters
    assert "model" not in parameters and "stage1" not in parameters


def test_proposal_generation_is_deterministic_for_fixed_input() -> None:
    series = _series((120, 300, 480, 660, 840))
    shots = continuous_shot_segments(1001, ())
    first = propose_active_windows("L01_V001", 10.0, 1001, series, (), shots)
    second = propose_active_windows("L01_V001", 10.0, 1001, series, (), shots)
    assert first == second


def test_bundle_allowlist_excludes_unlisted_heavy_artifacts(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    manifest = [{
        "candidate_id": "mb1v02_c001",
        "overview_sheet_path": "overview/mb1v02_c001.jpg",
        "dense_sheet_path": "dense/mb1v02_c001.jpg",
    }]
    for name in BUNDLE_FILES:
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if name == "mb1_v02_candidate_manifest.jsonl":
            path.write_text(json.dumps(manifest[0]) + "\n", encoding="utf-8")
        else:
            path.write_text("{}\n", encoding="utf-8")
    for name in ("overview/mb1v02_c001.jpg", "dense/mb1v02_c001.jpg"):
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"jpeg")
    (output / "raw.mp4").write_bytes(b"heavy")
    (output / "vectors.npy").write_bytes(b"heavy")
    archive = create_mb1_v02_bundle(output, tmp_path / "bundle.zip")
    with ZipFile(archive) as stream:
        names = stream.namelist()
    assert "mb1_v02_candidate_manifest.jsonl" in names
    assert len([name for name in names if name.endswith(".jpg")]) == 2
    assert not any(name.endswith((".mp4", ".npy")) for name in names)
