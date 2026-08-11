from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from triage_eg.experiments.mb1_v02 import render_contact_sheet
from triage_eg.experiments.mb1_v02.signals import ShotSegment
from triage_eg.experiments.mb1_v021 import runner, signals
from triage_eg.experiments.mb1_v021.runner import (
    BUNDLE_BASE_FILES,
    Proposal,
    build_source_video_pool,
    create_mb1_v021_bundle,
    load_and_validate_qc,
    resolve_frozen_seeds,
)
from triage_eg.experiments.mb1_v021.signals import (
    SignalSeries,
    WindowAdjustment,
    adaptive_window,
    dense_displayed_frames,
    dense_focus_frame,
    detect_soft_transition_runs,
    hard_cut_mask,
    is_seed_near_duplicate,
    robust_baseline,
    scan_local_video,
)
from triage_eg.experiments.moment_m1 import DecodedFrame, VideoInfo


def _series(count: int = 12) -> SignalSeries:
    zeros = np.zeros(count, dtype=np.float64)
    histograms = np.zeros((count, 32), dtype=np.float64)
    histograms[:, 0] = 1.0
    baseline = robust_baseline(np.full(count, 0.01))
    return SignalSeries(
        frame_indices=np.arange(count, dtype=np.int64),
        pixel_differences=zeros.copy(),
        histogram_differences=zeros.copy(),
        histograms=histograms,
        spatial_concentrations=zeros.copy(),
        pixel_robust_z=zeros.copy(),
        histogram_robust_z=zeros.copy(),
        pixel_percentiles=zeros.copy(),
        histogram_percentiles=zeros.copy(),
        pixel_baseline=baseline,
        histogram_baseline=baseline,
        stride_frames=1,
        decode_ms=0.0,
        signal_ms=0.0,
    )


def _proposal(center: int, score: float = 1.0) -> Proposal:
    return Proposal(
        video_id="L01_V001",
        fps=10.0,
        center_frame=center,
        window=WindowAdjustment(center - 20, center + 20, 4.0, 4.0, "NONE"),
        coarse_shot=ShotSegment(0, 1000),
        pre_activity={},
        center_activity={},
        post_activity={},
        transition_strength=0.1,
        before_after_visual_difference=0.1,
        spatial_activity_concentration=0.5,
        overall_activity=0.1,
        source_pool_origin="TEST",
        proposal_score=score,
    )


def test_thirteen_frozen_usable_seeds_resolve() -> None:
    manifest = []
    qc = []
    for number in range(1, 42):
        candidate_id = f"mb1v02_c{number:03d}"
        manifest.append(
            {
                "candidate_id": candidate_id,
                "video_id": f"L01_V{number:03d}",
                "fps": 25.0,
                "window_start_frame": 100,
                "window_end_frame": 200,
                "proposal_center_frame": 150,
                "overview_sheet_path": f"overview/{candidate_id}.jpg",
                "dense_sheet_path": f"dense/{candidate_id}.jpg",
            }
        )
        qc.append(
            {
                "candidate_id": candidate_id,
                "video_id": f"L01_V{number:03d}",
                "qc_status": "USABLE" if number <= 13 else "REJECT",
                "boundary_richness": "HIGH" if number <= 7 else "LOW",
            }
        )
    seeds = resolve_frozen_seeds(manifest, qc)
    assert len(seeds) == 13
    assert all(row["seed_class"] == "MB1_V02_USABLE_SEED" for row in seeds)


def test_ai_qc_hash_validation_accepts_exact_and_rejects_tamper(
    tmp_path: Path, monkeypatch
) -> None:
    rows = [
        {
            "candidate_id": f"mb1v02_c{number:03d}",
            "qc_status": "USABLE" if number <= 13 else "REJECT",
        }
        for number in range(1, 42)
    ]
    payload = "".join(json.dumps(row) + "\n" for row in rows).encode()
    expected = hashlib.sha256(payload).hexdigest()
    monkeypatch.setattr(runner, "EXPECTED_QC_SHA256", expected)
    path = tmp_path / "qc.jsonl"
    path.write_bytes(payload)
    assert len(load_and_validate_qc(path)) == 41
    path.write_bytes(payload + b"\n")
    try:
        load_and_validate_qc(path)
    except ValueError as error:
        assert "HASH_MISMATCH" in str(error)
    else:
        raise AssertionError("tampered AI-QC artifact was accepted")


def test_candidate_generation_does_not_read_semantic_interval_fields() -> None:
    source = inspect.getsource(runner._initial_proposals) + inspect.getsource(
        runner.prepare_mb1_v021_candidates
    )
    for forbidden in (
        "acceptable_start_frame",
        "acceptable_end_frame",
        "preferred_frame",
        "moment_type",
        "query_text",
    ):
        assert forbidden not in source


def test_expanded_source_pool_is_deterministic_and_priority_ordered() -> None:
    rt2 = [
        {"query_id": "q02", "source_video_id": "L01_V002", "difficulty_tags": []},
        {"query_id": "q01", "source_video_id": "L01_V001", "difficulty_tags": []},
    ]
    prior = {
        "selected_videos": [
            {"video_id": "L01_V003", "sampling_bucket": "GENERAL"},
            {"video_id": "L01_V001", "sampling_bucket": "GENERAL"},
        ]
    }
    expected = ["L01_V001", "L01_V002", "L01_V003"]
    assert [row["video_id"] for row in build_source_video_pool(rt2, prior)] == expected
    assert [
        row["video_id"] for row in build_source_video_pool(rt2[::-1], prior)
    ] == expected


def test_robust_normalization_handles_zero_mad_safely() -> None:
    baseline = robust_baseline(np.ones(20, dtype=np.float64))
    values = signals.robust_z(np.asarray([1.0, 2.0]), baseline)
    assert baseline.robust_scale == signals.EPSILON
    assert np.isfinite(values).all() and values[0] == 0.0 and values[1] > 0


def test_hard_cut_strong_dual_signal_veto() -> None:
    mask = hard_cut_mask(np.asarray([0.99, 0.94]), np.asarray([0.95, 0.98]))
    assert mask.tolist() == [True, False]


def test_hard_cut_extreme_single_signal_with_support_veto() -> None:
    mask = hard_cut_mask(np.asarray([0.998, 0.998]), np.asarray([0.80, 0.79]))
    assert mask.tolist() == [True, False]


def test_soft_crossfade_run_detector_on_gradual_transition() -> None:
    series = _series(10)
    series.pixel_robust_z[3:6] = 3.0
    series.histogram_robust_z[3:6] = 3.0
    series.histograms[6:, 0] = 0.0
    series.histograms[6:, 31] = 1.0
    runs = detect_soft_transition_runs(
        series, fps=15.0, reference_histogram_p95=0.2
    )
    assert runs == ((3, 5),)


class _LocalDecoder:
    info = VideoInfo(fps=10.0, total_frames=20)

    def decode_indices(self, frame_indices: list[int]) -> list[DecodedFrame]:
        return [
            DecodedFrame(
                "L01_V001",
                value,
                np.full((4, 4, 3), 0 if value < 6 else 255, dtype=np.uint8),
            )
            for value in frame_indices
        ]


def test_candidate_local_guard_catches_cut_missed_by_coarse(
    monkeypatch,
) -> None:
    def compact(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        value = int(image[0, 0, 0])
        histogram = np.zeros(32, dtype=np.float64)
        histogram[0 if value == 0 else 31] = 1.0
        return np.full((4, 4), value, dtype=np.uint8), histogram

    monkeypatch.setattr(signals, "_small_frame_features", compact)
    coarse = _series(20)
    assert not hard_cut_mask(
        coarse.pixel_percentiles, coarse.histogram_percentiles
    ).any()
    _, local = scan_local_video(_LocalDecoder(), 0, 12, coarse)
    assert 6 in local.hard_cut_frames


def test_adaptive_window_shrinks_to_clean_three_to_four_seconds() -> None:
    result = adaptive_window(
        center_frame=100,
        fps=10.0,
        shot_start_frame=60,
        shot_end_frame=150,
        veto_frames=(80, 116),
    )
    assert result is not None
    assert 3.0 <= result.final_duration_seconds < 4.0
    assert result.start_frame <= 100 <= result.end_frame
    assert result.reason == "TRIMMED_BOTH"


def test_adaptive_window_rejects_without_clean_three_seconds() -> None:
    assert (
        adaptive_window(
            center_frame=100,
            fps=10.0,
            shot_start_frame=80,
            shot_end_frame=130,
            veto_frames=(106,),
        )
        is None
    )


def test_seed_near_duplicate_exclusion_uses_iou_or_center_distance() -> None:
    seeds = [
        {
            "video_id": "L01_V001",
            "window_start_frame": 100,
            "window_end_frame": 140,
            "proposal_center_frame": 120,
        }
    ]
    assert is_seed_near_duplicate(
        video_id="L01_V001",
        start_frame=105,
        end_frame=145,
        center_frame=125,
        fps=10.0,
        seeds=seeds,
    )
    assert not is_seed_near_duplicate(
        video_id="L01_V002",
        start_frame=105,
        end_frame=145,
        center_frame=125,
        fps=10.0,
        seeds=seeds,
    )


def test_temporal_nms_and_per_video_cap_are_deterministic() -> None:
    proposals = [
        _proposal(100, 1.0),
        _proposal(120, 0.9),
        _proposal(200, 0.8),
        _proposal(300, 0.7),
        _proposal(400, 0.6),
    ]
    selected, counts = runner._nms_and_cap(proposals, fps=10.0)
    assert [item.center_frame for item in selected] == [100, 200, 300]
    assert counts["rejected_temporal_nms"] == 1
    assert counts["rejected_per_video_cap"] == 1


def test_dense_focus_uses_model_free_local_change_signal() -> None:
    series = _series(20)
    series.pixel_robust_z[7] = 5.0
    series.histogram_robust_z[7] = 4.0
    assert dense_focus_frame(series, 0, 19) == 7


def test_overview_dense_manifest_ids_match_rendered_labels(tmp_path: Path) -> None:
    overview_ids = signals.overview_displayed_frames(100, 220)
    dense_ids = dense_displayed_frames(100, 220, 160, 30.0)
    requested = sorted(set(overview_ids + dense_ids))
    frames = {
        value: DecodedFrame(
            "L01_V001", value, np.full((24, 32, 3), value % 255, dtype=np.uint8)
        )
        for value in requested
    }
    overview_labels = render_contact_sheet(
        tmp_path / "overview.jpg",
        candidate_id="mb1v021_c001",
        video_id="L01_V001",
        view_name="OVERVIEW",
        fps=30.0,
        frames=[frames[value] for value in overview_ids],
        quality=80,
    )
    dense_labels = render_contact_sheet(
        tmp_path / "dense.jpg",
        candidate_id="mb1v021_c001",
        video_id="L01_V001",
        view_name="DENSE_LOCAL_CHANGE",
        fps=30.0,
        frames=[frames[value] for value in dense_ids],
        quality=80,
    )
    assert overview_labels == overview_ids
    assert dense_labels == dense_ids


def test_bundle_excludes_raw_video_models_vectors_and_cache(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    manifest = {
        "candidate_id": "mb1v021_c001",
        "overview_sheet_path": "overview/mb1v021_c001.jpg",
        "dense_sheet_path": "dense/mb1v021_c001.jpg",
    }
    for name in BUNDLE_BASE_FILES:
        path = output / name
        if name == "mb1_v021_candidate_manifest.jsonl":
            path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
        else:
            path.write_text("{}\n", encoding="utf-8")
    for name in (
        "overview/mb1v021_c001.jpg",
        "dense/mb1v021_c001.jpg",
        "montages/overview_montage_001.jpg",
        "montages/dense_montage_001.jpg",
    ):
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"jpeg")
    (output / "raw.mp4").write_bytes(b"heavy")
    (output / "vectors.npy").write_bytes(b"heavy")
    archive = create_mb1_v021_bundle(output, tmp_path / "bundle.zip")
    with ZipFile(archive) as stream:
        names = stream.namelist()
    assert len([name for name in names if name.endswith(".jpg")]) == 4
    assert not any(name.endswith((".mp4", ".npy", ".bin")) for name in names)
