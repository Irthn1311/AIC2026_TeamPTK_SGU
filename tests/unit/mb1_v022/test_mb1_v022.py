from __future__ import annotations

import inspect
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest

from triage_eg.experiments.mb1_v021.signals import (
    SignalSeries,
    is_seed_near_duplicate,
    robust_baseline,
)
from triage_eg.experiments.mb1_v022 import runner
from triage_eg.experiments.mb1_v022.runner import BUNDLE_BASE_FILES
from triage_eg.experiments.mb1_v022.signals import (
    center_relative_features,
    context_window,
    cut_inside_window,
    final_continuity_audit,
    orb_transition,
    safe_dense_focus,
)
from triage_eg.experiments.moment_m1 import DecodedFrame, VideoInfo


def _series(fps: float = 20.0, seconds: float = 6.0) -> SignalSeries:
    count = int(fps * seconds) + 1
    frames = np.arange(count, dtype=np.int64)
    values = np.full(count, 0.01, dtype=np.float64)
    histograms = np.zeros((count, 32), dtype=np.float64)
    histograms[:, 0] = 1.0
    baseline = robust_baseline(values)
    return SignalSeries(
        frame_indices=frames,
        pixel_differences=values,
        histogram_differences=values.copy(),
        histograms=histograms,
        spatial_concentrations=np.full(count, 0.5),
        pixel_robust_z=np.zeros(count),
        histogram_robust_z=np.zeros(count),
        pixel_percentiles=np.zeros(count),
        histogram_percentiles=np.zeros(count),
        pixel_baseline=baseline,
        histogram_baseline=baseline,
        stride_frames=1,
        decode_ms=0.0,
        signal_ms=0.0,
    )


def test_center_at_window_start_is_rejected() -> None:
    result, reason = context_window(center_frame=0, fps=20.0, total_frames=200, cut_frames=())
    assert result is None and reason == "INSUFFICIENT_PRE_CONTEXT"


def test_center_at_window_end_is_rejected() -> None:
    result, reason = context_window(center_frame=199, fps=20.0, total_frames=200, cut_frames=())
    assert result is None and reason == "INSUFFICIENT_POST_CONTEXT"


def test_center_equal_coarse_shot_start_is_rejected() -> None:
    result, reason = context_window(center_frame=100, fps=20.0, total_frames=300, cut_frames=(100,))
    assert result is None and reason == "INSUFFICIENT_PRE_CONTEXT"


def test_less_than_minimum_pre_context_is_rejected() -> None:
    result, reason = context_window(center_frame=124, fps=20.0, total_frames=300, cut_frames=(100,))
    assert result is None and reason == "INSUFFICIENT_PRE_CONTEXT"


def test_less_than_minimum_post_context_is_rejected() -> None:
    result, reason = context_window(center_frame=176, fps=20.0, total_frames=300, cut_frames=(200,))
    assert result is None and reason == "INSUFFICIENT_POST_CONTEXT"


def test_valid_symmetric_three_second_window_is_possible() -> None:
    result, reason = context_window(
        center_frame=100, fps=20.0, total_frames=300, cut_frames=(60, 141)
    )
    assert reason is None and result is not None
    assert result.window.reason == "SYMMETRIC_SHRINK"
    assert result.window.final_duration_seconds == 3.0
    assert result.window.start_frame < 100 < result.window.end_frame


def test_profiles_are_relative_to_proposal_center() -> None:
    series = _series()
    series.pixel_differences[55:66] = 0.5
    first = center_relative_features(series, 60, 20.0)
    second = center_relative_features(series, 80, 20.0)
    assert first["center_activity"]["peak"] > second["center_activity"]["peak"]


def test_cut_coordinate_c_minus_one_to_c_is_consistent() -> None:
    assert not cut_inside_window(100, 100, 200)
    assert cut_inside_window(101, 100, 200)
    assert cut_inside_window(200, 100, 200)
    assert not cut_inside_window(201, 100, 200)


def test_cut_safety_margin_applies_outside_window() -> None:
    result, reason = context_window(
        center_frame=100, fps=20.0, total_frames=300, cut_frames=(64, 136)
    )
    assert result is None and reason in {
        "INSUFFICIENT_PRE_CONTEXT",
        "INSUFFICIENT_POST_CONTEXT",
        "KNOWN_CUT_SAFETY_MARGIN",
    }


def _texture(seed: int = 1) -> np.ndarray:
    cv2 = pytest.importorskip("cv2", reason="ORB tests require OpenCV")
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 256, (180, 320), dtype=np.uint8)
    cv2.circle(image, (150, 90), 35, 255, 3)
    cv2.rectangle(image, (30, 30), (90, 130), 0, 3)
    return image


def test_camera_translation_preserves_more_orb_than_replacement() -> None:
    cv2 = pytest.importorskip("cv2", reason="ORB tests require OpenCV")
    image = _texture(1)
    translated = cv2.warpAffine(
        image, np.float32([[1, 0, 5], [0, 1, 3]]), (image.shape[1], image.shape[0])
    )
    replacement = _texture(100)
    same = orb_transition(image, translated)
    cut = orb_transition(image, replacement)
    assert same.available and cut.available
    assert float(same.continuity) > float(cut.continuity)


def test_obvious_replacement_has_low_orb_continuity() -> None:
    pytest.importorskip("cv2", reason="ORB tests require OpenCV")

    class CutDecoder:
        info = VideoInfo(fps=20.0, total_frames=81)

        def decode_indices(self, frame_indices: list[int]) -> list[DecodedFrame]:
            first, second = _texture(2), _texture(200)
            return [
                DecodedFrame(
                    "L01_V001",
                    value,
                    np.repeat((first if value < 40 else second)[..., None], 3, axis=2),
                )
                for value in frame_indices
            ]

    audit = final_continuity_audit(CutDecoder(), 0, 80)
    assert audit.status == "AUTO_CONTINUITY_SCREEN_REJECT"
    assert audit.abrupt_transition_frames


def test_orb_unavailable_falls_back_without_rejection() -> None:
    pytest.importorskip("cv2", reason="ORB tests require OpenCV")

    class BlankDecoder:
        info = VideoInfo(fps=20.0, total_frames=81)

        def decode_indices(self, frame_indices: list[int]) -> list[DecodedFrame]:
            return [
                DecodedFrame("L01_V001", value, np.zeros((90, 160, 3), dtype=np.uint8))
                for value in frame_indices
            ]

    audit = final_continuity_audit(BlankDecoder(), 0, 80)
    assert audit.orb_available_fraction == 0.0
    assert audit.status == "AUTO_CONTINUITY_SCREEN_PASS"


def test_dense_focus_cannot_use_unsafe_edge_peak() -> None:
    series = _series(fps=20.0, seconds=4.0)
    series.pixel_robust_z[1] = 100
    series.pixel_robust_z[40] = 10
    assert safe_dense_focus(series, 0, 80, 40, 20.0) == 40


def test_seed_duplicate_exclusion_is_retained() -> None:
    seeds = [
        {
            "candidate_id": "mb1v02_c001",
            "video_id": "L01_V001",
            "window_start_frame": 100,
            "window_end_frame": 180,
            "proposal_center_frame": 140,
        }
    ]
    assert is_seed_near_duplicate(
        video_id="L01_V001",
        start_frame=110,
        end_frame=190,
        center_frame=150,
        fps=20.0,
        seeds=seeds,
    )
    assert not is_seed_near_duplicate(
        video_id="L01_V002",
        start_frame=110,
        end_frame=190,
        center_frame=150,
        fps=20.0,
        seeds=seeds,
    )


def test_continuity_cap_is_used_by_final_ranking() -> None:
    source = inspect.getsource(runner._verify_and_select)
    assert '"continuity_capped_before_after"' in source
    assert "base_transition_score * proposal.audit.continuity_quality" in source


def test_semantic_interval_fields_are_never_consumed() -> None:
    source = inspect.getsource(runner._initial_proposals) + inspect.getsource(
        runner.prepare_mb1_v022_candidates
    )
    for field in runner.FORBIDDEN_SEMANTIC_FIELDS:
        assert field not in source


def test_bundle_excludes_heavy_assets(tmp_path: Path) -> None:
    root = tmp_path / "output"
    root.mkdir()
    manifest = {
        "candidate_id": "mb1v022_c001",
        "overview_sheet_path": "overview/mb1v022_c001.jpg",
        "dense_sheet_path": "dense/mb1v022_c001.jpg",
    }
    for name in BUNDLE_BASE_FILES:
        path = root / name
        if name == "mb1_v022_candidate_manifest.jsonl":
            path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
        else:
            path.write_text("{}\n", encoding="utf-8")
    for name in (
        "overview/mb1v022_c001.jpg",
        "dense/mb1v022_c001.jpg",
        "montages/overview_montage_001.jpg",
        "montages/dense_montage_001.jpg",
    ):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"jpeg")
    (root / "raw.mp4").write_bytes(b"heavy")
    (root / "vectors.npy").write_bytes(b"heavy")
    archive = runner.create_mb1_v022_bundle(root, tmp_path / "bundle.zip")
    with ZipFile(archive) as stream:
        names = stream.namelist()
    assert not any(name.endswith((".mp4", ".npy", ".bin")) for name in names)
