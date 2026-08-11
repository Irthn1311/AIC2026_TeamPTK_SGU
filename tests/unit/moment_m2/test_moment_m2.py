from __future__ import annotations

import inspect
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest

from triage_eg.experiments.mb1_e1.runner import (
    FROZEN_M1_SETTINGS as MB1_E1_FROZEN_SETTINGS,
)
from triage_eg.experiments.mb1_e1.runner import (
    refine_inside_candidate_window as mb1_e1_refine,
)
from triage_eg.experiments.moment_m1 import DecodedFrame, VideoInfo
from triage_eg.experiments.moment_m2 import (
    A0_METHOD,
    BOUNDARY_LIKE_TYPES,
    aggregate_m2_metrics,
    centered_moving_average,
    create_m2_bundle,
    score_dense_window,
    select_dense_peak,
    smoothing_width_frames,
    solve_moment_plateau,
    solve_plateau_from_smoothed,
)
from triage_eg.experiments.moment_m2 import plateau as plateau_module
from triage_eg.experiments.moment_m2 import runner as runner_module
from triage_eg.retrieval.stage1b.writers import write_json, write_jsonl


def test_a0_reuses_exact_frozen_mb1_e1_m1_semantics() -> None:
    assert A0_METHOD == "FROZEN_M1_COARSE_TO_FINE"
    assert runner_module.refine_inside_candidate_window is mb1_e1_refine
    assert runner_module.FROZEN_M1_SETTINGS is MB1_E1_FROZEN_SETTINGS


class _Decoder:
    def __init__(self) -> None:
        self.info = VideoInfo(fps=25.0, total_frames=20)
        self.requested: list[int] = []

    def decode_indices(self, frame_indices: list[int]) -> list[DecodedFrame]:
        self.requested = list(frame_indices)
        return [
            DecodedFrame("L01_V001", index, np.full((2, 2, 3), index, dtype=np.uint8))
            for index in frame_indices
        ]


class _Encoder:
    def __init__(self) -> None:
        self.calls = 0

    def encode(self, frames: list[DecodedFrame]) -> np.ndarray:
        self.calls += 1
        output = np.zeros((len(frames), 512), dtype=np.float32)
        output[:, 0] = 1.0
        output[:, 1] = np.arange(1, len(frames) + 1, dtype=np.float32)
        return output


def test_a1_decodes_every_frame_in_inclusive_candidate_window() -> None:
    decoder, encoder = _Decoder(), _Encoder()
    text = np.zeros(512, dtype=np.float32)
    text[0] = 1.0
    frames, scores, _, timing = score_dense_window(
        decoder=decoder,
        image_encoder=encoder,
        text_embedding=text,
        window_start=4,
        window_end=9,
    )
    assert frames.tolist() == decoder.requested == [4, 5, 6, 7, 8, 9]
    assert len(scores) == timing["dense_frame_count"] == 6


def test_a1_tie_chooses_lower_actual_frame_index() -> None:
    frames = np.asarray([19, 11, 15], dtype=np.int64)
    scores = np.asarray([0.8, 0.8, 0.7], dtype=np.float32)
    assert select_dense_peak(frames, scores) == pytest.approx((11, 0.8))


def test_dense_scores_are_computed_once_and_shared_by_a1_a2() -> None:
    decoder, encoder = _Decoder(), _Encoder()
    text = np.zeros(512, dtype=np.float32)
    text[0] = 1.0
    frames, scores, _, timing = score_dense_window(
        decoder=decoder,
        image_encoder=encoder,
        text_embedding=text,
        window_start=0,
        window_end=4,
    )
    solve_moment_plateau(frames, scores, 25.0, "ACTION_VISIBILITY")
    assert encoder.calls == timing["dense_image_encoding_calls"] == 1
    source = inspect.getsource(runner_module.run_m2)
    assert source.count("score_dense_window(") == 1


def test_centered_smoothing_preserves_frame_coordinate_mapping() -> None:
    values = np.asarray([1, 2, 3, 4, 5], dtype=np.float64)
    assert smoothing_width_frames(25.0) == 5
    smoothed = centered_moving_average(values, 3)
    assert smoothed.tolist() == pytest.approx([1.5, 2.0, 3.0, 4.0, 4.5])
    solution = solve_moment_plateau(
        np.arange(100, 105), values, 12.5, "ACTION_VISIBILITY"
    )
    assert len(solution.smoothed_clip_scores) == 5
    assert 100 <= solution.smoothed_peak_frame <= 104


def test_half_prominence_threshold_is_exact() -> None:
    frames = np.arange(5)
    curve = np.asarray([0.0, 0.4, 1.0, 0.4, 0.0])
    solution = solve_plateau_from_smoothed(
        frames, curve, curve, 25.0, "ACTION_VISIBILITY", 5
    )
    assert solution.baseline_score == pytest.approx(0.4)
    assert solution.prominence == pytest.approx(0.6)
    assert solution.plateau_threshold == pytest.approx(0.7)


def test_peak_plateau_contains_smoothed_peak() -> None:
    solution = solve_moment_plateau(
        np.arange(8),
        np.asarray([0.0, 0.1, 0.7, 1.0, 0.8, 0.1, 0.0, -0.1]),
        25.0,
        "STATE",
    )
    assert solution.plateau_start_frame <= solution.smoothed_peak_frame
    assert solution.smoothed_peak_frame <= solution.plateau_end_frame


@pytest.mark.parametrize(
    "moment_type",
    sorted(BOUNDARY_LIKE_TYPES - {"LAST_OCCURRENCE", "TRANSITION_OFFSET"}),
)
def test_left_edge_moment_types_choose_plateau_start(moment_type: str) -> None:
    frames = np.arange(7)
    curve = np.asarray([0.0, 0.0, 0.8, 1.0, 0.8, 0.0, 0.0])
    solution = solve_plateau_from_smoothed(frames, curve, curve, 25.0, moment_type, 5)
    assert solution.prediction == solution.plateau_start_frame == 2


@pytest.mark.parametrize("moment_type", ["LAST_OCCURRENCE", "TRANSITION_OFFSET"])
def test_right_edge_moment_types_choose_plateau_end(moment_type: str) -> None:
    frames = np.arange(7)
    curve = np.asarray([0.0, 0.0, 0.8, 1.0, 0.8, 0.0, 0.0])
    solution = solve_plateau_from_smoothed(frames, curve, curve, 25.0, moment_type, 5)
    assert solution.prediction == solution.plateau_end_frame == 4


def test_action_visibility_chooses_highest_raw_score_inside_plateau() -> None:
    frames = np.arange(7)
    raw = np.asarray([0.0, 0.0, 0.8, 0.9, 1.0, 0.0, 0.0])
    smoothed = np.asarray([0.0, 0.0, 0.8, 1.0, 0.8, 0.0, 0.0])
    solution = solve_plateau_from_smoothed(
        frames, raw, smoothed, 25.0, "ACTION_VISIBILITY", 5
    )
    assert (solution.plateau_start_frame, solution.plateau_end_frame) == (2, 4)
    assert solution.prediction == 4


def test_extremum_falls_back_to_dense_peak_with_explicit_diagnostic() -> None:
    solution = solve_moment_plateau(
        np.arange(5), np.asarray([0.0, 0.2, 1.0, 0.1, 0.0]), 25.0, "EXTREMUM"
    )
    assert solution.prediction == solution.raw_dense_peak_frame == 2
    assert "EXTREMUM_TEMPORAL_SOLVER_NOT_IMPLEMENTED" in solution.diagnostics


def test_prediction_solver_has_no_gt_or_confidence_inputs() -> None:
    source = inspect.getsource(plateau_module).lower()
    for forbidden in (
        "acceptable_start_frame",
        "acceptable_end_frame",
        "preferred_frame",
        "annotation_confidence",
    ):
        assert forbidden not in source
    assert set(inspect.signature(solve_moment_plateau).parameters) == {
        "frame_indices",
        "raw_clip_scores",
        "fps",
        "moment_type",
    }


def test_interval_metrics_and_pairwise_regressions_are_visible() -> None:
    rows = [
        {
            "a0_distance_to_interval": 5,
            "a1_distance_to_interval": 3,
            "a2_distance_to_interval": 0,
            "a0_interval_hit": False,
            "a1_interval_hit": False,
            "a2_interval_hit": True,
            "a0_preferred_frame_error": 5,
            "a1_preferred_frame_error": 3,
            "a2_preferred_frame_error": 1,
        },
        {
            "a0_distance_to_interval": 0,
            "a1_distance_to_interval": 0,
            "a2_distance_to_interval": 2,
            "a0_interval_hit": True,
            "a1_interval_hit": True,
            "a2_interval_hit": False,
            "a0_preferred_frame_error": 1,
            "a1_preferred_frame_error": 1,
            "a2_preferred_frame_error": 2,
        },
    ]
    metrics = aggregate_m2_metrics(rows)
    assert metrics["A2_MOMENT_TYPE_PLATEAU"]["INTERVAL_HIT_RATE"] == 0.5
    assert metrics["A2_VS_A1"] == {
        "A2_WINS": 1,
        "A1_WINS": 1,
        "TIES": 0,
        "NEW_HITS": 1,
        "LOST_HITS": 1,
    }


def test_bundle_is_allowlisted_and_excludes_heavy_assets(tmp_path: Path) -> None:
    output = tmp_path / "m2"
    for name in ("m2_summary.json", "m2_metrics.json", "run_manifest.json"):
        write_json(output / name, {})
    for name in ("moment_results.jsonl", "moment_score_curves.jsonl", "issues.jsonl"):
        write_jsonl(output / name, [])
    write_json(output / "visuals/review_key.json", {})
    (output / "visuals/example_ab.jpg").write_bytes(b"small")
    (output / "benchmark").mkdir(parents=True)
    for name in (
        "mb1_ai_semantic_moments.jsonl",
        "mb1_candidate_manifest.jsonl",
        "mb1_ai_semantic_moments.sha256",
        "mb1_candidate_manifest.sha256",
    ):
        (output / "benchmark" / name).write_text("provenance\n", encoding="utf-8")
    (output / "cache").mkdir()
    np.save(output / "cache/scores.npy", np.ones(3))
    (output / "raw.mp4").write_bytes(b"heavy")
    bundle = create_m2_bundle(output, tmp_path / "bundle.zip")
    with ZipFile(bundle) as archive:
        names = set(archive.namelist())
    assert "visuals/example_ab.jpg" in names
    assert not any(name.endswith((".npy", ".mp4")) for name in names)
    assert names == {
        "m2_summary.json",
        "m2_metrics.json",
        "moment_results.jsonl",
        "moment_score_curves.jsonl",
        "run_manifest.json",
        "issues.jsonl",
        "benchmark/mb1_ai_semantic_moments.jsonl",
        "benchmark/mb1_ai_semantic_moments.sha256",
        "benchmark/mb1_candidate_manifest.jsonl",
        "benchmark/mb1_candidate_manifest.sha256",
        "visuals/review_key.json",
        "visuals/example_ab.jpg",
    }
