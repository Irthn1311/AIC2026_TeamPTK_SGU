from __future__ import annotations

import inspect
import json
from dataclasses import asdict
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pytest
import yaml

from triage_eg.experiments.moment_m3 import (
    M3InferenceCase,
    M3Settings,
    build_case_registry,
    build_metrics,
    build_state_signals,
    evaluate_predictions_only,
    local_window,
    solve_state_transition,
    validate_trusted_registry_row,
)
from triage_eg.experiments.moment_m3.solver import MOTION_WEIGHT


def _case(moment_type: str = "ONSET") -> M3InferenceCase:
    return M3InferenceCase("case", "L01_V001", moment_type, "event", "before", "after", 110)


def _solve(contrast: np.ndarray, moment_type: str = "ONSET", *, m1: int = 106):
    frames = np.arange(100, 100 + len(contrast), dtype=np.int64)
    angles = np.linspace(0.0, 0.1, len(frames))
    embeddings = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    return solve_state_transition(
        _case(moment_type),
        frame_ids=frames,
        contrast=contrast,
        image_embeddings=embeddings,
        fps=10.0,
        m1_prediction=m1,
        use_motion_tiebreaker=False,
    )


def test_state_contrast_is_after_minus_before() -> None:
    images = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    values = build_state_signals(
        images,
        before_text_embedding=np.asarray([1.0, 0.0]),
        after_text_embedding=np.asarray([0.0, 1.0]),
        event_text_embedding=np.asarray([1.0, 1.0]),
    )
    assert values["contrast"].tolist() == [-1.0, 1.0]


@pytest.mark.parametrize("moment_type", ["ONSET", "CONTACT"])
def test_low_to_high_transition_routes_and_finds_boundary(moment_type: str) -> None:
    solution, _ = _solve(np.asarray([-1.0] * 10 + [1.0] * 11), moment_type)
    assert abs(solution.prediction - 110) <= 1
    assert not solution.used_m1_fallback


def test_no_information_falls_back_to_valid_m1_prediction() -> None:
    solution, _ = _solve(np.zeros(21), m1=107)
    assert solution.prediction == 107
    assert solution.used_m1_fallback
    assert solution.fallback_reason == "LOW_INFORMATION_SIGNAL"


def test_single_frame_spike_is_not_persistence_valid_boundary() -> None:
    values = np.asarray([-1.0] * 10 + [1.0] + [-1.0] * 10)
    solution, _ = _solve(values)
    assert solution.valid_candidate_count == 0
    assert solution.used_m1_fallback
    assert solution.prediction == 106


def test_first_occurrence_selects_earliest_strong_persistent_change() -> None:
    values = np.asarray([-1.0] * 10 + [0.0] * 10 + [1.0] * 11)
    solution, _ = _solve(values, "FIRST_OCCURRENCE")
    assert solution.prediction == 110


def test_extremum_and_control_types_route_to_m1() -> None:
    for moment_type, reason in (
        ("EXTREMUM", "M3_EXTREMUM_UNSUPPORTED"),
        ("ACTION_VISIBILITY", "M3_TYPE_ROUTED_TO_M1"),
    ):
        solution, _ = _solve(np.asarray([-1.0] * 10 + [1.0] * 11), moment_type, m1=108)
        assert solution.prediction == 108
        assert solution.fallback_reason == reason


def test_conditional_case_is_excluded_from_primary_metrics() -> None:
    base = {
        "case_id": "x",
        "video_id": "L01_V001",
        "moment_type": "ONSET",
        "m1_distance": 2,
        "m3_a1_distance": 1,
        "m3_a2_distance": 1,
    }
    primary, _, secondary = build_metrics(
        [
            {**base, "primary_gate": True, "conditional": False},
            {**base, "case_id": "y", "primary_gate": False, "conditional": True},
        ]
    )
    assert primary["case_count"] == 1
    assert secondary["case_count"] == 1


def test_inference_contract_rejects_gt_and_solver_has_no_gt_parameter() -> None:
    with pytest.raises(TypeError):
        M3InferenceCase(
            "case",
            "L01_V001",
            "ONSET",
            "event",
            "before",
            "after",
            110,
            accepted_intervals=[[100, 101]],
        )
    assert "accepted" not in inspect.signature(solve_state_transition).parameters


def test_accepted_interval_is_consumed_by_evaluator_only() -> None:
    row = {
        "case_id": "case",
        "video_id": "L01_V001",
        "moment_type": "ONSET",
        "primary_gate": True,
        "conditional": False,
        "accepted_intervals": [[10, 12]],
    }
    output = evaluate_predictions_only(row, {"m1": 8, "m3_a1": 11, "m3_a2": 13})
    assert output["m3_a1_hit"] is True
    assert output["m1_distance"] == 2
    assert output["m3_a2_distance"] == 1


def test_exact_raw_frame_coordinates_are_preserved_without_timestamp_reconstruction() -> None:
    assert local_window(1_000_003, fps=25.0, total_frames=2_000_000) == (999_965, 1_000_041)
    source = inspect.getsource(local_window)
    assert "timestamp" not in source
    solution, _ = _solve(np.asarray([-1.0] * 10 + [1.0] * 11))
    assert solution.prediction >= 100


def test_motion_weight_and_all_algorithm_settings_are_frozen() -> None:
    assert MOTION_WEIGHT == 0.10
    assert asdict(M3Settings())["motion_weight"] == 0.10
    with pytest.raises(ValueError, match="parameter sweeps are forbidden"):
        M3Settings(motion_weight=0.2)


def test_experiment_config_has_exactly_three_arms_and_no_sweep() -> None:
    config = yaml.safe_load(Path("configs/experiments/moment_m3.yaml").read_text())
    assert config["parameter_sweep"] is False
    assert len(config["variants"]) == 3
    assert config["state_transition"]["motion_weight"] == 0.10


def _trusted_row() -> dict:
    return {
        "case_id": "m3_x",
        "source_candidate_id": "x",
        "source_version": "frozen",
        "video_id": "L01_V001",
        "moment_type": "ONSET",
        "semantic_event_vi": "bat dau",
        "semantic_event_en": "starts",
        "before_state_en": "not started",
        "after_state_en": "started",
        "accepted_intervals": [[10, 12]],
        "candidate_anchor_frame": 9,
        "candidate_anchor_source": "original",
        "primary_gate": True,
        "conditional": False,
        "annotation_confidence": "HIGH",
        "annotation_provenance": "original",
        "human_reviewed": True,
        "registry_status": "TRUSTED_METADATA_READY",
    }


def test_case_registry_rejects_missing_required_trusted_metadata() -> None:
    row = _trusted_row()
    row["before_state_en"] = None
    with pytest.raises(ValueError, match="M3_TRUSTED_METADATA_MISSING"):
        validate_trusted_registry_row(row)


def _write_zip(path: Path, name: str, rows: list[dict]) -> None:
    payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr(name, payload)


def test_frozen_seed_semantics_are_not_invented_from_reason_code(tmp_path: Path) -> None:
    new_ids = ("mb1v022_c005", "mb1v022_c013", "mb1v022_c014", "mb1v022_c015", "mb1v022_c017")
    qc = [
        {
            "candidate_id": candidate_id,
            "video_id": "L01_V001",
            "ai_qc": "CONDITIONAL" if candidate_id.endswith("014") else "USABLE",
            "semantic_event_vi": "su kien",
            "confidence": "HIGH",
            "interval_basis": "dense sheet",
        }
        for candidate_id in new_ids
    ]
    carry = [
        {
            "candidate_id": f"mb1v02_c{index:03d}",
            "video_id": "L01_V001",
            "prior_reason_code": "CONTACT_LIKE_REASON",
        }
        for index in range(13)
    ]
    ai_zip = tmp_path / "qc.zip"
    with ZipFile(ai_zip, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr(
            "mb1_v022_ai_qc_new_candidates_v01.jsonl",
            "".join(json.dumps(row) + "\n" for row in qc),
        )
        archive.writestr(
            "mb1_v022_frozen_seed_carryover_v01.jsonl",
            "".join(json.dumps(row) + "\n" for row in carry),
        )
    candidate_zip = tmp_path / "candidates.zip"
    _write_zip(
        candidate_zip,
        "mb1_v022_candidate_manifest.jsonl",
        [
            {"candidate_id": candidate_id, "video_id": "L01_V001", "proposal_center_frame": 20}
            for candidate_id in new_ids
        ],
    )
    registry, summary = build_case_registry(
        ai_qc_zip=ai_zip, notebook20_candidates_zip=candidate_zip
    )
    frozen = [row for row in registry if row["source_version"] == "MB1_V02_FROZEN"]
    assert len(frozen) == 13
    assert all(row["moment_type"] is None for row in frozen)
    assert all(not row["reason_code_used_to_infer_semantics"] for row in frozen)
    assert summary["frozen_seed_metadata_found"] == 0
    assert summary["primary_case_count"] == 4
    assert summary["benchmark_coverage"] == "TOO_SMALL_FOR_KEEP_DROP"
