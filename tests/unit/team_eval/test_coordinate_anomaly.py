from __future__ import annotations

from copy import deepcopy

from aic2026_eval.coordinate_anomaly import (
    classify_failure,
    decide_coordinate_contract,
)


def original_checks(*, clean: bool = False) -> list[dict]:
    failures = {
        "L21_V012": (113, 13323),
        "L21_V016": (135, 15147),
    }
    checks = []
    for index in range(1, 17):
        video_id = f"L21_V{index:03d}"
        samples = []
        for sample_index in range(3):
            btc_n = sample_index + 1
            frame_idx = index * 1000 + sample_index * 100
            correspondence_pass = True
            if not clean and video_id in failures and sample_index == 1:
                btc_n, frame_idx = failures[video_id]
                correspondence_pass = False
            samples.append(
                {
                    "btc_n": btc_n,
                    "mapping_frame_idx": frame_idx,
                    "actual_decoded_frame_idx": frame_idx,
                    "correspondence_pass": correspondence_pass,
                }
            )
        checks.append({"video_id": video_id, "status": "PASS", "samples": samples})
    return checks


def local_mapping() -> dict:
    return {"frame_idx_monotonic": True, "pts_time_monotonic": True}


def clean_neighbors() -> list[dict]:
    return [
        {
            "neighbor_btc_n": value,
            "target_btc_n": 10,
            "own_correspondence_pass": value != 10,
        }
        for value in range(8, 13)
    ]


def expanded(video_id: str, *, systematic: bool = False) -> list[dict]:
    rows = []
    for index in range(12):
        row = {
            "video_id": video_id,
            "btc_n": index + 1,
            "mapping_frame_idx": index * 100,
            "actual_decoded_frame_idx": index * 100,
            "raw_decode_identity_exact": True,
            "is_original_failed_row": index == 5,
            "correspondence_pass": index != 5,
        }
        if systematic and index in {3, 4}:
            row["correspondence_pass"] = False
            row["diagnostic_best_offset_frames"] = 12
        rows.append(row)
    return rows


def anomaly(video_id: str, btc_n: int, frame_idx: int, classification: str) -> dict:
    return {
        "video_id": video_id,
        "btc_n": btc_n,
        "mapping_frame_idx": frame_idx,
        "classification": classification,
        "raw_decode_identity_exact": True,
        "local_mapping_consistency": local_mapping(),
    }


def test_isolated_jpeg_failure_does_not_automatically_imply_drift() -> None:
    classification = classify_failure(
        {
            "raw_decode_identity_exact": True,
            "best_correlation": 0.4,
            "best_mae": 0.2,
            "best_offset_frames": 0,
        },
        clean_neighbors(),
        local_mapping(),
        expanded("L21_V012"),
    )
    assert classification == "LOCAL_BTC_KEYFRAME_OR_MAPPING_ROW_ANOMALY"


def test_consistent_neighboring_offset_is_real_coordinate_drift() -> None:
    classification = classify_failure(
        {
            "raw_decode_identity_exact": True,
            "best_correlation": 0.95,
            "best_mae": 0.04,
            "best_offset_frames": 12,
        },
        clean_neighbors(),
        local_mapping(),
        expanded("L21_V012", systematic=True),
    )
    assert classification == "REAL_COORDINATE_DRIFT"


def test_isolated_failed_row_with_clean_neighbors_is_local_offset() -> None:
    classification = classify_failure(
        {
            "raw_decode_identity_exact": True,
            "best_correlation": 0.97,
            "best_mae": 0.03,
            "best_offset_frames": -7,
        },
        clean_neighbors(),
        local_mapping(),
        expanded("L21_V016"),
    )
    assert classification == "LOCAL_KEYFRAME_EXTRACTION_OFFSET"


def test_documented_local_anomaly_pass_requires_all_evidence() -> None:
    diagnostics = [
        anomaly(
            "L21_V012",
            113,
            13323,
            "LOCAL_BTC_KEYFRAME_OR_MAPPING_ROW_ANOMALY",
        ),
        anomaly("L21_V016", 135, 15147, "LOCAL_KEYFRAME_EXTRACTION_OFFSET"),
    ]
    evidence = expanded("L21_V012") + expanded("L21_V016")
    decision = decide_coordinate_contract(original_checks(), diagnostics, evidence)
    assert decision["status"] == "PASS_WITH_DOCUMENTED_LOCAL_BTC_ANOMALIES"
    broken = deepcopy(evidence)
    broken[0]["raw_decode_identity_exact"] = False
    decision = decide_coordinate_contract(original_checks(), diagnostics, broken)
    assert decision["status"] == "UNRESOLVED"


def test_systematic_drift_remains_fail() -> None:
    diagnostics = [
        anomaly("L21_V012", 113, 13323, "REAL_COORDINATE_DRIFT"),
        anomaly(
            "L21_V016",
            135,
            15147,
            "LOCAL_BTC_KEYFRAME_OR_MAPPING_ROW_ANOMALY",
        ),
    ]
    decision = decide_coordinate_contract(
        original_checks(),
        diagnostics,
        expanded("L21_V012", systematic=True) + expanded("L21_V016"),
    )
    assert decision["status"] == "FAIL"


def test_all_clean_coordinate_evidence_remains_pass() -> None:
    decision = decide_coordinate_contract(original_checks(clean=True), [], [])
    assert decision["status"] == "PASS"
    assert decision["original_sample_count"] == decision["original_pass_count"] == 48
    assert decision["original_fail_count"] == 0
    broken = original_checks(clean=True)
    broken[0] = {"video_id": "L21_V001", "status": "RAW_DECODE_FAILED", "samples": []}
    assert decide_coordinate_contract(broken, [], [])["status"] == "FAIL"
