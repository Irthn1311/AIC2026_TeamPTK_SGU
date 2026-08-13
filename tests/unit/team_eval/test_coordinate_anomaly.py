from __future__ import annotations

from pathlib import Path

from aic2026_eval.coordinate_anomaly import (
    LOCAL_ASSET_OFFSET,
    PASS_WITH_OFFSETS,
    classify_failure,
    close_coordinate_policy,
    decide_coordinate_contract,
)


def original_checks(*, clean: bool = False, identity_mismatch: bool = False) -> list[dict]:
    failures = {"L21_V012": (113, 13323), "L21_V016": (135, 15147)}
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
            actual = frame_idx
            if identity_mismatch and video_id == "L21_V001" and sample_index == 0:
                actual += 1
            samples.append(
                {
                    "btc_n": btc_n,
                    "mapping_frame_idx": frame_idx,
                    "actual_decoded_frame_idx": actual,
                    "correspondence_pass": correspondence_pass,
                }
            )
        checks.append({"video_id": video_id, "status": "PASS", "samples": samples})
    return checks


def anomaly(classification: str = LOCAL_ASSET_OFFSET) -> dict:
    return {
        "video_id": "L21_V012",
        "btc_n": 112,
        "mapping_frame_idx": 13200,
        "classification": classification,
    }


def test_repeated_small_jpeg_offsets_do_not_imply_coordinate_drift() -> None:
    classification = classify_failure(
        {
            "raw_decode_identity_exact": True,
            "best_offset_frames": 1,
            "best_correlation": 0.9997,
        },
        [{"own_correspondence_pass": False}],
        {"frame_idx_monotonic": True, "pts_time_monotonic": True},
        [
            {"diagnostic_best_offset_frames": 1},
            {"diagnostic_best_offset_frames": 1},
        ],
    )
    assert classification == LOCAL_ASSET_OFFSET


def test_jpeg_audit_failure_is_scoring_accepted_and_recorded() -> None:
    diagnostics = [anomaly() for _ in range(6)]
    decision = decide_coordinate_contract(original_checks(), diagnostics, [])
    assert decision["status"] == PASS_WITH_OFFSETS
    assert decision["documented_btc_local_asset_offset_count"] == 6
    assert decision["btc_jpeg_correspondence_cleanliness"] == "HAS_LOCAL_1_TO_2_FRAME_OFFSETS"
    assert decision["jpeg_correspondence_used_as_hard_gate"] is False


def test_true_structural_coordinate_conflicts_still_fail() -> None:
    assert decide_coordinate_contract(
        original_checks(), [anomaly("REAL_COORDINATE_DRIFT")], []
    )["status"] == "FAIL"
    identity_decision = decide_coordinate_contract(original_checks(identity_mismatch=True), [], [])
    assert identity_decision["status"] == "FAIL"


def _mapping(path: Path, rows: list[tuple[int, int]]) -> None:
    path.write_text(
        "n,pts_time,fps,frame_idx\n"
        + "".join(f"{n},{frame / 25:.3f},25.0,{frame}\n" for n, frame in rows),
        encoding="utf-8",
    )


def _inventory(tmp_path: Path, video_id: str, rows: list[tuple[int, int]]) -> dict:
    root = tmp_path / video_id
    root.mkdir()
    video = root / "video.mp4"
    video.write_bytes(b"raw-video-present")
    mapping = root / "mapping.csv"
    _mapping(mapping, rows)
    return {
        "video_id": video_id,
        "video_path": str(video),
        "mapping_path": str(mapping),
        "total_frames": 30000,
    }


def test_final_policy_records_all_six_real_offsets_without_search(tmp_path: Path) -> None:
    inventory = [
        _inventory(tmp_path, "L21_V012", [(112, 13200), (113, 13323)]),
        _inventory(
            tmp_path,
            "L21_V016",
            [(134, 15000), (135, 15147), (137, 15300), (202, 22000)],
        ),
    ]
    decision, diagnostics, neighbor_rows, expanded = close_coordinate_policy(
        inventory, original_checks()
    )
    assert decision["status"] == PASS_WITH_OFFSETS
    assert decision["neighborhood_search_rerun"] is False
    assert len(diagnostics) == len(expanded) == 6
    assert neighbor_rows == []
    assert {row["classification"] for row in diagnostics} == {LOCAL_ASSET_OFFSET}
    assert all(
        row["best_raw_frame_idx"] == row["mapping_frame_idx"] + row["best_offset_frames"]
        for row in diagnostics
    )


def test_mapping_non_monotonic_coordinate_conflict_fails(tmp_path: Path) -> None:
    inventory = [_inventory(tmp_path, "L21_V001", [(1, 100), (2, 90)])]
    decision, _, _, _ = close_coordinate_policy(inventory, original_checks(clean=True))
    assert decision["status"] == "FAIL"
    assert {row["code"] for row in decision["structural_issues"]} == {
        "MAPPING_NON_MONOTONIC_COORDINATE_CONFLICT"
    }


def test_all_clean_coordinate_evidence_remains_pass() -> None:
    decision = decide_coordinate_contract(original_checks(clean=True), [], [])
    assert decision["status"] == "PASS"
    assert decision["original_sample_count"] == decision["original_pass_count"] == 48
    assert decision["original_fail_count"] == 0
