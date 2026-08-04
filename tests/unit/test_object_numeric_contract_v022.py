from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile

import pytest

from triage_eg.data.cross_asset_patch_v021 import _object_targets
from triage_eg.data.object_numeric_contract import (
    OUTPUT_NAMES,
    NumericLimits,
    inspect_file,
    parse_label,
    parse_numeric,
    run_survey,
    validate_root,
    write_outputs,
)


@pytest.mark.parametrize(
    ("value", "status", "parsed"),
    [
        ("0.25", "VALID", 0.25),
        (" 0.25 ", "VALID", 0.25),
        (2, "VALID", 2.0),
        (2.5, "VALID", 2.5),
        (True, "UNSUPPORTED_TYPE", None),
        (None, "NULL", None),
        ("   ", "EMPTY", None),
        ("abc", "INVALID_FORMAT", None),
        ("NaN", "NON_FINITE", None),
        ("Infinity", "NON_FINITE", None),
        ("-Infinity", "NON_FINITE", None),
        ("1e-2", "VALID", 0.01),
        ("1,25", "INVALID_FORMAT", None),
    ],
)
def test_numeric_parser(value, status: str, parsed: float | None) -> None:
    result = parse_numeric(value)
    assert result["parse_status"] == status
    assert result["parsed_value"] == parsed


def test_numeric_parser_preserves_raw_and_trimmed() -> None:
    result = parse_numeric(" 0.25 ")
    assert result["raw_value"] == " 0.25 "
    assert result["trimmed_value"] == "0.25"


@pytest.mark.parametrize(
    ("value", "status", "parsed"),
    [
        ("84", "VALID", 84),
        (" 392 ", "VALID", 392),
        (84, "VALID", 84),
        (84.0, "VALID", 84),
        (84.5, "INVALID_FORMAT", None),
        ("84.0", "INVALID_FORMAT", None),
        ("1e2", "INVALID_FORMAT", None),
        ("-2", "VALID", -2),
        ("", "EMPTY", None),
        (None, "NULL", None),
        ("NaN", "NON_FINITE", None),
        (False, "UNSUPPORTED_TYPE", None),
    ],
)
def test_label_parser(value, status: str, parsed: int | None) -> None:
    result = parse_label(value)
    assert result["parse_status"] == status
    assert result["parsed_value"] == parsed


def object_payload(*, coordinate="0.1", score="0.9", label="84", count: int = 1):
    return {
        "detection_boxes": [[coordinate, "0.2", "0.3", "0.4"] for _ in range(count)],
        "detection_class_entities": ["Person"] * count,
        "detection_class_labels": [label] * count,
        "detection_class_names": ["/m/person"] * count,
        "detection_scores": [score] * count,
    }


def write_object(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def inspect(tmp_path: Path, value: object):
    path = tmp_path / "001.json"
    write_object(path, value)
    return inspect_file(path, video_id="L21_V017", ordinal_n=1, max_bytes=1_048_576)


def test_valid_all_string_detection_integrated(tmp_path: Path) -> None:
    result, accumulators, issues = inspect(tmp_path, object_payload())
    assert not issues
    assert result["file_normalization_status"] == "VALID"
    assert accumulators["coordinate"]["valid"] == 4


def test_mixed_native_number_and_string_detection(tmp_path: Path) -> None:
    value = object_payload(coordinate=0.1, score=0.9, label=84)
    result, _, _ = inspect(tmp_path, value)
    assert result["valid_detection_count"] == 1


@pytest.mark.parametrize(
    ("field", "value", "code"),
    [
        ("coordinate", "bad", "COORDINATE_INVALID_FORMAT"),
        ("score", "bad", "SCORE_INVALID_FORMAT"),
        ("label", "84.0", "LABEL_INVALID_FORMAT"),
    ],
)
def test_invalid_numeric_detection(tmp_path: Path, field: str, value, code: str) -> None:
    kwargs = {field: value}
    result, _, issues = inspect(tmp_path, object_payload(**kwargs))
    assert result["invalid_detection_count"] == 1
    assert code in {item["code"] for item in issues}
    assert issues[0]["raw_value"] == value


def test_native_non_integer_float_label_issue(tmp_path: Path) -> None:
    _, _, issues = inspect(tmp_path, object_payload(label=2.5))
    assert "LABEL_NOT_INTEGER" in {item["code"] for item in issues}


def test_parallel_length_mismatch_is_not_truncated(tmp_path: Path) -> None:
    value = object_payload(count=2)
    value["detection_scores"].pop()
    result, accumulators, issues = inspect(tmp_path, value)
    assert result["detection_count"] == 0
    assert accumulators["coordinate"]["total"] == 0
    assert issues[0]["code"] == "OBJECT_PARALLEL_ARRAY_LENGTH_MISMATCH"


def test_no_silent_defaulting_or_dropping(tmp_path: Path) -> None:
    result, _, issues = inspect(tmp_path, object_payload(score=None))
    assert result["detection_count"] == 1
    assert result["valid_detection_count"] == 0
    assert result["invalid_detection_count"] == 1
    assert issues[0]["raw_value"] is None


def make_full_fixture(root: Path, *, count: int = 2) -> None:
    for _, _, path in _object_targets(root):
        write_object(path, object_payload(count=count))


def test_aggregate_min_max_and_per_position(tmp_path: Path) -> None:
    make_full_fixture(tmp_path)
    result = run_survey(tmp_path)
    coordinates = result.summary["coordinate_summary"]
    assert coordinates["min"] == 0.1
    assert coordinates["max"] == 0.4
    assert coordinates["per_position_min"] == [0.1, 0.2, 0.3, 0.4]
    assert coordinates["per_position_max"] == [0.1, 0.2, 0.3, 0.4]
    assert result.summary["score_summary"]["min"] == 0.9
    assert result.summary["label_summary"]["distinct_count"] == 1


def test_fixture_below_locked_detection_count_is_blocked(tmp_path: Path) -> None:
    make_full_fixture(tmp_path, count=2)
    result = run_survey(tmp_path)
    assert result.summary["readiness"]["status"] == "NUMERIC_CONTRACT_BLOCKED"


def test_full_locked_shape_is_ready(tmp_path: Path) -> None:
    make_full_fixture(tmp_path, count=100)
    result = run_survey(tmp_path)
    assert result.summary["files_inspected"] == 15
    assert result.summary["detections_observed"] == 1500
    assert result.summary["coordinate_summary"]["total_values"] == 6000
    assert result.summary["readiness"]["status"] == "READY_FOR_STAGE_0_DATA_AUDIT"


def test_zip_exact_members_and_excludes_itself(tmp_path: Path) -> None:
    data = tmp_path / "data"
    make_full_fixture(data, count=100)
    paths = write_outputs(run_survey(data), tmp_path / "output")
    with ZipFile(paths["zip"]) as archive:
        assert set(archive.namelist()) == set(OUTPUT_NAMES)
        assert "object_numeric_contract_v022.zip" not in archive.namelist()


def test_strict_root_protection(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        validate_root(tmp_path, strict_root=True)


def test_limit_cannot_exceed_fifteen() -> None:
    with pytest.raises(ValueError):
        NumericLimits(max_object_json_total=16)


def test_patch_does_not_call_id_scan(tmp_path: Path, monkeypatch) -> None:
    make_full_fixture(tmp_path)
    import triage_eg.data.cross_asset_survey as old

    monkeypatch.setattr(old, "collect_asset_paths", lambda *args, **kwargs: pytest.fail("scan"))
    assert run_survey(tmp_path).summary["id_set_scan_rerun"] is False


def test_cli_smoke_on_temporary_fixture(tmp_path: Path) -> None:
    make_full_fixture(tmp_path)
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/survey_object_numeric_contract_v022.py",
            "--dataset-root",
            str(tmp_path),
            "--no-write",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert '"survey_version": "0.2.2"' in completed.stdout
