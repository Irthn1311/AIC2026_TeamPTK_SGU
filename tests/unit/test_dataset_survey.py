from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from triage_eg.data.dataset_survey import (
    SurveyLimits,
    TraversalBudget,
    bounded_tree,
    candidate_video_id,
    classify_asset,
    inspect_csv,
    inspect_json,
    inspect_npy,
    survey_dataset,
    validate_dataset_root,
    write_survey_outputs,
)


def test_root_validation(tmp_path: Path) -> None:
    assert validate_dataset_root(tmp_path) == tmp_path.resolve()
    with pytest.raises(FileNotFoundError):
        validate_dataset_root(tmp_path / "missing")
    file_path = tmp_path / "file"
    file_path.write_text("x", encoding="utf-8")
    with pytest.raises(NotADirectoryError):
        validate_dataset_root(file_path)

    missing_result = survey_dataset(tmp_path / "not-attached")
    assert missing_result.summary["root_exists"] is False
    assert missing_result.summary["sample_issues"][0]["code"] == "ROOT_NOT_FOUND"


def test_bounded_tree_limits_depth_and_listing(tmp_path: Path) -> None:
    (tmp_path / "a" / "b").mkdir(parents=True)
    for index in range(4):
        (tmp_path / f"file_{index}.txt").write_text("x", encoding="utf-8")
    limits = SurveyLimits(max_depth=1, max_listed_per_directory=2)
    tree = bounded_tree(tmp_path, limits, TraversalBudget(100))[0]
    assert len(tree["entries"]) == 2
    assert tree["hidden_entries"] >= 1
    directory = next(entry for entry in tree["entries"] if entry["type"] == "directory")
    assert directory["truncated"] is True


@pytest.mark.parametrize(("path", "expected"), [
    ("videos/L01_V001.mp4", "VIDEO"),
    ("keyframes/L01_V001/000001.jpg", "KEYFRAME_IMAGE"),
    ("map-keyframes/L01_V001.csv", "KEYFRAME_MAPPING"),
    ("clip-features/L01_V001.npy", "CLIP_FEATURE"),
    ("objects/L01_V001.json", "OBJECT_JSON"),
    ("metadata/L01_V001.json", "METADATA"),
    ("misc/readme.txt", "UNKNOWN"),
])
def test_asset_classification(path: str, expected: str) -> None:
    assert classify_asset(path).asset_type == expected


def test_csv_header_and_row_limit(tmp_path: Path) -> None:
    path = tmp_path / "mapping.csv"
    path.write_text("n,pts_time,frame_idx\n0,0.0,0\n1,0.4,10\n2,0.8,20\n", encoding="utf-8")
    report = inspect_csv(path, max_rows=2)
    assert report["columns"] == ["n", "pts_time", "frame_idx"]
    assert report["rows_inspected"] == 2


def test_npy_is_inspected_with_bounded_rows(tmp_path: Path) -> None:
    path = tmp_path / "clip.npy"
    np.save(path, np.ones((10, 4), dtype=np.float32))
    report = inspect_npy(path, max_rows=3, seed=2026)
    assert report["shape"] == (10, 4)
    assert report["rows_inspected"] == 3


def test_json_bounded_inspection(tmp_path: Path) -> None:
    path = tmp_path / "objects.json"
    path.write_text(json.dumps({"objects": [{"bbox": [1, 2, 3, 4]}]}), encoding="utf-8")
    assert inspect_json(path, 1_024)["inspection_status"] == "INSPECTED"
    assert inspect_json(path, 1)["reason"] == "FILE_TOO_LARGE_TO_INSPECT"


def test_candidate_video_id_is_conservative() -> None:
    assert candidate_video_id("L01_V001_clip.npy") == "L01_V001"
    assert candidate_video_id("L01_V001.mp4") == "L01_V001"
    assert candidate_video_id("keyframes/L01_V001/000001.jpg") == "L01_V001"


def test_output_serialization_creates_only_expected_files(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "L01_V001.mp4").write_bytes(b"not-decoded")
    result = survey_dataset(dataset, limits=SurveyLimits(max_stat_operations=50))
    paths = write_survey_outputs(result, tmp_path / "output")
    assert {path.name for path in paths.values()} == {
        "dataset_survey.json", "dataset_survey.md", "sample_inventory.jsonl"
    }
    assert json.loads(paths["json"].read_text(encoding="utf-8"))["root_exists"] is True


def test_symlink_is_not_followed(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    root = tmp_path / "dataset"
    root.mkdir()
    link = root / "escape"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("Symlinks are unavailable in this environment")
    tree = bounded_tree(root, SurveyLimits(), TraversalBudget(20))[0]
    assert tree["entries"] == [{"name": "escape", "type": "symlink", "followed": False}]


def test_stat_limit_is_enforced(tmp_path: Path) -> None:
    for index in range(10):
        (tmp_path / f"{index}.mp4").write_bytes(b"x")
    result = survey_dataset(tmp_path, limits=SurveyLimits(max_stat_operations=3))
    assert result.summary["stat_operations"]["total"] == 3
    assert any(
        item["code"] == "INSPECTION_LIMIT_REACHED"
        for item in result.summary["sample_issues"]
    )
