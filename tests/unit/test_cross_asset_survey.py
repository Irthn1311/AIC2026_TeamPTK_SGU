from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from triage_eg.data.cross_asset_survey import (
    CrossAssetLimits,
    choose_samples,
    collect_asset_paths,
    compare_id_sets,
    filename_hypothesis,
    inspect_clip,
    inspect_object_json,
    parse_mapping,
    survey_cross_assets,
    validate_root,
    write_outputs,
)


def _mapping(path: Path, rows: list[tuple[int, float, float, int]]) -> None:
    path.write_text(
        "n,pts_time,fps,frame_idx\n"
        + "".join(f"{n},{pts},{fps},{frame}\n" for n, pts, fps, frame in rows),
        encoding="utf-8",
    )


def _dataset(tmp_path: Path, *, mismatch: bool = False) -> Path:
    root = tmp_path / "dataset"
    mapping_root = root / "map-keyframes-aic25-b1" / "map-keyframes"
    clip_root = root / "clip-features-32-aic25-b1" / "clip-features-32"
    metadata_root = root / "media-info-aic25-b1" / "media-info"
    object_root = root / "objects-aic25-b1" / "objects"
    keyframe_root = root / "keyframes" / "keyframes"
    for directory in (mapping_root, clip_root, metadata_root, object_root, keyframe_root):
        directory.mkdir(parents=True, exist_ok=True)
    ids = ["L21_V001", "L24_V002", "L28_V003"]
    for video_id in ids:
        partition_name = video_id.split("_", 1)[0]
        video_root = root / f"Videos_{partition_name}_a" / "video"
        video_root.mkdir(parents=True, exist_ok=True)
        (video_root / f"{video_id}.mp4").write_bytes(b"fake")
        rows = [(1, 0.0, 25.0, 0), (2, 0.04, 25.0, 1), (3, 0.04, 25.0, 1)]
        _mapping(mapping_root / f"{video_id}.csv", rows)
        np.save(clip_root / f"{video_id}.npy", np.ones((3, 4), dtype=np.float16))
        (metadata_root / f"{video_id}.json").write_text("{}", encoding="utf-8")
        objects = object_root / video_id
        objects.mkdir()
        keyframes = keyframe_root / f"Keyframes_{partition_name}" / "keyframes" / video_id
        keyframes.mkdir(parents=True)
        for n in (1, 2, 3):
            (keyframes / f"{n:03d}.jpg").write_bytes(b"")
            payload = {"detections": [{"class": "person", "score": 0.9, "bbox": [0, 1, 2, 3]}]}
            (objects / f"{n:03d}.json").write_text(json.dumps(payload), encoding="utf-8")
    if mismatch:
        (metadata_root / "L28_V003.json").unlink()
    return root


def test_exact_id_set_equality(tmp_path: Path) -> None:
    assets, _, _ = collect_asset_paths(_dataset(tmp_path))
    comparison = compare_id_sets(assets)
    assert comparison["all_equal"] is True
    assert comparison["intersection_count"] == 3
    assert comparison["object_coverage"]["covers_all_videos"] is True


def test_id_set_mismatch(tmp_path: Path) -> None:
    assets, _, _ = collect_asset_paths(_dataset(tmp_path, mismatch=True))
    comparison = compare_id_sets(assets)
    assert comparison["all_equal"] is False
    assert comparison["missing_from"]["metadata"] == ["L28_V003"]


def test_invalid_video_id_pattern_is_reported(tmp_path: Path) -> None:
    root = _dataset(tmp_path)
    bad = root / "map-keyframes-aic25-b1" / "map-keyframes" / "bad.csv"
    _mapping(bad, [(1, 0.0, 25.0, 0)])
    _, _, issues = collect_asset_paths(root)
    assert any(item["code"] == "INVALID_VIDEO_ID_PATTERN" for item in issues)


def test_mapping_parser_and_duplicate_detection(tmp_path: Path) -> None:
    path = tmp_path / "mapping.csv"
    _mapping(path, [(1, 0.0, 25.0, 0), (2, 0.04, 25.0, 1), (3, 0.04, 25.0, 1)])
    report = parse_mapping(path, 10)
    assert report["row_count"] == 3
    assert report["n_monotonic"] is True
    assert report["duplicate_frame_idx_groups"][0]["n_values"] == [2, 3]


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([1, 2, 3], "CSV_N"),
        ([0, 1, 2], "CSV_N_MINUS_ONE"),
        ([0, 4, 8], "FRAME_IDX"),
        ([7, 9, 11], "NO_SIMPLE_RELATION"),
    ],
)
def test_filename_hypotheses(values: list[int], expected: str, tmp_path: Path) -> None:
    path = tmp_path / "mapping.csv"
    _mapping(path, [(1, 0.0, 25.0, 0), (2, 0.16, 25.0, 4), (3, 0.32, 25.0, 8)])
    assert filename_hypothesis(values, parse_mapping(path, 10))["hypothesis"] == expected


def test_npy_row_count_uses_mmap_contract(tmp_path: Path) -> None:
    path = tmp_path / "features.npy"
    np.save(path, np.ones((7, 512), dtype=np.float16))
    assert inspect_clip(path) == {
        "shape": [7, 512],
        "ndim": 2,
        "row_count": 7,
        "dimension": 512,
        "dtype": "float16",
    }


def test_object_schema_and_bounded_read(tmp_path: Path) -> None:
    path = tmp_path / "001.json"
    path.write_text(
        json.dumps({"detections": [{"label": "car", "confidence": 0.8, "bbox": [1, 2, 3, 4]}]}),
        encoding="utf-8",
    )
    report = inspect_object_json(path, 1024)
    assert report["detection_count"] == 1
    assert report["candidate_bbox_fields"] == ["bbox"]
    assert report["bbox_order"] == "UNKNOWN"
    with pytest.raises(OverflowError):
        inspect_object_json(path, 1)


def test_empty_and_malformed_object_json(tmp_path: Path) -> None:
    empty = tmp_path / "empty.json"
    empty.write_text("[]", encoding="utf-8")
    assert inspect_object_json(empty, 100)["is_empty"] is True
    malformed = tmp_path / "bad.json"
    malformed.write_text("{", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        inspect_object_json(malformed, 100)


def test_cross_asset_count_mismatch_and_duplicate_case(tmp_path: Path) -> None:
    root = _dataset(tmp_path)
    (
        root / "keyframes" / "keyframes" / "Keyframes_L24" / "keyframes" / "L24_V002" / "003.jpg"
    ).unlink()
    result = survey_cross_assets(
        root,
        limits=CrossAssetLimits(max_videos=3),
        video_ids=["L21_V001", "L24_V002", "L28_V003"],
    )
    record = next(item for item in result.records if item["video_id"] == "L24_V002")
    assert record["mapping_equals_keyframes"] is False
    assert result.summary["duplicate_frame_idx_case_studies"]


def test_deterministic_sample_selection() -> None:
    ids = {"L21_V001", "L22_V001", "L23_V001", "L24_V001", "L25_V001"}
    first, _ = choose_samples(ids, requested=None, maximum=3, seed=2026)
    second, _ = choose_samples(ids, requested=None, maximum=3, seed=2026)
    assert first == second
    assert len(first) == 3


def test_output_serialization(tmp_path: Path) -> None:
    result = survey_cross_assets(
        _dataset(tmp_path),
        limits=CrossAssetLimits(max_videos=3),
        video_ids=["L21_V001", "L24_V002", "L28_V003"],
    )
    paths = write_outputs(result, tmp_path / "output")
    assert {path.name for path in paths.values()} == {
        "cross_asset_survey_v02.json",
        "cross_asset_survey_v02.md",
        "cross_asset_records.jsonl",
        "object_schema_samples.jsonl",
        "issues.jsonl",
    }
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    assert summary["survey_version"] == "0.2.0"


def test_strict_root_protection(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Strict dataset root"):
        validate_root(_dataset(tmp_path), strict_root=True)
