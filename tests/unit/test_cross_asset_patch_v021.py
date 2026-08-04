from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest

from triage_eg.data.cross_asset_patch_v021 import (
    DUPLICATE_GROUPS,
    EXPECTED_FIELDS,
    OUTPUT_NAMES,
    SAMPLE_PARTITIONS,
    PatchLimits,
    aggregate_object_schema,
    inspect_parallel_object,
    resolve_duplicate_cases,
    run_patch,
    validate_root,
    write_outputs,
)


def payload(count: int = 1) -> dict[str, list[object]]:
    return {
        "detection_boxes": [[0.1, 0.2, 0.3, 0.4] for _ in range(count)],
        "detection_class_entities": ["person" for _ in range(count)],
        "detection_class_labels": [1 for _ in range(count)],
        "detection_class_names": ["person" for _ in range(count)],
        "detection_scores": [0.9 for _ in range(count)],
    }


def inspect(tmp_path: Path, value: object, *, limit: int = 1_048_576):
    path = tmp_path / "001.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return inspect_parallel_object(
        path, video_id="L21_V017", ordinal_n=1, max_bytes=limit, max_boxes=20
    )


def test_valid_equal_length_parallel_arrays(tmp_path: Path) -> None:
    sample, issues = inspect(tmp_path, payload(2))
    assert sample["parallel_arrays_valid"] and not issues
    assert sample["detection_count"] == 2


def test_five_empty_arrays(tmp_path: Path) -> None:
    sample, _ = inspect(tmp_path, payload(0))
    assert sample["empty_detection_representation"] == "FIVE_EMPTY_PARALLEL_ARRAYS"
    assert sample["is_empty_detection"]


@pytest.mark.parametrize(
    ("mutation", "code"),
    [
        (lambda value: value.pop("detection_scores"), "OBJECT_FIELD_MISSING"),
        (lambda value: value.update(detection_scores="bad"), "OBJECT_FIELD_TYPE_MISMATCH"),
        (
            lambda value: value["detection_scores"].append(0.8),
            "OBJECT_PARALLEL_ARRAY_LENGTH_MISMATCH",
        ),
    ],
)
def test_parallel_array_failures(tmp_path: Path, mutation, code: str) -> None:
    value = payload()
    mutation(value)
    sample, issues = inspect(tmp_path, value)
    assert not sample["parallel_arrays_valid"]
    assert code in {item["code"] for item in issues}


def test_box_shape_four_and_normalized_scale(tmp_path: Path) -> None:
    sample, _ = inspect(tmp_path, payload())
    bbox = sample["bbox_observation"]
    assert bbox["box_length_distribution"] == {"4": 1}
    assert bbox["coordinate_scale_hypothesis"] == "NORMALIZED_0_1"
    assert bbox["bbox_order"] == "UNKNOWN"


@pytest.mark.parametrize(
    ("box", "code"),
    [
        ([1, 2, 3], "OBJECT_BOX_LENGTH_UNEXPECTED"),
        ([1, "x", 3, 4], "OBJECT_COORDINATE_NON_NUMERIC"),
    ],
)
def test_bad_box_observations(tmp_path: Path, box: list[object], code: str) -> None:
    value = payload()
    value["detection_boxes"] = [box]
    _, issues = inspect(tmp_path, value)
    assert code in {item["code"] for item in issues}


def test_score_numeric_summary(tmp_path: Path) -> None:
    sample, _ = inspect(tmp_path, payload())
    score = sample["field_observations"]["detection_scores"]
    assert score["numeric_min"] == 0.9
    assert score["score_range_hypothesis"] == "PROBABILITY_LIKE_0_1"


def test_non_numeric_score_issue(tmp_path: Path) -> None:
    value = payload()
    value["detection_scores"] = ["high"]
    _, issues = inspect(tmp_path, value)
    assert "OBJECT_SCORE_NON_NUMERIC" in {item["code"] for item in issues}


def test_malformed_json(tmp_path: Path) -> None:
    path = tmp_path / "001.json"
    path.write_text("{", encoding="utf-8")
    sample, _ = inspect_parallel_object(
        path, video_id="L21_V017", ordinal_n=1, max_bytes=20, max_boxes=2
    )
    assert sample["inspection_status"] == "MALFORMED"


def test_file_larger_than_limit(tmp_path: Path) -> None:
    sample, _ = inspect(tmp_path, payload(), limit=2)
    assert sample["inspection_status"] == "TOO_LARGE"


def test_extra_top_level_field(tmp_path: Path) -> None:
    value = payload()
    value["extra"] = []
    sample, _ = inspect(tmp_path, value)
    assert sample["extra_fields"] == ["extra"]


def test_aggregate_schema_summary(tmp_path: Path) -> None:
    one, _ = inspect(tmp_path, payload(2))
    summary = aggregate_object_schema([one])
    assert summary["files_inspected"] == 1
    assert summary["detection_count_total_in_sample"] == 2
    assert summary["score_item_types"] == {"float": 2}


def make_fixture(
    root: Path,
    *,
    missing_keyframe: tuple[str, int] | None = None,
    missing_object: tuple[str, int] | None = None,
    clip_rows: int = 2000,
) -> None:
    object_root = root / "objects-aic25-b1" / "objects"
    for video_id in SAMPLE_PARTITIONS:
        targets = {1}
        for group in DUPLICATE_GROUPS.get(video_id, ()):
            targets.update(group["n_values"])
        for ordinal in targets:
            path = object_root / video_id / f"{ordinal:03d}.json"
            if missing_object == (video_id, ordinal):
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload()), encoding="utf-8")
    for video_id, groups in DUPLICATE_GROUPS.items():
        key_dir = (
            root / "keyframes" / "keyframes" / SAMPLE_PARTITIONS[video_id] / "keyframes" / video_id
        )
        key_dir.mkdir(parents=True, exist_ok=True)
        for group in groups:
            for ordinal in group["n_values"]:
                if missing_keyframe != (video_id, ordinal):
                    (key_dir / f"{ordinal:03d}.jpg").write_bytes(b"fake")
        clip = root / "clip-features-32-aic25-b1" / "clip-features-32" / f"{video_id}.npy"
        clip.parent.mkdir(parents=True, exist_ok=True)
        np.save(clip, np.zeros((clip_rows, 1), dtype=np.float16))


def test_zero_padded_lookup_row_index_and_metadata(tmp_path: Path) -> None:
    make_fixture(tmp_path)
    cases, issues = resolve_duplicate_cases(tmp_path)
    asset = cases[0]["related_assets"][0]
    assert not issues
    assert asset["keyframe"]["path"].endswith("019.jpg")
    assert asset["object"]["path"].endswith("019.json")
    assert asset["clip"]["row_index"] == 18
    assert cases[0]["keyframe_sizes_equal"]


@pytest.mark.parametrize(
    ("missing_keyframe", "missing_object", "clip_rows", "code"),
    [
        (("L28_V019", 19), None, 2000, "MISSING_DUPLICATE_KEYFRAME"),
        (None, ("L28_V019", 19), 2000, "MISSING_DUPLICATE_OBJECT_JSON"),
        (None, None, 10, "CLIP_ROW_OUT_OF_RANGE"),
    ],
)
def test_duplicate_missing_assets(
    tmp_path: Path, missing_keyframe, missing_object, clip_rows: int, code: str
) -> None:
    make_fixture(
        tmp_path,
        missing_keyframe=missing_keyframe,
        missing_object=missing_object,
        clip_rows=clip_rows,
    )
    _, issues = resolve_duplicate_cases(tmp_path)
    assert code in {item["code"] for item in issues}


def test_different_keyframe_sizes_are_metadata_only(tmp_path: Path) -> None:
    make_fixture(tmp_path)
    key = tmp_path / "keyframes/keyframes/Keyframes_L28/keyframes/L28_V019/020.jpg"
    key.write_bytes(b"different")
    cases, _ = resolve_duplicate_cases(tmp_path)
    assert cases[0]["keyframe_sizes_equal"] is False
    assert cases[0]["classification"] == "DUPLICATE_MAPPING_PRESERVED"


def test_outputs_jsonl_markdown_and_zip(tmp_path: Path) -> None:
    dataset = tmp_path / "data"
    make_fixture(dataset)
    result = run_patch(dataset)
    paths = write_outputs(result, tmp_path / "out")
    assert paths["object_schema_samples_v021.jsonl"].read_text(encoding="utf-8").count("\n") == 15
    assert (tmp_path / "out/patch_report_v021.md").is_file()
    with ZipFile(paths["zip"]) as archive:
        assert set(archive.namelist()) == set(OUTPUT_NAMES)
        assert "cross_asset_survey_v021.zip" not in archive.namelist()


def test_strict_root_protection(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        validate_root(tmp_path, strict_root=True)


def test_patch_is_bounded_and_does_not_call_id_scan(tmp_path: Path, monkeypatch) -> None:
    make_fixture(tmp_path)
    import triage_eg.data.cross_asset_survey as old

    monkeypatch.setattr(
        old, "collect_asset_paths", lambda *args, **kwargs: pytest.fail("ID scan called")
    )
    result = run_patch(tmp_path, limits=PatchLimits(max_object_json_total=15))
    assert len(result.object_samples) == 15
    assert result.summary["id_set_scan_rerun"] is False
    assert result.summary["readiness"]["status"] == "READY_FOR_STAGE_0_IMPLEMENTATION_PLANNING"


def test_limit_cannot_exceed_fifteen() -> None:
    with pytest.raises(ValueError):
        PatchLimits(max_object_json_total=16)


def test_expected_fields_are_exact() -> None:
    assert len(EXPECTED_FIELDS) == 5


def test_cli_smoke_on_temporary_fixture(tmp_path: Path) -> None:
    make_fixture(tmp_path)
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/survey_cross_assets_v021.py",
            "--dataset-root",
            str(tmp_path),
            "--no-write",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert '"survey_version": "0.2.1"' in completed.stdout
