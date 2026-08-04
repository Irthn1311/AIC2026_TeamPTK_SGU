from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from triage_eg.data.stage0_audit.asset_resolver import resolve_assets
from triage_eg.data.stage0_audit.auditors import (
    audit_clip,
    audit_keyframes,
    audit_mapping,
    audit_metadata,
    audit_objects,
    probe_video,
)
from triage_eg.data.stage0_audit.contracts import AuditConfig
from triage_eg.data.stage0_audit.runner import run_audit
from triage_eg.data.stage0_audit.writers import FINAL_ARTIFACTS, create_bundle, write_jsonl


def assets(root: Path):
    return resolve_assets(
        root, "L01_V001", {"L01_V001": "Videos_L01"}, {"L01_V001": "Keyframes_L01"}
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mode": "bad"},
        {"clip_validation": "bad"},
        {"object_validation": "bad"},
        {"sample_size": 0},
        {"max_object_json_bytes": 0},
        {"ffprobe_timeout_seconds": 0},
        {"workers": 2},
        {"resume": True, "overwrite": True},
    ],
)
def test_invalid_config_is_fail_closed(tmp_path: Path, kwargs) -> None:
    with pytest.raises(ValueError):
        AuditConfig(dataset_root=tmp_path, output_root=tmp_path / "out", **kwargs)


def test_missing_mapping(tmp_path: Path) -> None:
    _, rows, issues = audit_mapping(tmp_path / "missing.csv", "L01_V001")
    assert not rows and issues[0].code == "MISSING_MAPPING"


def test_missing_keyframe_directory(tmp_path: Path) -> None:
    _, _, issues = audit_keyframes(tmp_path, assets(tmp_path), {1})
    assert "MISSING_KEYFRAME_DIRECTORY" in {item.code for item in issues}


def test_missing_clip(tmp_path: Path) -> None:
    _, issues = audit_clip(tmp_path, assets(tmp_path), 1, expected_dimension=512, mode="shape")
    assert issues[0].code == "MISSING_CLIP"


def test_missing_object_directory(tmp_path: Path) -> None:
    _, issues = audit_objects(tmp_path, assets(tmp_path), {1}, mode="full", max_bytes=100)
    assert "MISSING_OBJECT_DIRECTORY" in {item.code for item in issues}


def test_missing_metadata_is_warning(tmp_path: Path) -> None:
    _, issues = audit_metadata(tmp_path, assets(tmp_path))
    assert issues[0].code == "MISSING_METADATA" and issues[0].severity == "WARNING"


def test_missing_video_only_blocks_raw(tmp_path: Path) -> None:
    _, issues = probe_video(tmp_path, assets(tmp_path), timeout=1)
    assert issues[0].code == "MISSING_VIDEO"
    assert issues[0].blocks_raw_video_pipeline and not issues[0].blocks_btc_baseline


@pytest.mark.parametrize(
    ("content", "code"),
    [
        ("{", "OBJECT_JSON_MALFORMED"),
        ("[]", "OBJECT_TOP_LEVEL_INVALID"),
        (json.dumps({}), "OBJECT_FIELD_MISSING"),
        (
            json.dumps(
                {
                    "detection_boxes": {},
                    "detection_class_entities": [],
                    "detection_class_labels": [],
                    "detection_class_names": [],
                    "detection_scores": [],
                }
            ),
            "OBJECT_FIELD_TYPE_MISMATCH",
        ),
        (
            json.dumps(
                {
                    "detection_boxes": [["0.1"]],
                    "detection_class_entities": ["x"],
                    "detection_class_labels": ["1"],
                    "detection_class_names": ["x"],
                    "detection_scores": ["0.9"],
                }
            ),
            "OBJECT_BOX_LENGTH_INVALID",
        ),
    ],
)
def test_object_schema_failures(tmp_path: Path, content: str, code: str) -> None:
    asset = assets(tmp_path)
    path = asset.object_directory / "001.json"
    path.parent.mkdir(parents=True)
    path.write_text(content, encoding="utf-8")
    _, issues = audit_objects(tmp_path, asset, {1}, mode="full", max_bytes=10000)
    assert code in {item.code for item in issues}


@pytest.mark.parametrize(
    ("content", "code"), [("{", "METADATA_MALFORMED"), ("[]", "METADATA_TOP_LEVEL_INVALID")]
)
def test_metadata_malformed_or_top_level(tmp_path: Path, content: str, code: str) -> None:
    asset = assets(tmp_path)
    asset.metadata.parent.mkdir(parents=True)
    asset.metadata.write_text(content, encoding="utf-8")
    _, issues = audit_metadata(tmp_path, asset)
    assert issues[0].code == code


def test_numpy_scalars_are_json_serialized(tmp_path: Path) -> None:
    path = tmp_path / "values.jsonl"
    write_jsonl(path, [{"value": np.int64(7)}])
    assert json.loads(path.read_text(encoding="utf-8"))["value"] == 7


def test_empty_issue_jsonl_is_valid(tmp_path: Path) -> None:
    path = tmp_path / "issues.jsonl"
    write_jsonl(path, [])
    assert path.read_text(encoding="utf-8") == ""


def test_bundle_rejects_inside_output_root(tmp_path: Path) -> None:
    for name in FINAL_ARTIFACTS:
        (tmp_path / name).write_text("x", encoding="utf-8")
    with pytest.raises(ValueError):
        create_bundle(tmp_path, tmp_path / "bundle.zip")


def test_nonempty_output_requires_explicit_policy(tmp_path: Path) -> None:
    data, output = tmp_path / "data", tmp_path / "out"
    data.mkdir()
    output.mkdir()
    (output / "existing").write_text("x", encoding="utf-8")
    with pytest.raises(FileExistsError):
        run_audit(AuditConfig(dataset_root=data, output_root=output))


def test_strict_root_rejects_local_dataset(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    with pytest.raises(ValueError):
        run_audit(AuditConfig(dataset_root=data, output_root=tmp_path / "out", strict_root=True))


def test_stage0_source_has_no_frame_extraction_or_opencv() -> None:
    package = Path("src/triage_eg/data/stage0_audit")
    source = "\n".join(path.read_text(encoding="utf-8") for path in package.glob("*.py"))
    assert "import cv2" not in source
    assert "VideoCapture" not in source
    assert "extract_frame" not in source
