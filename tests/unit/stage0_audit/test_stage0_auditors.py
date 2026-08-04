from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from triage_eg.data.stage0_audit.asset_resolver import (
    discover_layout,
    resolve_assets,
    validate_video_id,
)
from triage_eg.data.stage0_audit.auditors import (
    audit_clip,
    audit_keyframes,
    audit_mapping,
    audit_metadata,
    audit_objects,
    probe_video,
)


def paths(root: Path, video_id: str = "L01_V001"):
    return resolve_assets(root, video_id, {video_id: "Videos_L01"}, {video_id: "Keyframes_L01"})


def mapping(path: Path, rows=None, columns="n,pts_time,fps,frame_idx") -> None:
    rows = rows or [(1, 0.0, 25.0, 0), (2, 0.04, 25.0, 1)]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        columns + "\n" + "".join(",".join(map(str, row)) + "\n" for row in rows), encoding="utf-8"
    )


def test_resolve_partitions_and_all_paths(tmp_path: Path) -> None:
    video = tmp_path / "Videos_L01/video/L01_V001.mp4"
    key = tmp_path / "keyframes/keyframes/Keyframes_L01/keyframes/L01_V001"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"x")
    key.mkdir(parents=True)
    videos, keys = discover_layout(tmp_path)
    resolved = resolve_assets(tmp_path, "L01_V001", videos, keys)
    assert resolved.video == video
    assert resolved.keyframe_directory == key
    assert resolved.mapping.name == "L01_V001.csv"
    assert resolved.clip.name == "L01_V001.npy"
    assert resolved.object_directory.name == "L01_V001"
    assert resolved.metadata.name == "L01_V001.json"


@pytest.mark.parametrize("value", ["bad", "L01-V001", "../L01_V001", ""])
def test_invalid_video_id(value: str) -> None:
    with pytest.raises(ValueError):
        validate_video_id(value)


def test_valid_mapping_preserves_authoritative_frame_idx(tmp_path: Path) -> None:
    path = tmp_path / "map.csv"
    mapping(path, [(1, 99.0, 25.0, 7)])
    record, rows, issues = audit_mapping(path, "L01_V001")
    assert record["status"] == "VALID"
    assert rows[0]["frame_idx"] == 7
    assert not issues


@pytest.mark.parametrize(
    ("rows", "code"),
    [
        ([(2, 0.0, 25, 0)], "MAPPING_N_INVALID"),
        ([(1, 0.0, 25, 0), (3, 0.1, 25, 2)], "MAPPING_N_NON_CONTIGUOUS"),
        ([(1, 0.1, 25, 1), (2, 0.0, 25, 2)], "MAPPING_PTS_NON_MONOTONIC"),
        ([(1, 0.0, 25, 2), (2, 0.1, 25, 1)], "MAPPING_FRAME_IDX_NON_MONOTONIC"),
        ([(1, 0.0, 0, 0)], "MAPPING_FPS_INVALID"),
    ],
)
def test_mapping_validation(tmp_path: Path, rows, code: str) -> None:
    path = tmp_path / "map.csv"
    mapping(path, rows)
    _, _, issues = audit_mapping(path, "L01_V001")
    assert code in {item.code for item in issues}


def test_mapping_missing_column(tmp_path: Path) -> None:
    path = tmp_path / "map.csv"
    mapping(path, columns="n,pts_time,fps")
    _, _, issues = audit_mapping(path, "L01_V001")
    assert issues[0].code == "MAPPING_COLUMN_MISMATCH"


def test_duplicate_frame_idx_is_preserved_warning(tmp_path: Path) -> None:
    path = tmp_path / "map.csv"
    mapping(path, [(1, 0, 25, 7), (2, 0.1, 25, 7)])
    record, rows, issues = audit_mapping(path, "L01_V001")
    assert len(rows) == 2
    assert record["duplicate_groups"] == {7: [1, 2]}
    duplicate = next(item for item in issues if item.code == "DUPLICATE_FRAME_IDX")
    assert duplicate.severity == "WARNING" and not duplicate.blocks_btc_baseline


def write_jpg(path: Path, body: bytes = b"ok") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\xff\xd8" + body + b"\xff\xd9")


def test_keyframe_exact_ordinals_header_only(tmp_path: Path) -> None:
    asset = paths(tmp_path)
    write_jpg(asset.keyframe_directory / "001.jpg")
    write_jpg(asset.keyframe_directory / "002.jpg")
    record, details, issues = audit_keyframes(tmp_path, asset, {1, 2})
    assert record["ordinal_set_matches_mapping"]
    assert details[1]["header_valid"]
    assert not issues


@pytest.mark.parametrize(
    ("setup", "expected", "code"),
    [
        (lambda directory: write_jpg(directory / "001.jpg"), {1, 2}, "KEYFRAME_MISSING"),
        (
            lambda directory: (write_jpg(directory / "001.jpg"), write_jpg(directory / "002.jpg")),
            {1},
            "KEYFRAME_EXTRA",
        ),
        (lambda directory: write_jpg(directory / "bad.jpg"), set(), "KEYFRAME_FILENAME_INVALID"),
        (
            lambda directory: (
                directory.mkdir(parents=True),
                (directory / "001.jpg").write_bytes(b""),
            ),
            {1},
            "KEYFRAME_FILE_EMPTY",
        ),
        (
            lambda directory: (
                directory.mkdir(parents=True),
                (directory / "001.jpg").write_bytes(b"xxxx"),
            ),
            {1},
            "KEYFRAME_HEADER_INVALID",
        ),
    ],
)
def test_keyframe_issues(tmp_path: Path, setup, expected: set[int], code: str) -> None:
    asset = paths(tmp_path)
    setup(asset.keyframe_directory)
    _, _, issues = audit_keyframes(tmp_path, asset, expected)
    assert code in {item.code for item in issues}


def save_clip(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)


def test_clip_full_mmap_norms(tmp_path: Path) -> None:
    asset = paths(tmp_path)
    save_clip(asset.clip, np.array([[3, 4, 0, 0], [0, 0, 0, 0]], dtype=np.float32))
    record, issues = audit_clip(tmp_path, asset, 2, expected_dimension=4, mode="full", chunk_rows=1)
    assert not issues
    assert record["norm_min"] == 0 and record["norm_max"] == 5
    assert record["norm_mean"] == 2.5


@pytest.mark.parametrize(
    ("array", "mapping_count", "dimension", "code"),
    [
        (np.zeros((2, 4)), 3, 4, "CLIP_ROW_COUNT_MISMATCH"),
        (np.zeros((2, 3)), 2, 4, "CLIP_DIMENSION_MISMATCH"),
        (np.zeros(4), 4, 4, "CLIP_NDIM_MISMATCH"),
        (np.array([[np.nan, 0, 0, 0]]), 1, 4, "CLIP_NON_FINITE"),
    ],
)
def test_clip_issues(tmp_path: Path, array, mapping_count: int, dimension: int, code: str) -> None:
    asset = paths(tmp_path)
    save_clip(asset.clip, array)
    _, issues = audit_clip(
        tmp_path, asset, mapping_count, expected_dimension=dimension, mode="full"
    )
    assert code in {item.code for item in issues}


def object_payload(coordinate="0.1", score="0.9", label="84", count=1):
    return {
        "detection_boxes": [[coordinate, " 0.2 ", "0.3", "0.4"] for _ in range(count)],
        "detection_class_entities": ["Person"] * count,
        "detection_class_labels": [label] * count,
        "detection_class_names": ["/m/person"] * count,
        "detection_scores": [score] * count,
    }


def write_object(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_object_valid_numeric_strings_whitespace_and_raw_policy(tmp_path: Path) -> None:
    asset = paths(tmp_path)
    write_object(asset.object_directory / "001.json", object_payload())
    record, issues = audit_objects(tmp_path, asset, {1}, mode="full", max_bytes=10000)
    assert not issues
    assert record["valid_detections"] == 1
    assert record["numeric_string_count"] == 6
    assert record["trimmed_whitespace_count"] == 1
    assert record["bbox_order"] == "UNKNOWN"


def test_object_native_numbers(tmp_path: Path) -> None:
    asset = paths(tmp_path)
    write_object(
        asset.object_directory / "001.json", object_payload(coordinate=0.1, score=0.9, label=84)
    )
    record, _ = audit_objects(tmp_path, asset, {1}, mode="full", max_bytes=10000)
    assert record["native_number_count"] == 3


@pytest.mark.parametrize(
    ("kwargs", "code"),
    [
        ({"coordinate": True}, "OBJECT_NUMERIC_INVALID"),
        ({"score": None}, "OBJECT_NUMERIC_INVALID"),
        ({"label": "84.0"}, "OBJECT_LABEL_INVALID"),
        ({"score": "NaN"}, "OBJECT_NUMERIC_INVALID"),
        ({"coordinate": "bad"}, "OBJECT_NUMERIC_INVALID"),
        ({"coordinate": "2.0"}, "OBJECT_COORDINATE_OUTSIDE_ZERO_ONE"),
        ({"score": "2.0"}, "OBJECT_SCORE_OUTSIDE_ZERO_ONE"),
    ],
)
def test_object_numeric_policy(tmp_path: Path, kwargs, code: str) -> None:
    asset = paths(tmp_path)
    write_object(asset.object_directory / "001.json", object_payload(**kwargs))
    record, issues = audit_objects(tmp_path, asset, {1}, mode="full", max_bytes=10000)
    assert record["detections_observed"] == 1
    assert code in {item.code for item in issues}


def test_object_parallel_mismatch_no_silent_drop(tmp_path: Path) -> None:
    asset = paths(tmp_path)
    payload = object_payload(count=2)
    payload["detection_scores"].pop()
    write_object(asset.object_directory / "001.json", payload)
    record, issues = audit_objects(tmp_path, asset, {1}, mode="full", max_bytes=10000)
    assert record["detections_observed"] == 0
    assert issues[-1].code == "OBJECT_PARALLEL_ARRAY_LENGTH_MISMATCH"


def metadata_payload():
    return {
        name: ([] if name == "keywords" else "x")
        for name in (
            "author",
            "channel_id",
            "channel_url",
            "description",
            "keywords",
            "length",
            "publish_date",
            "thumbnail_url",
            "title",
            "watch_url",
        )
    }


def test_metadata_valid_lengths_without_text_copy(tmp_path: Path) -> None:
    asset = paths(tmp_path)
    payload = metadata_payload()
    payload["description"] = "x" * 10000
    write_object(asset.metadata, payload)
    record, issues = audit_metadata(tmp_path, asset)
    assert not issues
    assert record["description_length"] == 10000
    assert "description" not in record


@pytest.mark.parametrize(
    ("mutation", "code"),
    [
        (lambda p: p.pop("title"), "METADATA_FIELD_MISSING"),
        (lambda p: p.update(title=None), "METADATA_FIELD_NULL"),
    ],
)
def test_metadata_missing_or_null(tmp_path: Path, mutation, code: str) -> None:
    asset = paths(tmp_path)
    payload = metadata_payload()
    mutation(payload)
    write_object(asset.metadata, payload)
    _, issues = audit_metadata(tmp_path, asset)
    assert code in {item.code for item in issues}


def ffprobe_payload(*, audio=True, nb_frames="10", avg="25/1", r="25/1"):
    streams = [
        {
            "codec_type": "video",
            "codec_name": "h264",
            "width": 640,
            "height": 480,
            "pix_fmt": "yuv420p",
            "avg_frame_rate": avg,
            "r_frame_rate": r,
            "time_base": "1/1000",
            "duration": "1.0",
            "nb_frames": nb_frames,
        }
    ]
    if audio:
        streams.append({"codec_type": "audio", "codec_name": "aac"})
    return {"streams": streams, "format": {"format_name": "mp4", "duration": "1.0"}}


def test_media_probe_valid_and_audio(tmp_path: Path) -> None:
    asset = paths(tmp_path)
    asset.video.parent.mkdir(parents=True)
    asset.video.write_bytes(b"fake")

    def run(*args, **kwargs):
        return subprocess.CompletedProcess(args[0], 0, json.dumps(ffprobe_payload()), "")

    record, issues = probe_video(tmp_path, asset, timeout=1, which=lambda _: "ffprobe", run=run)
    assert not issues
    assert record["probe_status"] == "SUCCESS" and record["has_audio_stream"]
    assert record["cfr_vfr_indicator"] == "CFR_LIKE"


def test_media_probe_missing_binary_only_blocks_raw(tmp_path: Path) -> None:
    asset = paths(tmp_path)
    asset.video.parent.mkdir(parents=True)
    asset.video.write_bytes(b"fake")
    _, issues = probe_video(tmp_path, asset, timeout=1, which=lambda _: None)
    assert issues[0].code == "FFPROBE_NOT_AVAILABLE"
    assert not issues[0].blocks_btc_baseline and issues[0].blocks_raw_video_pipeline


def test_media_probe_timeout(tmp_path: Path) -> None:
    asset = paths(tmp_path)
    asset.video.parent.mkdir(parents=True)
    asset.video.write_bytes(b"fake")

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("ffprobe", 1)

    _, issues = probe_video(tmp_path, asset, timeout=1, which=lambda _: "ffprobe", run=timeout)
    assert issues[0].code == "VIDEO_PROBE_FAILED"


@pytest.mark.parametrize(
    ("payload", "code"),
    [("bad", "VIDEO_PROBE_FAILED"), (json.dumps({"streams": []}), "VIDEO_STREAM_MISSING")],
)
def test_media_probe_bad_output(tmp_path: Path, payload: str, code: str) -> None:
    asset = paths(tmp_path)
    asset.video.parent.mkdir(parents=True)
    asset.video.write_bytes(b"fake")

    def run(*args, **kwargs):
        return subprocess.CompletedProcess(args[0], 0, payload, "")

    _, issues = probe_video(tmp_path, asset, timeout=1, which=lambda _: "ffprobe", run=run)
    assert issues[0].code == code


def test_media_missing_frames_audio_and_vfr_are_heuristics(tmp_path: Path) -> None:
    asset = paths(tmp_path)
    asset.video.parent.mkdir(parents=True)
    asset.video.write_bytes(b"fake")

    def run(*args, **kwargs):
        output = json.dumps(ffprobe_payload(audio=False, nb_frames="N/A", r="30/1"))
        return subprocess.CompletedProcess(args[0], 0, output, "")

    record, issues = probe_video(tmp_path, asset, timeout=1, which=lambda _: "ffprobe", run=run)
    codes = {item.code for item in issues}
    assert {"VIDEO_FRAME_COUNT_UNKNOWN", "VIDEO_AUDIO_MISSING", "VIDEO_VFR_INDICATOR"} <= codes
    assert record["cfr_vfr_indicator"] == "VFR_LIKE"
