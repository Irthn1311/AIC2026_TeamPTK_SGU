from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from system_tai.data.corpus_discovery import discover_corpus
from system_tai.refinement.video import VideoProbe
from system_tai.validation.readiness import (
    RawVideoPolicy,
    ReadinessStatus,
    ReadinessValidationLevel,
    main,
    validate_readiness,
)
from tests.phase3_helpers import create_corpus, feature_matrix


def _fixture_videos() -> dict[str, tuple[list[int], np.ndarray]]:
    return {
        "L21_V001": ([0, 10], feature_matrix([(0, 1.0), (1, 1.0)])),
        "L21_V002": ([0, 20], feature_matrix([(2, 1.0), (3, 1.0)])),
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_manifest_level_is_ready_and_does_not_expose_machine_paths(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    create_corpus(input_root, _fixture_videos())
    manifest_path = discover_corpus(input_root, portable=True).write(
        tmp_path / "portable.json",
        portable=True,
    )
    before = _sha256(manifest_path)
    report = validate_readiness(
        manifest_path,
        input_root=input_root,
        validation_level=ReadinessValidationLevel.MANIFEST,
        raw_video_policy=RawVideoPolicy.REQUIRED,
    )
    assert report.status is ReadinessStatus.READY
    assert report.video_count == 2
    assert report.feature_row_count == 4
    assert report.raw_video_present_count == 2
    assert report.feature_validated_video_count == 0
    assert report.issues == ()
    assert report.copied_source_artifacts is False
    assert str(tmp_path) not in json.dumps(report.to_payload())
    assert _sha256(manifest_path) == before


def test_missing_raw_video_is_warning_or_error_by_policy(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    create_corpus(input_root, _fixture_videos(), include_raw_video=False)
    manifest_path = discover_corpus(input_root, portable=True).write(
        tmp_path / "portable.json",
        portable=True,
    )
    optional = validate_readiness(
        manifest_path,
        input_root=input_root,
        raw_video_policy=RawVideoPolicy.OPTIONAL,
    )
    required = validate_readiness(
        manifest_path,
        input_root=input_root,
        raw_video_policy=RawVideoPolicy.REQUIRED,
    )
    assert optional.ready is True
    assert optional.issue_counts == {"ERROR": 0, "WARNING": 2}
    assert required.ready is False
    assert required.issue_counts == {"ERROR": 2, "WARNING": 0}
    assert [issue.video_id for issue in required.issues] == ["L21_V001", "L21_V002"]


def test_feature_level_accepts_duplicate_frame_ids_and_counts_rows(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    create_corpus(
        input_root,
        {"L21_V001": ([0, 0, 10], feature_matrix([(0, 1), (1, 1), (2, 1)]))},
    )
    manifest_path = discover_corpus(input_root, portable=True).write(
        tmp_path / "portable.json",
        portable=True,
    )
    report = validate_readiness(
        manifest_path,
        input_root=input_root,
        validation_level=ReadinessValidationLevel.FEATURES,
        raw_video_policy=RawVideoPolicy.REQUIRED,
    )
    assert report.ready is True
    assert report.feature_validated_video_count == 1
    assert report.duplicate_mapped_frame_row_count == 1


def test_feature_level_rejects_nonfinite_matrix_without_copying_source(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    dataset = create_corpus(input_root, _fixture_videos())
    manifest_path = discover_corpus(input_root, portable=True).write(
        tmp_path / "portable.json",
        portable=True,
    )
    clip = (
        dataset
        / "clip-features-32-aic25-b1"
        / "clip-features-32"
        / "L21_V001.npy"
    )
    matrix = np.load(clip, allow_pickle=False)
    matrix[0, 0] = np.nan
    np.save(clip, matrix)
    assert clip.stat().st_size == json.loads(manifest_path.read_text())["videos"][0][
        "clip_size_bytes"
    ]
    report = validate_readiness(
        manifest_path,
        input_root=input_root,
        validation_level=ReadinessValidationLevel.FEATURES,
    )
    assert report.ready is False
    failure = next(issue for issue in report.issues if issue.video_id == "L21_V001")
    assert failure.code == "FEATURE_VALIDATION_FAILED"
    assert "NaN or Infinity" in failure.message


def test_full_level_proves_original_frame_bounds_with_fake_probe(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    create_corpus(
        input_root,
        {"L21_V001": ([0, 10], feature_matrix([(0, 1), (1, 1)]))},
    )
    manifest_path = discover_corpus(input_root, portable=True).write(
        tmp_path / "portable.json",
        portable=True,
    )

    def short_probe(video) -> VideoProbe:
        return VideoProbe(
            video_id=video.video_id,
            raw_video_path=video.raw_video_path,
            decoder_backend="fake-probe",
            fps=30.0,
            total_frame_count=10,
            width=640,
            height=360,
            duration_seconds=1 / 3,
        )

    report = validate_readiness(
        manifest_path,
        input_root=input_root,
        validation_level=ReadinessValidationLevel.FULL,
        raw_video_policy=RawVideoPolicy.REQUIRED,
        video_prober=short_probe,
    )
    assert report.ready is False
    assert report.raw_video_probed_count == 0
    assert report.issues[0].code == "RAW_VIDEO_PROBE_FAILED"
    assert "max=10, upper=9" in report.issues[0].message


def test_full_level_passes_and_preserves_absolute_frame_ids(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    create_corpus(
        input_root,
        {"L21_V001": ([0, 10], feature_matrix([(0, 1), (1, 1)]))},
    )
    manifest_path = discover_corpus(input_root, portable=True).write(
        tmp_path / "portable.json",
        portable=True,
    )

    def valid_probe(video) -> VideoProbe:
        return VideoProbe(
            video_id=video.video_id,
            raw_video_path=video.raw_video_path,
            decoder_backend="fake-probe",
            fps=30.0,
            total_frame_count=11,
            width=640,
            height=360,
            duration_seconds=11 / 30,
        )

    report = validate_readiness(
        manifest_path,
        input_root=input_root,
        validation_level=ReadinessValidationLevel.FULL,
        raw_video_policy=RawVideoPolicy.REQUIRED,
        video_prober=valid_probe,
    )
    assert report.ready is True
    assert report.raw_video_probed_count == 1
    assert report.feature_validated_video_count == 1


def test_manifest_failure_is_structured_and_cli_returns_nonzero(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    report = validate_readiness(missing)
    assert report.status is ReadinessStatus.NOT_READY
    assert report.issues[0].code == "MANIFEST_LOAD_FAILED"
    destination = tmp_path / "report.json"
    exit_code = main(["--manifest", str(missing), "--output", str(destination)])
    assert exit_code == 2
    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["status"] == "NOT_READY"
    assert payload["ready"] is False


def test_cli_writes_deterministic_utf8_report_for_portable_manifest(
    tmp_path: Path,
) -> None:
    input_root = tmp_path / "input"
    create_corpus(input_root, _fixture_videos())
    manifest_path = discover_corpus(input_root, portable=True).write(
        tmp_path / "portable.json",
        portable=True,
    )
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    args = [
        "--manifest",
        str(manifest_path),
        "--input-root",
        str(input_root),
        "--validation-level",
        "features",
        "--raw-video-policy",
        "required",
    ]
    assert main([*args, "--output", str(first)]) == 0
    assert main([*args, "--output", str(second)]) == 0
    assert first.read_bytes() == second.read_bytes()
    assert not first.read_bytes().startswith(b"\xef\xbb\xbf")
    assert first.read_bytes().endswith(b"\n")
