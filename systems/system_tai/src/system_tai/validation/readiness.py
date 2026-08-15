"""Fail-closed corpus and index readiness validation.

This boundary audits whether a previously built feature manifest is safe to use.  It
does not build indexes, load a text model, change ranking, or copy source artifacts.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

from system_tai.data.corpus_discovery import (
    CorpusDiscoveryError,
    DiscoveredVideo,
    load_corpus_manifest,
)
from system_tai.features.btc_clip_store import (
    LoadedVideoFeatureStore,
    VideoFeatureStoreLoader,
)
from system_tai.refinement.video import (
    OpenCVVideoDecoder,
    RawVideoRecord,
    VideoProbe,
)


class ReadinessValidationLevel(StrEnum):
    MANIFEST = "manifest"
    FEATURES = "features"
    FULL = "full"


class RawVideoPolicy(StrEnum):
    OPTIONAL = "optional"
    REQUIRED = "required"


class ReadinessStatus(StrEnum):
    READY = "READY"
    NOT_READY = "NOT_READY"


class IssueSeverity(StrEnum):
    ERROR = "ERROR"
    WARNING = "WARNING"


@dataclass(frozen=True, slots=True)
class ReadinessIssue:
    severity: IssueSeverity
    code: str
    message: str
    video_id: str | None = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "severity": self.severity.value,
            "code": self.code,
            "video_id": self.video_id,
            "message": self.message,
        }


@dataclass(frozen=True, slots=True)
class ReadinessReport:
    status: ReadinessStatus
    validation_level: ReadinessValidationLevel
    raw_video_policy: RawVideoPolicy
    expected_embedding_dimension: int
    manifest_schema_version: int | None
    manifest_fingerprint: str | None
    manifest_portable: bool | None
    dataset_root_name: str | None
    video_count: int
    feature_row_count: int
    feature_validated_video_count: int
    raw_video_present_count: int
    raw_video_missing_count: int
    raw_video_probed_count: int
    duplicate_mapped_frame_row_count: int
    issue_counts: Mapping[str, int]
    issues: tuple[ReadinessIssue, ...]
    copied_source_artifacts: bool = False
    schema_version: int = 1

    @property
    def ready(self) -> bool:
        return self.status is ReadinessStatus.READY

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "status": self.status.value,
            "ready": self.ready,
            "validation_level": self.validation_level.value,
            "raw_video_policy": self.raw_video_policy.value,
            "expected_embedding_dimension": self.expected_embedding_dimension,
            "manifest_schema_version": self.manifest_schema_version,
            "manifest_fingerprint": self.manifest_fingerprint,
            "manifest_portable": self.manifest_portable,
            "dataset_root_name": self.dataset_root_name,
            "video_count": self.video_count,
            "feature_row_count": self.feature_row_count,
            "feature_validated_video_count": self.feature_validated_video_count,
            "raw_video_present_count": self.raw_video_present_count,
            "raw_video_missing_count": self.raw_video_missing_count,
            "raw_video_probed_count": self.raw_video_probed_count,
            "duplicate_mapped_frame_row_count": self.duplicate_mapped_frame_row_count,
            "issue_counts": dict(self.issue_counts),
            "issues": [issue.to_payload() for issue in self.issues],
            "copied_source_artifacts": self.copied_source_artifacts,
        }


class FeatureLoader(Protocol):
    def load(
        self,
        *,
        video_id: str,
        mapping_csv_path: Path,
        clip_npy_path: Path,
    ) -> LoadedVideoFeatureStore: ...


VideoProber = Callable[[DiscoveredVideo], VideoProbe]


def _issue(
    severity: IssueSeverity,
    code: str,
    message: str,
    *,
    video_id: str | None = None,
) -> ReadinessIssue:
    return ReadinessIssue(severity, code, message, video_id)


def _probe_with_opencv(video: DiscoveredVideo) -> VideoProbe:
    decoder = OpenCVVideoDecoder()
    return decoder.probe(RawVideoRecord(video.video_id, video.raw_video_path))


def _empty_report(
    *,
    level: ReadinessValidationLevel,
    raw_policy: RawVideoPolicy,
    expected_dimension: int,
    issue: ReadinessIssue,
) -> ReadinessReport:
    return ReadinessReport(
        status=ReadinessStatus.NOT_READY,
        validation_level=level,
        raw_video_policy=raw_policy,
        expected_embedding_dimension=expected_dimension,
        manifest_schema_version=None,
        manifest_fingerprint=None,
        manifest_portable=None,
        dataset_root_name=None,
        video_count=0,
        feature_row_count=0,
        feature_validated_video_count=0,
        raw_video_present_count=0,
        raw_video_missing_count=0,
        raw_video_probed_count=0,
        duplicate_mapped_frame_row_count=0,
        issue_counts={"ERROR": 1, "WARNING": 0},
        issues=(issue,),
    )


def validate_readiness(
    manifest_path: Path,
    *,
    input_root: Path | None = None,
    validation_level: ReadinessValidationLevel = ReadinessValidationLevel.MANIFEST,
    raw_video_policy: RawVideoPolicy = RawVideoPolicy.OPTIONAL,
    expected_embedding_dimension: int = 512,
    max_root_depth: int = 4,
    feature_loader: FeatureLoader | None = None,
    video_prober: VideoProber | None = None,
) -> ReadinessReport:
    """Validate a reusable manifest without changing any source artifact."""

    if expected_embedding_dimension <= 0:
        raise ValueError("expected_embedding_dimension must be positive")
    level = ReadinessValidationLevel(validation_level)
    raw_policy = RawVideoPolicy(raw_video_policy)
    try:
        manifest = load_corpus_manifest(
            manifest_path,
            input_root=input_root,
            validate_sources=True,
            max_root_depth=max_root_depth,
        )
    except (CorpusDiscoveryError, FileNotFoundError, OSError, ValueError) as exc:
        return _empty_report(
            level=level,
            raw_policy=raw_policy,
            expected_dimension=expected_embedding_dimension,
            issue=_issue(
                IssueSeverity.ERROR,
                "MANIFEST_LOAD_FAILED",
                f"{type(exc).__name__}: {exc}",
            ),
        )

    issues: list[ReadinessIssue] = []
    feature_validated = 0
    raw_probed = 0
    duplicate_rows = 0
    raw_present = sum(video.raw_video_path is not None for video in manifest.videos)
    raw_missing = len(manifest.videos) - raw_present

    for video in manifest.videos:
        if video.embedding_dimension != expected_embedding_dimension:
            issues.append(
                _issue(
                    IssueSeverity.ERROR,
                    "MANIFEST_DIMENSION_MISMATCH",
                    (
                        f"manifest dimension {video.embedding_dimension} does not match "
                        f"expected {expected_embedding_dimension}"
                    ),
                    video_id=video.video_id,
                )
            )
        if video.raw_video_path is None:
            severity = (
                IssueSeverity.ERROR
                if raw_policy is RawVideoPolicy.REQUIRED
                else IssueSeverity.WARNING
            )
            issues.append(
                _issue(
                    severity,
                    "RAW_VIDEO_MISSING",
                    "raw video is absent from the corpus manifest",
                    video_id=video.video_id,
                )
            )

    loaded_stores: dict[str, LoadedVideoFeatureStore] = {}
    if level in {ReadinessValidationLevel.FEATURES, ReadinessValidationLevel.FULL}:
        loader = feature_loader or VideoFeatureStoreLoader(
            expected_dimension=expected_embedding_dimension,
            memory_map=True,
        )
        for video in manifest.videos:
            try:
                store = loader.load(
                    video_id=video.video_id,
                    mapping_csv_path=video.mapping_csv_path,
                    clip_npy_path=video.clip_npy_path,
                )
                if store.descriptor.row_count != video.row_count:
                    raise ValueError(
                        "loaded feature row count does not match manifest: "
                        f"loaded={store.descriptor.row_count}, manifest={video.row_count}"
                    )
                loaded_stores[video.video_id] = store
                feature_validated += 1
                duplicate_rows += store.duplicate_frame_id_count
            except (FileNotFoundError, OSError, ValueError) as exc:
                issues.append(
                    _issue(
                        IssueSeverity.ERROR,
                        "FEATURE_VALIDATION_FAILED",
                        f"{type(exc).__name__}: {exc}",
                        video_id=video.video_id,
                    )
                )

    if level is ReadinessValidationLevel.FULL:
        prober = video_prober or _probe_with_opencv
        for video in manifest.videos:
            if video.raw_video_path is None:
                continue
            try:
                probe = prober(video)
                if probe.video_id != video.video_id:
                    raise ValueError(
                        f"probe video_id mismatch: {probe.video_id} != {video.video_id}"
                    )
                store = loaded_stores.get(video.video_id)
                if store is None:
                    raise ValueError("raw-frame bounds cannot be checked without feature mapping")
                out_of_bounds = [
                    record.frame_id
                    for record in store.mappings
                    if record.frame_id >= probe.total_frame_count
                ]
                if out_of_bounds:
                    raise ValueError(
                        "mapped frame_id exceeds raw-video bounds: "
                        f"max={max(out_of_bounds)}, upper={probe.total_frame_count - 1}"
                    )
                raw_probed += 1
            except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
                issues.append(
                    _issue(
                        IssueSeverity.ERROR,
                        "RAW_VIDEO_PROBE_FAILED",
                        f"{type(exc).__name__}: {exc}",
                        video_id=video.video_id,
                    )
                )

    ordered_issues = tuple(
        sorted(
            issues,
            key=lambda item: (
                0 if item.severity is IssueSeverity.ERROR else 1,
                item.video_id or "",
                item.code,
                item.message,
            ),
        )
    )
    error_count = sum(
        issue.severity is IssueSeverity.ERROR for issue in ordered_issues
    )
    warning_count = len(ordered_issues) - error_count
    return ReadinessReport(
        status=(
            ReadinessStatus.NOT_READY if error_count else ReadinessStatus.READY
        ),
        validation_level=level,
        raw_video_policy=raw_policy,
        expected_embedding_dimension=expected_embedding_dimension,
        manifest_schema_version=manifest.schema_version,
        manifest_fingerprint=manifest.fingerprint,
        manifest_portable=manifest.portable,
        dataset_root_name=manifest.dataset_root.name,
        video_count=len(manifest.videos),
        feature_row_count=manifest.total_rows,
        feature_validated_video_count=feature_validated,
        raw_video_present_count=raw_present,
        raw_video_missing_count=raw_missing,
        raw_video_probed_count=raw_probed,
        duplicate_mapped_frame_row_count=duplicate_rows,
        issue_counts={"ERROR": error_count, "WARNING": warning_count},
        issues=ordered_issues,
    )


def write_readiness_report(report: ReadinessReport, path: Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(
                report.to_payload(),
                stream,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--input-root", type=Path)
    parser.add_argument(
        "--validation-level",
        choices=tuple(level.value for level in ReadinessValidationLevel),
        default=ReadinessValidationLevel.MANIFEST.value,
    )
    parser.add_argument(
        "--raw-video-policy",
        choices=tuple(policy.value for policy in RawVideoPolicy),
        default=RawVideoPolicy.OPTIONAL.value,
    )
    parser.add_argument("--expected-dimension", type=int, default=512)
    parser.add_argument("--max-root-depth", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = validate_readiness(
            args.manifest,
            input_root=args.input_root,
            validation_level=ReadinessValidationLevel(args.validation_level),
            raw_video_policy=RawVideoPolicy(args.raw_video_policy),
            expected_embedding_dimension=args.expected_dimension,
            max_root_depth=args.max_root_depth,
        )
        destination = write_readiness_report(report, args.output)
    except (OSError, ValueError) as exc:
        print(f"readiness validation failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": report.status.value,
                "ready": report.ready,
                "validation_level": report.validation_level.value,
                "video_count": report.video_count,
                "feature_row_count": report.feature_row_count,
                "error_count": report.issue_counts["ERROR"],
                "warning_count": report.issue_counts["WARNING"],
                "report": str(destination),
                "copied_source_artifacts": False,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )
    return 0 if report.ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
