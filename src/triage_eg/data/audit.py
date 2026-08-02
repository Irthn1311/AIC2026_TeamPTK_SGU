"""Lightweight metadata-only dataset auditing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from triage_eg.common.schemas import VideoRecord
from triage_eg.data.manifest import validate_relative_paths, validate_unique_video_ids


@dataclass(frozen=True)
class DataAuditReport:
    """Summary of manifest consistency and expected local files."""

    total_videos: int
    total_frames: int
    total_duration_ms: int
    missing_paths: tuple[str, ...]
    duplicate_video_ids: tuple[str, ...]
    invalid_records: tuple[str, ...]


def audit_video_records(records: list[VideoRecord], data_root: str | Path) -> DataAuditReport:
    """Audit metadata and path existence without decoding any video."""

    root = Path(data_root)
    invalid_paths = validate_relative_paths(records)
    missing = [
        record.relative_path
        for record in records
        if record.relative_path not in invalid_paths and not (root / record.relative_path).is_file()
    ]
    invalid = [f"invalid relative path: {path}" for path in invalid_paths]
    return DataAuditReport(
        total_videos=len(records),
        total_frames=sum(record.total_frames for record in records),
        total_duration_ms=sum(record.duration_ms for record in records),
        missing_paths=tuple(missing),
        duplicate_video_ids=tuple(validate_unique_video_ids(records)),
        invalid_records=tuple(invalid),
    )

