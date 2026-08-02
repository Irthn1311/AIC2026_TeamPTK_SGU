"""Stable data contracts shared by TRIAGE-EG modules."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, is_dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, TypeVar


class SourcePolicy(StrEnum):
    """Supported origins for retrieval frames."""

    BTC_KEYFRAME = "BTC_KEYFRAME"
    SHOT_CENTER = "SHOT_CENTER"
    ADAPTIVE_MULTIFRAME = "ADAPTIVE_MULTIFRAME"
    UNIFORM_FALLBACK = "UNIFORM_FALLBACK"
    QUERY_LOCAL_DENSE = "QUERY_LOCAL_DENSE"


def _require_non_empty(value: str, field_name: str) -> None:
    if not value.strip():
        raise ValueError(f"{field_name} must not be empty")


@dataclass(frozen=True)
class VideoRecord:
    """Metadata for one source video; paths are relative to the configured data root."""

    video_id: str
    relative_path: str
    batch_id: str
    fps: float
    total_frames: int
    duration_ms: int
    width: int | None
    height: int | None
    has_audio: bool | None
    dataset_version: str

    def __post_init__(self) -> None:
        _require_non_empty(self.video_id, "video_id")
        if self.fps <= 0:
            raise ValueError("fps must be greater than zero")
        if self.total_frames < 0:
            raise ValueError("total_frames must be non-negative")
        if self.duration_ms < 0:
            raise ValueError("duration_ms must be non-negative")


@dataclass(frozen=True)
class ShotRecord:
    """Inclusive frame and time boundaries for one detected shot."""

    shot_id: str
    video_id: str
    start_frame: int
    end_frame: int
    start_time_ms: int
    end_time_ms: int
    detector_name: str
    detector_version: str
    confidence: float | None

    def __post_init__(self) -> None:
        if self.start_frame > self.end_frame:
            raise ValueError("start_frame must not exceed end_frame")
        if self.start_time_ms > self.end_time_ms:
            raise ValueError("start_time_ms must not exceed end_time_ms")


@dataclass(frozen=True)
class FrameRecord:
    """A retrieval frame mapped to its actual source-video frame identifier."""

    frame_uid: str
    video_id: str
    actual_frame_id: int
    timestamp_ms: int
    image_path: str
    source_policy: str
    shot_id: str | None
    dataset_version: str
    extraction_version: str

    def __post_init__(self) -> None:
        try:
            SourcePolicy(self.source_policy)
        except ValueError as error:
            supported = ", ".join(policy.value for policy in SourcePolicy)
            raise ValueError(f"Unsupported source_policy; expected one of: {supported}") from error
        if self.actual_frame_id < 0 or self.timestamp_ms < 0:
            raise ValueError("frame and timestamp identifiers must be non-negative")


@dataclass(frozen=True)
class FeatureRecord:
    """Metadata mapping one frame to one row in a feature matrix."""

    feature_uid: str
    frame_uid: str
    model_name: str
    model_version: str
    feature_row: int
    dimension: int
    normalized: bool
    artifact_path: str


@dataclass(frozen=True)
class CandidateFrame:
    """A ranked retrieval candidate expressed in submission frame coordinates."""

    video_id: str
    frame_id: int
    timestamp_ms: int
    frame_uid: str
    score: float
    rank: int
    source_branch: str


@dataclass(frozen=True)
class CandidateVideo:
    """Video-level aggregation of frame candidates."""

    video_id: str
    score: float
    best_frames: tuple[CandidateFrame, ...]
    matched_event_count: int
    source_branches: tuple[str, ...]


@dataclass(frozen=True)
class RunManifest:
    """Reproducibility metadata saved beside every generated artifact."""

    run_id: str
    created_at_utc: str
    git_commit: str
    config_path: str
    data_version: str
    artifact_name: str
    command: str
    status: str


def dataclass_to_dict(value: Any) -> dict[str, Any]:
    """Convert a dataclass instance to a JSON-compatible dictionary."""

    if not is_dataclass(value) or isinstance(value, type):
        raise TypeError("value must be a dataclass instance")
    return asdict(value)


def save_jsonl(records: Iterable[Any], path: str | Path) -> None:
    """Write mappings or dataclass instances as UTF-8 JSON Lines."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        for record in records:
            payload = dataclass_to_dict(record) if is_dataclass(record) else record
            if not isinstance(payload, Mapping):
                raise TypeError("JSONL records must be mappings or dataclass instances")
            stream.write(json.dumps(payload, ensure_ascii=False) + "\n")


RecordT = TypeVar("RecordT")


def load_jsonl(path: str | Path, record_type: type[RecordT] | None = None) -> list[Any]:
    """Read JSON Lines, optionally constructing a dataclass-compatible type."""

    records: list[Any] = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected an object on JSONL line {line_number}")
            records.append(record_type(**payload) if record_type else payload)
    return records
