"""Dependency-free data contracts for the system_tai skeleton.

These records define internal shapes only. They do not establish an accepted
team checkpoint schema or an official BTC submission schema.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class FrameIndexBase(StrEnum):
    """Declared coordinate base for original BTC video frame indexes."""

    ZERO = "zero_based"
    ONE = "one_based"
    UNKNOWN = "unknown"


def _require_text(value: str, field_name: str) -> None:
    if not value.strip():
        raise ValueError(f"{field_name} must not be empty")


@dataclass(frozen=True, slots=True)
class BenchmarkVideoRecord:
    """Authoritative video bounds used to validate original-frame coordinates."""

    video_id: str
    video_path: Path
    fps: float
    duration_seconds: float
    total_frames: int
    frame_index_base: FrameIndexBase
    codec: str | None = None
    width: int | None = None
    height: int | None = None

    def __post_init__(self) -> None:
        _require_text(self.video_id, "video_id")
        if not str(self.video_path).strip():
            raise ValueError("video_path must not be empty")
        if not math.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("fps must be positive")
        if not math.isfinite(self.duration_seconds) or self.duration_seconds <= 0:
            raise ValueError("duration_seconds must be positive")
        if self.total_frames <= 0:
            raise ValueError("total_frames must be positive")
        if self.frame_index_base is FrameIndexBase.UNKNOWN:
            raise ValueError("frame_index_base must be known")
        if (self.width is None) != (self.height is None):
            raise ValueError("width and height must be provided together")
        if self.width is not None and (self.width <= 0 or self.height is None or self.height <= 0):
            raise ValueError("resolution dimensions must be positive")


@dataclass(frozen=True, slots=True)
class FrameRecord:
    """Mapping record whose actual_frame_id is an original BTC frame index."""

    video_id: str
    actual_frame_id: int
    keyframe_order: int | None
    clip_row: int
    pts_time: float
    fps: float
    mapping_version: str
    physical_row: int = 0
    keyframe_filename: str | None = None

    def __post_init__(self) -> None:
        _require_text(self.video_id, "video_id")
        _require_text(self.mapping_version, "mapping_version")
        if self.actual_frame_id < 0:
            raise ValueError("actual_frame_id must be non-negative")
        if self.keyframe_order is not None and self.keyframe_order < 0:
            raise ValueError("keyframe_order must be non-negative when provided")
        if self.clip_row < 0:
            raise ValueError("clip_row must be non-negative")
        if not math.isfinite(self.pts_time) or self.pts_time < 0:
            raise ValueError("pts_time must be non-negative")
        if not math.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("fps must be positive")
        if self.physical_row < 0:
            raise ValueError("physical_row must be non-negative")


@dataclass(frozen=True, slots=True)
class FeatureRecord:
    video_id: str
    actual_frame_id: int
    clip_row: int
    encoder_id: str
    dimension: int

    def __post_init__(self) -> None:
        _require_text(self.video_id, "video_id")
        _require_text(self.encoder_id, "encoder_id")
        if self.actual_frame_id < 0 or self.clip_row < 0:
            raise ValueError("frame and feature-row indexes must be non-negative")
        if self.dimension <= 0:
            raise ValueError("dimension must be positive")


@dataclass(frozen=True, slots=True)
class RetrievalHit:
    clip_row: int
    score: float
    rank: int

    def __post_init__(self) -> None:
        if self.clip_row < 0:
            raise ValueError("clip_row must be non-negative")
        if self.rank < 1:
            raise ValueError("rank must start at one")


@dataclass(frozen=True, slots=True)
class CandidateFrame:
    query_id: str
    video_id: str
    actual_frame_id: int
    score: float
    retrieval_rank: int
    clip_row: int

    def __post_init__(self) -> None:
        _require_text(self.query_id, "query_id")
        _require_text(self.video_id, "video_id")
        if self.actual_frame_id < 0 or self.clip_row < 0:
            raise ValueError("frame and feature-row indexes must be non-negative")
        if self.retrieval_rank < 1:
            raise ValueError("retrieval_rank must start at one")


@dataclass(frozen=True, slots=True)
class RankedKISRecord:
    query_id: str
    rank: int
    video_id: str
    actual_frame_id: int
    score: float

    def __post_init__(self) -> None:
        _require_text(self.query_id, "query_id")
        _require_text(self.video_id, "video_id")
        if self.rank < 1:
            raise ValueError("rank must start at one")
        if self.actual_frame_id < 0:
            raise ValueError("actual_frame_id must be non-negative")


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    code: str
    message: str
    line_number: int | None = None


@dataclass(frozen=True, slots=True)
class ValidationResult:
    valid: bool
    errors: tuple[ValidationIssue, ...] = ()
    warnings: tuple[ValidationIssue, ...] = ()
