"""Dependency-free data contracts for the system_tai skeleton.

These records define internal shapes only. They do not establish an accepted
team checkpoint schema or an official BTC submission schema.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any


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
class VideoFeatureStore:
    """Immutable descriptor for one validated BTC mapping/feature pair."""

    video_id: str
    mapping_csv_path: Path
    clip_npy_path: Path
    row_count: int
    embedding_dimension: int
    normalized: bool

    def __post_init__(self) -> None:
        _require_text(self.video_id, "video_id")
        if self.row_count <= 0:
            raise ValueError("row_count must be positive")
        if self.embedding_dimension <= 0:
            raise ValueError("embedding_dimension must be positive")


@dataclass(frozen=True, slots=True)
class FrameMappingRecord:
    """Physical mapping row whose frame_id is the BTC CSV frame_idx exactly."""

    clip_row: int
    keyframe_order: int
    frame_id: int
    pts_time: float
    fps: float

    def __post_init__(self) -> None:
        if self.clip_row < 0:
            raise ValueError("clip_row must be non-negative")
        if self.keyframe_order < 0:
            raise ValueError("keyframe_order must be non-negative")
        if self.frame_id < 0:
            raise ValueError("frame_id must be non-negative")
        if not math.isfinite(self.pts_time) or self.pts_time < 0:
            raise ValueError("pts_time must be finite and non-negative")
        if not math.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("fps must be finite and positive")


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
    video_id: str
    frame_id: int
    clip_row: int
    keyframe_order: int
    score: float
    rank: int
    source: str
    diagnostic_metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        _require_text(self.video_id, "video_id")
        _require_text(self.source, "source")
        if self.frame_id < 0 or self.clip_row < 0 or self.keyframe_order < 0:
            raise ValueError("frame and feature-row indexes must be non-negative")
        if not math.isfinite(self.score):
            raise ValueError("score must be finite")
        if self.rank < 1:
            raise ValueError("rank must start at one")
        if self.diagnostic_metadata is not None:
            object.__setattr__(
                self,
                "diagnostic_metadata",
                MappingProxyType(dict(self.diagnostic_metadata)),
            )


@dataclass(frozen=True, slots=True)
class KISQuery:
    query_id: str
    text: str
    top_k: int

    def __post_init__(self) -> None:
        _require_text(self.query_id, "query_id")
        _require_text(self.text, "text")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")


@dataclass(frozen=True, slots=True)
class KISResult:
    query_id: str
    ranked_candidates: tuple[CandidateFrame, ...]

    def __post_init__(self) -> None:
        _require_text(self.query_id, "query_id")
        expected_ranks = tuple(range(1, len(self.ranked_candidates) + 1))
        observed_ranks = tuple(candidate.rank for candidate in self.ranked_candidates)
        if observed_ranks != expected_ranks:
            raise ValueError("candidate ranks must be contiguous and start at one")


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
    query_id: str | None = None


@dataclass(frozen=True, slots=True)
class ValidationResult:
    valid: bool
    errors: tuple[ValidationIssue, ...] = ()
    warnings: tuple[ValidationIssue, ...] = ()
