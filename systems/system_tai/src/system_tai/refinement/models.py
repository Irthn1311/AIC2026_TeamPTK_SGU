"""Immutable Phase 4 domain models and validated configuration."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any

from system_tai.refinement.video import CoarseDecodeStrategy
from system_tai.retrieval.multi_query import QueryVariant


class MissingRawVideoPolicy(StrEnum):
    KEEP_ORIGINAL = "keep-original"
    SKIP_CANDIDATE = "skip-candidate"
    FAIL_QUERY = "fail-query"


class CandidateFailurePolicy(StrEnum):
    KEEP_ORIGINAL = "keep-original"
    SKIP_CANDIDATE = "skip-candidate"
    FAIL_QUERY = "fail-query"


class RefinementStatus(StrEnum):
    REFINED = "REFINED"
    KEEP_ORIGINAL = "KEEP_ORIGINAL"
    SKIPPED = "SKIPPED"
    NOT_REFINED = "NOT_REFINED"
    FAILED = "FAILED"


@dataclass(frozen=True, slots=True)
class RefinementConfig:
    top_candidates_to_refine: int = 20
    window_before_seconds: float = 5.0
    window_after_seconds: float = 5.0
    coarse_stride_frames: int = 15
    coarse_top_n: int = 3
    fine_radius_frames: int = 30
    fine_stride_frames: int = 1
    image_batch_size: int = 32
    max_decoded_frames_per_candidate: int = 500
    output_top_k: int = 100
    device: str = "cpu"
    missing_raw_video_policy: MissingRawVideoPolicy = MissingRawVideoPolicy.KEEP_ORIGINAL
    candidate_failure_policy: CandidateFailurePolicy = CandidateFailurePolicy.KEEP_ORIGINAL
    allow_model_download: bool = False
    clip_cache_dir: Path | None = None
    rrf_constant: float = 60.0
    coarse_decode_strategy: CoarseDecodeStrategy = CoarseDecodeStrategy.SEQUENTIAL

    def __post_init__(self) -> None:
        if not 1 <= self.top_candidates_to_refine <= 100:
            raise ValueError("top_candidates_to_refine must be between 1 and 100")
        if not 1 <= self.output_top_k <= 100:
            raise ValueError("output_top_k must be between 1 and 100")
        for field, value in (
            ("window_before_seconds", self.window_before_seconds),
            ("window_after_seconds", self.window_after_seconds),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{field} must be finite and non-negative")
        for field, value in (
            ("coarse_stride_frames", self.coarse_stride_frames),
            ("coarse_top_n", self.coarse_top_n),
            ("fine_stride_frames", self.fine_stride_frames),
            ("image_batch_size", self.image_batch_size),
            ("max_decoded_frames_per_candidate", self.max_decoded_frames_per_candidate),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{field} must be a positive integer")
        if type(self.fine_radius_frames) is not int or self.fine_radius_frames < 0:
            raise ValueError("fine_radius_frames must be a non-negative integer")
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("device must be cpu or cuda")
        if not isinstance(self.missing_raw_video_policy, MissingRawVideoPolicy):
            raise ValueError("invalid missing_raw_video_policy")
        if not isinstance(self.candidate_failure_policy, CandidateFailurePolicy):
            raise ValueError("invalid candidate_failure_policy")
        if not math.isfinite(self.rrf_constant) or self.rrf_constant <= 0:
            raise ValueError("rrf_constant must be finite and positive")
        if not isinstance(self.coarse_decode_strategy, CoarseDecodeStrategy):
            raise ValueError("invalid coarse_decode_strategy")


@dataclass(frozen=True, slots=True)
class Q3AnchorRefinementConfig:
    """Opt-in budget for refining only authoritative Q3 anchor substitutions."""

    enabled: bool = False
    max_extra_q3_anchors: int = 6

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise ValueError("enabled must be a boolean")
        if type(self.max_extra_q3_anchors) is not int or not (
            1 <= self.max_extra_q3_anchors <= 100
        ):
            raise ValueError("max_extra_q3_anchors must be in [1, 100]")


@dataclass(frozen=True, slots=True)
class Phase3Candidate:
    query_id: str
    rank: int
    video_id: str
    frame_id: int
    retrieval_score: float
    retrieval_provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not self.query_id.strip() or not self.video_id.strip():
            raise ValueError("candidate query_id and video_id must not be empty")
        if self.rank < 1 or self.frame_id < 0:
            raise ValueError("candidate rank must start at one and frame_id must be non-negative")
        if not math.isfinite(self.retrieval_score):
            raise ValueError("candidate retrieval_score must be finite")
        object.__setattr__(
            self,
            "retrieval_provenance",
            MappingProxyType(dict(self.retrieval_provenance)),
        )


@dataclass(frozen=True, slots=True)
class RefinementQuery:
    query_id: str
    variants: tuple[QueryVariant, ...]
    candidates: tuple[Phase3Candidate, ...]

    def __post_init__(self) -> None:
        if not self.query_id.strip() or not self.variants or not self.candidates:
            raise ValueError("refinement query requires ID, variants, and candidates")
        if any(candidate.query_id != self.query_id for candidate in self.candidates):
            raise ValueError("candidate query_id mismatch")
        if [candidate.rank for candidate in self.candidates] != list(
            range(1, len(self.candidates) + 1)
        ):
            raise ValueError("Phase 3 candidate ranks must be contiguous from one")


@dataclass(frozen=True, slots=True)
class RefinedCandidate:
    query_id: str
    original_candidate_rank: int
    video_id: str
    candidate_frame_id: int
    refined_frame_id: int | None
    candidate_timestamp_seconds: float | None
    refined_timestamp_seconds: float | None
    fps: float | None
    total_frame_count: int | None
    window_start_frame: int | None
    window_end_frame: int | None
    coarse_frame_ids: tuple[int, ...]
    fine_frame_ids: tuple[int, ...]
    coarse_sample_count: int
    fine_sample_count: int
    decoded_frame_count: int
    encoded_image_count: int
    refinement_fusion_score: float | None
    variant_hit_count: int
    best_individual_rank: int | None
    per_variant_provenance: tuple[Mapping[str, Any], ...]
    decoder_backend: str | None
    raw_video_path: Path | None
    status: RefinementStatus
    warnings: tuple[str, ...]
    failure_reason: str | None
    original_retrieval_provenance: Mapping[str, Any]
    timings: Mapping[str, float]

    def __post_init__(self) -> None:
        if not self.query_id.strip() or not self.video_id.strip():
            raise ValueError("refined candidate identifiers must not be empty")
        if self.original_candidate_rank < 1 or self.candidate_frame_id < 0:
            raise ValueError("invalid original candidate rank/frame")
        if self.refined_frame_id is not None and self.refined_frame_id < 0:
            raise ValueError("refined_frame_id must be non-negative when provided")
        if self.window_start_frame is not None and self.window_end_frame is not None:
            if not self.window_start_frame <= self.window_end_frame:
                raise ValueError("refinement window is invalid")
            if self.refined_frame_id is not None and not (
                self.window_start_frame <= self.refined_frame_id <= self.window_end_frame
            ):
                raise ValueError("refined_frame_id is outside the bounded window")
        object.__setattr__(
            self,
            "per_variant_provenance",
            tuple(MappingProxyType(dict(item)) for item in self.per_variant_provenance),
        )
        object.__setattr__(
            self,
            "original_retrieval_provenance",
            MappingProxyType(dict(self.original_retrieval_provenance)),
        )
        object.__setattr__(self, "timings", MappingProxyType(dict(self.timings)))


class QueryRefinementError(RuntimeError):
    """Explicit query-level failure selected by a refinement policy."""
