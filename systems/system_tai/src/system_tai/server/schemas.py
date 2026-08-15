"""Pydantic schemas for the system_tai FastAPI gateway conforming to Sheet 09 Accepted V1."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Generic, Literal, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


class RequestContext(BaseModel):
    dataset_batch: str = "B-04"
    expected_index_version: str = "clip-vit-b32-v1"
    expected_mapping_version: str = "frame-mapping-v1"


class ResponseMeta(BaseModel):
    request_id: str
    api_contract_version: str = "1.0"
    dataset_batch: str = "B-04"
    index_version: str = "clip-vit-b32-v1"
    mapping_version: str = "frame-mapping-v1"
    latency_ms: float = 0.0
    server_time: str = Field(
        default_factory=lambda: datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    )


class ApiResponse(BaseModel, Generic[T]):
    meta: ResponseMeta
    data: T


# --- Common Visual DTOs ---
class FrameNeighbor(BaseModel):
    id: int
    timestamp: str


class CandidateItem(BaseModel):
    videoId: str
    frameId: int
    timestamp: str
    score: float
    badges: list[str] = Field(default_factory=list)
    neighbors: list[FrameNeighbor] = Field(default_factory=list)


# --- System Config & Health ---
class SystemConfigData(BaseModel):
    supported_tasks: list[str] = Field(default_factory=lambda: ["KIS", "Q&A", "TRAKE"])
    task_schemas: dict[str, Any] = Field(
        default_factory=lambda: {
            "KIS": ["video_id", "frame_id"],
            "Q&A": ["video_id", "frame_id", "answer"],
            "TRAKE": ["video_id", "frame_id_1...frame_id_n"],
        }
    )
    max_answers: int = 100
    interaction_modes: list[str] = Field(
        default_factory=lambda: ["interactive", "automatic"]
    )
    processing_modes: list[str] = Field(default_factory=lambda: ["sync", "async"])
    capabilities: dict[str, bool] = Field(
        default_factory=lambda: {
            "visual": True,
            "ocr": True,
            "object": True,
            "metadata": True,
            "asr": True,
            "temporal_refinement": True,
        }
    )


class HealthData(BaseModel):
    status: str = "live"
    device: str = "auto"
    video_count: int = 873
    feature_rows: int = 177321


# --- KIS Search & Refine ---
class KisSearchRequest(BaseModel):
    query_id: str | None = None
    request_context: RequestContext = Field(default_factory=RequestContext)
    query: str
    query_en: str | None = None
    filters: list[str] = Field(default_factory=list)
    query_variants: list[str] = Field(default_factory=list)
    top_k: int = 100
    interaction_mode: str = "interactive"


class KisSearchData(BaseModel):
    execution_id: str
    normalized_query: str
    total_candidates: int
    candidates: list[CandidateItem] = Field(default_factory=list)
    timings: dict[str, float] = Field(default_factory=dict)


class KisRefineRequest(BaseModel):
    execution_id: str | None = None
    request_context: RequestContext = Field(default_factory=RequestContext)
    video_id: str
    center_actual_frame_id: int
    window_seconds: float = 1.0


class KisRefineData(BaseModel):
    execution_id: str
    moment_found: bool = True
    video_id: str
    semantic_interval: list[int] = Field(default_factory=list)
    neighboring_frames: list[FrameNeighbor] = Field(default_factory=list)
    recommended_frame: int
    evidence_summary: str


# --- Q&A Ask, Localize & Verify ---
class QaAskRequest(BaseModel):
    query_id: str | None = None
    request_context: RequestContext = Field(default_factory=RequestContext)
    event_description: str
    question: str
    event_description_en: str | None = None
    question_en: str | None = None
    temporal_relation: str = "during"
    suggested_answer_type: str = "auto"
    top_k: int = 100
    interaction_mode: str = "interactive"


class QaAnswerItem(BaseModel):
    videoId: str
    frameId: int
    answer: str
    confidence: float
    validation: str = "VALID"


class QaAskData(BaseModel):
    execution_id: str
    normalized_event: str
    normalized_question: str
    detected_answer_type: str = "auto"
    total_candidates: int
    candidates: list[CandidateItem] = Field(default_factory=list)
    answers: list[QaAnswerItem] = Field(default_factory=list)
    timings: dict[str, float] = Field(default_factory=dict)


class QaLocalizeRequest(BaseModel):
    execution_id: str | None = None
    request_context: RequestContext = Field(default_factory=RequestContext)
    video_id: str
    anchor_actual_frame_id: int
    temporal_relation: str = "during"


class QaLocalizeData(BaseModel):
    execution_id: str
    evidence_found: bool = True
    video_id: str
    evidence_interval: list[int] = Field(default_factory=list)
    representative_frames: list[int] = Field(default_factory=list)
    recommended_frame: int
    evidence_summary: str
    answer_hypotheses: list[dict[str, Any]] = Field(default_factory=list)


class QaVerifyRequest(BaseModel):
    execution_id: str | None = None
    request_context: RequestContext = Field(default_factory=RequestContext)
    video_id: str
    actual_frame_id: int
    canonical_answer: str


class QaVerifyData(BaseModel):
    execution_id: str
    normalized_answer: str
    supported: bool = True
    confidence: float = 1.0
    answer_evidence_consistency: bool = True
    evidence_summary: str
    verification_reasons: list[str] = Field(default_factory=list)


# --- TRAKE Search & Verify ---
class TrakeQueryRequest(BaseModel):
    query_id: str | None = None
    request_context: RequestContext = Field(default_factory=RequestContext)
    events: list[str]
    events_en: list[str] | None = None
    top_k_videos: int = 32
    top_k_per_event: int = 10
    top_k_chains: int = 100
    beam_width: int = 100
    interaction_mode: str = "interactive"


class TrakeChainItem(BaseModel):
    videoId: str
    frames: list[int]
    confidence: float


class TrakeQueryData(BaseModel):
    execution_id: str
    status: str = "completed"
    top_chains: list[TrakeChainItem] = Field(default_factory=list)
    chains: list[TrakeChainItem] = Field(default_factory=list)
    candidates: list[CandidateItem] = Field(default_factory=list)
    timings: dict[str, float] = Field(default_factory=dict)


class TrakeVerifyRequest(BaseModel):
    execution_id: str | None = None
    request_context: RequestContext = Field(default_factory=RequestContext)
    video_id: str
    events: list[str]
    actual_frame_ids: list[int]


class TrakeVerifyData(BaseModel):
    execution_id: str
    valid: bool = True
    same_video: bool = True
    complete_events: bool = True
    correct_order: bool = True
    gap_valid: bool = True
    evidence_consistency: bool = True
    confidence: float = 1.0
    violations: list[str] = Field(default_factory=list)


# --- Evidence & Video Frame Detail DTOs ---
class EvidenceDetailData(BaseModel):
    video_id: str
    actual_frame_id: int
    timestamp: str
    visual_feature_available: bool = True
    ocr_text: str | None = None
    asr_transcript: str | None = None
    object_detections: list[dict[str, Any]] = Field(default_factory=list)
    neighboring_keyframes: list[int] = Field(default_factory=list)


class VideoFrameItem(BaseModel):
    actual_frame_id: int
    keyframe_order: int
    timestamp: str
    pts_time: float = 0.0


class VideoFramesData(BaseModel):
    video_id: str
    fps: float = 25.0
    duration_seconds: float = 0.0
    total_frames: int = 0
    keyframe_count: int = 0
    frames: list[VideoFrameItem] = Field(default_factory=list)


# --- Submission Validate & Export ---
class SubmissionRecord(BaseModel):
    video_id: str
    frame_id: int | None = None
    frame_ids: list[int] | None = None
    answer: str | None = None


class SubmissionValidateRequest(BaseModel):
    request_context: RequestContext = Field(default_factory=RequestContext)
    task_type: Literal["KIS", "Q&A", "TRAKE"]
    records: list[SubmissionRecord]


class SubmissionValidateData(BaseModel):
    valid: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    duplicate_count: int = 0
    record_count: int = 0
    submission_schema_version: str = "1.0"
