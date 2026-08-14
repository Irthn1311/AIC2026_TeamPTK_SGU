"""Pydantic schemas for the system_tai FastAPI gateway."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


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


# --- KIS Schemas ---
class KisSearchRequest(BaseModel):
    query: str
    query_en: str | None = None
    top_k: int = 100
    refine_top_n: int = 3


class KisSearchResponse(BaseModel):
    query_id: str
    query: str
    candidates: list[CandidateItem]
    timings: dict[str, float] = Field(default_factory=dict)


# --- QA Schemas ---
class QaAskRequest(BaseModel):
    event_description: str
    question: str
    event_description_en: str | None = None
    question_en: str | None = None
    top_k: int = 100


class QaAnswerItem(BaseModel):
    videoId: str
    frameId: int
    answer: str
    confidence: float
    validation: Literal["VALID", "INVALID"] = "VALID"


class QaAskResponse(BaseModel):
    query_id: str
    question_type: str
    candidates: list[CandidateItem]
    answers: list[QaAnswerItem]
    timings: dict[str, float] = Field(default_factory=dict)


# --- TRAKE Schemas ---
class TrakeQueryRequest(BaseModel):
    events: list[str]
    events_en: list[str] | None = None
    top_k: int = 100


class TrakeChainItem(BaseModel):
    videoId: str
    frames: list[int]
    confidence: float


class TrakeQueryResponse(BaseModel):
    query_id: str
    chains: list[TrakeChainItem]
    candidates: list[CandidateItem] = Field(default_factory=list)
    timings: dict[str, float] = Field(default_factory=dict)


# --- Health Schema ---
class HealthResponse(BaseModel):
    status: str
    device: str
    active_tasks: list[str]
    video_count: int
    feature_rows: int
