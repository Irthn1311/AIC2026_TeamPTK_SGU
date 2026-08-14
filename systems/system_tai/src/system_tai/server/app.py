"""FastAPI Gateway application for system_tai."""

from __future__ import annotations

import time
import uuid
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from system_tai.kis.session_schema import (
    QAQueryRequest as EngineQARequest,
)
from system_tai.kis.session_schema import (
    QueryRequest as EngineKISRequest,
)
from system_tai.kis.session_schema import (
    TRAKEQueryRequest as EngineTRAKERequest,
)

from .schemas import (
    CandidateItem,
    FrameNeighbor,
    HealthResponse,
    KisSearchRequest,
    KisSearchResponse,
    QaAnswerItem,
    QaAskRequest,
    QaAskResponse,
    TrakeChainItem,
    TrakeQueryRequest,
    TrakeQueryResponse,
)


def _format_timestamp(frame_id: int, fps: float = 30.0) -> str:
    seconds = int(frame_id / fps)
    mins = seconds // 60
    secs = seconds % 60
    return f"{mins:02d}:{secs:02d}"


def create_app(engine: Any = None) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        engine: Optional KISRuntimeSessionEngine instance. If None, mock mode is active.
    """
    app = FastAPI(
        title="system_tai API Gateway",
        description="RESTful API Gateway for Textual KIS, VideoQA, and TRAKE retrieval.",
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # State container on app
    app.state.engine = engine

    @app.get("/api/v1/health", response_model=HealthResponse)
    async def health_check() -> HealthResponse:
        active_engine = app.state.engine
        if active_engine is not None:
            status_data = active_engine.handle_health()
            return HealthResponse(
                status="ready",
                device=str(getattr(active_engine, "device", "auto")),
                active_tasks=["KIS", "Q&A", "TRAKE"],
                video_count=len(status_data.get("video_ids", [])),
                feature_rows=status_data.get("feature_rows", 0),
            )
        return HealthResponse(
            status="ready_mock",
            device="cpu",
            active_tasks=["KIS", "Q&A", "TRAKE"],
            video_count=873,
            feature_rows=177321,
        )

    @app.post("/api/v1/kis/search", response_model=KisSearchResponse)
    async def search_kis(req: KisSearchRequest) -> KisSearchResponse:
        t0 = time.perf_counter()
        query_id = f"KIS-{uuid.uuid4().hex[:8].upper()}"
        active_engine = app.state.engine

        if active_engine is not None:
            engine_req = EngineKISRequest(
                request_id=f"req-{uuid.uuid4().hex[:6]}",
                query_id=query_id,
                query_text=req.query,
                query_text_en=req.query_en,
                top_k=req.top_k,
                refine_top_n=req.refine_top_n,
            )
            result = active_engine.handle_query(engine_req)
            candidates = [
                CandidateItem(
                    videoId=c.video_id,
                    frameId=c.frame_id,
                    timestamp=_format_timestamp(c.frame_id),
                    score=float(c.score),
                    badges=[f"Rank #{c.rank}"],
                    neighbors=[
                        FrameNeighbor(
                            id=max(0, c.frame_id - 30),
                            timestamp=_format_timestamp(max(0, c.frame_id - 30)),
                        ),
                        FrameNeighbor(
                            id=c.frame_id,
                            timestamp=_format_timestamp(c.frame_id),
                        ),
                        FrameNeighbor(
                            id=c.frame_id + 30,
                            timestamp=_format_timestamp(c.frame_id + 30),
                        ),
                    ],
                )
                for c in result.candidates[:req.top_k]
            ]
            return KisSearchResponse(
                query_id=query_id,
                query=req.query,
                candidates=candidates,
                timings={"total_seconds": time.perf_counter() - t0},
            )

        # Mock fallback when engine is not booted
        mock_candidates = [
            CandidateItem(
                videoId="L21_V001",
                frameId=2250,
                timestamp="01:15",
                score=0.92,
                badges=["Top Pick", "Visual Focus"],
                neighbors=[
                    FrameNeighbor(id=2220, timestamp="01:14"),
                    FrameNeighbor(id=2250, timestamp="01:15"),
                    FrameNeighbor(id=2280, timestamp="01:16"),
                ],
            ),
            CandidateItem(
                videoId="L21_V005",
                frameId=1440,
                timestamp="00:48",
                score=0.88,
                badges=["High Confidence"],
                neighbors=[
                    FrameNeighbor(id=1410, timestamp="00:47"),
                    FrameNeighbor(id=1440, timestamp="00:48"),
                    FrameNeighbor(id=1470, timestamp="00:49"),
                ],
            ),
        ]
        return KisSearchResponse(
            query_id=query_id,
            query=req.query,
            candidates=mock_candidates,
            timings={"total_seconds": time.perf_counter() - t0},
        )

    @app.post("/api/v1/qa/ask", response_model=QaAskResponse)
    async def ask_qa(req: QaAskRequest) -> QaAskResponse:
        t0 = time.perf_counter()
        query_id = f"QA-{uuid.uuid4().hex[:8].upper()}"
        active_engine = app.state.engine

        if active_engine is not None:
            engine_req = EngineQARequest(
                request_id=f"req-{uuid.uuid4().hex[:6]}",
                query_id=query_id,
                event_description=req.event_description,
                question=req.question,
                event_description_en=req.event_description_en,
                question_en=req.question_en,
                output_top_k=req.top_k,
            )
            result = active_engine.handle_qa_query(engine_req)
            answers = [
                QaAnswerItem(
                    videoId=p.video_id,
                    frameId=p.frame_id,
                    answer=p.answer,
                    confidence=float(p.confidence) if p.confidence is not None else 1.0,
                    validation="VALID",
                )
                for p in result.predictions
            ]
            candidates = [
                CandidateItem(
                    videoId=a.videoId,
                    frameId=a.frameId,
                    timestamp=_format_timestamp(a.frameId),
                    score=a.confidence,
                    badges=[f"Answer: {a.answer}"],
                    neighbors=[],
                )
                for a in answers
            ]
            return QaAskResponse(
                query_id=query_id,
                question_type=result.question_type.value,
                candidates=candidates,
                answers=answers,
                timings={"total_seconds": time.perf_counter() - t0},
            )

        # Mock fallback
        mock_answers = [
            QaAnswerItem(
                videoId="L21_V005",
                frameId=1440,
                answer="Trâu",
                confidence=0.95,
                validation="VALID",
            ),
            QaAnswerItem(
                videoId="L21_V016",
                frameId=890,
                answer="Dệt",
                confidence=0.91,
                validation="VALID",
            ),
        ]
        return QaAskResponse(
            query_id=query_id,
            question_type="OBJECT_ENTITY",
            candidates=[
                CandidateItem(
                    videoId=a.videoId,
                    frameId=a.frameId,
                    timestamp=_format_timestamp(a.frameId),
                    score=a.confidence,
                    badges=[f"Answer: {a.answer}"],
                    neighbors=[],
                )
                for a in mock_answers
            ],
            answers=mock_answers,
            timings={"total_seconds": time.perf_counter() - t0},
        )

    @app.post("/api/v1/trake/query", response_model=TrakeQueryResponse)
    async def query_trake(req: TrakeQueryRequest) -> TrakeQueryResponse:
        t0 = time.perf_counter()
        query_id = f"TRAKE-{uuid.uuid4().hex[:8].upper()}"
        active_engine = app.state.engine

        if active_engine is not None:
            engine_req = EngineTRAKERequest(
                request_id=f"req-{uuid.uuid4().hex[:6]}",
                query_id=query_id,
                event_descriptions=req.events,
                event_descriptions_en=req.events_en,
                output_top_k=req.top_k,
            )
            result = active_engine.handle_trake_query(engine_req)
            chains = [
                TrakeChainItem(
                    videoId=p.video_id,
                    frames=list(p.frame_ids),
                    confidence=float(p.confidence) if p.confidence is not None else 1.0,
                )
                for p in result.predictions
            ]
            return TrakeQueryResponse(
                query_id=query_id,
                chains=chains,
                candidates=[],
                timings={"total_seconds": time.perf_counter() - t0},
            )

        # Mock fallback
        mock_chains = [
            TrakeChainItem(
                videoId="L21_V007",
                frames=[300, 650, 980],
                confidence=0.89,
            )
        ]
        return TrakeQueryResponse(
            query_id=query_id,
            chains=mock_chains,
            candidates=[],
            timings={"total_seconds": time.perf_counter() - t0},
        )

    return app
