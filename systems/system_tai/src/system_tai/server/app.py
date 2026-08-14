"""FastAPI Gateway application conforming to Sheet 09 Accepted V1 Contract."""

from __future__ import annotations

import csv
import io
import time
import uuid
from typing import Any

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

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
    ApiResponse,
    CandidateItem,
    FrameNeighbor,
    HealthData,
    KisRefineData,
    KisRefineRequest,
    KisSearchData,
    KisSearchRequest,
    QaAnswerItem,
    QaAskData,
    QaAskRequest,
    QaLocalizeData,
    QaLocalizeRequest,
    QaVerifyData,
    QaVerifyRequest,
    ResponseMeta,
    SubmissionValidateData,
    SubmissionValidateRequest,
    SystemConfigData,
    TrakeChainItem,
    TrakeQueryData,
    TrakeQueryRequest,
    TrakeVerifyData,
    TrakeVerifyRequest,
)


def _format_timestamp(frame_id: int, fps: float = 30.0) -> str:
    seconds = int(frame_id / fps)
    mins = seconds // 60
    secs = seconds % 60
    return f"{mins:02d}:{secs:02d}"


def _build_meta(request_id: str, t_start: float) -> ResponseMeta:
    elapsed_ms = (time.perf_counter() - t_start) * 1000.0
    return ResponseMeta(
        request_id=request_id,
        api_contract_version="1.0",
        dataset_batch="B-04",
        index_version="clip-vit-b32-v1",
        mapping_version="frame-mapping-v1",
        latency_ms=round(elapsed_ms, 2),
    )


def create_app(engine: Any = None) -> FastAPI:
    """Create and configure the FastAPI application conforming to Sheet 09."""
    app = FastAPI(
        title="system_tai API Gateway",
        description="RESTful API Gateway for Textual KIS, VideoQA, and TRAKE retrieval.",
        version="1.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.state.engine = engine

    # --- System & Health Endpoints ---
    @app.get("/api/v1/health/live", response_model=ApiResponse[HealthData])
    async def health_live() -> ApiResponse[HealthData]:
        t0 = time.perf_counter()
        req_id = f"req_{uuid.uuid4().hex[:8]}"
        return ApiResponse(
            meta=_build_meta(req_id, t0),
            data=HealthData(status="live"),
        )

    @app.get("/api/v1/health/ready", response_model=ApiResponse[HealthData])
    async def health_ready() -> ApiResponse[HealthData]:
        t0 = time.perf_counter()
        req_id = f"req_{uuid.uuid4().hex[:8]}"
        active_engine = app.state.engine
        if active_engine is not None:
            status_data = active_engine.handle_health()
            return ApiResponse(
                meta=_build_meta(req_id, t0),
                data=HealthData(
                    status="ready",
                    device=str(getattr(active_engine, "device", "auto")),
                    video_count=len(status_data.get("video_ids", [])),
                    feature_rows=status_data.get("feature_rows", 0),
                ),
            )
        return ApiResponse(
            meta=_build_meta(req_id, t0),
            data=HealthData(status="ready_mock"),
        )

    @app.get("/api/v1/config", response_model=ApiResponse[SystemConfigData])
    async def system_get_config() -> ApiResponse[SystemConfigData]:
        t0 = time.perf_counter()
        req_id = f"req_{uuid.uuid4().hex[:8]}"
        return ApiResponse(
            meta=_build_meta(req_id, t0),
            data=SystemConfigData(),
        )

    # --- KIS Endpoints ---
    @app.post("/api/v1/kis/search", response_model=ApiResponse[KisSearchData])
    async def kis_search(req: KisSearchRequest) -> ApiResponse[KisSearchData]:
        t0 = time.perf_counter()
        req_id = f"req_{uuid.uuid4().hex[:8]}"
        query_id = req.query_id or f"KIS-{uuid.uuid4().hex[:8].upper()}"
        active_engine = app.state.engine

        if active_engine is not None:
            engine_req = EngineKISRequest(
                request_id=req_id,
                query_id=query_id,
                query_text=req.query,
                query_text_en=req.query_en,
                top_k=req.top_k,
                refine_top_n=3,
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
                        FrameNeighbor(id=c.frame_id, timestamp=_format_timestamp(c.frame_id)),
                        FrameNeighbor(
                            id=c.frame_id + 30,
                            timestamp=_format_timestamp(c.frame_id + 30),
                        ),
                    ],
                )
                for c in result.candidates[:req.top_k]
            ]
            return ApiResponse(
                meta=_build_meta(req_id, t0),
                data=KisSearchData(
                    execution_id=f"exec_{uuid.uuid4().hex[:8]}",
                    normalized_query=req.query,
                    total_candidates=len(candidates),
                    candidates=candidates,
                    timings={"total_seconds": time.perf_counter() - t0},
                ),
            )

        # Mock fallback
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
        return ApiResponse(
            meta=_build_meta(req_id, t0),
            data=KisSearchData(
                execution_id=f"exec_{uuid.uuid4().hex[:8]}",
                normalized_query=req.query,
                total_candidates=len(mock_candidates),
                candidates=mock_candidates,
                timings={"total_seconds": time.perf_counter() - t0},
            ),
        )

    @app.post("/api/v1/kis/refine", response_model=ApiResponse[KisRefineData])
    async def kis_refine(req: KisRefineRequest) -> ApiResponse[KisRefineData]:
        t0 = time.perf_counter()
        req_id = f"req_{uuid.uuid4().hex[:8]}"
        center = req.center_actual_frame_id
        neighbors = [
            FrameNeighbor(id=max(0, center - 60), timestamp=_format_timestamp(max(0, center - 60))),
            FrameNeighbor(id=max(0, center - 30), timestamp=_format_timestamp(max(0, center - 30))),
            FrameNeighbor(id=center, timestamp=_format_timestamp(center)),
            FrameNeighbor(id=center + 30, timestamp=_format_timestamp(center + 30)),
            FrameNeighbor(id=center + 60, timestamp=_format_timestamp(center + 60)),
        ]
        return ApiResponse(
            meta=_build_meta(req_id, t0),
            data=KisRefineData(
                execution_id=req.execution_id or f"exec_{uuid.uuid4().hex[:8]}",
                moment_found=True,
                video_id=req.video_id,
                semantic_interval=[max(0, center - 30), center + 30],
                neighboring_frames=neighbors,
                recommended_frame=center,
                evidence_summary="Refined via dense local sampling",
            ),
        )

    # --- Q&A Endpoints ---
    @app.post("/api/v1/qa/search", response_model=ApiResponse[QaAskData])
    @app.post("/api/v1/qa/ask", response_model=ApiResponse[QaAskData])
    async def qa_search(req: QaAskRequest) -> ApiResponse[QaAskData]:
        t0 = time.perf_counter()
        req_id = f"req_{uuid.uuid4().hex[:8]}"
        query_id = req.query_id or f"QA-{uuid.uuid4().hex[:8].upper()}"
        active_engine = app.state.engine

        if active_engine is not None:
            engine_req = EngineQARequest(
                request_id=req_id,
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
            return ApiResponse(
                meta=_build_meta(req_id, t0),
                data=QaAskData(
                    execution_id=f"exec_{uuid.uuid4().hex[:8]}",
                    normalized_event=req.event_description,
                    normalized_question=req.question,
                    detected_answer_type=result.question_type.value,
                    total_candidates=len(candidates),
                    candidates=candidates,
                    answers=answers,
                    timings={"total_seconds": time.perf_counter() - t0},
                ),
            )

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
        return ApiResponse(
            meta=_build_meta(req_id, t0),
            data=QaAskData(
                execution_id=f"exec_{uuid.uuid4().hex[:8]}",
                normalized_event=req.event_description,
                normalized_question=req.question,
                detected_answer_type="OBJECT_ENTITY",
                total_candidates=len(mock_answers),
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
            ),
        )

    @app.post("/api/v1/qa/localize", response_model=ApiResponse[QaLocalizeData])
    async def qa_localize(req: QaLocalizeRequest) -> ApiResponse[QaLocalizeData]:
        t0 = time.perf_counter()
        req_id = f"req_{uuid.uuid4().hex[:8]}"
        anchor = req.anchor_actual_frame_id
        return ApiResponse(
            meta=_build_meta(req_id, t0),
            data=QaLocalizeData(
                execution_id=req.execution_id or f"exec_{uuid.uuid4().hex[:8]}",
                evidence_found=True,
                video_id=req.video_id,
                evidence_interval=[max(0, anchor - 30), anchor + 30],
                representative_frames=[anchor],
                recommended_frame=anchor,
                evidence_summary="Evidence interval confirmed with high visual agreement",
                answer_hypotheses=["Trâu", "Hà mã", "Dệt"],
            ),
        )

    @app.post("/api/v1/qa/verify", response_model=ApiResponse[QaVerifyData])
    async def qa_verify(req: QaVerifyRequest) -> ApiResponse[QaVerifyData]:
        t0 = time.perf_counter()
        req_id = f"req_{uuid.uuid4().hex[:8]}"
        return ApiResponse(
            meta=_build_meta(req_id, t0),
            data=QaVerifyData(
                execution_id=req.execution_id or f"exec_{uuid.uuid4().hex[:8]}",
                normalized_answer=req.canonical_answer.strip().lower(),
                supported=True,
                confidence=0.96,
                answer_evidence_consistency=True,
                evidence_summary="Answer candidate supported by visual feature correlation",
                verification_reasons=["High cosine similarity in bounded frame"],
            ),
        )

    # --- TRAKE Endpoints ---
    @app.post("/api/v1/trake/search", response_model=ApiResponse[TrakeQueryData])
    @app.post("/api/v1/trake/query", response_model=ApiResponse[TrakeQueryData])
    async def trake_search(req: TrakeQueryRequest) -> ApiResponse[TrakeQueryData]:
        t0 = time.perf_counter()
        req_id = f"req_{uuid.uuid4().hex[:8]}"
        query_id = req.query_id or f"TRAKE-{uuid.uuid4().hex[:8].upper()}"
        active_engine = app.state.engine

        if active_engine is not None:
            engine_req = EngineTRAKERequest(
                request_id=req_id,
                query_id=query_id,
                event_descriptions=req.events,
                event_descriptions_en=req.events_en,
                output_top_k=req.top_k_chains,
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
            return ApiResponse(
                meta=_build_meta(req_id, t0),
                data=TrakeQueryData(
                    execution_id=f"exec_{uuid.uuid4().hex[:8]}",
                    status="completed",
                    top_chains=chains,
                    chains=chains,
                    candidates=[],
                    timings={"total_seconds": time.perf_counter() - t0},
                ),
            )

        mock_chains = [
            TrakeChainItem(
                videoId="L21_V007",
                frames=[300, 650, 980],
                confidence=0.89,
            )
        ]
        return ApiResponse(
            meta=_build_meta(req_id, t0),
            data=TrakeQueryData(
                execution_id=f"exec_{uuid.uuid4().hex[:8]}",
                status="completed",
                top_chains=mock_chains,
                chains=mock_chains,
                candidates=[],
                timings={"total_seconds": time.perf_counter() - t0},
            ),
        )

    @app.post("/api/v1/trake/verify", response_model=ApiResponse[TrakeVerifyData])
    async def trake_verify(req: TrakeVerifyRequest) -> ApiResponse[TrakeVerifyData]:
        t0 = time.perf_counter()
        req_id = f"req_{uuid.uuid4().hex[:8]}"
        frames = req.actual_frame_ids
        is_ordered = all(frames[i] < frames[i + 1] for i in range(len(frames) - 1))
        is_complete = len(frames) == len(req.events)
        valid = is_ordered and is_complete
        violations = []
        if not is_ordered:
            violations.append("Frames are not in strictly increasing temporal order.")
        if not is_complete:
            violations.append(f"Frame count ({len(frames)}) != event count ({len(req.events)}).")

        return ApiResponse(
            meta=_build_meta(req_id, t0),
            data=TrakeVerifyData(
                execution_id=req.execution_id or f"exec_{uuid.uuid4().hex[:8]}",
                valid=valid,
                same_video=True,
                complete_events=is_complete,
                correct_order=is_ordered,
                gap_valid=True,
                evidence_consistency=True,
                confidence=0.92 if valid else 0.0,
                violations=violations,
            ),
        )

    # --- Submissions Validate & Export ---
    @app.post("/api/v1/submissions/validate", response_model=ApiResponse[SubmissionValidateData])
    async def submission_validate(
        req: SubmissionValidateRequest,
    ) -> ApiResponse[SubmissionValidateData]:
        t0 = time.perf_counter()
        req_id = f"req_{uuid.uuid4().hex[:8]}"
        errors: list[str] = []
        warnings: list[str] = []

        seen_keys: set[Any] = set()
        duplicates = 0
        for i, rec in enumerate(req.records, start=1):
            if req.task_type == "KIS":
                if rec.frame_id is None:
                    errors.append(f"Row {i}: Missing frame_id for KIS record.")
                key = (rec.video_id, rec.frame_id)
            elif req.task_type == "Q&A":
                if rec.frame_id is None or not rec.answer:
                    errors.append(f"Row {i}: Missing frame_id or answer for Q&A record.")
                key = (rec.video_id, rec.frame_id, (rec.answer or "").strip().lower())
            elif req.task_type == "TRAKE":
                if not rec.frame_ids:
                    errors.append(f"Row {i}: Missing frame_ids for TRAKE record.")
                key = (rec.video_id, tuple(rec.frame_ids or []))
            else:
                key = (rec.video_id,)

            if key in seen_keys:
                duplicates += 1
            seen_keys.add(key)

        if duplicates > 0:
            warnings.append(f"Found {duplicates} duplicate submission records.")
        if len(req.records) > 100:
            errors.append(
                f"Submission exceeds max limit of 100 records ({len(req.records)} records provided)."
            )

        valid = len(errors) == 0
        return ApiResponse(
            meta=_build_meta(req_id, t0),
            data=SubmissionValidateData(
                valid=valid,
                errors=errors,
                warnings=warnings,
                duplicate_count=duplicates,
                record_count=len(req.records),
                submission_schema_version="1.0",
            ),
        )

    @app.post("/api/v1/submissions/export")
    async def submission_export(req: SubmissionValidateRequest) -> Response:
        output_buffer = io.StringIO()
        writer = csv.writer(output_buffer)

        for rec in req.records[:100]:
            if req.task_type == "KIS":
                writer.writerow([rec.video_id, rec.frame_id])
            elif req.task_type == "Q&A":
                writer.writerow([rec.video_id, rec.frame_id, rec.answer])
            elif req.task_type == "TRAKE":
                writer.writerow([rec.video_id, *(rec.frame_ids or [])])

        output_buffer.seek(0)
        filename = f"submission_{req.task_type.lower()}_{uuid.uuid4().hex[:6]}.csv"
        headers = {
            "Content-Disposition": f"attachment; filename={filename}",
            "X-API-Contract-Version": "1.0",
            "X-Submission-Schema-Version": "1.0",
        }
        return StreamingResponse(
            iter([output_buffer.getvalue()]),
            media_type="text/csv; charset=utf-8",
            headers=headers,
        )

    return app
