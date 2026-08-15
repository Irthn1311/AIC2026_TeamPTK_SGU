"""Consolidated Execution Trace and Observability module for system_tai."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass
class StageTiming:
    stage_name: str
    duration_seconds: float
    start_time_offset: float


@dataclass
class ExecutionTrace:
    request_id: str
    query_id: str
    task_type: str  # KIS | Q&A | TRAKE
    timestamp_utc: str = field(
        default_factory=lambda: datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    )
    device: str = "auto"
    stage_timings: list[StageTiming] = field(default_factory=list)
    total_duration_seconds: float = 0.0
    candidate_count: int = 0
    top_candidates: list[dict[str, Any]] = field(default_factory=list)
    evidence_summary: dict[str, Any] = field(default_factory=dict)
    validation_status: str = "VALID"  # VALID | WARNING | ERROR
    validation_messages: list[str] = field(default_factory=list)
    failure_reason: str | None = None
    telemetry: dict[str, Any] = field(default_factory=dict)

    def add_stage(self, name: str, duration_seconds: float, offset: float = 0.0) -> None:
        self.stage_timings.append(
            StageTiming(
                stage_name=name,
                duration_seconds=round(duration_seconds, 4),
                start_time_offset=round(offset, 4),
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save_json(self, output_path: Path | str) -> None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)


class TraceContext:
    """Context manager for tracing stage latencies."""

    def __init__(self, trace: ExecutionTrace, stage_name: str) -> None:
        self.trace = trace
        self.stage_name = stage_name
        self.t0 = 0.0

    def __enter__(self) -> TraceContext:
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        duration = time.perf_counter() - self.t0
        self.trace.add_stage(self.stage_name, duration)
        if exc_val is not None:
            self.trace.validation_status = "ERROR"
            self.trace.failure_reason = str(exc_val)
