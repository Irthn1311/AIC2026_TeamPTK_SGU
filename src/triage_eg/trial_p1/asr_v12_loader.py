"""Fail-closed loader and query-local retrieval for the frozen ASR v1.2 bundle."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from triage_eg.fs1_v11.contracts import WHISPER_ID, WHISPER_REVISION

REQUIRED_FILES = (
    "asr_transcripts_v12.jsonl",
    "asr_lexical_index_v12.json",
    "asr_audio_inventory_v12.jsonl",
    "asr_performance_report_v12.json",
)
FORBIDDEN_FRAME_FIELDS = frozenset({"frame_id", "frame_ids", "actual_frame_id"})


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise RuntimeError(f"ASR_V12_JSONL_ROW_NOT_OBJECT:{path.name}:{line_number}")
        rows.append(value)
    return rows


def _tokens(value: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(re.findall(r"\w+", str(value).casefold(), re.UNICODE)))


def _validate_seconds(row: dict[str, Any], *, context: str) -> None:
    forbidden = FORBIDDEN_FRAME_FIELDS.intersection(row)
    if forbidden:
        raise RuntimeError(f"ASR_V12_INFERRED_FRAME_FIELD_FORBIDDEN:{context}:{sorted(forbidden)}")
    try:
        start, end = float(row["start_seconds"]), float(row["end_seconds"])
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(f"ASR_V12_SECONDS_REQUIRED:{context}") from error
    if not math.isfinite(start + end) or start < 0 or end < start:
        raise RuntimeError(f"ASR_V12_SECONDS_INVALID:{context}:{start}:{end}")


@dataclass(frozen=True)
class ASRV12Validation:
    status: str
    model_id: str
    model_revision: str
    inventory_audio_video_count: int
    transcript_video_count: int
    pass_video_count: int
    lexical_term_count: int
    manifest_file_count: int
    coverage_complete: bool
    performance_status: str

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


class ASRV12Loader:
    """Validated ASR evidence stays in seconds until a canonical mapper is injected."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve(strict=True)
        missing = [name for name in REQUIRED_FILES if not (self.root / name).is_file()]
        if missing:
            raise RuntimeError(f"ASR_V12_REQUIRED_FILES_MISSING:{missing}")
        self.inventory = _read_jsonl(self.root / REQUIRED_FILES[2])
        self.transcripts = _read_jsonl(self.root / REQUIRED_FILES[0])
        lexical = json.loads((self.root / REQUIRED_FILES[1]).read_text(encoding="utf-8"))
        performance = json.loads((self.root / REQUIRED_FILES[3]).read_text(encoding="utf-8"))
        if not isinstance(lexical, dict) or not lexical:
            raise RuntimeError("ASR_V12_LEXICAL_INDEX_EMPTY")
        if not isinstance(performance, dict):
            raise RuntimeError("ASR_V12_PERFORMANCE_REPORT_INVALID")
        self.lexical_index: dict[str, list[dict[str, Any]]] = lexical
        self.performance = performance
        self.manifest_paths = tuple(sorted(self.root.glob("*_manifest.json")))
        if not self.manifest_paths:
            raise RuntimeError("ASR_V12_MANIFEST_FILES_MISSING")
        self._transcript_by_video = self._validate_and_index()

    def _validate_and_index(self) -> dict[str, dict[str, Any]]:
        expected = {
            str(row["video_id"])
            for row in self.inventory
            if row.get("has_audio") is True and row.get("probe_status") == "PASS"
        }
        indexed: dict[str, dict[str, Any]] = {}
        pass_ids: set[str] = set()
        for row in self.transcripts:
            video_id = str(row.get("video_id", ""))
            if not video_id or video_id in indexed:
                raise RuntimeError(f"ASR_V12_TRANSCRIPT_VIDEO_ID_INVALID:{video_id}")
            indexed[video_id] = row
            if row.get("status") != "PASS":
                continue
            if row.get("model_id") != WHISPER_ID or row.get("model_revision") != WHISPER_REVISION:
                raise RuntimeError(f"ASR_V12_MODEL_PROVENANCE_MISMATCH:{video_id}")
            previous_end = 0.0
            for index, segment in enumerate(row.get("segments", [])):
                _validate_seconds(segment, context=f"transcript:{video_id}:{index}")
                start, end = float(segment["start_seconds"]), float(segment["end_seconds"])
                if start < previous_end:
                    raise RuntimeError(f"ASR_V12_TIMESTAMPS_NOT_MONOTONIC:{video_id}:{index}")
                previous_end = end
            pass_ids.add(video_id)
        if set(indexed) != expected:
            missing = sorted(expected - set(indexed))
            unexpected = sorted(set(indexed) - expected)
            raise RuntimeError(
                f"ASR_V12_CORPUS_COVERAGE_MISMATCH:missing={missing}:unexpected={unexpected}"
            )
        if not pass_ids:
            raise RuntimeError("ASR_V12_NO_PASS_TRANSCRIPTS")
        for term, spans in self.lexical_index.items():
            if not str(term).strip() or not isinstance(spans, list) or not spans:
                raise RuntimeError(f"ASR_V12_LEXICAL_TERM_INVALID:{term!r}")
            for index, span in enumerate(spans):
                if not isinstance(span, dict):
                    raise RuntimeError(f"ASR_V12_LEXICAL_SPAN_INVALID:{term}:{index}")
                _validate_seconds(span, context=f"lexical:{term}:{index}")
                if str(span.get("video_id", "")) not in pass_ids:
                    raise RuntimeError(f"ASR_V12_LEXICAL_VIDEO_NOT_PASS:{term}:{index}")
        status = str(self.performance.get("status", ""))
        if status != "PASS":
            raise RuntimeError(f"ASR_V12_PERFORMANCE_GATE_NOT_PASS:{status or 'MISSING'}")
        self.validation = ASRV12Validation(
            status="PASS",
            model_id=WHISPER_ID,
            model_revision=WHISPER_REVISION,
            inventory_audio_video_count=len(expected),
            transcript_video_count=len(indexed),
            pass_video_count=len(pass_ids),
            lexical_term_count=len(self.lexical_index),
            manifest_file_count=len(self.manifest_paths),
            coverage_complete=True,
            performance_status=status,
        )
        return indexed

    def retrieve_spans(
        self,
        query_text: str,
        *,
        video_ids: set[str] | None = None,
        max_spans: int = 5,
    ) -> list[dict[str, Any]]:
        """Retrieve bounded local spans; no frame identifiers are materialized here."""

        if max_spans < 1:
            return []
        scores: dict[tuple[str, float, float, str], dict[str, Any]] = {}
        for token in _tokens(query_text):
            for span in self.lexical_index.get(token, []):
                video_id = str(span["video_id"])
                if video_ids is not None and video_id not in video_ids:
                    continue
                key = (
                    video_id,
                    float(span["start_seconds"]),
                    float(span["end_seconds"]),
                    str(span.get("text", "")),
                )
                row = scores.setdefault(
                    key,
                    {
                        "video_id": video_id,
                        "start_seconds": key[1],
                        "end_seconds": key[2],
                        "text": key[3],
                        "matched_terms": [],
                        "provenance": {
                            "video_id": video_id,
                            "source": "ASR_V12_LEXICAL_INDEX",
                            "model_id": WHISPER_ID,
                            "model_revision": WHISPER_REVISION,
                        },
                    },
                )
                row["matched_terms"].append(token)
        ordered = sorted(
            scores.values(),
            key=lambda row: (
                -len(set(row["matched_terms"])),
                row["video_id"],
                row["start_seconds"],
                row["end_seconds"],
                row["text"],
            ),
        )
        return [
            {**row, "matched_terms": sorted(set(row["matched_terms"])), "asr_rank": rank}
            for rank, row in enumerate(ordered[:max_spans], 1)
        ]

    def rank_video_hypotheses(
        self, query_text: str, *, max_videos: int = 100
    ) -> list[dict[str, Any]]:
        """Rank-level ASR branch for KIS; never adds raw ASR scores to visual scores."""

        spans = self.retrieve_spans(query_text, max_spans=max(len(self.transcripts) * 20, 1))
        grouped: dict[str, list[dict[str, Any]]] = {}
        for span in spans:
            grouped.setdefault(str(span["video_id"]), []).append(span)
        ordered = sorted(
            grouped.items(),
            key=lambda item: (
                -max(len(span["matched_terms"]) for span in item[1]),
                -len(item[1]),
                item[0],
            ),
        )
        return [
            {
                "video_id": video_id,
                "rank": rank,
                "best_span": min(rows, key=lambda row: (row["asr_rank"], row["start_seconds"])),
                "fusion_contract": "RANK_LEVEL_COMPLEMENTARY_BRANCH_ONLY",
            }
            for rank, (video_id, rows) in enumerate(ordered[:max_videos], 1)
        ]

    @staticmethod
    def map_span_to_frame(
        span: dict[str, Any], canonical_mapper: Callable[[str, float], int]
    ) -> dict[str, Any]:
        """The injected FrameMap/BTC mapper is the only permitted seconds-to-frame boundary."""

        _validate_seconds(span, context="canonical_map_input")
        video_id = str(span["video_id"])
        midpoint_seconds = (float(span["start_seconds"]) + float(span["end_seconds"])) / 2
        frame_id = canonical_mapper(video_id, midpoint_seconds)
        if not isinstance(frame_id, int) or frame_id < 0:
            raise RuntimeError("ASR_V12_CANONICAL_MAPPER_RETURNED_INVALID_FRAME")
        return {**span, "frame_id": frame_id, "frame_mapping_source": "INJECTED_CANONICAL_FRAMEMAP"}


__all__ = ["ASRV12Loader", "ASRV12Validation", "REQUIRED_FILES"]
