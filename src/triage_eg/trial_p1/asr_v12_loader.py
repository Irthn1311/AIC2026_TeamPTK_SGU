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

ASR_V12_SOURCE_TYPE = "ASR_V12_EXACT"
ASR_EXTERNAL_V3_SOURCE_TYPE = "ASR_EXTERNAL_V3_VALIDATED"
ASR_EXTERNAL_V3_PROVENANCE = "VALIDATED_EXTERNAL_NOT_REPRODUCIBLE"

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
                        "chunk_id": str(
                            span.get("chunk_id")
                            or f"{video_id}:{key[1]:.6f}:{key[2]:.6f}"
                        ),
                        "source_type": ASR_V12_SOURCE_TYPE,
                        "matched_terms": [],
                        "provenance": {
                            "source_type": ASR_V12_SOURCE_TYPE,
                            "provenance_level": "EXACT_PINNED_REPRODUCIBLE",
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


@dataclass(frozen=True)
class ASRExternalV3Validation:
    status: str
    source_type: str
    provenance_level: str
    transcript_video_count: int
    transcript_chunk_count: int
    pass_chunk_count: int
    no_speech_chunk_count: int
    lexical_term_count: int
    e5_vector_count: int
    coverage_complete: bool

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


class ASRExternalV3Loader:
    """Fail-closed adapter for the explicitly non-Whisper-exact external V3 bundle."""

    REQUIRED_FILES = (
        "asr_external_v3_transcripts.jsonl",
        "asr_external_v3_lexical_index.json",
        "asr_external_v3_e5_manifest.json",
        "asr_external_v3_video_coverage.json",
        "asr_external_v3_timestamp_audit.json",
        "asr_external_v3_provenance.json",
    )

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve(strict=True)
        missing = [name for name in self.REQUIRED_FILES if not (self.root / name).is_file()]
        if missing:
            raise RuntimeError(f"ASR_EXTERNAL_V3_REQUIRED_FILES_MISSING:{missing}")
        self.transcripts = _read_jsonl(self.root / self.REQUIRED_FILES[0])
        lexical = json.loads((self.root / self.REQUIRED_FILES[1]).read_text(encoding="utf-8"))
        e5 = json.loads((self.root / self.REQUIRED_FILES[2]).read_text(encoding="utf-8"))
        coverage = json.loads((self.root / self.REQUIRED_FILES[3]).read_text(encoding="utf-8"))
        timestamp = json.loads((self.root / self.REQUIRED_FILES[4]).read_text(encoding="utf-8"))
        provenance = json.loads((self.root / self.REQUIRED_FILES[5]).read_text(encoding="utf-8"))
        if provenance.get("source_type") != ASR_EXTERNAL_V3_SOURCE_TYPE:
            raise RuntimeError("ASR_EXTERNAL_V3_SOURCE_TYPE_MISMATCH")
        if provenance.get("provenance_level") != ASR_EXTERNAL_V3_PROVENANCE:
            raise RuntimeError("ASR_EXTERNAL_V3_PROVENANCE_LEVEL_MISMATCH")
        if provenance.get("is_whisper_v12_exact") is not False:
            raise RuntimeError("ASR_EXTERNAL_V3_WHISPER_ALIAS_FORBIDDEN")
        if not str(timestamp.get("status", "")).startswith("PASS") or (
            coverage.get("coverage_complete") is not True
        ):
            raise RuntimeError("ASR_EXTERNAL_V3_AUDIT_GATE_NOT_PASS")
        if not str(e5.get("alignment_status", "")).startswith("PASS"):
            raise RuntimeError("ASR_EXTERNAL_V3_E5_ALIGNMENT_NOT_PASS")
        if lexical.get("source_type") != ASR_EXTERNAL_V3_SOURCE_TYPE:
            raise RuntimeError("ASR_EXTERNAL_V3_LEXICAL_SOURCE_TYPE_MISMATCH")
        chunks = lexical.get("chunks")
        postings = lexical.get("postings")
        if not isinstance(chunks, dict) or not chunks or not isinstance(postings, dict):
            raise RuntimeError("ASR_EXTERNAL_V3_LEXICAL_INDEX_INVALID")
        self.lexical_chunks: dict[str, dict[str, Any]] = chunks
        self.lexical_postings: dict[str, list[str]] = postings
        self.e5_manifest = e5
        self.coverage = coverage
        self.timestamp_audit = timestamp
        self.provenance = provenance
        self._validate_rows()

    def _validate_rows(self) -> None:
        chunk_ids: set[str] = set()
        video_ids: set[str] = set()
        pass_count = 0
        no_speech_count = 0
        for index, row in enumerate(self.transcripts):
            if row.get("source_type") != ASR_EXTERNAL_V3_SOURCE_TYPE:
                raise RuntimeError(f"ASR_EXTERNAL_V3_ROW_SOURCE_TYPE_MISMATCH:{index}")
            _validate_seconds(row, context=f"external_v3:{index}")
            chunk_id = str(row.get("chunk_id", ""))
            video_id = str(row.get("video_id", ""))
            if not chunk_id or chunk_id in chunk_ids or not video_id:
                raise RuntimeError(f"ASR_EXTERNAL_V3_ROW_ID_INVALID:{index}:{chunk_id}")
            chunk_ids.add(chunk_id)
            video_ids.add(video_id)
            if row.get("status") == "PASS":
                pass_count += 1
            elif row.get("status") == "NO_SPEECH" and row.get("is_no_speech") is True:
                no_speech_count += 1
            else:
                raise RuntimeError(f"ASR_EXTERNAL_V3_ROW_STATUS_INVALID:{index}")
        lexical_ids = set(self.lexical_chunks)
        if not lexical_ids.issubset(chunk_ids):
            raise RuntimeError("ASR_EXTERNAL_V3_LEXICAL_CHUNK_UNKNOWN")
        for term, ids in self.lexical_postings.items():
            if not str(term).strip() or not isinstance(ids, list) or not ids:
                raise RuntimeError(f"ASR_EXTERNAL_V3_LEXICAL_POSTING_INVALID:{term!r}")
            if not set(ids).issubset(lexical_ids):
                raise RuntimeError(f"ASR_EXTERNAL_V3_LEXICAL_POSTING_CHUNK_UNKNOWN:{term}")
        expected_videos = int(self.coverage.get("canonical_video_count", -1))
        if len(video_ids) != expected_videos:
            raise RuntimeError(
                f"ASR_EXTERNAL_V3_VIDEO_COVERAGE_MISMATCH:{len(video_ids)}:{expected_videos}"
            )
        vector_count = int(self.e5_manifest.get("vector_count", -1))
        if len(lexical_ids) != vector_count:
            raise RuntimeError(
                f"ASR_EXTERNAL_V3_E5_LEXICAL_COUNT_MISMATCH:{len(lexical_ids)}:{vector_count}"
            )
        self.validation = ASRExternalV3Validation(
            status="PASS",
            source_type=ASR_EXTERNAL_V3_SOURCE_TYPE,
            provenance_level=ASR_EXTERNAL_V3_PROVENANCE,
            transcript_video_count=len(video_ids),
            transcript_chunk_count=len(self.transcripts),
            pass_chunk_count=pass_count,
            no_speech_chunk_count=no_speech_count,
            lexical_term_count=len(self.lexical_postings),
            e5_vector_count=vector_count,
            coverage_complete=True,
        )

    def retrieve_spans(
        self,
        query_text: str,
        *,
        video_ids: set[str] | None = None,
        max_spans: int = 5,
    ) -> list[dict[str, Any]]:
        if max_spans < 1:
            return []
        scored: dict[str, dict[str, Any]] = {}
        query_tokens = tuple(token for token in _tokens(query_text) if len(token) >= 2)
        corpus_size = len(self.lexical_chunks)
        for token in query_tokens:
            posting = self.lexical_postings.get(token, [])
            inverse_document_frequency = math.log((corpus_size + 1) / (len(posting) + 1)) + 1
            for chunk_id in posting:
                chunk = self.lexical_chunks[chunk_id]
                video_id = str(chunk["video_id"])
                if video_ids is not None and video_id not in video_ids:
                    continue
                row = scored.setdefault(
                    chunk_id,
                    {
                        "source_type": ASR_EXTERNAL_V3_SOURCE_TYPE,
                        "video_id": video_id,
                        "chunk_id": chunk_id,
                        "start_seconds": float(chunk["start_seconds"]),
                        "end_seconds": float(chunk["end_seconds"]),
                        "text": str(chunk["text"]),
                        "e5_row_index": chunk.get("e5_row_index"),
                        "matched_terms": [],
                        "lexical_score": 0.0,
                        "retrieval_metadata": {
                            "branch": "LEXICAL",
                            "score_semantics": "UNWEIGHTED_UNIQUE_QUERY_TOKEN_OVERLAP",
                        },
                        "provenance": {
                            "source_type": ASR_EXTERNAL_V3_SOURCE_TYPE,
                            "provenance_level": ASR_EXTERNAL_V3_PROVENANCE,
                            "transcription_engine": "UNKNOWN_EXTERNAL",
                            "transcription_exact_revision": None,
                        },
                    },
                )
                row["matched_terms"].append(token)
                row["lexical_score"] += inverse_document_frequency
        ordered = sorted(
            scored.values(),
            key=lambda row: (
                -row["lexical_score"],
                -len(set(row["matched_terms"])),
                row["video_id"],
                row["start_seconds"],
                row["chunk_id"],
            ),
        )
        return [
            {**row, "matched_terms": sorted(set(row["matched_terms"])), "asr_rank": rank}
            for rank, row in enumerate(ordered[:max_spans], 1)
        ]

    def rank_video_hypotheses(
        self, query_text: str, *, max_videos: int = 100
    ) -> list[dict[str, Any]]:
        spans = self.retrieve_spans(query_text, max_spans=max(len(self.transcripts), 1))
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
                "source_type": ASR_EXTERNAL_V3_SOURCE_TYPE,
            }
            for rank, (video_id, rows) in enumerate(ordered[:max_videos], 1)
        ]

    @staticmethod
    def map_span_to_frame(
        span: dict[str, Any], canonical_mapper: Callable[[str, float], int]
    ) -> dict[str, Any]:
        _validate_seconds(span, context="external_v3_canonical_map_input")
        midpoint = (float(span["start_seconds"]) + float(span["end_seconds"])) / 2
        frame_id = canonical_mapper(str(span["video_id"]), midpoint)
        if not isinstance(frame_id, int) or frame_id < 0:
            raise RuntimeError("ASR_EXTERNAL_V3_CANONICAL_MAPPER_RETURNED_INVALID_FRAME")
        return {
            **span,
            "frame_id": frame_id,
            "frame_mapping_source": "INJECTED_CANONICAL_FRAMEMAP",
        }


def load_asr_evidence(root: str | Path, source_type: str) -> ASRV12Loader | ASRExternalV3Loader:
    """Select a source explicitly; filesystem contents never silently determine the label."""

    if source_type == ASR_V12_SOURCE_TYPE:
        return ASRV12Loader(root)
    if source_type == ASR_EXTERNAL_V3_SOURCE_TYPE:
        return ASRExternalV3Loader(root)
    raise RuntimeError(f"ASR_SOURCE_TYPE_UNSUPPORTED:{source_type}")


__all__ = [
    "ASR_EXTERNAL_V3_SOURCE_TYPE",
    "ASR_V12_SOURCE_TYPE",
    "ASRExternalV3Loader",
    "ASRExternalV3Validation",
    "ASRV12Loader",
    "ASRV12Validation",
    "REQUIRED_FILES",
    "load_asr_evidence",
]
