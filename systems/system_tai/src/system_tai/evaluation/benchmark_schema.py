"""Typed, immutable schemas for the Phase 2.5 KIS benchmark."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml


class BenchmarkLanguage(StrEnum):
    VIETNAMESE = "vi"
    ENGLISH = "en"


class VariantType(StrEnum):
    VIETNAMESE_DIRECT = "vietnamese_direct"
    ENGLISH_TRANSLATION = "english_translation"
    ENGLISH_EXPANSION = "english_expansion"


class AnnotationStatus(StrEnum):
    DRAFT = "draft"
    VERIFIED = "verified"


@dataclass(frozen=True, slots=True)
class BenchmarkIssue:
    code: str
    message: str
    query_id: str | None = None
    field: str | None = None


@dataclass(frozen=True, slots=True)
class RelevantFrame:
    """A human-authored positive in the official BTC frame coordinate."""

    video_id: str
    frame_id: int


@dataclass(frozen=True, slots=True)
class BenchmarkQuery:
    query_id: str
    language: BenchmarkLanguage
    text: str
    semantic_group_id: str
    variant_type: VariantType
    relevant_frames: tuple[RelevantFrame, ...]
    relevant_video_ids: tuple[str, ...]
    annotation_notes: str
    annotation_status: AnnotationStatus
    source_scope: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class KISBenchmark:
    schema_version: int
    benchmark_id: str
    description: str
    queries: tuple[BenchmarkQuery, ...]


@dataclass(frozen=True, slots=True)
class BenchmarkParseResult:
    benchmark: KISBenchmark | None
    errors: tuple[BenchmarkIssue, ...]

    @property
    def valid(self) -> bool:
        return self.benchmark is not None and not self.errors


def _text(
    item: dict[str, Any],
    field: str,
    issues: list[BenchmarkIssue],
    *,
    query_id: str | None,
    allow_empty: bool = False,
) -> str | None:
    value = item.get(field)
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        issues.append(
            BenchmarkIssue(
                code="INVALID_TEXT_FIELD",
                message=f"{field} must be a string"
                + ("" if allow_empty else " and must not be empty"),
                query_id=query_id,
                field=field,
            )
        )
        return None
    return value


def _string_list(
    item: dict[str, Any],
    field: str,
    issues: list[BenchmarkIssue],
    *,
    query_id: str | None,
    required: bool,
) -> tuple[str, ...] | None:
    value = item.get(field, [])
    if not isinstance(value, list) or any(
        not isinstance(entry, str) or not entry.strip() for entry in value
    ):
        issues.append(
            BenchmarkIssue(
                code="INVALID_STRING_LIST",
                message=f"{field} must be a list of non-empty strings",
                query_id=query_id,
                field=field,
            )
        )
        return None
    if required and not value:
        issues.append(
            BenchmarkIssue(
                code="EMPTY_SOURCE_SCOPE",
                message="source_scope must describe at least one audited video",
                query_id=query_id,
                field=field,
            )
        )
        return None
    return tuple(value)


def _parse_relevant_frames(
    item: dict[str, Any],
    issues: list[BenchmarkIssue],
    *,
    query_id: str | None,
) -> tuple[RelevantFrame, ...] | None:
    value = item.get("relevant_frames", [])
    if not isinstance(value, list):
        issues.append(
            BenchmarkIssue(
                code="INVALID_RELEVANT_FRAMES",
                message="relevant_frames must be a list",
                query_id=query_id,
                field="relevant_frames",
            )
        )
        return None
    parsed: list[RelevantFrame] = []
    for index, entry in enumerate(value):
        field = f"relevant_frames[{index}]"
        if not isinstance(entry, dict):
            issues.append(
                BenchmarkIssue(
                    code="INVALID_RELEVANT_FRAME",
                    message="each relevant frame must be an object",
                    query_id=query_id,
                    field=field,
                )
            )
            continue
        video_id = entry.get("video_id")
        frame_id = entry.get("frame_id")
        if not isinstance(video_id, str) or not video_id.strip():
            issues.append(
                BenchmarkIssue(
                    code="INVALID_RELEVANT_VIDEO_ID",
                    message="relevant frame video_id must be a non-empty string",
                    query_id=query_id,
                    field=f"{field}.video_id",
                )
            )
            continue
        if type(frame_id) is not int or frame_id < 0:
            issues.append(
                BenchmarkIssue(
                    code="INVALID_RELEVANT_FRAME_ID",
                    message="relevant frame_id must be a non-negative integer",
                    query_id=query_id,
                    field=f"{field}.frame_id",
                )
            )
            continue
        parsed.append(RelevantFrame(video_id=video_id, frame_id=frame_id))
    return tuple(parsed)


def parse_benchmark_payload(payload: Any) -> BenchmarkParseResult:
    issues: list[BenchmarkIssue] = []
    if not isinstance(payload, dict):
        return BenchmarkParseResult(
            benchmark=None,
            errors=(
                BenchmarkIssue(
                    code="INVALID_BENCHMARK_ROOT",
                    message="benchmark root must be an object",
                ),
            ),
        )
    schema_version = payload.get("schema_version")
    if type(schema_version) is not int or schema_version < 1:
        issues.append(
            BenchmarkIssue(
                code="INVALID_SCHEMA_VERSION",
                message="schema_version must be a positive integer",
                field="schema_version",
            )
        )
    benchmark_id = _text(payload, "benchmark_id", issues, query_id=None)
    description = _text(
        payload,
        "description",
        issues,
        query_id=None,
        allow_empty=True,
    )
    raw_queries = payload.get("queries")
    if not isinstance(raw_queries, list):
        issues.append(
            BenchmarkIssue(
                code="INVALID_QUERIES",
                message="queries must be a list",
                field="queries",
            )
        )
        raw_queries = []

    queries: list[BenchmarkQuery] = []
    for index, raw_query in enumerate(raw_queries):
        if not isinstance(raw_query, dict):
            issues.append(
                BenchmarkIssue(
                    code="INVALID_QUERY",
                    message="each query must be an object",
                    field=f"queries[{index}]",
                )
            )
            continue
        raw_query_id = raw_query.get("query_id")
        query_id = raw_query_id if isinstance(raw_query_id, str) else None
        issue_count = len(issues)
        parsed_query_id = _text(raw_query, "query_id", issues, query_id=query_id)
        text = _text(raw_query, "text", issues, query_id=query_id)
        group = _text(raw_query, "semantic_group_id", issues, query_id=query_id)
        notes = _text(
            raw_query,
            "annotation_notes",
            issues,
            query_id=query_id,
            allow_empty=True,
        )
        source_scope = _string_list(
            raw_query,
            "source_scope",
            issues,
            query_id=query_id,
            required=True,
        )
        relevant_video_ids = _string_list(
            raw_query,
            "relevant_video_ids",
            issues,
            query_id=query_id,
            required=False,
        )
        relevant_frames = _parse_relevant_frames(raw_query, issues, query_id=query_id)
        try:
            language = BenchmarkLanguage(raw_query.get("language"))
        except (TypeError, ValueError):
            language = None
            issues.append(
                BenchmarkIssue(
                    code="UNSUPPORTED_LANGUAGE",
                    message="language must be one of: vi, en",
                    query_id=query_id,
                    field="language",
                )
            )
        try:
            variant_type = VariantType(raw_query.get("variant_type"))
        except (TypeError, ValueError):
            variant_type = None
            issues.append(
                BenchmarkIssue(
                    code="UNSUPPORTED_VARIANT_TYPE",
                    message="unsupported variant_type",
                    query_id=query_id,
                    field="variant_type",
                )
            )
        try:
            annotation_status = AnnotationStatus(raw_query.get("annotation_status"))
        except (TypeError, ValueError):
            annotation_status = None
            issues.append(
                BenchmarkIssue(
                    code="UNSUPPORTED_ANNOTATION_STATUS",
                    message="annotation_status must be draft or verified",
                    query_id=query_id,
                    field="annotation_status",
                )
            )
        if len(issues) != issue_count:
            continue
        assert parsed_query_id is not None
        assert text is not None
        assert group is not None
        assert notes is not None
        assert source_scope is not None
        assert relevant_video_ids is not None
        assert relevant_frames is not None
        assert language is not None
        assert variant_type is not None
        assert annotation_status is not None
        queries.append(
            BenchmarkQuery(
                query_id=parsed_query_id,
                language=language,
                text=text,
                semantic_group_id=group,
                variant_type=variant_type,
                relevant_frames=relevant_frames,
                relevant_video_ids=relevant_video_ids,
                annotation_notes=notes,
                annotation_status=annotation_status,
                source_scope=source_scope,
            )
        )

    if issues:
        return BenchmarkParseResult(benchmark=None, errors=tuple(issues))
    assert isinstance(schema_version, int)
    assert benchmark_id is not None
    assert description is not None
    return BenchmarkParseResult(
        benchmark=KISBenchmark(
            schema_version=schema_version,
            benchmark_id=benchmark_id,
            description=description,
            queries=tuple(queries),
        ),
        errors=(),
    )


def load_benchmark(path: Path) -> BenchmarkParseResult:
    source = Path(path)
    if not source.is_file():
        return BenchmarkParseResult(
            benchmark=None,
            errors=(
                BenchmarkIssue(
                    code="BENCHMARK_FILE_NOT_FOUND",
                    message=f"benchmark file not found: {source}",
                ),
            ),
        )
    try:
        text = source.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        return BenchmarkParseResult(
            benchmark=None,
            errors=(
                BenchmarkIssue(
                    code="INVALID_UTF8",
                    message=f"benchmark is not valid UTF-8: {exc}",
                ),
            ),
        )
    try:
        payload = json.loads(text) if source.suffix.lower() == ".json" else yaml.safe_load(text)
    except (json.JSONDecodeError, yaml.YAMLError) as exc:
        return BenchmarkParseResult(
            benchmark=None,
            errors=(
                BenchmarkIssue(
                    code="INVALID_BENCHMARK_SYNTAX",
                    message=f"cannot parse benchmark: {exc}",
                ),
            ),
        )
    return parse_benchmark_payload(payload)
