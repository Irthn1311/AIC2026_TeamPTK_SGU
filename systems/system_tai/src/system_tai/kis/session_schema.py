"""Request, response, and configuration schemas for long-lived operational session."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from system_tai.refinement.models import RefinementConfig
from system_tai.retrieval.multi_query import (
    QueryLanguage,
    QueryVariant,
    QueryVariantType,
)


class SessionProtocolError(RuntimeError):
    """Base exception for session protocol errors."""


class MalformedRequestError(SessionProtocolError):
    """Request line could not be parsed as valid JSON."""

    def __init__(self, message: str, line_number: int = 1) -> None:
        super().__init__(message)
        self.line_number = line_number


class UnknownRequestTypeError(SessionProtocolError):
    """Request type is unrecognized."""


class InvalidRequestError(SessionProtocolError):
    """Request JSON is valid but violates field requirements."""


class DuplicateRequestIdError(SessionProtocolError):
    """Request ID was previously submitted within this session."""


@dataclass(frozen=True, slots=True)
class SessionConfig:
    input_root: Path = field(default_factory=lambda: Path("/kaggle/input"))
    reuse_manifest: Path | None = None
    manifest_cache: Path | None = None
    output_root: Path = field(default_factory=lambda: Path("/kaggle/working/system_tai_operational_session"))
    device: str = "auto"
    allow_model_download: bool = False
    clip_cache_dir: Path | None = None
    rrf_constant: float = 60.0
    chunk_size: int = 4096
    default_top_k_per_variant: int = 100
    default_output_top_k: int = 100
    default_refine_top_n: int = 3
    max_requests: int | None = None
    continue_on_request_error: bool = True
    fail_fast_protocol: bool = False
    session_id: str | None = None
    refinement_config: RefinementConfig = field(default_factory=RefinementConfig)

    def __post_init__(self) -> None:
        if self.reuse_manifest is not None and self.manifest_cache is not None:
            raise ValueError("reuse_manifest and manifest_cache are mutually exclusive")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu, or cuda")
        if self.rrf_constant <= 0 or not math.isfinite(self.rrf_constant):
            raise ValueError("rrf_constant must be positive and finite")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if not 1 <= self.default_top_k_per_variant <= 1000:
            raise ValueError("default_top_k_per_variant must be between 1 and 1000")
        if not 1 <= self.default_output_top_k <= 100:
            raise ValueError("default_output_top_k must be between 1 and 100")
        if not 0 <= self.default_refine_top_n <= self.default_output_top_k:
            raise ValueError("default_refine_top_n must be between 0 and default_output_top_k")
        if self.max_requests is not None and self.max_requests <= 0:
            raise ValueError("max_requests must be positive if set")


@dataclass(frozen=True, slots=True)
class HealthRequest:
    request_id: str
    type: str = "health"

    def __post_init__(self) -> None:
        if not self.request_id or not self.request_id.strip():
            raise ValueError("request_id must be non-empty")


@dataclass(frozen=True, slots=True)
class QueryRequest:
    request_id: str
    query_id: str
    query_vi: str
    query_en: str | None = None
    query_en_expansion: str | None = None
    weight_vi: float = 1.0
    weight_en: float = 1.0
    weight_en_expansion: float = 1.0
    top_k_per_variant: int = 100
    output_top_k: int = 100
    refine_top_n: int = 3
    type: str = "query"

    def __post_init__(self) -> None:
        if not self.request_id or not self.request_id.strip():
            raise ValueError("request_id must be non-empty")
        if not self.query_id or not self.query_id.strip():
            raise ValueError("query_id must be non-empty")
        if not self.query_vi or not self.query_vi.strip():
            raise ValueError("query_vi must be non-empty")
        if not math.isfinite(self.weight_vi) or self.weight_vi <= 0:
            raise ValueError("weight_vi must be finite and > 0")
        if self.query_en and self.query_en.strip():
            if not math.isfinite(self.weight_en) or self.weight_en <= 0:
                raise ValueError("weight_en must be finite and > 0")
        if self.query_en_expansion and self.query_en_expansion.strip():
            if not math.isfinite(self.weight_en_expansion) or self.weight_en_expansion <= 0:
                raise ValueError("weight_en_expansion must be finite and > 0")
        if not 1 <= self.top_k_per_variant <= 1000:
            raise ValueError("top_k_per_variant must be in range [1, 1000]")
        if not 1 <= self.output_top_k <= 100:
            raise ValueError("output_top_k must be in range [1, 100]")
        if not 0 <= self.refine_top_n <= self.output_top_k:
            raise ValueError("refine_top_n must be in range [0, output_top_k]")

    def variants(self) -> tuple[QueryVariant, ...]:
        result: list[QueryVariant] = [
            QueryVariant(
                variant_id=f"{self.query_id}::v1_vi",
                text=self.query_vi.strip(),
                language=QueryLanguage.VIETNAMESE,
                variant_type=QueryVariantType.VIETNAMESE_DIRECT,
                weight=self.weight_vi,
            )
        ]
        if self.query_en and self.query_en.strip():
            result.append(
                QueryVariant(
                    variant_id=f"{self.query_id}::v2_en",
                    text=self.query_en.strip(),
                    language=QueryLanguage.ENGLISH,
                    variant_type=QueryVariantType.ENGLISH_TRANSLATION,
                    weight=self.weight_en,
                )
            )
        if self.query_en_expansion and self.query_en_expansion.strip():
            result.append(
                QueryVariant(
                    variant_id=f"{self.query_id}::v3_en_exp",
                    text=self.query_en_expansion.strip(),
                    language=QueryLanguage.ENGLISH,
                    variant_type=QueryVariantType.ENGLISH_EXPANSION,
                    weight=self.weight_en_expansion,
                )
            )
        return tuple(result)


@dataclass(frozen=True, slots=True)
class ShutdownRequest:
    request_id: str
    type: str = "shutdown"

    def __post_init__(self) -> None:
        if not self.request_id or not self.request_id.strip():
            raise ValueError("request_id must be non-empty")


def parse_session_request(
    line: str,
    line_number: int = 1,
    *,
    default_top_k_per_variant: int = 100,
    default_output_top_k: int = 100,
    default_refine_top_n: int = 3,
) -> HealthRequest | QueryRequest | ShutdownRequest:
    raw = line.strip()
    if not raw:
        raise MalformedRequestError(
            f"line {line_number}: request line is empty",
            line_number=line_number,
        )
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise MalformedRequestError(
            f"line {line_number}: invalid JSON: {exc}",
            line_number=line_number,
        ) from exc

    if not isinstance(data, dict):
        raise MalformedRequestError(
            f"line {line_number}: JSON request root must be an object",
            line_number=line_number,
        )

    req_type = data.get("type")
    if not req_type or not isinstance(req_type, str):
        raise InvalidRequestError("missing or non-string request 'type'")

    request_id = data.get("request_id")
    if request_id is None or not isinstance(request_id, str) or not request_id.strip():
        raise InvalidRequestError("missing or empty 'request_id'")

    req_type_clean = req_type.strip().lower()
    if req_type_clean == "health":
        return HealthRequest(request_id=request_id.strip())
    if req_type_clean == "shutdown":
        return ShutdownRequest(request_id=request_id.strip())
    if req_type_clean == "query":
        query_id = data.get("query_id")
        query_vi = data.get("query_vi")
        if not query_id or not isinstance(query_id, str) or not query_id.strip():
            raise InvalidRequestError("query request requires non-empty 'query_id'")
        if not query_vi or not isinstance(query_vi, str) or not query_vi.strip():
            raise InvalidRequestError("query request requires non-empty 'query_vi'")

        try:
            return QueryRequest(
                request_id=request_id.strip(),
                query_id=query_id.strip(),
                query_vi=query_vi.strip(),
                query_en=data.get("query_en"),
                query_en_expansion=data.get("query_en_expansion"),
                weight_vi=float(data.get("weight_vi", 1.0)),
                weight_en=float(data.get("weight_en", 1.0)),
                weight_en_expansion=float(data.get("weight_en_expansion", 1.0)),
                top_k_per_variant=int(data.get("top_k_per_variant", default_top_k_per_variant)),
                output_top_k=int(data.get("output_top_k", default_output_top_k)),
                refine_top_n=int(data.get("refine_top_n", default_refine_top_n)),
            )
        except (TypeError, ValueError) as exc:
            raise InvalidRequestError(f"invalid query request fields: {exc}") from exc

    raise UnknownRequestTypeError(f"unknown request type '{req_type}'")


def format_json_response(data: dict[str, Any]) -> str:
    """Formats response as compact, single-line JSON."""
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))
