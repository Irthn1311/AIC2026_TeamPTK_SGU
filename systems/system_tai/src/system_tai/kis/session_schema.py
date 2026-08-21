"""Request, response, and configuration schemas for long-lived operational session."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from system_tai.qa.grounding import QAVideoConditionedEvidenceConfig
from system_tai.qa.object_provider import ObjectAnswerProviderConfig
from system_tai.qa.ocr_provider import OCRAnswerProviderConfig
from system_tai.qa.visual_ontology import VisualOntologyConfig
from system_tai.refinement.models import (
    Q3AnchorRefinementConfig,
    RefinementConfig,
    SharedRawRegionRefinementConfig,
)
from system_tai.retrieval.multi_query import (
    QueryLanguage,
    QueryVariant,
    QueryVariantType,
)
from system_tai.retrieval.video_restricted import VideoConditionedKeyframeConfig
from system_tai.trake.video_first import TRAKEVideoFirstConfig


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
    output_root: Path = field(
        default_factory=lambda: Path("/kaggle/working/system_tai_operational_session")
    )
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
    video_conditioned_keyframe_config: VideoConditionedKeyframeConfig = field(
        default_factory=VideoConditionedKeyframeConfig
    )
    q3_anchor_refinement_config: Q3AnchorRefinementConfig = field(
        default_factory=Q3AnchorRefinementConfig
    )
    trake_video_first_config: TRAKEVideoFirstConfig = field(
        default_factory=TRAKEVideoFirstConfig
    )
    trake_shared_raw_region_config: SharedRawRegionRefinementConfig = field(
        default_factory=SharedRawRegionRefinementConfig
    )
    qa_video_conditioned_evidence_config: QAVideoConditionedEvidenceConfig = field(
        default_factory=QAVideoConditionedEvidenceConfig
    )
    qa_object_answer_provider_config: ObjectAnswerProviderConfig = field(
        default_factory=ObjectAnswerProviderConfig
    )
    qa_ocr_answer_provider_config: OCRAnswerProviderConfig = field(
        default_factory=OCRAnswerProviderConfig
    )
    qa_visual_ontology_config: VisualOntologyConfig = field(
        default_factory=VisualOntologyConfig
    )
    qa_unsupported_provider_fallback: bool = False
    enable_dynamic_translation: bool = False
    translation_model_name: str = "vinai/vinai-translate-vi2en-v2"
    translation_cache_dir: Path | None = None
    translation_device: str = "auto"
    translation_allow_model_download: bool = False
    translation_revision: str | None = (
        "ae7baa85da07dbe8e23ac26a9f5ef560c17e2138"
    )
    translation_max_clip_tokens: int = 75

    def __post_init__(self) -> None:
        if not 1 <= self.translation_max_clip_tokens <= 75:
            raise ValueError("translation_max_clip_tokens must be between 1 and 75")
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
        if not isinstance(
            self.video_conditioned_keyframe_config,
            VideoConditionedKeyframeConfig,
        ):
            raise ValueError(
                "video_conditioned_keyframe_config must be "
                "VideoConditionedKeyframeConfig"
            )
        if not isinstance(self.q3_anchor_refinement_config, Q3AnchorRefinementConfig):
            raise ValueError(
                "q3_anchor_refinement_config must be Q3AnchorRefinementConfig"
            )
        if (
            self.q3_anchor_refinement_config.enabled
            and not self.video_conditioned_keyframe_config.enabled
        ):
            raise ValueError("Q3 anchor refinement requires Q3 keyframe conditioning")
        if not isinstance(self.trake_video_first_config, TRAKEVideoFirstConfig):
            raise ValueError("trake_video_first_config must be TRAKEVideoFirstConfig")
        if not isinstance(
            self.trake_shared_raw_region_config,
            SharedRawRegionRefinementConfig,
        ):
            raise ValueError(
                "trake_shared_raw_region_config must be "
                "SharedRawRegionRefinementConfig"
            )
        if not isinstance(
            self.qa_video_conditioned_evidence_config,
            QAVideoConditionedEvidenceConfig,
        ):
            raise ValueError(
                "qa_video_conditioned_evidence_config must be "
                "QAVideoConditionedEvidenceConfig"
            )
        if not isinstance(
            self.qa_object_answer_provider_config,
            ObjectAnswerProviderConfig,
        ):
            raise ValueError(
                "qa_object_answer_provider_config must be ObjectAnswerProviderConfig"
            )
        if (
            self.qa_object_answer_provider_config.enabled
            and not self.qa_video_conditioned_evidence_config.enabled
        ):
            raise ValueError("QA object evidence requires QA video-conditioned evidence")
        if not isinstance(
            self.qa_ocr_answer_provider_config,
            OCRAnswerProviderConfig,
        ):
            raise ValueError(
                "qa_ocr_answer_provider_config must be OCRAnswerProviderConfig"
            )
        if (
            self.qa_ocr_answer_provider_config.enabled
            and not self.qa_video_conditioned_evidence_config.enabled
        ):
            raise ValueError("QA OCR evidence requires QA video-conditioned evidence")
        if not isinstance(self.qa_visual_ontology_config, VisualOntologyConfig):
            raise ValueError(
                "qa_visual_ontology_config must be VisualOntologyConfig"
            )
        if (
            self.qa_visual_ontology_config.enabled
            and not self.qa_video_conditioned_evidence_config.enabled
        ):
            raise ValueError(
                "QA visual ontology requires QA video-conditioned evidence"
            )
        if (
            self.qa_visual_ontology_config.enabled
            and self.qa_object_answer_provider_config.enabled
        ):
            raise ValueError(
                "QA visual ontology and QA object evidence are mutually exclusive"
            )

    @classmethod
    def from_yaml(cls, yaml_path: str | Path, **overrides: Any) -> SessionConfig:
        """Load SessionConfig from a production YAML configuration file."""
        try:
            import yaml
        except ImportError:
            import subprocess
            import sys

            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pyyaml"], check=False)
            import yaml

        p = Path(yaml_path)
        if not p.exists():
            raise FileNotFoundError(f"Configuration file not found at {p}")
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}

        sys_cfg = data.get("system", {})
        kis_cfg = data.get("kis", {})
        ret_cfg = data.get("retrieval", {})

        kwargs: dict[str, Any] = {
            "device": sys_cfg.get("device", "auto"),
            "allow_model_download": sys_cfg.get("allow_model_download", True),
            "chunk_size": sys_cfg.get("chunk_size", 50000),
            "rrf_constant": ret_cfg.get("rrf_k", 60.0),
            "default_output_top_k": kis_cfg.get("top_k_candidates", 100),
            "default_refine_top_n": kis_cfg.get("refine_top_n_anchors", 3),
            "enable_dynamic_translation": kis_cfg.get("enable_dynamic_translation", False),
            "translation_model_name": kis_cfg.get(
                "translation_model_name",
                "vinai/vinai-translate-vi2en-v2",
            ),
            "translation_revision": kis_cfg.get(
                "translation_revision",
                "ae7baa85da07dbe8e23ac26a9f5ef560c17e2138",
            ),
            "translation_allow_model_download": kis_cfg.get(
                "translation_allow_model_download",
                False,
            ),
            "translation_max_clip_tokens": kis_cfg.get(
                "translation_max_clip_tokens",
                75,
            ),
        }
        kwargs.update(overrides)
        return cls(**kwargs)


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
    include_vi_variant: bool = True
    type: str = "query"

    def __post_init__(self) -> None:
        if not self.request_id or not self.request_id.strip():
            raise ValueError("request_id must be non-empty")
        if not self.query_id or not self.query_id.strip():
            raise ValueError("query_id must be non-empty")
        if not self.query_vi or not self.query_vi.strip():
            raise ValueError("query_vi must be non-empty")
        if type(self.include_vi_variant) is not bool:
            raise ValueError("include_vi_variant must be a boolean")
        if not self.include_vi_variant and not (
            self.query_en and self.query_en.strip()
        ):
            raise ValueError(
                "query_en must be non-empty when include_vi_variant is false"
            )
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
        result: list[QueryVariant] = []
        if self.include_vi_variant:
            result.append(
                QueryVariant(
                    variant_id=f"{self.query_id}::v1_vi",
                    text=self.query_vi.strip(),
                    language=QueryLanguage.VIETNAMESE,
                    variant_type=QueryVariantType.VIETNAMESE_DIRECT,
                    weight=self.weight_vi,
                )
            )
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
class QAQueryRequest:
    request_id: str
    query_id: str
    event_description: str
    question: str
    event_description_en: str | None = None
    question_en: str | None = None
    include_vi_variant: bool = True
    top_k_per_variant: int = 100
    output_top_k: int = 100
    refine_top_n: int = 3
    type: str = "qa_query"

    def __post_init__(self) -> None:
        if not self.request_id or not self.request_id.strip():
            raise ValueError("request_id must be non-empty")
        if not self.query_id or not self.query_id.strip():
            raise ValueError("query_id must be non-empty")
        if not self.event_description or not self.event_description.strip():
            raise ValueError("event_description must be non-empty")
        if not self.question or not self.question.strip():
            raise ValueError("question must be non-empty")

        if self.event_description_en is not None:
            if not isinstance(self.event_description_en, str) or not (
                self.event_description_en.strip()
            ):
                raise ValueError("event_description_en must be a non-empty string when provided")

        if self.question_en is not None:
            if not isinstance(self.question_en, str) or not self.question_en.strip():
                raise ValueError("question_en must be a non-empty string when provided")

        if type(self.include_vi_variant) is not bool:
            raise ValueError("include_vi_variant must be a boolean")
        if not self.include_vi_variant and not (
            self.event_description_en and self.event_description_en.strip()
        ):
            raise ValueError(
                "event_description_en must be non-empty when include_vi_variant is false"
            )

        if type(self.top_k_per_variant) is not int or not (1 <= self.top_k_per_variant <= 1000):
            raise ValueError("top_k_per_variant must be integer in range [1, 1000]")
        if type(self.output_top_k) is not int or not (1 <= self.output_top_k <= 100):
            raise ValueError("output_top_k must be integer in range [1, 100]")
        if type(self.refine_top_n) is not int or not (1 <= self.refine_top_n <= self.output_top_k):
            raise ValueError("refine_top_n must be integer in range [1, output_top_k]")

    def variants(self) -> tuple[QueryVariant, ...]:
        result: list[QueryVariant] = []
        if self.include_vi_variant:
            result.append(
                QueryVariant(
                    variant_id=f"{self.query_id}::v1_vi",
                    text=self.event_description.strip(),
                    language=QueryLanguage.VIETNAMESE,
                    variant_type=QueryVariantType.VIETNAMESE_DIRECT,
                    weight=1.0,
                )
            )
        if self.event_description_en and self.event_description_en.strip():
            result.append(
                QueryVariant(
                    variant_id=f"{self.query_id}::v2_en",
                    text=self.event_description_en.strip(),
                    language=QueryLanguage.ENGLISH,
                    variant_type=QueryVariantType.ENGLISH_TRANSLATION,
                    weight=1.0,
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


@dataclass(frozen=True, slots=True)
class TRAKEQueryRequest:
    request_id: str
    query_id: str
    events: tuple[dict[str, Any], ...]
    include_vi_variant: bool = True
    top_k_per_variant: int = 100
    event_candidate_top_k: int = 100
    output_top_k: int = 100
    beam_width: int = 100
    refine_top_n: int = 3
    type: str = "trake_query"

    def __post_init__(self) -> None:
        if not self.request_id or not self.request_id.strip():
            raise ValueError("request_id must be non-empty")
        if not self.query_id or not self.query_id.strip():
            raise ValueError("query_id must be non-empty")
        if not isinstance(self.events, (tuple, list)) or len(self.events) == 0:
            raise ValueError("events must be a non-empty list of event objects")
        if type(self.include_vi_variant) is not bool:
            raise ValueError("include_vi_variant must be a boolean")

        for idx, ev in enumerate(self.events):
            if not isinstance(ev, dict):
                raise ValueError(f"event at index {idx} must be an object")
            desc = ev.get("description")
            if not desc or not isinstance(desc, str) or not desc.strip():
                raise ValueError(f"event at index {idx} requires non-empty string 'description'")
            desc_en = ev.get("description_en")
            if desc_en is not None:
                if not isinstance(desc_en, str) or not desc_en.strip():
                    raise ValueError(
                        f"event at index {idx} 'description_en' must be non-empty string"
                    )
            if not self.include_vi_variant and desc_en is None:
                raise ValueError(
                    "every event requires non-empty 'description_en' when "
                    "include_vi_variant is false"
                )

        if type(self.top_k_per_variant) is not int or not (1 <= self.top_k_per_variant <= 1000):
            raise ValueError("top_k_per_variant must be integer in range [1, 1000]")
        if type(self.event_candidate_top_k) is not int or not (
            1 <= self.event_candidate_top_k <= 100
        ):
            raise ValueError("event_candidate_top_k must be integer in range [1, 100]")
        if type(self.output_top_k) is not int or not (1 <= self.output_top_k <= 100):
            raise ValueError("output_top_k must be integer in range [1, 100]")
        if type(self.beam_width) is not int or not (1 <= self.beam_width <= 1000):
            raise ValueError("beam_width must be integer in range [1, 1000]")
        if (
            type(self.refine_top_n) is not int
            or not (0 <= self.refine_top_n <= self.output_top_k)
            or self.refine_top_n > 100
        ):
            raise ValueError("refine_top_n must be integer in range [0, min(output_top_k, 100)]")


def parse_session_request(
    line: str,
    line_number: int = 1,
    *,
    default_top_k_per_variant: int = 100,
    default_output_top_k: int = 100,
    default_refine_top_n: int = 3,
) -> HealthRequest | QueryRequest | QAQueryRequest | TRAKEQueryRequest | ShutdownRequest:
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
                include_vi_variant=data.get("include_vi_variant", True),
            )
        except (TypeError, ValueError) as exc:
            raise InvalidRequestError(f"invalid query request fields: {exc}") from exc
    if req_type_clean == "qa_query":
        query_id = data.get("query_id")
        event_desc = data.get("event_description")
        question = data.get("question")
        if not query_id or not isinstance(query_id, str) or not query_id.strip():
            raise InvalidRequestError("qa_query request requires non-empty 'query_id'")
        if not event_desc or not isinstance(event_desc, str) or not event_desc.strip():
            raise InvalidRequestError("qa_query request requires non-empty 'event_description'")
        if not question or not isinstance(question, str) or not question.strip():
            raise InvalidRequestError("qa_query request requires non-empty 'question'")

        top_k_pv_raw = data.get("top_k_per_variant", default_top_k_per_variant)
        out_k_raw = data.get("output_top_k", default_output_top_k)
        refine_n_raw = data.get("refine_top_n", default_refine_top_n)

        if (
            type(top_k_pv_raw) is not int
            or type(out_k_raw) is not int
            or type(refine_n_raw) is not int
        ):
            msg = "top_k_per_variant, output_top_k, and refine_top_n must be strict integers"
            raise InvalidRequestError(msg)

        try:
            return QAQueryRequest(
                request_id=request_id.strip(),
                query_id=query_id.strip(),
                event_description=event_desc.strip(),
                question=question.strip(),
                event_description_en=data.get("event_description_en"),
                question_en=data.get("question_en"),
                include_vi_variant=data.get("include_vi_variant", True),
                top_k_per_variant=top_k_pv_raw,
                output_top_k=out_k_raw,
                refine_top_n=refine_n_raw,
            )
        except (TypeError, ValueError) as exc:
            raise InvalidRequestError(f"invalid qa_query request fields: {exc}") from exc
    if req_type_clean == "trake_query":
        query_id = data.get("query_id")
        events_raw = data.get("events")
        if not query_id or not isinstance(query_id, str) or not query_id.strip():
            raise InvalidRequestError("trake_query request requires non-empty 'query_id'")
        if not isinstance(events_raw, list) or len(events_raw) == 0:
            raise InvalidRequestError("trake_query request requires non-empty 'events' array")

        top_k_pv_raw = data.get("top_k_per_variant", default_top_k_per_variant)
        event_cand_k_raw = data.get("event_candidate_top_k", default_output_top_k)
        out_k_raw = data.get("output_top_k", default_output_top_k)
        beam_w_raw = data.get("beam_width", 100)
        refine_n_raw = data.get("refine_top_n", default_refine_top_n)

        if (
            type(top_k_pv_raw) is not int
            or type(event_cand_k_raw) is not int
            or type(out_k_raw) is not int
            or type(beam_w_raw) is not int
            or type(refine_n_raw) is not int
        ):
            msg = "numeric parameters must be strict integers"
            raise InvalidRequestError(msg)

        try:
            return TRAKEQueryRequest(
                request_id=request_id.strip(),
                query_id=query_id.strip(),
                events=tuple(events_raw),
                include_vi_variant=data.get("include_vi_variant", True),
                top_k_per_variant=top_k_pv_raw,
                event_candidate_top_k=event_cand_k_raw,
                output_top_k=out_k_raw,
                beam_width=beam_w_raw,
                refine_top_n=refine_n_raw,
            )
        except (TypeError, ValueError) as exc:
            raise InvalidRequestError(f"invalid trake_query request fields: {exc}") from exc

    raise UnknownRequestTypeError(f"unknown request type '{req_type}'")


def format_json_response(data: dict[str, Any]) -> str:
    """Formats response as compact, single-line JSON."""
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))
