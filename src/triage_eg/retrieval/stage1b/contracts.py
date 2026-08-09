"""Typed Stage 1B configuration and encoder candidate contracts."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

STAGE1B_VERSION = "0.1.1"
EVIDENCE_SOURCES = {
    "AUTHORITATIVE",
    "REPOSITORY_SOURCE",
    "EMPIRICAL_PROBE",
    "USER_ASSERTED",
    "HYPOTHESIS",
}
COMPATIBILITY_STATUSES = {
    "NOT_TESTED",
    "BLOCKED",
    "REJECTED",
    "UNVERIFIED",
    "USER_ASSERTED",
    "VERIFIED",
}


@dataclass(frozen=True)
class CompatibilityGate:
    minimum_completed_samples: int = 20
    pairwise_cosine_mean_min: float = 0.995
    pairwise_cosine_min_min: float = 0.98
    target_top1_rate_min: float = 0.95
    target_top5_rate_min: float = 1.0
    require_dimension_512: bool = True
    require_all_finite: bool = True
    implementation_equivalence_cosine_min: float = 0.999999

    def __post_init__(self) -> None:
        rates = (
            self.pairwise_cosine_mean_min,
            self.pairwise_cosine_min_min,
            self.target_top1_rate_min,
            self.target_top5_rate_min,
            self.implementation_equivalence_cosine_min,
        )
        if self.minimum_completed_samples <= 0 or any(not -1 <= value <= 1 for value in rates):
            raise ValueError("Invalid Stage 1B compatibility gate")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> CompatibilityGate:
        return cls(**value)


@dataclass(frozen=True)
class CandidateContract:
    candidate_id: str
    enabled: bool
    implementation: str
    architecture: str
    pretrained: str
    checkpoint_path: str | None
    output_dimension: int = 512
    tokenizer: str | None = None
    context_length: int | None = None
    image_preprocessing: dict[str, Any] = field(default_factory=dict)
    text_preprocessing: dict[str, Any] = field(default_factory=dict)
    image_embedding_normalization: bool = True
    text_embedding_normalization: bool = True
    runtime_dtype: str = "float32"
    evidence_source: str = "HYPOTHESIS"
    compatibility_status: str = "NOT_TESTED"
    checkpoint_sha256: str | None = None
    source_root: str | None = None
    asset_manifest_path: str | None = None
    text_truncate: bool = False
    device: str = "auto"
    batch_size: int = 16
    runtime_priority: int = 100
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", self.candidate_id):
            raise ValueError("candidate_id must be a safe path component")
        if self.implementation not in {"openai_clip", "open_clip", "custom", "mock", "unknown"}:
            raise ValueError("Unsupported candidate implementation")
        if self.evidence_source not in EVIDENCE_SOURCES:
            raise ValueError("Unknown candidate evidence_source")
        if self.compatibility_status not in COMPATIBILITY_STATUSES:
            raise ValueError("Unknown candidate compatibility_status")
        if (
            self.output_dimension <= 0
            or self.runtime_dtype != "float32"
            or self.runtime_priority < 0
            or self.batch_size <= 0
            or self.device not in {"auto", "cpu", "cuda", "cuda:0"}
        ):
            raise ValueError("Candidate output contract is invalid")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> CandidateContract:
        allowed = {item.name for item in cls.__dataclass_fields__.values()}
        data = {key: item for key, item in value.items() if key in allowed}
        data["notes"] = tuple(data.get("notes", ()))
        return cls(**data)

    def reproducible(self) -> bool:
        preprocess = self.image_preprocessing
        if self.implementation == "openai_clip" and self.source_root:
            return bool(
                self.architecture == "ViT-B/32"
                and self.pretrained == "openai"
                and self.checkpoint_path
                and self.tokenizer == "official clip.tokenize"
                and self.context_length
                and preprocess.get("source") == "official_clip_load_return_value"
                and preprocess.get("manual_preprocess_override") is False
                and {"strip", "lowercase", "unicode_normalization"} <= set(self.text_preprocessing)
            )
        required_image = {"resize", "crop", "interpolation", "convert_rgb", "mean", "std"}
        required_text = {"strip", "lowercase", "unicode_normalization"}
        required = (
            self.implementation not in {"unknown", "mock"}
            and bool(self.architecture)
            and bool(self.pretrained)
            and bool(self.checkpoint_path)
            and bool(self.tokenizer)
            and self.context_length is not None
            and self.context_length > 0
            and required_image <= set(preprocess)
            and all(preprocess[key] is not None for key in required_image)
            and required_text <= set(self.text_preprocessing)
        )
        return bool(required)


class MultimodalEncoder(Protocol):
    def encode_images(self, paths: list[Path]) -> np.ndarray: ...

    def encode_text(self, texts: list[str]) -> np.ndarray: ...


@dataclass(frozen=True)
class Stage1BConfig:
    repo_root: Path
    dataset_root: Path
    stage0_root: Path
    stage1_root: Path
    output_root: Path
    candidate_config: Path
    smoke_queries: Path | None = None
    sample_size: int = 50
    seed: int = 2026
    candidate_ids: tuple[str, ...] = ()
    overwrite: bool = False
    reuse_results: bool = False
    strict_root: bool = False
    run_text_smoke: bool = True
    build_git_commit: str | None = None

    def __post_init__(self) -> None:
        if not 1 <= self.sample_size <= 100:
            raise ValueError("sample_size must be between 1 and 100")
        if self.overwrite and self.reuse_results:
            raise ValueError("overwrite and reuse_results cannot both be enabled")
