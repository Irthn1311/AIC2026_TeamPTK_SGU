"""Typed contracts for the Stage 1D Vietnamese translation-bridge ablation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from triage_eg.retrieval.stage1c.contracts import StructuralFlagConfig

STAGE1D_VERSION = "0.1.0"
TRANSLATOR_MODEL_ID = "Helsinki-NLP/opus-mt-vi-en"
TRANSLATOR_REVISION = "c8d2853e77f5fae31124d993e0b35176b1c8914e"
TRANSLATOR_ARCHITECTURE = "MarianMT"
ARMS = ("EN_DIRECT", "VI_DIRECT", "VI_TRANSLATED_EN")
REVIEW_LABELS = frozenset({"RELEVANT", "PARTIAL", "IRRELEVANT", "UNCERTAIN"})
ISSUE_CODES = frozenset(
    {
        "STAGE1C_FROZEN_BASELINE_INVALID",
        "STAGE1C_QUERY_SUITE_FINGERPRINT_MISMATCH",
        "STAGE1_INDEX_FINGERPRINT_MISMATCH",
        "STAGE1B_ENCODER_NOT_VERIFIED",
        "TRANSLATOR_ASSET_NOT_FOUND",
        "TRANSLATOR_ASSET_MANIFEST_INVALID",
        "TRANSLATOR_REVISION_MISMATCH",
        "TRANSLATOR_FILE_HASH_MISMATCH",
        "TRANSLATOR_DEPENDENCY_NOT_AVAILABLE",
        "TRANSLATOR_LOAD_FAILED",
        "TRANSLATION_FAILED",
        "TRANSLATION_EMPTY",
        "CLIP_TEXT_ENCODING_FAILED",
        "TRANSLATED_QUERY_SEARCH_FAILED",
        "KEYFRAME_RESOLUTION_FAILED",
        "KIS_EXPORT_FAILED",
        "CONTACT_SHEET_RENDER_FAILED",
        "HIGH_INITIAL_FRAME_CONCENTRATION",
        "HIGH_SINGLE_VIDEO_CONCENTRATION",
        "HIGH_EXACT_VECTOR_DUPLICATION",
        "REVIEW_LABEL_INVALID",
        "REVIEW_IDENTITY_MISMATCH",
        "REVIEW_DUPLICATE",
        "REVIEW_INCOMPLETE",
    }
)


@dataclass(frozen=True)
class TranslatorConfig:
    model_id: str = TRANSLATOR_MODEL_ID
    exact_revision: str = TRANSLATOR_REVISION
    device: str = "cpu"
    batch_size: int = 8
    local_files_only: bool = True

    def __post_init__(self) -> None:
        if self.model_id != TRANSLATOR_MODEL_ID:
            raise ValueError("Only the locked OPUS-MT vi-en model is accepted")
        if self.exact_revision != TRANSLATOR_REVISION:
            raise ValueError("Translator revision must match the locked revision")
        if self.device not in {"cpu", "cuda", "cuda:0", "auto"}:
            raise ValueError("Unsupported translator device")
        if self.batch_size <= 0 or not self.local_files_only:
            raise ValueError("Translator must use positive batches and local_files_only")


@dataclass(frozen=True)
class GenerationConfig:
    do_sample: bool = False
    num_beams: int = 4
    max_new_tokens: int = 64
    length_penalty: float = 1.0
    early_stopping: bool = True

    def __post_init__(self) -> None:
        if self.do_sample:
            raise ValueError("Stage 1D translation must be deterministic")
        if self.num_beams <= 0 or self.max_new_tokens <= 0:
            raise ValueError("Generation limits must be positive")

    def as_generate_kwargs(self) -> dict[str, Any]:
        return {
            "do_sample": self.do_sample,
            "num_beams": self.num_beams,
            "max_new_tokens": self.max_new_tokens,
            "length_penalty": self.length_penalty,
            "early_stopping": self.early_stopping,
        }


@dataclass(frozen=True)
class RetrievalConfig:
    frame_top_k: int = 50
    kis_top_k: int = 100
    contact_sheet_top_k: int = 20

    def __post_init__(self) -> None:
        if not 1 <= self.contact_sheet_top_k <= self.frame_top_k:
            raise ValueError("contact_sheet_top_k must be within frame_top_k")
        if self.kis_top_k < self.frame_top_k:
            raise ValueError("kis_top_k must cover frame_top_k")


@dataclass(frozen=True)
class ReviewConfig:
    top_k: int = 5
    seed: int = 2026
    blinded: bool = True

    def __post_init__(self) -> None:
        if self.top_k <= 0 or not self.blinded:
            raise ValueError("Stage 1D review must be positive and blinded")


@dataclass(frozen=True)
class Stage1DConfig:
    repo_root: Path
    dataset_root: Path
    stage0_root: Path
    stage1_root: Path
    stage1b_root: Path
    stage1c_root: Path
    clip_asset_root: Path
    translator_asset_root: Path
    output_root: Path
    translator: TranslatorConfig = field(default_factory=TranslatorConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    review: ReviewConfig = field(default_factory=ReviewConfig)
    structural_flags: StructuralFlagConfig = field(default_factory=StructuralFlagConfig)
    query_suite: Path | None = None
    pair_ids: tuple[str, ...] = ()
    clip_device: str = "auto"
    overwrite: bool = False
    reuse_translations: bool = False
    reuse_results: bool = False
    strict_root: bool = False
    skip_contact_sheets: bool = False
    build_git_commit: str | None = None
    stage1c_materialization: str = "DIRECT_DIRECTORY"
    translator_asset_materialization: str = "DIRECT_DIRECTORY"
    expected_query_count: int = 28
    expected_pair_count: int = 14

    def __post_init__(self) -> None:
        if self.overwrite and self.reuse_results:
            raise ValueError("overwrite and reuse_results cannot both be enabled")
        if self.clip_device not in {"cpu", "cuda", "cuda:0", "auto"}:
            raise ValueError("Unsupported CLIP device")
        if self.expected_query_count != self.expected_pair_count * 2:
            raise ValueError("Expected query count must be twice the pair count")
        if self.review.top_k > self.retrieval.frame_top_k:
            raise ValueError("Review top-k must fit within retrieval top-k")
        modes = {"DIRECT_DIRECTORY", "EXTRACTED_ZIP"}
        if (
            self.stage1c_materialization not in modes
            or self.translator_asset_materialization not in modes
        ):
            raise ValueError("Invalid input materialization mode")


@dataclass(frozen=True)
class Stage1DResult:
    output_root: Path
    summary: dict[str, Any]
    reused: bool
