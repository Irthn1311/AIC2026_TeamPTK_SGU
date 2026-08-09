"""Typed contracts for Stage 1C qualitative text-retrieval evaluation."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

STAGE1C_VERSION = "0.1.0"
QUERY_SUITE_VERSION = "0.1.0"

LANGUAGES = frozenset({"en", "vi"})
CATEGORIES = frozenset(
    {
        "OBJECT",
        "ACTION",
        "SCENE",
        "ATTRIBUTE",
        "SPATIAL_RELATION",
        "MULTI_CONCEPT",
        "EVENT",
        "DIFFICULT",
    }
)
DIFFICULTIES = frozenset({"EASY", "MEDIUM", "HARD"})
REVIEW_LABELS = frozenset({"RELEVANT", "PARTIAL", "IRRELEVANT", "UNCERTAIN"})
FAILURE_TAGS = frozenset(
    {
        "OBJECT_MISMATCH",
        "ACTION_MISMATCH",
        "ATTRIBUTE_MISMATCH",
        "SCENE_MISMATCH",
        "SPATIAL_RELATION_MISMATCH",
        "MULTI_CONCEPT_PARTIAL",
        "INITIAL_FRAME_DOMINATION",
        "SAME_VIDEO_DOMINATION",
        "EXACT_VECTOR_DUPLICATION",
        "GENERIC_VISUAL_MATCH",
        "LANGUAGE_DEGRADATION",
        "OTHER",
    }
)
ISSUE_CODES = frozenset(
    {
        "STAGE1_INDEX_NOT_READY",
        "STAGE1B_ENCODER_NOT_VERIFIED",
        "STAGE1B_MODEL_SPACE_NOT_VERIFIED",
        "ENCODER_ASSET_NOT_FOUND",
        "ENCODER_LOAD_FAILED",
        "QUERY_SUITE_INVALID",
        "QUERY_PAIR_INVALID",
        "QUERY_ENCODING_FAILED",
        "QUERY_TOKENIZATION_FAILED",
        "QUERY_SEARCH_FAILED",
        "KEYFRAME_RESOLUTION_FAILED",
        "CONTACT_SHEET_RENDER_FAILED",
        "KIS_EXPORT_FAILED",
        "HIGH_INITIAL_FRAME_CONCENTRATION",
        "HIGH_SINGLE_VIDEO_CONCENTRATION",
        "HIGH_EXACT_VECTOR_DUPLICATION",
        "REVIEW_LABEL_INVALID",
        "REVIEW_ROW_IDENTITY_MISMATCH",
        "REVIEW_DUPLICATE_JUDGMENT",
        "REVIEW_INCOMPLETE",
    }
)


@dataclass(frozen=True)
class QueryRecord:
    query_id: str
    pair_id: str
    language: str
    category: str
    difficulty: str
    text: str
    notes: str = ""

    def __post_init__(self) -> None:
        safe = r"[A-Za-z0-9][A-Za-z0-9._-]*"
        if not re.fullmatch(safe, self.query_id) or not re.fullmatch(safe, self.pair_id):
            raise ValueError("QUERY_SUITE_INVALID: query_id and pair_id must be safe")
        if self.language not in LANGUAGES:
            raise ValueError(f"QUERY_SUITE_INVALID: invalid language {self.language!r}")
        if self.category not in CATEGORIES:
            raise ValueError(f"QUERY_SUITE_INVALID: invalid category {self.category!r}")
        if self.difficulty not in DIFFICULTIES:
            raise ValueError(f"QUERY_SUITE_INVALID: invalid difficulty {self.difficulty!r}")
        if not self.text.strip():
            raise ValueError("QUERY_SUITE_INVALID: query text must be non-empty")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> QueryRecord:
        required = {"query_id", "pair_id", "language", "category", "difficulty", "text"}
        missing = sorted(required - set(value))
        if missing:
            raise ValueError(f"QUERY_SUITE_INVALID: missing fields {missing}")
        return cls(
            query_id=str(value["query_id"]),
            pair_id=str(value["pair_id"]),
            language=str(value["language"]),
            category=str(value["category"]),
            difficulty=str(value["difficulty"]),
            text=str(value["text"]),
            notes=str(value.get("notes", "")),
        )


@dataclass(frozen=True)
class StructuralFlagConfig:
    initial_frame_rate_top20_warn: float = 0.50
    top_video_share_top20_warn: float = 0.40
    exact_duplicate_rate_top20_warn: float = 0.30

    def __post_init__(self) -> None:
        values = (
            self.initial_frame_rate_top20_warn,
            self.top_video_share_top20_warn,
            self.exact_duplicate_rate_top20_warn,
        )
        if any(not 0 <= value <= 1 for value in values):
            raise ValueError("Structural review heuristics must be between zero and one")


@dataclass(frozen=True)
class Stage1CConfig:
    repo_root: Path
    dataset_root: Path
    stage0_root: Path
    stage1_root: Path
    stage1b_root: Path
    encoder_asset_root: Path
    query_suite: Path
    output_root: Path
    frame_top_k: int = 50
    kis_top_k: int = 100
    review_top_k: int = 10
    contact_sheet_top_k: int = 20
    query_ids: tuple[str, ...] = ()
    languages: tuple[str, ...] = ()
    categories: tuple[str, ...] = ()
    device: str = "auto"
    batch_size: int = 16
    overwrite: bool = False
    reuse_results: bool = False
    strict_root: bool = False
    skip_contact_sheets: bool = False
    build_git_commit: str | None = None
    structural_flags: StructuralFlagConfig = field(default_factory=StructuralFlagConfig)

    def __post_init__(self) -> None:
        if self.overwrite and self.reuse_results:
            raise ValueError("overwrite and reuse_results cannot both be enabled")
        if not 1 <= self.review_top_k <= self.frame_top_k:
            raise ValueError("review_top_k must be within frame_top_k")
        if not 1 <= self.contact_sheet_top_k <= self.frame_top_k:
            raise ValueError("contact_sheet_top_k must be within frame_top_k")
        if self.kis_top_k < self.frame_top_k or self.batch_size <= 0:
            raise ValueError("kis_top_k must cover frame_top_k and batch_size must be positive")
        if self.device not in {"auto", "cpu", "cuda", "cuda:0"}:
            raise ValueError("Unsupported Stage 1C device")
        if set(self.languages) - LANGUAGES:
            raise ValueError("Invalid Stage 1C language filter")
        if set(self.categories) - CATEGORIES:
            raise ValueError("Invalid Stage 1C category filter")


@dataclass(frozen=True)
class Stage1CResult:
    output_root: Path
    summary: dict[str, Any]
    reused: bool
