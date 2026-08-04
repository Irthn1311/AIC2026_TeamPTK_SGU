"""Stage 1 typed contracts."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

STAGE1_VERSION = "0.1.1"


@dataclass(frozen=True)
class EncoderContract:
    implementation: str | None = None
    model_name: str | None = None
    pretrained: str | None = None
    checkpoint_path: str | None = None
    tokenizer: str | None = None
    text_preprocessing: str | None = None
    image_preprocessing: str | None = None
    output_dimension: int = 512
    normalize_text_embedding: bool = True
    evidence_source: str = "NONE"
    compatibility_status: str = "BLOCKED"

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> EncoderContract:
        allowed = {field.name for field in cls.__dataclass_fields__.values()}
        return cls(**{key: item for key, item in value.items() if key in allowed})


class TextEncoder(Protocol):
    def encode_text(self, texts: list[str]) -> Any: ...


@dataclass(frozen=True)
class SearchConfig:
    stage1_root: Path
    query_id: str
    top_k: int = 100
    metric: str = "cosine"
    search_chunk_rows: int = 16_384
    video_grouping: str = "max"
    max_predictions: int = 100
    csv_header: bool = True

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", self.query_id):
            raise ValueError("query_id must be a safe single path component")
        if self.top_k <= 0 or self.max_predictions <= 0 or self.search_chunk_rows <= 0:
            raise ValueError("search limits must be positive")
        if self.metric not in {"cosine", "dot"}:
            raise ValueError("metric must be cosine or dot")
        if self.video_grouping not in {"max", "mean_top_k"}:
            raise ValueError("video_grouping must be max or mean_top_k")
