"""YAML configuration loading for Stage 1D."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .contracts import (
    STAGE1D_VERSION,
    GenerationConfig,
    RetrievalConfig,
    ReviewConfig,
    TranslatorConfig,
)


def load_stage1d_yaml(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve(strict=True)
    value = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("stage1d_version") != STAGE1D_VERSION:
        raise ValueError("Invalid Stage 1D configuration")
    allowed = {"stage1d_version", "translator", "generation", "retrieval", "review"}
    unexpected = sorted(set(value) - allowed)
    if unexpected:
        raise ValueError(f"Unsupported Stage 1D config keys: {unexpected}")
    return value


def settings_from_yaml(
    path: str | Path,
) -> tuple[TranslatorConfig, GenerationConfig, RetrievalConfig, ReviewConfig]:
    value = load_stage1d_yaml(path)
    return (
        TranslatorConfig(**dict(value.get("translator", {}))),
        GenerationConfig(**dict(value.get("generation", {}))),
        RetrievalConfig(**dict(value.get("retrieval", {}))),
        ReviewConfig(**dict(value.get("review", {}))),
    )

