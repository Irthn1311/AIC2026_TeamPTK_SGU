"""Contracts and configuration for the Stage 2A operational runtime."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

STAGE2_VERSION = "0.1.0"
MAX_TOP_K = 100
SUPPORTED_LANGUAGES = {"en", "vi", "auto"}
ISSUE_CODES = {
    "STAGE1_INDEX_NOT_READY",
    "STAGE1_INDEX_FINGERPRINT_MISMATCH",
    "STAGE1B_ENCODER_NOT_VERIFIED",
    "STAGE1B_MODEL_SPACE_NOT_VERIFIED",
    "STAGE1E_LANGUAGE_PATH_NOT_FROZEN",
    "STAGE1E_CONTRACT_INVALID",
    "LANGUAGE_UNSUPPORTED",
    "LANGUAGE_AMBIGUOUS",
    "TRANSLATOR_ASSET_NOT_FOUND",
    "TRANSLATOR_REVISION_MISMATCH",
    "TRANSLATOR_LOAD_FAILED",
    "TRANSLATION_FAILED",
    "TRANSLATION_EMPTY",
    "CLIP_ASSET_NOT_FOUND",
    "CLIP_LOAD_FAILED",
    "CLIP_ENCODING_FAILED",
    "SEARCH_FAILED",
    "KIS_EXPORT_FAILED",
}


class Stage2RuntimeError(ValueError):
    """Fail-closed operational error carrying a stable issue code."""

    def __init__(self, code: str, message: str = "") -> None:
        if code not in ISSUE_CODES:
            raise ValueError(f"Unknown Stage 2 issue code: {code}")
        self.code = code
        super().__init__(f"{code}: {message}" if message else code)


@dataclass(frozen=True)
class QueryRequest:
    query_id: str
    text: str
    language: str = "auto"
    top_k: int = MAX_TOP_K

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", self.query_id):
            raise ValueError("query_id must be a safe path component")
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("query text must be non-empty")
        if self.language not in SUPPORTED_LANGUAGES:
            raise Stage2RuntimeError("LANGUAGE_UNSUPPORTED", self.language)
        if not 1 <= self.top_k <= MAX_TOP_K:
            raise ValueError("top_k must be between 1 and 100")


@dataclass(frozen=True)
class Stage2RuntimeConfig:
    stage1_root: Path
    stage1b_root: Path
    stage1e_root: Path
    clip_asset_root: Path
    translator_asset_root: Path
    output_root: Path
    stage1d_config: Path
    default_language: str = "auto"
    default_top_k: int = MAX_TOP_K
    clip_device: str = "cpu"
    clip_batch_size: int = 32
    translator_device: str = "cpu"
    translator_batch_size: int = 8
    translator_lazy_load: bool = True
    search_backend: str = "existing_stage1_exact"
    max_top_k: int = MAX_TOP_K
    search_chunk_rows: int = 16_384
    build_git_commit: str | None = None
    hardware_mode: str = "auto"
    video_backend: str = "auto"
    auto_nvdec_promoted: bool = False

    def __post_init__(self) -> None:
        if self.default_language not in SUPPORTED_LANGUAGES:
            raise ValueError("invalid default language")
        if not 1 <= self.default_top_k <= self.max_top_k <= MAX_TOP_K:
            raise ValueError("invalid Stage 2 top-k limits")
        if self.clip_device not in {"cpu", "cuda", "cuda:0", "auto"}:
            raise ValueError("invalid CLIP device")
        if self.translator_device not in {"cpu", "cuda", "cuda:0", "auto"}:
            raise ValueError("invalid translator device")
        if self.clip_batch_size <= 0 or self.translator_batch_size <= 0:
            raise ValueError("batch sizes must be positive")
        if self.search_backend != "existing_stage1_exact" or self.search_chunk_rows <= 0:
            raise ValueError("Stage 2A requires the existing exact Stage 1 backend")
        if self.hardware_mode not in {"auto", "cpu", "gpu"}:
            raise ValueError("invalid hardware mode")
        if self.video_backend not in {"auto", "opencv", "nvdec"}:
            raise ValueError("invalid video backend")


def load_stage2_settings(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve(strict=True)
    value = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("stage2_version") != STAGE2_VERSION:
        raise ValueError("Invalid Stage 2 runtime configuration")
    allowed = {"stage2_version", "runtime", "hardware", "video", "clip", "translator", "search"}
    if set(value) - allowed:
        raise ValueError(f"Unsupported Stage 2 config keys: {sorted(set(value) - allowed)}")
    return value


def config_from_yaml(
    path: str | Path,
    *,
    stage1_root: Path,
    stage1b_root: Path,
    stage1e_root: Path,
    clip_asset_root: Path,
    translator_asset_root: Path,
    output_root: Path,
    stage1d_config: Path,
    build_git_commit: str | None = None,
) -> Stage2RuntimeConfig:
    value = load_stage2_settings(path)
    runtime, clip = value.get("runtime", {}), value.get("clip", {})
    translator, search = value.get("translator", {}), value.get("search", {})
    hardware, video = value.get("hardware", {}), value.get("video", {})
    return Stage2RuntimeConfig(
        stage1_root=stage1_root,
        stage1b_root=stage1b_root,
        stage1e_root=stage1e_root,
        clip_asset_root=clip_asset_root,
        translator_asset_root=translator_asset_root,
        output_root=output_root,
        stage1d_config=stage1d_config,
        default_language=runtime.get("default_language", "auto"),
        default_top_k=int(runtime.get("default_top_k", 100)),
        clip_device=clip.get("device", "cpu"),
        clip_batch_size=int(clip.get("batch_size", 32)),
        translator_device=translator.get("device", "cpu"),
        translator_batch_size=int(translator.get("batch_size", 8)),
        translator_lazy_load=bool(translator.get("lazy_load", True)),
        search_backend=search.get("backend", "existing_stage1_exact"),
        max_top_k=int(search.get("max_top_k", 100)),
        search_chunk_rows=int(search.get("chunk_rows", 16_384)),
        build_git_commit=build_git_commit,
        hardware_mode=hardware.get("mode", "auto"),
        video_backend=video.get("backend", "auto"),
        auto_nvdec_promoted=bool(video.get("auto_nvdec_promoted", False)),
    )


__all__ = [
    "MAX_TOP_K",
    "QueryRequest",
    "STAGE2_VERSION",
    "Stage2RuntimeConfig",
    "Stage2RuntimeError",
    "config_from_yaml",
    "load_stage2_settings",
]
