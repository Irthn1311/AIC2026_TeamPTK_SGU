"""Offline Stage 1B multimodal encoder adapters."""

from triage_eg.retrieval.stage1b.adapters.openai_clip_official import (
    OfficialOpenAIClipAdapter,
    preflight_official_openai_clip,
    resolve_official_asset_paths,
)

__all__ = [
    "OfficialOpenAIClipAdapter",
    "preflight_official_openai_clip",
    "resolve_official_asset_paths",
]
