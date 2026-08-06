"""Video catalog and frame-mapping interfaces."""

from system_tai.data.corpus_discovery import (
    CorpusDiscoveryError,
    CorpusManifest,
    DiscoveredVideo,
    DiscoveryMetrics,
    DiscoveryValidation,
    discover_corpus,
    load_corpus_manifest,
    load_or_build_manifest_cache,
)
from system_tai.data.frame_mapping import FrameMappingLoader
from system_tai.data.video_catalog import BenchmarkVideoCatalog

__all__ = [
    "BenchmarkVideoCatalog",
    "CorpusDiscoveryError",
    "CorpusManifest",
    "DiscoveryMetrics",
    "DiscoveryValidation",
    "DiscoveredVideo",
    "FrameMappingLoader",
    "discover_corpus",
    "load_or_build_manifest_cache",
    "load_corpus_manifest",
]
