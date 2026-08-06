"""Video catalog and frame-mapping interfaces."""

from system_tai.data.corpus_discovery import (
    CorpusDiscoveryError,
    CorpusManifest,
    DiscoveredVideo,
    discover_corpus,
    load_corpus_manifest,
)
from system_tai.data.frame_mapping import FrameMappingLoader
from system_tai.data.video_catalog import BenchmarkVideoCatalog

__all__ = [
    "BenchmarkVideoCatalog",
    "CorpusDiscoveryError",
    "CorpusManifest",
    "DiscoveredVideo",
    "FrameMappingLoader",
    "discover_corpus",
    "load_corpus_manifest",
]
