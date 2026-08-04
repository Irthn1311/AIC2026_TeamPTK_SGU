"""Video catalog and frame-mapping interfaces."""

from .frame_mapping import FrameMappingLoader
from .video_catalog import BenchmarkVideoCatalog

__all__ = ["BenchmarkVideoCatalog", "FrameMappingLoader"]
