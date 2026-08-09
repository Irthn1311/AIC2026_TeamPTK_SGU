"""Reference experiment RT1: whole-query, unordered, and DANTE DP arms."""

from .contracts import RT1Event, RT1Query, RT1Settings, load_rt1_queries, load_rt1_settings
from .dante import DanteAlignment, dante_monotonic_dp
from .runner import RT1RunnerConfig, create_rt1_bundle, run_reference_rt1
from .scoring import (
    VideoRows,
    build_video_row_groups,
    rank_dante_dp,
    rank_unordered_event_max,
    top_k_video_overlap,
)

__all__ = [
    "DanteAlignment",
    "RT1Event",
    "RT1Query",
    "RT1RunnerConfig",
    "RT1Settings",
    "VideoRows",
    "build_video_row_groups",
    "create_rt1_bundle",
    "dante_monotonic_dp",
    "load_rt1_queries",
    "load_rt1_settings",
    "rank_dante_dp",
    "rank_unordered_event_max",
    "run_reference_rt1",
    "top_k_video_overlap",
]
