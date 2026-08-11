"""MB1 v0.2 boundary-rich semantic-moment candidate preparation."""

from .runner import (
    MB1_V02_MODE,
    MB1_V02_VERSION,
    MB1V02Config,
    annotation_schema_v02,
    build_source_video_pool,
    create_mb1_v02_bundle,
    preflight_mb1_v02,
    prepare_mb1_v02_candidates,
    render_contact_sheet,
)
from .signals import (
    CandidateProposal,
    ScanSeries,
    ShotSegment,
    continuous_shot_segments,
    dense_displayed_frames,
    detect_hard_cuts,
    overview_displayed_frames,
    propose_active_windows,
    scan_low_resolution_video,
)

__all__ = [
    "CandidateProposal",
    "MB1V02Config",
    "MB1_V02_MODE",
    "MB1_V02_VERSION",
    "ScanSeries",
    "ShotSegment",
    "annotation_schema_v02",
    "build_source_video_pool",
    "continuous_shot_segments",
    "create_mb1_v02_bundle",
    "dense_displayed_frames",
    "detect_hard_cuts",
    "overview_displayed_frames",
    "preflight_mb1_v02",
    "prepare_mb1_v02_candidates",
    "propose_active_windows",
    "render_contact_sheet",
    "scan_low_resolution_video",
]
