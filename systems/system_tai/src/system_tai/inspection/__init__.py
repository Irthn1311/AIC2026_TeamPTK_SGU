"""Static inspection artifacts for system_tai outputs."""

from system_tai.inspection.candidate_report import (
    CandidateInspectionArtifact,
    InspectionMode,
    KeyframeThumbnailIndex,
    ThumbnailResolver,
    build_candidate_inspection,
    resolve_keyframe_path,
)

__all__ = [
    "CandidateInspectionArtifact",
    "InspectionMode",
    "KeyframeThumbnailIndex",
    "ThumbnailResolver",
    "build_candidate_inspection",
    "resolve_keyframe_path",
]
