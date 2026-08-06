"""Static inspection artifacts for system_tai outputs."""

from system_tai.inspection.candidate_report import (
    CandidateInspectionArtifact,
    build_candidate_inspection,
    resolve_keyframe_path,
)

__all__ = [
    "CandidateInspectionArtifact",
    "build_candidate_inspection",
    "resolve_keyframe_path",
]
