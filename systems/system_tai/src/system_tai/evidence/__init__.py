"""Task-neutral evidence artifact readers and ASR transcript providers."""

from .asr_provider import (
    ASRSegment,
    WhisperASRExtractor,
)
from .object_artifacts import (
    BTC_OBJECT_ARTIFACT_SCHEMA,
    ObjectArtifactError,
    ObjectArtifactIndex,
    ObjectDetection,
    ObjectFrameEvidence,
    resolve_object_artifact_root,
)

__all__ = [
    "ASRSegment",
    "BTC_OBJECT_ARTIFACT_SCHEMA",
    "ObjectArtifactError",
    "ObjectArtifactIndex",
    "ObjectDetection",
    "ObjectFrameEvidence",
    "WhisperASRExtractor",
    "resolve_object_artifact_root",
]
