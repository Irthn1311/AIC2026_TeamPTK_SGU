"""Retrieval interfaces and candidate construction."""

from .candidates import CandidateConstructor
from .multi_query import (
    QueryLanguage,
    QueryVariant,
    QueryVariantType,
    WeightedRRFRetriever,
)
from .vector_search import ExactNumpyRetriever, VectorSearch
from .video_evidence import (
    FullCorpusVideoMaximaOutcome,
    RestrictedFrameHit,
    VideoMaximumHit,
    VideoRestrictedFeatureSearcher,
    VideoRestrictedSearchOutcome,
)
from .video_restricted import (
    VIDEO_CONDITIONED_KEYFRAME_DIVERSITY,
    VideoConditionedKeyframeConfig,
    VideoConditionedKeyframeDiversity,
)

__all__ = [
    "CandidateConstructor",
    "ExactNumpyRetriever",
    "QueryLanguage",
    "QueryVariant",
    "QueryVariantType",
    "VectorSearch",
    "FullCorpusVideoMaximaOutcome",
    "RestrictedFrameHit",
    "VideoMaximumHit",
    "VideoRestrictedFeatureSearcher",
    "VideoRestrictedSearchOutcome",
    "VIDEO_CONDITIONED_KEYFRAME_DIVERSITY",
    "VideoConditionedKeyframeConfig",
    "VideoConditionedKeyframeDiversity",
    "WeightedRRFRetriever",
]
