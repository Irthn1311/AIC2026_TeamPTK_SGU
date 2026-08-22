"""Retrieval interfaces, FAISS vector index, BM25 text search, and multimodal fusion."""

from .bm25_search import (
    BM25Document,
    BM25Index,
    BM25TextRetriever,
    tokenize_text,
)
from .candidates import CandidateConstructor
from .faiss_index import (
    FaissVectorIndex,
    FaissVectorRetriever,
    VectorRecord,
)
from .hybrid_fusion import HybridFusionRetriever
from .multi_query import (
    QueryLanguage,
    QueryVariant,
    QueryVariantType,
    WeightedRRFRetriever,
)
from .semantic_query import (
    CompiledSemanticQuery,
    CompiledSemanticVariant,
    SemanticQueryConfig,
    SemanticUnitRole,
    VietnameseSemanticUnit,
    compile_vietnamese_semantic_query,
    decompose_vietnamese_semantic_units,
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
    "BM25Document",
    "BM25Index",
    "BM25TextRetriever",
    "CandidateConstructor",
    "CompiledSemanticQuery",
    "CompiledSemanticVariant",
    "ExactNumpyRetriever",
    "FaissVectorIndex",
    "FaissVectorRetriever",
    "FullCorpusVideoMaximaOutcome",
    "HybridFusionRetriever",
    "QueryLanguage",
    "QueryVariant",
    "QueryVariantType",
    "RestrictedFrameHit",
    "SemanticQueryConfig",
    "SemanticUnitRole",
    "VectorRecord",
    "VectorSearch",
    "VideoConditionedKeyframeConfig",
    "VideoConditionedKeyframeDiversity",
    "VIDEO_CONDITIONED_KEYFRAME_DIVERSITY",
    "VideoMaximumHit",
    "VideoRestrictedFeatureSearcher",
    "VideoRestrictedSearchOutcome",
    "VietnameseSemanticUnit",
    "WeightedRRFRetriever",
    "compile_vietnamese_semantic_query",
    "decompose_vietnamese_semantic_units",
    "tokenize_text",
]
