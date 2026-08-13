"""
Query Understanding Module for AIC 2026 Multimodal Video Retrieval
==================================================================
Provides query decomposition, intent classification, entity extraction,
dynamic query routing, and confidence-calibrated late fusion weights.
"""

from src.query_understanding.schemas import (
    DynamicFusionDecision,
    FusionWeights,
    IntentEnum,
    QueryUnderstandingResult,
)
from src.query_understanding.parser import BaseQueryParser, RuleBasedQueryParser
from src.query_understanding.router import QueryRouter

__all__ = [
    "IntentEnum",
    "QueryUnderstandingResult",
    "FusionWeights",
    "DynamicFusionDecision",
    "BaseQueryParser",
    "RuleBasedQueryParser",
    "QueryRouter",
]
