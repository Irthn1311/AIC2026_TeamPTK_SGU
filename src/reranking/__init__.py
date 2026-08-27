"""
Reranking module for AIC System.

Exports BaseReranker interface and concrete implementations:
- CLIPReranker        — Normalized CLIP/fusion score blending (Trụ 6)
- TemporalReranker    — Temporal density + video-context scoring (Trụ 6)
- OCRRelevanceReranker — OCR keyword matching boost
- ConstraintFilter    — Must-have / negation constraint enforcement (Trụ 5)
- GeminiReranker      — Gemini Vision semantic verification (Trụ 7)
"""

from src.reranking.base import BaseReranker
from src.reranking.clip_reranker import CLIPReranker
from src.reranking.temporal_reranker import TemporalReranker
from src.reranking.ocr_reranker import OCRRelevanceReranker
from src.reranking.constraint_filter import ConstraintFilter
from src.reranking.gemini_reranker import GeminiReranker

__all__ = [
    "BaseReranker",
    "CLIPReranker",
    "TemporalReranker",
    "OCRRelevanceReranker",
    "ConstraintFilter",
    "GeminiReranker",
]
