"""
Fusion module for AIC System.

- NormalizedScoreFusion (NSF) — Trụ 3: preserves cosine signal (preferred)
- ReciprocalRankFusion  (RRF) — legacy rank-based fusion
"""

from src.fusion.normalized_score_fusion import NormalizedScoreFusion
from src.fusion.reciprocal_rank import ReciprocalRankFusion

__all__ = [
    "NormalizedScoreFusion",
    "ReciprocalRankFusion",
]
