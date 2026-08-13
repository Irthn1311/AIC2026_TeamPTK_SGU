"""
Base Reranker Interface for AIC System.
All rerankers take a list of SearchResult candidates and return a re-ordered list.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List

from src.common.types import SearchResult


class BaseReranker(ABC):
    """Abstract Base Class for candidate rerankers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name identifier of the reranker."""
        pass

    @abstractmethod
    def rerank(
        self,
        query: Any,
        candidates: List[SearchResult],
        top_k: int = 50,
    ) -> List[SearchResult]:
        """
        Rerank a candidate list based on specialized scoring logic.

        Args:
            query: Natural language query string or parsed query object.
            candidates: Ranked list of SearchResult objects from fusion.
            top_k: Number of candidates to return after reranking.

        Returns:
            Re-ordered List[SearchResult] with updated scores/metadata.
        """
        pass
