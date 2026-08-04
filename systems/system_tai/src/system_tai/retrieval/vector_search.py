"""Vector-search interface for an externally supplied compatible query vector."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from system_tai.common.schemas import RetrievalHit


class VectorSearch:
    def search(
        self,
        query_vector: Sequence[float],
        feature_matrix: Any,
        *,
        top_k: int,
    ) -> Sequence[RetrievalHit]:
        raise NotImplementedError("VectorSearch.search is not implemented")
