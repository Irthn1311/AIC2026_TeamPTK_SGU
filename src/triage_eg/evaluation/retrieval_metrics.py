"""Generic diagnostic retrieval metrics."""

from collections.abc import Sequence


def reciprocal_rank(relevance: Sequence[bool]) -> float:
    """Return reciprocal rank of the first relevant result."""

    for rank, relevant in enumerate(relevance, start=1):
        if relevant:
            return 1.0 / rank
    return 0.0


def recall_at_k(relevance: Sequence[bool], total_relevant: int, k: int) -> float:
    """Return retrieved relevant items divided by known relevant items."""

    if total_relevant <= 0 or k <= 0:
        raise ValueError("total_relevant and k must be greater than zero")
    return sum(relevance[:k]) / total_relevant
