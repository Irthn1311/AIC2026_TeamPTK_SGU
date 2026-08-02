"""Replaceable vector-index interface."""

from collections.abc import Sequence
from typing import Protocol

import numpy as np


class VectorIndex(Protocol):
    """Contract for NumPy and future FAISS vector indexes."""

    @property
    def size(self) -> int:
        """Return indexed vector count."""
        ...

    @property
    def dimension(self) -> int:
        """Return indexed vector dimension."""
        ...

    def build(self, vectors: np.ndarray, ids: Sequence[str]) -> None:
        """Build an index over external string identifiers."""
        ...

    def search(self, query_vectors: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return score and external-ID matrices."""
        ...
