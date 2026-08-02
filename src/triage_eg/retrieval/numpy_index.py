"""In-memory NumPy brute-force cosine index."""

from collections.abc import Sequence

import numpy as np


class NumPyFlatCosineIndex:
    """Exact cosine search baseline suitable only for small collections."""

    def __init__(self) -> None:
        self._vectors: np.ndarray | None = None
        self._ids: np.ndarray | None = None

    @property
    def size(self) -> int:
        """Return indexed vector count."""

        return 0 if self._vectors is None else self._vectors.shape[0]

    @property
    def dimension(self) -> int:
        """Return vector dimension, or zero before build."""

        return 0 if self._vectors is None else self._vectors.shape[1]

    def build(self, vectors: np.ndarray, ids: Sequence[str]) -> None:
        """Normalize and retain a two-dimensional vector matrix."""

        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
            raise ValueError("vectors must be a non-empty two-dimensional matrix")
        if matrix.shape[0] != len(ids):
            raise ValueError("ids count must match vector count")
        if len(set(ids)) != len(ids):
            raise ValueError("index ids must be unique")
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError("zero vectors cannot be indexed for cosine search")
        self._vectors = matrix / norms
        self._ids = np.asarray(ids, dtype=str)

    def search(self, query_vectors: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        """Search one or more queries and return descending scores and IDs."""

        if self._vectors is None or self._ids is None:
            raise RuntimeError("Index must be built before search")
        queries = np.asarray(query_vectors, dtype=np.float32)
        if queries.ndim != 2 or queries.shape[1] != self.dimension:
            raise ValueError(f"queries must have shape (n, {self.dimension})")
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        norms = np.linalg.norm(queries, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError("zero query vectors are invalid for cosine search")
        similarities = (queries / norms) @ self._vectors.T
        result_count = min(top_k, self.size)
        order = np.argsort(-similarities, axis=1, kind="stable")[:, :result_count]
        return np.take_along_axis(similarities, order, axis=1), self._ids[order]
