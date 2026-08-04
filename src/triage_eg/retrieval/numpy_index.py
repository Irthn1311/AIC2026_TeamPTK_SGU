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


class NumPyMemmapExactIndex:
    """Chunked exact cosine/dot search over a NumPy matrix and stored norms."""

    def __init__(
        self,
        vectors: np.ndarray,
        norms: np.ndarray,
        *,
        metric: str = "cosine",
        chunk_rows: int = 16_384,
    ) -> None:
        if metric not in {"cosine", "dot"}:
            raise ValueError("metric must be cosine or dot")
        if vectors.ndim != 2 or vectors.shape[0] == 0 or vectors.shape[1] == 0:
            raise ValueError("vectors must be a non-empty 2D matrix")
        if norms.shape != (vectors.shape[0],):
            raise ValueError("norms must have one value per vector row")
        if chunk_rows <= 0:
            raise ValueError("chunk_rows must be positive")
        if not np.isfinite(norms).all() or np.any(norms <= 0):
            raise ValueError("stored norms must be finite and positive")
        self._vectors = vectors
        self._norms = norms
        self.metric = metric
        self.chunk_rows = chunk_rows

    @property
    def size(self) -> int:
        return int(self._vectors.shape[0])

    @property
    def dimension(self) -> int:
        return int(self._vectors.shape[1])

    def vectors_at(self, rows: np.ndarray) -> np.ndarray:
        """Return selected stored rows as float32 for sanity/benchmark queries."""

        indices = np.asarray(rows, dtype=np.int64)
        if indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= self.size):
            raise IndexError("vector row selection is out of range")
        return np.asarray(self._vectors[indices], dtype=np.float32)

    @staticmethod
    def _bounded_topk(
        scores: np.ndarray, rows: np.ndarray, result_count: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Select top-k without sorting the complete score array, including stable ties."""

        if len(scores) <= result_count:
            selected = np.arange(len(scores))
        else:
            boundary = np.partition(scores, len(scores) - result_count)[
                len(scores) - result_count
            ]
            better = np.flatnonzero(scores > boundary)
            tied = np.flatnonzero(scores == boundary)
            tied = tied[np.argsort(rows[tied], kind="stable")]
            selected = np.concatenate((better, tied[: result_count - len(better)]))
        order = np.lexsort((rows[selected], -scores[selected]))
        selected = selected[order]
        return scores[selected], rows[selected]

    def search(self, query_vectors: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        queries = np.asarray(query_vectors, dtype=np.float32)
        if queries.ndim == 1:
            queries = queries.reshape(1, -1)
        if queries.ndim != 2 or queries.shape[0] == 0 or queries.shape[1] != self.dimension:
            raise ValueError(f"queries must have shape (n, {self.dimension})")
        if not np.isfinite(queries).all():
            raise ValueError("query vectors must be finite")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        query_norms = np.linalg.norm(queries, axis=1)
        if np.any(query_norms == 0):
            raise ValueError("zero-norm query vectors are invalid")
        result_count = min(top_k, self.size)
        best_scores = [np.empty(0, dtype=np.float32) for _ in queries]
        best_rows = [np.empty(0, dtype=np.int64) for _ in queries]
        for start in range(0, self.size, self.chunk_rows):
            stop = min(start + self.chunk_rows, self.size)
            chunk = np.asarray(self._vectors[start:stop], dtype=np.float32)
            chunk_scores = chunk @ queries.T
            if self.metric == "cosine":
                denominator = self._norms[start:stop, None] * query_norms[None, :]
                chunk_scores = chunk_scores / denominator
            rows = np.arange(start, stop, dtype=np.int64)
            for query_index in range(len(queries)):
                combined_scores = np.concatenate(
                    (best_scores[query_index], chunk_scores[:, query_index].astype(np.float32))
                )
                combined_rows = np.concatenate((best_rows[query_index], rows))
                best_scores[query_index], best_rows[query_index] = self._bounded_topk(
                    combined_scores, combined_rows, result_count
                )
        all_scores = np.stack(best_scores)
        all_rows = np.stack(best_rows)
        return all_scores, all_rows
