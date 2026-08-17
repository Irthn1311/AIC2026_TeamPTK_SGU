"""Exact float32 scoring backend for the persisted SigLIP2 diagnostic index."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from triage_eg.retrieval.numpy_index import NumPyMemmapExactIndex

from .index import validate_siglip2_index


class Siglip2ExactBackend:
    def __init__(
        self,
        index_root: str | Path,
        *,
        stage1_root: str | Path | None = None,
        chunk_rows: int = 16_384,
    ) -> None:
        validated = validate_siglip2_index(index_root, stage1_root=stage1_root)
        self.manifest: dict[str, Any] = validated["manifest"]
        self._backend = NumPyMemmapExactIndex(
            validated["vectors"], validated["norms"], metric="cosine", chunk_rows=chunk_rows
        )

    @property
    def size(self) -> int:
        return self._backend.size

    @property
    def dimension(self) -> int:
        return self._backend.dimension

    def score_all(self, query_vector: np.ndarray) -> np.ndarray:
        return self._backend.score_all(np.asarray(query_vector, dtype=np.float32))

    def score_many_all(self, query_vectors: np.ndarray) -> np.ndarray:
        return self._backend.score_many_all(np.asarray(query_vectors, dtype=np.float32))

    def search(self, query_vectors: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        return self._backend.search(np.asarray(query_vectors, dtype=np.float32), top_k)


__all__ = ["Siglip2ExactBackend"]
