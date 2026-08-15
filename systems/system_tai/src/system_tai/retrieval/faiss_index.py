"""FAISS-backed Vector Index and Retriever with transparent CPU/GPU and NumPy fallback."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from system_tai.common.schemas import CandidateFrame, KISQuery, KISResult
from system_tai.features.btc_clip_store import FeatureStoreRegistry
from system_tai.features.query_encoder import TextEncoder

try:
    import faiss  # type: ignore[import-untyped]
    HAS_FAISS = True
except ImportError:
    faiss = None
    HAS_FAISS = False


def _normalize_vector(
    vector: Sequence[float] | NDArray[np.number], *, expected_dimension: int
) -> NDArray[np.float32]:
    array = np.asarray(vector, dtype=np.float32)
    if array.shape != (expected_dimension,):
        raise ValueError(
            f"query vector shape mismatch: observed={array.shape}, expected=({expected_dimension},)"
        )
    if not np.isfinite(array).all():
        raise ValueError("query vector contains NaN or Infinity")
    norm = float(np.linalg.norm(array))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("query vector must have a finite non-zero norm")
    return np.asarray(array / norm, dtype=np.float32)


@dataclass(frozen=True, slots=True)
class VectorRecord:
    video_id: str
    actual_frame_id: int
    clip_row: int
    keyframe_order: int


class FaissVectorIndex:
    """FAISS-compatible high-performance vector index supporting FlatIP and IVF."""

    def __init__(
        self,
        dimension: int,
        *,
        index_type: str = "flat",
        nlist: int = 100,
        nprobe: int = 10,
        force_numpy: bool = False,
    ) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self.dimension = dimension
        self.index_type = index_type.lower()
        self.nlist = nlist
        self.nprobe = nprobe
        self.records: list[VectorRecord] = []
        self._vectors: list[np.ndarray] = []
        self._faiss_index: Any = None
        self._is_trained = False
        self.use_faiss = HAS_FAISS and not force_numpy

        if self.use_faiss:
            if self.index_type == "flat":
                self._faiss_index = faiss.IndexFlatIP(self.dimension)
                self._is_trained = True
            elif self.index_type == "ivf":
                quantizer = faiss.IndexFlatIP(self.dimension)
                self._faiss_index = faiss.IndexIVFFlat(
                    quantizer, self.dimension, self.nlist, faiss.METRIC_INNER_PRODUCT
                )
                self._faiss_index.nprobe = self.nprobe
            else:
                raise ValueError(f"unsupported index_type: {self.index_type}")

    @property
    def total_vectors(self) -> int:
        return len(self.records)

    def add(
        self,
        matrix: NDArray[np.float32],
        records: Sequence[VectorRecord],
    ) -> None:
        """Add a matrix of vectors and corresponding metadata records."""
        arr = np.asarray(matrix, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != self.dimension:
            raise ValueError(
                f"matrix shape mismatch: observed={arr.shape}, expected=(N, {self.dimension})"
            )
        if len(records) != arr.shape[0]:
            raise ValueError(
                f"record count mismatch: {len(records)} records vs {arr.shape[0]} rows"
            )
        if not np.isfinite(arr).all():
            raise ValueError("matrix contains NaN or Inf values")

        # Normalize unit vectors
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError("matrix contains zero-norm vector")
        unit_vectors = arr / norms

        self.records.extend(records)

        if self.use_faiss:
            if self.index_type == "ivf" and not self._is_trained:
                self._faiss_index.train(unit_vectors)
                self._is_trained = True
            self._faiss_index.add(unit_vectors)
        else:
            self._vectors.append(unit_vectors)

    def search(
        self,
        query_vector: Sequence[float] | NDArray[np.number],
        top_k: int,
    ) -> list[tuple[VectorRecord, float]]:
        """Search top-K nearest neighbors for the given query vector."""
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if not self.records:
            return []

        q_unit = _normalize_vector(query_vector, expected_dimension=self.dimension)

        if self.use_faiss and self._faiss_index is not None:
            q_batch = np.ascontiguousarray(q_unit[None, :], dtype=np.float32)
            k = min(top_k, len(self.records))
            scores_batch, indices_batch = self._faiss_index.search(q_batch, k)
            scores = scores_batch[0]
            indices = indices_batch[0]

            results = []
            for idx, score in zip(indices, scores):
                if idx < 0 or idx >= len(self.records):
                    continue
                results.append((self.records[idx], float(score)))
            return results

        # NumPy fallback search
        all_vecs = np.vstack(self._vectors) if self._vectors else np.empty((0, self.dimension), dtype=np.float32)
        scores = all_vecs @ q_unit
        k = min(top_k, len(self.records))
        top_indices = np.argsort(-scores)[:k]

        return [(self.records[i], float(scores[i])) for i in top_indices]


class FaissVectorRetriever:
    """Drop-in Vector Retriever using FaissVectorIndex."""

    def __init__(
        self,
        registry: FeatureStoreRegistry,
        text_encoder: TextEncoder,
        *,
        index_type: str = "flat",
        force_numpy: bool = False,
    ) -> None:
        if text_encoder.dimension != registry.embedding_dimension:
            raise ValueError(
                "text encoder/feature dimension mismatch: "
                f"encoder={text_encoder.dimension}, features={registry.embedding_dimension}"
            )
        self.registry = registry
        self.text_encoder = text_encoder
        self.index = FaissVectorIndex(
            dimension=registry.embedding_dimension,
            index_type=index_type,
            force_numpy=force_numpy,
        )
        self._build_index()

    def _build_index(self) -> None:
        for store in self.registry.stores:
            desc = store.descriptor
            records = [
                VectorRecord(
                    video_id=desc.video_id,
                    actual_frame_id=row.actual_frame_id,
                    clip_row=row.clip_row,
                    keyframe_order=row.keyframe_order if row.keyframe_order is not None else 0,
                )
                for row in desc.rows
            ]
            self.index.add(np.asarray(store.matrix, dtype=np.float32), records)

    def retrieve(self, query: KISQuery) -> KISResult:
        return self.search_vector(
            query_id=query.query_id,
            query_vector=self.text_encoder.encode(query.text),
            top_k=query.top_k,
        )

    def search_vector(
        self,
        *,
        query_id: str,
        query_vector: Sequence[float] | NDArray[np.number],
        top_k: int,
    ) -> KISResult:
        if not query_id.strip():
            raise ValueError("query_id must not be empty")
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        raw_results = self.index.search(query_vector, top_k=top_k * 2)

        # Deduplicate per (video_id, actual_frame_id) keeping highest score
        best_by_identity: dict[tuple[str, int], tuple[VectorRecord, float]] = {}
        for rec, score in raw_results:
            key = (rec.video_id, rec.actual_frame_id)
            if key not in best_by_identity or score > best_by_identity[key][1]:
                best_by_identity[key] = (rec, score)

        # Sort: score desc, video asc, frame asc, clip_row asc
        sorted_candidates = sorted(
            best_by_identity.values(),
            key=lambda item: (-item[1], item[0].video_id, item[0].actual_frame_id, item[0].clip_row),
        )[:top_k]

        candidates = tuple(
            CandidateFrame(
                video_id=rec.video_id,
                frame_id=rec.actual_frame_id,
                clip_row=rec.clip_row,
                keyframe_order=rec.keyframe_order,
                score=float(score),
                rank=rank,
                source="faiss_clip",
                diagnostic_metadata={
                    "dimension": self.index.dimension,
                    "index_type": self.index.index_type,
                    "use_faiss": self.index.use_faiss,
                },
            )
            for rank, (rec, score) in enumerate(sorted_candidates, start=1)
        )

        return KISResult(
            query_id=query_id,
            ranked_candidates=candidates,
        )
