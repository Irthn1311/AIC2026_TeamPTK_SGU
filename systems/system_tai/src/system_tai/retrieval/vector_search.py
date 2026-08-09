"""Deterministic exact NumPy cosine retrieval for the Phase 2 KIS baseline."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from system_tai.common.schemas import CandidateFrame, KISQuery, KISResult, RetrievalHit
from system_tai.features.btc_clip_store import FeatureStoreRegistry
from system_tai.features.query_encoder import TextEncoder


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
class _ScoredCandidate:
    video_id: str
    frame_id: int
    clip_row: int
    keyframe_order: int
    score: float


def _candidate_sort_key(candidate: _ScoredCandidate) -> tuple[float, str, int, int]:
    return (-candidate.score, candidate.video_id, candidate.frame_id, candidate.clip_row)


class ExactNumpyRetriever:
    def __init__(
        self,
        registry: FeatureStoreRegistry,
        text_encoder: TextEncoder,
        *,
        chunk_size: int = 4096,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if text_encoder.dimension != registry.embedding_dimension:
            raise ValueError(
                "text encoder/feature dimension mismatch: "
                f"encoder={text_encoder.dimension}, features={registry.embedding_dimension}"
            )
        self.registry = registry
        self.text_encoder = text_encoder
        self.chunk_size = chunk_size

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
        query_unit = _normalize_vector(
            query_vector, expected_dimension=self.registry.embedding_dimension
        )
        best_by_identity: dict[tuple[str, int], _ScoredCandidate] = {}
        for store in self.registry.stores:
            row_count = store.descriptor.row_count
            for start in range(0, row_count, self.chunk_size):
                stop = min(start + self.chunk_size, row_count)
                chunk = np.asarray(store.matrix[start:stop], dtype=np.float32)
                if not np.isfinite(chunk).all():
                    raise ValueError(
                        f"non-finite feature chunk: video={store.descriptor.video_id}, "
                        f"rows=[{start}, {stop})"
                    )
                norms = np.linalg.norm(chunk, axis=1)
                if np.any(norms == 0):
                    raise ValueError(f"zero-norm feature row in {store.descriptor.video_id}")
                scores = (chunk @ query_unit) / norms
                if not np.isfinite(scores).all():
                    raise ValueError("cosine computation produced NaN or Infinity")
                for local_row in range(stop - start):
                    clip_row = start + local_row
                    mapping = store.mappings[clip_row]
                    candidate = _ScoredCandidate(
                        video_id=store.descriptor.video_id,
                        frame_id=mapping.frame_id,
                        clip_row=clip_row,
                        keyframe_order=mapping.keyframe_order,
                        score=float(scores[local_row]),
                    )
                    identity = (candidate.video_id, candidate.frame_id)
                    existing = best_by_identity.get(identity)
                    if existing is None or _candidate_sort_key(
                        candidate
                    ) < _candidate_sort_key(existing):
                        best_by_identity[identity] = candidate
        ranked = sorted(best_by_identity.values(), key=_candidate_sort_key)[:top_k]
        candidates = tuple(
            CandidateFrame(
                video_id=item.video_id,
                frame_id=item.frame_id,
                clip_row=item.clip_row,
                keyframe_order=item.keyframe_order,
                score=float(item.score),
                rank=rank,
                source="clip_exact",
                diagnostic_metadata={
                    "feature_dimension": self.registry.embedding_dimension,
                    "chunk_size": self.chunk_size,
                },
            )
            for rank, item in enumerate(ranked, start=1)
        )
        return KISResult(query_id=query_id, ranked_candidates=candidates)

    def search_vectors(
        self,
        *,
        query_ids: Sequence[str],
        query_vectors: Sequence[Sequence[float] | NDArray[np.number]],
        top_k: int,
    ) -> dict[str, KISResult]:
        if not query_ids:
            raise ValueError("query_ids must not be empty")
        if len(query_ids) != len(query_vectors):
            raise ValueError("query_ids and query_vectors length mismatch")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        for qid in query_ids:
            if not isinstance(qid, str) or not qid.strip():
                raise ValueError("query_id must not be empty")
        if len(set(query_ids)) != len(query_ids):
            raise ValueError("query_ids must be unique")

        query_units = [
            _normalize_vector(
                vector, expected_dimension=self.registry.embedding_dimension
            )
            for vector in query_vectors
        ]
        best_by_query: list[dict[tuple[str, int], _ScoredCandidate]] = [
            {} for _ in query_units
        ]
        for store in self.registry.stores:
            row_count = store.descriptor.row_count
            for start in range(0, row_count, self.chunk_size):
                stop = min(start + self.chunk_size, row_count)
                chunk = np.asarray(store.matrix[start:stop], dtype=np.float32)
                if not np.isfinite(chunk).all():
                    raise ValueError(
                        f"non-finite feature chunk: video={store.descriptor.video_id}, "
                        f"rows=[{start}, {stop})"
                    )
                norms = np.linalg.norm(chunk, axis=1)
                if np.any(norms == 0):
                    raise ValueError(f"zero-norm feature row in {store.descriptor.video_id}")

                score_vectors: list[NDArray[np.float32]] = []
                for query_unit in query_units:
                    scores = (chunk @ query_unit) / norms
                    if not np.isfinite(scores).all():
                        raise ValueError("cosine computation produced NaN or Infinity")
                    score_vectors.append(scores)

                video_id = store.descriptor.video_id
                for local_row in range(stop - start):
                    clip_row = start + local_row
                    mapping = store.mappings[clip_row]
                    frame_id = mapping.frame_id
                    keyframe_order = mapping.keyframe_order
                    identity = (video_id, frame_id)

                    for q_idx in range(len(query_units)):
                        score = float(score_vectors[q_idx][local_row])
                        candidate = _ScoredCandidate(
                            video_id=video_id,
                            frame_id=frame_id,
                            clip_row=clip_row,
                            keyframe_order=keyframe_order,
                            score=score,
                        )
                        existing = best_by_query[q_idx].get(identity)
                        if existing is None or _candidate_sort_key(
                            candidate
                        ) < _candidate_sort_key(existing):
                            best_by_query[q_idx][identity] = candidate

        results: dict[str, KISResult] = {}
        for q_idx, query_id in enumerate(query_ids):
            ranked = sorted(best_by_query[q_idx].values(), key=_candidate_sort_key)[:top_k]
            candidates = tuple(
                CandidateFrame(
                    video_id=item.video_id,
                    frame_id=item.frame_id,
                    clip_row=item.clip_row,
                    keyframe_order=item.keyframe_order,
                    score=float(item.score),
                    rank=rank,
                    source="clip_exact",
                    diagnostic_metadata={
                        "feature_dimension": self.registry.embedding_dimension,
                        "chunk_size": self.chunk_size,
                    },
                )
                for rank, item in enumerate(ranked, start=1)
            )
            results[query_id] = KISResult(query_id=query_id, ranked_candidates=candidates)

        return results


class VectorSearch:
    """Compatibility exact search over a single unlabelled feature matrix."""

    def search(
        self,
        query_vector: Sequence[float],
        feature_matrix: NDArray[np.number],
        *,
        top_k: int,
    ) -> Sequence[RetrievalHit]:
        matrix = np.asarray(feature_matrix)
        if matrix.ndim != 2:
            raise ValueError("feature_matrix must be two-dimensional")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        query = _normalize_vector(query_vector, expected_dimension=matrix.shape[1])
        working = np.asarray(matrix, dtype=np.float32)
        if not np.isfinite(working).all():
            raise ValueError("feature_matrix contains NaN or Infinity")
        norms = np.linalg.norm(working, axis=1)
        if np.any(norms == 0):
            raise ValueError("feature_matrix contains a zero-norm row")
        scores = (working @ query) / norms
        rows = sorted(range(matrix.shape[0]), key=lambda row: (-float(scores[row]), row))
        return tuple(
            RetrievalHit(clip_row=row, score=float(scores[row]), rank=rank)
            for rank, row in enumerate(rows[:top_k], start=1)
        )
