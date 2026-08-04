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


def exact_cosine_self_diagnostics(
    vectors: np.ndarray,
    norms: np.ndarray,
    query_rows: np.ndarray,
    *,
    top_k: int = 5,
    diagnostic_top_k: int = 100,
    tie_tolerance: float = 1e-6,
    chunk_rows: int = 16_384,
) -> list[dict[str, object]]:
    """Compute tie-aware full-corpus self ranks without materializing all scores."""

    if vectors.ndim != 2 or vectors.shape[0] == 0 or vectors.shape[1] == 0:
        raise ValueError("vectors must be a non-empty 2D matrix")
    if norms.ndim != 1:
        raise ValueError("norms must be one-dimensional")
    rows = np.asarray(query_rows, dtype=np.int64)
    if rows.ndim != 1 or len(rows) == 0:
        raise ValueError("query_rows must be a non-empty one-dimensional array")
    if min(top_k, diagnostic_top_k, chunk_rows) <= 0 or tie_tolerance < 0:
        raise ValueError("diagnostic limits must be positive and tolerance non-negative")

    scan_size = min(len(vectors), len(norms))
    results: list[dict[str, object]] = []
    valid_result_indices: list[int] = []
    valid_queries: list[np.ndarray] = []
    direct_scores: list[float] = []
    query_norm_values: list[float] = []
    for query_row in rows:
        resolvable = 0 <= query_row < len(vectors) and query_row < len(norms)
        result: dict[str, object] = {
            "global_row": int(query_row),
            "query_row_resolvable": bool(resolvable),
            "corpus_shape_valid": len(vectors) == len(norms),
            "query_norm": None,
            "stored_norm": None,
            "direct_self_score": None,
            "self_score_abs_error_from_one": None,
            "self_score_finite": False,
            "search_self_score": None,
            "search_self_score_abs_delta_from_direct": None,
            "search_self_score_finite": False,
            "search_self_score_consistent": False,
            "raw_higher_count": None,
            "strictly_better_beyond_tolerance_count": None,
            "tie_equivalent_count": None,
            "rank_higher_count": None,
            "exact_equal_before_count": None,
            "actual_deterministic_rank": None,
            "included_top_k": False,
            "queried_row_top1": False,
            "queried_row_present_in_diagnostic_top_k": False,
            "non_finite_corpus_score_count": None,
            "diagnostic_top_candidates": [],
        }
        results.append(result)
        if not resolvable:
            continue
        query = np.asarray(vectors[int(query_row)], dtype=np.float32)
        query_norm = float(np.linalg.norm(query))
        stored_norm = float(norms[int(query_row)])
        direct_score = float("nan")
        if (
            np.isfinite(query).all()
            and np.isfinite(query_norm)
            and query_norm > 0
            and np.isfinite(stored_norm)
            and stored_norm > 0
        ):
            direct_score = float(
                np.float32(np.dot(query, query) / (query_norm * stored_norm))
            )
        result.update(
            {
                "query_norm": query_norm if np.isfinite(query_norm) else None,
                "stored_norm": stored_norm if np.isfinite(stored_norm) else None,
                "direct_self_score": direct_score if np.isfinite(direct_score) else None,
                "self_score_abs_error_from_one": (
                    abs(direct_score - 1.0) if np.isfinite(direct_score) else None
                ),
                "self_score_finite": bool(np.isfinite(direct_score)),
            }
        )
        if not np.isfinite(direct_score):
            continue
        valid_result_indices.append(len(results) - 1)
        valid_queries.append(query)
        direct_scores.append(direct_score)
        query_norm_values.append(query_norm)

    if not valid_queries or scan_size == 0:
        return results

    queries = np.stack(valid_queries).astype(np.float32, copy=False)
    direct = np.asarray(direct_scores, dtype=np.float32)
    query_norms = np.asarray(query_norm_values, dtype=np.float32)
    raw_higher = np.zeros(len(queries), dtype=np.int64)
    strict_higher = np.zeros(len(queries), dtype=np.int64)
    ties = np.zeros(len(queries), dtype=np.int64)
    rank_far_higher = np.zeros(len(queries), dtype=np.int64)
    search_self_scores = np.full(len(queries), np.nan, dtype=np.float32)
    rank_score_counts: list[dict[float, int]] = [{} for _ in queries]
    rank_score_before_counts: list[dict[float, int]] = [{} for _ in queries]
    non_finite = np.zeros(len(queries), dtype=np.int64)
    best_scores = [np.empty(0, dtype=np.float32) for _ in queries]
    best_rows = [np.empty(0, dtype=np.int64) for _ in queries]
    result_count = min(diagnostic_top_k, scan_size)

    for start in range(0, scan_size, chunk_rows):
        stop = min(start + chunk_rows, scan_size)
        chunk = np.asarray(vectors[start:stop], dtype=np.float32)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            scores = (chunk @ queries.T) / (norms[start:stop, None] * query_norms[None, :])
        chunk_global_rows = np.arange(start, stop, dtype=np.int64)
        for query_index in range(len(queries)):
            values = scores[:, query_index].astype(np.float32, copy=False)
            finite = np.isfinite(values)
            non_finite[query_index] += int(np.sum(~finite))
            finite_values = values[finite]
            finite_rows = chunk_global_rows[finite]
            self_score = direct[query_index]
            query_row = int(rows[valid_result_indices[query_index]])
            query_position = query_row - start
            if 0 <= query_position < len(values):
                search_self_scores[query_index] = values[query_position]
            other_rows = finite_rows != query_row
            other_values = finite_values[other_rows]
            raw_higher[query_index] += int(np.sum(other_values > self_score))
            strict_higher[query_index] += int(
                np.sum(other_values > self_score + tie_tolerance)
            )
            ties[query_index] += int(
                np.sum(np.abs(finite_values - self_score) <= tie_tolerance)
            )
            rank_window_tolerance = (
                2 * tie_tolerance + 4 * float(np.finfo(np.float32).eps)
            )
            other_rank_window = (
                np.abs(other_values - self_score) <= rank_window_tolerance
            )
            rank_far_higher[query_index] += int(
                np.sum((other_values > self_score) & ~other_rank_window)
            )
            rank_window = (
                np.abs(finite_values - self_score) <= rank_window_tolerance
            )
            window_values = finite_values[rank_window]
            window_rows = finite_rows[rank_window]
            unique_scores, score_counts = np.unique(
                window_values, return_counts=True
            )
            for score, count in zip(unique_scores, score_counts, strict=True):
                key = float(score)
                rank_score_counts[query_index][key] = (
                    rank_score_counts[query_index].get(key, 0)
                    + int(count)
                )
            before_scores, before_counts = np.unique(
                window_values[window_rows < query_row], return_counts=True
            )
            for score, count in zip(before_scores, before_counts, strict=True):
                key = float(score)
                rank_score_before_counts[query_index][key] = (
                    rank_score_before_counts[query_index].get(key, 0)
                    + int(count)
                )
            combined_scores = np.concatenate((best_scores[query_index], finite_values))
            combined_rows = np.concatenate((best_rows[query_index], finite_rows))
            best_scores[query_index], best_rows[query_index] = (
                NumPyMemmapExactIndex._bounded_topk(
                    combined_scores, combined_rows, result_count
                )
            )

    for query_index, result_index in enumerate(valid_result_indices):
        result = results[result_index]
        query_row = int(result["global_row"])
        selected_scores = best_scores[query_index]
        selected_rows = best_rows[query_index]
        requested_rows = selected_rows[: min(top_k, len(selected_rows))]
        diagnostic_positions = np.flatnonzero(selected_rows == query_row)
        search_self_score = float(search_self_scores[query_index])
        search_self_finite = bool(np.isfinite(search_self_score))
        search_self_delta = (
            abs(search_self_score - float(direct[query_index]))
            if search_self_finite
            else None
        )
        search_self_consistent = bool(
            search_self_delta is not None and search_self_delta <= tie_tolerance
        )
        rank_higher_count: int | None = None
        exact_equal_before_count: int | None = None
        actual_deterministic_rank: int | None = None
        if search_self_consistent:
            near_higher = sum(
                count
                for score, count in rank_score_counts[query_index].items()
                if score > search_self_score
            )
            rank_higher_count = int(rank_far_higher[query_index] + near_higher)
            exact_equal_before_count = rank_score_before_counts[query_index].get(
                search_self_score, 0
            )
            actual_deterministic_rank = (
                1 + rank_higher_count + exact_equal_before_count
            )
            if diagnostic_positions.size and actual_deterministic_rank != int(
                diagnostic_positions[0] + 1
            ):
                raise RuntimeError(
                    "Full-corpus rank counts disagree with diagnostic candidate order"
                )
        candidates = []
        for rank, (score, candidate_row) in enumerate(
            zip(selected_scores, selected_rows, strict=True), start=1
        ):
            candidates.append(
                {
                    "rank": rank,
                    "global_row": int(candidate_row),
                    "score": float(score),
                    "delta_from_self": float(score - direct[query_index]),
                    "within_tie_tolerance": bool(
                        abs(float(score - direct[query_index])) <= tie_tolerance
                    ),
                }
            )
        result.update(
            {
                "raw_higher_count": int(raw_higher[query_index]),
                "strictly_better_beyond_tolerance_count": int(
                    strict_higher[query_index]
                ),
                "tie_equivalent_count": int(ties[query_index]),
                "search_self_score": search_self_score if search_self_finite else None,
                "search_self_score_abs_delta_from_direct": search_self_delta,
                "search_self_score_finite": search_self_finite,
                "search_self_score_consistent": search_self_consistent,
                "rank_higher_count": rank_higher_count,
                "exact_equal_before_count": exact_equal_before_count,
                "actual_deterministic_rank": actual_deterministic_rank,
                "included_top_k": bool(np.any(requested_rows == query_row)),
                "queried_row_top1": bool(len(selected_rows) and selected_rows[0] == query_row),
                "queried_row_present_in_diagnostic_top_k": bool(
                    len(diagnostic_positions)
                ),
                "non_finite_corpus_score_count": int(non_finite[query_index]),
                "diagnostic_top_candidates": candidates,
            }
        )
    return results
