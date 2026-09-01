"""Task-neutral exact CLIP search over complete or selected video stores."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
from numpy.typing import NDArray

from system_tai.features.btc_clip_store import (
    FeatureStoreRegistry,
    LoadedVideoFeatureStore,
)


@dataclass(frozen=True, slots=True)
class RestrictedFrameHit:
    video_id: str
    frame_id: int
    clip_row: int
    keyframe_order: int
    pts_time: float
    cosine_score: float
    rank: int
    selection_source: str = "RAW"
    raw_local_rank: int = 0


@dataclass(frozen=True, slots=True)
class VideoMaximumHit:
    query_id: str
    video_id: str
    frame_id: int
    clip_row: int
    keyframe_order: int
    cosine_score: float
    rank: int
    top_m_score: float = 0.0
    top_m_peaks: tuple[tuple[int, float], ...] = ()


@dataclass(frozen=True, slots=True)
class VideoRestrictedSearchOutcome:
    rankings: Mapping[str, Mapping[str, tuple[RestrictedFrameHit, ...]]]
    physical_rows_scored: int
    video_store_scan_count: int
    candidate_selection_telemetry: Mapping[str, Mapping[str, Mapping[str, int]]] | None = None

    def __post_init__(self) -> None:
        frozen = {
            query_id: MappingProxyType(dict(per_video))
            for query_id, per_video in self.rankings.items()
        }
        object.__setattr__(self, "rankings", MappingProxyType(frozen))


@dataclass(frozen=True, slots=True)
class FullCorpusVideoMaximaOutcome:
    rankings: Mapping[str, tuple[VideoMaximumHit, ...]]
    physical_rows_scored: int
    video_store_scan_count: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "rankings", MappingProxyType(dict(self.rankings)))


@dataclass(frozen=True, slots=True)
class _UnrankedFrameHit:
    video_id: str
    frame_id: int
    clip_row: int
    keyframe_order: int
    pts_time: float
    cosine_score: float


def _frame_sort_key(hit: _UnrankedFrameHit) -> tuple[float, int, int]:
    return (-hit.cosine_score, hit.frame_id, hit.clip_row)


def _normalize_query_matrix(
    query_vectors: (
        Sequence[Sequence[float] | NDArray[np.number]] | NDArray[np.number]
    ),
    *,
    expected_dimension: int,
    already_normalized: bool = False,
) -> NDArray[np.float32]:
    if len(query_vectors) == 0:
        raise ValueError("query_vectors must not be empty")
    units: list[NDArray[np.float32]] = []
    for index, vector in enumerate(query_vectors):
        array = np.asarray(vector, dtype=np.float32)
        if array.shape != (expected_dimension,):
            raise ValueError(
                "query vector shape mismatch: "
                f"index={index}, observed={array.shape}, "
                f"expected=({expected_dimension},)"
            )
        if not np.isfinite(array).all():
            raise ValueError(f"query vector {index} contains NaN or Infinity")
        norm = float(np.linalg.norm(array))
        if not math.isfinite(norm) or norm <= 0:
            raise ValueError(f"query vector {index} must have a finite non-zero norm")
        units.append(
            np.asarray(array if already_normalized else array / norm, dtype=np.float32)
        )
    return np.stack(units).astype(np.float32, copy=False)


def select_hybrid_candidates(
    all_ordered: Sequence[_UnrankedFrameHit],
    *,
    total_budget: int,
    raw_top_k: int = 10,
    min_pts_gap_seconds: float = 5.0,
    compulsory_frame_ids: Sequence[int] | None = None,
) -> tuple[tuple[RestrictedFrameHit, ...], dict[str, object]]:
    """Select candidates using an equal-budget hybrid policy: Raw Top-K + Diverse Slots + Tail-Fill + Extras."""
    if type(total_budget) is not int or total_budget <= 0:
        raise ValueError("total_budget must be a positive integer")
    if type(raw_top_k) is not int or raw_top_k < 0:
        raise ValueError("raw_top_k must be a non-negative integer")
    if not math.isfinite(min_pts_gap_seconds) or min_pts_gap_seconds < 0:
        raise ValueError("min_pts_gap_seconds must be a non-negative finite float")

    total_available = len(all_ordered)
    raw_budget = min(raw_top_k, total_budget, total_available)
    raw_hits = list(all_ordered[:raw_budget])
    selected_pts = [hit.pts_time for hit in raw_hits]
    selected_fids = {hit.frame_id for hit in raw_hits}

    diverse_budget = total_budget - len(raw_hits)
    diverse_hits: list[_UnrankedFrameHit] = []

    if diverse_budget > 0:
        for hit in all_ordered[raw_budget:]:
            if len(diverse_hits) >= diverse_budget:
                break
            if hit.frame_id in selected_fids:
                continue
            if not any(abs(hit.pts_time - p) < min_pts_gap_seconds for p in selected_pts):
                diverse_hits.append(hit)
                selected_pts.append(hit.pts_time)
                selected_fids.add(hit.frame_id)

    tail_fill_hits: list[_UnrankedFrameHit] = []
    if len(raw_hits) + len(diverse_hits) < min(total_budget, total_available):
        for hit in all_ordered[raw_budget:]:
            if hit.frame_id in selected_fids:
                continue
            tail_fill_hits.append(hit)
            selected_fids.add(hit.frame_id)
            if len(raw_hits) + len(diverse_hits) + len(tail_fill_hits) >= min(total_budget, total_available):
                break

    normal_selected = raw_hits + diverse_hits + tail_fill_hits

    # Compulsory frames from DP temporal chain added as extras outside nominal K
    compulsory_set = set(compulsory_frame_ids or ())
    missing_compulsory: list[_UnrankedFrameHit] = []
    if compulsory_set:
        for hit in all_ordered:
            if hit.frame_id in compulsory_set and hit.frame_id not in selected_fids:
                missing_compulsory.append(hit)
                selected_fids.add(hit.frame_id)

    raw_ranks = {hit.frame_id: idx for idx, hit in enumerate(all_ordered, start=1)}

    hits_with_source: list[tuple[_UnrankedFrameHit, str]] = (
        [(hit, "RAW") for hit in raw_hits]
        + [(hit, "DIVERSE") for hit in diverse_hits]
        + [(hit, "TAIL_FILL") for hit in tail_fill_hits]
        + [(hit, "TEMPORAL_CHAIN_COMPULSORY") for hit in missing_compulsory]
    )

    final_hits = tuple(
        RestrictedFrameHit(
            video_id=hit.video_id,
            frame_id=hit.frame_id,
            clip_row=hit.clip_row,
            keyframe_order=hit.keyframe_order,
            pts_time=hit.pts_time,
            cosine_score=hit.cosine_score,
            rank=rank,
            selection_source=source,
            raw_local_rank=raw_ranks.get(hit.frame_id, 0),
        )
        for rank, (hit, source) in enumerate(hits_with_source, start=1)
    )

    telemetry = {
        "nominal_budget": total_budget,
        "normal_candidate_count": len(normal_selected),
        "compulsory_extra_count": len(missing_compulsory),
        "effective_candidate_count": len(final_hits),
        "raw_count": len(raw_hits),
        "diverse_count": len(diverse_hits),
        "tail_fill_count": len(tail_fill_hits),
    }
    return final_hits, telemetry


def rank_store_frames(
    store: LoadedVideoFeatureStore,
    *,
    query_ids: Sequence[str],
    query_vectors: (
        Sequence[Sequence[float] | NDArray[np.number]] | NDArray[np.number]
    ),
    expected_dimension: int,
    chunk_size: int,
    per_query_cap: int | Mapping[str, int] | None = None,
    query_vectors_are_normalized: bool = False,
    compulsory_frame_ids: Sequence[int] | None = None,
    enable_temporal_diversity: bool = False,
    temporal_diversity_gap_seconds: float = 5.0,
    raw_top_k: int = 10,
    return_telemetry: bool = False,
) -> dict[str, tuple[RestrictedFrameHit, ...]] | tuple[dict[str, tuple[RestrictedFrameHit, ...]], dict[str, dict[str, int]]]:
    """Rank distinct absolute frames for several vectors in one store traversal."""

    if not query_ids or len(query_ids) != len(query_vectors):
        raise ValueError("query_ids and query_vectors must have equal non-zero length")
    if len(set(query_ids)) != len(query_ids):
        raise ValueError("query_ids must be unique")
    if any(not isinstance(query_id, str) or not query_id.strip() for query_id in query_ids):
        raise ValueError("query_ids must contain non-empty strings")
    if type(chunk_size) is not int or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    if per_query_cap is not None:
        if isinstance(per_query_cap, Mapping):
            for qid in query_ids:
                if qid not in per_query_cap:
                    raise ValueError(f"Missing per_query_cap for query_id: '{qid}'")
                cap_val = per_query_cap[qid]
                if type(cap_val) is not int or cap_val <= 0:
                    raise ValueError(f"per_query_cap for '{qid}' must be a positive integer, got {cap_val}")
        elif type(per_query_cap) is not int or per_query_cap <= 0:
            raise ValueError("per_query_cap must be a positive integer or Mapping[str, int] when provided")
    if store.descriptor.embedding_dimension != expected_dimension:
        raise ValueError(
            f"feature dimension mismatch for {store.descriptor.video_id}: "
            f"{store.descriptor.embedding_dimension} != {expected_dimension}"
        )
    if store.matrix.ndim != 2 or store.matrix.shape[0] != len(store.mappings):
        raise ValueError(f"invalid feature/mapping shape for {store.descriptor.video_id}")

    query_matrix = _normalize_query_matrix(
        query_vectors,
        expected_dimension=expected_dimension,
        already_normalized=query_vectors_are_normalized,
    )
    best_by_query_frame: list[dict[int, _UnrankedFrameHit]] = [
        {} for _ in query_ids
    ]
    row_count = store.descriptor.row_count
    for start in range(0, row_count, chunk_size):
        stop = min(start + chunk_size, row_count)
        chunk = np.asarray(store.matrix[start:stop], dtype=np.float32)
        if not np.isfinite(chunk).all():
            raise ValueError(
                f"non-finite feature chunk: video={store.descriptor.video_id}, "
                f"rows=[{start}, {stop})"
            )
        norms = np.linalg.norm(chunk, axis=1)
        if np.any(~np.isfinite(norms)) or np.any(norms <= 0):
            raise ValueError(f"non-finite or zero-norm feature row in {store.descriptor.video_id}")
        if len(query_ids) == 1:
            scores = ((chunk @ query_matrix[0]) / norms)[:, None]
        else:
            scores = (chunk @ query_matrix.T) / norms[:, None]
        if not np.isfinite(scores).all():
            raise ValueError("restricted cosine computation produced NaN or Infinity")

        for local_row in range(stop - start):
            clip_row = start + local_row
            mapping = store.mappings[clip_row]
            for query_index in range(len(query_ids)):
                hit = _UnrankedFrameHit(
                    video_id=store.descriptor.video_id,
                    frame_id=mapping.frame_id,
                    clip_row=clip_row,
                    keyframe_order=mapping.keyframe_order,
                    pts_time=mapping.pts_time,
                    cosine_score=float(scores[local_row, query_index]),
                )
                existing = best_by_query_frame[query_index].get(hit.frame_id)
                if existing is None or _frame_sort_key(hit) < _frame_sort_key(existing):
                    best_by_query_frame[query_index][hit.frame_id] = hit

    rankings: dict[str, tuple[RestrictedFrameHit, ...]] = {}
    telemetries: dict[str, dict[str, int]] = {}
    compulsory_set = set(compulsory_frame_ids or ())
    for query_index, query_id in enumerate(query_ids):
        all_ordered = sorted(
            best_by_query_frame[query_index].values(),
            key=_frame_sort_key,
        )
        current_cap = (
            per_query_cap[query_id]
            if isinstance(per_query_cap, Mapping)
            else per_query_cap
        )
        if enable_temporal_diversity and current_cap is not None:
            current_raw_k = min(raw_top_k, current_cap)
            hybrid_hits, tel = select_hybrid_candidates(
                all_ordered,
                total_budget=current_cap,
                raw_top_k=current_raw_k,
                min_pts_gap_seconds=temporal_diversity_gap_seconds,
                compulsory_frame_ids=compulsory_frame_ids,
            )
            rankings[query_id] = hybrid_hits
            telemetries[query_id] = tel
        else:
            if current_cap is not None:
                capped = all_ordered[:current_cap]
                if compulsory_set:
                    capped_fids = {hit.frame_id for hit in capped}
                    missing_compulsory = [
                        hit for hit in all_ordered[current_cap:]
                        if hit.frame_id in compulsory_set and hit.frame_id not in capped_fids
                    ]
                    ordered = capped + missing_compulsory
                else:
                    missing_compulsory = []
                    ordered = capped
            else:
                capped = all_ordered
                missing_compulsory = []
                ordered = all_ordered

            raw_ranks = {hit.frame_id: idx for idx, hit in enumerate(all_ordered, start=1)}
            capped_fid_set = {hit.frame_id for hit in capped}

            rankings[query_id] = tuple(
                RestrictedFrameHit(
                    video_id=hit.video_id,
                    frame_id=hit.frame_id,
                    clip_row=hit.clip_row,
                    keyframe_order=hit.keyframe_order,
                    pts_time=hit.pts_time,
                    cosine_score=hit.cosine_score,
                    rank=rank,
                    selection_source="RAW" if hit.frame_id in capped_fid_set else "TEMPORAL_CHAIN_COMPULSORY",
                    raw_local_rank=raw_ranks.get(hit.frame_id, 0),
                )
                for rank, hit in enumerate(ordered, start=1)
            )
            telemetries[query_id] = {
                "nominal_budget": current_cap if current_cap is not None else len(all_ordered),
                "normal_candidate_count": len(capped),
                "compulsory_extra_count": len(missing_compulsory),
                "effective_candidate_count": len(ordered),
                "raw_count": len(capped),
                "diverse_count": 0,
                "tail_fill_count": 0,
            }
    if return_telemetry:
        return rankings, telemetries
    return rankings


class VideoRestrictedFeatureSearcher:
    """Exact, bounded task-neutral search over loaded BTC keyframe features."""

    def __init__(self, registry: FeatureStoreRegistry, *, chunk_size: int = 4096) -> None:
        if type(chunk_size) is not int or chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer")
        self.registry = registry
        self.chunk_size = chunk_size

    def search_video_maxima(
        self,
        *,
        query_ids: Sequence[str],
        query_vectors: (
            Sequence[Sequence[float] | NDArray[np.number]] | NDArray[np.number]
        ),
        top_m_evidence_cap: int = 1,
        top_m_min_frame_gap: int = 60,
        top_m_weights: Sequence[float] = (1.0,),
    ) -> FullCorpusVideoMaximaOutcome:
        """Return every video's best keyframe/Top-M evidence for each query without global Top-K."""

        if type(top_m_evidence_cap) is not int or top_m_evidence_cap <= 0:
            raise ValueError("top_m_evidence_cap must be a positive integer")
        if type(top_m_min_frame_gap) is not int or top_m_min_frame_gap < 0:
            raise ValueError("top_m_min_frame_gap must be a non-negative integer")
        if not top_m_weights or any(
            not math.isfinite(w) or w <= 0 for w in top_m_weights
        ):
            raise ValueError("top_m_weights must contain positive finite floats")

        by_query: dict[str, list[VideoMaximumHit]] = {
            query_id: [] for query_id in query_ids
        }
        for store in sorted(self.registry.stores, key=lambda s: s.descriptor.video_id):
            video_id = store.descriptor.video_id
            per_store = rank_store_frames(
                store,
                query_ids=query_ids,
                query_vectors=query_vectors,
                expected_dimension=self.registry.embedding_dimension,
                chunk_size=self.chunk_size,
                per_query_cap=None,
            )
            assert isinstance(per_store, dict)
            for query_id in query_ids:
                hits = per_store[query_id]
                best_hit = hits[0]

                # Top-M Multi-Evidence Pooling
                if top_m_evidence_cap <= 1:
                    top_m_val = float(best_hit.cosine_score)
                    selected_peaks = ()
                else:
                    selected_peaks: list[tuple[int, float]] = []
                    weights = list(top_m_weights[:top_m_evidence_cap])
                    if len(weights) < top_m_evidence_cap:
                        weights.extend([weights[-1]] * (top_m_evidence_cap - len(weights)))

                    for h in hits:
                        if len(selected_peaks) >= top_m_evidence_cap:
                            break
                        if not selected_peaks:
                            selected_peaks.append((h.frame_id, float(h.cosine_score)))
                        else:
                            if all(abs(h.frame_id - peak_fid) >= top_m_min_frame_gap for peak_fid, _ in selected_peaks):
                                selected_peaks.append((h.frame_id, float(h.cosine_score)))

                    selected_scores = [p[1] for p in selected_peaks]
                    w_sum = sum(weights[:len(selected_scores)])
                    top_m_val = (
                        sum(w * s for w, s in zip(weights, selected_scores)) / w_sum
                        if w_sum > 0
                        else float(best_hit.cosine_score)
                    )
                    selected_peaks = tuple(selected_peaks)

                by_query[query_id].append(
                    VideoMaximumHit(
                        query_id=query_id,
                        video_id=best_hit.video_id,
                        frame_id=best_hit.frame_id,
                        clip_row=best_hit.clip_row,
                        keyframe_order=best_hit.keyframe_order,
                        cosine_score=best_hit.cosine_score,
                        rank=0,
                        top_m_score=top_m_val,
                        top_m_peaks=selected_peaks,
                    )
                )

        rankings: dict[str, tuple[VideoMaximumHit, ...]] = {}
        for query_id in query_ids:
            # Sort by top_m_score if multi-evidence, else cosine_score
            ordered = sorted(
                by_query[query_id],
                key=lambda hit: (
                    -(hit.top_m_score if top_m_evidence_cap > 1 else hit.cosine_score),
                    hit.video_id,
                ),
            )
            rankings[query_id] = tuple(
                VideoMaximumHit(
                    query_id=hit.query_id,
                    video_id=hit.video_id,
                    frame_id=hit.frame_id,
                    clip_row=hit.clip_row,
                    keyframe_order=hit.keyframe_order,
                    cosine_score=hit.cosine_score,
                    rank=rank,
                    top_m_score=hit.top_m_score,
                    top_m_peaks=hit.top_m_peaks,
                )
                for rank, hit in enumerate(ordered, start=1)
            )
        return FullCorpusVideoMaximaOutcome(
            rankings=rankings,
            physical_rows_scored=self.registry.total_rows,
            video_store_scan_count=len(self.registry.stores),
        )

    def search_selected_videos(
        self,
        *,
        video_ids: Sequence[str],
        query_ids: Sequence[str],
        query_vectors: (
            Sequence[Sequence[float] | NDArray[np.number]] | NDArray[np.number]
        ),
        per_query_result_cap: int | Mapping[str, int],
        compulsory_frame_ids_by_video: Mapping[str, Sequence[int]] | None = None,
        enable_temporal_diversity: bool = False,
        temporal_diversity_gap_seconds: float = 5.0,
        raw_top_k: int = 10,
    ) -> VideoRestrictedSearchOutcome:
        if not video_ids:
            raise ValueError("video_ids must not be empty")
        if len(set(video_ids)) != len(video_ids):
            raise ValueError("video_ids must be unique")
        if isinstance(per_query_result_cap, Mapping):
            for qid in query_ids:
                if qid not in per_query_result_cap:
                    raise ValueError(f"Missing per_query_result_cap for query_id: '{qid}'")
                cap_val = per_query_result_cap[qid]
                if type(cap_val) is not int or cap_val <= 0:
                    raise ValueError(f"per_query_result_cap for '{qid}' must be a positive integer, got {cap_val}")
        elif type(per_query_result_cap) is not int or per_query_result_cap <= 0:
            raise ValueError("per_query_result_cap must be a positive integer or Mapping[str, int]")

        selected = tuple(sorted(video_ids))
        rankings: dict[str, dict[str, tuple[RestrictedFrameHit, ...]]] = {
            query_id: {} for query_id in query_ids
        }
        telemetries: dict[str, dict[str, dict[str, int]]] = {
            query_id: {} for query_id in query_ids
        }
        rows_scored = 0
        compulsory_map = compulsory_frame_ids_by_video or {}
        for video_id in selected:
            store = self.registry.get(video_id)
            per_store, per_store_telemetry = rank_store_frames(
                store,
                query_ids=query_ids,
                query_vectors=query_vectors,
                expected_dimension=self.registry.embedding_dimension,
                chunk_size=self.chunk_size,
                per_query_cap=per_query_result_cap,
                compulsory_frame_ids=compulsory_map.get(video_id),
                enable_temporal_diversity=enable_temporal_diversity,
                temporal_diversity_gap_seconds=temporal_diversity_gap_seconds,
                raw_top_k=raw_top_k,
                return_telemetry=True,
            )
            rows_scored += store.descriptor.row_count
            for query_id in query_ids:
                rankings[query_id][video_id] = per_store[query_id]
                telemetries[query_id][video_id] = per_store_telemetry[query_id]
        return VideoRestrictedSearchOutcome(
            rankings=rankings,
            physical_rows_scored=rows_scored,
            video_store_scan_count=len(selected),
            candidate_selection_telemetry=telemetries,
        )
