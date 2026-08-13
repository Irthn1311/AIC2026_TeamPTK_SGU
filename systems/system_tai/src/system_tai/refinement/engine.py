"""Deterministic bounded coarse-to-fine exact-frame refinement engine."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from system_tai.common.schemas import CandidateFrame, KISResult
from system_tai.refinement.clip_encoder import RefinementEncoder
from system_tai.refinement.models import (
    CandidateFailurePolicy,
    MissingRawVideoPolicy,
    Phase3Candidate,
    QueryRefinementError,
    RefinedCandidate,
    RefinementConfig,
    RefinementQuery,
    RefinementStatus,
)
from system_tai.refinement.video import (
    DecodedFrame,
    DecodeRequest,
    RawVideoError,
    RawVideoRecord,
    RawVideoRegistry,
    VideoDecoder,
    VideoProbe,
)
from system_tai.retrieval.multi_query import QueryVariant

FrameEmbeddingKey = tuple[str, int]
FrameEmbeddingCache = MutableMapping[FrameEmbeddingKey, NDArray[np.float32]]


def _encode_frames_with_cache(
    *,
    video_id: str,
    frames: Sequence[DecodedFrame],
    encoder: RefinementEncoder,
    batch_size: int,
    frame_embedding_cache: FrameEmbeddingCache | None,
) -> NDArray[np.float32]:
    if not frames:
        return encoder.encode_images(
            [],
            batch_size=batch_size,
        )

    if frame_embedding_cache is None:
        return encoder.encode_images(
            [frame.image for frame in frames],
            batch_size=batch_size,
        )

    missing_indices: list[int] = []
    missing_keys: list[FrameEmbeddingKey] = []
    seen_missing: set[FrameEmbeddingKey] = set()

    for idx, frame in enumerate(frames):
        key: FrameEmbeddingKey = (video_id, frame.absolute_frame_id)
        if key not in frame_embedding_cache and key not in seen_missing:
            seen_missing.add(key)
            missing_indices.append(idx)
            missing_keys.append(key)

    if missing_keys:
        missing_images = [frames[i].image for i in missing_indices]
        encoded_misses = encoder.encode_images(
            missing_images,
            batch_size=batch_size,
        )
        if len(encoded_misses) != len(missing_keys):
            raise ValueError("Encoded image count mismatch with missing keys")
        for key, row in zip(missing_keys, encoded_misses):
            frame_embedding_cache[key] = np.asarray(row, dtype=np.float32).copy()

    result_rows: list[NDArray[np.float32]] = []
    for frame in frames:
        key = (video_id, frame.absolute_frame_id)
        result_rows.append(frame_embedding_cache[key])

    return np.vstack(result_rows).astype(np.float32)


@dataclass(frozen=True, slots=True)
class LocalFrameFusion:
    absolute_frame_id: int
    fusion_score: float
    variant_hit_count: int
    best_individual_rank: int
    per_variant_provenance: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True, slots=True)
class QueryRefinementOutcome:
    query_id: str
    result: KISResult
    candidates: tuple[RefinedCandidate, ...]
    warnings: tuple[str, ...]
    timings: Mapping[str, float | int]


@dataclass(frozen=True, slots=True)
class SelectedRefinementOutcome:
    candidates: tuple[RefinedCandidate, ...]
    warnings: tuple[str, ...]
    timings: Mapping[str, float | int]


@dataclass(frozen=True, slots=True)
class SharedRefinementGroup:
    """One semantic scoring group inside a shared raw-decode query batch."""

    query_id: str
    variants: tuple[QueryVariant, ...]
    candidates: tuple[Phase3Candidate, ...]
    text_embeddings: NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class SharedRefinementBatchOutcome:
    groups: tuple[SelectedRefinementOutcome, ...]
    timings: Mapping[str, float | int | bool]


@dataclass(frozen=True, slots=True)
class _SelectedAnchorWork:
    candidate: Phase3Candidate
    probe: VideoProbe
    window_start: int
    window_end: int
    coarse_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _SharedCandidateWork:
    group_index: int
    candidate: Phase3Candidate
    variants: tuple[QueryVariant, ...]
    text_embeddings: NDArray[np.float32]
    probe: VideoProbe
    window_start: int
    window_end: int
    coarse_ids: tuple[int, ...]


def build_frame_window(
    candidate_frame_id: int,
    *,
    fps: float,
    total_frame_count: int,
    before_seconds: float,
    after_seconds: float,
) -> tuple[int, int]:
    if candidate_frame_id < 0 or candidate_frame_id >= total_frame_count:
        raise RawVideoError(f"candidate_frame_id outside raw-video bounds: {candidate_frame_id}")
    start = max(0, candidate_frame_id - int(round(before_seconds * fps)))
    end = min(
        total_frame_count - 1,
        candidate_frame_id + int(round(after_seconds * fps)),
    )
    return start, end


def coarse_frame_ids(
    start_frame: int,
    end_frame: int,
    *,
    stride: int,
    candidate_frame_id: int,
) -> tuple[int, ...]:
    sampled = set(range(start_frame, end_frame + 1, stride))
    sampled.add(candidate_frame_id)
    return tuple(sorted(sampled))


def fine_frame_ids(
    winners: Sequence[int],
    *,
    window_start: int,
    window_end: int,
    radius: int,
    stride: int,
) -> tuple[int, ...]:
    sampled: set[int] = set()
    for winner in sorted(set(winners)):
        fine_start = max(window_start, winner - radius)
        fine_end = min(window_end, winner + radius)
        sampled.update(range(fine_start, fine_end + 1, stride))
        sampled.add(winner)
    return tuple(sorted(sampled))


def fuse_local_frame_rankings(
    frame_ids: Sequence[int],
    image_embeddings: NDArray[np.float32],
    variants: Sequence[QueryVariant],
    text_embeddings: NDArray[np.float32],
    *,
    rrf_constant: float,
) -> tuple[LocalFrameFusion, ...]:
    per_frame = _rank_local_frames(
        frame_ids,
        image_embeddings,
        variants,
        text_embeddings,
    )
    return _fuse_ranked_frames(per_frame, rrf_constant=rrf_constant)


def _rank_local_frames(
    frame_ids: Sequence[int],
    image_embeddings: NDArray[np.float32],
    variants: Sequence[QueryVariant],
    text_embeddings: NDArray[np.float32],
) -> dict[int, list[dict[str, Any]]]:
    resolved_frames = tuple(frame_ids)
    if image_embeddings.ndim != 2 or image_embeddings.shape[0] != len(resolved_frames):
        raise ValueError("image embeddings/frame IDs row mismatch")
    if text_embeddings.ndim != 2 or text_embeddings.shape[0] != len(variants):
        raise ValueError("text embeddings/query variants row mismatch")
    if image_embeddings.shape[1] != text_embeddings.shape[1]:
        raise ValueError("image/text embedding dimension mismatch")
    image_matrix = np.asarray(image_embeddings, dtype=np.float32)
    text_matrix = np.asarray(text_embeddings, dtype=np.float32)
    if not np.isfinite(image_matrix).all() or not np.isfinite(text_matrix).all():
        raise ValueError("local refinement embeddings contain NaN or Infinity")
    image_norms = np.linalg.norm(image_matrix, axis=1, keepdims=True)
    text_norms = np.linalg.norm(text_matrix, axis=1, keepdims=True)
    if np.any(image_norms <= 0) or np.any(text_norms <= 0):
        raise ValueError("local refinement embeddings contain a zero norm")
    scores = np.asarray(
        (image_matrix / image_norms) @ (text_matrix / text_norms).T,
        dtype=np.float32,
    )
    per_frame: dict[int, list[dict[str, Any]]] = {frame_id: [] for frame_id in resolved_frames}
    for variant_index, variant in enumerate(variants):
        ordered_rows = sorted(
            range(len(resolved_frames)),
            key=lambda row: (-float(scores[row, variant_index]), resolved_frames[row]),
        )
        for rank, row in enumerate(ordered_rows, start=1):
            per_frame[resolved_frames[row]].append(
                {
                    "variant_id": variant.variant_id,
                    "variant_type": variant.variant_type.value,
                    "language": variant.language.value,
                    "weight": variant.weight,
                    "rank": rank,
                    "cosine_score": float(scores[row, variant_index]),
                }
            )
    return per_frame


def _fuse_ranked_frames(
    per_frame: Mapping[int, Sequence[Mapping[str, Any]]],
    *,
    rrf_constant: float,
) -> tuple[LocalFrameFusion, ...]:
    fused: list[LocalFrameFusion] = []
    for frame_id in sorted(per_frame):
        provenance = tuple(sorted(per_frame[frame_id], key=lambda item: item["variant_id"]))
        fusion_score = sum(
            float(item["weight"]) / (rrf_constant + int(item["rank"])) for item in provenance
        )
        fused.append(
            LocalFrameFusion(
                absolute_frame_id=frame_id,
                fusion_score=float(fusion_score),
                variant_hit_count=len(provenance),
                best_individual_rank=min(int(item["rank"]) for item in provenance),
                per_variant_provenance=provenance,
            )
        )
    return tuple(
        sorted(
            fused,
            key=lambda item: (
                -item.fusion_score,
                -item.variant_hit_count,
                item.best_individual_rank,
                item.absolute_frame_id,
            ),
        )
    )


def _empty_candidate_timings() -> dict[str, float]:
    return {
        "video_probe_seconds": 0.0,
        "video_open_seconds": 0.0,
        "coarse_decode_seconds": 0.0,
        "coarse_encode_seconds": 0.0,
        "coarse_score_seconds": 0.0,
        "coarse_fusion_seconds": 0.0,
        "fine_decode_seconds": 0.0,
        "fine_encode_seconds": 0.0,
        "fine_score_seconds": 0.0,
        "fine_fusion_seconds": 0.0,
        "candidate_total_seconds": 0.0,
        "coarse_sparse_request_count": 0,
        "coarse_sparse_success_count": 0,
        "coarse_sparse_fallback_count": 0,
        "coarse_requested_frame_count": 0,
        "coarse_decoded_frame_count": 0,
        "fine_requested_frame_count": 0,
        "fine_decoded_frame_count": 0,
    }


class ExactFrameRefiner:
    def __init__(
        self,
        *,
        raw_videos: RawVideoRegistry,
        decoder: VideoDecoder,
        encoder: RefinementEncoder,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.raw_videos = raw_videos
        self.decoder = decoder
        self.encoder = encoder
        self.clock = clock
        self._probe_cache: dict[str, VideoProbe] = {}

    def refine_query(
        self,
        query: RefinementQuery,
        config: RefinementConfig,
        *,
        precomputed_text_embeddings: NDArray[np.float32] | None = None,
        frame_embedding_cache: FrameEmbeddingCache | None = None,
    ) -> QueryRefinementOutcome:
        query_start = self.clock()
        text_start = self.clock()
        if precomputed_text_embeddings is not None:
            if not isinstance(precomputed_text_embeddings, np.ndarray):
                raise ValueError("precomputed_text_embeddings must be a numpy array")
            if precomputed_text_embeddings.dtype != np.float32:
                raise ValueError("precomputed_text_embeddings must be float32")
            if precomputed_text_embeddings.ndim != 2:
                raise ValueError("precomputed_text_embeddings must be exactly 2-dimensional")
            if precomputed_text_embeddings.shape[0] != len(query.variants):
                raise ValueError(
                    "precomputed_text_embeddings rows must match number of query variants"
                )
            if not np.isfinite(precomputed_text_embeddings).all():
                raise ValueError(
                    "precomputed_text_embeddings contains non-finite values (NaN/Infinity)"
                )
            if np.any(np.linalg.norm(precomputed_text_embeddings, axis=1) <= 0):
                raise ValueError("precomputed_text_embeddings contains a zero-norm row")
            text_embeddings = precomputed_text_embeddings
            text_encode_seconds = 0.0
        else:
            texts = [variant.text for variant in query.variants]
            text_embeddings = self.encoder.encode_texts(texts)
            text_encode_seconds = self.clock() - text_start
        refined: list[RefinedCandidate] = []
        warnings: list[str] = []
        aggregate = _empty_candidate_timings()
        for index, candidate in enumerate(query.candidates):
            if index >= config.top_candidates_to_refine:
                record = self._unchanged_candidate(
                    candidate,
                    status=RefinementStatus.NOT_REFINED,
                    warning="candidate outside top_candidates_to_refine",
                )
            else:
                record = self._process_candidate(
                    candidate,
                    query.variants,
                    text_embeddings,
                    config,
                    frame_embedding_cache=frame_embedding_cache,
                )
            refined.append(record)
            warnings.extend(record.warnings)
            for key in aggregate:
                aggregate[key] += float(record.timings.get(key, 0.0))

        result, dedup_warnings = self._build_final_result(
            query.query_id,
            tuple(refined),
            config.output_top_k,
        )
        warnings.extend(dedup_warnings)
        counts = {
            "decoded_frame_count": sum(item.decoded_frame_count for item in refined),
            "encoded_image_count": sum(item.encoded_image_count for item in refined),
            "refined_candidate_count": sum(
                item.status is RefinementStatus.REFINED for item in refined
            ),
            "kept_original_count": sum(
                item.status in {RefinementStatus.KEEP_ORIGINAL, RefinementStatus.NOT_REFINED}
                for item in refined
            ),
            "skipped_candidate_count": sum(
                item.status is RefinementStatus.SKIPPED for item in refined
            ),
            "failed_candidate_count": sum(item.failure_reason is not None for item in refined),
            "missing_raw_video_count": sum(
                any("raw video missing" in warning for warning in item.warnings) for item in refined
            ),
        }
        timings: dict[str, float | int] = {
            **aggregate,
            "text_encode_seconds": text_encode_seconds,
            "query_total_seconds": self.clock() - query_start,
            **counts,
        }
        return QueryRefinementOutcome(
            query_id=query.query_id,
            result=result,
            candidates=tuple(refined),
            warnings=tuple(sorted(set(warnings))),
            timings=timings,
        )

    def refine_selected_candidates(
        self,
        *,
        query_id: str,
        variants: tuple[QueryVariant, ...],
        candidates: tuple[Phase3Candidate, ...],
        config: RefinementConfig,
        precomputed_text_embeddings: NDArray[np.float32],
        frame_embedding_cache: FrameEmbeddingCache,
    ) -> SelectedRefinementOutcome:
        """Refine an explicit bounded candidate set with grouped video regions."""
        started = self.clock()
        if not query_id.strip() or not variants:
            raise ValueError("selected refinement requires query_id and variants")
        if any(candidate.query_id != query_id for candidate in candidates):
            raise ValueError("selected refinement candidate query_id mismatch")
        ranks = [candidate.rank for candidate in candidates]
        if len(ranks) != len(set(ranks)):
            raise ValueError("selected refinement candidate ranks must be unique")
        if precomputed_text_embeddings.dtype != np.float32:
            raise ValueError("precomputed_text_embeddings must be float32")
        if precomputed_text_embeddings.shape != (
            len(variants),
            self.encoder.dimension,
        ):
            raise ValueError("precomputed_text_embeddings shape mismatch")
        if not np.isfinite(precomputed_text_embeddings).all() or np.any(
            np.linalg.norm(precomputed_text_embeddings, axis=1) <= 0
        ):
            raise ValueError("precomputed_text_embeddings must be finite and non-zero")

        records: list[RefinedCandidate] = []
        warnings: list[str] = []
        metrics: dict[str, float | int] = {
            "q3_anchor_refinement_seconds": 0.0,
            "unique_q3_coarse_frame_count": 0,
            "unique_q3_fine_frame_count": 0,
            "frame_embedding_cache_hit_count": 0,
            "frame_embedding_cache_miss_count": 0,
            "merged_temporal_region_count": 0,
            "decoded_frame_count": 0,
            "encoded_image_count": 0,
        }
        by_video: dict[str, list[Phase3Candidate]] = {}
        for candidate in sorted(candidates, key=lambda item: item.rank):
            by_video.setdefault(candidate.video_id, []).append(candidate)

        for video_id in sorted(by_video):
            video_candidates = tuple(by_video[video_id])
            try:
                raw_record = self.raw_videos.get(video_id)
            except Exception as exc:
                for candidate in video_candidates:
                    records.append(self._handle_candidate_failure(candidate, exc, config))
                continue
            if raw_record.raw_video_path is None:
                records.extend(
                    self._handle_missing_raw(candidate, raw_record, config)
                    for candidate in video_candidates
                )
                continue
            try:
                probe = self._probe_cache.get(video_id)
                if probe is None:
                    probe = self.decoder.probe(raw_record)
                    self._probe_cache[video_id] = probe
                works = tuple(
                    self._selected_anchor_work(candidate, probe, config)
                    for candidate in video_candidates
                )
            except Exception as exc:
                records.extend(
                    self._handle_candidate_failure(
                        candidate,
                        exc,
                        config,
                        raw_video_path=raw_record.raw_video_path,
                    )
                    for candidate in video_candidates
                )
                continue

            regions = self._merge_selected_regions(
                works,
                max_span=config.max_decoded_frames_per_candidate,
            )
            metrics["merged_temporal_region_count"] += len(regions)
            for region in regions:
                try:
                    region_records, region_metrics = self._refine_selected_region(
                        region,
                        variants=variants,
                        text_embeddings=precomputed_text_embeddings,
                        config=config,
                        frame_embedding_cache=frame_embedding_cache,
                    )
                    records.extend(region_records)
                    for key, value in region_metrics.items():
                        metrics[key] += value
                except Exception as exc:
                    records.extend(
                        self._handle_candidate_failure(
                            work.candidate,
                            exc,
                            config,
                            raw_video_path=work.probe.raw_video_path,
                        )
                        for work in region
                    )

        ordered_records = tuple(sorted(records, key=lambda item: item.original_candidate_rank))
        for record in ordered_records:
            warnings.extend(record.warnings)
        metrics["q3_anchor_refinement_seconds"] = self.clock() - started
        return SelectedRefinementOutcome(
            candidates=ordered_records,
            warnings=tuple(sorted(set(warnings))),
            timings=metrics,
        )

    def refine_shared_candidate_groups(
        self,
        groups: tuple[SharedRefinementGroup, ...],
        *,
        config: RefinementConfig,
        frame_embedding_cache: FrameEmbeddingCache | None = None,
    ) -> SharedRefinementBatchOutcome:
        """Refine semantic groups while sharing exact raw frames within one query.

        Sampling, scoring, fusion, and candidate failure policies are identical to
        the legacy candidate refiner.  Only raw decode orchestration is shared.
        """
        started = self.clock()
        if not groups:
            raise ValueError("shared refinement requires at least one group")
        resolved_embedding_cache: FrameEmbeddingCache = (
            frame_embedding_cache if frame_embedding_cache is not None else {}
        )
        raw_frame_cache: dict[tuple[str, int], DecodedFrame] = {}
        records: list[list[RefinedCandidate]] = [[] for _ in groups]
        warnings: list[list[str]] = [[] for _ in groups]
        works: list[_SharedCandidateWork] = []
        candidate_node_count = 0
        raw_decode_before = 0

        for group_index, group in enumerate(groups):
            self._validate_shared_group(group)
            candidate_node_count += len(group.candidates)
            raw_decode_before += 2 * len(group.candidates)
            for candidate in group.candidates:
                try:
                    raw_record = self.raw_videos.get(candidate.video_id)
                except Exception as exc:
                    record = self._handle_candidate_failure(candidate, exc, config)
                    records[group_index].append(record)
                    warnings[group_index].extend(record.warnings)
                    continue
                if raw_record.raw_video_path is None:
                    record = self._handle_missing_raw(candidate, raw_record, config)
                    records[group_index].append(record)
                    warnings[group_index].extend(record.warnings)
                    continue
                try:
                    probe = self._probe_cache.get(candidate.video_id)
                    if probe is None:
                        probe = self.decoder.probe(raw_record)
                        self._probe_cache[candidate.video_id] = probe
                    selected = self._selected_anchor_work(candidate, probe, config)
                    works.append(
                        _SharedCandidateWork(
                            group_index=group_index,
                            candidate=candidate,
                            variants=group.variants,
                            text_embeddings=group.text_embeddings,
                            probe=probe,
                            window_start=selected.window_start,
                            window_end=selected.window_end,
                            coarse_ids=selected.coarse_ids,
                        )
                    )
                except Exception as exc:
                    record = self._handle_candidate_failure(
                        candidate,
                        exc,
                        config,
                        raw_video_path=raw_record.raw_video_path,
                    )
                    records[group_index].append(record)
                    warnings[group_index].extend(record.warnings)

        telemetry: dict[str, float | int | bool] = {
            "shared_raw_region_refinement_enabled": True,
            "refinement_candidate_node_count": candidate_node_count,
            "unique_video_count": len(
                {
                    candidate.video_id
                    for group in groups
                    for candidate in group.candidates
                }
            ),
            "coarse_requested_frame_count": sum(len(work.coarse_ids) for work in works),
            "coarse_unique_requested_frame_count": len(
                {(work.candidate.video_id, frame) for work in works for frame in work.coarse_ids}
            ),
            "fine_requested_frame_count": 0,
            "fine_unique_requested_frame_count": 0,
            "raw_decode_request_count_before_estimate": raw_decode_before,
            "raw_decode_request_count_actual": 0,
            "decoded_frame_count_actual": 0,
            "frame_cache_hit_count": 0,
            "frame_embedding_cache_hit_count": 0,
            "coalesced_region_count": 0,
        }

        fine_by_work: dict[tuple[int, int], tuple[int, ...]] = {}
        coarse_warnings: dict[tuple[int, int], tuple[str, ...]] = {}
        active: list[_SharedCandidateWork] = []
        for video_id in sorted({work.candidate.video_id for work in works}):
            video_works = tuple(work for work in works if work.candidate.video_id == video_id)
            requests = tuple((work, work.coarse_ids) for work in video_works)
            for region in self._merge_shared_stage_regions(
                requests,
                max_span=config.max_decoded_frames_per_candidate,
            ):
                telemetry["coalesced_region_count"] += 1
                try:
                    frames, region_warnings = self._decode_shared_stage_region(
                        region,
                        raw_frame_cache=raw_frame_cache,
                        config=config,
                        coarse=True,
                        telemetry=telemetry,
                    )
                    embedding_hits = sum(len(ids) for _, ids in region) - len(frames)
                    embedding_hits += sum(
                        (video_id, frame.absolute_frame_id) in resolved_embedding_cache
                        for frame in frames
                    )
                    telemetry["frame_embedding_cache_hit_count"] += embedding_hits
                    embeddings = _encode_frames_with_cache(
                        video_id=video_id,
                        frames=frames,
                        encoder=self.encoder,
                        batch_size=config.image_batch_size,
                        frame_embedding_cache=resolved_embedding_cache,
                    )
                    embedding_by_id = {
                        frame.absolute_frame_id: embeddings[index]
                        for index, frame in enumerate(frames)
                    }
                    region_fine: dict[tuple[int, int], tuple[int, ...]] = {}
                    region_coarse_warnings: dict[
                        tuple[int, int], tuple[str, ...]
                    ] = {}
                    region_active: list[_SharedCandidateWork] = []
                    for work, requested_ids in region:
                        candidate_embeddings = np.vstack(
                            [embedding_by_id[frame_id] for frame_id in requested_ids]
                        ).astype(np.float32)
                        ranked = fuse_local_frame_rankings(
                            requested_ids,
                            candidate_embeddings,
                            work.variants,
                            work.text_embeddings,
                            rrf_constant=config.rrf_constant,
                        )
                        winners = tuple(
                            item.absolute_frame_id for item in ranked[: config.coarse_top_n]
                        )
                        key = (work.group_index, work.candidate.rank)
                        region_fine[key] = fine_frame_ids(
                            winners,
                            window_start=work.window_start,
                            window_end=work.window_end,
                            radius=config.fine_radius_frames,
                            stride=config.fine_stride_frames,
                        )
                        region_coarse_warnings[key] = region_warnings
                        region_active.append(work)
                    fine_by_work.update(region_fine)
                    coarse_warnings.update(region_coarse_warnings)
                    active.extend(region_active)
                except Exception:
                    self._legacy_fallback_region(
                        region,
                        records=records,
                        warnings=warnings,
                        config=config,
                        frame_embedding_cache=resolved_embedding_cache,
                    )

        telemetry["fine_requested_frame_count"] = sum(
            len(fine_by_work[(work.group_index, work.candidate.rank)]) for work in active
        )
        telemetry["fine_unique_requested_frame_count"] = len(
            {
                (work.candidate.video_id, frame)
                for work in active
                for frame in fine_by_work[(work.group_index, work.candidate.rank)]
            }
        )

        for video_id in sorted({work.candidate.video_id for work in active}):
            video_works = tuple(work for work in active if work.candidate.video_id == video_id)
            requests = tuple(
                (work, fine_by_work[(work.group_index, work.candidate.rank)])
                for work in video_works
            )
            for region in self._merge_shared_stage_regions(
                requests,
                max_span=config.max_decoded_frames_per_candidate,
            ):
                telemetry["coalesced_region_count"] += 1
                try:
                    frames, region_warnings = self._decode_shared_stage_region(
                        region,
                        raw_frame_cache=raw_frame_cache,
                        config=config,
                        coarse=False,
                        telemetry=telemetry,
                    )
                    embedding_hits = sum(len(ids) for _, ids in region) - len(frames)
                    embedding_hits += sum(
                        (video_id, frame.absolute_frame_id) in resolved_embedding_cache
                        for frame in frames
                    )
                    telemetry["frame_embedding_cache_hit_count"] += embedding_hits
                    embeddings = _encode_frames_with_cache(
                        video_id=video_id,
                        frames=frames,
                        encoder=self.encoder,
                        batch_size=config.image_batch_size,
                        frame_embedding_cache=resolved_embedding_cache,
                    )
                    embedding_by_id = {
                        frame.absolute_frame_id: embeddings[index]
                        for index, frame in enumerate(frames)
                    }
                    region_records: list[tuple[int, RefinedCandidate]] = []
                    for work, requested_ids in region:
                        candidate_embeddings = np.vstack(
                            [embedding_by_id[frame_id] for frame_id in requested_ids]
                        ).astype(np.float32)
                        ranked = fuse_local_frame_rankings(
                            requested_ids,
                            candidate_embeddings,
                            work.variants,
                            work.text_embeddings,
                            rrf_constant=config.rrf_constant,
                        )
                        winner = ranked[0]
                        key = (work.group_index, work.candidate.rank)
                        record = RefinedCandidate(
                            query_id=work.candidate.query_id,
                            original_candidate_rank=work.candidate.rank,
                            video_id=video_id,
                            candidate_frame_id=work.candidate.frame_id,
                            refined_frame_id=winner.absolute_frame_id,
                            candidate_timestamp_seconds=(
                                work.candidate.frame_id / work.probe.fps
                            ),
                            refined_timestamp_seconds=(
                                winner.absolute_frame_id / work.probe.fps
                            ),
                            fps=work.probe.fps,
                            total_frame_count=work.probe.total_frame_count,
                            window_start_frame=work.window_start,
                            window_end_frame=work.window_end,
                            coarse_frame_ids=work.coarse_ids,
                            fine_frame_ids=requested_ids,
                            coarse_sample_count=len(work.coarse_ids),
                            fine_sample_count=len(requested_ids),
                            decoded_frame_count=len(work.coarse_ids) + len(requested_ids),
                            encoded_image_count=len(work.coarse_ids) + len(requested_ids),
                            refinement_fusion_score=winner.fusion_score,
                            variant_hit_count=winner.variant_hit_count,
                            best_individual_rank=winner.best_individual_rank,
                            per_variant_provenance=winner.per_variant_provenance,
                            decoder_backend=work.probe.decoder_backend,
                            raw_video_path=work.probe.raw_video_path,
                            status=RefinementStatus.REFINED,
                            warnings=tuple(
                                sorted(set((*coarse_warnings[key], *region_warnings)))
                            ),
                            failure_reason=None,
                            original_retrieval_provenance=(
                                work.candidate.retrieval_provenance
                            ),
                            timings=_empty_candidate_timings(),
                        )
                        region_records.append((work.group_index, record))
                    for group_index, record in region_records:
                        records[group_index].append(record)
                        warnings[group_index].extend(record.warnings)
                except Exception:
                    self._legacy_fallback_region(
                        region,
                        records=records,
                        warnings=warnings,
                        config=config,
                        frame_embedding_cache=resolved_embedding_cache,
                    )

        outcomes: list[SelectedRefinementOutcome] = []
        for index, group in enumerate(groups):
            ordered = tuple(sorted(records[index], key=lambda item: item.original_candidate_rank))
            if len(ordered) != len(group.candidates):
                raise RuntimeError("shared refinement did not resolve every candidate")
            outcomes.append(
                SelectedRefinementOutcome(
                    candidates=ordered,
                    warnings=tuple(sorted(set(warnings[index]))),
                    timings={},
                )
            )
        telemetry["shared_refinement_seconds"] = self.clock() - started
        raw_frame_cache.clear()
        return SharedRefinementBatchOutcome(groups=tuple(outcomes), timings=telemetry)

    def _validate_shared_group(self, group: SharedRefinementGroup) -> None:
        if not group.query_id.strip() or not group.variants or not group.candidates:
            raise ValueError("shared refinement group requires ID, variants, and candidates")
        if any(candidate.query_id != group.query_id for candidate in group.candidates):
            raise ValueError("shared refinement candidate query_id mismatch")
        ranks = [candidate.rank for candidate in group.candidates]
        if ranks != list(range(1, len(ranks) + 1)):
            raise ValueError("shared refinement candidate ranks must be contiguous from one")
        if group.text_embeddings.dtype != np.float32 or group.text_embeddings.shape != (
            len(group.variants),
            self.encoder.dimension,
        ):
            raise ValueError("shared refinement text embedding shape/dtype mismatch")
        if not np.isfinite(group.text_embeddings).all() or np.any(
            np.linalg.norm(group.text_embeddings, axis=1) <= 0
        ):
            raise ValueError("shared refinement text embeddings must be finite and non-zero")

    @staticmethod
    def _merge_shared_stage_regions(
        requests: tuple[tuple[_SharedCandidateWork, tuple[int, ...]], ...],
        *,
        max_span: int,
    ) -> tuple[tuple[tuple[_SharedCandidateWork, tuple[int, ...]], ...], ...]:
        regions: list[list[tuple[_SharedCandidateWork, tuple[int, ...]]]] = []
        for request in sorted(
            requests,
            key=lambda item: (
                item[1][0],
                item[1][-1],
                item[0].group_index,
                item[0].candidate.rank,
            ),
        ):
            if not regions:
                regions.append([request])
                continue
            current = regions[-1]
            current_start = min(item[1][0] for item in current)
            current_end = max(item[1][-1] for item in current)
            merged_end = max(current_end, request[1][-1])
            if request[1][0] <= current_end + 1 and merged_end - current_start + 1 <= max_span:
                current.append(request)
            else:
                regions.append([request])
        return tuple(tuple(region) for region in regions)

    def _decode_shared_stage_region(
        self,
        region: tuple[tuple[_SharedCandidateWork, tuple[int, ...]], ...],
        *,
        raw_frame_cache: dict[tuple[str, int], DecodedFrame],
        config: RefinementConfig,
        coarse: bool,
        telemetry: dict[str, float | int | bool],
    ) -> tuple[tuple[DecodedFrame, ...], tuple[str, ...]]:
        probe = region[0][0].probe
        video_id = region[0][0].candidate.video_id
        requested_ids = tuple(sorted({frame for _, ids in region for frame in ids}))
        logical_requested_count = sum(len(ids) for _, ids in region)
        missing_ids = tuple(
            frame for frame in requested_ids if (video_id, frame) not in raw_frame_cache
        )
        telemetry["frame_cache_hit_count"] += (
            logical_requested_count - len(missing_ids)
        )
        decode_warnings: tuple[str, ...] = ()
        if missing_ids:
            telemetry["raw_decode_request_count_actual"] += 1
            decode = self._decode_selected_frames(
                probe=probe,
                frame_ids=missing_ids,
                config=config,
                coarse=coarse,
            )
            telemetry["decoded_frame_count_actual"] += decode.decoded_frame_count
            decode_warnings = decode.warnings
            for frame in decode.frames:
                raw_frame_cache[(video_id, frame.absolute_frame_id)] = frame
        return (
            tuple(raw_frame_cache[(video_id, frame)] for frame in requested_ids),
            decode_warnings,
        )

    def _legacy_fallback_region(
        self,
        region: tuple[tuple[_SharedCandidateWork, tuple[int, ...]], ...],
        *,
        records: list[list[RefinedCandidate]],
        warnings: list[list[str]],
        config: RefinementConfig,
        frame_embedding_cache: FrameEmbeddingCache,
    ) -> None:
        for work, _ in region:
            record = self._process_candidate(
                work.candidate,
                work.variants,
                work.text_embeddings,
                config,
                frame_embedding_cache=frame_embedding_cache,
            )
            records[work.group_index].append(record)
            warnings[work.group_index].extend(record.warnings)

    @staticmethod
    def _selected_anchor_work(
        candidate: Phase3Candidate,
        probe: VideoProbe,
        config: RefinementConfig,
    ) -> _SelectedAnchorWork:
        start, end = build_frame_window(
            candidate.frame_id,
            fps=probe.fps,
            total_frame_count=probe.total_frame_count,
            before_seconds=config.window_before_seconds,
            after_seconds=config.window_after_seconds,
        )
        return _SelectedAnchorWork(
            candidate=candidate,
            probe=probe,
            window_start=start,
            window_end=end,
            coarse_ids=coarse_frame_ids(
                start,
                end,
                stride=config.coarse_stride_frames,
                candidate_frame_id=candidate.frame_id,
            ),
        )

    @staticmethod
    def _merge_selected_regions(
        works: tuple[_SelectedAnchorWork, ...],
        *,
        max_span: int,
    ) -> tuple[tuple[_SelectedAnchorWork, ...], ...]:
        regions: list[list[_SelectedAnchorWork]] = []
        for work in sorted(
            works,
            key=lambda item: (
                item.window_start,
                item.window_end,
                item.candidate.rank,
            ),
        ):
            if not regions:
                regions.append([work])
                continue
            current = regions[-1]
            merged_start = min(item.window_start for item in current)
            merged_end = max(item.window_end for item in current)
            candidate_end = max(merged_end, work.window_end)
            if (
                work.window_start <= merged_end + 1
                and candidate_end - merged_start + 1 <= max_span
            ):
                current.append(work)
            else:
                regions.append([work])
        return tuple(tuple(region) for region in regions)

    def _decode_selected_frames(
        self,
        *,
        probe: VideoProbe,
        frame_ids: tuple[int, ...],
        config: RefinementConfig,
        coarse: bool,
    ) -> Any:
        request = DecodeRequest(
            probe=probe,
            frame_ids=frame_ids,
            max_decoded_frames=config.max_decoded_frames_per_candidate,
        )
        if (
            coarse
            and config.coarse_decode_strategy == "sparse-verified"
            and hasattr(self.decoder, "decode_sparse_verified")
        ):
            return self.decoder.decode_sparse_verified(request, fallback_to_sequential=True)
        return self.decoder.decode(request)

    def _refine_selected_region(
        self,
        region: tuple[_SelectedAnchorWork, ...],
        *,
        variants: tuple[QueryVariant, ...],
        text_embeddings: NDArray[np.float32],
        config: RefinementConfig,
        frame_embedding_cache: FrameEmbeddingCache,
    ) -> tuple[tuple[RefinedCandidate, ...], dict[str, int]]:
        probe = region[0].probe
        video_id = region[0].candidate.video_id
        coarse_ids = tuple(sorted({frame for work in region for frame in work.coarse_ids}))
        coarse_decode = self._decode_selected_frames(
            probe=probe,
            frame_ids=coarse_ids,
            config=config,
            coarse=True,
        )
        coarse_hits = sum(
            (video_id, frame.absolute_frame_id) in frame_embedding_cache
            for frame in coarse_decode.frames
        )
        coarse_misses = len(coarse_decode.frames) - coarse_hits
        coarse_embeddings = _encode_frames_with_cache(
            video_id=video_id,
            frames=coarse_decode.frames,
            encoder=self.encoder,
            batch_size=config.image_batch_size,
            frame_embedding_cache=frame_embedding_cache,
        )
        coarse_by_id = {
            frame.absolute_frame_id: coarse_embeddings[index]
            for index, frame in enumerate(coarse_decode.frames)
        }

        fine_by_rank: dict[int, tuple[int, ...]] = {}
        for work in region:
            candidate_embeddings = np.vstack(
                [coarse_by_id[frame_id] for frame_id in work.coarse_ids]
            ).astype(np.float32)
            coarse_ranked = fuse_local_frame_rankings(
                work.coarse_ids,
                candidate_embeddings,
                variants,
                text_embeddings,
                rrf_constant=config.rrf_constant,
            )
            winners = tuple(
                item.absolute_frame_id for item in coarse_ranked[: config.coarse_top_n]
            )
            fine_by_rank[work.candidate.rank] = fine_frame_ids(
                winners,
                window_start=work.window_start,
                window_end=work.window_end,
                radius=config.fine_radius_frames,
                stride=config.fine_stride_frames,
            )

        fine_ids = tuple(sorted({frame for ids in fine_by_rank.values() for frame in ids}))
        fine_decode = self._decode_selected_frames(
            probe=probe,
            frame_ids=fine_ids,
            config=config,
            coarse=False,
        )
        fine_hits = sum(
            (video_id, frame.absolute_frame_id) in frame_embedding_cache
            for frame in fine_decode.frames
        )
        fine_misses = len(fine_decode.frames) - fine_hits
        fine_embeddings = _encode_frames_with_cache(
            video_id=video_id,
            frames=fine_decode.frames,
            encoder=self.encoder,
            batch_size=config.image_batch_size,
            frame_embedding_cache=frame_embedding_cache,
        )
        fine_by_id = {
            frame.absolute_frame_id: fine_embeddings[index]
            for index, frame in enumerate(fine_decode.frames)
        }

        records: list[RefinedCandidate] = []
        decode_warnings = tuple(
            sorted(set((*coarse_decode.warnings, *fine_decode.warnings)))
        )
        for work in region:
            candidate_fine_ids = fine_by_rank[work.candidate.rank]
            candidate_fine_embeddings = np.vstack(
                [fine_by_id[frame_id] for frame_id in candidate_fine_ids]
            ).astype(np.float32)
            fine_ranked = fuse_local_frame_rankings(
                candidate_fine_ids,
                candidate_fine_embeddings,
                variants,
                text_embeddings,
                rrf_constant=config.rrf_constant,
            )
            winner = fine_ranked[0]
            records.append(
                RefinedCandidate(
                    query_id=work.candidate.query_id,
                    original_candidate_rank=work.candidate.rank,
                    video_id=video_id,
                    candidate_frame_id=work.candidate.frame_id,
                    refined_frame_id=winner.absolute_frame_id,
                    candidate_timestamp_seconds=work.candidate.frame_id / probe.fps,
                    refined_timestamp_seconds=winner.absolute_frame_id / probe.fps,
                    fps=probe.fps,
                    total_frame_count=probe.total_frame_count,
                    window_start_frame=work.window_start,
                    window_end_frame=work.window_end,
                    coarse_frame_ids=work.coarse_ids,
                    fine_frame_ids=candidate_fine_ids,
                    coarse_sample_count=len(work.coarse_ids),
                    fine_sample_count=len(candidate_fine_ids),
                    decoded_frame_count=len(work.coarse_ids) + len(candidate_fine_ids),
                    encoded_image_count=len(work.coarse_ids) + len(candidate_fine_ids),
                    refinement_fusion_score=winner.fusion_score,
                    variant_hit_count=winner.variant_hit_count,
                    best_individual_rank=winner.best_individual_rank,
                    per_variant_provenance=winner.per_variant_provenance,
                    decoder_backend=probe.decoder_backend,
                    raw_video_path=probe.raw_video_path,
                    status=RefinementStatus.REFINED,
                    warnings=decode_warnings,
                    failure_reason=None,
                    original_retrieval_provenance=work.candidate.retrieval_provenance,
                    timings=_empty_candidate_timings(),
                )
            )
        return tuple(records), {
            "unique_q3_coarse_frame_count": len(coarse_ids),
            "unique_q3_fine_frame_count": len(fine_ids),
            "frame_embedding_cache_hit_count": coarse_hits + fine_hits,
            "frame_embedding_cache_miss_count": coarse_misses + fine_misses,
            "decoded_frame_count": (
                coarse_decode.decoded_frame_count + fine_decode.decoded_frame_count
            ),
            "encoded_image_count": coarse_misses + fine_misses,
        }

    def _process_candidate(
        self,
        candidate: Phase3Candidate,
        variants: tuple[QueryVariant, ...],
        text_embeddings: NDArray[np.float32],
        config: RefinementConfig,
        *,
        frame_embedding_cache: FrameEmbeddingCache | None = None,
    ) -> RefinedCandidate:
        record = self.raw_videos.get(candidate.video_id)
        if record.raw_video_path is None:
            return self._handle_missing_raw(candidate, record, config)
        try:
            return self._refine_candidate(
                candidate,
                record,
                variants,
                text_embeddings,
                config,
                frame_embedding_cache=frame_embedding_cache,
            )
        except Exception as exc:
            return self._handle_candidate_failure(
                candidate, exc, config, raw_video_path=record.raw_video_path
            )

    def _refine_candidate(
        self,
        candidate: Phase3Candidate,
        raw_record: RawVideoRecord,
        variants: tuple[QueryVariant, ...],
        text_embeddings: NDArray[np.float32],
        config: RefinementConfig,
        *,
        frame_embedding_cache: FrameEmbeddingCache | None = None,
    ) -> RefinedCandidate:
        candidate_start = self.clock()
        timings = _empty_candidate_timings()
        probe = self._probe_cache.get(candidate.video_id)
        if probe is None:
            probe_start = self.clock()
            probe = self.decoder.probe(raw_record)
            timings["video_probe_seconds"] = self.clock() - probe_start
            self._probe_cache[candidate.video_id] = probe
        start_frame, end_frame = build_frame_window(
            candidate.frame_id,
            fps=probe.fps,
            total_frame_count=probe.total_frame_count,
            before_seconds=config.window_before_seconds,
            after_seconds=config.window_after_seconds,
        )
        coarse_ids = coarse_frame_ids(
            start_frame,
            end_frame,
            stride=config.coarse_stride_frames,
            candidate_frame_id=candidate.frame_id,
        )
        timings["coarse_requested_frame_count"] += len(coarse_ids)
        coarse_req = DecodeRequest(
            probe=probe,
            frame_ids=coarse_ids,
            max_decoded_frames=config.max_decoded_frames_per_candidate,
        )
        if config.coarse_decode_strategy == "sparse-verified" and hasattr(
            self.decoder, "decode_sparse_verified"
        ):
            timings["coarse_sparse_request_count"] += 1
            coarse_decode = self.decoder.decode_sparse_verified(
                coarse_req, fallback_to_sequential=True
            )
            if coarse_decode.decode_strategy == "sparse_verified":
                timings["coarse_sparse_success_count"] += 1
            elif coarse_decode.decode_strategy == "sparse_verified_fallback_sequential":
                timings["coarse_sparse_fallback_count"] += 1
        else:
            coarse_decode = self.decoder.decode(coarse_req)

        timings["coarse_decoded_frame_count"] += coarse_decode.decoded_frame_count
        timings["video_open_seconds"] += coarse_decode.video_open_seconds
        timings["coarse_decode_seconds"] = coarse_decode.decode_seconds
        coarse_encode_start = self.clock()
        coarse_embeddings = _encode_frames_with_cache(
            video_id=candidate.video_id,
            frames=coarse_decode.frames,
            encoder=self.encoder,
            batch_size=config.image_batch_size,
            frame_embedding_cache=frame_embedding_cache,
        )
        timings["coarse_encode_seconds"] = self.clock() - coarse_encode_start
        coarse_score_start = self.clock()
        coarse_per_frame = _rank_local_frames(
            [frame.absolute_frame_id for frame in coarse_decode.frames],
            coarse_embeddings,
            variants,
            text_embeddings,
        )
        timings["coarse_score_seconds"] = self.clock() - coarse_score_start
        coarse_fusion_start = self.clock()
        coarse_ranked = _fuse_ranked_frames(coarse_per_frame, rrf_constant=config.rrf_constant)
        timings["coarse_fusion_seconds"] = self.clock() - coarse_fusion_start
        winners = tuple(item.absolute_frame_id for item in coarse_ranked[: config.coarse_top_n])
        fine_ids = fine_frame_ids(
            winners,
            window_start=start_frame,
            window_end=end_frame,
            radius=config.fine_radius_frames,
            stride=config.fine_stride_frames,
        )
        timings["fine_requested_frame_count"] += len(fine_ids)
        fine_decode = self.decoder.decode(
            DecodeRequest(
                probe=probe,
                frame_ids=fine_ids,
                max_decoded_frames=config.max_decoded_frames_per_candidate,
            )
        )
        timings["fine_decoded_frame_count"] += fine_decode.decoded_frame_count
        timings["video_open_seconds"] += fine_decode.video_open_seconds
        timings["fine_decode_seconds"] = fine_decode.decode_seconds
        fine_encode_start = self.clock()
        fine_embeddings = _encode_frames_with_cache(
            video_id=candidate.video_id,
            frames=fine_decode.frames,
            encoder=self.encoder,
            batch_size=config.image_batch_size,
            frame_embedding_cache=frame_embedding_cache,
        )
        timings["fine_encode_seconds"] = self.clock() - fine_encode_start
        fine_score_start = self.clock()
        fine_per_frame = _rank_local_frames(
            [frame.absolute_frame_id for frame in fine_decode.frames],
            fine_embeddings,
            variants,
            text_embeddings,
        )
        timings["fine_score_seconds"] = self.clock() - fine_score_start
        fine_fusion_start = self.clock()
        fine_ranked = _fuse_ranked_frames(fine_per_frame, rrf_constant=config.rrf_constant)
        timings["fine_fusion_seconds"] = self.clock() - fine_fusion_start
        winner = fine_ranked[0]
        timings["candidate_total_seconds"] = self.clock() - candidate_start
        decoded_count = coarse_decode.decoded_frame_count + fine_decode.decoded_frame_count
        encoded_count = len(coarse_decode.frames) + len(fine_decode.frames)
        return RefinedCandidate(
            query_id=candidate.query_id,
            original_candidate_rank=candidate.rank,
            video_id=candidate.video_id,
            candidate_frame_id=candidate.frame_id,
            refined_frame_id=winner.absolute_frame_id,
            candidate_timestamp_seconds=candidate.frame_id / probe.fps,
            refined_timestamp_seconds=winner.absolute_frame_id / probe.fps,
            fps=probe.fps,
            total_frame_count=probe.total_frame_count,
            window_start_frame=start_frame,
            window_end_frame=end_frame,
            coarse_frame_ids=coarse_ids,
            fine_frame_ids=fine_ids,
            coarse_sample_count=len(coarse_ids),
            fine_sample_count=len(fine_ids),
            decoded_frame_count=decoded_count,
            encoded_image_count=encoded_count,
            refinement_fusion_score=winner.fusion_score,
            variant_hit_count=winner.variant_hit_count,
            best_individual_rank=winner.best_individual_rank,
            per_variant_provenance=winner.per_variant_provenance,
            decoder_backend=probe.decoder_backend,
            raw_video_path=probe.raw_video_path,
            status=RefinementStatus.REFINED,
            warnings=tuple(sorted(set((*coarse_decode.warnings, *fine_decode.warnings)))),
            failure_reason=None,
            original_retrieval_provenance=candidate.retrieval_provenance,
            timings=timings,
        )

    def _handle_missing_raw(
        self,
        candidate: Phase3Candidate,
        record: RawVideoRecord,
        config: RefinementConfig,
    ) -> RefinedCandidate:
        reason = f"raw video missing for {candidate.video_id}"
        if config.missing_raw_video_policy is MissingRawVideoPolicy.FAIL_QUERY:
            raise QueryRefinementError(reason)
        if config.missing_raw_video_policy is MissingRawVideoPolicy.SKIP_CANDIDATE:
            return self._unchanged_candidate(
                candidate,
                status=RefinementStatus.SKIPPED,
                warning=reason,
                refined_frame_id=None,
                raw_video_path=record.raw_video_path,
            )
        return self._unchanged_candidate(
            candidate,
            status=RefinementStatus.KEEP_ORIGINAL,
            warning=reason,
            raw_video_path=record.raw_video_path,
        )

    def _handle_candidate_failure(
        self,
        candidate: Phase3Candidate,
        exc: Exception,
        config: RefinementConfig,
        *,
        raw_video_path: Any = None,
    ) -> RefinedCandidate:
        reason = f"{type(exc).__name__}: {exc}"
        if config.candidate_failure_policy is CandidateFailurePolicy.FAIL_QUERY:
            raise QueryRefinementError(reason) from exc
        if config.candidate_failure_policy is CandidateFailurePolicy.SKIP_CANDIDATE:
            return self._unchanged_candidate(
                candidate,
                status=RefinementStatus.SKIPPED,
                warning=reason,
                refined_frame_id=None,
                failure_reason=reason,
                raw_video_path=raw_video_path,
            )
        return self._unchanged_candidate(
            candidate,
            status=RefinementStatus.KEEP_ORIGINAL,
            warning=reason,
            failure_reason=reason,
            raw_video_path=raw_video_path,
        )

    @staticmethod
    def _unchanged_candidate(
        candidate: Phase3Candidate,
        *,
        status: RefinementStatus,
        warning: str,
        refined_frame_id: int | None = None,
        raw_video_path: Any = None,
        failure_reason: str | None = None,
    ) -> RefinedCandidate:
        final_frame = (
            candidate.frame_id
            if refined_frame_id is None and status is not RefinementStatus.SKIPPED
            else refined_frame_id
        )
        return RefinedCandidate(
            query_id=candidate.query_id,
            original_candidate_rank=candidate.rank,
            video_id=candidate.video_id,
            candidate_frame_id=candidate.frame_id,
            refined_frame_id=final_frame,
            candidate_timestamp_seconds=None,
            refined_timestamp_seconds=None,
            fps=None,
            total_frame_count=None,
            window_start_frame=None,
            window_end_frame=None,
            coarse_frame_ids=(),
            fine_frame_ids=(),
            coarse_sample_count=0,
            fine_sample_count=0,
            decoded_frame_count=0,
            encoded_image_count=0,
            refinement_fusion_score=None,
            variant_hit_count=0,
            best_individual_rank=None,
            per_variant_provenance=(),
            decoder_backend=None,
            raw_video_path=raw_video_path,
            status=status,
            warnings=(warning,),
            failure_reason=failure_reason,
            original_retrieval_provenance=candidate.retrieval_provenance,
            timings=_empty_candidate_timings(),
        )

    @staticmethod
    def _build_final_result(
        query_id: str,
        candidates: tuple[RefinedCandidate, ...],
        output_top_k: int,
    ) -> tuple[KISResult, tuple[str, ...]]:
        selected: list[RefinedCandidate] = []
        seen: set[tuple[str, int]] = set()
        warnings: list[str] = []
        for candidate in sorted(candidates, key=lambda item: item.original_candidate_rank):
            if candidate.refined_frame_id is None or candidate.status is RefinementStatus.SKIPPED:
                continue
            identity = (candidate.video_id, candidate.refined_frame_id)
            if identity in seen:
                warnings.append(
                    "refined duplicate removed: "
                    f"{candidate.video_id}/{candidate.refined_frame_id} "
                    f"from original rank {candidate.original_candidate_rank}"
                )
                continue
            seen.add(identity)
            selected.append(candidate)
            if len(selected) >= output_top_k:
                break
        if len(selected) < min(output_top_k, len(candidates)):
            warnings.append(
                f"refined output contains {len(selected)} records after policies/deduplication"
            )
        output = tuple(
            CandidateFrame(
                video_id=candidate.video_id,
                frame_id=int(candidate.refined_frame_id),
                clip_row=int(candidate.original_retrieval_provenance.get("clip_row_diagnostic", 0)),
                keyframe_order=int(
                    candidate.original_retrieval_provenance.get("keyframe_order_diagnostic", 0)
                ),
                score=float(candidate.original_retrieval_provenance.get("fusion_score", 0.0)),
                rank=rank,
                source="raw_video_exact_frame_refinement",
                diagnostic_metadata={
                    "original_candidate_rank": candidate.original_candidate_rank,
                    "candidate_frame_id": candidate.candidate_frame_id,
                    "refinement_status": candidate.status.value,
                },
            )
            for rank, candidate in enumerate(selected, start=1)
        )
        return KISResult(query_id=query_id, ranked_candidates=output), tuple(warnings)
