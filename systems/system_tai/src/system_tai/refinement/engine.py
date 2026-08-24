"""Deterministic bounded coarse-to-fine exact-frame refinement engine."""

from __future__ import annotations

import math
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
    SelectedVideoTimelineScoutConfig,
    SelectedVideoVisualVerifierConfig,
    VisualVerifierFailurePolicy,
)
from system_tai.refinement.video import (
    DecodedFrame,
    DecodeRequest,
    RawVideoError,
    RawVideoRecord,
    RawVideoRegistry,
    SparseDecodeRequest,
    VideoDecoder,
    VideoProbe,
)
from system_tai.refinement.visual_verifier import (
    StructuredVisualVerifier,
    VisualVerificationError,
    VisualVerificationInput,
    VisualVerificationResult,
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
class TimelineScoutOutcome:
    """Automatically discovered raw-video anchors and bounded audit telemetry."""

    candidates: tuple[Phase3Candidate, ...]
    warnings: tuple[str, ...]
    trace: Mapping[str, Any]
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


def timeline_sparse_frame_ids(
    probe: VideoProbe,
    *,
    sample_stride_seconds: float,
    max_samples: int,
) -> tuple[int, ...]:
    """Uniformly cover the probed full timeline, including both endpoints."""
    if not math.isfinite(sample_stride_seconds) or sample_stride_seconds <= 0:
        raise ValueError("sample_stride_seconds must be finite and positive")
    if type(max_samples) is not int or max_samples < 2:
        raise ValueError("max_samples must be at least two")
    if probe.total_frame_count == 1:
        return (0,)
    stride = max(1, int(round(sample_stride_seconds * probe.fps)))
    sampled = tuple(range(0, probe.total_frame_count, stride))
    if sampled[-1] != probe.total_frame_count - 1:
        sampled = (*sampled, probe.total_frame_count - 1)
    if len(sampled) <= max_samples:
        return sampled
    last = probe.total_frame_count - 1
    # Integer interpolation is deterministic, bounded, and guarantees tail coverage.
    return tuple((index * last) // (max_samples - 1) for index in range(max_samples))


def select_timeline_regions(
    ranked: Sequence[LocalFrameFusion],
    *,
    fps: float,
    max_regions: int,
    minimum_gap_seconds: float,
) -> tuple[LocalFrameFusion, ...]:
    """Temporal NMS over automatically scored timeline samples."""
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    if type(max_regions) is not int or max_regions <= 0:
        raise ValueError("max_regions must be a positive integer")
    if not math.isfinite(minimum_gap_seconds) or minimum_gap_seconds <= 0:
        raise ValueError("minimum_gap_seconds must be finite and positive")
    minimum_gap_frames = max(1, int(round(minimum_gap_seconds * fps)))
    selected: list[LocalFrameFusion] = []
    for candidate in ranked:
        if any(
            abs(candidate.absolute_frame_id - prior.absolute_frame_id)
            < minimum_gap_frames
            for prior in selected
        ):
            continue
        selected.append(candidate)
        if len(selected) >= max_regions:
            break
    return tuple(selected)


def build_visual_verification_shortlist(
    ranked: Sequence[LocalFrameFusion],
    *,
    total_frame_count: int,
    shortlist_size: int,
    coverage_bins: int,
) -> tuple[LocalFrameFusion, ...]:
    """Combine global CLIP leaders with deterministic full-timeline coverage.

    Coverage candidates are the sampled frames nearest each temporal-bin midpoint,
    independent of CLIP score. This prevents a semantically subtle late scene from
    being excluded solely because broad exercise frames dominate CLIP ranking.
    """
    if total_frame_count <= 0:
        raise ValueError("total_frame_count must be positive")
    if shortlist_size <= 0:
        raise ValueError("shortlist_size must be positive")
    if not 1 <= coverage_bins <= shortlist_size:
        raise ValueError("coverage_bins must be in [1, shortlist_size]")
    if not ranked:
        return ()
    rank_index = {item.absolute_frame_id: index for index, item in enumerate(ranked)}
    chosen: dict[int, LocalFrameFusion] = {}
    global_budget = max(0, shortlist_size - coverage_bins)
    for item in ranked[:global_budget]:
        chosen[item.absolute_frame_id] = item
    for bin_index in range(coverage_bins):
        start = (bin_index * total_frame_count) // coverage_bins
        end = ((bin_index + 1) * total_frame_count) // coverage_bins
        bin_candidates = [
            item
            for item in ranked
            if start <= item.absolute_frame_id < end
        ]
        if bin_candidates:
            midpoint = (start + end - 1) / 2.0
            representative = min(
                bin_candidates,
                key=lambda item: (
                    abs(item.absolute_frame_id - midpoint),
                    item.absolute_frame_id,
                ),
            )
            chosen[representative.absolute_frame_id] = representative
    for item in ranked:
        if len(chosen) >= shortlist_size:
            break
        chosen[item.absolute_frame_id] = item
    return tuple(
        sorted(chosen.values(), key=lambda item: rank_index[item.absolute_frame_id])
    )


def rank_visually_verified_timeline_frames(
    shortlist: Sequence[LocalFrameFusion],
    results: Sequence[VisualVerificationResult],
) -> tuple[LocalFrameFusion, ...]:
    """Rank calibrated VLM results by conjunction, then CLIP fallbacks.

    VLM and raw CLIP score scales are never added or compared numerically. A partial
    result set is valid: discriminative verified frames form the leading partition while
    failed candidates and non-discriminative positive plateaus retain their original CLIP
    order in the fallback partition. Within the verified partition, the weakest visible
    predicate precedes aggregate coverage and match score so one missing count, attribute,
    action, or relation cannot be hidden by a strong broad-scene match.
    """
    by_frame = {item.absolute_frame_id: item for item in shortlist}
    result_by_frame = {item.absolute_frame_id: item for item in results}
    if len(result_by_frame) != len(results):
        raise ValueError("visual verifier returned duplicate frame identities")
    if not set(result_by_frame).issubset(by_frame):
        raise ValueError("visual verifier result identity mismatch")
    if not result_by_frame:
        raise ValueError("visual verifier returned no successful results")
    plateau_frame_ids = non_discriminative_visual_plateau_frame_ids(results)
    trusted_result_by_frame = {
        frame_id: result
        for frame_id, result in result_by_frame.items()
        if frame_id not in plateau_frame_ids and result.eligible_for_promotion
    }
    verified = sorted(
        (
            item
            for item in shortlist
            if item.absolute_frame_id in trusted_result_by_frame
        ),
        key=lambda item: (
            -int(
                trusted_result_by_frame[
                    item.absolute_frame_id
                ].all_visible_requirements_satisfied
            ),
            -trusted_result_by_frame[item.absolute_frame_id].predicate_bottleneck_score,
            -trusted_result_by_frame[item.absolute_frame_id].requirement_coverage,
            -trusted_result_by_frame[item.absolute_frame_id].match_score,
            -item.fusion_score,
            item.absolute_frame_id,
        ),
    )
    fallbacks = [
        item
        for item in shortlist
        if item.absolute_frame_id not in trusted_result_by_frame
    ]
    ordered = [*verified, *fallbacks]
    return tuple(
        LocalFrameFusion(
            absolute_frame_id=item.absolute_frame_id,
            fusion_score=(
                trusted_result_by_frame[item.absolute_frame_id].match_score
                if item.absolute_frame_id in trusted_result_by_frame
                else item.fusion_score
            ),
            variant_hit_count=item.variant_hit_count,
            best_individual_rank=item.best_individual_rank,
            per_variant_provenance=(
                *item.per_variant_provenance,
                (
                    {
                        "visual_verification": dict(
                            result_by_frame[item.absolute_frame_id].to_trace()
                        )
                    }
                    if item.absolute_frame_id in trusted_result_by_frame
                    else (
                        {
                            "visual_verification": {
                                **dict(
                                    result_by_frame[item.absolute_frame_id].to_trace()
                                ),
                                "calibration_status": (
                                    "ABSTAIN_NON_DISCRIMINATIVE_POSITIVE_PLATEAU"
                                ),
                            }
                        }
                        if item.absolute_frame_id in plateau_frame_ids
                        else {
                            "visual_verification": {
                                "status": "CANDIDATE_FALLBACK_CLIP"
                            }
                        }
                    )
                ),
            ),
        )
        for item in ordered
    )


def non_discriminative_visual_plateau_frame_ids(
    results: Sequence[VisualVerificationResult],
    *,
    minimum_plateau_size: int = 3,
) -> frozenset[int]:
    """Find repeated positive VLM templates that cannot distinguish frames.

    Plateau candidates remain in output but their VLM score abstains from promotion,
    leaving them in canonical CLIP order. Repeated negative results are harmless and are
    not calibrated by this policy.
    """

    if type(minimum_plateau_size) is not int or minimum_plateau_size < 2:
        raise ValueError("minimum_plateau_size must be an integer of at least 2")
    frames_by_signature: dict[tuple[Any, ...], list[int]] = {}
    for result in results:
        if (
            not result.all_visible_requirements_satisfied
            or not result.eligible_for_promotion
        ):
            continue
        frames_by_signature.setdefault(result.semantic_signature, []).append(
            result.absolute_frame_id
        )
    return frozenset(
        frame_id
        for frame_ids in frames_by_signature.values()
        if len(frame_ids) >= minimum_plateau_size
        for frame_id in frame_ids
    )


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
        visual_verifier: StructuredVisualVerifier | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.raw_videos = raw_videos
        self.decoder = decoder
        self.encoder = encoder
        self.visual_verifier = visual_verifier
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

    def scout_selected_video_timelines(
        self,
        *,
        query_id: str,
        query_vi: str,
        query_en: str,
        variants: tuple[QueryVariant, ...],
        ranked_video_ids: tuple[str, ...],
        rank_slots: tuple[Phase3Candidate, ...],
        config: SelectedVideoTimelineScoutConfig,
        visual_verifier_config: SelectedVideoVisualVerifierConfig,
        refinement_config: RefinementConfig,
        precomputed_text_embeddings: NDArray[np.float32],
        frame_embedding_cache: FrameEmbeddingCache,
    ) -> TimelineScoutOutcome:
        """Find semantic regions across complete system-nominated video timelines.

        This stage consumes only the retrieval-produced video order and probed raw
        video metadata. It does not accept target frames, timestamps, or labels.
        Returned anchors inherit existing same-video rank slots so downstream
        refinement cannot change the canonical video/rank sequence.
        """
        started = self.clock()
        if not config.enabled:
            return TimelineScoutOutcome(
                candidates=(),
                warnings=(),
                trace={"enabled": False},
                timings={
                    "timeline_scout_seconds": 0.0,
                    "timeline_video_count": 0,
                    "timeline_sample_count": 0,
                    "timeline_decoded_frame_count": 0,
                    "timeline_encoded_image_count": 0,
                    "timeline_region_count": 0,
                },
            )
        if not query_id.strip() or not variants:
            raise ValueError("timeline scout requires query_id and variants")
        if not query_vi.strip() or not query_en.strip():
            raise ValueError("timeline scout requires Vietnamese and English query text")
        if visual_verifier_config.enabled and self.visual_verifier is None:
            raise ValueError("visual verifier is enabled but no verifier was initialized")
        if precomputed_text_embeddings.shape != (
            len(variants),
            self.encoder.dimension,
        ):
            raise ValueError("timeline scout text embedding shape mismatch")
        if not np.isfinite(precomputed_text_embeddings).all() or np.any(
            np.linalg.norm(precomputed_text_embeddings, axis=1) <= 0
        ):
            raise ValueError("timeline scout text embeddings must be finite and non-zero")

        unique_ranked_videos = tuple(dict.fromkeys(ranked_video_ids))
        slots_by_video: dict[str, list[Phase3Candidate]] = {}
        for slot in sorted(rank_slots, key=lambda item: item.rank):
            if slot.query_id != query_id:
                raise ValueError("timeline scout rank-slot query_id mismatch")
            slots_by_video.setdefault(slot.video_id, []).append(slot)
        selected_videos = tuple(
            video_id
            for video_id in unique_ranked_videos
            if video_id in slots_by_video
        )[: config.max_videos]

        anchors: list[Phase3Candidate] = []
        warnings: list[str] = []
        video_traces: list[dict[str, Any]] = []
        sample_count = 0
        decoded_count = 0
        encoded_count = 0
        visual_verified_count = 0
        visual_verifier_seconds = 0.0
        for nomination_rank, video_id in enumerate(selected_videos, start=1):
            try:
                raw_record = self.raw_videos.get(video_id)
                if raw_record.raw_video_path is None:
                    warnings.append(f"timeline scout raw video missing for {video_id}")
                    continue
                probe = self._probe_cache.get(video_id)
                if probe is None:
                    probe = self.decoder.probe(raw_record)
                    self._probe_cache[video_id] = probe
                frame_ids = timeline_sparse_frame_ids(
                    probe,
                    sample_stride_seconds=config.sample_stride_seconds,
                    max_samples=config.max_samples_per_video,
                )
                if not hasattr(self.decoder, "decode_sparse_verified"):
                    raise RawVideoError(
                        "timeline scout requires verified sparse absolute-frame decoding"
                    )
                request = SparseDecodeRequest(
                    probe=probe,
                    frame_ids=frame_ids,
                    max_decoded_frames=config.max_samples_per_video,
                )
                decoded = self.decoder.decode_sparse_verified(
                    request,
                    fallback_to_sequential=False,
                )
                cache_hits = sum(
                    (video_id, frame.absolute_frame_id) in frame_embedding_cache
                    for frame in decoded.frames
                )
                embeddings = _encode_frames_with_cache(
                    video_id=video_id,
                    frames=decoded.frames,
                    encoder=self.encoder,
                    batch_size=refinement_config.image_batch_size,
                    frame_embedding_cache=frame_embedding_cache,
                )
                fused = fuse_local_frame_rankings(
                    frame_ids,
                    embeddings,
                    variants,
                    precomputed_text_embeddings,
                    rrf_constant=refinement_config.rrf_constant,
                )
                verification_trace: dict[str, Any] = {"enabled": False}
                ranked_for_selection = fused
                if visual_verifier_config.enabled:
                    execution_trace = dict(visual_verifier_config.execution_trace())
                    if visual_verifier_config.cpu_fast_profile_applied:
                        warnings.append(
                            "visual verifier CPU-fast profile applied for "
                            f"{video_id}: {execution_trace['effective']}"
                        )
                    shortlist = build_visual_verification_shortlist(
                        fused,
                        total_frame_count=probe.total_frame_count,
                        shortlist_size=(
                            visual_verifier_config.effective_shortlist_per_video
                        ),
                        coverage_bins=visual_verifier_config.effective_coverage_bins,
                    )
                    frame_index = {
                        frame.absolute_frame_id: index
                        for index, frame in enumerate(decoded.frames)
                    }
                    verification_inputs: list[VisualVerificationInput] = []
                    for item in shortlist:
                        center_index = frame_index[item.absolute_frame_id]
                        start_index = max(
                            0,
                            center_index
                            - visual_verifier_config.effective_neighbor_sample_radius,
                        )
                        end_index = min(
                            len(decoded.frames),
                            center_index
                            + visual_verifier_config.effective_neighbor_sample_radius
                            + 1,
                        )
                        center = decoded.frames[center_index]
                        verification_inputs.append(
                            VisualVerificationInput(
                                video_id=video_id,
                                absolute_frame_id=center.absolute_frame_id,
                                timestamp_seconds=center.timestamp_seconds,
                                images=tuple(
                                    frame.image
                                    for frame in decoded.frames[start_index:end_index]
                                ),
                            )
                        )
                    verify_started = self.clock()
                    try:
                        assert self.visual_verifier is not None
                        try:
                            verification_results = self.visual_verifier.verify(
                                query_vi=query_vi,
                                query_en=query_en,
                                candidates=verification_inputs,
                            )
                        finally:
                            visual_verifier_seconds += (
                                self.clock() - verify_started
                            )
                        candidate_failures = tuple(
                            getattr(self.visual_verifier, "last_failures", ())
                        )
                        recovered_retries = tuple(
                            getattr(
                                self.visual_verifier,
                                "last_recovered_retries",
                                (),
                            )
                        )
                        if candidate_failures and (
                            visual_verifier_config.failure_policy
                            is VisualVerifierFailurePolicy.FAIL_QUERY
                        ):
                            raise VisualVerificationError(
                                "visual verifier candidate failures: "
                                + "; ".join(
                                    f"{item.video_id}/{item.absolute_frame_id}: "
                                    f"{item.retry_error}"
                                    for item in candidate_failures
                                )
                            )
                        if not verification_results:
                            raise VisualVerificationError(
                                "visual verifier returned no successful candidates"
                            )
                        visual_verified_count += len(verification_results)
                        plateau_frame_ids = (
                            non_discriminative_visual_plateau_frame_ids(
                                verification_results
                            )
                        )
                        predicate_contract = tuple(
                            getattr(
                                self.visual_verifier,
                                "last_predicate_contract",
                                (),
                            )
                        )
                        ranked_for_selection = rank_visually_verified_timeline_frames(
                            shortlist,
                            verification_results,
                        )
                        failure_traces = [
                            dict(item.to_trace()) for item in candidate_failures
                        ]
                        if candidate_failures:
                            warnings.append(
                                "visual verifier candidate-local fallback to CLIP for "
                                f"{video_id}: "
                                + ", ".join(
                                    str(item.absolute_frame_id)
                                    for item in candidate_failures
                                )
                            )
                        verification_trace = {
                            "enabled": True,
                            "status": (
                                "PARTIAL_SUCCESS"
                                if candidate_failures
                                else "SUCCESS"
                            ),
                            "provider": dict(self.visual_verifier.identifiers),
                            "execution": execution_trace,
                            "shortlist_frame_ids": [
                                item.absolute_frame_id for item in shortlist
                            ],
                            "results": [
                                dict(item.to_trace()) for item in verification_results
                            ],
                            "predicate_contract": [
                                dict(item.to_prompt()) for item in predicate_contract
                            ],
                            "strictly_promotable_candidate_count": sum(
                                item.eligible_for_promotion
                                for item in verification_results
                            ),
                            "successful_candidate_count": len(
                                verification_results
                            ),
                            "failed_candidate_count": len(candidate_failures),
                            "failures": failure_traces,
                            "recovered_retries": [
                                dict(item) for item in recovered_retries
                            ],
                            "calibration": {
                                "policy": (
                                    "fixed-contract-fail-closed-and-positive-plateau-abstains"
                                ),
                                "minimum_plateau_size": 3,
                                "abstained_frame_ids": sorted(plateau_frame_ids),
                                "abstained_candidate_count": len(
                                    plateau_frame_ids
                                ),
                            },
                        }
                    except Exception as exc:
                        if (
                            visual_verifier_config.failure_policy
                            is VisualVerifierFailurePolicy.FAIL_QUERY
                        ):
                            raise VisualVerificationError(
                                f"visual verifier failed for {video_id}: {exc}"
                            ) from exc
                        warnings.append(
                            f"visual verifier fallback to CLIP for {video_id}: {exc}"
                        )
                        verification_trace = {
                            "enabled": True,
                            "status": "FALLBACK_CLIP",
                            "execution": execution_trace,
                            "failure_reason": str(exc),
                        }
                regions = select_timeline_regions(
                    ranked_for_selection,
                    fps=probe.fps,
                    max_regions=config.max_regions_per_video,
                    minimum_gap_seconds=config.minimum_region_gap_seconds,
                )
                available_slots = slots_by_video[video_id]
                assigned = tuple(zip(available_slots, regions, strict=False))
                for slot, region in assigned:
                    anchors.append(
                        Phase3Candidate(
                            query_id=query_id,
                            rank=slot.rank,
                            video_id=video_id,
                            frame_id=region.absolute_frame_id,
                            retrieval_score=region.fusion_score,
                            retrieval_provenance={
                                **dict(slot.retrieval_provenance),
                                "timeline_scout": True,
                                "timeline_nomination_rank": nomination_rank,
                                "timeline_original_slot_frame_id": slot.frame_id,
                                "timeline_sample_count": len(frame_ids),
                                "timeline_region_fusion_score": region.fusion_score,
                                "timeline_variant_hit_count": region.variant_hit_count,
                                "timeline_best_individual_rank": (
                                    region.best_individual_rank
                                ),
                                "timeline_per_variant_provenance": (
                                    region.per_variant_provenance
                                ),
                                "timeline_visual_verifier_enabled": (
                                    visual_verifier_config.enabled
                                ),
                            },
                        )
                    )
                sample_count += len(frame_ids)
                decoded_count += decoded.decoded_frame_count
                encoded_count += len(decoded.frames) - cache_hits
                video_traces.append(
                    {
                        "video_id": video_id,
                        "nomination_rank": nomination_rank,
                        "fps": probe.fps,
                        "total_frame_count": probe.total_frame_count,
                        "sample_count": len(frame_ids),
                        "first_sample_frame_id": frame_ids[0],
                        "last_sample_frame_id": frame_ids[-1],
                        "selected_region_frame_ids": [
                            item.absolute_frame_id for item in regions
                        ],
                        "assigned_rank_slots": [slot.rank for slot, _ in assigned],
                        "decoder_backend": decoded.decoder_backend,
                        "visual_verification": verification_trace,
                        "warnings": decoded.warnings,
                    }
                )
            except VisualVerificationError:
                raise
            except Exception as exc:
                warnings.append(f"timeline scout failed for {video_id}: {exc}")

        ordered_anchors = tuple(sorted(anchors, key=lambda item: item.rank))
        timings: dict[str, float | int] = {
            "timeline_scout_seconds": self.clock() - started,
            "timeline_video_count": len(video_traces),
            "timeline_sample_count": sample_count,
            "timeline_decoded_frame_count": decoded_count,
            "timeline_encoded_image_count": encoded_count,
            "timeline_region_count": len(ordered_anchors),
            "timeline_visual_verified_candidate_count": visual_verified_count,
            "timeline_visual_verifier_seconds": visual_verifier_seconds,
        }
        return TimelineScoutOutcome(
            candidates=ordered_anchors,
            warnings=tuple(sorted(set(warnings))),
            trace={
                "enabled": True,
                "selection_source": "system_video_first_nomination",
                "hard_coded_target": False,
                "visual_verifier_enabled": visual_verifier_config.enabled,
                "ranked_video_ids_considered": list(selected_videos),
                "videos": video_traces,
                "selected_anchors": [
                    {
                        "rank": item.rank,
                        "video_id": item.video_id,
                        "frame_id": item.frame_id,
                    }
                    for item in ordered_anchors
                ],
                "warnings": tuple(sorted(set(warnings))),
            },
            timings=timings,
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
