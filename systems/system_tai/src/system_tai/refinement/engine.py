"""Deterministic bounded coarse-to-fine exact-frame refinement engine."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
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
    DecodeRequest,
    RawVideoError,
    RawVideoRecord,
    RawVideoRegistry,
    VideoDecoder,
    VideoProbe,
)
from system_tai.retrieval.multi_query import QueryVariant


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
    ) -> QueryRefinementOutcome:
        query_start = self.clock()
        text_start = self.clock()
        text_embeddings = self.encoder.encode_texts([variant.text for variant in query.variants])
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

    def _process_candidate(
        self,
        candidate: Phase3Candidate,
        variants: tuple[QueryVariant, ...],
        text_embeddings: NDArray[np.float32],
        config: RefinementConfig,
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
        coarse_decode = self.decoder.decode(
            DecodeRequest(
                probe=probe,
                frame_ids=coarse_ids,
                max_decoded_frames=config.max_decoded_frames_per_candidate,
            )
        )
        timings["video_open_seconds"] += coarse_decode.video_open_seconds
        timings["coarse_decode_seconds"] = coarse_decode.decode_seconds
        coarse_encode_start = self.clock()
        coarse_embeddings = self.encoder.encode_images(
            [frame.image for frame in coarse_decode.frames],
            batch_size=config.image_batch_size,
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
        fine_decode = self.decoder.decode(
            DecodeRequest(
                probe=probe,
                frame_ids=fine_ids,
                max_decoded_frames=config.max_decoded_frames_per_candidate,
            )
        )
        timings["video_open_seconds"] += fine_decode.video_open_seconds
        timings["fine_decode_seconds"] = fine_decode.decode_seconds
        fine_encode_start = self.clock()
        fine_embeddings = self.encoder.encode_images(
            [frame.image for frame in fine_decode.frames],
            batch_size=config.image_batch_size,
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
