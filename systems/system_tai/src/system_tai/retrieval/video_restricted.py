"""Video-conditioned exact keyframe search and rank-slot-preserving diversity."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from system_tai.common.schemas import CandidateFrame, KISResult
from system_tai.features.btc_clip_store import FeatureStoreRegistry, LoadedVideoFeatureStore

VIDEO_CONDITIONED_KEYFRAME_DIVERSITY = "VIDEO_CONDITIONED_KEYFRAME_DIVERSITY"


@dataclass(frozen=True, slots=True)
class VideoConditionedKeyframeConfig:
    """Bounded opt-in configuration for Q3.1 keyframe diversity."""

    enabled: bool = False
    selected_video_global_rank_cap: int = 50
    max_selected_videos: int = 50
    max_anchors_per_video: int = 3
    minimum_anchor_gap_seconds: float = 5.0
    preserve_first_video_occurrence: bool = True

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise ValueError("enabled must be a boolean")
        if not 1 <= self.selected_video_global_rank_cap <= 100:
            raise ValueError("selected_video_global_rank_cap must be in [1, 100]")
        if not 1 <= self.max_selected_videos <= 100:
            raise ValueError("max_selected_videos must be in [1, 100]")
        if not 1 <= self.max_anchors_per_video <= 100:
            raise ValueError("max_anchors_per_video must be in [1, 100]")
        if (
            not math.isfinite(self.minimum_anchor_gap_seconds)
            or self.minimum_anchor_gap_seconds < 0
        ):
            raise ValueError("minimum_anchor_gap_seconds must be finite and non-negative")
        if self.preserve_first_video_occurrence is not True:
            raise ValueError("Q3.1 requires preserve_first_video_occurrence=true")


@dataclass(frozen=True, slots=True)
class RestrictedKeyframeCandidate:
    video_id: str
    frame_id: int
    clip_row: int
    keyframe_order: int
    pts_time: float
    cosine_score: float
    restricted_rank: int


@dataclass(frozen=True, slots=True)
class VideoConditioningOutcome:
    result: KISResult
    trace: Mapping[str, Any]
    selected_video_count: int
    restricted_keyframe_rows_scored: int
    anchor_count: int
    substitution_count: int
    selected_videos_with_no_replacement_capacity: int
    total_same_video_replacement_slots: int
    restricted_search_seconds: float
    conditioning_seconds: float


def _normalize_query(
    query_vector: Sequence[float] | NDArray[np.number], *, expected_dimension: int
) -> NDArray[np.float32]:
    vector = np.asarray(query_vector, dtype=np.float32)
    if vector.shape != (expected_dimension,):
        raise ValueError(
            "query vector shape mismatch: "
            f"observed={vector.shape}, expected=({expected_dimension},)"
        )
    if not np.isfinite(vector).all():
        raise ValueError("query vector contains NaN or Infinity")
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError("query vector must have a finite non-zero norm")
    return np.asarray(vector / norm, dtype=np.float32)


def _restricted_sort_key(
    candidate: RestrictedKeyframeCandidate,
) -> tuple[float, int, int]:
    return (-candidate.cosine_score, candidate.frame_id, candidate.clip_row)


class VideoConditionedKeyframeDiversity:
    """Search selected video stores and diversify only their existing rank slots."""

    def __init__(
        self,
        registry: FeatureStoreRegistry,
        *,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.registry = registry
        self.clock = clock

    def condition(
        self,
        *,
        global_result: KISResult,
        query_vector: Sequence[float] | NDArray[np.number],
        config: VideoConditionedKeyframeConfig,
    ) -> VideoConditioningOutcome:
        started = self.clock()
        if not config.enabled:
            return VideoConditioningOutcome(
                result=global_result,
                trace={
                    "policy": VIDEO_CONDITIONED_KEYFRAME_DIVERSITY,
                    "enabled": False,
                },
                selected_video_count=0,
                restricted_keyframe_rows_scored=0,
                anchor_count=0,
                substitution_count=0,
                selected_videos_with_no_replacement_capacity=0,
                total_same_video_replacement_slots=0,
                restricted_search_seconds=0.0,
                conditioning_seconds=0.0,
            )

        query_unit = _normalize_query(
            query_vector, expected_dimension=self.registry.embedding_dimension
        )
        global_candidates = tuple(global_result.ranked_candidates)
        identities = [(item.video_id, item.frame_id) for item in global_candidates]
        if len(set(identities)) != len(identities):
            raise ValueError("global KIS result contains duplicate video/frame identities")

        slots_by_video: dict[str, list[int]] = {}
        first_rank_by_video: dict[str, int] = {}
        for index, candidate in enumerate(global_candidates):
            slots_by_video.setdefault(candidate.video_id, []).append(index)
            first_rank_by_video.setdefault(candidate.video_id, candidate.rank)
        selected_video_ids = tuple(
            video_id
            for video_id, _rank in sorted(
                (
                    (video_id, rank)
                    for video_id, rank in first_rank_by_video.items()
                    if rank <= config.selected_video_global_rank_cap
                ),
                key=lambda item: (item[1], item[0]),
            )[: config.max_selected_videos]
        )

        original_identity_set = set(identities)
        output = list(global_candidates)
        per_video_trace: list[dict[str, Any]] = []
        restricted_rows_scored = 0
        restricted_seconds = 0.0
        anchor_count = 0
        substitution_count = 0
        no_capacity_count = 0
        replacement_slot_count = 0

        for video_id in selected_video_ids:
            store = self.registry.get(video_id)
            search_started = self.clock()
            ranking = self._score_store(store, query_unit)
            restricted_seconds += self.clock() - search_started
            restricted_rows_scored += store.descriptor.row_count

            slot_indexes = slots_by_video[video_id]
            first_candidate = global_candidates[slot_indexes[0]]
            replacement_slots = slot_indexes[1:]
            replacement_slot_count += len(replacement_slots)
            if not replacement_slots:
                no_capacity_count += 1

            anchors = self._select_anchors(
                ranking,
                config=config,
                excluded_identities=original_identity_set,
            )
            anchor_count += len(anchors)

            substitutions: list[dict[str, Any]] = []
            next_anchor = 0
            for slot_index in replacement_slots:
                if next_anchor >= len(anchors):
                    break
                anchor = anchors[next_anchor]
                next_anchor += 1
                original = global_candidates[slot_index]
                metadata = dict(original.diagnostic_metadata or {})
                metadata.update(
                    {
                        "q3_policy": VIDEO_CONDITIONED_KEYFRAME_DIVERSITY,
                        "score_semantics": "ORIGINAL_GLOBAL_SLOT_SCORE",
                        "original_global_rank": original.rank,
                        "original_frame_id": original.frame_id,
                        "restricted_cosine_score": anchor.cosine_score,
                        "restricted_rank": anchor.restricted_rank,
                        "anchor_pts_time": anchor.pts_time,
                        "anchor_index": next_anchor,
                    }
                )
                output[slot_index] = CandidateFrame(
                    video_id=video_id,
                    frame_id=anchor.frame_id,
                    clip_row=anchor.clip_row,
                    keyframe_order=anchor.keyframe_order,
                    score=original.score,
                    rank=original.rank,
                    source="video_conditioned_keyframe_diversity",
                    diagnostic_metadata=metadata,
                )
                substitution_count += 1
                substitutions.append(
                    {
                        "rank": original.rank,
                        "original_frame_id": original.frame_id,
                        "replacement_frame_id": anchor.frame_id,
                        "restricted_rank": anchor.restricted_rank,
                    }
                )

            per_video_trace.append(
                {
                    "video_id": video_id,
                    "first_occurrence_rank": first_candidate.rank,
                    "first_occurrence_frame_id": first_candidate.frame_id,
                    "restricted_keyframe_rows_scored": store.descriptor.row_count,
                    "output_slot_count": len(slot_indexes),
                    "available_replacement_slot_count": len(replacement_slots),
                    "anchors": [self._anchor_payload(anchor) for anchor in anchors],
                    "substitutions": substitutions,
                    "uninserted_anchor_count": max(0, len(anchors) - len(substitutions)),
                }
            )

        conditioned = KISResult(
            query_id=global_result.query_id,
            ranked_candidates=tuple(output),
        )
        conditioned_identities = [
            (item.video_id, item.frame_id) for item in conditioned.ranked_candidates
        ]
        if len(set(conditioned_identities)) != len(conditioned_identities):
            raise ValueError("video conditioning introduced duplicate video/frame identities")
        if [item.rank for item in conditioned.ranked_candidates] != [
            item.rank for item in global_candidates
        ]:
            raise AssertionError("video conditioning changed the rank sequence")
        if [item.video_id for item in conditioned.ranked_candidates] != [
            item.video_id for item in global_candidates
        ]:
            raise AssertionError("video conditioning changed the video sequence")
        for video_id in selected_video_ids:
            first_index = slots_by_video[video_id][0]
            if conditioned.ranked_candidates[first_index] != global_candidates[first_index]:
                raise AssertionError("video conditioning changed a first video occurrence")

        conditioning_seconds = self.clock() - started
        trace = {
            "policy": VIDEO_CONDITIONED_KEYFRAME_DIVERSITY,
            "enabled": True,
            "config": {
                "selected_video_global_rank_cap": config.selected_video_global_rank_cap,
                "max_selected_videos": config.max_selected_videos,
                "max_anchors_per_video": config.max_anchors_per_video,
                "minimum_anchor_gap_seconds": config.minimum_anchor_gap_seconds,
                "preserve_first_video_occurrence": config.preserve_first_video_occurrence,
            },
            "selected_video_ids": list(selected_video_ids),
            "selected_video_count": len(selected_video_ids),
            "restricted_keyframe_rows_scored": restricted_rows_scored,
            "anchor_count": anchor_count,
            "substitution_count": substitution_count,
            "selected_videos_with_no_replacement_capacity": no_capacity_count,
            "total_same_video_replacement_slots": replacement_slot_count,
            "restricted_search_seconds": restricted_seconds,
            "conditioning_seconds": conditioning_seconds,
            "videos": per_video_trace,
        }
        return VideoConditioningOutcome(
            result=conditioned,
            trace=trace,
            selected_video_count=len(selected_video_ids),
            restricted_keyframe_rows_scored=restricted_rows_scored,
            anchor_count=anchor_count,
            substitution_count=substitution_count,
            selected_videos_with_no_replacement_capacity=no_capacity_count,
            total_same_video_replacement_slots=replacement_slot_count,
            restricted_search_seconds=restricted_seconds,
            conditioning_seconds=conditioning_seconds,
        )

    @staticmethod
    def _score_store(
        store: LoadedVideoFeatureStore,
        query_unit: NDArray[np.float32],
    ) -> tuple[RestrictedKeyframeCandidate, ...]:
        matrix = np.asarray(store.matrix, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] != len(store.mappings):
            raise ValueError(f"invalid feature/mapping shape for {store.descriptor.video_id}")
        if not np.isfinite(matrix).all():
            raise ValueError(f"non-finite feature row in {store.descriptor.video_id}")
        norms = np.linalg.norm(matrix, axis=1)
        if np.any(~np.isfinite(norms)) or np.any(norms <= 0):
            raise ValueError(f"non-finite or zero-norm feature row in {store.descriptor.video_id}")
        scores = (matrix @ query_unit) / norms
        if not np.isfinite(scores).all():
            raise ValueError("restricted cosine computation produced NaN or Infinity")

        best_by_frame: dict[int, RestrictedKeyframeCandidate] = {}
        for clip_row, mapping in enumerate(store.mappings):
            candidate = RestrictedKeyframeCandidate(
                video_id=store.descriptor.video_id,
                frame_id=mapping.frame_id,
                clip_row=clip_row,
                keyframe_order=mapping.keyframe_order,
                pts_time=mapping.pts_time,
                cosine_score=float(scores[clip_row]),
                restricted_rank=0,
            )
            existing = best_by_frame.get(candidate.frame_id)
            if existing is None or _restricted_sort_key(candidate) < _restricted_sort_key(
                existing
            ):
                best_by_frame[candidate.frame_id] = candidate
        ordered = sorted(best_by_frame.values(), key=_restricted_sort_key)
        return tuple(
            RestrictedKeyframeCandidate(
                video_id=item.video_id,
                frame_id=item.frame_id,
                clip_row=item.clip_row,
                keyframe_order=item.keyframe_order,
                pts_time=item.pts_time,
                cosine_score=item.cosine_score,
                restricted_rank=rank,
            )
            for rank, item in enumerate(ordered, start=1)
        )

    @staticmethod
    def _select_anchors(
        ranking: Sequence[RestrictedKeyframeCandidate],
        *,
        config: VideoConditionedKeyframeConfig,
        excluded_identities: set[tuple[str, int]],
    ) -> tuple[RestrictedKeyframeCandidate, ...]:
        selected: list[RestrictedKeyframeCandidate] = []
        accepted_times: list[float] = []
        for candidate in ranking:
            if (candidate.video_id, candidate.frame_id) in excluded_identities:
                continue
            if all(
                abs(candidate.pts_time - pts_time) >= config.minimum_anchor_gap_seconds
                for pts_time in accepted_times
            ):
                selected.append(candidate)
                accepted_times.append(candidate.pts_time)
                if len(selected) >= config.max_anchors_per_video:
                    break
        return tuple(selected)

    @staticmethod
    def _anchor_payload(candidate: RestrictedKeyframeCandidate) -> dict[str, Any]:
        return {
            "video_id": candidate.video_id,
            "frame_id": candidate.frame_id,
            "clip_row": candidate.clip_row,
            "keyframe_order": candidate.keyframe_order,
            "pts_time": candidate.pts_time,
            "restricted_cosine_score": candidate.cosine_score,
            "restricted_rank": candidate.restricted_rank,
        }
