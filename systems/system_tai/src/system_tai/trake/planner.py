import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from system_tai.preliminary.schemas import TRAKEPrediction

from .models import TRAKEEventCandidate, TRAKEQuery


@dataclass(frozen=True, slots=True)
class TRAKEPathState:
    video_id: str
    frame_ids: tuple[int, ...]
    candidate_ranks: tuple[int, ...]
    path_score: float


def plan_trake_paths(
    query: TRAKEQuery,
    event_candidates: Sequence[Sequence[TRAKEEventCandidate]],
    beam_width: int = 100,
    output_top_k: int = 100,
    rrf_constant: float = 60.0,
) -> tuple[tuple[TRAKEPrediction, ...], dict[str, Any]]:
    """Deterministic bounded temporal beam search / DP for TRAKE.

    Produces rank-scored temporal paths f_0 <= f_1 <= ... <= f_(N-1)
    where all frames come from candidate pools of the SAME video.
    """
    # 1. Parameter validation
    if type(beam_width) is not int or beam_width < 1 or beam_width > 1000:
        raise ValueError("beam_width must be an integer in range 1..1000")
    if type(output_top_k) is not int or output_top_k < 1 or output_top_k > 100:
        raise ValueError("output_top_k must be an integer in range 1..100")
    if (
        type(rrf_constant) is bool
        or not isinstance(rrf_constant, (int, float))
        or not math.isfinite(float(rrf_constant))
        or float(rrf_constant) <= 0
    ):
        raise ValueError("rrf_constant must be a finite positive number > 0")

    num_events = len(query.events)
    if len(event_candidates) != num_events:
        raise ValueError(
            f"Candidate pool count ({len(event_candidates)}) != event count ({num_events})"
        )

    event_candidate_counts = [len(pool) for pool in event_candidates]
    diagnostics: dict[str, Any] = {
        "query_id": query.query_id,
        "event_candidate_counts": event_candidate_counts,
        "candidate_video_count": 0,
        "complete_video_count": 0,
        "complete_path_count_before_global_topk": 0,
        "beam_width": beam_width,
        "output_top_k": output_top_k,
        "rrf_constant": float(rrf_constant),
        "zero_output_reason": None,
    }

    # 2. Check for empty pools
    if any(cnt == 0 for cnt in event_candidate_counts):
        diagnostics["zero_output_reason"] = "empty_candidate_pool"
        return (), diagnostics

    # 3. Validate & deduplicate input candidate pools
    deduped_pools: list[list[TRAKEEventCandidate]] = []
    all_candidate_videos: set[str] = set()

    for idx, pool in enumerate(event_candidates):
        seen_ranks: set[int] = set()
        by_semantic_key: dict[tuple[str, int], list[TRAKEEventCandidate]] = {}

        for cand in pool:
            if cand.query_id != query.query_id:
                raise ValueError(
                    f"Candidate query_id mismatch at event {idx}: expected {query.query_id}"
                )
            if cand.event_index != idx:
                raise ValueError(
                    f"Candidate event_index mismatch at pool {idx}: got {cand.event_index}"
                )
            if cand.rank in seen_ranks:
                raise ValueError(
                    f"Duplicate candidate rank {cand.rank} in event pool {idx}"
                )
            seen_ranks.add(cand.rank)
            all_candidate_videos.add(cand.video_id)

            key = (cand.video_id, cand.frame_id)
            if key not in by_semantic_key:
                by_semantic_key[key] = []
            by_semantic_key[key].append(cand)

        # Resolve duplicate semantic candidates (same event_index, video_id, frame_id)
        resolved_pool: list[TRAKEEventCandidate] = []
        for key, cands in by_semantic_key.items():
            if len(cands) == 1:
                resolved_pool.append(cands[0])
            else:
                ranks = [c.rank for c in cands]
                if len(set(ranks)) < len(ranks):
                    raise ValueError(f"Conflicting identical rank duplicate for key {key}")
                best_cand = min(cands, key=lambda c: c.rank)
                resolved_pool.append(best_cand)

        resolved_pool.sort(key=lambda c: (c.rank, c.video_id, c.frame_id))
        deduped_pools.append(resolved_pool)

    diagnostics["candidate_video_count"] = len(all_candidate_videos)

    # 4. Group candidates by video_id
    video_map: dict[str, dict[int, list[TRAKEEventCandidate]]] = {}
    for idx, pool in enumerate(deduped_pools):
        for cand in pool:
            if cand.video_id not in video_map:
                video_map[cand.video_id] = {e: [] for e in range(num_events)}
            video_map[cand.video_id][idx].append(cand)

    complete_videos = [
        vid
        for vid, event_dict in video_map.items()
        if all(len(event_dict[e]) > 0 for e in range(num_events))
    ]
    complete_videos.sort()
    diagnostics["complete_video_count"] = len(complete_videos)

    if not complete_videos:
        diagnostics["zero_output_reason"] = "no_complete_video"
        return (), diagnostics

    # 5. Bounded Temporal Beam Search per video
    all_final_states: list[TRAKEPathState] = []

    for vid in complete_videos:
        event_dict = video_map[vid]

        # Event 0 states initialization
        current_states: list[TRAKEPathState] = []
        for c0 in event_dict[0]:
            score = 1.0 / (rrf_constant + c0.rank)
            current_states.append(
                TRAKEPathState(
                    video_id=vid,
                    frame_ids=(c0.frame_id,),
                    candidate_ranks=(c0.rank,),
                    path_score=score,
                )
            )

        current_states = _prune_and_dedup_states(current_states, beam_width)

        video_failed = False
        for e in range(1, num_events):
            next_states: list[TRAKEPathState] = []
            e_cands = event_dict[e]

            for state in current_states:
                last_frame = state.frame_ids[-1]
                for ce in e_cands:
                    if ce.frame_id >= last_frame:  # NON-DECREASING TEMPORAL ORDER
                        score_delta = 1.0 / (rrf_constant + ce.rank)
                        new_state = TRAKEPathState(
                            video_id=vid,
                            frame_ids=state.frame_ids + (ce.frame_id,),
                            candidate_ranks=state.candidate_ranks + (ce.rank,),
                            path_score=state.path_score + score_delta,
                        )
                        next_states.append(new_state)

            if not next_states:
                video_failed = True
                break

            current_states = _prune_and_dedup_states(next_states, beam_width)

        if not video_failed and current_states:
            all_final_states.extend(current_states)

    if not all_final_states:
        diagnostics["zero_output_reason"] = "no_temporal_valid_path"
        return (), diagnostics

    # 6. Global Deduplication and Ranking across all videos
    global_by_key: dict[tuple[str, tuple[int, ...]], TRAKEPathState] = {}
    for state in all_final_states:
        key = (state.video_id, state.frame_ids)
        if key not in global_by_key:
            global_by_key[key] = state
        else:
            existing = global_by_key[key]
            if state.path_score > existing.path_score:
                global_by_key[key] = state
            elif math.isclose(state.path_score, existing.path_score, rel_tol=1e-9, abs_tol=1e-9):
                if state.candidate_ranks < existing.candidate_ranks:
                    global_by_key[key] = state

    deduped_global_states = list(global_by_key.values())
    diagnostics["complete_path_count_before_global_topk"] = len(deduped_global_states)

    # Sort globally:
    # 1. path_score descending
    # 2. video_id ascending
    # 3. frame_ids lexicographically ascending
    # 4. candidate_ranks lexicographically ascending
    deduped_global_states.sort(
        key=lambda s: (-s.path_score, s.video_id, s.frame_ids, s.candidate_ranks)
    )

    top_states = deduped_global_states[:output_top_k]

    predictions = tuple(
        TRAKEPrediction(
            query_id=query.query_id,
            rank=rank_idx + 1,
            video_id=st.video_id,
            frame_ids=st.frame_ids,
        )
        for rank_idx, st in enumerate(top_states)
    )

    return predictions, diagnostics


def _prune_and_dedup_states(
    states: list[TRAKEPathState], beam_width: int
) -> list[TRAKEPathState]:
    """Deduplicate states with identical frame_ids and retain top beam_width states."""
    by_frame_ids: dict[tuple[int, ...], TRAKEPathState] = {}
    for st in states:
        key = st.frame_ids
        if key not in by_frame_ids:
            by_frame_ids[key] = st
        else:
            existing = by_frame_ids[key]
            if st.path_score > existing.path_score:
                by_frame_ids[key] = st
            elif math.isclose(st.path_score, existing.path_score, rel_tol=1e-9, abs_tol=1e-9):
                if st.candidate_ranks < existing.candidate_ranks:
                    by_frame_ids[key] = st

    unique_states = list(by_frame_ids.values())
    unique_states.sort(key=lambda s: (-s.path_score, s.frame_ids, s.candidate_ranks))
    return unique_states[:beam_width]
