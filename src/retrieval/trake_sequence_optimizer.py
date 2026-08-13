"""
TRAKE Sequence-Aware Temporal Optimizer
========================================
Dynamic Programming / Viterbi-style optimizer for finding the globally
optimal temporal sequence across N ordered events within a single video.

Key invariant:
    timestamp(f_1) < timestamp(f_2) < ... < timestamp(f_N)

Does NOT mutate candidate lists. Returns a new optimal sequence.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("aic.trake_optimizer")


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class Candidate:
    """Normalized retrieval candidate for one event."""
    event_index: int
    video_id: str
    frame_id: int
    timestamp_seconds: float
    fused_score: float
    visual_score: float = 0.0
    ocr_score: float = 0.0
    asr_score: float = 0.0
    object_score: float = 0.0
    keyframe_name: str = ""
    keyframe_url: str = ""
    timestamp_text: str = ""


@dataclass
class SequenceResult:
    """Result of sequence optimization for one video."""
    video_id: str
    sequence: List[Optional[Candidate]]  # Index = event_index, None = unresolved
    sequence_score: float = 0.0
    coverage: float = 0.0
    coverage_count: int = 0
    total_events: int = 0
    retrieval_score: float = 0.0
    temporal_valid: bool = False
    missing_events: List[int] = field(default_factory=list)
    confidence_per_event: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Confidence Scoring
# ---------------------------------------------------------------------------

def compute_candidate_confidence(
    candidate: Candidate,
    all_candidates_for_event: List[Candidate],
) -> str:
    """
    Simple confidence classification based on:
    - Absolute score
    - Margin to next-best candidate
    
    Returns: 'HIGH', 'MEDIUM', 'LOW'
    """
    if not all_candidates_for_event:
        return "UNRESOLVED"

    score = candidate.fused_score
    sorted_scores = sorted(
        [c.fused_score for c in all_candidates_for_event], reverse=True
    )

    # Margin: how much better is this candidate vs the mean of others
    if len(sorted_scores) > 1:
        best = sorted_scores[0]
        second = sorted_scores[1]
        margin = best - second
    else:
        margin = score

    if score >= 0.70 and margin >= 0.10:
        return "HIGH"
    elif score >= 0.45 or margin >= 0.05:
        return "MEDIUM"
    else:
        return "LOW"


# ---------------------------------------------------------------------------
# Transition Score
# ---------------------------------------------------------------------------

def transition_score(
    prev: Candidate,
    curr: Candidate,
    strict_order: bool = True,
) -> float:
    """
    Simple transition score between consecutive events.
    
    Rules:
    - If temporal order violated: return -inf (blocks transition)
    - If same frame: return -inf (blocks duplicate)
    - Otherwise: 0.0 (no bonus/penalty in v1)
    
    Future: could add light penalty for unnaturally close timestamps.
    """
    if strict_order:
        if curr.timestamp_seconds <= prev.timestamp_seconds:
            return float("-inf")
        if curr.frame_id == prev.frame_id:
            return float("-inf")
    else:
        # Non-strict: only block exact same timestamp
        if curr.timestamp_seconds < prev.timestamp_seconds:
            return float("-inf")
        if (curr.timestamp_seconds == prev.timestamp_seconds and
                curr.frame_id == prev.frame_id):
            return float("-inf")

    return 0.0


# ---------------------------------------------------------------------------
# Dynamic Programming Sequence Optimizer
# ---------------------------------------------------------------------------

def optimize_sequence(
    event_candidates: Dict[int, List[Candidate]],
    total_events: int,
    strict_order: bool = True,
) -> Tuple[List[Optional[Candidate]], float]:
    """
    Find the globally optimal sequence f_1, f_2, ..., f_N that maximizes
    total score subject to strict temporal ordering.
    
    Supports skipping missing/incompatible intermediate events by allowing
    transitions from any prior event i' < i.
    
    Algorithm:
        dp[i][j] = best total score for a valid subsequence ending at candidate j of event i.
        dp[i][j] = score(i, j) + max_{(i'<i, k)} (dp[i'][k] + transition((i',k), (i,j)))
    
    Args:
        event_candidates: {event_index: [Candidate, ...]} — candidates per event.
                          Events may be missing entirely.
        total_events: Total number of events N.
        strict_order: If True, require strict < ordering on timestamps.
    
    Returns:
        (optimal_sequence, total_score)
        optimal_sequence[i] is the chosen Candidate for event i, or None if unresolved.
    """
    if total_events == 0:
        return [], 0.0

    events_with_candidates = [i for i in range(total_events) if i in event_candidates and event_candidates[i]]

    if not events_with_candidates:
        return [None] * total_events, 0.0

    # Sort candidates by timestamp for each event
    sorted_candidates: Dict[int, List[Candidate]] = {}
    for ei in events_with_candidates:
        sorted_candidates[ei] = sorted(
            event_candidates[ei], key=lambda c: c.timestamp_seconds
        )

    # dp_score[ei][j]: best cumulative score ending at candidate j of event ei
    # dp_prev[ei][j]: Tuple (prev_event_idx, prev_cand_idx) or None
    dp_score: Dict[int, List[float]] = {}
    dp_prev: Dict[int, List[Optional[Tuple[int, int]]]] = {}

    for pos, curr_event in enumerate(events_with_candidates):
        cands_curr = sorted_candidates[curr_event]
        n_curr = len(cands_curr)
        scores: List[float] = [0.0] * n_curr
        backptrs: List[Optional[Tuple[int, int]]] = [None] * n_curr

        for j in range(n_curr):
            cand_curr = cands_curr[j]
            best_prev_val = float("-inf")
            best_prev_ptr: Optional[Tuple[int, int]] = None

            # Look across all earlier events in events_with_candidates
            for prev_pos in range(pos):
                prev_event = events_with_candidates[prev_pos]
                cands_prev = sorted_candidates[prev_event]
                prev_scores = dp_score[prev_event]

                for k, cand_prev in enumerate(cands_prev):
                    if prev_scores[k] == float("-inf"):
                        continue

                    t_score = transition_score(cand_prev, cand_curr, strict_order)
                    if t_score == float("-inf"):
                        continue

                    val = prev_scores[k] + t_score
                    if val > best_prev_val:
                        best_prev_val = val
                        best_prev_ptr = (prev_event, k)

            if best_prev_ptr is not None and best_prev_val > float("-inf"):
                scores[j] = cand_curr.fused_score + best_prev_val
                backptrs[j] = best_prev_ptr
            else:
                scores[j] = cand_curr.fused_score
                backptrs[j] = None

        dp_score[curr_event] = scores
        dp_prev[curr_event] = backptrs

    # Find the global maximum across all events and all candidates
    best_total_score = float("-inf")
    best_end_event: Optional[int] = None
    best_end_cand_idx: Optional[int] = None

    for ei in events_with_candidates:
        for j, score in enumerate(dp_score[ei]):
            if score > best_total_score:
                best_total_score = score
                best_end_event = ei
                best_end_cand_idx = j

    if best_end_event is None or best_end_cand_idx is None or best_total_score <= float("-inf"):
        return [None] * total_events, 0.0

    # Backtrack following backpointers
    chosen_cands: Dict[int, Candidate] = {}
    curr_ptr: Optional[Tuple[int, int]] = (best_end_event, best_end_cand_idx)

    while curr_ptr is not None:
        ei, ci = curr_ptr
        chosen_cands[ei] = sorted_candidates[ei][ci]
        curr_ptr = dp_prev[ei][ci]

    # Build sequence of length total_events
    result: List[Optional[Candidate]] = [None] * total_events
    for ei, cand in chosen_cands.items():
        if 0 <= ei < total_events:
            result[ei] = cand

    return result, best_total_score


# ---------------------------------------------------------------------------
# Sequence-Aware Video Ranking
# ---------------------------------------------------------------------------

def rank_videos_by_sequence(
    video_candidates: Dict[str, Dict[int, List[Candidate]]],
    total_events: int,
    strict_order: bool = True,
    sequence_weight: float = 0.60,
    coverage_weight: float = 0.20,
    retrieval_weight: float = 0.20,
    top_k_videos: int = 5,
) -> List[SequenceResult]:
    """
    For each video:
    1. Run DP sequence optimizer
    2. Compute sequence score, coverage, retrieval score
    3. Rank videos by weighted combination
    
    Returns: Sorted list of SequenceResult (best first), up to top_k_videos.
    """
    results: List[SequenceResult] = []

    for video_id, event_cands in video_candidates.items():
        sequence, seq_score = optimize_sequence(
            event_cands, total_events, strict_order
        )

        # Compute coverage
        resolved = [i for i, c in enumerate(sequence) if c is not None]
        missing = [i for i, c in enumerate(sequence) if c is None]
        coverage_count = len(resolved)
        coverage = coverage_count / total_events if total_events > 0 else 0.0

        # Check temporal validity of resolved sequence
        temporal_valid = True
        resolved_cands = [sequence[i] for i in resolved]
        for idx in range(1, len(resolved_cands)):
            prev_c = resolved_cands[idx - 1]
            curr_c = resolved_cands[idx]
            if prev_c and curr_c:
                if curr_c.timestamp_seconds <= prev_c.timestamp_seconds:
                    temporal_valid = False
                    break
                if curr_c.frame_id == prev_c.frame_id:
                    temporal_valid = False
                    break

        # Retrieval score: average of all candidate scores across all events
        all_scores = []
        for ei, cands in event_cands.items():
            all_scores.extend(c.fused_score for c in cands)
        retrieval_score = sum(all_scores) / len(all_scores) if all_scores else 0.0

        # Confidence per event
        confidences: List[str] = []
        for ei in range(total_events):
            chosen = sequence[ei]
            if chosen is None:
                confidences.append("UNRESOLVED")
            else:
                event_cand_list = event_cands.get(ei, [])
                confidences.append(
                    compute_candidate_confidence(chosen, event_cand_list)
                )

        result = SequenceResult(
            video_id=video_id,
            sequence=sequence,
            sequence_score=seq_score,
            coverage=coverage,
            coverage_count=coverage_count,
            total_events=total_events,
            retrieval_score=retrieval_score,
            temporal_valid=temporal_valid,
            missing_events=missing,
            confidence_per_event=confidences,
        )
        results.append(result)

    # Normalize scores for ranking
    max_seq_score = max((r.sequence_score for r in results), default=1.0)
    max_ret_score = max((r.retrieval_score for r in results), default=1.0)
    if max_seq_score <= 0:
        max_seq_score = 1.0
    if max_ret_score <= 0:
        max_ret_score = 1.0

    def final_score(r: SequenceResult) -> float:
        seq_norm = r.sequence_score / max_seq_score if max_seq_score > 0 else 0.0
        cov = r.coverage
        ret_norm = r.retrieval_score / max_ret_score if max_ret_score > 0 else 0.0
        return (
            sequence_weight * seq_norm
            + coverage_weight * cov
            + retrieval_weight * ret_norm
        )

    # Sort: temporal_valid first, then by final_score
    results.sort(
        key=lambda r: (r.temporal_valid, final_score(r)),
        reverse=True,
    )

    # Attach final score to each result for API output
    for r in results:
        r._final_score = final_score(r)  # type: ignore[attr-defined]

    return results[:top_k_videos]


# ---------------------------------------------------------------------------
# Constrained Re-Search Interface (Hook for future implementation)
# ---------------------------------------------------------------------------

def constrained_re_search_candidates(
    event_index: int,
    lower_bound_seconds: float,
    upper_bound_seconds: float,
    all_keyframes: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Interface for constrained re-search when an event is unresolved
    but neighboring events define a temporal window.
    
    Example: E2=100s, E4=300s, E3 unresolved
             → search keyframes where 100 < timestamp < 300
    
    Current implementation: returns keyframes in the window for manual selection.
    Future: could run targeted retrieval within the temporal window.
    """
    filtered = [
        kf for kf in all_keyframes
        if lower_bound_seconds < float(kf.get("pts_time", 0.0)) < upper_bound_seconds
    ]
    return filtered
