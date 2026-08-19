"""
Frame Selector — picks the best keyframe(s) from a ranked candidate list.

For KIS / Q&A: selects 1 best keyframe from the fused+reranked list.
For TRAKE:     selects 1 best keyframe per event step (called N times).

Also de-duplicates nearby frames within the same shot to avoid
submitting near-identical frames for different queries.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from src.common.types import SearchResult, EvidenceResult
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Minimum temporal gap (seconds) between selected keyframes
# to avoid selecting the same moment twice
_MIN_TEMPORAL_GAP = 1.0


class FrameSelector:
    """
    Selects the single best keyframe (or one per event) from ranked results.

    Usage — KIS / Q&A:
        selector = FrameSelector()
        evidence = selector.select_best(ranked_results, query_id="q001")

    Usage — TRAKE (called per event):
        evidence = selector.select_best(event_results, query_id="q001_event1")
    """

    def select_best(
        self,
        results: List[SearchResult],
        query_id: str = "",
        explanation: str = "",
    ) -> Optional[EvidenceResult]:
        """
        Return the top-ranked result as an EvidenceResult.

        Args:
            results:     Ranked list (highest score first), typically after fusion+rerank
            query_id:    For logging
            explanation: Human-readable reason (filled by LLM reranker in Sprint 4)

        Returns:
            EvidenceResult or None if results list is empty
        """
        if not results:
            logger.warning(f"[FrameSelector] Empty result list for query_id='{query_id}'")
            return None

        best = results[0]
        evidence = EvidenceResult(
            video_id=best.video_id,
            frame_idx=best.frame_idx,
            n=best.n,
            pts_time=best.pts_time,
            confidence=best.score,
            explanation=explanation or f"Top result from {best.retriever_source} (score={best.score:.4f})",
            top_results=results[:20],
        )

        logger.debug(
            f"[FrameSelector] query='{query_id}' → "
            f"{best.video_id} frame_idx={best.frame_idx} "
            f"pts={best.pts_time:.2f}s score={best.score:.4f}"
        )
        return evidence

    def select_diverse(
        self,
        results: List[SearchResult],
        n_select: int = 1,
        min_gap_seconds: float = _MIN_TEMPORAL_GAP,
    ) -> List[SearchResult]:
        """
        Select up to n_select results that are temporally diverse
        (i.e., not all from the same moment of the same video).

        Used when we want to present multiple candidate frames to the user.

        Args:
            results:          Ranked candidates (highest score first)
            n_select:         Number of frames to select
            min_gap_seconds:  Minimum time gap between selected frames
                              within the same video

        Returns:
            Subset of results (up to n_select), temporally diverse
        """
        selected: List[SearchResult] = []
        # Per-video: track pts_times of already-selected frames
        selected_times: Dict[str, List[float]] = {}

        for result in results:
            if len(selected) >= n_select:
                break

            vid = result.video_id
            ts = result.pts_time

            # Check temporal gap within same video
            if vid in selected_times:
                too_close = any(
                    abs(ts - prev_ts) < min_gap_seconds
                    for prev_ts in selected_times[vid]
                )
                if too_close:
                    continue

            selected.append(result)
            selected_times.setdefault(vid, []).append(ts)

        return selected

    def select_per_event(
        self,
        event_results: Dict[int, List[SearchResult]],
        enforce_temporal_order: bool = True,
    ) -> Dict[int, Optional[SearchResult]]:
        """
        Select one keyframe per event for TRAKE alignment.

        Args:
            event_results:         {event_id: ranked_results}
            enforce_temporal_order: If True, ensures pts_time of event N+1
                                    is strictly greater than event N

        Returns:
            {event_id: best_SearchResult_or_None}
        """
        selections: Dict[int, Optional[SearchResult]] = {}
        last_pts: float = -1.0
        used_keyframe_ids = set()

        for event_id in sorted(event_results.keys()):
            candidates = event_results[event_id]

            if not candidates:
                selections[event_id] = None
                continue

            if not enforce_temporal_order:
                # Pick top candidate not already used
                chosen = next((c for c in candidates if c.keyframe_id not in used_keyframe_ids), candidates[0])
                selections[event_id] = chosen
                used_keyframe_ids.add(chosen.keyframe_id)
                continue

            # Pick the best candidate that is temporally after the previous event (+0.5s gap) and not already used
            chosen = None
            for c in candidates:
                if c.pts_time > (last_pts + 0.5) and c.keyframe_id not in used_keyframe_ids:
                    chosen = c
                    break

            if chosen is None:
                # Fallback 1: Any candidate after last_pts not already used
                for c in candidates:
                    if c.pts_time > last_pts and c.keyframe_id not in used_keyframe_ids:
                        chosen = c
                        break

            if chosen is None:
                # Fallback 2: Any unused candidate
                for c in candidates:
                    if c.keyframe_id not in used_keyframe_ids:
                        chosen = c
                        break

            if chosen is None:
                chosen = candidates[0]

            selections[event_id] = chosen
            used_keyframe_ids.add(chosen.keyframe_id)
            last_pts = max(last_pts, chosen.pts_time)

        return selections
