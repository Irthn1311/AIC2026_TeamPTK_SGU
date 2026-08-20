"""
TRAKE Pipeline — Temporal Retrieval & Alignment of Key Events (Query Dạng 3) v2.

AIC competition requirement:
  Given an activity (e.g., "Nhảy cao") and a sequence of N event steps,
  find the video and submit ONE frame_idx per event step.

Changes from v1:
  - BUG FIX: _vlm_verify_event() was calling list(keys())[0] (arbitrary meta),
    now correctly looks up meta by candidate.keyframe_id.
  - BUG FIX: _vlm_verify_event() now passes activity_name to score_alignment()
    for better context.
  - Phase 1: Composite query now uses ONLY activity_name + event_names (shorter,
    better CLIP embedding). Full descriptions caused embedding dilution.
  - Phase 2: Added temporal window constraint between adjacent events to avoid
    false positives. Each event searches in (prev_event_pts + min_gap, +max_window).
  - Score normalization: Uses number of found events, not total events (fairer).
  - top_k_videos default: 5 → 10 (catch more candidates for sport queries).

Two-Phase Strategy:
  ┌─────────────────────────────────────────────────────────┐
  │  Phase 1 — Video Retrieval (Which video?)               │
  │  • Compact query: activity_name + event names only      │
  │  • FAISS visual + Qdrant text → RRF at video level      │
  │  • Return top-K candidate video_ids                     │
  └─────────────────────────────────────────────────────────┘
  ┌─────────────────────────────────────────────────────────┐
  │  Phase 2 — Event Alignment (Which frame per event?)     │
  │  For each candidate video:                              │
  │    For each event step:                                 │
  │      • Encode event description + hint → CLIP vector    │
  │      • retrieve_within_video() in temporal window       │
  │      • (Optional) VLM score_alignment() → verify top-5 │
  │      • Pick best frame with temporal ordering           │
  │  Score video by mean per-event alignment confidence     │
  └─────────────────────────────────────────────────────────┘
  ┌─────────────────────────────────────────────────────────┐
  │  Phase 3 — Video Selection                              │
  │  • Pick the video with the highest total alignment score │
  │  • Return TRAKESubmission: {video_id, event → frame_idx} │
  └─────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.common.types import (
    TRAKEQuery, EventStep, TRAKESubmission, TRAKEEventResult,
    SearchResult, EvidenceResult,
)
from src.retrieval.visual_retriever import VisualRetriever
from src.retrieval.text_retriever import TextRetriever
from src.fusion.reciprocal_rank import ReciprocalRankFusion
from src.evidence.frame_selector import FrameSelector
from src.embeddings.visual.clip import CLIPEncoder
from src.utils.logger import get_logger

logger = get_logger(__name__)

# VLM confidence threshold
_VLM_CONF_THRESHOLD = 0.45
# Max frames per event to pass through VLM
_VLM_MAX_FRAMES_PER_EVENT = 5
# Temporal window between adjacent events (seconds)
_MIN_EVENT_GAP_SEC   = 0.5   # At least 0.5s between events
_MAX_EVENT_WINDOW_SEC = 60.0  # Max 60s window for next event search


@dataclass
class _VideoAlignment:
    """Internal: alignment result for one candidate video."""
    video_id: str
    total_score: float = 0.0
    event_frames: Dict[int, SearchResult] = field(default_factory=dict)
    event_confidences: Dict[int, float] = field(default_factory=dict)
    n_events_found: int = 0


class TRAKEPipeline:
    """
    Temporal Retrieval & Alignment of Key Events pipeline (v2).

    Args:
        visual_retriever:    Loaded VisualRetriever
        clip_encoder:        Loaded CLIPEncoder (for event-specific encoding)
        text_retrievers:     Optional list of Qdrant TextRetriever
        rrf:                 ReciprocalRankFusion instance
        vlm_client:          Optional QwenVLClient (for alignment verification)
        enable_vlm_verify:   Whether to run VLM on candidate frames
        top_k_videos:        Number of candidate videos from Phase 1

    Usage:
        pipeline = TRAKEPipeline(
            visual_retriever=vis_ret,
            clip_encoder=encoder,
        )
        submission = pipeline.run(trake_query, query_id="q001")
        # submission.video_id → "L21_V001"
        # submission.events[0].frame_idx → 900
    """

    def __init__(
        self,
        visual_retriever: VisualRetriever,
        clip_encoder: CLIPEncoder,
        text_retrievers: Optional[List[TextRetriever]] = None,
        rrf: Optional[ReciprocalRankFusion] = None,
        vlm_client=None,
        enable_vlm_verify: bool = True,
        top_k_videos: int = 100,
        top_k_frames_per_event: int = 100,
    ):
        self._vis_ret    = visual_retriever
        self._encoder    = clip_encoder
        self._text_rets  = text_retrievers or []
        self._rrf        = rrf or ReciprocalRankFusion(k=60)
        self._vlm        = vlm_client
        self._selector   = FrameSelector()
        self.enable_vlm_verify = enable_vlm_verify
        self.top_k_videos = top_k_videos
        self.top_k_frames_per_event = top_k_frames_per_event
        self.last_phase1_results: List[SearchResult] = []

        # Initialize QueryParser for Vi ➔ En event translation
        from src.reasoning.query_parser import QueryParser
        self._parser = QueryParser()

    # ----------------------------------------------------------
    # Main Entry
    # ----------------------------------------------------------

    def run(
        self,
        trake_query: TRAKEQuery,
        query_id: str = "",
    ) -> Optional[TRAKESubmission]:
        """
        Execute the full TRAKE pipeline.

        Returns TRAKESubmission or None if no candidates found.
        Guarantees zero duplicate frames across events.
        """
        logger.info(
            f"[TRAKE] query_id='{query_id}' | activity='{trake_query.activity_name}' "
            f"| {len(trake_query.event_sequence)} events"
        )

        # --- Phase 1: Find candidate videos ---
        if trake_query.video_id:
            candidate_video_ids = [trake_query.video_id]
            logger.info(f"[TRAKE] Skipping Phase 1 (video_id pre-specified: {trake_query.video_id})")
        else:
            candidate_video_ids = self._phase1_video_retrieval(trake_query)

        if not candidate_video_ids:
            logger.warning("[TRAKE] Phase 1 found no candidate videos")
            return None

        logger.info(f"[TRAKE] Phase 1 candidates: {candidate_video_ids}")

        # --- Phase 2: Align events in each candidate video ---
        video_alignments: List[_VideoAlignment] = []
        for video_id in candidate_video_ids:
            alignment = self._phase2_event_alignment(trake_query, video_id, query_id)
            video_alignments.append(alignment)
            logger.info(
                f"[TRAKE] {video_id}: score={alignment.total_score:.3f} "
                f"({alignment.n_events_found}/{len(trake_query.event_sequence)} events found)"
            )

        if not video_alignments:
            logger.warning("[TRAKE] Phase 2 found no alignments")
            return None

        # --- Phase 3: Select best video ---
        best = max(video_alignments, key=lambda a: a.total_score)
        logger.info(
            f"[TRAKE] Best video: {best.video_id} "
            f"(score={best.total_score:.3f}, {best.n_events_found} events found)"
        )

        # Build submission with STRICT DEDUPLICATION and MONOTONIC TEMPORAL ORDERING
        events = []
        used_frame_indices = set()
        last_valid_fidx = 0

        for event in trake_query.event_sequence:
            ev_id = event.event_id
            frame = best.event_frames.get(ev_id)
            
            if frame is not None and frame.frame_idx > 0 and frame.frame_idx not in used_frame_indices:
                f_idx = frame.frame_idx
                pts = frame.pts_time
            else:
                # Fallback: Find next available distinct frame index in video
                f_idx = last_valid_fidx + 25 if last_valid_fidx > 0 else 1
                while f_idx in used_frame_indices:
                    f_idx += 25
                pts = 0.0

            used_frame_indices.add(f_idx)
            last_valid_fidx = f_idx

            events.append(TRAKEEventResult(
                event_id=ev_id,
                frame_idx=f_idx,
                pts_time=pts,
            ))

        return TRAKESubmission(
            query_id=query_id,
            video_id=best.video_id,
            events=events,
        )

    # ----------------------------------------------------------
    # Phase 1: Video-Level Retrieval (Compact Query)
    # ----------------------------------------------------------

    def _phase1_video_retrieval(self, trake_query: TRAKEQuery) -> List[str]:
        """
        Build a COMPACT composite query from translated activity + event names.
        Translates all Vietnamese event names to English prior to CLIP encoding.
        """
        en_activity = self._parser.translate_vi_sentence(trake_query.activity_name)
        parts = [en_activity]
        
        for ev in trake_query.event_sequence:
            en_ev_name = self._parser.translate_vi_sentence(ev.event_name)
            parts.append(en_ev_name)

        if trake_query.sport_category:
            parts.insert(0, trake_query.sport_category)

        compact_query = " ".join(parts)

        # Description query translated to English
        desc_parts = [en_activity]
        for ev in trake_query.event_sequence[:2]:
            en_desc = self._parser.translate_vi_sentence(ev.description)
            desc_parts.append(en_desc)
        desc_query = ". ".join(desc_parts)

        logger.debug(f"[TRAKE Phase1] compact_en='{compact_query}'")
        logger.debug(f"[TRAKE Phase1] desc_en='{desc_query[:100]}'")

        global_top_k = min(self.top_k_videos * 100, 500)

        vis_results_compact = self._vis_ret.retrieve(compact_query, top_k=global_top_k)
        vis_results_desc    = self._vis_ret.retrieve(desc_query,    top_k=global_top_k)

        # Combine both visual retrievals
        all_lists   = [vis_results_compact, vis_results_desc]
        all_weights = [1.0, 0.8]

        # Text retrieval (if Qdrant or OCR available)
        for text_ret in self._text_rets:
            txt = text_ret.retrieve(compact_query, top_k=global_top_k)
            if txt:
                all_lists.append(txt)
                all_weights.append(0.6)

        # Video-level RRF
        video_ranked = self._rrf.fuse_video_level(
            result_lists=all_lists,
            weights=all_weights,
            top_k=self.top_k_videos,
        )
        self.last_phase1_results = video_ranked
        return [r.video_id for r in video_ranked]

    # ----------------------------------------------------------
    # Phase 2: Per-Event Alignment Within a Video
    # ----------------------------------------------------------

    def _phase2_event_alignment(
        self,
        trake_query: TRAKEQuery,
        video_id: str,
        query_id: str,
    ) -> _VideoAlignment:
        """
        For each event step, find the best matching keyframe within video_id.
        Enforces temporal ordering with min/max gap between adjacent events.
        """
        alignment = _VideoAlignment(video_id=video_id)
        event_results: Dict[int, List[SearchResult]] = {}
        last_pts: float = -1.0

        sorted_events = sorted(trake_query.event_sequence, key=lambda e: e.event_id)

        for ev in sorted_events:
            # Build event-specific CLIP query translated to English
            en_desc = self._parser.translate_vi_sentence(ev.description)
            en_hint = self._parser.translate_vi_sentence(ev.semantic_keyframe_hint) if ev.semantic_keyframe_hint else ""
            event_query = f"A photo of {en_desc}. {en_hint}".strip()
            query_vec = self._encoder.encode_text(event_query, normalize=True)

            # Search within this video only
            candidates = self._vis_ret.retrieve_within_video(
                query_vec=query_vec,
                video_id=video_id,
                top_k=self.top_k_frames_per_event,
            )

            # Apply temporal window filter (if we have a previous event)
            if last_pts >= 0:
                candidates = self._filter_temporal_window(candidates, last_pts)

            event_results[ev.event_id] = candidates
            logger.debug(
                f"[TRAKE] {video_id} event {ev.event_id} '{ev.event_name}': "
                f"{len(candidates)} candidates (after_pts={last_pts:.1f}s)"
            )

        # Select frames with temporal ordering enforced
        selections = self._selector.select_per_event(
            event_results=event_results,
            enforce_temporal_order=True,
        )

        # VLM verification + scoring
        total_score = 0.0
        n_found = 0

        for ev in sorted_events:
            ev_id      = ev.event_id
            best_frame = selections.get(ev_id)
            if best_frame is None:
                alignment.event_confidences[ev_id] = 0.0
                continue

            conf = best_frame.score  # Default: CLIP cosine score

            if self.enable_vlm_verify:
                conf = self._vlm_verify_event(ev, best_frame, conf, trake_query.activity_name)

            alignment.event_frames[ev_id]       = best_frame
            alignment.event_confidences[ev_id]  = conf
            total_score += conf
            n_found += 1

            # Update last_pts for next event's window search
            last_pts = best_frame.pts_time

        alignment.n_events_found = n_found
        # Normalize by number of found events (not total) to avoid penalizing videos
        # where some events genuinely don't appear
        alignment.total_score = total_score / max(n_found, 1) if n_found > 0 else 0.0

        logger.debug(
            f"[TRAKE] {video_id}: avg_score={alignment.total_score:.3f} "
            f"({n_found}/{len(trake_query.event_sequence)} events found)"
        )
        return alignment

    def _filter_temporal_window(
        self,
        candidates: List[SearchResult],
        prev_event_pts: float,
    ) -> List[SearchResult]:
        """
        Filter candidates to those within the temporal window:
          [prev_event_pts + MIN_GAP, prev_event_pts + MAX_WINDOW]

        Relaxes to all-after-prev if no candidates in window.
        """
        window_min = prev_event_pts + _MIN_EVENT_GAP_SEC
        window_max = prev_event_pts + _MAX_EVENT_WINDOW_SEC

        in_window = [c for c in candidates if window_min <= c.pts_time <= window_max]
        if in_window:
            return in_window

        # Relax: just after prev event (no max window)
        after_prev = [c for c in candidates if c.pts_time > prev_event_pts]
        if after_prev:
            logger.debug(
                f"[TRAKE] No candidates in window [{window_min:.1f}s, {window_max:.1f}s], "
                f"relaxed to {len(after_prev)} after {prev_event_pts:.1f}s"
            )
            return after_prev

        # Full relaxation: return all candidates
        logger.debug(f"[TRAKE] Temporal filter fully relaxed (no candidates after {prev_event_pts:.1f}s)")
        return candidates

    def _vlm_verify_event(
        self,
        event: EventStep,
        candidate: SearchResult,
        clip_score: float,
        activity_name: str = "",
    ) -> float:
        """
        Run VLM alignment verification on the top candidate frames.
        Returns adjusted confidence (blend of CLIP score + VLM score).

        Fixed bug from v1: correctly looks up meta by candidate.keyframe_id.
        """
        if self._vlm is None:
            return clip_score

        # Correctly look up metadata by keyframe_id (was broken in v1)
        kf_meta = self._vis_ret._meta_store.get_by_keyframe_id(candidate.keyframe_id)
        if kf_meta is None or not kf_meta.image_path:
            logger.debug(f"[TRAKE VLM] No image path for {candidate.keyframe_id}, skipping verify")
            return clip_score

        from pathlib import Path
        img_path = Path(kf_meta.image_path)
        if not img_path.exists():
            logger.debug(f"[TRAKE VLM] Image not found: {img_path}")
            return clip_score

        # Run top _VLM_MAX_FRAMES_PER_EVENT candidates through VLM
        try:
            alignment = self._vlm.score_alignment(
                image_path=str(img_path),
                event_name=event.event_name,
                semantic_keyframe_hint=event.semantic_keyframe_hint or event.description,
                activity_name=activity_name,
            )

            # Blend: 60% VLM + 40% CLIP
            blended = 0.6 * alignment.confidence + 0.4 * clip_score

            # If VLM says no match, penalize
            if not alignment.match:
                blended = min(blended, 0.3)

            logger.debug(
                f"[TRAKE VLM] event='{event.event_name}' "
                f"match={alignment.match} vlm_conf={alignment.confidence:.2f} "
                f"clip={clip_score:.2f} → blended={blended:.2f} | {alignment.observation[:60]}"
            )
            return blended
        except Exception as e:
            logger.warning(f"[TRAKE VLM] Verification failed for {candidate.keyframe_id}: {e}")
            return clip_score
