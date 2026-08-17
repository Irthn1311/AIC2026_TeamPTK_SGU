# ==============================================================================================================
# QA Candidate Selection & Provenance Utilities
# ==============================================================================================================

from __future__ import annotations

from typing import Any, Mapping


def select_fourth_unique_primary_candidate(
    candidates: list[Any],
) -> tuple[int, Any, dict[str, Any]] | None:
    """
    Selects the 4th primary evidence candidate ordered strictly by video nomination rank.

    Fail-closed policy:
    - Missing, None, or unparseable local_anchor_rank -> strictly ineligible.
    - Missing, None, or unparseable video_nomination_rank -> strictly ineligible.
    - local_anchor_rank != 1 -> strictly ineligible.
    - video_nomination_rank < 1 -> strictly ineligible.
    - Dedup: Ensures exactly one primary candidate per unique nominated video (first seen).
    - Sort key: (video_nomination_rank, video_id, frame_id).

    Returns:
        tuple of (primary_index_0_based, candidate_object, provenance_info) or None if < 4 unique primary candidates.
    """
    valid_primaries: list[tuple[int, str, int, Any]] = []
    seen_videos: set[tuple[int, str]] = set()

    for cand in candidates:
        video_id: str | None = None
        frame_id: int | None = None
        local_rank: int | None = None
        nom_rank: int | None = None

        if isinstance(cand, tuple) and len(cand) >= 2:
            # (ev_cand, ref_cand) tuple from runtime.py
            ev_cand, ref_cand = cand[0], cand[1]
            video_id = getattr(ev_cand, "video_id", None) or getattr(ref_cand, "video_id", None)
            frame_id = getattr(ev_cand, "frame_id", None) or getattr(ref_cand, "refined_frame_id", None) or getattr(ref_cand, "candidate_frame_id", None)

            raw_local = getattr(ref_cand, "local_anchor_rank", None)
            raw_nom = getattr(ref_cand, "video_nomination_rank", None)
            if raw_local is None and hasattr(ev_cand, "provenance") and isinstance(ev_cand.provenance, Mapping):
                raw_local = ev_cand.provenance.get("local_anchor_rank")
            if raw_nom is None and hasattr(ev_cand, "provenance") and isinstance(ev_cand.provenance, Mapping):
                raw_nom = ev_cand.provenance.get("video_nomination_rank")

            try:
                if raw_local is not None:
                    local_rank = int(raw_local)
                if raw_nom is not None:
                    nom_rank = int(raw_nom)
            except (ValueError, TypeError):
                continue
        elif isinstance(cand, Mapping):
            # Dict representation from engine.py / constructor
            video_id = cand.get("video_id")
            raw_frame = cand.get("frame_id")
            raw_local = cand.get("local_anchor_rank")
            raw_nom = cand.get("video_nomination_rank")
            try:
                if raw_frame is not None:
                    frame_id = int(raw_frame)
                if raw_local is not None:
                    local_rank = int(raw_local)
                if raw_nom is not None:
                    nom_rank = int(raw_nom)
            except (ValueError, TypeError):
                continue
        elif hasattr(cand, "provenance") and isinstance(cand.provenance, Mapping):
            # QAEvidenceCandidate with provenance dict
            video_id = getattr(cand, "video_id", None)
            raw_frame = getattr(cand, "frame_id", None)
            raw_local = cand.provenance.get("local_anchor_rank")
            raw_nom = cand.provenance.get("video_nomination_rank")
            try:
                if raw_frame is not None:
                    frame_id = int(raw_frame)
                if raw_local is not None:
                    local_rank = int(raw_local)
                if raw_nom is not None:
                    nom_rank = int(raw_nom)
            except (ValueError, TypeError):
                continue

        # Strict fail-closed validation
        if (
            video_id is None
            or frame_id is None
            or local_rank is None
            or nom_rank is None
            or local_rank != 1
            or nom_rank < 1
        ):
            continue

        vid_key = (nom_rank, str(video_id))
        if vid_key in seen_videos:
            continue
        seen_videos.add(vid_key)

        valid_primaries.append((nom_rank, str(video_id), int(frame_id), cand))

    valid_primaries.sort(key=lambda item: (item[0], item[1], item[2]))

    if len(valid_primaries) < 4:
        return None

    target_nom_rank, target_vid, target_frame, target_cand = valid_primaries[3]
    provenance = {
        "primary_index_0_based": 3,
        "primary_index_1_based": 4,
        "video_nomination_rank": target_nom_rank,
        "local_anchor_rank": 1,
        "video_id": target_vid,
        "frame_id": target_frame,
        "source_key": f"{target_vid}:{target_frame}:{target_nom_rank}:1",
    }
    return 3, target_cand, provenance
