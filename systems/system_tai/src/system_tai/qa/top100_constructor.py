"""Ranked Top-100 Answer List Constructor for Video Q&A (QA-R1.1 Interleaved Anti-Starvation).

Conforms to Master PDF Decision 10 with interleaved coverage and early close temporal depth.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from system_tai.preliminary.matching import NormalizedAliasAnswerMatcher
from system_tai.preliminary.schemas import QAPrediction
from system_tai.preliminary.validation import validate_ranked_top100


def construct_ranked_qa_top100(
    query_id: str,
    scored_candidates: Sequence[dict[str, Any]],
    output_top_k: int = 100,
    *,
    expand_temporal: bool = True,
    return_provenance: bool = False,
) -> list[QAPrediction] | tuple[list[QAPrediction], list[dict[str, Any]]]:
    """Build an optimal, metric-aware Top-100 prediction list for Video Q&A.

    QA-R1.1 Interleaved Anti-Starvation Allocation Geometry:
    1. Phase 1 (Top 1..5 Primary & Close Temporal Depth):
       - Candidates 1..5 emit primary + close offsets (±30, +15)
       - Ranks ~1..16: Preserves elite early temporal depth for top candidates.
    2. Phase 2 (Anti-Starvation Video Coverage for Candidates 6..32):
       - Candidates 6..N emit 1 primary prediction each
       - Ranks ~17..43: Guarantees all 32 candidate videos receive a slot by ~rank 43.
    3. Phase 3 (Alternative Answer Hypotheses for Top 10 Candidates):
       - Ranks ~44..53: Hypothesis #2 for top candidates.
    4. Phase 4 (Medium Temporal Neighborhood for Top 16 Candidates):
       - Ranks ~54..80: Round-robin offsets [±60, ±45, ±90, ±120].
    5. Phase 5 (Close Offsets for Candidates 6..32):
       - Ranks ~81..95: Offsets [±30, ±60] for candidate videos 6..32.
    6. Phase 6 (Wide Temporal Fallback):
       - Ranks ~96..100: Offsets [±150..±300] to fill remaining slots up to target_k.

    Guarantees:
    - Invariant 1: Top 5 candidates have immediate close temporal depth (±30) before rank 17.
    - Invariant 2: All 32 nominated candidate videos are represented by ~rank 43 (Anti-Starvation).
    - Invariant 3: Frame bounds clamping (0 <= frame_id <= total_frames).
    - Invariant 4: Contiguous ranks 1..N and max 100 predictions.
    """
    if not scored_candidates or output_top_k <= 0:
        return ([], []) if return_provenance else []

    target_k = min(100, max(1, output_top_k))
    matcher = NormalizedAliasAnswerMatcher(strip_punctuation=True)
    seen_keys: set[tuple[str, int, str]] = set()
    predictions: list[QAPrediction] = []
    provenance_records: list[dict[str, Any]] = []

    def _try_add(
        video_id: str,
        frame_id: int,
        answer: str,
        *,
        max_frame: int | None = None,
        candidate_rank: int | None = None,
        slot_source: str = "PRIMARY",
        offset_frames: int = 0,
    ) -> bool:
        if len(predictions) >= target_k:
            return False
        if not answer or not isinstance(answer, str) or not answer.strip():
            return False
        norm_ans = matcher.normalize(answer)
        if not norm_ans:
            return False
        fid = int(frame_id)
        if fid < 0:
            fid = 0
        if max_frame is not None and fid > max_frame:
            fid = max_frame
        key = (video_id, fid, norm_ans)
        if key in seen_keys:
            return False
        seen_keys.add(key)
        final_rank = len(predictions) + 1
        predictions.append(
            QAPrediction(
                query_id=query_id,
                rank=final_rank,
                video_id=video_id,
                frame_id=fid,
                answer=answer.strip(),
            )
        )
        provenance_records.append(
            {
                "query_id": query_id,
                "final_rank": final_rank,
                "video_id": video_id,
                "frame_id": fid,
                "answer": answer.strip(),
                "candidate_rank": candidate_rank,
                "slot_source": slot_source,
                "offset_frames": offset_frames,
            }
        )
        return True

    if not expand_temporal:
        for idx, c in enumerate(scored_candidates):
            if not c.get("answers"):
                continue
            max_f = c.get("total_frames") or c.get("max_frame_id")
            cand_rank = c.get("evidence_rank", idx + 1)
            _try_add(
                c["video_id"],
                c["frame_id"],
                c["answers"][0],
                max_frame=max_f,
                candidate_rank=cand_rank,
                slot_source="PRIMARY_UNEXPANDED",
                offset_frames=0,
            )
        errors = validate_ranked_top100(
            predictions,
            expected_task="qa",
            expected_query_id=query_id,
        )
        if errors:
            msg = "; ".join(e.message for e in errors)
            raise ValueError(f"P0-A QA validation failed: {msg}")
        return (predictions, provenance_records) if return_provenance else predictions

    num_candidates = len(scored_candidates)

    # --------------------------------------------------------------------------
    # Phase 1: Top 1..5 Primary & Close Temporal Depth (±30, +15)
    # Target Ranks: ~1..16
    # --------------------------------------------------------------------------
    for idx in range(min(5, num_candidates)):
        c = scored_candidates[idx]
        max_f = c.get("total_frames") or c.get("max_frame_id")
        cand_rank = c.get("evidence_rank", idx + 1)
        if not c.get("answers"):
            continue
        # Primary frame
        _try_add(
            c["video_id"],
            c["frame_id"],
            c["answers"][0],
            max_frame=max_f,
            candidate_rank=cand_rank,
            slot_source="PRIMARY",
            offset_frames=0,
        )
        # Close positive offset (+30)
        _try_add(
            c["video_id"],
            c["frame_id"] + 30,
            c["answers"][0],
            max_frame=max_f,
            candidate_rank=cand_rank,
            slot_source="CLOSE_OFFSET",
            offset_frames=30,
        )
        # Close negative offset (-30)
        _try_add(
            c["video_id"],
            c["frame_id"] - 30,
            c["answers"][0],
            max_frame=max_f,
            candidate_rank=cand_rank,
            slot_source="CLOSE_OFFSET",
            offset_frames=-30,
        )
        # Top-1 gets additional fine offset (+15)
        if idx == 0:
            _try_add(
                c["video_id"],
                c["frame_id"] + 15,
                c["answers"][0],
                max_frame=max_f,
                candidate_rank=cand_rank,
                slot_source="CLOSE_OFFSET",
                offset_frames=15,
            )

    # --------------------------------------------------------------------------
    # Phase 2: Anti-Starvation Video Coverage for Remaining Candidates 6..32
    # Target Ranks: ~17..43
    # --------------------------------------------------------------------------
    for idx in range(5, num_candidates):
        c = scored_candidates[idx]
        max_f = c.get("total_frames") or c.get("max_frame_id")
        cand_rank = c.get("evidence_rank", idx + 1)
        if c.get("answers"):
            _try_add(
                c["video_id"],
                c["frame_id"],
                c["answers"][0],
                max_frame=max_f,
                candidate_rank=cand_rank,
                slot_source="PRIMARY",
                offset_frames=0,
            )

    # --------------------------------------------------------------------------
    # Phase 3: Alternative Answer Hypotheses for Top 10 Candidates
    # Target Ranks: ~44..53
    # --------------------------------------------------------------------------
    for idx in range(min(10, num_candidates)):
        c = scored_candidates[idx]
        if len(c.get("answers", [])) > 1:
            max_f = c.get("total_frames") or c.get("max_frame_id")
            cand_rank = c.get("evidence_rank", idx + 1)
            _try_add(
                c["video_id"],
                c["frame_id"],
                c["answers"][1],
                max_frame=max_f,
                candidate_rank=cand_rank,
                slot_source="ALT_ANSWER",
                offset_frames=0,
            )
            _try_add(
                c["video_id"],
                c["frame_id"] + 30,
                c["answers"][1],
                max_frame=max_f,
                candidate_rank=cand_rank,
                slot_source="ALT_ANSWER_OFFSET",
                offset_frames=30,
            )

    # --------------------------------------------------------------------------
    # Phase 4: Medium Temporal Neighborhood for Top 16 Candidates (Round-Robin)
    # Target Ranks: ~54..80
    # --------------------------------------------------------------------------
    medium_offsets = [60, -60, 45, -45, 90, -90, 120, -120]
    for offset in medium_offsets:
        if len(predictions) >= target_k:
            break
        for idx in range(min(16, num_candidates)):
            if len(predictions) >= target_k:
                break
            c = scored_candidates[idx]
            if c.get("answers"):
                max_f = c.get("total_frames") or c.get("max_frame_id")
                cand_rank = c.get("evidence_rank", idx + 1)
                _try_add(
                    c["video_id"],
                    c["frame_id"] + offset,
                    c["answers"][0],
                    max_frame=max_f,
                    candidate_rank=cand_rank,
                    slot_source="MEDIUM_OFFSET",
                    offset_frames=offset,
                )

    # --------------------------------------------------------------------------
    # Phase 5: Close Temporal Offsets for Candidates 6..32 (±30, ±60)
    # Target Ranks: ~81..95
    # --------------------------------------------------------------------------
    for offset in [30, -30, 60, -60]:
        if len(predictions) >= target_k:
            break
        for idx in range(5, num_candidates):
            if len(predictions) >= target_k:
                break
            c = scored_candidates[idx]
            if c.get("answers"):
                max_f = c.get("total_frames") or c.get("max_frame_id")
                cand_rank = c.get("evidence_rank", idx + 1)
                _try_add(
                    c["video_id"],
                    c["frame_id"] + offset,
                    c["answers"][0],
                    max_frame=max_f,
                    candidate_rank=cand_rank,
                    slot_source="LATE_CANDIDATE_OFFSET",
                    offset_frames=offset,
                )

    # --------------------------------------------------------------------------
    # Phase 6: Wide Temporal Neighborhood Fallback up to target_k (Max 100)
    # Target Ranks: ~96..100
    # --------------------------------------------------------------------------
    wider_offsets = [150, -150, 180, -180, 210, -210, 240, -240, 270, -270, 300, -300]
    for offset in wider_offsets:
        if len(predictions) >= target_k:
            break
        for idx in range(num_candidates):
            if len(predictions) >= target_k:
                break
            c = scored_candidates[idx]
            if c.get("answers"):
                max_f = c.get("total_frames") or c.get("max_frame_id")
                cand_rank = c.get("evidence_rank", idx + 1)
                _try_add(
                    c["video_id"],
                    c["frame_id"] + offset,
                    c["answers"][0],
                    max_frame=max_f,
                    candidate_rank=cand_rank,
                    slot_source="WIDE_OFFSET",
                    offset_frames=offset,
                )

    # Final Validation
    errors = validate_ranked_top100(
        predictions,
        expected_task="qa",
        expected_query_id=query_id,
    )
    if errors:
        msg = "; ".join(e.message for e in errors)
        raise ValueError(f"P0-A QA validation failed: {msg}")

    return (predictions, provenance_records) if return_provenance else predictions
