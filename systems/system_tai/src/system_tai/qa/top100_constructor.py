"""Ranked Top-100 Answer List Constructor for Video Q&A (QA-R1 Diversity & Anti-Starvation).

Conforms to Master PDF Decision 10 with guaranteed candidate video coverage before deep temporal expansion.
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
) -> list[QAPrediction]:
    """Build an optimal, metric-aware Top-100 prediction list for Video Q&A.

    Balances:
    - Primary candidate precision in Top-1..5 (R@1, R@5)
    - Broad candidate video diversity across all nominated videos (R@20, R@50)
    - Local temporal neighborhood coverage & alternative answer hypotheses (R@50, R@100)

    Guarantees the Anti-Starvation Invariant:
    Every candidate in scored_candidates receives at least one primary slot
    before deep multi-frame temporal offsets consume the quota.
    """
    if not scored_candidates or output_top_k <= 0:
        return []

    target_k = min(100, max(1, output_top_k))
    matcher = NormalizedAliasAnswerMatcher(strip_punctuation=True)
    seen_keys: set[tuple[str, int, str]] = set()
    predictions: list[QAPrediction] = []

    def _try_add(
        video_id: str,
        frame_id: int,
        answer: str,
        max_frame: int | None = None,
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
        predictions.append(
            QAPrediction(
                query_id=query_id,
                rank=len(predictions) + 1,
                video_id=video_id,
                frame_id=fid,
                answer=answer.strip(),
            )
        )
        return True

    if not expand_temporal:
        for c in scored_candidates:
            if not c.get("answers"):
                continue
            max_f = c.get("total_frames") or c.get("max_frame_id")
            _try_add(c["video_id"], c["frame_id"], c["answers"][0], max_frame=max_f)
        errors = validate_ranked_top100(
            predictions,
            expected_task="qa",
            expected_query_id=query_id,
        )
        if errors:
            msg = "; ".join(e.message for e in errors)
            raise ValueError(f"P0-A QA validation failed: {msg}")
        return predictions

    num_candidates = len(scored_candidates)

    # --------------------------------------------------------------------------
    # Phase 1: Elite Precision (Top-1..3 Candidates + Tight Immediate Offsets)
    # Target Ranks: 1..10
    # --------------------------------------------------------------------------
    c0 = scored_candidates[0]
    max0 = c0.get("total_frames") or c0.get("max_frame_id")
    _try_add(c0["video_id"], c0["frame_id"], c0["answers"][0], max_frame=max0)
    _try_add(c0["video_id"], c0["frame_id"] + 30, c0["answers"][0], max_frame=max0)
    _try_add(c0["video_id"], c0["frame_id"] - 30, c0["answers"][0], max_frame=max0)

    if num_candidates > 1:
        c1 = scored_candidates[1]
        max1 = c1.get("total_frames") or c1.get("max_frame_id")
        _try_add(c1["video_id"], c1["frame_id"], c1["answers"][0], max_frame=max1)

    _try_add(c0["video_id"], c0["frame_id"] + 15, c0["answers"][0], max_frame=max0)

    if num_candidates > 1:
        _try_add(c1["video_id"], c1["frame_id"] + 30, c1["answers"][0], max_frame=max1)

    if num_candidates > 2:
        c2 = scored_candidates[2]
        max2 = c2.get("total_frames") or c2.get("max_frame_id")
        _try_add(c2["video_id"], c2["frame_id"], c2["answers"][0], max_frame=max2)

    if len(c0.get("answers", [])) > 1:
        _try_add(c0["video_id"], c0["frame_id"], c0["answers"][1], max_frame=max0)

    if num_candidates > 1:
        _try_add(c1["video_id"], c1["frame_id"] - 30, c1["answers"][0], max_frame=max1)

    if num_candidates > 2:
        _try_add(c2["video_id"], c2["frame_id"] + 30, c2["answers"][0], max_frame=max2)

    # --------------------------------------------------------------------------
    # Phase 2: Anti-Starvation Primary Video Coverage (Candidates 3..N)
    # Emits 1 primary prediction for EVERY candidate video in scored_candidates
    # Target Ranks: ~11..40
    # --------------------------------------------------------------------------
    for c in scored_candidates[3:]:
        max_f = c.get("total_frames") or c.get("max_frame_id")
        if c.get("answers"):
            _try_add(c["video_id"], c["frame_id"], c["answers"][0], max_frame=max_f)

    # --------------------------------------------------------------------------
    # Phase 3: Close Local Temporal Offsets for Candidates 3..10 (±30)
    # Target Ranks: ~41..55
    # --------------------------------------------------------------------------
    for c in scored_candidates[3:10]:
        max_f = c.get("total_frames") or c.get("max_frame_id")
        if c.get("answers"):
            _try_add(c["video_id"], c["frame_id"] + 30, c["answers"][0], max_frame=max_f)
            _try_add(c["video_id"], c["frame_id"] - 30, c["answers"][0], max_frame=max_f)

    # --------------------------------------------------------------------------
    # Phase 4: Alternate Answer Hypotheses for Top 10 Candidates
    # Target Ranks: ~56..65
    # --------------------------------------------------------------------------
    for c in scored_candidates[:10]:
        if len(c.get("answers", [])) > 1:
            max_f = c.get("total_frames") or c.get("max_frame_id")
            _try_add(c["video_id"], c["frame_id"], c["answers"][1], max_frame=max_f)
            _try_add(c["video_id"], c["frame_id"] + 30, c["answers"][1], max_frame=max_f)

    # --------------------------------------------------------------------------
    # Phase 5: Medium Temporal Neighborhood for Top 16 Candidates (Round-Robin)
    # Target Ranks: ~66..85
    # --------------------------------------------------------------------------
    medium_offsets = [60, -60, 45, -45, 90, -90, 120, -120]
    for offset in medium_offsets:
        if len(predictions) >= target_k:
            break
        for c in scored_candidates[:16]:
            if len(predictions) >= target_k:
                break
            if c.get("answers"):
                max_f = c.get("total_frames") or c.get("max_frame_id")
                _try_add(c["video_id"], c["frame_id"] + offset, c["answers"][0], max_frame=max_f)

    # --------------------------------------------------------------------------
    # Phase 6: Temporal Offsets for Candidates 11..N (±30, ±60)
    # Target Ranks: ~86..95
    # --------------------------------------------------------------------------
    for offset in [30, -30, 60, -60]:
        if len(predictions) >= target_k:
            break
        for c in scored_candidates[10:]:
            if len(predictions) >= target_k:
                break
            if c.get("answers"):
                max_f = c.get("total_frames") or c.get("max_frame_id")
                _try_add(c["video_id"], c["frame_id"] + offset, c["answers"][0], max_frame=max_f)

    # --------------------------------------------------------------------------
    # Phase 7: Wide Temporal Neighborhood Fallback up to target_k (Max 100)
    # Target Ranks: ~96..100
    # --------------------------------------------------------------------------
    wider_offsets = [150, -150, 180, -180, 210, -210, 240, -240, 270, -270, 300, -300]
    for offset in wider_offsets:
        if len(predictions) >= target_k:
            break
        for c in scored_candidates:
            if len(predictions) >= target_k:
                break
            if c.get("answers"):
                max_f = c.get("total_frames") or c.get("max_frame_id")
                _try_add(c["video_id"], c["frame_id"] + offset, c["answers"][0], max_frame=max_f)

    # Final Validation
    errors = validate_ranked_top100(
        predictions,
        expected_task="qa",
        expected_query_id=query_id,
    )
    if errors:
        msg = "; ".join(e.message for e in errors)
        raise ValueError(f"P0-A QA validation failed: {msg}")

    return predictions
