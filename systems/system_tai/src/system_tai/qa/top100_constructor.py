"""Ranked Top-100 Answer List Constructor for Video Q&A conforming to Master PDF Decision 10."""

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
    - Primary candidate precision (R@1)
    - Local temporal neighborhood coverage (R@5, R@20)
    - Candidate video diversity and alternative answers (R@50, R@100)
    """
    if not scored_candidates or output_top_k <= 0:
        return []

    target_k = min(100, max(1, output_top_k))
    matcher = NormalizedAliasAnswerMatcher(strip_punctuation=True)
    seen_keys: set[tuple[str, int, str]] = set()
    predictions: list[QAPrediction] = []

    def _try_add(video_id: str, frame_id: int, answer: str) -> bool:
        if len(predictions) >= target_k:
            return False
        if not answer or not isinstance(answer, str) or not answer.strip():
            return False
        norm_ans = matcher.normalize(answer)
        if not norm_ans:
            return False
        fid = int(frame_id)
        if fid < 0:
            return False
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
            if not c["answers"]:
                continue
            norm_ans = matcher.normalize(c["answers"][0])
            key = (c["video_id"], int(c["frame_id"]), norm_ans)
            if key in seen_keys or int(c["frame_id"]) < 0:
                continue
            seen_keys.add(key)
            predictions.append(
                QAPrediction(
                    query_id=query_id,
                    rank=len(predictions) + 1,
                    video_id=c["video_id"],
                    frame_id=int(c["frame_id"]),
                    answer=c["answers"][0].strip(),
                )
            )
        errors = validate_ranked_top100(
            predictions,
            expected_task="qa",
            expected_query_id=query_id,
        )
        if errors:
            msg = "; ".join(e.message for e in errors)
            raise ValueError(f"P0-A QA validation failed: {msg}")
        return predictions

    # --------------------------------------------------------------------------
    # Tier 1: Primary Top-1 candidate (Best video, best frame, best answer)
    # --------------------------------------------------------------------------
    c0 = scored_candidates[0]
    _try_add(c0["video_id"], c0["frame_id"], c0["answers"][0])

    # --------------------------------------------------------------------------
    # Tier 2: Top-1 immediate temporal neighbors (+30, -30) & Top-2 primary
    # --------------------------------------------------------------------------
    _try_add(c0["video_id"], max(0, c0["frame_id"] + 30), c0["answers"][0])
    _try_add(c0["video_id"], max(0, c0["frame_id"] - 30), c0["answers"][0])
    if len(scored_candidates) > 1:
        c1 = scored_candidates[1]
        _try_add(c1["video_id"], c1["frame_id"], c1["answers"][0])
    _try_add(c0["video_id"], max(0, c0["frame_id"] + 15), c0["answers"][0])

    # --------------------------------------------------------------------------
    # Tier 3: Primary predictions for candidates 2..10 & their close temporal neighbors
    # --------------------------------------------------------------------------
    for c in scored_candidates[1:10]:
        _try_add(c["video_id"], c["frame_id"], c["answers"][0])
        _try_add(c["video_id"], max(0, c["frame_id"] + 30), c["answers"][0])
        _try_add(c["video_id"], max(0, c["frame_id"] - 30), c["answers"][0])

    # --------------------------------------------------------------------------
    # Tier 4: Alternative answer hypotheses for top 3 candidates
    # --------------------------------------------------------------------------
    for c in scored_candidates[:3]:
        if len(c.get("answers", [])) > 1:
            alt_ans = c["answers"][1]
            _try_add(c["video_id"], c["frame_id"], alt_ans)
            _try_add(c["video_id"], max(0, c["frame_id"] + 30), alt_ans)

    # --------------------------------------------------------------------------
    # Tier 5: Medium temporal neighborhood for top 10 candidates (+60, -60, +45, -45, +90, -90, +120, -120)
    # --------------------------------------------------------------------------
    medium_offsets = [60, -60, 45, -45, 90, -90, 120, -120]
    for offset in medium_offsets:
        for c in scored_candidates[:10]:
            _try_add(c["video_id"], max(0, c["frame_id"] + offset), c["answers"][0])

    # --------------------------------------------------------------------------
    # Tier 6: Candidates 11..N and their temporal offsets
    # --------------------------------------------------------------------------
    for c in scored_candidates[10:]:
        _try_add(c["video_id"], c["frame_id"], c["answers"][0])
        for offset in [30, -30, 60, -60, 90, -90]:
            _try_add(c["video_id"], max(0, c["frame_id"] + offset), c["answers"][0])

    # --------------------------------------------------------------------------
    # Tier 7: Wider temporal coverage up to target_k (max 100)
    # --------------------------------------------------------------------------
    wider_offsets = [150, -150, 180, -180, 210, -210, 240, -240, 270, -270, 300, -300]
    for offset in wider_offsets:
        if len(predictions) >= target_k:
            break
        for c in scored_candidates:
            if len(predictions) >= target_k:
                break
            _try_add(c["video_id"], max(0, c["frame_id"] + offset), c["answers"][0])

    # Final Validation
    errors = validate_ranked_top100(
        predictions,
        expected_task="qa",
        expected_query_id=query_id,
    )
    if errors:
        msg = "; ".join(e.message for e in errors)
        raise ValueError(f"construct_ranked_qa_top100 validation failed for {query_id}: {msg}")

    return predictions
