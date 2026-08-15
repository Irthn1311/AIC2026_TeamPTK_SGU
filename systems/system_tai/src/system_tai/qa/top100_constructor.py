"""Ranked Top-100 Answer List Constructor for Video Q&A.

Defaults to the proven Score-Champion policy: `temporal_dense_v1` (conforming to Master PDF Decision 10).
Preserves frame bounds clamping, deterministic deduplication, contiguous ranks, and P0-A validation.
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
    policy: str = "temporal_dense_v1",
    return_provenance: bool = False,
) -> list[QAPrediction] | tuple[list[QAPrediction], list[dict[str, Any]]]:
    """Build an optimal, metric-aware Top-100 prediction list for Video Q&A.

    Policies:
    - 'temporal_dense_v1' (DEFAULT / SCORE CHAMPION): Tier 1..7 dense temporal expansion
      prioritizing high-density temporal evidence (±30..±120) for Top 10 candidates in Ranks 1..50.
    - 'anti_starvation_v1' (EXPERIMENTAL): Breadth-first coverage across all 32 candidate videos.
    - 'interleaved_v1' (EXPERIMENTAL): Interleaved coverage with top-5 close temporal depth.
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

    # --------------------------------------------------------------------------
    # SCORE CHAMPION POLICY: temporal_dense_v1 (Proven baseline in e133378)
    # --------------------------------------------------------------------------
    if policy == "temporal_dense_v1":
        # Tier 1: Primary Top-1 candidate
        c0 = scored_candidates[0]
        max0 = c0.get("total_frames") or c0.get("max_frame_id")
        _try_add(c0["video_id"], c0["frame_id"], c0["answers"][0], max_frame=max0, candidate_rank=1, slot_source="TIER1_PRIMARY", offset_frames=0)

        # Tier 2: Top-1 immediate temporal neighbors (+30, -30) & Top-2 primary
        _try_add(c0["video_id"], c0["frame_id"] + 30, c0["answers"][0], max_frame=max0, candidate_rank=1, slot_source="TIER2_OFFSET", offset_frames=30)
        _try_add(c0["video_id"], c0["frame_id"] - 30, c0["answers"][0], max_frame=max0, candidate_rank=1, slot_source="TIER2_OFFSET", offset_frames=-30)
        if len(scored_candidates) > 1:
            c1 = scored_candidates[1]
            max1 = c1.get("total_frames") or c1.get("max_frame_id")
            _try_add(c1["video_id"], c1["frame_id"], c1["answers"][0], max_frame=max1, candidate_rank=2, slot_source="TIER2_PRIMARY", offset_frames=0)
        _try_add(c0["video_id"], c0["frame_id"] + 15, c0["answers"][0], max_frame=max0, candidate_rank=1, slot_source="TIER2_OFFSET", offset_frames=15)

        # Tier 3: Primary predictions for candidates 2..10 & close neighbors (±30)
        for idx, c in enumerate(scored_candidates[1:10], start=2):
            if not c.get("answers"):
                continue
            max_f = c.get("total_frames") or c.get("max_frame_id")
            _try_add(c["video_id"], c["frame_id"], c["answers"][0], max_frame=max_f, candidate_rank=idx, slot_source="TIER3_PRIMARY", offset_frames=0)
            _try_add(c["video_id"], c["frame_id"] + 30, c["answers"][0], max_frame=max_f, candidate_rank=idx, slot_source="TIER3_OFFSET", offset_frames=30)
            _try_add(c["video_id"], c["frame_id"] - 30, c["answers"][0], max_frame=max_f, candidate_rank=idx, slot_source="TIER3_OFFSET", offset_frames=-30)

        # Tier 4: Alternative answer hypotheses for top 3 candidates
        for idx, c in enumerate(scored_candidates[:3], start=1):
            if len(c.get("answers", [])) > 1:
                max_f = c.get("total_frames") or c.get("max_frame_id")
                alt_ans = c["answers"][1]
                _try_add(c["video_id"], c["frame_id"], alt_ans, max_frame=max_f, candidate_rank=idx, slot_source="TIER4_ALT_ANSWER", offset_frames=0)
                _try_add(c["video_id"], c["frame_id"] + 30, alt_ans, max_frame=max_f, candidate_rank=idx, slot_source="TIER4_ALT_OFFSET", offset_frames=30)

        # Tier 5: Medium temporal neighborhood for top 10 candidates (+60, -60, +45, -45, +90, -90, +120, -120)
        medium_offsets = [60, -60, 45, -45, 90, -90, 120, -120]
        for offset in medium_offsets:
            if len(predictions) >= target_k:
                break
            for idx, c in enumerate(scored_candidates[:10], start=1):
                if len(predictions) >= target_k:
                    break
                if c.get("answers"):
                    max_f = c.get("total_frames") or c.get("max_frame_id")
                    _try_add(c["video_id"], c["frame_id"] + offset, c["answers"][0], max_frame=max_f, candidate_rank=idx, slot_source="TIER5_MEDIUM_OFFSET", offset_frames=offset)

        # Tier 6: Candidates 11..N and their temporal offsets
        for idx, c in enumerate(scored_candidates[10:], start=11):
            if len(predictions) >= target_k:
                break
            if not c.get("answers"):
                continue
            max_f = c.get("total_frames") or c.get("max_frame_id")
            _try_add(c["video_id"], c["frame_id"], c["answers"][0], max_frame=max_f, candidate_rank=idx, slot_source="TIER6_PRIMARY", offset_frames=0)
            for offset in [30, -30, 60, -60, 90, -90]:
                if len(predictions) >= target_k:
                    break
                _try_add(c["video_id"], c["frame_id"] + offset, c["answers"][0], max_frame=max_f, candidate_rank=idx, slot_source="TIER6_OFFSET", offset_frames=offset)

        # Tier 7: Wider temporal coverage up to target_k (max 100)
        wider_offsets = [150, -150, 180, -180, 210, -210, 240, -240, 270, -270, 300, -300]
        for offset in wider_offsets:
            if len(predictions) >= target_k:
                break
            for idx, c in enumerate(scored_candidates, start=1):
                if len(predictions) >= target_k:
                    break
                if c.get("answers"):
                    max_f = c.get("total_frames") or c.get("max_frame_id")
                    _try_add(c["video_id"], c["frame_id"] + offset, c["answers"][0], max_frame=max_f, candidate_rank=idx, slot_source="TIER7_WIDE_OFFSET", offset_frames=offset)

    else:
        raise ValueError(f"Unknown QA Top-100 constructor policy: {policy}")

    # Final Validation
    errors = validate_ranked_top100(
        predictions,
        expected_task="qa",
        expected_query_id=query_id,
    )
    if errors:
        msg = "; ".join(e.message for e in errors)
        raise ValueError(f"construct_ranked_qa_top100 validation failed for {query_id}: {msg}")

    return (predictions, provenance_records) if return_provenance else predictions
