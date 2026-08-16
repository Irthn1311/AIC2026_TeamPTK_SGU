"""Unit tests for QA-R2F3: Tier 3 Negative-Offset-First Interleaving in Top-100 constructor."""

from __future__ import annotations

import pytest

from system_tai.preliminary.schemas import QAPrediction
from system_tai.qa.top100_constructor import construct_ranked_qa_top100


def _build_synthetic_candidates(num_videos: int = 20, seeds_per_video: int = 2) -> list[dict]:
    """Generate a clean synthetic candidate pool with primaries and secondaries."""
    candidates = []
    # Primary anchors (local_anchor_rank = 1)
    for v_rank in range(1, num_videos + 1):
        vid = f"V{v_rank:03d}"
        candidates.append(
            {
                "video_id": vid,
                "frame_id": v_rank * 1000,
                "answers": [f"answer_{vid}", f"alt_answer_{vid}"],
                "scores": [0.9, 0.7],
                "evidence_rank": len(candidates) + 1,
                "video_nomination_rank": v_rank,
                "local_anchor_rank": 1,
                "total_frames": 50000,
            }
        )
    # Secondary anchors (local_anchor_rank = 2)
    if seeds_per_video >= 2:
        for v_rank in range(1, num_videos + 1):
            vid = f"V{v_rank:03d}"
            candidates.append(
                {
                    "video_id": vid,
                    "frame_id": v_rank * 1000 + 500,
                    "answers": [f"answer_{vid}", f"alt_answer_{vid}"],
                    "scores": [0.8, 0.6],
                    "evidence_rank": len(candidates) + 1,
                    "video_nomination_rank": v_rank,
                    "local_anchor_rank": 2,
                    "total_frames": 50000,
                }
            )
    return candidates


def test_tier3_negative_offset_first_off_preserves_current_behavior_exactly():
    """Invariant 1: Feature OFF reproduces primary-first (direct -> +30 -> -30) exactly tuple-for-tuple."""
    cands = _build_synthetic_candidates(num_videos=20, seeds_per_video=2)
    preds_r2f2 = construct_ranked_qa_top100(
        query_id="TEST-Q1",
        scored_candidates=cands,
        output_top_k=100,
        tier3_primary_first=True,
        tier3_negative_offset_first=False,
    )
    preds_default = construct_ranked_qa_top100(
        query_id="TEST-Q1",
        scored_candidates=cands,
        output_top_k=100,
        tier3_primary_first=True,
    )
    assert len(preds_r2f2) == 100
    assert len(preds_default) == 100
    for p1, p2 in zip(preds_r2f2, default_preds := preds_default):
        assert (p1.rank, p1.video_id, p1.frame_id, p1.answer) == (p2.rank, p2.video_id, p2.frame_id, p2.answer)


def test_tier1_and_tier2_prefix_unchanged_under_sign_swap():
    """Invariant 2: First 5 slots (Tier 1 & Tier 2) are completely identical under sign swap."""
    cands = _build_synthetic_candidates(num_videos=20, seeds_per_video=2)
    
    preds_pos_first, prov_pos = construct_ranked_qa_top100(
        query_id="TEST-Q1",
        scored_candidates=cands,
        output_top_k=100,
        return_provenance=True,
        tier3_primary_first=True,
        tier3_negative_offset_first=False,
    )
    preds_neg_first, prov_neg = construct_ranked_qa_top100(
        query_id="TEST-Q1",
        scored_candidates=cands,
        output_top_k=100,
        return_provenance=True,
        tier3_primary_first=True,
        tier3_negative_offset_first=True,
    )
    
    for i in range(5):
        assert (preds_pos_first[i].rank, preds_pos_first[i].video_id, preds_pos_first[i].frame_id, preds_pos_first[i].answer) == (
            preds_neg_first[i].rank, preds_neg_first[i].video_id, preds_neg_first[i].frame_id, preds_neg_first[i].answer
        )
        assert prov_pos[i]["slot_source"] == prov_neg[i]["slot_source"]


def test_tier3_directs_occur_before_both_sign_phases():
    """Invariant 3: Direct anchors for candidates 3..10 occur before any ±30 offsets."""
    cands = _build_synthetic_candidates(num_videos=20, seeds_per_video=2)
    
    preds_neg, prov_neg = construct_ranked_qa_top100(
        query_id="TEST-Q1",
        scored_candidates=cands,
        output_top_k=100,
        return_provenance=True,
        tier3_primary_first=True,
        tier3_negative_offset_first=True,
    )
    
    t3_primary_indices = [idx for idx, p in enumerate(prov_neg) if p["slot_source"] == "TIER3_PRIMARY"]
    t3_offset_indices = [idx for idx, p in enumerate(prov_neg) if p["slot_source"] == "TIER3_OFFSET"]
    
    assert len(t3_primary_indices) == 8  # Candidates 3..10
    assert len(t3_offset_indices) == 18  # 9 candidates * 2 offsets
    assert max(t3_primary_indices) < min(t3_offset_indices)
    assert t3_primary_indices == list(range(5, 13))  # Ranks 6..13


def test_minus30_phase_precedes_plus30_phase_in_candidate_order():
    """Invariant 4 & 5: -30 phase (ranks 14..22) precedes +30 phase (ranks 23..31), each in candidate order 2..10."""
    cands = _build_synthetic_candidates(num_videos=20, seeds_per_video=2)
    
    preds_neg, prov_neg = construct_ranked_qa_top100(
        query_id="TEST-Q1",
        scored_candidates=cands,
        output_top_k=100,
        return_provenance=True,
        tier3_primary_first=True,
        tier3_negative_offset_first=True,
    )
    
    # Phase T3-B (-30) should occupy indices 13..21 (ranks 14..22)
    minus30_slots = prov_neg[13:22]
    for idx, slot in enumerate(minus30_slots, start=2):
        assert slot["slot_source"] == "TIER3_OFFSET"
        assert slot["offset_frames"] == -30
        assert slot["candidate_rank"] == idx
        assert slot["video_id"] == f"V{idx:03d}"
    
    # Phase T3-C (+30) should occupy indices 22..30 (ranks 23..31)
    plus30_slots = prov_neg[22:31]
    for idx, slot in enumerate(plus30_slots, start=2):
        assert slot["slot_source"] == "TIER3_OFFSET"
        assert slot["offset_frames"] == 30
        assert slot["candidate_rank"] == idx
        assert slot["video_id"] == f"V{idx:03d}"


def test_candidate_7_minus30_emitted_at_rank_19():
    """Invariant 6: In treatment, Candidate 7's -30 offset is emitted at exact Rank 19 (<= 20)."""
    cands = _build_synthetic_candidates(num_videos=20, seeds_per_video=2)
    
    _, prov_neg = construct_ranked_qa_top100(
        query_id="TEST-Q1",
        scored_candidates=cands,
        output_top_k=100,
        return_provenance=True,
        tier3_primary_first=True,
        tier3_negative_offset_first=True,
    )
    
    cand7_minus30 = [
        (idx + 1, p) for idx, p in enumerate(prov_neg)
        if p.get("candidate_rank") == 7 and p.get("offset_frames") == -30 and p.get("slot_source") == "TIER3_OFFSET"
    ]
    assert len(cand7_minus30) == 1
    assert cand7_minus30[0][0] == 19
    assert cand7_minus30[0][0] <= 20


def test_tier3_attempted_and_emitted_tuple_set_identical_under_sign_swap():
    """Invariant 7: Multiset of unique (video_id, frame_id, answer) emitted by Tier 3 is identical between pos-first and neg-first."""
    cands = _build_synthetic_candidates(num_videos=20, seeds_per_video=2)
    
    _, prov_pos = construct_ranked_qa_top100(
        query_id="TEST-Q1",
        scored_candidates=cands,
        output_top_k=100,
        return_provenance=True,
        tier3_primary_first=True,
        tier3_negative_offset_first=False,
    )
    _, prov_neg = construct_ranked_qa_top100(
        query_id="TEST-Q1",
        scored_candidates=cands,
        output_top_k=100,
        return_provenance=True,
        tier3_primary_first=True,
        tier3_negative_offset_first=True,
    )
    
    t3_tuples_pos = {
        (p["video_id"], p["frame_id"], p["answer"])
        for p in prov_pos if p["slot_source"] in ("TIER3_PRIMARY", "TIER3_OFFSET")
    }
    t3_tuples_neg = {
        (p["video_id"], p["frame_id"], p["answer"])
        for p in prov_neg if p["slot_source"] in ("TIER3_PRIMARY", "TIER3_OFFSET")
    }
    assert t3_tuples_pos == t3_tuples_neg
    assert len(t3_tuples_neg) == 26


def test_tier3_dedup_collision_fixture_preserves_unique_emitted_set():
    """Invariant 8: When candidates contain deliberate frame collisions, the unique emitted set after Tier 3 is identical under sign swap."""
    # Build candidate pool where candidate 4 has frame collision with candidate 3 offset
    cands = [
        {"video_id": "V001", "frame_id": 1000, "answers": ["ans1"], "video_nomination_rank": 1, "local_anchor_rank": 1},
        {"video_id": "V002", "frame_id": 2000, "answers": ["ans2"], "video_nomination_rank": 2, "local_anchor_rank": 1},
        {"video_id": "V003", "frame_id": 3000, "answers": ["ans3"], "video_nomination_rank": 3, "local_anchor_rank": 1},
        {"video_id": "V003", "frame_id": 3030, "answers": ["ans3"], "video_nomination_rank": 3, "local_anchor_rank": 2},  # Deliberate collision with V003 +30
        {"video_id": "V004", "frame_id": 4000, "answers": ["ans4"], "video_nomination_rank": 4, "local_anchor_rank": 1},
        {"video_id": "V005", "frame_id": 5000, "answers": ["ans5"], "video_nomination_rank": 5, "local_anchor_rank": 1},
    ]
    
    _, prov_pos = construct_ranked_qa_top100(
        query_id="COLLISION-Q",
        scored_candidates=cands,
        output_top_k=100,
        return_provenance=True,
        tier3_primary_first=True,
        tier3_negative_offset_first=False,
    )
    _, prov_neg = construct_ranked_qa_top100(
        query_id="COLLISION-Q",
        scored_candidates=cands,
        output_top_k=100,
        return_provenance=True,
        tier3_primary_first=True,
        tier3_negative_offset_first=True,
    )
    
    tuples_pos = {(p["video_id"], p["frame_id"], p["answer"]) for p in prov_pos}
    tuples_neg = {(p["video_id"], p["frame_id"], p["answer"]) for p in prov_neg}
    assert tuples_pos == tuples_neg


def test_downstream_first_slot_and_rank_identical_between_pos_and_neg_first():
    """Invariant 9: Downstream phases (Tier 4, Phase A, Primary 11-12, Secondary MB) start at exact same index."""
    cands = _build_synthetic_candidates(num_videos=20, seeds_per_video=2)
    
    _, prov_pos = construct_ranked_qa_top100(
        query_id="TEST-Q1",
        scored_candidates=cands,
        output_top_k=100,
        return_provenance=True,
        secondary_temporal_micro_budget=True,
        primary_11_12_micro_coverage=True,
        tier3_primary_first=True,
        tier3_negative_offset_first=False,
    )
    _, prov_neg = construct_ranked_qa_top100(
        query_id="TEST-Q1",
        scored_candidates=cands,
        output_top_k=100,
        return_provenance=True,
        secondary_temporal_micro_budget=True,
        primary_11_12_micro_coverage=True,
        tier3_primary_first=True,
        tier3_negative_offset_first=True,
    )
    
    # Primary 11-12 slots appear at exact same indices
    prim_idx_pos = [idx for idx, p in enumerate(prov_pos) if p["slot_source"] == "TIER5_PRIMARY_MICRO_COVERAGE"]
    prim_idx_neg = [idx for idx, p in enumerate(prov_neg) if p["slot_source"] == "TIER5_PRIMARY_MICRO_COVERAGE"]
    assert prim_idx_pos == prim_idx_neg
    
    # Secondary micro-budget slots appear at exact same indices
    sec_idx_pos = [idx for idx, p in enumerate(prov_pos) if p["slot_source"] == "TIER5_SECONDARY_MICRO_OFFSET"]
    sec_idx_neg = [idx for idx, p in enumerate(prov_neg) if p["slot_source"] == "TIER5_SECONDARY_MICRO_OFFSET"]
    assert sec_idx_pos == sec_idx_neg


def test_target_k_bounds_strictly_respected():
    """Invariant 10: target_k <= 100 is strictly respected."""
    cands = _build_synthetic_candidates(num_videos=20, seeds_per_video=2)
    preds = construct_ranked_qa_top100(
        query_id="TEST-Q1",
        scored_candidates=cands,
        output_top_k=30,
        tier3_primary_first=True,
        tier3_negative_offset_first=True,
    )
    assert len(preds) == 30
    assert preds[-1].rank == 30
