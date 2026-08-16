"""Unit tests for QA-R2F2: Tier 3 Primary-First Interleaving in Top-100 constructor."""

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


def test_tier3_primary_first_off_reproduces_legacy_exactly():
    """Invariant 1: Feature OFF reproduces legacy Top-100 predictions tuple-for-tuple."""
    cands = _build_synthetic_candidates(num_videos=20, seeds_per_video=2)
    legacy_preds = construct_ranked_qa_top100(
        query_id="TEST-Q1",
        scored_candidates=cands,
        output_top_k=100,
        tier3_primary_first=False,
    )
    default_preds = construct_ranked_qa_top100(
        query_id="TEST-Q1",
        scored_candidates=cands,
        output_top_k=100,
    )
    assert len(legacy_preds) == 100
    assert len(default_preds) == 100
    for p1, p2 in zip(legacy_preds, default_preds):
        assert (p1.rank, p1.video_id, p1.frame_id, p1.answer) == (p2.rank, p2.video_id, p2.frame_id, p2.answer)


def test_tier1_and_tier2_prefix_unchanged():
    """Invariant 2: First 5 slots (Tier 1 & Tier 2) are completely identical between OFF and ON."""
    cands = _build_synthetic_candidates(num_videos=20, seeds_per_video=2)
    
    preds_off, prov_off = construct_ranked_qa_top100(
        query_id="TEST-Q1",
        scored_candidates=cands,
        output_top_k=100,
        return_provenance=True,
        tier3_primary_first=False,
    )
    preds_on, prov_on = construct_ranked_qa_top100(
        query_id="TEST-Q1",
        scored_candidates=cands,
        output_top_k=100,
        return_provenance=True,
        tier3_primary_first=True,
    )
    
    for i in range(5):
        assert (preds_off[i].rank, preds_off[i].video_id, preds_off[i].frame_id, preds_off[i].answer) == (
            preds_on[i].rank, preds_on[i].video_id, preds_on[i].frame_id, preds_on[i].answer
        )
        assert prov_off[i]["slot_source"] == prov_on[i]["slot_source"]


def test_tier3_direct_primaries_appear_before_all_tier3_offsets():
    """Invariant 3: In Tier 3, all candidates 3..10 direct primaries appear before any ±30 offsets."""
    cands = _build_synthetic_candidates(num_videos=20, seeds_per_video=2)
    
    preds_on, prov_on = construct_ranked_qa_top100(
        query_id="TEST-Q1",
        scored_candidates=cands,
        output_top_k=100,
        return_provenance=True,
        tier3_primary_first=True,
    )
    
    t3_primary_indices = [
        idx for idx, p in enumerate(prov_on)
        if p["slot_source"] == "TIER3_PRIMARY"
    ]
    t3_offset_indices = [
        idx for idx, p in enumerate(prov_on)
        if p["slot_source"] == "TIER3_OFFSET"
    ]
    
    # 8 primaries in Tier 3 (candidates 3..10, since candidate 2 was emitted in Tier 2)
    assert len(t3_primary_indices) == 8
    assert len(t3_offset_indices) == 18  # 9 candidates (2..10) * 2 offsets (+30, -30)
    
    # All primaries must appear before any offsets
    assert max(t3_primary_indices) < min(t3_offset_indices)
    
    # The primary indices should be consecutive: 5..12 (0-based indices, ranks 6..13)
    assert t3_primary_indices == list(range(5, 13))


def test_tier3_plus30_and_minus30_phases_follow_candidate_order():
    """Invariant 4 & 5: +30 phase precedes -30 phase, each in candidate order 2..10."""
    cands = _build_synthetic_candidates(num_videos=20, seeds_per_video=2)
    
    preds_on, prov_on = construct_ranked_qa_top100(
        query_id="TEST-Q1",
        scored_candidates=cands,
        output_top_k=100,
        return_provenance=True,
        tier3_primary_first=True,
    )
    
    # Phase T3-B (+30) should occupy indices 13..21 (ranks 14..22)
    plus30_slots = prov_on[13:22]
    for idx, slot in enumerate(plus30_slots, start=2):
        assert slot["slot_source"] == "TIER3_OFFSET"
        assert slot["offset_frames"] == 30
        assert slot["candidate_rank"] == idx
        assert slot["video_id"] == f"V{idx:03d}"
    
    # Phase T3-C (-30) should occupy indices 22..30 (ranks 23..31)
    minus30_slots = prov_on[22:31]
    for idx, slot in enumerate(minus30_slots, start=2):
        assert slot["slot_source"] == "TIER3_OFFSET"
        assert slot["offset_frames"] == -30
        assert slot["candidate_rank"] == idx
        assert slot["video_id"] == f"V{idx:03d}"


def test_candidate_7_plus30_moves_to_rank_19_or_20():
    """Invariant 6: In a candidate list, Candidate 7's +30 offset is emitted at exact Rank 19 (<=20)."""
    cands = _build_synthetic_candidates(num_videos=20, seeds_per_video=2)
    
    preds_on, prov_on = construct_ranked_qa_top100(
        query_id="TEST-Q1",
        scored_candidates=cands,
        output_top_k=100,
        return_provenance=True,
        tier3_primary_first=True,
    )
    
    cand7_plus30_on = [
        (idx + 1, p) for idx, p in enumerate(prov_on)
        if p.get("candidate_rank") == 7 and p.get("offset_frames") == 30 and p.get("slot_source") == "TIER3_OFFSET"
    ]
    assert len(cand7_plus30_on) == 1
    # Rank 19 is strictly <= 20
    assert cand7_plus30_on[0][0] <= 20


def test_tier3_emits_same_tuple_set_without_introducing_new_slots():
    """Invariant 7: Multiset of (video_id, frame_id, answer) emitted in Tier 3 is identical between OFF and ON."""
    cands = _build_synthetic_candidates(num_videos=20, seeds_per_video=2)
    
    _, prov_off = construct_ranked_qa_top100(
        query_id="TEST-Q1",
        scored_candidates=cands,
        output_top_k=100,
        return_provenance=True,
        tier3_primary_first=False,
    )
    _, prov_on = construct_ranked_qa_top100(
        query_id="TEST-Q1",
        scored_candidates=cands,
        output_top_k=100,
        return_provenance=True,
        tier3_primary_first=True,
    )
    
    t3_tuples_off = {
        (p["video_id"], p["frame_id"], p["answer"])
        for p in prov_off if p["slot_source"] in ("TIER3_PRIMARY", "TIER3_OFFSET")
    }
    t3_tuples_on = {
        (p["video_id"], p["frame_id"], p["answer"])
        for p in prov_on if p["slot_source"] in ("TIER3_PRIMARY", "TIER3_OFFSET")
    }
    assert t3_tuples_off == t3_tuples_on
    assert len(t3_tuples_off) == 26  # 8 primaries + 18 offsets = 26 unique admissions in Tier 3


def test_downstream_phases_start_at_exact_same_index_and_chain_properly():
    """Invariant 8: Downstream Tier 4, Phase A, Primary 11-12, and Secondary MB chain cleanly with same indices."""
    cands = _build_synthetic_candidates(num_videos=20, seeds_per_video=2)
    
    _, prov_off = construct_ranked_qa_top100(
        query_id="TEST-Q1",
        scored_candidates=cands,
        output_top_k=100,
        return_provenance=True,
        secondary_temporal_micro_budget=True,
        primary_11_12_micro_coverage=True,
        tier3_primary_first=False,
    )
    _, prov_on = construct_ranked_qa_top100(
        query_id="TEST-Q1",
        scored_candidates=cands,
        output_top_k=100,
        return_provenance=True,
        secondary_temporal_micro_budget=True,
        primary_11_12_micro_coverage=True,
        tier3_primary_first=True,
    )
    
    # Primary 11-12 slots should appear at the exact same indices in both OFF and ON
    prim_idx_off = [idx for idx, p in enumerate(prov_off) if p["slot_source"] == "TIER5_PRIMARY_MICRO_COVERAGE"]
    prim_idx_on = [idx for idx, p in enumerate(prov_on) if p["slot_source"] == "TIER5_PRIMARY_MICRO_COVERAGE"]
    assert prim_idx_off == prim_idx_on
    
    # Secondary micro-budget slots should appear at the exact same indices in both OFF and ON
    sec_idx_off = [idx for idx, p in enumerate(prov_off) if p["slot_source"] == "TIER5_SECONDARY_MICRO_OFFSET"]
    sec_idx_on = [idx for idx, p in enumerate(prov_on) if p["slot_source"] == "TIER5_SECONDARY_MICRO_OFFSET"]
    assert sec_idx_off == sec_idx_on


def test_target_k_bounds_strictly_respected():
    """Invariant 9: target_k <= 100 is strictly respected."""
    cands = _build_synthetic_candidates(num_videos=20, seeds_per_video=2)
    preds = construct_ranked_qa_top100(
        query_id="TEST-Q1",
        scored_candidates=cands,
        output_top_k=25,
        tier3_primary_first=True,
    )
    assert len(preds) == 25
    assert preds[-1].rank == 25
