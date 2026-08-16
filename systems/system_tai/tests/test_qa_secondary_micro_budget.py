"""Unit tests for QA-R2E.1: Secondary-Seed Late Micro-Budget in Top-100 constructor."""

from __future__ import annotations

import pytest

from system_tai.preliminary.schemas import QAPrediction
from system_tai.qa.top100_constructor import construct_ranked_qa_top100


def _build_synthetic_candidates(num_videos: int = 15, seeds_per_video: int = 2) -> list[dict]:
    """Generate a clean synthetic candidate pool with 2 seeds per video."""
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


def test_micro_budget_off_reproduces_legacy_exactly():
    """Invariant 1: Feature OFF reproduces legacy Top-100 predictions tuple-for-tuple."""
    cands = _build_synthetic_candidates(num_videos=12, seeds_per_video=2)
    legacy_preds = construct_ranked_qa_top100(
        query_id="TEST-Q1",
        scored_candidates=cands,
        output_top_k=100,
        secondary_temporal_micro_budget=False,
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


def test_phase_a_prefix_unchanged_when_on():
    """Invariant 2: All outputs emitted before the micro-budget insertion point (Tiers 1..4 + Phase A) remain unchanged."""
    cands = _build_synthetic_candidates(num_videos=12, seeds_per_video=2)
    
    preds_off, prov_off = construct_ranked_qa_top100(
        query_id="TEST-Q1",
        scored_candidates=cands,
        output_top_k=100,
        return_provenance=True,
        secondary_temporal_micro_budget=False,
    )
    preds_on, prov_on = construct_ranked_qa_top100(
        query_id="TEST-Q1",
        scored_candidates=cands,
        output_top_k=100,
        return_provenance=True,
        secondary_temporal_micro_budget=True,
    )
    
    # Identify the index where micro-budget starts
    micro_indices = [
        idx for idx, p in enumerate(prov_on)
        if p["slot_source"] == "TIER5_SECONDARY_MICRO_OFFSET"
    ]
    assert len(micro_indices) > 0, "Expected micro-budget slots to be emitted"
    first_micro_idx = micro_indices[0]
    
    # All slots before the first micro-budget slot must be identical to legacy
    for i in range(first_micro_idx):
        p_off = preds_off[i]
        p_on = preds_on[i]
        assert (p_off.rank, p_off.video_id, p_off.frame_id, p_off.answer) == (p_on.rank, p_on.video_id, p_on.frame_id, p_on.answer)
        assert prov_off[i]["slot_source"] == prov_on[i]["slot_source"]


def test_eligibility_requires_local_rank_2_and_nomination_rank_le_10():
    """Invariant 3: Only local_anchor_rank == 2 and video_nomination_rank <= 10 are eligible."""
    cands = _build_synthetic_candidates(num_videos=15, seeds_per_video=2)
    
    preds_on, prov_on = construct_ranked_qa_top100(
        query_id="TEST-Q1",
        scored_candidates=cands,
        output_top_k=100,
        return_provenance=True,
        secondary_temporal_micro_budget=True,
    )
    
    micro_slots = [p for p in prov_on if p["slot_source"] == "TIER5_SECONDARY_MICRO_OFFSET"]
    vids_in_micro = {p["video_id"] for p in micro_slots}
    
    # Videos 1..10 should be present
    for r in range(1, 11):
        assert f"V{r:03d}" in vids_in_micro
    # Videos 11+ must NOT be in micro slots
    for r in range(11, 16):
        assert f"V{r:03d}" not in vids_in_micro


def test_missing_or_invalid_metadata_makes_candidate_ineligible():
    """Invariant 4: Missing or non-int metadata makes candidate ineligible (never infers from list position)."""
    # 10 primary candidates
    cands = [
        {"video_id": f"V{i:03d}", "frame_id": i * 1000, "answers": [f"ans{i}"], "video_nomination_rank": i, "local_anchor_rank": 1}
        for i in range(1, 11)
    ]
    # Secondary candidates with various metadata states
    cands.extend([
        # Candidate with missing local_anchor_rank
        {"video_id": "V001", "frame_id": 1500, "answers": ["ans1"], "video_nomination_rank": 1, "local_anchor_rank": None},
        # Candidate with string nomination rank
        {"video_id": "V002", "frame_id": 2500, "answers": ["ans2"], "video_nomination_rank": "2", "local_anchor_rank": 2},
        # Valid secondary candidate
        {"video_id": "V003", "frame_id": 3500, "answers": ["ans3"], "video_nomination_rank": 3, "local_anchor_rank": 2},
    ])
    
    preds, prov = construct_ranked_qa_top100(
        query_id="TEST-Q1",
        scored_candidates=cands,
        output_top_k=100,
        return_provenance=True,
        secondary_temporal_micro_budget=True,
    )
    micro_slots = [p for p in prov if p["slot_source"] == "TIER5_SECONDARY_MICRO_OFFSET"]
    micro_vids = [p["video_id"] for p in micro_slots]
    
    assert "V001" not in micro_vids
    assert "V002" not in micro_vids
    assert "V003" in micro_vids


def test_deterministic_nomination_ordering_and_symmetric_offsets():
    """Invariant 5 & 6: Emits -30 then +30 in nomination order (synthetic anchor = 1500 -> 1470, 1530)."""
    # 10 primary candidates
    cands = [
        {"video_id": f"V{i:03d}", "frame_id": i * 1000, "answers": [f"ans{i}"], "video_nomination_rank": i, "local_anchor_rank": 1}
        for i in range(1, 11)
    ]
    # Deliberately out of nomination order in secondary list
    cands.extend([
        {"video_id": "V002", "frame_id": 2500, "answers": ["ans2"], "video_nomination_rank": 2, "local_anchor_rank": 2},
        {"video_id": "V001", "frame_id": 1500, "answers": ["ans1"], "video_nomination_rank": 1, "local_anchor_rank": 2},
    ])
    
    preds, prov = construct_ranked_qa_top100(
        query_id="TEST-Q1",
        scored_candidates=cands,
        output_top_k=100,
        return_provenance=True,
        secondary_temporal_micro_budget=True,
    )
    
    micro_slots = [p for p in prov if p["slot_source"] == "TIER5_SECONDARY_MICRO_OFFSET"]
    # Should emit V001 (-30: 1470, +30: 1530) then V002 (-30: 2470, +30: 2530)
    assert len(micro_slots) == 4
    assert micro_slots[0]["video_id"] == "V001" and micro_slots[0]["frame_id"] == 1470 and micro_slots[0]["offset_frames"] == -30
    assert micro_slots[1]["video_id"] == "V001" and micro_slots[1]["frame_id"] == 1530 and micro_slots[1]["offset_frames"] == 30
    assert micro_slots[2]["video_id"] == "V002" and micro_slots[2]["frame_id"] == 2470 and micro_slots[2]["offset_frames"] == -30
    assert micro_slots[3]["video_id"] == "V002" and micro_slots[3]["frame_id"] == 2530 and micro_slots[3]["offset_frames"] == 30


def test_micro_budget_capped_at_20_successful_slots():
    """Invariant 7: Total secondary micro-budget emissions never exceed 20 slots."""
    cands = _build_synthetic_candidates(num_videos=20, seeds_per_video=2)
    
    preds, prov = construct_ranked_qa_top100(
        query_id="TEST-Q1",
        scored_candidates=cands,
        output_top_k=100,
        return_provenance=True,
        secondary_temporal_micro_budget=True,
    )
    
    micro_slots = [p for p in prov if p["slot_source"] == "TIER5_SECONDARY_MICRO_OFFSET"]
    assert len(micro_slots) == 20  # Exactly 10 videos * 2 offsets = 20 slots max


def test_dedup_collision_does_not_incorrectly_consume_budget():
    """Invariant 8: When an offset collides, budget only counts successful additions."""
    # 10 primary candidates
    cands = [
        {"video_id": f"V{i:03d}", "frame_id": i * 1000, "answers": [f"ans{i}"], "video_nomination_rank": i, "local_anchor_rank": 1}
        for i in range(1, 11)
    ]
    # V001 secondary anchor is 1060.
    # Its -30 offset is 1030 (which collides with Tier 2 offset +30 of primary anchor 1000).
    # Its +30 offset is 1090 (new).
    cands.append(
        {"video_id": "V001", "frame_id": 1060, "answers": ["ans1"], "video_nomination_rank": 1, "local_anchor_rank": 2}
    )
    
    preds, prov = construct_ranked_qa_top100(
        query_id="TEST-Q1",
        scored_candidates=cands,
        output_top_k=100,
        return_provenance=True,
        secondary_temporal_micro_budget=True,
    )
    
    micro_slots = [p for p in prov if p["slot_source"] == "TIER5_SECONDARY_MICRO_OFFSET"]
    # 1030 was already added in Tier 2, so only 1090 is added in micro-budget
    assert len(micro_slots) == 1
    assert micro_slots[0]["frame_id"] == 1090


def test_target_k_bounds_respected():
    """Invariant 9: Target_k <= 100 is strictly respected."""
    cands = _build_synthetic_candidates(num_videos=20, seeds_per_video=2)
    preds = construct_ranked_qa_top100(
        query_id="TEST-Q1",
        scored_candidates=cands,
        output_top_k=80,
        secondary_temporal_micro_budget=True,
    )
    assert len(preds) == 80
    assert preds[-1].rank == 80
