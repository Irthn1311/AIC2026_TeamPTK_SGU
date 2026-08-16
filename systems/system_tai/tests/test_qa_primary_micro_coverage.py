"""Unit tests for QA-R2F1: Primary 11-12 Late Micro-Coverage in Top-100 constructor."""

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


def test_primary_micro_coverage_off_reproduces_legacy_exactly():
    """Invariant 1: Feature OFF reproduces legacy Top-100 predictions tuple-for-tuple."""
    cands = _build_synthetic_candidates(num_videos=20, seeds_per_video=2)
    legacy_preds = construct_ranked_qa_top100(
        query_id="TEST-Q1",
        scored_candidates=cands,
        output_top_k=100,
        primary_11_12_micro_coverage=False,
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


def test_only_local_anchor_rank_1_and_nomination_ranks_11_12_eligible():
    """Invariant 2 & 3: Only local_anchor_rank == 1 and nomination ranks in {11, 12} are eligible."""
    cands = _build_synthetic_candidates(num_videos=20, seeds_per_video=2)
    
    preds_on, prov_on = construct_ranked_qa_top100(
        query_id="TEST-Q1",
        scored_candidates=cands,
        output_top_k=100,
        return_provenance=True,
        primary_11_12_micro_coverage=True,
    )
    
    prim_slots = [p for p in prov_on if p["slot_source"] == "TIER5_PRIMARY_MICRO_COVERAGE"]
    assert len(prim_slots) == 2
    
    emitted_vids = [p["video_id"] for p in prim_slots]
    assert emitted_vids == ["V011", "V012"]
    
    # Ranks 10 and 13 must not be present in TIER5_PRIMARY_MICRO_COVERAGE
    for p in prim_slots:
        assert p["video_id"] not in ("V010", "V013")
        assert p["offset_frames"] == 0


def test_missing_or_invalid_metadata_makes_candidate_ineligible():
    """Invariant 4: Missing or invalid metadata makes candidate ineligible (never infers from position)."""
    cands = [
        {"video_id": f"V{i:03d}", "frame_id": i * 1000, "answers": [f"ans{i}"], "video_nomination_rank": i, "local_anchor_rank": 1}
        for i in range(1, 11)
    ]
    cands.extend([
        # Candidate with missing local_anchor_rank
        {"video_id": "V011", "frame_id": 11000, "answers": ["ans11"], "video_nomination_rank": 11, "local_anchor_rank": None},
        # Candidate with string nomination rank
        {"video_id": "V012", "frame_id": 12000, "answers": ["ans12"], "video_nomination_rank": "12", "local_anchor_rank": 1},
        # Candidate with secondary local_anchor_rank
        {"video_id": "V011", "frame_id": 11500, "answers": ["ans11"], "video_nomination_rank": 11, "local_anchor_rank": 2},
    ])
    
    preds, prov = construct_ranked_qa_top100(
        query_id="TEST-Q1",
        scored_candidates=cands,
        output_top_k=100,
        return_provenance=True,
        primary_11_12_micro_coverage=True,
    )
    prim_slots = [p for p in prov if p["slot_source"] == "TIER5_PRIMARY_MICRO_COVERAGE"]
    assert len(prim_slots) == 0


def test_deterministic_nomination_ordering_11_before_12():
    """Invariant 5: Emits nomination rank 11 before 12 regardless of candidate list ordering."""
    cands = [
        {"video_id": f"V{i:03d}", "frame_id": i * 1000, "answers": [f"ans{i}"], "video_nomination_rank": i, "local_anchor_rank": 1}
        for i in range(1, 11)
    ]
    # Intentionally reverse order in candidate list: 12 before 11
    cands.extend([
        {"video_id": "V012", "frame_id": 12000, "answers": ["ans12"], "video_nomination_rank": 12, "local_anchor_rank": 1},
        {"video_id": "V011", "frame_id": 11000, "answers": ["ans11"], "video_nomination_rank": 11, "local_anchor_rank": 1},
    ])
    
    preds, prov = construct_ranked_qa_top100(
        query_id="TEST-Q1",
        scored_candidates=cands,
        output_top_k=100,
        return_provenance=True,
        primary_11_12_micro_coverage=True,
    )
    prim_slots = [p for p in prov if p["slot_source"] == "TIER5_PRIMARY_MICRO_COVERAGE"]
    assert len(prim_slots) == 2
    assert prim_slots[0]["video_id"] == "V011" and prim_slots[0]["frame_id"] == 11000
    assert prim_slots[1]["video_id"] == "V012" and prim_slots[1]["frame_id"] == 12000


def test_max_successful_emissions_capped_at_2():
    """Invariant 6: Total primary micro-coverage emissions never exceed 2."""
    cands = [
        {"video_id": f"V{i:03d}", "frame_id": i * 1000, "answers": [f"ans{i}"], "video_nomination_rank": i, "local_anchor_rank": 1}
        for i in range(1, 11)
    ]
    # Extra candidates claiming rank 11/12
    cands.extend([
        {"video_id": "V011", "frame_id": 11000, "answers": ["ans11"], "video_nomination_rank": 11, "local_anchor_rank": 1},
        {"video_id": "V012", "frame_id": 12000, "answers": ["ans12"], "video_nomination_rank": 12, "local_anchor_rank": 1},
        {"video_id": "V011b", "frame_id": 11001, "answers": ["ans11b"], "video_nomination_rank": 11, "local_anchor_rank": 1},
    ])
    
    preds, prov = construct_ranked_qa_top100(
        query_id="TEST-Q1",
        scored_candidates=cands,
        output_top_k=100,
        return_provenance=True,
        primary_11_12_micro_coverage=True,
    )
    prim_slots = [p for p in prov if p["slot_source"] == "TIER5_PRIMARY_MICRO_COVERAGE"]
    assert len(prim_slots) == 2


def test_dedup_collision_does_not_incorrectly_consume_budget():
    """Invariant 7: Collision on direct frame does not count toward budget."""
    cands = [
        {"video_id": f"V{i:03d}", "frame_id": i * 1000, "answers": [f"ans{i}"], "video_nomination_rank": i, "local_anchor_rank": 1}
        for i in range(1, 11)
    ]
    # V011 direct anchor has same (video_id, frame_id) as Candidate 1
    cands.extend([
        {"video_id": "V001", "frame_id": 1000, "answers": ["ans1"], "video_nomination_rank": 11, "local_anchor_rank": 1},
        {"video_id": "V012", "frame_id": 12000, "answers": ["ans12"], "video_nomination_rank": 12, "local_anchor_rank": 1},
    ])
    
    preds, prov = construct_ranked_qa_top100(
        query_id="TEST-Q1",
        scored_candidates=cands,
        output_top_k=100,
        return_provenance=True,
        primary_11_12_micro_coverage=True,
    )
    prim_slots = [p for p in prov if p["slot_source"] == "TIER5_PRIMARY_MICRO_COVERAGE"]
    assert len(prim_slots) == 1
    assert prim_slots[0]["video_id"] == "V012"


def test_phase_a_prefix_unchanged_and_chains_with_secondary_micro_budget():
    """Invariant 8: Phase A prefix is unchanged; Primary 11-12 and Secondary MB chain cleanly."""
    cands = _build_synthetic_candidates(num_videos=20, seeds_per_video=2)
    
    preds, prov = construct_ranked_qa_top100(
        query_id="TEST-Q1",
        scored_candidates=cands,
        output_top_k=100,
        return_provenance=True,
        secondary_temporal_micro_budget=True,
        primary_11_12_micro_coverage=True,
    )
    
    # Check that primary 11-12 slots appear immediately before secondary micro-budget slots
    prim_indices = [idx for idx, p in enumerate(prov) if p["slot_source"] == "TIER5_PRIMARY_MICRO_COVERAGE"]
    sec_indices = [idx for idx, p in enumerate(prov) if p["slot_source"] == "TIER5_SECONDARY_MICRO_OFFSET"]
    
    assert len(prim_indices) == 2
    assert len(sec_indices) == 20
    assert max(prim_indices) < min(sec_indices), "Primary 11-12 micro-coverage must precede secondary micro-budget"


def test_target_k_bounds_strictly_respected():
    """Invariant 9: target_k <= 100 is strictly respected."""
    cands = _build_synthetic_candidates(num_videos=20, seeds_per_video=2)
    preds = construct_ranked_qa_top100(
        query_id="TEST-Q1",
        scored_candidates=cands,
        output_top_k=75,
        primary_11_12_micro_coverage=True,
    )
    assert len(preds) == 75
    assert preds[-1].rank == 75
