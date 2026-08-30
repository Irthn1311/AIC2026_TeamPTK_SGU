"""
Baseline Equivalence Test
=========================
Verifies that when all experimental feature flags are False (default state),
the system maintains exact equivalence to commit 5d9db1d.
"""

import pytest
from system_tai.kis.video_first import (
    KISVideoFirstConfig,
    fuse_restricted_frames,
    FusedVideoEvidence,
)
from system_tai.retrieval.video_evidence import (
    RestrictedFrameHit,
    VideoRestrictedSearchOutcome,
)
from system_tai.retrieval.semantic_query import (
    SemanticQueryConfig,
)
from system_tai.retrieval.multi_query import (
    QueryLanguage,
    QueryVariant,
    QueryVariantType,
    WeightedRRFRetriever,
)


def test_default_feature_flags_are_strictly_false():
    sq_cfg = SemanticQueryConfig()
    assert sq_cfg.enable_visual_content_filter is False

    vf_cfg = KISVideoFirstConfig()
    assert vf_cfg.enable_candidate_union is False
    assert vf_cfg.enable_score_normalization is False
    assert vf_cfg.enable_late_interaction is False
    assert vf_cfg.enable_positive_chain_bonus is False

    # Exact 5d9db1d hyperparameter defaults
    assert vf_cfg.video_nomination_depth == 100
    assert vf_cfg.restricted_frames_per_video_per_variant == 10
    assert vf_cfg.restricted_frame_min_gap == 0
    assert vf_cfg.top_m_evidence_cap == 3
    assert vf_cfg.top_m_min_frame_gap == 60
    assert vf_cfg.top_m_weights == (0.6, 0.3, 0.1)
    assert vf_cfg.coverage_threshold == 0.75
    assert vf_cfg.selected_video_cap == 32


def test_fuse_restricted_frames_equivalence():
    v1 = QueryVariant(variant_id="v1", text="query 1", language=QueryLanguage.ENGLISH, variant_type=QueryVariantType.ENGLISH_TRANSLATION, weight=1.0)
    v2 = QueryVariant(variant_id="v2", text="query 2", language=QueryLanguage.ENGLISH, variant_type=QueryVariantType.ENGLISH_TRANSLATION, weight=0.5)
    variants = (v1, v2)

    restricted_hits = {
        "v1": {
            "L30_V046": (
                RestrictedFrameHit(video_id="L30_V046", frame_id=2425, clip_row=0, keyframe_order=1, pts_time=97.0, cosine_score=0.45, rank=1),
                RestrictedFrameHit(video_id="L30_V046", frame_id=4865, clip_row=1, keyframe_order=2, pts_time=194.6, cosine_score=0.40, rank=2),
            )
        },
        "v2": {
            "L30_V046": (
                RestrictedFrameHit(video_id="L30_V046", frame_id=2425, clip_row=0, keyframe_order=1, pts_time=97.0, cosine_score=0.42, rank=1),
                RestrictedFrameHit(video_id="L30_V046", frame_id=4865, clip_row=1, keyframe_order=2, pts_time=194.6, cosine_score=0.38, rank=2),
            )
        },
    }

    restricted = VideoRestrictedSearchOutcome(
        rankings=restricted_hits,
        physical_rows_scored=4,
        video_store_scan_count=1,
    )

    selected_videos = (
        FusedVideoEvidence(
            video_id="L30_V046",
            rank=1,
            fusion_score=0.88,
            variant_hit_count=2,
            primary_coverage_count=2,
            best_individual_rank=1,
            per_variant=(),
        ),
    )

    weighted_rrf = WeightedRRFRetriever(object())

    result = fuse_restricted_frames(
        query_id="Q1",
        variants=variants,
        restricted=restricted,
        selected_videos=selected_videos,
        weighted_rrf=weighted_rrf,
        output_top_k=10,
        rrf_constant=60.0,
    )

    assert len(result.ranked_candidates) == 2
    # Because 2425 and 4865 are > 75 frames apart, both are retained in primary tier
    assert result.ranked_candidates[0].video_id == "L30_V046"
    assert result.ranked_candidates[0].frame_id == 2425
    assert result.ranked_candidates[1].frame_id == 4865
    assert "scores_by_variant" in result.ranked_candidates[0].diagnostic_metadata


def test_manual_reference_schema_and_tri_state_evaluator():
    import json
    from pathlib import Path
    
    ref_file = Path(__file__).resolve().parent.parent / "benchmarks" / "manual_kis_reference_v1.json"
    assert ref_file.exists(), "manual_kis_reference_v1.json must exist"
    
    data = json.loads(ref_file.read_text(encoding="utf-8"))
    assert data["schema_version"] == "1.0"
    queries = data["queries"]
    assert len(queries) == 5

    for q in queries:
        assert q["official_gt"] is None
        assert q["annotation_status"] == "VIDEO_ONLY_VERIFIED"
        assert q["human_annotated_intervals"] == []
        assert isinstance(q["human_verified_video_id"], str)
        assert len(q["human_verified_video_id"]) > 0
