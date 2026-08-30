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
        assert q["annotation_status"] in ("VIDEO_ONLY_VERIFIED", "FRAME_INTERVAL_VERIFIED")
        assert isinstance(q["human_verified_video_id"], str)
        assert len(q["human_verified_video_id"]) > 0

    p1_1 = next(q for q in queries if "p1-1" in q["query_id"])
    assert p1_1["annotation_status"] == "FRAME_INTERVAL_VERIFIED"
    assert p1_1["human_annotated_intervals"] == [[6600, 6850]]


def test_runner_py_compile_syntax():
    import py_compile
    from pathlib import Path

    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    runner_script = repo_root / "scratch" / "run_kaggle_v2a_causal_closure.py"
    assert runner_script.exists(), "run_kaggle_v2a_causal_closure.py must exist"

    # Must compile cleanly without SyntaxError, IndentationError, or TabError
    py_compile.compile(str(runner_script), doraise=True)


def test_pts_time_and_variant_scores_preservation_through_rrf():
    """Verify that pts_time and scores_by_variant are strictly preserved across fusion and serialization."""
    from system_tai.common.schemas import CandidateFrame, KISResult

    v1 = QueryVariant(variant_id="vi_primary", text="nón bảo hiểm", language=QueryLanguage.VIETNAMESE, variant_type=QueryVariantType.VIETNAMESE_DIRECT, weight=1.0)
    v2 = QueryVariant(variant_id="en_translated", text="helmet", language=QueryLanguage.ENGLISH, variant_type=QueryVariantType.ENGLISH_TRANSLATION, weight=0.5)

    hit1 = CandidateFrame(video_id="L30_V046", frame_id=2425, clip_row=0, keyframe_order=5, score=0.45, rank=1, source="clip", diagnostic_metadata={"pts_time": 97.123})
    hit2 = CandidateFrame(video_id="L30_V046", frame_id=2425, clip_row=0, keyframe_order=5, score=0.42, rank=1, source="clip", diagnostic_metadata={"pts_time": 97.123})

    rrf = WeightedRRFRetriever(object())
    fused_res = rrf.fuse_rankings(
        query_id="Q-TEST",
        variants=(v1, v2),
        rankings={
            "vi_primary": KISResult(query_id="vi_primary", ranked_candidates=(hit1,)),
            "en_translated": KISResult(query_id="en_translated", ranked_candidates=(hit2,)),
        },
        output_top_k=10,
    )

    top_cand = fused_res.ranked_candidates[0]
    assert top_cand.diagnostic_metadata.get("pts_time") == 97.123, "pts_time must be preserved through WeightedRRFRetriever.fuse_rankings"

    # Now verify fuse_restricted_frames preserves pts_time and produces scores_by_variant
    restricted = VideoRestrictedSearchOutcome(
        rankings={
            "vi_primary": {"L30_V046": (RestrictedFrameHit(video_id="L30_V046", frame_id=2425, clip_row=0, keyframe_order=5, pts_time=97.123, cosine_score=0.45, rank=1),)},
            "en_translated": {"L30_V046": (RestrictedFrameHit(video_id="L30_V046", frame_id=2425, clip_row=0, keyframe_order=5, pts_time=97.123, cosine_score=0.42, rank=1),)},
        },
        physical_rows_scored=1,
        video_store_scan_count=1,
    )
    sel_vids = (FusedVideoEvidence(video_id="L30_V046", rank=1, fusion_score=0.9, variant_hit_count=2, primary_coverage_count=2, best_individual_rank=1, per_variant=()),)

    final_res = fuse_restricted_frames(
        query_id="Q-TEST",
        variants=(v1, v2),
        restricted=restricted,
        selected_videos=sel_vids,
        weighted_rrf=rrf,
        output_top_k=10,
        rrf_constant=60.0,
    )

    final_cand = final_res.ranked_candidates[0]
    diag = final_cand.diagnostic_metadata
    assert diag.get("pts_time") == 97.123, "pts_time must be preserved in final CandidateFrame"
    assert "scores_by_variant" in diag
    assert diag["scores_by_variant"] == {"vi_primary": 0.45, "en_translated": 0.42}

