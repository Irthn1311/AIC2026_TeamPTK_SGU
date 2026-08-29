"""Comprehensive invariant tests for KIS V2-A Retrieval Foundation.

Validates the 5 essential invariants:
1. Length Invariance (adding low-scoring frames does not alter video score)
2. Duplicate Invariance (temporal neighborhood duplicates are suppressed)
3. Permutation Invariance (frame ordering does not alter score)
4. Clause-Scale Invariance (clause-local normalization prevents generic dominance)
5. Adaptive-K Monotonicity (flatter score distributions never decrease K)
"""

from __future__ import annotations

import math
import pytest

from system_tai.kis.video_first import (
    KISVideoFirstConfig,
    compute_adaptive_video_budget_v2,
    normalize_clause_scores,
    fuse_video_maxima_v2,
    VariantVideoEvidence,
    FusedVideoEvidence,
)
from system_tai.retrieval.multi_query import QueryLanguage, QueryVariant, QueryVariantType
from system_tai.retrieval.video_evidence import FullCorpusVideoMaximaOutcome, VideoMaximumHit


def test_invariant_1_length_invariance() -> None:
    """Appending low-scoring frames must not increase or alter top-M evidence score."""
    from pathlib import Path
    from system_tai.common.schemas import FrameMappingRecord, VideoFeatureStore
    from system_tai.features.btc_clip_store import FeatureStoreRegistry, LoadedVideoFeatureStore
    from system_tai.retrieval.video_evidence import VideoRestrictedFeatureSearcher
    import numpy as np

    base_mappings = (
        FrameMappingRecord(clip_row=0, keyframe_order=1, frame_id=100, pts_time=4.0, fps=25.0),
        FrameMappingRecord(clip_row=1, keyframe_order=2, frame_id=300, pts_time=12.0, fps=25.0),
        FrameMappingRecord(clip_row=2, keyframe_order=3, frame_id=500, pts_time=20.0, fps=25.0),
    )
    base_matrix = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.8, 0.0, 0.0, 0.0],
        [0.6, 0.0, 0.0, 0.0],
    ], dtype=np.float32)

    store_a = LoadedVideoFeatureStore(
        descriptor=VideoFeatureStore(video_id="video_short", mapping_csv_path=Path("m.csv"), clip_npy_path=Path("c.npy"), row_count=3, embedding_dimension=4, normalized=True),
        matrix=base_matrix,
        mappings=base_mappings,
    )

    long_mappings = list(base_mappings)
    long_matrix_rows = [base_matrix[0], base_matrix[1], base_matrix[2]]
    for fid in range(1000, 1100):
        long_mappings.append(FrameMappingRecord(clip_row=len(long_mappings), keyframe_order=len(long_mappings), frame_id=fid, pts_time=float(fid)/25.0, fps=25.0))
        long_matrix_rows.append(np.array([0.05, 0.0, 0.0, 0.0], dtype=np.float32))

    store_b = LoadedVideoFeatureStore(
        descriptor=VideoFeatureStore(video_id="video_long", mapping_csv_path=Path("m.csv"), clip_npy_path=Path("c.npy"), row_count=len(long_mappings), embedding_dimension=4, normalized=True),
        matrix=np.array(long_matrix_rows, dtype=np.float32),
        mappings=tuple(long_mappings),
    )

    registry = FeatureStoreRegistry(stores=(store_a, store_b))
    searcher = VideoRestrictedFeatureSearcher(registry=registry)

    q_vec = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    maxima = searcher.search_video_maxima(
        query_ids=("q1",),
        query_vectors=q_vec,
        top_m_evidence_cap=3,
        top_m_min_frame_gap=60,
        top_m_weights=(0.6, 0.3, 0.1),
    )

    score_short = next(h for h in maxima.rankings["q1"] if h.video_id == "video_short").top_m_score
    score_long = next(h for h in maxima.rankings["q1"] if h.video_id == "video_long").top_m_score

    assert math.isclose(score_short, score_long, rel_tol=1e-5)


def test_invariant_2_duplicate_invariance() -> None:
    """Duplicate frames in the same temporal neighborhood (gap < 60) must be suppressed."""
    from pathlib import Path
    from system_tai.common.schemas import FrameMappingRecord, VideoFeatureStore
    from system_tai.features.btc_clip_store import FeatureStoreRegistry, LoadedVideoFeatureStore
    from system_tai.retrieval.video_evidence import VideoRestrictedFeatureSearcher
    import numpy as np

    dup_mappings = (
        FrameMappingRecord(clip_row=0, keyframe_order=1, frame_id=100, pts_time=4.0, fps=25.0),
        FrameMappingRecord(clip_row=1, keyframe_order=2, frame_id=102, pts_time=4.08, fps=25.0),
        FrameMappingRecord(clip_row=2, keyframe_order=3, frame_id=105, pts_time=4.20, fps=25.0),
    )
    dup_matrix = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.99, 0.0, 0.0, 0.0],
        [0.98, 0.0, 0.0, 0.0],
    ], dtype=np.float32)

    store = LoadedVideoFeatureStore(
        descriptor=VideoFeatureStore(video_id="video_dup", mapping_csv_path=Path("m.csv"), clip_npy_path=Path("c.npy"), row_count=3, embedding_dimension=4, normalized=True),
        matrix=dup_matrix,
        mappings=dup_mappings,
    )

    registry = FeatureStoreRegistry(stores=(store,))
    searcher = VideoRestrictedFeatureSearcher(registry=registry)

    q_vec = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    maxima = searcher.search_video_maxima(
        query_ids=("q1",),
        query_vectors=q_vec,
        top_m_evidence_cap=3,
        top_m_min_frame_gap=60,
        top_m_weights=(0.6, 0.3, 0.1),
    )

    hit = maxima.rankings["q1"][0]
    assert math.isclose(hit.top_m_score, 1.0, rel_tol=1e-5)


def test_invariant_3_permutation_invariance() -> None:
    """Frame order in matrix must not change the Top-M evidence score."""
    from pathlib import Path
    from system_tai.common.schemas import FrameMappingRecord, VideoFeatureStore
    from system_tai.features.btc_clip_store import FeatureStoreRegistry, LoadedVideoFeatureStore
    from system_tai.retrieval.video_evidence import VideoRestrictedFeatureSearcher
    import numpy as np

    mappings_fwd = (
        FrameMappingRecord(clip_row=0, keyframe_order=1, frame_id=100, pts_time=4.0, fps=25.0),
        FrameMappingRecord(clip_row=1, keyframe_order=2, frame_id=300, pts_time=12.0, fps=25.0),
        FrameMappingRecord(clip_row=2, keyframe_order=3, frame_id=500, pts_time=20.0, fps=25.0),
    )
    mat_fwd = np.array([[0.9, 0, 0, 0], [0.7, 0, 0, 0], [0.5, 0, 0, 0]], dtype=np.float32)

    mappings_rev = (
        FrameMappingRecord(clip_row=0, keyframe_order=1, frame_id=500, pts_time=20.0, fps=25.0),
        FrameMappingRecord(clip_row=1, keyframe_order=2, frame_id=300, pts_time=12.0, fps=25.0),
        FrameMappingRecord(clip_row=2, keyframe_order=3, frame_id=100, pts_time=4.0, fps=25.0),
    )
    mat_rev = np.array([[0.5, 0, 0, 0], [0.7, 0, 0, 0], [0.9, 0, 0, 0]], dtype=np.float32)

    store_fwd = LoadedVideoFeatureStore(descriptor=VideoFeatureStore("v_fwd", Path("m.csv"), Path("c.npy"), 3, 4, True), matrix=mat_fwd, mappings=mappings_fwd)
    store_rev = LoadedVideoFeatureStore(descriptor=VideoFeatureStore("v_rev", Path("m.csv"), Path("c.npy"), 3, 4, True), matrix=mat_rev, mappings=mappings_rev)

    searcher = VideoRestrictedFeatureSearcher(FeatureStoreRegistry(stores=(store_fwd, store_rev)))
    q_vec = np.array([[1.0, 0, 0, 0]], dtype=np.float32)
    maxima = searcher.search_video_maxima(
        query_ids=("q1",),
        query_vectors=q_vec,
        top_m_evidence_cap=3,
        top_m_min_frame_gap=60,
        top_m_weights=(0.6, 0.3, 0.1),
    )

    score_fwd = next(h for h in maxima.rankings["q1"] if h.video_id == "v_fwd").top_m_score
    score_rev = next(h for h in maxima.rankings["q1"] if h.video_id == "v_rev").top_m_score
    assert math.isclose(score_fwd, score_rev, rel_tol=1e-5)


def test_invariant_4_clause_scale_invariance() -> None:
    """Clause-local percentile normalization maps raw cosine distribution into (0, 1]."""
    raw_a = {"v1": 0.40, "v2": 0.38, "v3": 0.35}
    raw_b = {"v2": 0.25, "v1": 0.22, "v3": 0.20}

    norm_a = normalize_clause_scores(raw_a)
    norm_b = normalize_clause_scores(raw_b)

    assert math.isclose(norm_a["v1"], 1.0)
    assert math.isclose(norm_a["v2"], 2.0 / 3.0)
    assert math.isclose(norm_a["v3"], 1.0 / 3.0)

    assert math.isclose(norm_b["v2"], 1.0)
    assert math.isclose(norm_b["v1"], 2.0 / 3.0)
    assert math.isclose(norm_b["v3"], 1.0 / 3.0)


def test_invariant_5_adaptive_k_monotonicity() -> None:
    """Flatter score distributions must never decrease nominated K (always in {32, 48, 64})."""
    confident_scores = [1.0] + [0.3 - 0.005 * i for i in range(31)]
    k_conf, diag_conf = compute_adaptive_video_budget_v2(confident_scores, clause_count=2, has_attributes=False)
    assert k_conf == 32
    assert diag_conf.chosen_k == 32

    flat_scores = [0.500 - 0.0001 * i for i in range(32)]
    k_flat, diag_flat = compute_adaptive_video_budget_v2(flat_scores, clause_count=2, has_attributes=False)
    assert k_flat == 64
    assert diag_flat.chosen_k == 64
    assert diag_flat.is_flat is True

    k_complex, diag_complex = compute_adaptive_video_budget_v2(confident_scores, clause_count=5, has_attributes=True)
    assert k_complex >= 48
    assert k_complex in {48, 64}

    assert k_flat >= k_conf


def test_invariant_6_compulsory_dp_frame_preservation() -> None:
    """Winning DP chain frames outside per-query cap must be preserved and boosted in restricted search."""
    from pathlib import Path
    from system_tai.common.schemas import FrameMappingRecord, VideoFeatureStore
    from system_tai.features.btc_clip_store import FeatureStoreRegistry, LoadedVideoFeatureStore
    from system_tai.retrieval.video_evidence import VideoRestrictedFeatureSearcher
    import numpy as np

    mappings = tuple(
        FrameMappingRecord(clip_row=i, keyframe_order=i+1, frame_id=i*100, pts_time=float(i)*4.0, fps=25.0)
        for i in range(20)
    )
    matrix = np.zeros((20, 4), dtype=np.float32)
    for i in range(20):
        matrix[i, 0] = 1.0 - (0.04 * i)
    matrix[:, 1:] = 0.0

    store = LoadedVideoFeatureStore(
        descriptor=VideoFeatureStore(video_id="video_chain", mapping_csv_path=Path("m.csv"), clip_npy_path=Path("c.npy"), row_count=20, embedding_dimension=4, normalized=True),
        matrix=matrix,
        mappings=mappings,
    )
    registry = FeatureStoreRegistry(stores=(store,))
    searcher = VideoRestrictedFeatureSearcher(registry=registry)

    outcome = searcher.search_selected_videos(
        video_ids=("video_chain",),
        query_ids=("q1",),
        query_vectors=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        per_query_result_cap=5,
        compulsory_frame_ids_by_video={"video_chain": [1500]},
    )

    hits = outcome.rankings["q1"]["video_chain"]
    retained_fids = [h.frame_id for h in hits]
    assert 1500 in retained_fids
    assert len(retained_fids) == 6


def test_invariant_7_temporal_frame_gap_diversity() -> None:
    """Restricted frame fusion enforces minimum frame gap between same-video candidates unless chain winner."""
    from system_tai.common.schemas import KISResult
    from system_tai.kis.video_first import (
        FusedVideoEvidence,
        TemporalChainDiagnostic,
        fuse_restricted_frames,
    )
    from system_tai.retrieval.multi_query import QueryVariant, QueryVariantType, QueryLanguage, WeightedRRFRetriever
    from system_tai.retrieval.video_evidence import VideoRestrictedSearchOutcome, RestrictedFrameHit

    var = QueryVariant(variant_id="v1", weight=1.0, variant_type=QueryVariantType.VIETNAMESE_DIRECT, language=QueryLanguage.VIETNAMESE, text="query")
    hits = (
        RestrictedFrameHit(video_id="vid_1", frame_id=100, clip_row=0, keyframe_order=1, pts_time=4.0, cosine_score=0.90, rank=1),
        RestrictedFrameHit(video_id="vid_1", frame_id=110, clip_row=1, keyframe_order=2, pts_time=4.4, cosine_score=0.88, rank=2),
        RestrictedFrameHit(video_id="vid_1", frame_id=120, clip_row=2, keyframe_order=3, pts_time=4.8, cosine_score=0.85, rank=3),
        RestrictedFrameHit(video_id="vid_1", frame_id=800, clip_row=4, keyframe_order=5, pts_time=32.0, cosine_score=0.70, rank=4),
    )
    restricted = VideoRestrictedSearchOutcome(
        rankings={"v1": {"vid_1": hits}},
        physical_rows_scored=4,
        video_store_scan_count=1,
    )
    diag = TemporalChainDiagnostic(
        is_temporal_compound=True,
        temporal_scene_count=2,
        has_valid_chain=True,
        selected_chain_frames=(100, 800),
        chain_score=0.80,
        soft_and_score=0.80,
        balance_ratio=1.0,
        temporal_multiplier=1.35,
    )
    selected_videos = (
        FusedVideoEvidence(
            video_id="vid_1",
            rank=1,
            fusion_score=0.95,
            variant_hit_count=1,
            primary_coverage_count=1,
            best_individual_rank=1,
            per_variant=(),
            temporal_chain=diag,
        ),
    )
    res = fuse_restricted_frames(
        query_id="q1",
        variants=(var,),
        restricted=restricted,
        selected_videos=selected_videos,
        weighted_rrf=WeightedRRFRetriever(object()),
        output_top_k=10,
        rrf_constant=60.0,
        temporal_chain_bonus=0.05,
        min_frame_gap=50,
    )
    fids = [c.frame_id for c in res.ranked_candidates]
    assert 100 in fids
    assert 800 in fids
    assert 110 not in fids
    assert 120 not in fids
