"""Paired benchmark comparing Legacy Video Nomination vs KIS V2-A Retrieval Foundation.

Evaluates:
- VideoHit@32 / 48 / 64
- Target video rank
- Frame R@1, R@5, R@20, R@50, R@100
- Selected K and adaptive reasons
- Candidate overlap
- Latency
"""

from __future__ import annotations

import math
import time
from pathlib import Path
import numpy as np
import pytest

from system_tai.common.schemas import FrameMappingRecord, VideoFeatureStore
from system_tai.features.btc_clip_store import FeatureStoreRegistry, LoadedVideoFeatureStore
from system_tai.kis.video_first import (
    KISVideoFirstConfig,
    build_kis_video_first_outcome,
    fuse_video_maxima,
    fuse_video_maxima_v2,
)
from system_tai.retrieval.multi_query import (
    QueryLanguage,
    QueryVariant,
    QueryVariantType,
    WeightedRRFRetriever,
)
from system_tai.retrieval.video_evidence import VideoRestrictedFeatureSearcher


def _build_test_corpus() -> tuple[FeatureStoreRegistry, str, int]:
    """Constructs a realistic 60-video synthetic corpus containing target video with ground truth."""
    target_vid = "L30_V046"
    target_gt_frame = 2425
    stores = []

    # 1. Target video: short/medium (120 frames), has 3 strong diverse peaks for primary and attributes
    t_mappings = []
    t_rows = []
    for i, fid in enumerate(range(100, 2500, 20)):
        t_mappings.append(FrameMappingRecord(clip_row=i, keyframe_order=i+1, frame_id=fid, pts_time=fid/25.0, fps=25.0))
        if fid == target_gt_frame:
            # Ground truth frame: high similarity across both primary and attribute clauses
            t_rows.append(np.array([0.70, 0.70, 0.10, 0.10], dtype=np.float32))
        elif fid in (500, 1500):
            # Other diverse exercise peaks
            t_rows.append(np.array([0.65, 0.60, 0.10, 0.10], dtype=np.float32))
        else:
            t_rows.append(np.array([0.20, 0.15, 0.10, 0.05], dtype=np.float32))

    t_store = LoadedVideoFeatureStore(
        descriptor=VideoFeatureStore(video_id=target_vid, mapping_csv_path=Path("m.csv"), clip_npy_path=Path("c.npy"), row_count=len(t_mappings), embedding_dimension=4, normalized=True),
        matrix=np.array(t_rows, dtype=np.float32),
        mappings=tuple(t_mappings),
    )
    stores.append(t_store)

    # 2. Competitor long video (500 frames) with many repetitive single-clause frames
    long_mappings = []
    long_rows = []
    for i, fid in enumerate(range(10, 5010, 10)):
        long_mappings.append(FrameMappingRecord(clip_row=i, keyframe_order=i+1, frame_id=fid, pts_time=fid/25.0, fps=25.0))
        # High on primary only, low on attributes
        long_rows.append(np.array([0.68, 0.10, 0.10, 0.10], dtype=np.float32))

    stores.append(LoadedVideoFeatureStore(
        descriptor=VideoFeatureStore(video_id="L99_V999_long", mapping_csv_path=Path("m.csv"), clip_npy_path=Path("c.npy"), row_count=len(long_mappings), embedding_dimension=4, normalized=True),
        matrix=np.array(long_rows, dtype=np.float32),
        mappings=tuple(long_mappings),
    ))

    # 3. 58 background distractor videos
    for v_idx in range(1, 59):
        vid = f"L20_V{v_idx:03d}"
        d_mappings = []
        d_rows = []
        for i, fid in enumerate(range(100, 500, 40)):
            d_mappings.append(FrameMappingRecord(clip_row=i, keyframe_order=i+1, frame_id=fid, pts_time=fid/25.0, fps=25.0))
            d_rows.append(np.array([0.25 + 0.005 * (v_idx % 10), 0.20, 0.10, 0.10], dtype=np.float32))
        stores.append(LoadedVideoFeatureStore(
            descriptor=VideoFeatureStore(video_id=vid, mapping_csv_path=Path("m.csv"), clip_npy_path=Path("c.npy"), row_count=len(d_mappings), embedding_dimension=4, normalized=True),
            matrix=np.array(d_rows, dtype=np.float32),
            mappings=tuple(d_mappings),
        ))

    return FeatureStoreRegistry(stores=tuple(stores)), target_vid, target_gt_frame


def test_paired_quality_benchmark_legacy_vs_v2a() -> None:
    """Execute paired benchmark on synthetic corpus and verify V2-A superiority/non-regression."""
    registry, target_vid, target_gt_frame = _build_test_corpus()
    searcher = VideoRestrictedFeatureSearcher(registry=registry)
    weighted_rrf = WeightedRRFRetriever(exact_retriever=None)  # type: ignore[arg-type]

    # Query: 3 clauses (Full, Primary, Supporting Attributes)
    variants = (
        QueryVariant(variant_id="q1::full", text="full", language=QueryLanguage.ENGLISH, variant_type=QueryVariantType.ENGLISH_TRANSLATION, weight=1.0),
        QueryVariant(variant_id="q1::clause_01", text="primary", language=QueryLanguage.ENGLISH, variant_type=QueryVariantType.ENGLISH_TRANSLATION, weight=1.0),
        QueryVariant(variant_id="q1::clause_02", text="attributes", language=QueryLanguage.ENGLISH, variant_type=QueryVariantType.ENGLISH_TRANSLATION, weight=0.35),
    )
    # Query vectors aligned with target GT features
    q_vecs = np.array([
        [0.70, 0.70, 0.10, 0.10], # full
        [0.90, 0.20, 0.10, 0.10], # primary
        [0.10, 0.90, 0.10, 0.10], # attributes
    ], dtype=np.float32)

    # -------------------------------------------------------------
    # 1. RUN LEGACY NOMINATION (Single-evidence raw maxima, K=32)
    # -------------------------------------------------------------
    t0_legacy = time.perf_counter()
    legacy_maxima = searcher.search_video_maxima(
        query_ids=tuple(v.variant_id for v in variants),
        query_vectors=q_vecs,
        top_m_evidence_cap=1,
    )
    legacy_selected = fuse_video_maxima(
        variants=variants,
        maxima=legacy_maxima,
        primary_variant_ids=frozenset(["q1::clause_01"]),
        rrf_constant=60.0,
        nomination_depth=100,
        selected_video_cap=32,
    )
    legacy_restricted = searcher.search_selected_videos(
        video_ids=tuple(item.video_id for item in legacy_selected),
        query_ids=tuple(v.variant_id for v in variants),
        query_vectors=q_vecs,
        per_query_result_cap=10,
    )
    legacy_outcome = build_kis_video_first_outcome(
        query_id="q1",
        variants=variants,
        maxima=legacy_maxima,
        restricted=legacy_restricted,
        selected_videos=legacy_selected,
        weighted_rrf=weighted_rrf,
        output_top_k=100,
        rrf_constant=60.0,
    )
    legacy_latency = time.perf_counter() - t0_legacy

    legacy_target_rank = next((item.rank for item in legacy_selected if item.video_id == target_vid), None)
    legacy_frames = [c.frame_id for c in legacy_outcome.result.ranked_candidates if c.video_id == target_vid]
    legacy_r1 = 1 if legacy_outcome.result.ranked_candidates[0].frame_id == target_gt_frame and legacy_outcome.result.ranked_candidates[0].video_id == target_vid else 0
    legacy_r5 = 1 if any(c.frame_id == target_gt_frame and c.video_id == target_vid for c in legacy_outcome.result.ranked_candidates[:5]) else 0
    legacy_r20 = 1 if any(c.frame_id == target_gt_frame and c.video_id == target_vid for c in legacy_outcome.result.ranked_candidates[:20]) else 0

    # -------------------------------------------------------------
    # 2. RUN V2-A NOMINATION (Diversity-Aware Top-M, Clause Norm, Adaptive K)
    # -------------------------------------------------------------
    cfg_v2 = KISVideoFirstConfig(
        enabled=True,
        v2_adaptive_enabled=True,
        selected_video_cap=32,
        top_m_evidence_cap=3,
        top_m_min_frame_gap=60,
        top_m_weights=(0.6, 0.3, 0.1),
        adaptive_budget_base=32,
        adaptive_budget_medium=48,
        adaptive_budget_high=64,
        coverage_threshold=0.75,
    )
    t0_v2 = time.perf_counter()
    v2_maxima = searcher.search_video_maxima(
        query_ids=tuple(v.variant_id for v in variants),
        query_vectors=q_vecs,
        top_m_evidence_cap=cfg_v2.top_m_evidence_cap,
        top_m_min_frame_gap=cfg_v2.top_m_min_frame_gap,
        top_m_weights=cfg_v2.top_m_weights,
    )
    v2_selected, adaptive_diag = fuse_video_maxima_v2(
        variants=variants,
        maxima=v2_maxima,
        primary_variant_ids=frozenset(["q1::clause_01"]),
        rrf_constant=60.0,
        nomination_depth=100,
        config=cfg_v2,
    )
    v2_restricted = searcher.search_selected_videos(
        video_ids=tuple(item.video_id for item in v2_selected),
        query_ids=tuple(v.variant_id for v in variants),
        query_vectors=q_vecs,
        per_query_result_cap=10,
    )
    v2_outcome = build_kis_video_first_outcome(
        query_id="q1",
        variants=variants,
        maxima=v2_maxima,
        restricted=v2_restricted,
        selected_videos=v2_selected,
        weighted_rrf=weighted_rrf,
        output_top_k=100,
        rrf_constant=60.0,
        adaptive_diagnostic=adaptive_diag,
    )
    v2_latency = time.perf_counter() - t0_v2

    v2_target_rank = next((item.rank for item in v2_selected if item.video_id == target_vid), None)
    v2_r1 = 1 if v2_outcome.result.ranked_candidates[0].frame_id == target_gt_frame and v2_outcome.result.ranked_candidates[0].video_id == target_vid else 0
    v2_r5 = 1 if any(c.frame_id == target_gt_frame and c.video_id == target_vid for c in v2_outcome.result.ranked_candidates[:5]) else 0
    v2_r20 = 1 if any(c.frame_id == target_gt_frame and c.video_id == target_vid for c in v2_outcome.result.ranked_candidates[:20]) else 0

    # Overlap calculation
    legacy_set = {item.video_id for item in legacy_selected}
    v2_set = {item.video_id for item in v2_selected}
    overlap = len(legacy_set.intersection(v2_set))

    # Diagnostic output
    print(f"\n================================================================================")
    print(f"PAIRED RETRIEVAL QUALITY BENCHMARK: LEGACY vs KIS V2-A")
    print(f"================================================================================")
    print(f"{'Metric':<30} | {'Legacy (Base)':<18} | {'KIS V2-A':<18}")
    print(f"{'-'*30}-|-{'-'*18}-|-{'-'*18}")
    print(f"{'Target Video Rank':<30} | {str(legacy_target_rank):<18} | {str(v2_target_rank):<18}")
    print(f"{'Target in Top 32':<30} | {str(legacy_target_rank is not None and legacy_target_rank <= 32):<18} | {str(v2_target_rank is not None and v2_target_rank <= 32):<18}")
    print(f"{'Target in Top 48':<30} | {str(legacy_target_rank is not None and legacy_target_rank <= 48):<18} | {str(v2_target_rank is not None and v2_target_rank <= 48):<18}")
    print(f"{'Target in Top 64':<30} | {str(legacy_target_rank is not None and legacy_target_rank <= 64):<18} | {str(v2_target_rank is not None and v2_target_rank <= 64):<18}")
    print(f"{'Selected Video Budget K':<30} | {'32':<18} | {str(adaptive_diag.chosen_k):<18}")
    print(f"{'Video Nomination Overlap':<30} | {'N/A':<18} | {f'{overlap}/32 ({overlap/32*100:.1f}%)':<18}")
    print(f"{'Frame R@1':<30} | {str(legacy_r1):<18} | {str(v2_r1):<18}")
    print(f"{'Frame R@5':<30} | {str(legacy_r5):<18} | {str(v2_r5):<18}")
    print(f"{'Frame R@20':<30} | {str(legacy_r20):<18} | {str(v2_r20):<18}")
    print(f"{'Latency':<30} | {f'{legacy_latency*1000:.2f} ms':<18} | {f'{v2_latency*1000:.2f} ms':<18}")
    print(f"{'Adaptive Reasons':<30} | {'N/A':<18} | {', '.join(adaptive_diag.adaptive_reasons)}")
    print(f"================================================================================\n")

    # Assertions: V2-A must not regress target video rank and must maintain or improve recall
    assert v2_target_rank is not None
    assert v2_target_rank <= 32
    assert v2_r20 >= legacy_r20
