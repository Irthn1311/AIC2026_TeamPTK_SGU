"""Unit test suite for Phase C Paraphrase Ensemble and Invariant Fusion."""

import math
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pytest
from system_tai.retrieval.multi_query import QueryLanguage, QueryVariant, QueryVariantType
from system_tai.retrieval.semantic_query import (
    CompiledParaphraseGroup,
    CompiledParaphraseEnsemble,
    CompiledSemanticQuery,
    CompiledSemanticVariant,
    SemanticQueryConfig,
    SemanticUnitRole,
    VietnameseSemanticUnit,
    allocate_hierarchical_quotas,
    compute_normalized_ensemble_weights,
)
from system_tai.translation.paraphrase_sidecar_provider import (
    ImmutableParaphraseEnsembleSidecarProvider,
    canonical_sidecar_sha256,
)


def _make_mock_compiled_query(
    query_id: str,
    group_id: str,
    n_variants: int,
    raw_weights: list[float] | None = None,
) -> CompiledSemanticQuery:
    if raw_weights is None:
        raw_weights = [1.0] * n_variants
    assert len(raw_weights) == n_variants

    units = [
        VietnameseSemanticUnit(
            unit_id=f"{query_id}::{group_id}::unit_{i:02d}",
            text=f"unit text {i}",
            role=SemanticUnitRole.PRIMARY_SCENE if i == 0 else SemanticUnitRole.SUPPORTING_ATTRIBUTE,
            weight=raw_weights[i],
            temporal_index=None,
        )
        for i in range(n_variants)
    ]
    variants = [
        CompiledSemanticVariant(
            query_variant=QueryVariant(
                variant_id=f"{query_id}::{group_id}::semantic_{i:02d}_s01",
                text=f"segment text {i}",
                language=QueryLanguage.ENGLISH,
                variant_type=QueryVariantType.ENGLISH_TRANSLATION,
                weight=raw_weights[i],
            ),
            semantic_unit_id=units[i].unit_id,
            semantic_role=units[i].role,
            source_vietnamese=units[i].text,
            raw_english=f"segment text {i}",
            segment_index=1,
            segment_count=1,
            clip_token_count=10,
            temporal_index=None,
        )
        for i in range(n_variants)
    ]
    return CompiledSemanticQuery(
        query_id=query_id,
        source_vietnamese="mock vi",
        units=tuple(units),
        variants=tuple(variants),
        provider_name="mock_provider",
    )


def test_deterministic_divmod_hierarchical_quota_allocation():
    """Verify hierarchical divmod quota allocation preserves exact sum B_sem and stable deterministic order."""
    g0 = _make_mock_compiled_query("q1", "g0", 3)
    g1 = _make_mock_compiled_query("q1", "g1", 2)
    groups = [g0, g1]

    # Test B=20*3=60 across 2 groups:
    # Level 1: divmod(60, 2) -> q_grp=30, r_grp=0 -> B0=30, B1=30
    # Level 2: G0 has 3 vars -> divmod(30, 3) -> 10, 10, 10
    #          G1 has 2 vars -> divmod(30, 2) -> 15, 15
    quotas = allocate_hierarchical_quotas(groups, total_semantic_budget=60)
    assert sum(quotas.values()) == 60
    assert quotas[g0.variants[0].query_variant.variant_id] == 10
    assert quotas[g0.variants[1].query_variant.variant_id] == 10
    assert quotas[g0.variants[2].query_variant.variant_id] == 10
    assert quotas[g1.variants[0].query_variant.variant_id] == 15
    assert quotas[g1.variants[1].query_variant.variant_id] == 15

    # Test with remainders at both levels: B=37
    # Level 1: divmod(37, 2) -> q=18, r=1 -> B0=19, B1=18
    # Level 2: G0: divmod(19, 3) -> q=6, r=1 -> 7, 6, 6
    #          G1: divmod(18, 2) -> q=9, r=0 -> 9, 9
    quotas_rem = allocate_hierarchical_quotas(groups, total_semantic_budget=37)
    assert sum(quotas_rem.values()) == 37
    assert quotas_rem[g0.variants[0].query_variant.variant_id] == 7
    assert quotas_rem[g0.variants[1].query_variant.variant_id] == 6
    assert quotas_rem[g0.variants[2].query_variant.variant_id] == 6
    assert quotas_rem[g1.variants[0].query_variant.variant_id] == 9
    assert quotas_rem[g1.variants[1].query_variant.variant_id] == 9


def test_insufficient_budget_raises_fail_fast():
    """Verify ValueError is raised if B_sem < sum(|V_i|)."""
    g0 = _make_mock_compiled_query("q1", "g0", 3)
    g1 = _make_mock_compiled_query("q1", "g1", 4)
    # Total variants = 7. Budget 6 must raise.
    with pytest.raises(ValueError, match="Insufficient semantic budget"):
        allocate_hierarchical_quotas([g0, g1], total_semantic_budget=6)


def test_weight_mass_conservation():
    """Verify weight mass conservation: each group has mass W_0/N, and total ensemble mass equals W_0."""
    g0 = _make_mock_compiled_query("q1", "g0", 3, [1.0, 1.0, 0.35])
    g1 = _make_mock_compiled_query("q1", "g1", 2, [1.0, 0.5])
    g2 = _make_mock_compiled_query("q1", "g2", 1, [1.0])
    groups = [g0, g1, g2]

    w0 = 2.35  # baseline mass
    norm_weights = compute_normalized_ensemble_weights(groups, baseline_weight_mass=w0)

    # 1. Total sum must equal W_0
    assert abs(sum(norm_weights.values()) - w0) < 1e-9

    # 2. Each group must have mass W_0 / 3
    target_group_mass = w0 / 3.0
    for g in groups:
        grp_mass = sum(norm_weights[v.query_variant.variant_id] for v in g.variants)
        assert abs(grp_mass - target_group_mass) < 1e-9

    # 3. Relative intra-group proportions must be exact
    # In G0: ratio v0:v1:v2 is 1.0:1.0:0.35
    w_g0_0 = norm_weights[g0.variants[0].query_variant.variant_id]
    w_g0_1 = norm_weights[g0.variants[1].query_variant.variant_id]
    w_g0_2 = norm_weights[g0.variants[2].query_variant.variant_id]
    assert abs(w_g0_0 - w_g0_1) < 1e-9
    assert abs((w_g0_0 / w_g0_2) - (1.0 / 0.35)) < 1e-9


def test_paraphrase_sidecar_cross_group_difference_allowed():
    """Verify same Vietnamese text mapping to different English translations across different groups is allowed."""
    import json
    import tempfile
    sidecar_data = {
        "$schema_version": "1.0.0",
        "sidecar_id": "test-cross-group-sidecar",
        "target_queries_count": 1,
        "queries": {
            "p1-5": {
                "query_vi": "món ăn ngon",
                "query_vi_sha256": "419f118a4d72382cddaa74144b24b55b823831b896c778806153df34931acaac",
                "paraphrase_groups": [
                    {
                        "group_id": "group_a",
                        "units": [
                            {"vi_text": "món ăn ngon", "en_text": "delicious dish"}
                        ]
                    },
                    {
                        "group_id": "group_b",
                        "units": [
                            {"vi_text": "món ăn ngon", "en_text": "tasty food"}
                        ]
                    }
                ]
            }
        }
    }
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "cross_group.json"
        p.write_text(json.dumps(sidecar_data), encoding="utf-8")

        provider = ImmutableParaphraseEnsembleSidecarProvider(p)
        assert provider.translate_unit("p1-5", "group_a", "món ăn ngon") == "delicious dish"
        assert provider.translate_unit("p1-5", "group_b", "món ăn ngon") == "tasty food"


def test_paraphrase_sidecar_intra_group_collision_and_duplicate_group_raises():
    """Verify intra-group conflicting translation or duplicate group_id raises ValueError."""
    import json
    import tempfile
    # 1. Duplicate group_id
    dup_group = {
        "sidecar_id": "test-dup",
        "target_queries_count": 1,
        "queries": {
            "q1": {
                "query_vi": "test",
                "paraphrase_groups": [
                    {"group_id": "g1", "units": [{"vi_text": "a", "en_text": "b"}]},
                    {"group_id": "g1", "units": [{"vi_text": "a", "en_text": "c"}]},
                ]
            }
        }
    }
    with tempfile.TemporaryDirectory() as td:
        p1 = Path(td) / "dup_group.json"
        p1.write_text(json.dumps(dup_group), encoding="utf-8")
        with pytest.raises(ValueError, match="duplicate group_id"):
            ImmutableParaphraseEnsembleSidecarProvider(p1)

    # 2. Conflicting translation within same group
    conflict = {
        "sidecar_id": "test-conflict",
        "target_queries_count": 1,
        "queries": {
            "q1": {
                "query_vi": "test",
                "paraphrase_groups": [
                    {
                        "group_id": "g1",
                        "units": [
                            {"vi_text": "a", "en_text": "first translation"},
                            {"vi_text": "a", "en_text": "conflicting translation"},
                        ]
                    }
                ]
            }
        }
    }
    with tempfile.TemporaryDirectory() as td:
        p2 = Path(td) / "conflict.json"
        p2.write_text(json.dumps(conflict), encoding="utf-8")
        with pytest.raises(ValueError, match="Conflicting translation"):
            ImmutableParaphraseEnsembleSidecarProvider(p2)


def test_canonical_paraphrase_ensemble_sidecar_loads_and_validates():
    """Verify the production benchmark paraphrase sidecar loads cleanly and verifies integrity."""
    sidecar_path = Path("scratch/benchmarks/translation_ablation/paraphrase_ensemble_p1_focus_v1.json")
    assert sidecar_path.exists()

    expected_sha = "1bb2a15e7f55d9b1947552cdd33f5dba52b4316444781ff8d883aa359f163cf2"
    actual_sha = canonical_sidecar_sha256(sidecar_path)
    assert actual_sha == expected_sha

    provider = ImmutableParaphraseEnsembleSidecarProvider(sidecar_path, expected_sha)
    assert provider.sidecar_id == "paraphrase-ensemble-p1-focus-v1"

    # Negative controls have N=1 group
    for q in ["query-p1-1-kis", "query-p1-2-kis", "query-p1-4-kis", "query-p1-6-kis"]:
        groups = provider.get_paraphrase_groups(q)
        assert len(groups) == 1
        assert groups[0]["group_id"] == "group_canonical_new"

    # Treatment query P1-5 has N=2 groups
    p15_groups = provider.get_paraphrase_groups("query-p1-5-kis")
    assert len(p15_groups) == 2
    assert {g["group_id"] for g in p15_groups} == {"group_canonical_new", "group_candidate_old"}


def test_group_aware_video_nomination_and_chain_isolation():
    """Verify group-aware nomination computes independent group DP chains and fuses using 1/N RRF without cross-group temporal corruption."""
    from system_tai.kis.video_first import KISVideoFirstConfig, fuse_video_maxima_v2_paraphrase_ensemble
    from system_tai.retrieval.video_evidence import FullCorpusVideoMaximaOutcome, VideoMaximumHit

    # Create two groups: G0 (2 temporal scene variants), G1 (1 full scene variant)
    g0 = _make_mock_compiled_query("q1", "g0", 2, [1.0, 1.0])
    g1 = _make_mock_compiled_query("q1", "g1", 1, [1.0])

    # Mark G0 variants as temporal scenes
    u0_temp = VietnameseSemanticUnit(
        unit_id="q1::g0::unit_00",
        text="scene 1",
        role=SemanticUnitRole.TEMPORAL_SCENE,
        weight=1.0,
        temporal_index=1,
    )
    u1_temp = VietnameseSemanticUnit(
        unit_id="q1::g0::unit_01",
        text="scene 2",
        role=SemanticUnitRole.TEMPORAL_SCENE,
        weight=1.0,
        temporal_index=2,
    )
    v0 = CompiledSemanticVariant(
        query_variant=QueryVariant(variant_id="q1::g0::v0", text="scene 1 en", language=QueryLanguage.ENGLISH, variant_type=QueryVariantType.ENGLISH_TRANSLATION, weight=1.0),
        semantic_unit_id=u0_temp.unit_id,
        semantic_role=u0_temp.role,
        source_vietnamese=u0_temp.text,
        raw_english="scene 1 en",
        segment_index=1,
        segment_count=1,
        clip_token_count=5,
        temporal_index=1,
    )
    v1 = CompiledSemanticVariant(
        query_variant=QueryVariant(variant_id="q1::g0::v1", text="scene 2 en", language=QueryLanguage.ENGLISH, variant_type=QueryVariantType.ENGLISH_TRANSLATION, weight=1.0),
        semantic_unit_id=u1_temp.unit_id,
        semantic_role=u1_temp.role,
        source_vietnamese=u1_temp.text,
        raw_english="scene 2 en",
        segment_index=1,
        segment_count=1,
        clip_token_count=5,
        temporal_index=2,
    )
    g0_temporal = CompiledSemanticQuery(
        query_id="q1",
        source_vietnamese="scene 1 rồi scene 2",
        units=(u0_temp, u1_temp),
        variants=(v0, v1),
        provider_name="mock",
    )

    norm_w = {"q1::g0::v0": 1.0, "q1::g0::v1": 1.0, g1.variants[0].query_variant.variant_id: 1.0}
    ensemble = CompiledParaphraseEnsemble(
        query_id="q1",
        source_vietnamese="scene 1 rồi scene 2",
        groups=(
            CompiledParaphraseGroup(group_id="g0", source_text="scene 1 rồi scene 2", compiled_query=g0_temporal, group_weight_mass=0.5),
            CompiledParaphraseGroup(group_id="g1", source_text="scene 1 rồi scene 2", compiled_query=g1, group_weight_mass=0.5),
        ),
        normalized_weights=norm_w,
        hierarchical_quotas_c1={"q1::g0::v0": 10, "q1::g0::v1": 10, g1.variants[0].query_variant.variant_id: 20},
        hierarchical_quotas_c2={"q1::g0::v0": 20, "q1::g0::v1": 20, g1.variants[0].query_variant.variant_id: 20},
        provider_name="mock",
        baseline_weight_mass=2.0,
        total_semantic_budget_c1=40,
        total_semantic_budget_c2=60,
    )

    # Video records for "V01" and "V02"
    # For G0: V01 has valid chain (frame 10 @ score 0.9, frame 80 @ score 0.85) -> high temporal score
    # For G1: V02 has high score (0.95)
    rec_v0_v01 = VideoMaximumHit(query_id="q1::g0::v0", video_id="V01", rank=1, frame_id=10, clip_row=0, keyframe_order=0, cosine_score=0.90, top_m_score=0.90, top_m_peaks=((10, 0.90),))
    rec_v0_v02 = VideoMaximumHit(query_id="q1::g0::v0", video_id="V02", rank=2, frame_id=5, clip_row=0, keyframe_order=0, cosine_score=0.40, top_m_score=0.40, top_m_peaks=((5, 0.40),))

    rec_v1_v01 = VideoMaximumHit(query_id="q1::g0::v1", video_id="V01", rank=1, frame_id=80, clip_row=1, keyframe_order=1, cosine_score=0.85, top_m_score=0.85, top_m_peaks=((80, 0.85),))
    rec_v1_v02 = VideoMaximumHit(query_id="q1::g0::v1", video_id="V02", rank=2, frame_id=12, clip_row=1, keyframe_order=1, cosine_score=0.40, top_m_score=0.40, top_m_peaks=((12, 0.40),))

    g1_var_id = g1.variants[0].query_variant.variant_id
    rec_g1_v01 = VideoMaximumHit(query_id=g1_var_id, video_id="V01", rank=2, frame_id=15, clip_row=0, keyframe_order=0, cosine_score=0.70, top_m_score=0.70, top_m_peaks=((15, 0.70),))
    rec_g1_v02 = VideoMaximumHit(query_id=g1_var_id, video_id="V02", rank=1, frame_id=20, clip_row=0, keyframe_order=0, cosine_score=0.95, top_m_score=0.95, top_m_peaks=((20, 0.95),))

    maxima = FullCorpusVideoMaximaOutcome(
        rankings={
            "q1::g0::v0": (rec_v0_v01, rec_v0_v02),
            "q1::g0::v1": (rec_v1_v01, rec_v1_v02),
            g1_var_id: (rec_g1_v02, rec_g1_v01),
        },
        physical_rows_scored=200,
        video_store_scan_count=1,
    )

    cfg = KISVideoFirstConfig(enabled=True, v2_adaptive_enabled=True, top_m_min_frame_gap=60)
    selected, diag = fuse_video_maxima_v2_paraphrase_ensemble(
        ensemble=ensemble,
        maxima=maxima,
        rrf_constant=60.0,
        nomination_depth=100,
        config=cfg,
    )

    assert len(selected) == 2
    # V01 should have preserved the temporal chain from G0 (frame 10 and frame 80)
    v01_evidence = next(item for item in selected if item.video_id == "V01")
    assert v01_evidence.temporal_chain is not None
    assert v01_evidence.temporal_chain.has_valid_chain is True
    assert tuple(v01_evidence.temporal_chain.selected_chain_frames) == (10, 80)


def test_fixture_level_c0_c1_c2_negative_controls_parity():
    """Verify that when N=1 (negative controls contract), C0, C1, and C2 produce identical hierarchical quotas and weights."""
    g0 = _make_mock_compiled_query("q_neg", "group_canonical_new", 3, [1.0, 1.0, 0.35])
    w0 = 2.35
    b_sem = 60

    # Quotas for C1 vs C2 when N=1:
    quotas_c1 = allocate_hierarchical_quotas([g0], total_semantic_budget=b_sem)
    quotas_c2 = {v.query_variant.variant_id: 20 for v in g0.variants}
    assert quotas_c1 == quotas_c2

    norm_weights = compute_normalized_ensemble_weights([g0], baseline_weight_mass=w0)
    for v in g0.variants:
        assert abs(norm_weights[v.query_variant.variant_id] - v.query_variant.weight) < 1e-9


def test_paraphrase_sidecar_expected_group_hashes_real_values():
    """Verify expected_group_hashes and variant counts return real non-None values across all 5 queries."""
    sidecar_path = Path("scratch/benchmarks/translation_ablation/paraphrase_ensemble_p1_focus_v1.json")
    provider = ImmutableParaphraseEnsembleSidecarProvider(sidecar_path)

    # Treatment query P1-5
    p15_hashes = provider.expected_group_hashes("query-p1-5-kis")
    assert p15_hashes == {
        "group_canonical_new": "243b0f915c63",
        "group_candidate_old": "99f24deaaf56",
    }
    p15_counts = provider.expected_group_variant_counts("query-p1-5-kis")
    assert p15_counts == {"group_canonical_new": 3, "group_candidate_old": 3}

    # Also check short ID resolution
    assert provider.expected_group_hashes("p1-5") == p15_hashes

    # Negative controls
    assert provider.expected_group_hashes("query-p1-1-kis") == {"group_canonical_new": "7874635d65ba"}
    assert provider.expected_group_hashes("query-p1-2-kis") == {"group_canonical_new": "55fb2063a648"}
    assert provider.expected_group_hashes("query-p1-4-kis") == {"group_canonical_new": "d6ca9eddc122"}
    assert provider.expected_group_hashes("query-p1-6-kis") == {"group_canonical_new": "52f4082c4a3c"}


def test_group_aware_disjoint_video_rankings_no_key_error():
    """Verify that when 2 groups have completely disjoint top videos, fusion evaluates all corpus videos without KeyError."""
    from system_tai.kis.video_first import KISVideoFirstConfig, fuse_video_maxima_v2_paraphrase_ensemble
    from system_tai.retrieval.video_evidence import FullCorpusVideoMaximaOutcome, VideoMaximumHit

    g0 = _make_mock_compiled_query("q_disjoint", "g0", 1)
    g1 = _make_mock_compiled_query("q_disjoint", "g1", 1)

    v0_id = g0.variants[0].query_variant.variant_id
    v1_id = g1.variants[0].query_variant.variant_id

    # 4 videos in corpus: V1, V2, V3, V4
    # G0 ranks: V1 (#1), V2 (#2), V3 (#3), V4 (#4)
    # G1 ranks: V3 (#1), V4 (#2), V1 (#3), V2 (#4)
    hits_v0 = (
        VideoMaximumHit(query_id=v0_id, video_id="V1", rank=1, frame_id=10, clip_row=0, keyframe_order=0, cosine_score=0.9, top_m_score=0.9, top_m_peaks=((10, 0.9),)),
        VideoMaximumHit(query_id=v0_id, video_id="V2", rank=2, frame_id=10, clip_row=0, keyframe_order=0, cosine_score=0.8, top_m_score=0.8, top_m_peaks=((10, 0.8),)),
        VideoMaximumHit(query_id=v0_id, video_id="V3", rank=3, frame_id=10, clip_row=0, keyframe_order=0, cosine_score=0.5, top_m_score=0.5, top_m_peaks=((10, 0.5),)),
        VideoMaximumHit(query_id=v0_id, video_id="V4", rank=4, frame_id=10, clip_row=0, keyframe_order=0, cosine_score=0.4, top_m_score=0.4, top_m_peaks=((10, 0.4),)),
    )
    hits_v1 = (
        VideoMaximumHit(query_id=v1_id, video_id="V3", rank=1, frame_id=20, clip_row=0, keyframe_order=0, cosine_score=0.95, top_m_score=0.95, top_m_peaks=((20, 0.95),)),
        VideoMaximumHit(query_id=v1_id, video_id="V4", rank=2, frame_id=20, clip_row=0, keyframe_order=0, cosine_score=0.85, top_m_score=0.85, top_m_peaks=((20, 0.85),)),
        VideoMaximumHit(query_id=v1_id, video_id="V1", rank=3, frame_id=20, clip_row=0, keyframe_order=0, cosine_score=0.6, top_m_score=0.6, top_m_peaks=((20, 0.6),)),
        VideoMaximumHit(query_id=v1_id, video_id="V2", rank=4, frame_id=20, clip_row=0, keyframe_order=0, cosine_score=0.3, top_m_score=0.3, top_m_peaks=((20, 0.3),)),
    )

    maxima = FullCorpusVideoMaximaOutcome(
        rankings={v0_id: hits_v0, v1_id: hits_v1},
        physical_rows_scored=400,
        video_store_scan_count=1,
    )

    ensemble = CompiledParaphraseEnsemble(
        query_id="q_disjoint",
        source_vietnamese="test vi",
        groups=(
            CompiledParaphraseGroup(group_id="g0", source_text="test vi", compiled_query=g0, group_weight_mass=0.5),
            CompiledParaphraseGroup(group_id="g1", source_text="test vi", compiled_query=g1, group_weight_mass=0.5),
        ),
        normalized_weights={v0_id: 0.5, v1_id: 0.5},
        hierarchical_quotas_c1={v0_id: 10, v1_id: 10},
        hierarchical_quotas_c2={v0_id: 20, v1_id: 20},
        provider_name="mock",
        baseline_weight_mass=1.0,
        total_semantic_budget_c1=20,
        total_semantic_budget_c2=40,
    )

    cfg = KISVideoFirstConfig(enabled=True, v2_adaptive_enabled=False, selected_video_cap=2)
    selected, diag = fuse_video_maxima_v2_paraphrase_ensemble(
        ensemble=ensemble,
        maxima=maxima,
        rrf_constant=60.0,
        nomination_depth=100,
        config=cfg,
    )

    assert len(selected) == 2
    # Ranks must be contiguous 1..2
    assert [item.rank for item in selected] == [1, 2]
    # No KeyError occurred and all 4 videos were scored


def test_c0_immutable_sidecar_provider_contract():
    """Verify C0 sidecar provider loads canonical T-new sidecar and has all query translations."""
    from system_tai.translation.sidecar_provider import ImmutableSidecarTranslationProvider

    tnew_path = Path("scratch/benchmarks/translation_ablation/translation_p1_focus_v2_new.json")
    assert tnew_path.exists()

    expected_sha = "545bd4a37c57af53713a1d9f382241ef729c287a1817a5671fdc923115b0be2a"
    provider = ImmutableSidecarTranslationProvider(tnew_path, expected_content_sha256=expected_sha)
    assert provider.sidecar_id == "translation-p1-focus-v2-new"

    for qid in ["query-p1-1-kis", "query-p1-2-kis", "query-p1-4-kis", "query-p1-5-kis", "query-p1-6-kis"]:
        exp_h = provider.expected_semantic_hash(qid)
        assert isinstance(exp_h, str) and len(exp_h) == 12


def test_session_config_validates_and_constructs():
    """Verify SessionConfig constructs cleanly with runner fields."""
    from system_tai.kis.session_schema import SessionConfig, KISVideoFirstConfig
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        vf_cfg = KISVideoFirstConfig(
            enabled=True,
            v2_adaptive_enabled=True,
            selected_video_cap=64,
            restricted_frames_per_video_per_variant=20,
            enable_paraphrase_ensemble=True,
            paraphrase_ensemble_mode="EQUAL_BUDGET",
        )
        cfg = SessionConfig(
            input_root=tdp / "features",
            reuse_manifest=tdp / "manifest.json",
            output_root=tdp / "out",
            enable_dynamic_translation=True,
            allow_model_download=False,
            rrf_constant=60.0,
            kis_video_first_config=vf_cfg,
        )
        assert cfg.kis_video_first_config.enable_paraphrase_ensemble is True
        assert cfg.kis_video_first_config.selected_video_cap == 64


def test_non_ensemble_quota_preservation_k10():
    """Verify that when paraphrase ensemble is disabled, semantic and VI caps both use restricted_frames_per_video_per_variant."""
    from system_tai.kis.session_schema import KISVideoFirstConfig

    cfg_k10 = KISVideoFirstConfig(
        enabled=True,
        v2_adaptive_enabled=True,
        restricted_frames_per_video_per_variant=10,
        enable_vi_localization_variant=True,
        enable_paraphrase_ensemble=False,
    )
    assert cfg_k10.restricted_frames_per_video_per_variant == 10
    assert cfg_k10.enable_paraphrase_ensemble is False


def test_query_manifest_not_allowed_as_corpus_manifest():
    """Verify that passing a query benchmark manifest to load_corpus_manifest raises ValueError."""
    from system_tai.data.corpus_discovery import load_corpus_manifest
    query_manifest = Path("systems/system_tai/benchmarks/frozen_kis_v2a_stress_manifest.json")
    assert query_manifest.exists()

    with pytest.raises(ValueError, match="unsupported feature manifest"):
        load_corpus_manifest(query_manifest)


def test_altered_vietnamese_query_text_raises_fail_fast():
    """Verify that providing altered Vietnamese query text for an existing query_id raises ValueError."""
    sidecar_path = Path("scratch/benchmarks/translation_ablation/paraphrase_ensemble_p1_focus_v1.json")
    provider = ImmutableParaphraseEnsembleSidecarProvider(sidecar_path)

    # Valid original text
    original_p15_text = provider._query_meta["query-p1-5-kis"]["query_vi"]
    provider.validate_query_vi("query-p1-5-kis", original_p15_text)

    # Altered text must fail fast
    with pytest.raises(ValueError, match="Vietnamese text altered"):
        provider.validate_query_vi("query-p1-5-kis", "Một người phụ nữ mặc áo khoác đỏ")


def test_canonical_projection_digest_ieee754_hex_format():
    """Verify canonical_projection_digest produces exact deterministic IEEE-754 64-bit uppercase hex digests."""
    from system_tai.retrieval.canonical_projection import canonical_projection_digest, float_to_ieee754_hex

    # Test float hex encoding
    assert float_to_ieee754_hex(1.0) == "3FF0000000000000"
    assert float_to_ieee754_hex(0.0) == "0000000000000000"

    candidates = [
        {"rank": 1, "video_id": "L26_V035", "frame_id": 100, "fusion_score": 0.95},
        {"rank": 2, "video_id": "L30_V046", "frame_id": 200, "fusion_score": 0.85},
    ]
    digest = canonical_projection_digest(candidates)
    assert digest == "310f5d76d10a1d905ff0cd4a151eea5e02b9bc9e1765cf889481193a8c0f63cd"


def test_missing_clip_tokenizer_raises_translation_error_fail_fast():
    """Verify that when OpenAI CLIP is not installed, TokenBudgetGuard raises TranslationError."""
    from unittest.mock import patch
    from system_tai.translation.provider import TokenBudgetGuard, TranslationError

    guard = TokenBudgetGuard(max_tokens=75)
    with patch.dict("sys.modules", {"clip": None}):
        with pytest.raises(TranslationError, match="OpenAI CLIP must be installed"):
            guard.count_tokens("A test query text")


def _create_mock_corpus_manifest(dataset_root: Path):
    from unittest.mock import MagicMock
    mm = MagicMock()
    mm.dataset_root = dataset_root
    mm.schema_version = "v1"
    mm.discovery_version = "v1"
    mm.path_mode = "portable"
    mm.fingerprint = "fp_mock_1234567890abcdef"
    mm.identifiers = {}
    mm.videos = ()
    mm.video_count = 100
    mm.total_frames = 1000
    mm.embedding_dimension = 512
    return mm


def _create_mock_raw_video_registry():
    from unittest.mock import MagicMock
    mock_raw = MagicMock()
    mock_raw.records = ()
    mock_rec = MagicMock()
    mock_rec.raw_video_path = None
    mock_rec.video_id = "V01"
    mock_raw.get.return_value = mock_rec
    return mock_raw


def test_non_video_first_runtime_executes_without_unbound_error():
    """Verify OperationalKISRuntime.handle_query executes without UnboundLocalError when video-first is disabled."""
    import tempfile
    import numpy as np
    from unittest.mock import MagicMock
    from system_tai.kis.session_schema import SessionConfig, KISVideoFirstConfig, QueryRequest
    from system_tai.kis.session_engine import OperationalKISRuntime
    from system_tai.retrieval.multi_query import WeightedRRFRetriever

    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        cfg = SessionConfig(
            input_root=tdp,
            output_root=tdp,
            enable_dynamic_translation=False,
            kis_video_first_config=KISVideoFirstConfig(enabled=False),
        )

        mock_encoder = MagicMock()
        mock_encoder.dimension = 512
        mock_encoder.identifiers = {"device": "cpu", "model": "ViT-B/32"}
        mock_encoder.encode_texts.side_effect = lambda texts: np.ones((len(texts), 512), dtype=np.float32)

        mock_registry = MagicMock()
        mock_registry.embedding_dimension = 512
        mock_registry.total_rows = 10
        mock_registry.stores = ()
        mock_registry.keys.return_value = ("L30_V046",)
        mock_registry.get_store.return_value = None

        mock_manifest = _create_mock_corpus_manifest(tdp)

        from system_tai.common.schemas import CandidateFrame
        mock_retriever = MagicMock()
        mock_cand = CandidateFrame(
            video_id="L30_V046",
            frame_id=100,
            clip_row=0,
            keyframe_order=0,
            score=0.9,
            rank=1,
            source="exact_retriever",
        )
        mock_result = MagicMock()
        mock_result.candidates = (mock_cand,)
        mock_result.ranked_candidates = (mock_cand,)
        mock_retriever.search_vector.return_value = mock_result

        runtime = OperationalKISRuntime(
            config=cfg,
            manifest_path=tdp / "manifest.json",
            manifest=mock_manifest,
            registry=mock_registry,
            raw_video_registry=_create_mock_raw_video_registry(),
            shared_encoder=mock_encoder,
            decoder=MagicMock(),
            translation_provider=None,
            token_budget_guard=None,
        )
        runtime.exact_retriever = mock_retriever
        runtime.weighted_rrf = WeightedRRFRetriever(exact_retriever=None)  # type: ignore[arg-type]

        req = QueryRequest(
            request_id="req_non_vf_1",
            query_id="query_p1_1",
            query_vi="test query",
            query_en="test query en",
        )
        resp = runtime.handle_query(req)
        assert resp["status"] == "SUCCESS"


def test_non_ensemble_k10_passes_exact_result_cap_to_searcher():
    """Verify that when restricted_frames_per_video_per_variant=10, search_selected_videos receives 10."""
    import tempfile
    import numpy as np
    from unittest.mock import MagicMock
    from system_tai.kis.session_schema import SessionConfig, KISVideoFirstConfig, QueryRequest
    from system_tai.kis.session_engine import OperationalKISRuntime
    from system_tai.retrieval.video_evidence import (
        FullCorpusVideoMaximaOutcome,
        VideoMaximumHit,
        VideoRestrictedSearchOutcome,
        RestrictedFrameHit,
    )

    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        cap_received = []

        mock_searcher = MagicMock()
        mock_searcher.search_video_maxima.return_value = FullCorpusVideoMaximaOutcome(
            rankings={"query_p1_1::semantic_01_s01": (VideoMaximumHit(query_id="query_p1_1::semantic_01_s01", video_id="V01", frame_id=10, clip_row=0, keyframe_order=0, cosine_score=0.9, rank=1),)},
            physical_rows_scored=10,
            video_store_scan_count=1,
        )
        def fake_search_selected_videos(*, per_query_result_cap, **kwargs):
            cap_received.append(per_query_result_cap)
            return VideoRestrictedSearchOutcome(
                rankings={"query_p1_1::semantic_01_s01": {"V01": (RestrictedFrameHit(video_id="V01", frame_id=10, clip_row=0, keyframe_order=0, pts_time=1.0, cosine_score=0.9, rank=1),)}},
                physical_rows_scored=10,
                video_store_scan_count=1,
            )
        mock_searcher.search_selected_videos.side_effect = fake_search_selected_videos

        cfg = SessionConfig(
            input_root=tdp,
            output_root=tdp,
            enable_dynamic_translation=True,
            kis_video_first_config=KISVideoFirstConfig(
                enabled=True,
                restricted_frames_per_video_per_variant=10,
                enable_paraphrase_ensemble=False,
            ),
        )

        mock_trans = MagicMock()
        mock_trans.provider_name = "mock_provider"
        mock_trans.sidecar_id = "mock_sidecar"
        mock_trans.translate_many.side_effect = lambda texts, **kwargs: tuple(f"En {t}" for t in texts)
        mock_trans.expected_semantic_hash.return_value = None
        mock_trans.expected_variant_count.return_value = None
        mock_trans.sidecar_metadata.return_value = {"sidecar_id": "mock_sidecar"}

        mock_guard = MagicMock()
        mock_guard.split_for_clip.side_effect = lambda t: (t,)
        mock_guard.count_tokens.return_value = 5
        mock_guard.max_tokens = 75

        mock_encoder = MagicMock()
        mock_encoder.dimension = 512
        mock_encoder.identifiers = {"device": "cpu", "model": "ViT-B/32"}
        mock_encoder.encode_texts.side_effect = lambda texts: np.ones((len(texts), 512), dtype=np.float32)

        runtime = OperationalKISRuntime(
            config=cfg,
            manifest_path=tdp / "manifest.json",
            manifest=_create_mock_corpus_manifest(tdp),
            registry=MagicMock(embedding_dimension=512, total_rows=10, stores=(), keys=lambda: ("V01",), get_store=lambda x: None),
            raw_video_registry=_create_mock_raw_video_registry(),
            shared_encoder=mock_encoder,
            decoder=MagicMock(),
            translation_provider=mock_trans,
            token_budget_guard=mock_guard,
        )
        runtime.video_restricted_searcher = mock_searcher

        req = QueryRequest(
            request_id="req_k10",
            query_id="query_p1_1",
            query_vi="test query vi",
        )
        runtime.handle_query(req)
        assert len(cap_received) == 1
        # The result cap passed to search_selected_videos must be 10 or a mapping with 10 (not 20!)
        if isinstance(cap_received[0], dict):
            assert all(v == 10 for v in cap_received[0].values())
        else:
            assert cap_received[0] == 10


def test_phase_c_audit_runner_smoke_with_mock_bootstrap():
    """Verify run_phase_c_audit executes end-to-end with bootstrap lifecycle."""
    import tempfile
    import numpy as np
    from unittest.mock import patch, MagicMock
    from scratch.run_phase_c_paraphrase_ensemble import run_phase_c_audit
    from system_tai.kis.session_engine import OperationalKISRuntime
    from system_tai.retrieval.multi_query import WeightedRRFRetriever
    from system_tai.translation.provider import TokenBudgetGuard
    from system_tai.retrieval.video_evidence import (
        FullCorpusVideoMaximaOutcome,
        VideoMaximumHit,
        VideoRestrictedSearchOutcome,
        RestrictedFrameHit,
    )

    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        out_dir = tdp / "phase_c_out"

        query_manifest = Path("systems/system_tai/benchmarks/frozen_kis_v2a_stress_manifest.json")
        tnew_sidecar = Path("scratch/benchmarks/translation_ablation/translation_p1_focus_v2_new.json")
        para_sidecar = Path("scratch/benchmarks/translation_ablation/paraphrase_ensemble_p1_focus_v1.json")
        manual_ref = Path("systems/system_tai/benchmarks/manual_kis_reference_v1.json")

        def mock_bootstrap(config, *, translation_provider=None, **kwargs):
            mock_encoder = MagicMock()
            mock_encoder.dimension = 512
            mock_encoder.identifiers = {"device": "cpu", "model": "ViT-B/32"}
            mock_encoder.encode_texts.side_effect = lambda texts: np.ones((len(texts), 512), dtype=np.float32)

            mock_searcher = MagicMock()
            def fake_maxima(*, query_ids, **kwargs):
                return FullCorpusVideoMaximaOutcome(
                    rankings={qid: tuple(VideoMaximumHit(query_id=qid, video_id=f"L{i:02d}_V001", frame_id=i*10, clip_row=0, keyframe_order=0, cosine_score=0.9 - i*0.005, rank=i) for i in range(1, 101)) for qid in query_ids},
                    physical_rows_scored=100,
                    video_store_scan_count=1,
                )
            mock_searcher.search_video_maxima.side_effect = fake_maxima

            def fake_restricted(*, video_ids, query_ids, **kwargs):
                return VideoRestrictedSearchOutcome(
                    rankings={
                        qid: {
                            vid: (
                                RestrictedFrameHit(video_id=vid, frame_id=10, clip_row=0, keyframe_order=0, pts_time=1.0, cosine_score=0.9, rank=1),
                                RestrictedFrameHit(video_id=vid, frame_id=20, clip_row=1, keyframe_order=1, pts_time=2.0, cosine_score=0.8, rank=2),
                            )
                            for vid in video_ids
                        }
                        for qid in query_ids
                    },
                    physical_rows_scored=100,
                    video_store_scan_count=1,
                    candidate_selection_telemetry={qid: {vid: {"effective_candidate_count": 20, "compulsory_extra_count": 0} for vid in video_ids} for qid in query_ids},
                )
            mock_searcher.search_selected_videos.side_effect = fake_restricted

            real_guard = TokenBudgetGuard(max_tokens=75)
            class _SimpleWordTokenizer:
                @staticmethod
                def encode(text: str) -> list[str]:
                    return text.split()
            real_guard._clip_tokenizer = _SimpleWordTokenizer()

            runtime = OperationalKISRuntime(
                config=config,
                manifest_path=tdp / "manifest.json",
                manifest=_create_mock_corpus_manifest(tdp),
                registry=MagicMock(embedding_dimension=512, total_rows=100, stores=(), keys=lambda: tuple(f"L{i:02d}_V001" for i in range(1, 101)), get_store=lambda x: None),
                raw_video_registry=_create_mock_raw_video_registry(),
                shared_encoder=mock_encoder,
                decoder=MagicMock(),
                translation_provider=translation_provider,
                token_budget_guard=real_guard,
            )
            runtime.video_restricted_searcher = mock_searcher
            runtime.weighted_rrf = WeightedRRFRetriever(exact_retriever=None)  # type: ignore[arg-type]
            return runtime

        with patch.object(OperationalKISRuntime, "bootstrap", side_effect=mock_bootstrap):
            res = run_phase_c_audit(
                query_manifest_path=query_manifest,
                input_root=tdp,
                output_dir=out_dir,
                tnew_sidecar_path=tnew_sidecar,
                paraphrase_sidecar_path=para_sidecar,
                manual_ref_path=manual_ref,
                allow_model_download=False,
                strict_corpus_gate=False,
            )
            assert "arms" in res
            assert "C0_baseline" in res["arms"]
            assert "C1_equal_budget_ensemble" in res["arms"]
            assert "C2_expanded_retention_ensemble" in res["arms"]
            assert (out_dir / "phase_c_paraphrase_ensemble_audit.json").exists()


def test_preseeded_stale_candidates_not_used_by_audit():
    """Verify run_phase_c_audit uses runtime output records directly, ignoring any pre-existing/stale candidate files."""
    import tempfile
    import json
    import numpy as np
    from unittest.mock import patch, MagicMock
    from scratch.run_phase_c_paraphrase_ensemble import run_phase_c_audit
    from system_tai.kis.session_engine import OperationalKISRuntime
    from system_tai.retrieval.multi_query import WeightedRRFRetriever
    from system_tai.translation.provider import TokenBudgetGuard
    from system_tai.retrieval.video_evidence import (
        FullCorpusVideoMaximaOutcome,
        VideoMaximumHit,
        VideoRestrictedSearchOutcome,
        RestrictedFrameHit,
    )

    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        out_dir = tdp / "phase_c_out"

        # Pre-seed a bogus stale candidate file inside the output directory structure
        stale_dir = out_dir / "C0_baseline" / "requests" / "stale_p1_5"
        stale_dir.mkdir(parents=True, exist_ok=True)
        stale_file = stale_dir / "candidates.json"
        stale_file.write_text(
            json.dumps({"records": [{"rank": 1, "video_id": "BOGUS_VIDEO", "frame_id": 999, "fusion_score": 1.0}] * 100}),
            encoding="utf-8",
        )

        query_manifest = Path("systems/system_tai/benchmarks/frozen_kis_v2a_stress_manifest.json")
        tnew_sidecar = Path("scratch/benchmarks/translation_ablation/translation_p1_focus_v2_new.json")
        para_sidecar = Path("scratch/benchmarks/translation_ablation/paraphrase_ensemble_p1_focus_v1.json")
        manual_ref = Path("systems/system_tai/benchmarks/manual_kis_reference_v1.json")

        def mock_bootstrap(config, *, translation_provider=None, **kwargs):
            mock_encoder = MagicMock()
            mock_encoder.dimension = 512
            mock_encoder.identifiers = {"device": "cpu", "model": "ViT-B/32"}
            mock_encoder.encode_texts.side_effect = lambda texts: np.ones((len(texts), 512), dtype=np.float32)

            mock_searcher = MagicMock()
            def fake_maxima(*, query_ids, **kwargs):
                return FullCorpusVideoMaximaOutcome(
                    rankings={qid: tuple(VideoMaximumHit(query_id=qid, video_id=f"L{i:02d}_V001", frame_id=i*10, clip_row=0, keyframe_order=0, cosine_score=0.9 - i*0.005, rank=i) for i in range(1, 101)) for qid in query_ids},
                    physical_rows_scored=100,
                    video_store_scan_count=1,
                )
            mock_searcher.search_video_maxima.side_effect = fake_maxima

            def fake_restricted(*, video_ids, query_ids, **kwargs):
                return VideoRestrictedSearchOutcome(
                    rankings={
                        qid: {
                            vid: (
                                RestrictedFrameHit(video_id=vid, frame_id=10, clip_row=0, keyframe_order=0, pts_time=1.0, cosine_score=0.9, rank=1),
                                RestrictedFrameHit(video_id=vid, frame_id=20, clip_row=1, keyframe_order=1, pts_time=2.0, cosine_score=0.8, rank=2),
                            )
                            for vid in video_ids
                        }
                        for qid in query_ids
                    },
                    physical_rows_scored=100,
                    video_store_scan_count=1,
                    candidate_selection_telemetry={qid: {vid: {"effective_candidate_count": 20, "compulsory_extra_count": 0} for vid in video_ids} for qid in query_ids},
                )
            mock_searcher.search_selected_videos.side_effect = fake_restricted

            real_guard = TokenBudgetGuard(max_tokens=75)
            class _SimpleWordTokenizer:
                @staticmethod
                def encode(text: str) -> list[str]:
                    return text.split()
            real_guard._clip_tokenizer = _SimpleWordTokenizer()

            runtime = OperationalKISRuntime(
                config=config,
                manifest_path=tdp / "manifest.json",
                manifest=_create_mock_corpus_manifest(tdp),
                registry=MagicMock(embedding_dimension=512, total_rows=100, stores=(), keys=lambda: tuple(f"L{i:02d}_V001" for i in range(1, 101)), get_store=lambda x: None),
                raw_video_registry=_create_mock_raw_video_registry(),
                shared_encoder=mock_encoder,
                decoder=MagicMock(),
                translation_provider=translation_provider,
                token_budget_guard=real_guard,
            )
            runtime.video_restricted_searcher = mock_searcher
            runtime.weighted_rrf = WeightedRRFRetriever(exact_retriever=None)  # type: ignore[arg-type]
            return runtime

        with patch.object(OperationalKISRuntime, "bootstrap", side_effect=mock_bootstrap):
            res = run_phase_c_audit(
                query_manifest_path=query_manifest,
                input_root=tdp,
                output_dir=out_dir,
                tnew_sidecar_path=tnew_sidecar,
                paraphrase_sidecar_path=para_sidecar,
                manual_ref_path=manual_ref,
                allow_model_download=False,
                strict_corpus_gate=False,
            )
            c0_p15 = res["arms"]["C0_baseline"]["query-p1-5-kis"]
            assert not any(r["video_id"] == "BOGUS_VIDEO" for r in c0_p15["records"])
            assert "stale_p1_5" not in c0_p15["candidates_json_path"]
