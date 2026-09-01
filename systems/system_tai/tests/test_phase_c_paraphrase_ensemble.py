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
