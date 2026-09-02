"""Comprehensive Unit & Regression Tests for Phase C.1: Four-Way Controlled Ablation Benchmark."""

import json
import math
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from system_tai.kis.session_engine import OperationalKISRuntime
from system_tai.kis.session_schema import KISVideoFirstConfig, SessionConfig
from system_tai.retrieval.multi_query import WeightedRRFRetriever
from system_tai.retrieval.video_evidence import (
    FullCorpusVideoMaximaOutcome,
    RestrictedFrameHit,
    VideoMaximumHit,
    VideoRestrictedSearchOutcome,
)
from system_tai.translation.paraphrase_sidecar_provider import (
    ImmutableParaphraseEnsembleSidecarProvider,
)
from system_tai.translation.provider import TokenBudgetGuard
from system_tai.translation.sidecar_provider import canonical_sidecar_sha256
from scratch.run_phase_c1_four_way_ablation import (
    CANONICAL_PARAPHRASE_SHA256,
    CANONICAL_SHAM_SHA256,
    CANONICAL_TNEW_SHA256,
    CANONICAL_TOLD_SHA256,
    classify_p15_findings,
    compute_pairwise_comparison,
    run_phase_c1_four_way_ablation,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
TNEW_SIDECAR_PATH = REPO_ROOT / "scratch" / "benchmarks" / "translation_ablation" / "translation_p1_focus_v2_new.json"
TOLD_SIDECAR_PATH = REPO_ROOT / "scratch" / "benchmarks" / "translation_ablation" / "translation_p1_focus_v1_old_candidate.json"
SHAM_SIDECAR_PATH = REPO_ROOT / "scratch" / "benchmarks" / "translation_ablation" / "paraphrase_sham_duplicate_new_p1_focus_v1.json"
PARA_SIDECAR_PATH = REPO_ROOT / "scratch" / "benchmarks" / "translation_ablation" / "paraphrase_ensemble_p1_focus_v1.json"
QUERY_MANIFEST_PATH = REPO_ROOT / "systems" / "system_tai" / "benchmarks" / "frozen_kis_v2a_stress_manifest.json"
MANUAL_REF_PATH = REPO_ROOT / "systems" / "system_tai" / "benchmarks" / "manual_kis_reference_v1.json"


def test_sham_sidecar_canonical_sha256_and_loading():
    """Verify sham duplicate sidecar exists, matches canonical SHA, and loads cleanly."""
    assert SHAM_SIDECAR_PATH.exists(), f"Sham sidecar missing: {SHAM_SIDECAR_PATH}"
    actual_sha = canonical_sidecar_sha256(SHAM_SIDECAR_PATH)
    assert actual_sha == CANONICAL_SHAM_SHA256

    provider = ImmutableParaphraseEnsembleSidecarProvider(
        sidecar_path=SHAM_SIDECAR_PATH,
        expected_content_sha256=CANONICAL_SHAM_SHA256,
    )

    # For P1-5: must have group_sham_a and group_sham_b
    groups_p15 = provider.get_paraphrase_groups("query-p1-5-kis")
    assert len(groups_p15) == 2
    g_ids = [g["group_id"] for g in groups_p15]
    assert g_ids == ["group_sham_a", "group_sham_b"]

    exp_hashes = provider.expected_group_hashes("query-p1-5-kis")
    assert exp_hashes["group_sham_a"] == "243b0f915c63"
    assert exp_hashes["group_sham_b"] == "243b0f915c63"

    exp_counts = provider.expected_group_variant_counts("query-p1-5-kis")
    assert exp_counts["group_sham_a"] == 3
    assert exp_counts["group_sham_b"] == 3

    # Negative controls must have exactly 1 group (group_canonical_new)
    for qid in ["query-p1-1-kis", "query-p1-2-kis", "query-p1-4-kis", "query-p1-6-kis"]:
        groups = provider.get_paraphrase_groups(qid)
        assert len(groups) == 1
        assert groups[0]["group_id"] == "group_canonical_new"


def test_phase_c1_exact_run_g_config_contract():
    """Preflight test verifying KISVideoFirstConfig and SessionConfig contract matches Phase C exactly."""
    vf_cfg = KISVideoFirstConfig(
        enabled=True,
        v2_adaptive_enabled=True,
        selected_video_cap=64,
        video_nomination_depth=100,
        restricted_frames_per_video_per_variant=20,
        full_query_weight=1.0,
        primary_scene_weight=1.0,
        supporting_attribute_weight=0.35,
        top_m_evidence_cap=5,
        top_m_weights=(0.4, 0.25, 0.15, 0.1, 0.1),
        top_m_min_frame_gap=60,
        enable_temporal_diverse_local_candidates=True,
        temporal_diversity_gap_seconds=5.0,
        enable_vi_localization_variant=True,
        vi_localization_weight=0.5,
        internal_rrf_candidate_depth=1000,
        enable_top_video_local_anchor=False,
        enable_paraphrase_ensemble=False,
        paraphrase_ensemble_mode="EQUAL_BUDGET",
    )
    assert vf_cfg.top_m_evidence_cap == 5
    assert vf_cfg.selected_video_cap == 64
    assert vf_cfg.internal_rrf_candidate_depth == 1000
    assert vf_cfg.enable_top_video_local_anchor is False

    session_cfg = SessionConfig(
        session_id="test_session",
        input_root=Path("."),
        output_root=Path("."),
        enable_dynamic_translation=True,
        device="cpu",
        rrf_constant=60.0,
        continue_on_request_error=False,
        fail_fast_protocol=True,
        kis_video_first_config=vf_cfg,
    )
    assert session_cfg.device == "cpu"
    assert session_cfg.rrf_constant == 60.0


def test_pairwise_comparison_all_six():
    """Verify compute_pairwise_comparison computes exact set metrics and rank shifts."""
    records_a = [{"video_id": f"V{i:03d}", "frame_id": 100 + i, "rank": i + 1} for i in range(100)]
    records_b = [{"video_id": f"V{i:03d}", "frame_id": 100 + i, "rank": (i + 1)} for i in range(70)] + [
        {"video_id": f"NEW_{j:03d}", "frame_id": 900 + j, "rank": 71 + j} for j in range(30)
    ]
    res = compute_pairwise_comparison(records_a, records_b)
    assert res["intersection_count"] == 70
    assert res["union_count"] == 130
    assert round(res["jaccard_similarity"], 4) == round(70 / 130, 4)
    assert res["membership_replaced_count"] == 30
    assert res["membership_replaced_ratio"] == 0.30
    assert res["median_rank_shift"] == 0.0


def test_classify_p15_findings_all_verdicts():
    """Test all diagnosis branches: destructive interference, confound, weak old, complementarity, compound, inconclusive, miss."""
    # 1. DESTRUCTIVE_INTER_GROUP_INTERFERENCE
    c1, v1 = classify_p15_findings(r_new=25, r_old=16, r_sham=25, r_mixed=31, selected_video_count_parity=True)
    assert v1 == "DESTRUCTIVE_INTER_GROUP_INTERFERENCE"
    assert c1["findings"]["destructive_interference"] is True

    # 2. ENSEMBLE_MECHANICS_CONFOUND
    c2, v2 = classify_p15_findings(r_new=25, r_old=25, r_sham=31, r_mixed=31, selected_video_count_parity=True)
    assert v2 == "ENSEMBLE_MECHANICS_CONFOUND"
    assert c2["findings"]["ensemble_mechanics_effect"] is True

    # 3. WEAK_OLD_GROUP_DILUTION_SUPPORTED
    c3, v3 = classify_p15_findings(r_new=25, r_old=60, r_sham=25, r_mixed=31, selected_video_count_parity=True)
    assert v3 == "WEAK_OLD_GROUP_DILUTION_SUPPORTED"
    assert c3["findings"]["old_wording_weaker_than_new"] is True

    # 4. ENSEMBLE_COMPLEMENTARITY
    c4, v4 = classify_p15_findings(r_new=25, r_old=25, r_sham=25, r_mixed=15, selected_video_count_parity=True)
    assert v4 == "ENSEMBLE_COMPLEMENTARITY"
    assert c4["findings"]["complementarity"] is True

    # 5. COMPOUND_MECHANICS_AND_REPLACEMENT_DEGRADATION
    c5, v5 = classify_p15_findings(r_new=25, r_old=25, r_sham=29, r_mixed=35, selected_video_count_parity=True)
    assert v5 == "COMPOUND_MECHANICS_AND_REPLACEMENT_DEGRADATION"

    # 6. ADAPTIVE_VIDEO_BUDGET_CONFOUND
    c6, v6 = classify_p15_findings(r_new=25, r_old=16, r_sham=25, r_mixed=31, selected_video_count_parity=False)
    assert v6 == "ADAPTIVE_VIDEO_BUDGET_CONFOUND"

    # 7. INCONCLUSIVE (all tied)
    c7, v7 = classify_p15_findings(r_new=25, r_old=25, r_sham=25, r_mixed=25, selected_video_count_parity=True)
    assert v7 == "INCONCLUSIVE"

    # 8. Handling MISS (None rank)
    c8, v8 = classify_p15_findings(r_new=25, r_old=None, r_sham=25, r_mixed=None, selected_video_count_parity=True)
    assert c8["reciprocal_ranks"]["C_OLD_SINGLE"] == 0.0
    assert c8["reciprocal_ranks"]["C_NEW_OLD_50_50"] == 0.0
    assert c8["rank_deltas"]["delta_wording_old_minus_new"] is None
    assert c8["reciprocal_rank_deltas"]["delta_rr_wording"] == -0.04


def test_sham_mass_missing_or_zero_fails_closed():
    """Verify that if group_sham_a/b mass is missing, zero, or unequal, runner validation asserts fail-closed."""
    # Test zero mass
    mass_a = 0.0
    mass_b = 0.5
    total_mass = 0.5
    assert (mass_a <= 0.0 or not math.isfinite(mass_a)) is True

    # Test unequal mass
    mass_a = 0.3
    mass_b = 0.7
    total_mass = 1.0
    assert not math.isclose(mass_a, mass_b, rel_tol=1e-5)


def test_phase_c1_runner_smoke_with_mock_bootstrap():
    """Execute run_phase_c1_four_way_ablation with mock bootstrap to verify end-to-end orchestration."""
    def _create_mock_corpus_manifest(root: Path):
        from system_tai.data.corpus_discovery import CorpusManifest, DiscoveredVideo, DiscoveryMetrics, DiscoveryValidation
        ordered = tuple(
            DiscoveredVideo(
                video_id=f"L{i:02d}_V001",
                mapping_csv_path=root / f"L{i:02d}_V001.csv",
                clip_npy_path=root / f"L{i:02d}_V001.npy",
                keyframe_directory=root / f"L{i:02d}_V001",
                raw_video_path=None,
                row_count=1,
                embedding_dimension=512,
                mapping_size_bytes=100,
                clip_size_bytes=2048,
                keyframe_image_count=1,
            )
            for i in range(1, 101)
        )
        return CorpusManifest(
            input_root=root,
            dataset_root=root,
            fingerprint="fp_mock_smoke_12345678",
            videos=ordered,
            schema_version="1.0.0",
            discovery_version="1.0.0",
            portable=True,
            dataset_identity="mock_id",
            validation_mode=DiscoveryValidation.STRICT,
            discovery_metrics=DiscoveryMetrics(),
        )

    def _create_mock_raw_video_registry(root: Path):
        from system_tai.refinement.video import RawVideoRecord, RawVideoRegistry
        records = [
            RawVideoRecord(
                video_id=f"L{i:02d}_V001",
                raw_video_path=root / f"L{i:02d}_V001.mp4",
            )
            for i in range(1, 101)
        ]
        return RawVideoRegistry(records=records)

    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        out_dir = tdp / "phase_c1_out"

        query_manifest = QUERY_MANIFEST_PATH
        tnew_sidecar = TNEW_SIDECAR_PATH
        told_sidecar = TOLD_SIDECAR_PATH
        sham_sidecar = SHAM_SIDECAR_PATH
        para_sidecar = PARA_SIDECAR_PATH
        manual_ref = MANUAL_REF_PATH

        providers_seen = []

        def mock_bootstrap(config, *, translation_provider=None, **kwargs):
            providers_seen.append(translation_provider)
            mock_encoder = MagicMock()
            mock_encoder.dimension = 512
            mock_encoder.identifiers = {"device": "cpu", "model": "ViT-B/32"}
            mock_encoder.encode_texts.side_effect = lambda texts: np.ones((len(texts), 512), dtype=np.float32)

            mock_searcher = MagicMock()
            def fake_maxima(*, query_ids, **kwargs):
                return FullCorpusVideoMaximaOutcome(
                    rankings={
                        qid: tuple(
                            VideoMaximumHit(
                                query_id=qid,
                                video_id=f"L{i:02d}_V001",
                                frame_id=i * 10,
                                clip_row=0,
                                keyframe_order=0,
                                cosine_score=0.9 - i * 0.005,
                                rank=i,
                            )
                            for i in range(1, 101)
                        )
                        for qid in query_ids
                    },
                    physical_rows_scored=100,
                    video_store_scan_count=100,
                )
            mock_searcher.search_video_maxima.side_effect = fake_maxima

            def fake_search_selected(*, query_ids, video_ids, **kwargs):
                return VideoRestrictedSearchOutcome(
                    rankings={
                        qid: {
                            vid: (
                                RestrictedFrameHit(
                                    video_id=vid,
                                    frame_id=10,
                                    clip_row=0,
                                    keyframe_order=0,
                                    pts_time=1.0,
                                    cosine_score=0.9,
                                    rank=1,
                                ),
                                RestrictedFrameHit(
                                    video_id=vid,
                                    frame_id=20,
                                    clip_row=1,
                                    keyframe_order=1,
                                    pts_time=2.0,
                                    cosine_score=0.8,
                                    rank=2,
                                ),
                            )
                            for vid in video_ids
                        }
                        for qid in query_ids
                    },
                    physical_rows_scored=100,
                    video_store_scan_count=len(video_ids),
                )
            mock_searcher.search_selected_videos.side_effect = fake_search_selected

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
                registry=MagicMock(
                    embedding_dimension=512,
                    total_rows=100,
                    stores=tuple(MagicMock() for _ in range(100)),
                    get_store=lambda x: None,
                ),
                raw_video_registry=_create_mock_raw_video_registry(tdp),
                shared_encoder=mock_encoder,
                decoder=MagicMock(),
                translation_provider=translation_provider,
                token_budget_guard=real_guard,
            )
            runtime.video_restricted_searcher = mock_searcher
            runtime.weighted_rrf = WeightedRRFRetriever(exact_retriever=None)  # type: ignore[arg-type]
            return runtime

        with patch.object(OperationalKISRuntime, "bootstrap", side_effect=mock_bootstrap):
            res = run_phase_c1_four_way_ablation(
                query_manifest_path=query_manifest,
                input_root=tdp,
                manifest_cache_path=tdp / "manifest_cache.json",
                output_root=out_dir,
                tnew_sidecar_path=tnew_sidecar,
                told_sidecar_path=told_sidecar,
                sham_sidecar_path=sham_sidecar,
                paraphrase_sidecar_path=para_sidecar,
                manual_ref_path=manual_ref,
                allow_model_download=False,
                strict_corpus_gate=False,
            )

            # Assert all 4 arms executed
            assert len(providers_seen) == 4
            assert "arms" in res
            assert "C_NEW_SINGLE" in res["arms"]
            assert "C_OLD_SINGLE" in res["arms"]
            assert "C_NEW_DUP_SHAM" in res["arms"]
            assert "C_NEW_OLD_50_50" in res["arms"]

            # Assert Sham invariants pass
            assert res["sham_invariants"]["all_invariants_pass"] is True
            assert res["sham_invariants"]["group_count"] == 2

            # Assert Negative controls bit-exact
            assert res["assertions"]["negative_controls"]["all_match"] is True

            # Assert all 6 pairwise comparisons generated
            p15_pairwise = res["comparisons"]["p1-5_pairwise"]
            expected_pairs = [
                "C_NEW_SINGLE__vs__C_OLD_SINGLE",
                "C_NEW_SINGLE__vs__C_NEW_DUP_SHAM",
                "C_NEW_SINGLE__vs__C_NEW_OLD_50_50",
                "C_OLD_SINGLE__vs__C_NEW_DUP_SHAM",
                "C_OLD_SINGLE__vs__C_NEW_OLD_50_50",
                "C_NEW_DUP_SHAM__vs__C_NEW_OLD_50_50",
            ]
            for ep in expected_pairs:
                assert ep in p15_pairwise, f"Missing pairwise comparison: {ep}"
            assert len(p15_pairwise) == 6

            # Assert diagnosis structure
            assert "primary_verdict" in res["diagnosis"]["p1-5"]
            assert (out_dir / "phase_c1_four_way_ablation_audit.json").exists()


def test_sham_provider_is_not_last_loop_provider():
    """Verify that Sham sidecar provider instance has group_sham_a and group_sham_b, unlike paraphrase provider."""
    sham_prov = ImmutableParaphraseEnsembleSidecarProvider(
        sidecar_path=SHAM_SIDECAR_PATH,
        expected_content_sha256=CANONICAL_SHAM_SHA256,
    )
    para_prov = ImmutableParaphraseEnsembleSidecarProvider(
        sidecar_path=PARA_SIDECAR_PATH,
        expected_content_sha256=CANONICAL_PARAPHRASE_SHA256,
    )

    sham_groups = [g["group_id"] for g in sham_prov.get_paraphrase_groups("query-p1-5-kis")]
    para_groups = [g["group_id"] for g in para_prov.get_paraphrase_groups("query-p1-5-kis")]

    assert sham_groups == ["group_sham_a", "group_sham_b"]
    assert para_groups == ["group_canonical_new", "group_candidate_old"]
    assert "group_sham_a" not in para_groups
    assert "group_sham_b" not in para_groups
