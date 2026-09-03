"""Unit & Regression Tests for KIS V2-A.3 Release Candidate 1 (KIS_V2A_RC1) E2E Closure."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from system_tai.kis.session_engine import OperationalKISRuntime
from system_tai.kis.session_schema import KISVideoFirstConfig, SessionConfig
from system_tai.retrieval.multi_query import WeightedRRFRetriever
from system_tai.retrieval.video_evidence import (
    FullCorpusVideoMaximaOutcome,
    RestrictedFrameHit,
    VideoMaximumHit,
    VideoRestrictedSearchOutcome,
)
from system_tai.translation.provider import TokenBudgetGuard
from system_tai.translation.sidecar_provider import canonical_sidecar_sha256
from scratch.run_kis_v2a_rc1_e2e_closure import (
    CANONICAL_GOLDEN_DIGESTS_SHA256,
    CANONICAL_MANUAL_REF_SHA256,
    CANONICAL_QUERY_MANIFEST_SHA256,
    CANONICAL_TNEW_SHA256,
    CANONICAL_TOLD_SHA256,
    EXPECTED_FULL_CORPUS_FINGERPRINT,
    EXPECTED_OPENAI_CLIP_COMMIT,
    RELEASE_CANDIDATE_ID,
    get_git_commit_sha,
    run_kis_v2a_rc1_e2e_closure,
)

TNEW_SIDECAR_PATH = REPO_ROOT / "scratch" / "benchmarks" / "translation_ablation" / "translation_p1_focus_v2_new.json"
TOLD_SIDECAR_PATH = REPO_ROOT / "scratch" / "benchmarks" / "translation_ablation" / "translation_p1_focus_v1_old_candidate.json"
QUERY_MANIFEST_PATH = REPO_ROOT / "systems" / "system_tai" / "benchmarks" / "frozen_kis_v2a_stress_manifest.json"
MANUAL_REF_PATH = REPO_ROOT / "systems" / "system_tai" / "benchmarks" / "manual_kis_reference_v1.json"
GOLDEN_DIGESTS_PATH = REPO_ROOT / "systems" / "system_tai" / "benchmarks" / "golden_phase_c1_c_new_single_digests.json"


def test_rc1_frozen_config_exact_contract():
    """Verify that the RC1 frozen config matches the strict release specifications."""
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

    assert vf_cfg.v2_adaptive_enabled is True
    assert vf_cfg.selected_video_cap == 64
    assert vf_cfg.top_m_evidence_cap == 5
    assert vf_cfg.top_m_weights == (0.4, 0.25, 0.15, 0.1, 0.1)
    assert vf_cfg.internal_rrf_candidate_depth == 1000
    assert vf_cfg.enable_top_video_local_anchor is False
    assert vf_cfg.enable_paraphrase_ensemble is False


def test_rc1_device_not_cpu_raises_fail_closed():
    """Verify that device != 'cpu' raises ValueError fail-closed."""
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        with pytest.raises(ValueError, match="device must be 'cpu'"):
            run_kis_v2a_rc1_e2e_closure(
                query_manifest_path=QUERY_MANIFEST_PATH,
                input_root=tdp,
                manifest_cache_path=tdp / "manifest_cache.json",
                output_root=tdp / "out",
                tnew_sidecar_path=TNEW_SIDECAR_PATH,
                manual_ref_path=MANUAL_REF_PATH,
                expected_commit="dummy_commit",
                device="cuda",
            )


def test_rc1_general_policy_requires_strict_corpus_gate():
    """Verify that policy 'general' strictly requires strict_corpus_gate=True."""
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        with pytest.raises(ValueError, match="strictly requires --strict-corpus-gate"):
            run_kis_v2a_rc1_e2e_closure(
                query_manifest_path=QUERY_MANIFEST_PATH,
                input_root=tdp,
                manifest_cache_path=tdp / "manifest_cache.json",
                output_root=tdp / "out",
                tnew_sidecar_path=TNEW_SIDECAR_PATH,
                manual_ref_path=MANUAL_REF_PATH,
                expected_commit="dummy_commit",
                policy="general",
                strict_corpus_gate=False,
                device="cpu",
            )


def test_rc1_mismatched_commit_raises_fail_closed():
    """Verify that mismatched git commit raises AssertionError."""
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        with pytest.raises(AssertionError, match="Active git commit .* does not match"):
            run_kis_v2a_rc1_e2e_closure(
                query_manifest_path=QUERY_MANIFEST_PATH,
                input_root=tdp,
                manifest_cache_path=tdp / "manifest_cache.json",
                output_root=tdp / "out",
                tnew_sidecar_path=TNEW_SIDECAR_PATH,
                manual_ref_path=MANUAL_REF_PATH,
                expected_commit="0000000000000000000000000000000000000000",
                device="cpu",
            )


def test_rc1_policy_selection_requires_told_sidecar():
    """Verify that policy 'benchmark_tuned' fails fast if no valid told_sidecar is provided."""
    git_sha = get_git_commit_sha()
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        with pytest.raises(FileNotFoundError, match="requires a valid --told-sidecar path"):
            run_kis_v2a_rc1_e2e_closure(
                query_manifest_path=QUERY_MANIFEST_PATH,
                input_root=tdp,
                manifest_cache_path=tdp / "manifest_cache.json",
                output_root=tdp / "out",
                tnew_sidecar_path=TNEW_SIDECAR_PATH,
                manual_ref_path=MANUAL_REF_PATH,
                expected_commit=git_sha,
                told_sidecar_path=None,
                policy="benchmark_tuned",
                strict_corpus_gate=False,
                device="cpu",
            )


def test_rc1_golden_fixture_canonical_sha_integrity():
    """Verify that golden_phase_c1_c_new_single_digests.json matches canonical SHA256."""
    assert GOLDEN_DIGESTS_PATH.exists()
    computed_sha = canonical_sidecar_sha256(GOLDEN_DIGESTS_PATH)
    assert computed_sha == CANONICAL_GOLDEN_DIGESTS_SHA256

    data = json.loads(GOLDEN_DIGESTS_PATH.read_text(encoding="utf-8"))
    assert data["provenance"]["full_corpus_fingerprint"] == EXPECTED_FULL_CORPUS_FINGERPRINT
    assert data["provenance"]["source_phase_c1_commit"] == "184bd7dece0f719eecb9641a2aa7b5a0f88eee3b"
    assert len(data["digests"]) == 5
    assert data["target_ranks"]["query-p1-2-kis"] == 2  # Verified rank #2


def test_rc1_custom_golden_path_tampered_sha_raises():
    """Verify that passing an arbitrary golden digests file with wrong SHA raises AssertionError."""
    git_sha = get_git_commit_sha()
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        tampered_golden = tdp / "tampered_golden.json"
        tampered_golden.write_text(json.dumps({"tampered": True}), encoding="utf-8")

        with pytest.raises(AssertionError, match="Golden digests fixture canonical SHA mismatch"):
            run_kis_v2a_rc1_e2e_closure(
                query_manifest_path=QUERY_MANIFEST_PATH,
                input_root=tdp,
                manifest_cache_path=tdp / "manifest_cache.json",
                output_root=tdp / "out",
                tnew_sidecar_path=TNEW_SIDECAR_PATH,
                manual_ref_path=MANUAL_REF_PATH,
                expected_commit=git_sha,
                golden_digests_path=tampered_golden,
                told_sidecar_path=TOLD_SIDECAR_PATH,
                policy="benchmark_tuned",
                strict_corpus_gate=False,
                device="cpu",
            )


def test_rc1_clip_commit_mismatch_raises_fail_closed():
    """Verify that an installed CLIP commit other than EXPECTED_OPENAI_CLIP_COMMIT raises AssertionError."""
    git_sha = get_git_commit_sha()
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        with patch("scratch.run_kis_v2a_rc1_e2e_closure.get_installed_clip_commit", return_value="1111111111111111111111111111111111111111"):
            with pytest.raises(AssertionError, match="Installed OpenAI CLIP commit mismatch"):
                run_kis_v2a_rc1_e2e_closure(
                    query_manifest_path=QUERY_MANIFEST_PATH,
                    input_root=tdp,
                    manifest_cache_path=tdp / "manifest_cache.json",
                    output_root=tdp / "out",
                    tnew_sidecar_path=TNEW_SIDECAR_PATH,
                    manual_ref_path=MANUAL_REF_PATH,
                    expected_commit=git_sha,
                    golden_digests_path=GOLDEN_DIGESTS_PATH,
                    told_sidecar_path=TOLD_SIDECAR_PATH,
                    policy="benchmark_tuned",
                    strict_corpus_gate=False,
                    device="cpu",
                )


def test_rc1_e2e_closure_runner_smoke_with_mock_bootstrap():
    """Verify run_kis_v2a_rc1_e2e_closure executes two clean sessions and asserts full determinism."""
    git_sha = get_git_commit_sha()

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
            fingerprint=EXPECTED_FULL_CORPUS_FINGERPRINT,
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
        out_dir = tdp / "rc1_closure_out"

        query_manifest = QUERY_MANIFEST_PATH
        tnew_sidecar = TNEW_SIDECAR_PATH
        told_sidecar = TOLD_SIDECAR_PATH
        manual_ref = MANUAL_REF_PATH

        sessions_bootstrapped = []

        def mock_bootstrap(config, *, translation_provider=None, **kwargs):
            sessions_bootstrapped.append((config.output_root, translation_provider))
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
            res = run_kis_v2a_rc1_e2e_closure(
                query_manifest_path=query_manifest,
                input_root=tdp,
                manifest_cache_path=tdp / "manifest_cache.json",
                output_root=out_dir,
                tnew_sidecar_path=tnew_sidecar,
                manual_ref_path=manual_ref,
                expected_commit=git_sha,
                golden_digests_path=GOLDEN_DIGESTS_PATH,
                told_sidecar_path=told_sidecar,
                policy="benchmark_tuned",
                strict_corpus_gate=False,
                device="cpu",
            )

            # Assert 2 clean sessions executed with distinct providers
            assert len(sessions_bootstrapped) == 2
            out_1, prov_1 = sessions_bootstrapped[0]
            out_2, prov_2 = sessions_bootstrapped[1]
            assert out_1 == out_dir / "run_1"
            assert out_2 == out_dir / "run_2"
            assert prov_1 is not prov_2

            # Assert Bit-Exact Parity
            assert res["verification_gates"]["two_pass_projection_bit_exact"] is True
            assert res["verification_gates"]["two_pass_selected_videos_bit_exact"] is True

            # Assert Release Manifest exists
            manifest_file = out_dir / "kis_v2a_rc1_closure_manifest.json"
            assert manifest_file.exists()
            manifest_data = json.loads(manifest_file.read_text(encoding="utf-8"))
            assert manifest_data["release_candidate"] == RELEASE_CANDIDATE_ID
            # In mock smoke mode (strict_corpus_gate=False), it is correctly marked NOT release-qualified
            assert manifest_data["release_qualified"] is False
            assert manifest_data["is_production_default"] is False
            assert manifest_data["policy"] == "EXPERIMENTAL_BENCHMARK_TUNED"
            assert len(manifest_data["queries"]) == 5
            for qid, qinfo in manifest_data["queries"].items():
                assert qinfo["two_pass_projection_bit_exact"] is True
                assert qinfo["two_pass_selected_seq_bit_exact"] is True
                assert qinfo["canonical_projection_digest"] is not None
