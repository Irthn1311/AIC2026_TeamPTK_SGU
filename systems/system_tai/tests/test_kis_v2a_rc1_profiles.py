"""Comprehensive verification suite for official KIS_V2A_RC1 Replay profile.

Covers:
- Profile factory and single source of truth validation
- Packaged sidecar resource integrity
- Bootstrap fail-closed enforcement (no silent fallback)
- CLI argument resolution (defaults vs explicit conflicts)
- Session manifest enrichment
- Wheel packaging verification
- CLI JSONL execution and unknown query fail-fast rejection
"""

from __future__ import annotations

import dataclasses
import io
import json
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from system_tai.kis.profiles import (
    CANONICAL_PORTABLE_CORPUS_FINGERPRINT,
    CANONICAL_RC1_TNEW_SHA256,
    EXPECTED_CLIP_CHECKPOINT_SHA256,
    EXPECTED_FULL_CORPUS_FINGERPRINT,
    EXPECTED_OPENAI_CLIP_COMMIT,
    KIS_V2A_RC1_REPLAY_PROFILE_NAME,
    OFFICIAL_RC1_REPLAY_SIDECAR,
    apply_kis_v2a_rc1_replay_profile,
    get_kis_v2a_rc1_replay_translation_provider,
    get_kis_v2a_rc1_video_first_config,
    validate_kis_v2a_rc1_replay_config,
)
from system_tai.kis.session import build_parser, run_session, session_config_from_args
from system_tai.kis.session_engine import OperationalKISRuntime
from system_tai.kis.session_schema import SessionConfig
from system_tai.translation.sidecar_provider import (
    ImmutableSidecarTranslationProvider,
    TranslationError,
    canonical_sidecar_sha256,
)


def test_rc1_replay_profile_contract_and_resource_sha():
    """Verify get_kis_v2a_rc1_video_first_config and packaged resource SHA."""
    vf = get_kis_v2a_rc1_video_first_config()
    assert vf.enabled is True
    assert vf.v2_adaptive_enabled is True
    assert vf.selected_video_cap == 64
    assert vf.video_nomination_depth == 100
    assert vf.restricted_frames_per_video_per_variant == 20
    assert vf.top_m_evidence_cap == 5
    assert vf.top_m_weights == (0.4, 0.25, 0.15, 0.1, 0.1)
    assert vf.internal_rrf_candidate_depth == 1000
    assert vf.enable_top_video_local_anchor is False
    assert vf.enable_paraphrase_ensemble is False

    assert OFFICIAL_RC1_REPLAY_SIDECAR.is_file()
    csha = canonical_sidecar_sha256(OFFICIAL_RC1_REPLAY_SIDECAR)
    assert csha == CANONICAL_RC1_TNEW_SHA256

    provider = get_kis_v2a_rc1_replay_translation_provider()
    assert isinstance(provider, ImmutableSidecarTranslationProvider)
    assert provider.sidecar_path == OFFICIAL_RC1_REPLAY_SIDECAR


def test_rc1_replay_config_validation_and_rejection():
    """Verify single source of truth validator accepts valid and rejects tampered configurations."""
    import dataclasses
    base = SessionConfig()
    rc1_cfg = apply_kis_v2a_rc1_replay_profile(base)
    # Valid config passes cleanly
    validate_kis_v2a_rc1_replay_config(rc1_cfg)
    assert rc1_cfg.chunk_size == 4096
    assert rc1_cfg.default_refine_top_n == 3
    assert rc1_cfg.refinement_config.top_candidates_to_refine == 20

    # Rejection of device != cpu
    with pytest.raises(ValueError, match="strictly requires device='cpu'"):
        validate_kis_v2a_rc1_replay_config(dataclasses.replace(rc1_cfg, device="cuda"))

    # Rejection of allow_model_download=True
    with pytest.raises(ValueError, match="strictly requires allow_model_download=False"):
        validate_kis_v2a_rc1_replay_config(dataclasses.replace(rc1_cfg, allow_model_download=True))

    # Tampered cap
    tampered_vf = dataclasses.replace(rc1_cfg.kis_video_first_config, selected_video_cap=32)
    with pytest.raises(ValueError, match="strictly requires exact KISVideoFirstConfig match"):
        validate_kis_v2a_rc1_replay_config(dataclasses.replace(rc1_cfg, kis_video_first_config=tampered_vf))

    # Tampered default_refine_top_n
    with pytest.raises(ValueError, match="strictly requires default_refine_top_n=3"):
        validate_kis_v2a_rc1_replay_config(dataclasses.replace(rc1_cfg, default_refine_top_n=0))

    # Tampered chunk_size
    with pytest.raises(ValueError, match="strictly requires chunk_size=4096"):
        validate_kis_v2a_rc1_replay_config(dataclasses.replace(rc1_cfg, chunk_size=256))

    # Tampered refinement config fields (e.g. window_before_seconds)
    with pytest.raises(ValueError, match="strictly requires exact RefinementConfig match"):
        tampered_ref = dataclasses.replace(rc1_cfg.refinement_config, window_before_seconds=99.0)
        validate_kis_v2a_rc1_replay_config(dataclasses.replace(rc1_cfg, refinement_config=tampered_ref))

    # Tampered top_candidates_to_refine
    with pytest.raises(ValueError, match="strictly requires exact RefinementConfig match"):
        tampered_ref2 = dataclasses.replace(rc1_cfg.refinement_config, top_candidates_to_refine=1)
        validate_kis_v2a_rc1_replay_config(dataclasses.replace(rc1_cfg, refinement_config=tampered_ref2))


def test_rc1_replay_request_validation_fail_closed():
    """Verify validate_kis_v2a_rc1_replay_request rejects parameter tampering in request JSONL."""
    from system_tai.kis.profiles import validate_kis_v2a_rc1_replay_request
    from system_tai.kis.session_schema import QueryRequest

    # Valid request passes
    valid_req = QueryRequest(
        request_id="req-1",
        query_id="query-p1-1-kis",
        query_vi="test query",
        output_top_k=100,
        top_k_per_variant=100,
        refine_top_n=3,
    )
    validate_kis_v2a_rc1_replay_request(valid_req)

    # Tampered output_top_k=50 raises ValueError
    bad_output_k = dataclasses.replace(valid_req, output_top_k=50)
    with pytest.raises(ValueError, match="strictly requires request output_top_k=100"):
        validate_kis_v2a_rc1_replay_request(bad_output_k)

    # Tampered top_k_per_variant=50 raises ValueError
    bad_variant_k = dataclasses.replace(valid_req, top_k_per_variant=50)
    with pytest.raises(ValueError, match="strictly requires request top_k_per_variant=100"):
        validate_kis_v2a_rc1_replay_request(bad_variant_k)

    # Tampered refine_top_n=0 raises ValueError
    bad_refine_n = dataclasses.replace(valid_req, refine_top_n=0)
    with pytest.raises(ValueError, match="strictly requires request refine_top_n=3"):
        validate_kis_v2a_rc1_replay_request(bad_refine_n)

    # Manual English variant prohibited
    bad_en = dataclasses.replace(valid_req, query_en="manual translation")
    with pytest.raises(ValueError, match="accepts Vietnamese input only"):
        validate_kis_v2a_rc1_replay_request(bad_en)


def test_rc1_replay_bootstrap_requires_provider_fail_closed():
    """Verify OperationalKISRuntime.bootstrap enforces explicit provider, model check, and environment."""
    base = SessionConfig()
    rc1_cfg = apply_kis_v2a_rc1_replay_profile(base)

    # Missing provider
    with pytest.raises(ValueError, match="Profile 'kis-v2a-rc1-replay' strictly requires an explicit ImmutableSidecarTranslationProvider"):
        OperationalKISRuntime.bootstrap(config=rc1_cfg, translation_provider=None)

    # Tampered provider sidecar hash
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json") as f:
        f.write('{"tampered": true}')
        tampered_path = Path(f.name)

    try:
        tampered_provider = ImmutableSidecarTranslationProvider(
            sidecar_path=tampered_path,
        )
        mock_manifest = MagicMock(
            schema_version="1.0.0",
            fingerprint=EXPECTED_FULL_CORPUS_FINGERPRINT,
            videos=tuple(MagicMock(video_id=f"v_{i}", raw_video_path=None) for i in range(873)),
            write=MagicMock(),
        )
        fake_model_prov = {
            "clip_source_commit": EXPECTED_OPENAI_CLIP_COMMIT,
            "checkpoint_path": "/fake/ViT-B-32.pt",
            "checkpoint_sha256": EXPECTED_CLIP_CHECKPOINT_SHA256,
            "verified_bit_exact": True,
        }
        with patch("system_tai.kis.session_engine.discover_corpus", return_value=mock_manifest), \
             patch("system_tai.kis.profiles.validate_kis_v2a_rc1_replay_model_pre_bootstrap", return_value=fake_model_prov):
            mock_registry = MagicMock(
                embedding_dimension=512,
                total_rows=177321,
                stores=tuple(MagicMock() for _ in range(873)),
                get_store=lambda x: None,
            )
            with pytest.raises(AssertionError, match="Sidecar canonical SHA256 mismatch"):
                OperationalKISRuntime.bootstrap(
                    config=rc1_cfg,
                    translation_provider=tampered_provider,
                    registry_loader=lambda path: mock_registry,
                    encoder_factory=MagicMock(),
                )
    finally:
        if tampered_path.exists():
            tampered_path.unlink()


def test_cli_argparse_profile_default_vs_explicit():
    """Verify CLI accepts bare profile or matching explicit flags, and rejects conflicting flags."""
    parser = build_parser()

    # 1. Bare profile -> successfully builds frozen RC1 configuration
    args_bare = parser.parse_args(["--profile", "kis-v2a-rc1-replay"])
    cfg_bare = session_config_from_args(args_bare)
    assert cfg_bare.profile_name == KIS_V2A_RC1_REPLAY_PROFILE_NAME
    assert cfg_bare.device == "cpu"
    assert cfg_bare.default_refine_top_n == 3
    assert cfg_bare.chunk_size == 4096
    assert cfg_bare.refinement_config.top_candidates_to_refine == 20
    assert cfg_bare.kis_video_first_config.selected_video_cap == 64
    assert cfg_bare.kis_video_first_config.internal_rrf_candidate_depth == 1000

    # 2. Explicit matching flags -> accepted
    args_matching = parser.parse_args([
        "--profile", "kis-v2a-rc1-replay",
        "--device", "cpu",
        "--kis-selected-video-cap", "64",
        "--rrf-constant", "60.0",
        "--default-output-top-k", "100",
        "--chunk-size", "4096",
        "--default-refine-top-n", "3",
    ])
    cfg_matching = session_config_from_args(args_matching)
    assert cfg_matching.device == "cpu"
    assert cfg_matching.chunk_size == 4096
    assert cfg_matching.default_refine_top_n == 3
    assert cfg_matching.refinement_config.top_candidates_to_refine == 20
    assert cfg_matching.kis_video_first_config.selected_video_cap == 64

    # 3. Explicit conflicting device -> raises ValueError
    args_bad_dev = parser.parse_args(["--profile", "kis-v2a-rc1-replay", "--device", "cuda"])
    with pytest.raises(ValueError, match="strictly requires --device cpu"):
        session_config_from_args(args_bad_dev)

    # 4. Explicit conflicting cap -> raises ValueError
    args_bad_cap = parser.parse_args(["--profile", "kis-v2a-rc1-replay", "--kis-selected-video-cap", "32"])
    with pytest.raises(ValueError, match="strictly requires --kis-selected-video-cap 64"):
        session_config_from_args(args_bad_cap)

    # 5. Explicit conflicting local anchor -> raises ValueError
    args_bad_anchor = parser.parse_args(["--profile", "kis-v2a-rc1-replay", "--enable-kis-multi-anchor-refinement"])
    with pytest.raises(ValueError, match="strictly requires local anchor refinement to be OFF"):
        session_config_from_args(args_bad_anchor)


def test_session_manifest_records_rc1_profile_metadata():
    """Verify _save_session_manifest records profile_name, provider identity, model_provenance, and config."""
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        base = SessionConfig(output_root=tdp)
        rc1_cfg = apply_kis_v2a_rc1_replay_profile(base)
        provider = get_kis_v2a_rc1_replay_translation_provider()

        mock_runtime = MagicMock()
        mock_runtime.session_id = "test-rc1-manifest"
        mock_runtime.start_time_utc = "2026-09-03T12:00:00Z"
        mock_runtime.config = rc1_cfg
        mock_runtime.output_root = tdp
        mock_runtime.manifest_path = tdp / "feature_manifest.json"
        mock_runtime.manifest = MagicMock(schema_version="1.0.0", fingerprint=EXPECTED_FULL_CORPUS_FINGERPRINT, videos=tuple(range(873)))
        mock_runtime.registry = MagicMock(total_rows=177321)
        mock_runtime.raw_video_registry = MagicMock(records=())
        mock_runtime.shared_encoder = MagicMock(identifiers={"device": "cpu", "model": "ViT-B/32"})
        mock_runtime.translation_provider = provider
        mock_runtime.bootstrap_timings = {}
        mock_runtime.model_provenance = {
            "clip_source_commit": EXPECTED_OPENAI_CLIP_COMMIT,
            "checkpoint_path": "/fake/ViT-B-32.pt",
            "checkpoint_sha256": EXPECTED_CLIP_CHECKPOINT_SHA256,
            "verified_bit_exact": True,
        }
        mock_runtime._request_count = 0
        mock_runtime._successful_query_count = 0
        mock_runtime._failed_query_count = 0
        mock_runtime._health_request_count = 0
        mock_runtime._malformed_request_count = 0

        OperationalKISRuntime._save_session_manifest(mock_runtime, shutdown_reason="test_done")

        manifest_file = tdp / "session_manifest.json"
        assert manifest_file.is_file()
        data = json.loads(manifest_file.read_text(encoding="utf-8"))
        assert data["profile_name"] == "kis-v2a-rc1-replay"
        assert data["translation_provider_mode"] == "immutable_sidecar"
        assert data["model_provenance"]["clip_source_commit"] == EXPECTED_OPENAI_CLIP_COMMIT
        assert data["model_provenance"]["checkpoint_sha256"] == EXPECTED_CLIP_CHECKPOINT_SHA256
        assert data["translation_provider_identity"]["type"] == "ImmutableSidecarTranslationProvider"
        assert data["translation_provider_identity"]["sidecar_sha256"] == CANONICAL_RC1_TNEW_SHA256
        assert data["kis_video_first_config"]["selected_video_cap"] == 64
        assert data["kis_video_first_config"]["internal_rrf_candidate_depth"] == 1000


def test_wheel_package_contains_official_sidecar_resource():
    """Verify building setuptools wheel dynamically into temporary directory includes translation_p1_focus_v2_new.json."""
    with tempfile.TemporaryDirectory() as td:
        repo_root = Path(__file__).resolve().parent.parent.parent.parent
        package_dir = repo_root / "systems" / "system_tai"
        out_dir = Path(td) / "wheel_dist"
        out_dir.mkdir(parents=True, exist_ok=True)

        subprocess.run(
            [sys.executable, "-m", "build", "--wheel", str(package_dir), "--outdir", str(out_dir)],
            check=True,
            capture_output=True,
        )
        wheels = list(out_dir.glob("*.whl"))
        assert len(wheels) > 0, "No built wheel produced in temporary directory!"

        latest_wheel = wheels[0]
        with zipfile.ZipFile(latest_wheel, "r") as z:
            names = z.namelist()
            expected_resource = "system_tai/kis/resources/translation_p1_focus_v2_new.json"
            assert expected_resource in names, f"Resource {expected_resource} not packaged in wheel! Wheel contents: {names}"


def test_cli_run_session_with_mock_runtime_injects_rc1_provider():
    """Verify run_session with --profile kis-v2a-rc1-replay injects provider into bootstrap."""
    parser = build_parser()
    args = parser.parse_args(["--profile", "kis-v2a-rc1-replay"])

    captured_bootstrap_kwargs = {}

    def mock_bootstrap(config, *, translation_provider=None, **kwargs):
        captured_bootstrap_kwargs["config"] = config
        captured_bootstrap_kwargs["translation_provider"] = translation_provider
        rt = MagicMock()
        rt.session_id = "test-mock-session"
        rt.handle_health.return_value = {"type": "health", "request_id": "req-1", "status": "ok"}
        rt.handle_shutdown.return_value = {"type": "shutdown", "request_id": "req-2", "status": "ok"}
        rt.close = MagicMock()
        return rt

    in_stream = io.StringIO('{"type": "health", "request_id": "req-1"}\n{"type": "shutdown", "request_id": "req-2"}\n')
    out_stream = io.StringIO()
    err_stream = io.StringIO()

    with patch.object(OperationalKISRuntime, "bootstrap", side_effect=mock_bootstrap):
        ret = run_session(args, stdin=in_stream, stdout=out_stream, stderr=err_stream)

    assert ret == 0
    assert captured_bootstrap_kwargs["config"].profile_name == KIS_V2A_RC1_REPLAY_PROFILE_NAME
    assert isinstance(captured_bootstrap_kwargs["translation_provider"], ImmutableSidecarTranslationProvider)
    assert captured_bootstrap_kwargs["translation_provider"].expected_content_sha256 == CANONICAL_RC1_TNEW_SHA256


def test_cli_unknown_query_rejected_fail_closed():
    """Condition 4: Verify unknown query in separate session causes TranslationError via handle_query and exits with code 1."""
    parser = build_parser()
    args = parser.parse_args(["--profile", "kis-v2a-rc1-replay"])

    mock_rt = MagicMock()
    mock_rt.session_id = "test-fail-fast"
    mock_rt._request_count = 1
    # CLI calls runtime.handle_query(request)
    mock_rt.handle_query.side_effect = TranslationError("Query 'query-unseen-6' not found in immutable sidecar")
    mock_rt.handle_error.return_value = {
        "type": "error",
        "request_id": "req-unknown",
        "error_code": "QUERY_EXECUTION_FAILED",
        "error_type": "TranslationError",
        "message": "TranslationError: Sidecar translation miss for query_id='query-unseen-6', text_sha='abc'",
        "session_continues": False,
    }

    in_stream = io.StringIO('{"type": "query", "request_id": "req-unknown", "query_id": "query-unseen-6", "query_vi": "Một cảnh quay chưa từng thấy trong benchmark"}\n')
    out_stream = io.StringIO()
    err_stream = io.StringIO()

    ret = run_session(args, runtime=mock_rt, stdin=in_stream, stdout=out_stream, stderr=err_stream)
    assert ret == 1, "Session must exit with code 1 upon query execution failure under fail-fast protocol!"
    out_text = out_stream.getvalue()
    assert "QUERY_EXECUTION_FAILED" in out_text
    assert "TranslationError" in out_text


def test_cli_argparse_rc1_replay_refinement_flags_rejected_fail_fast():
    """Verify that passing non-default refinement flags under kis-v2a-rc1-replay is rejected fail-fast."""
    parser = build_parser()

    bad_flags = [
        (["--window-before-seconds", "99"], "strictly requires --window-before-seconds 5.0"),
        (["--window-after-seconds", "99"], "strictly requires --window-after-seconds 5.0"),
        (["--coarse-stride-frames", "999"], "strictly requires --coarse-stride-frames 15"),
        (["--coarse-top-n", "10"], "strictly requires --coarse-top-n 3"),
        (["--fine-radius-frames", "50"], "strictly requires --fine-radius-frames 30"),
        (["--fine-stride-frames", "5"], "strictly requires --fine-stride-frames 1"),
        (["--image-batch-size", "7"], "strictly requires --image-batch-size 32"),
        (["--max-decoded-frames-per-candidate", "100"], "strictly requires --max-decoded-frames-per-candidate 500"),
        (["--missing-raw-video-policy", "fail-query"], "strictly requires --missing-raw-video-policy keep-original"),
        (["--candidate-failure-policy", "fail-query"], "strictly requires --candidate-failure-policy keep-original"),
        (["--coarse-decode-strategy", "sparse-verified"], "strictly requires --coarse-decode-strategy sequential"),
        (["--continue-on-request-error"], "strictly requires fail-fast protocol"),
        (["--translation-allow-model-download"], "strictly requires translation_allow_model_download=False"),
        (["--translation-device", "cpu"], "strictly requires translation_device 'auto'"),
        (["--translation-device", "cuda"], "strictly requires translation_device 'auto'"),
        (["--translation-cache-dir", "/tmp/cache"], "does not allow specifying translation_cache_dir"),
        (["--kis-visual-verifier-allow-model-download"], "strictly requires kis_visual_verifier_allow_model_download=False"),
        # 5 anchor dormant tuning flags
        (["--kis-anchor-video-rank-cap", "99"], "strictly requires default --kis-anchor-video-rank-cap 20"),
        (["--kis-anchor-max-videos", "99"], "strictly requires default --kis-anchor-max-videos 5"),
        (["--kis-anchors-per-video", "99"], "strictly requires default --kis-anchors-per-video 6"),
        (["--kis-anchor-min-gap-seconds", "99.0"], "strictly requires default --kis-anchor-min-gap-seconds 2.0"),
        (["--kis-max-extra-raw-anchors", "99"], "strictly requires default --kis-max-extra-raw-anchors 12"),
        # 5 timeline dormant tuning flags
        (["--kis-timeline-max-videos", "99"], "strictly requires default --kis-timeline-max-videos 3"),
        (["--kis-timeline-sample-stride-seconds", "99.0"], "strictly requires default --kis-timeline-sample-stride-seconds 1.0"),
        (["--kis-timeline-max-samples-per-video", "99"], "strictly requires default --kis-timeline-max-samples-per-video 300"),
        (["--kis-timeline-max-regions-per-video", "99"], "strictly requires default --kis-timeline-max-regions-per-video 3"),
        (["--kis-timeline-min-region-gap-seconds", "99.0"], "strictly requires default --kis-timeline-min-region-gap-seconds 5.0"),
        # 10 verifier dormant tuning flags
        (["--kis-visual-verifier-model", "fake-vlm"], "strictly requires default --kis-visual-verifier-model"),
        (["--kis-visual-verifier-revision", "v2"], "does not allow specifying --kis-visual-verifier-revision"),
        (["--kis-visual-verifier-cache-dir", "/tmp/vlm"], "does not allow specifying --kis-visual-verifier-cache-dir"),
        (["--kis-visual-verifier-shortlist-per-video", "99"], "strictly requires default --kis-visual-verifier-shortlist-per-video 32"),
        (["--kis-visual-verifier-coverage-bins", "99"], "strictly requires default --kis-visual-verifier-coverage-bins 12"),
        (["--kis-visual-verifier-temporal-evidence-window-seconds", "99.0"], "strictly requires default --kis-visual-verifier-temporal-evidence-window-seconds 6.0"),
        (["--kis-visual-verifier-neighbor-radius", "99"], "strictly requires default --kis-visual-verifier-neighbor-radius 1"),
        (["--kis-visual-verifier-max-new-tokens", "99"], "strictly requires default --kis-visual-verifier-max-new-tokens 512"),
        (["--kis-visual-verifier-execution-mode", "cpu-fast"], "strictly requires default --kis-visual-verifier-execution-mode 'auto'"),
        (["--kis-visual-verifier-failure-policy", "fail-query"], "strictly requires default --kis-visual-verifier-failure-policy 'fallback-clip'"),
    ]
    for flag_args, match_msg in bad_flags:
        args = parser.parse_args(["--profile", "kis-v2a-rc1-replay"] + flag_args)
        with pytest.raises(ValueError, match=match_msg):
            session_config_from_args(args)


def test_validate_kis_v2a_rc1_replay_config_bit_exact_subconfigs():
    """Verify validate_kis_v2a_rc1_replay_config rejects non-default disabled subconfigs."""
    from system_tai.kis.profiles import apply_kis_v2a_rc1_replay_profile, validate_kis_v2a_rc1_replay_config
    from system_tai.refinement.models import (
        SelectedVideoTimelineScoutConfig,
        SelectedVideoVisualVerifierConfig,
    )
    from system_tai.retrieval.video_restricted import VideoConditionedKeyframeConfig
    import dataclasses

    base = SessionConfig()
    cfg = apply_kis_v2a_rc1_replay_profile(base)
    validate_kis_v2a_rc1_replay_config(cfg)

    # 1. Non-default VideoConditionedKeyframeConfig (even if enabled=False)
    bad_anchor_cfg = dataclasses.replace(cfg, video_conditioned_keyframe_config=VideoConditionedKeyframeConfig(enabled=False, max_anchors_per_video=99))
    with pytest.raises(ValueError, match="strictly requires default VideoConditionedKeyframeConfig"):
        validate_kis_v2a_rc1_replay_config(bad_anchor_cfg)

    # 2. Non-default SelectedVideoTimelineScoutConfig (even if enabled=False)
    bad_scout_cfg = dataclasses.replace(cfg, selected_video_timeline_scout_config=SelectedVideoTimelineScoutConfig(enabled=False, max_videos=10))
    with pytest.raises(ValueError, match="strictly requires default SelectedVideoTimelineScoutConfig"):
        validate_kis_v2a_rc1_replay_config(bad_scout_cfg)

    # 3. Non-default SelectedVideoVisualVerifierConfig (even if enabled=False)
    bad_verifier_cfg = dataclasses.replace(cfg, selected_video_visual_verifier_config=SelectedVideoVisualVerifierConfig(enabled=False, shortlist_per_video=16))
    with pytest.raises(ValueError, match="strictly requires default SelectedVideoVisualVerifierConfig"):
        validate_kis_v2a_rc1_replay_config(bad_verifier_cfg)

    # 4. Non-default translation_max_clip_tokens
    bad_tokens_cfg = dataclasses.replace(cfg, translation_max_clip_tokens=74)
    with pytest.raises(ValueError, match="strictly requires translation_max_clip_tokens=75"):
        validate_kis_v2a_rc1_replay_config(bad_tokens_cfg)

    # 5. Non-default translation_device
    bad_dev_cfg = dataclasses.replace(cfg, translation_device="cpu")
    with pytest.raises(ValueError, match="strictly requires translation_device='auto'"):
        validate_kis_v2a_rc1_replay_config(bad_dev_cfg)

    # 6. Non-default translation_cache_dir
    bad_cache_cfg = dataclasses.replace(cfg, translation_cache_dir=Path("/tmp/cache"))
    with pytest.raises(ValueError, match="strictly requires translation_cache_dir=None"):
        validate_kis_v2a_rc1_replay_config(bad_cache_cfg)

    # 7. Non-default translation_revision
    bad_rev_cfg = dataclasses.replace(cfg, translation_revision="other")
    with pytest.raises(ValueError, match="strictly requires translation_revision="):
        validate_kis_v2a_rc1_replay_config(bad_rev_cfg)


def test_in_tree_qualification_runner_contract_and_manifest_schema():
    """Verify in-tree qualification runner constants and session_manifest schema alignment."""
    runner_path = (
        Path(__file__).resolve().parent.parent
        / "benchmarks"
        / "run_kis_v2a_rc1_replay_qualification.py"
    )
    assert runner_path.is_file(), f"Missing in-tree qualification runner at {runner_path}"

    import importlib.util
    spec = importlib.util.spec_from_file_location("qualification_runner", runner_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # 1. Assert runner constants match canonical frozen constants
    assert mod.EXPECTED_CLIP_COMMIT == EXPECTED_OPENAI_CLIP_COMMIT
    assert mod.EXPECTED_CLIP_CHECKPOINT_SHA256 == EXPECTED_CLIP_CHECKPOINT_SHA256
    assert mod.CANONICAL_PORTABLE_CORPUS_FINGERPRINT == CANONICAL_PORTABLE_CORPUS_FINGERPRINT
    assert mod.CANONICAL_ABSOLUTE_CORPUS_FINGERPRINT == EXPECTED_FULL_CORPUS_FINGERPRINT
    assert mod.HISTORICAL_RC1_COMMIT == "d3b2507b97af03ae9e7067b97e79ecc8488f551c"
    assert mod.HISTORICAL_REPLAY_TAG_COMMIT == "edfebede48f437479dfb03c7131aae64d863b240"
    assert mod.HISTORICAL_VALID_COMMITS == (
        "d3b2507b97af03ae9e7067b97e79ecc8488f551c",
        "edfebede48f437479dfb03c7131aae64d863b240",
    )
    assert mod.GOLDEN_DIGESTS == {
        "query-p1-1-kis": "1ec8d8c03122de1a9e3083a7addadcbcb4f845b2c18deec16d978baf72e19c3e",
        "query-p1-2-kis": "47a486ec387785ef95552642df7ed05542f2bfe80d37004dcc4a7287aca3219e",
        "query-p1-4-kis": "2d7f5ebacb8f040ed1c252feecfcbdee59e44550101a7754c76ebc1935f9770e",
        "query-p1-5-kis": "0eccbb6d600bb2945cd80fe0126e5a2a5106b414da171ce79fd87ca487e3cb62",
        "query-p1-6-kis": "9d4cd4ef703a106b515f087413d79a3755051aceaaf7be7de460511c9dfda6fb",
    }
    assert mod.EXPECTED_TARGET_COARSE_RANKS == {
        "query-p1-1-kis": 1,
        "query-p1-2-kis": 25,
        "query-p1-4-kis": 1,
        "query-p1-5-kis": 31,
        "query-p1-6-kis": 19,
    }
    assert mod.EXPECTED_TARGET_RANKS == {
        "query-p1-1-kis": 1,
        "query-p1-2-kis": 2,
        "query-p1-4-kis": 1,
        "query-p1-5-kis": 25,
        "query-p1-6-kis": 1,
    }
    assert mod.FROZEN_SELECTED_SEQUENCE_DIGESTS == {
        "query-p1-1-kis": "acf04f853f89070868f76fa9eec014b2d39aa4776100c5980ba9c32df46b1a20",
        "query-p1-2-kis": "86f875cfd66ecc1398c8c22731c36fe4a7faea825ee15d9aaee9db7fc8e5bfbb",
        "query-p1-4-kis": "393c06fb91975e47854d9c792376fb117bceca28ea03d65b7194fbe064c1264b",
        "query-p1-5-kis": "80eb8ad5c38b2211ea4cfb87b7da7dfcae7975bf22e70e5b746c863dd24501a3",
        "query-p1-6-kis": "193ca3cc581d484c9ecf80ee0e816a7f05c48b067a9cfd414a9ca8c772cb3f0e",
    }

    # Verify runner CLI required arguments & safety guard
    with pytest.raises(SystemExit):
        with patch.object(sys, "argv", ["runner", "--input-root", "/fake"]):
            with patch("sys.stderr", new=io.StringIO()):
                mod.main()

    with pytest.raises(ValueError, match="Safety guard: output_root cannot be /kaggle, /kaggle/input, or /kaggle/working directly"):
        with patch.object(
            sys,
            "argv",
            ["runner", "--expected-commit", "dummy", "--input-root", "/fake_in", "--output-root", "/kaggle/working"],
        ):
            mod.main()

    # 2. Verify session_manifest top-level fields match runner schema expectations
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        base = SessionConfig(output_root=tdp)
        rc1_cfg = apply_kis_v2a_rc1_replay_profile(base)
        provider = get_kis_v2a_rc1_replay_translation_provider()

        mock_runtime = MagicMock()
        mock_runtime.session_id = "test-schema"
        mock_runtime.start_time_utc = "2026-09-03T12:00:00Z"
        mock_runtime.config = rc1_cfg
        mock_runtime.output_root = tdp
        mock_runtime.manifest_path = tdp / "feature_manifest.json"
        mock_runtime.manifest = MagicMock(schema_version="1.0.0", fingerprint=CANONICAL_PORTABLE_CORPUS_FINGERPRINT, videos=tuple(range(873)))
        mock_runtime.registry = MagicMock(total_rows=177321)
        mock_runtime.raw_video_registry = MagicMock(records=())
        mock_runtime.shared_encoder = MagicMock(identifiers={"device": "cpu", "model": "ViT-B/32"})
        mock_runtime.translation_provider = provider
        mock_runtime.bootstrap_timings = {}
        mock_runtime.model_provenance = {
            "clip_source_commit": EXPECTED_OPENAI_CLIP_COMMIT,
            "checkpoint_path": "/fake/ViT-B-32.pt",
            "checkpoint_sha256": EXPECTED_CLIP_CHECKPOINT_SHA256,
            "verified_bit_exact": True,
        }
        mock_runtime._request_count = 0
        mock_runtime._successful_query_count = 0
        mock_runtime._failed_query_count = 0
        mock_runtime._health_request_count = 0
        mock_runtime._malformed_request_count = 0

        OperationalKISRuntime._save_session_manifest(mock_runtime, shutdown_reason="test_done")
        manifest_file = tdp / "session_manifest.json"
        data = json.loads(manifest_file.read_text(encoding="utf-8"))

        # Verify top-level fields asserted by runner
        assert data.get("manifest_fingerprint") == CANONICAL_PORTABLE_CORPUS_FINGERPRINT
        assert data.get("video_count") == 873
        assert data.get("feature_row_count") == 177321
        assert data.get("model_provenance", {}).get("verified_bit_exact") is True


def test_qualification_runner_safety_guards_and_deep_corpus_discovery():
    """Verify runner fail-safe guards (coupled manifest args, parent directory rmtree protection)

    and bounded deep corpus discovery (max_depth=4).
    """
    runner_path = (
        Path(__file__).resolve().parent.parent
        / "benchmarks"
        / "run_kis_v2a_rc1_replay_qualification.py"
    )
    import importlib.util
    spec = importlib.util.spec_from_file_location("qualification_runner_guard", runner_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # 1. Manifest path provided without SHA256 raises ValueError
    with pytest.raises(ValueError, match="must be provided together"):
        with patch.object(
            sys,
            "argv",
            [
                "runner",
                "--expected-commit", "dummy",
                "--input-root", "/fake_in",
                "--output-root", "/fake_out/sub",
                "--historical-manifest", "/fake/manifest.json",
            ],
        ):
            mod.main()

    # 2. SHA256 provided without manifest path raises ValueError
    with pytest.raises(ValueError, match="must be provided together"):
        with patch.object(
            sys,
            "argv",
            [
                "runner",
                "--expected-commit", "dummy",
                "--input-root", "/fake_in",
                "--output-root", "/fake_out/sub",
                "--historical-manifest-sha256", "0" * 64,
            ],
        ):
            mod.main()

    # 3. Output root is parent of repo_root raises ValueError
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        fake_repo = tdp / "workspace" / "repo"
        fake_repo.mkdir(parents=True)
        parent_out = tdp / "workspace"
        with pytest.raises(ValueError, match="Safety guard: output_root contains repo_root"):
            with patch.object(
                sys,
                "argv",
                [
                    "runner",
                    "--expected-commit", "dummy",
                    "--repo-root", str(fake_repo),
                    "--input-root", str(tdp / "dataset"),
                    "--output-root", str(parent_out),
                ],
            ):
                mod.main()

    # 4. Output root is parent of input_root raises ValueError
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        fake_input = tdp / "data" / "corpus"
        fake_input.mkdir(parents=True)
        parent_out = tdp / "data"
        with pytest.raises(ValueError, match="Safety guard: output_root contains input_root"):
            with patch.object(
                sys,
                "argv",
                [
                    "runner",
                    "--expected-commit", "dummy",
                    "--repo-root", str(tdp / "repo"),
                    "--input-root", str(fake_input),
                    "--output-root", str(parent_out),
                ],
            ):
                mod.main()

    # 5. Output root equals repo_root raises ValueError
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        fake_repo = tdp / "repo"
        fake_repo.mkdir(parents=True)
        with pytest.raises(ValueError, match="Safety guard: output_root contains repo_root"):
            with patch.object(
                sys,
                "argv",
                [
                    "runner",
                    "--expected-commit", "dummy",
                    "--repo-root", str(fake_repo),
                    "--input-root", str(tdp / "input"),
                    "--output-root", str(fake_repo),
                ],
            ):
                mod.main()

    # 6. Deep corpus discovery (dataset nested multiple levels deep e.g. /kaggle/input/datasets/user/dataset-aic)
    from system_tai.data.corpus_discovery import resolve_dataset_root
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        deep_dataset = tdp / "datasets" / "user" / "dataset-aic"
        (deep_dataset / "map-keyframes").mkdir(parents=True)
        (deep_dataset / "clip-features").mkdir(parents=True)
        (deep_dataset / "keyframes").mkdir(parents=True)

        resolved = resolve_dataset_root(tdp, max_depth=4)
        assert resolved == deep_dataset.resolve()
