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
        "message": "TranslationError: Query 'query-unseen-6' not found in immutable sidecar",
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
    ]
    for flag_args, match_msg in bad_flags:
        args = parser.parse_args(["--profile", "kis-v2a-rc1-replay"] + flag_args)
        with pytest.raises(ValueError, match=match_msg):
            session_config_from_args(args)

