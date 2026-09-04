"""Official Named Operational Profiles for system_tai KIS retrieval.

This module is the SINGLE SOURCE OF TRUTH for operational profiles, particularly
the officially promoted benchmark replay profile: 'kis-v2a-rc1-replay'.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import importlib.metadata
from pathlib import Path
from typing import Any

from system_tai.kis.session_schema import SessionConfig
from system_tai.kis.video_first import KISVideoFirstConfig
from system_tai.refinement.models import (
    CandidateFailurePolicy,
    CoarseDecodeStrategy,
    MissingRawVideoPolicy,
    Q3AnchorRefinementConfig,
    RefinementConfig,
    SelectedVideoTimelineScoutConfig,
    SelectedVideoVisualVerifierConfig,
)
from system_tai.retrieval.video_restricted import VideoConditionedKeyframeConfig
from system_tai.translation.sidecar_provider import (
    ImmutableSidecarTranslationProvider,
    canonical_sidecar_sha256,
)

# Profile Identifiers
KIS_V2A_RC1_REPLAY_PROFILE_NAME = "kis-v2a-rc1-replay"
SUPPORTED_PROFILES = ("legacy", KIS_V2A_RC1_REPLAY_PROFILE_NAME)

# Provenance Anchors
CANONICAL_RC1_TNEW_SHA256 = "545bd4a37c57af53713a1d9f382241ef729c287a1817a5671fdc923115b0be2a"
CANONICAL_PORTABLE_CORPUS_FINGERPRINT = "b0c5ea97a9d5e10dbb7e77dba18d153191218935e2a3275ef888e0a8a83ed6e4"
CANONICAL_ABSOLUTE_CORPUS_FINGERPRINT = "398bb60c6ea1c8ebbd787c801836ef96a8398795b61fd6808e996f4ef19c0fa2"
EXPECTED_FULL_CORPUS_FINGERPRINT = CANONICAL_ABSOLUTE_CORPUS_FINGERPRINT
ALLOWED_CORPUS_FINGERPRINTS = (
    CANONICAL_PORTABLE_CORPUS_FINGERPRINT,
    CANONICAL_ABSOLUTE_CORPUS_FINGERPRINT,
)
EXPECTED_OPENAI_CLIP_COMMIT = "d05afc436d78f1c48dc0dbf8e5980a9d471f35f6"
EXPECTED_CLIP_CHECKPOINT_SHA256 = "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af"

# Official Resource Location
OFFICIAL_RC1_REPLAY_SIDECAR = (
    Path(__file__).resolve().parent / "resources" / "translation_p1_focus_v2_new.json"
)


def compute_file_sha256(path: Path | str) -> str:
    """Compute SHA-256 hash of a file on disk."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_installed_clip_commit() -> str:
    """Read commit_id directly from direct_url.json of installed OpenAI CLIP package."""
    try:
        dist = importlib.metadata.distribution("clip")
        raw = dist.read_text("direct_url.json")
        if not raw:
            raise FileNotFoundError("Missing direct_url.json in clip dist-info metadata")
        data = json.loads(raw)
        commit = data.get("vcs_info", {}).get("commit_id")
        if not commit:
            raise ValueError("No vcs_info.commit_id found in clip direct_url.json")
        return str(commit).strip()
    except Exception as exc:
        raise AssertionError(
            f"Fail-closed: Unable to verify installed OpenAI CLIP git commit via direct_url.json: {exc}"
        ) from exc


def get_kis_v2a_rc1_video_first_config() -> KISVideoFirstConfig:
    """Return the exact, immutable KISVideoFirstConfig frozen for KIS_V2A_RC1 Replay."""
    return KISVideoFirstConfig(
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
        adaptive_budget_base=32,
        adaptive_budget_medium=48,
        adaptive_budget_high=64,
        coverage_threshold=0.75,
        enable_temporal_diverse_local_candidates=True,
        temporal_diversity_gap_seconds=5.0,
        enable_vi_localization_variant=True,
        vi_localization_weight=0.5,
        internal_rrf_candidate_depth=1000,
        enable_top_video_local_anchor=False,
        enable_paraphrase_ensemble=False,
        paraphrase_ensemble_mode="EQUAL_BUDGET",
        collect_fusion_trace=False,
    )


def get_kis_v2a_rc1_replay_translation_provider(
    sidecar_path: Path | None = None,
) -> ImmutableSidecarTranslationProvider:
    """Return an authenticated ImmutableSidecarTranslationProvider for RC1 replay."""
    resolved = Path(sidecar_path) if sidecar_path is not None else OFFICIAL_RC1_REPLAY_SIDECAR
    if not resolved.is_file():
        raise FileNotFoundError(
            f"Fail-closed: Official RC1 replay sidecar not found at {resolved}. "
            "Ensure package resources are installed properly."
        )
    return ImmutableSidecarTranslationProvider(
        sidecar_path=resolved,
        expected_content_sha256=CANONICAL_RC1_TNEW_SHA256,
    )


def get_kis_v2a_rc1_refinement_config() -> RefinementConfig:
    """Historical baseline RefinementConfig for RC1."""
    return RefinementConfig(
        top_candidates_to_refine=20,
        window_before_seconds=5.0,
        window_after_seconds=5.0,
        coarse_stride_frames=15,
        coarse_top_n=3,
        fine_radius_frames=30,
        fine_stride_frames=1,
        image_batch_size=32,
        max_decoded_frames_per_candidate=500,
        output_top_k=100,
        device="cpu",
        missing_raw_video_policy=MissingRawVideoPolicy.KEEP_ORIGINAL,
        candidate_failure_policy=CandidateFailurePolicy.KEEP_ORIGINAL,
        allow_model_download=False,
        clip_cache_dir=None,
        rrf_constant=60.0,
        coarse_decode_strategy=CoarseDecodeStrategy.SEQUENTIAL,
    )


def apply_kis_v2a_rc1_replay_profile(config: SessionConfig) -> SessionConfig:
    """Apply the frozen KIS_V2A_RC1 replay profile onto a base SessionConfig using dataclasses.replace."""
    vf_cfg = get_kis_v2a_rc1_video_first_config()
    clean_ref_cfg = get_kis_v2a_rc1_refinement_config()
    return dataclasses.replace(
        config,
        profile_name=KIS_V2A_RC1_REPLAY_PROFILE_NAME,
        device="cpu",
        allow_model_download=False,
        enable_dynamic_translation=True,
        translation_provider_mode="immutable_sidecar",
        translation_allow_model_download=False,
        rrf_constant=60.0,
        chunk_size=4096,
        default_top_k_per_variant=100,
        default_output_top_k=100,
        default_refine_top_n=3,
        continue_on_request_error=False,
        fail_fast_protocol=True,
        kis_video_first_config=vf_cfg,
        video_conditioned_keyframe_config=VideoConditionedKeyframeConfig(enabled=False),
        q3_anchor_refinement_config=Q3AnchorRefinementConfig(enabled=False),
        selected_video_timeline_scout_config=SelectedVideoTimelineScoutConfig(enabled=False),
        selected_video_visual_verifier_config=SelectedVideoVisualVerifierConfig(enabled=False),
        refinement_config=clean_ref_cfg,
    )


def validate_kis_v2a_rc1_replay_config(config: SessionConfig) -> None:
    """Single Source of Truth validator for KIS_V2A_RC1 Replay configuration."""
    if config.profile_name != KIS_V2A_RC1_REPLAY_PROFILE_NAME:
        raise AssertionError(
            f"Fail-closed: Expected profile_name '{KIS_V2A_RC1_REPLAY_PROFILE_NAME}', got '{config.profile_name}'"
        )
    if config.device != "cpu":
        raise ValueError(f"Profile '{KIS_V2A_RC1_REPLAY_PROFILE_NAME}' strictly requires device='cpu', got '{config.device}'")
    if config.allow_model_download is not False:
        raise ValueError(f"Profile '{KIS_V2A_RC1_REPLAY_PROFILE_NAME}' strictly requires allow_model_download=False")
    if config.enable_dynamic_translation is not True:
        raise ValueError(f"Profile '{KIS_V2A_RC1_REPLAY_PROFILE_NAME}' strictly requires enable_dynamic_translation=True")
    if config.translation_provider_mode != "immutable_sidecar":
        raise ValueError(
            f"Profile '{KIS_V2A_RC1_REPLAY_PROFILE_NAME}' strictly requires translation_provider_mode='immutable_sidecar', got '{config.translation_provider_mode}'"
        )
    if config.translation_allow_model_download is not False:
        raise ValueError(f"Profile '{KIS_V2A_RC1_REPLAY_PROFILE_NAME}' strictly requires translation_allow_model_download=False")
    if config.rrf_constant != 60.0:
        raise ValueError(f"Profile '{KIS_V2A_RC1_REPLAY_PROFILE_NAME}' strictly requires rrf_constant=60.0, got {config.rrf_constant}")
    if config.chunk_size != 4096:
        raise ValueError(f"Profile '{KIS_V2A_RC1_REPLAY_PROFILE_NAME}' strictly requires chunk_size=4096, got {config.chunk_size}")
    if config.default_top_k_per_variant != 100:
        raise ValueError(f"Profile '{KIS_V2A_RC1_REPLAY_PROFILE_NAME}' strictly requires default_top_k_per_variant=100, got {config.default_top_k_per_variant}")
    if config.default_output_top_k != 100:
        raise ValueError(f"Profile '{KIS_V2A_RC1_REPLAY_PROFILE_NAME}' strictly requires default_output_top_k=100, got {config.default_output_top_k}")
    if config.default_refine_top_n != 3:
        raise ValueError(f"Profile '{KIS_V2A_RC1_REPLAY_PROFILE_NAME}' strictly requires default_refine_top_n=3, got {config.default_refine_top_n}")
    if config.continue_on_request_error is not False:
        raise ValueError(f"Profile '{KIS_V2A_RC1_REPLAY_PROFILE_NAME}' strictly requires continue_on_request_error=False")
    if config.fail_fast_protocol is not True:
        raise ValueError(f"Profile '{KIS_V2A_RC1_REPLAY_PROFILE_NAME}' strictly requires fail_fast_protocol=True")
    if config.video_conditioned_keyframe_config != VideoConditionedKeyframeConfig(enabled=False):
        raise ValueError(
            f"Profile '{KIS_V2A_RC1_REPLAY_PROFILE_NAME}' strictly requires default VideoConditionedKeyframeConfig(enabled=False), got {config.video_conditioned_keyframe_config}"
        )
    if config.q3_anchor_refinement_config != Q3AnchorRefinementConfig(enabled=False):
        raise ValueError(
            f"Profile '{KIS_V2A_RC1_REPLAY_PROFILE_NAME}' strictly requires default Q3AnchorRefinementConfig(enabled=False), got {config.q3_anchor_refinement_config}"
        )
    if config.selected_video_timeline_scout_config != SelectedVideoTimelineScoutConfig(enabled=False):
        raise ValueError(
            f"Profile '{KIS_V2A_RC1_REPLAY_PROFILE_NAME}' strictly requires default SelectedVideoTimelineScoutConfig(enabled=False), got {config.selected_video_timeline_scout_config}"
        )
    if config.selected_video_visual_verifier_config != SelectedVideoVisualVerifierConfig(enabled=False):
        raise ValueError(
            f"Profile '{KIS_V2A_RC1_REPLAY_PROFILE_NAME}' strictly requires default SelectedVideoVisualVerifierConfig(enabled=False), got {config.selected_video_visual_verifier_config}"
        )

    expected_vf = get_kis_v2a_rc1_video_first_config()
    if config.kis_video_first_config != expected_vf:
        raise ValueError(
            f"Profile '{KIS_V2A_RC1_REPLAY_PROFILE_NAME}' strictly requires exact KISVideoFirstConfig match!\n"
            f"Expected: {expected_vf}\n"
            f"Got:      {config.kis_video_first_config}"
        )

    expected_ref = get_kis_v2a_rc1_refinement_config()
    if config.refinement_config != expected_ref:
        raise ValueError(
            f"Profile '{KIS_V2A_RC1_REPLAY_PROFILE_NAME}' strictly requires exact RefinementConfig match!\n"
            f"Expected: {expected_ref}\n"
            f"Got:      {config.refinement_config}"
        )


def validate_kis_v2a_rc1_replay_request(request: Any) -> None:
    """Fail-closed validation to prevent request JSONL from overriding frozen RC1 replay contract."""
    output_top_k = getattr(request, "output_top_k", None)
    if output_top_k is not None and output_top_k != 100:
        raise ValueError(
            f"Profile '{KIS_V2A_RC1_REPLAY_PROFILE_NAME}' strictly requires request output_top_k=100 (or default), got {output_top_k}"
        )
    top_k_per_variant = getattr(request, "top_k_per_variant", None)
    if top_k_per_variant is not None and top_k_per_variant != 100:
        raise ValueError(
            f"Profile '{KIS_V2A_RC1_REPLAY_PROFILE_NAME}' strictly requires request top_k_per_variant=100 (or default), got {top_k_per_variant}"
        )
    refine_top_n = getattr(request, "refine_top_n", None)
    if refine_top_n is not None and refine_top_n != 3:
        raise ValueError(
            f"Profile '{KIS_V2A_RC1_REPLAY_PROFILE_NAME}' strictly requires request refine_top_n=3 (or default), got {refine_top_n}"
        )
    include_vi = getattr(request, "include_vi_variant", True)
    if include_vi is not True:
        raise ValueError(
            f"Profile '{KIS_V2A_RC1_REPLAY_PROFILE_NAME}' strictly requires request include_vi_variant=True"
        )
    if getattr(request, "query_en", None) is not None or getattr(request, "query_en_expansion", None) is not None:
        raise ValueError(
            f"Profile '{KIS_V2A_RC1_REPLAY_PROFILE_NAME}' accepts Vietnamese input only; manual English variants are prohibited"
        )


def validate_kis_v2a_rc1_replay_model_pre_bootstrap(
    checkpoint_path: Path | None = None,
    clip_cache_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Validate OpenAI CLIP source commit and ViT-B-32 checkpoint SHA256 BEFORE clip.load()."""
    # 1. Inspect direct_url.json
    observed_commit = get_installed_clip_commit()
    if observed_commit.lower() != EXPECTED_OPENAI_CLIP_COMMIT.lower():
        raise AssertionError(
            f"Fail-closed: Installed OpenAI CLIP commit mismatch! Expected {EXPECTED_OPENAI_CLIP_COMMIT}, got {observed_commit}"
        )

    # 2. Inspect checkpoint file
    resolved_ckpt = None
    if clip_cache_dir is not None:
        resolved_ckpt = Path(clip_cache_dir) / "ViT-B-32.pt"
    elif checkpoint_path is not None:
        p = Path(checkpoint_path)
        resolved_ckpt = p / "ViT-B-32.pt" if p.is_dir() else p
    else:
        default_ckpt = Path.home() / ".cache" / "clip" / "ViT-B-32.pt"
        if default_ckpt.is_file():
            resolved_ckpt = default_ckpt

    if resolved_ckpt is None or not resolved_ckpt.is_file():
        raise FileNotFoundError(
            f"Fail-closed: CLIP ViT-B-32 checkpoint not found at {resolved_ckpt}. "
            "Pre-provision model before bootstrapping replay profile!"
        )

    actual_ckpt_sha = compute_file_sha256(resolved_ckpt)
    if actual_ckpt_sha.lower() != EXPECTED_CLIP_CHECKPOINT_SHA256.lower():
        raise AssertionError(
            f"Fail-closed: CLIP ViT-B-32.pt SHA256 mismatch! Expected {EXPECTED_CLIP_CHECKPOINT_SHA256}, got {actual_ckpt_sha}"
        )

    return {
        "clip_source_commit": observed_commit,
        "checkpoint_path": str(resolved_ckpt),
        "checkpoint_sha256": actual_ckpt_sha,
        "verified_bit_exact": True,
    }


def validate_kis_v2a_rc1_replay_environment(
    manifest: Any,
    registry: Any,
    translation_provider: Any,
    shared_encoder: Any,
) -> None:
    """Validate runtime environment invariants for KIS_V2A_RC1 Replay."""
    # 1. Translation Provider Invariants
    if translation_provider is None:
        raise ValueError(f"Profile '{KIS_V2A_RC1_REPLAY_PROFILE_NAME}' requires an explicit translation_provider!")
    if not isinstance(translation_provider, ImmutableSidecarTranslationProvider):
        raise TypeError(
            f"Profile '{KIS_V2A_RC1_REPLAY_PROFILE_NAME}' requires ImmutableSidecarTranslationProvider, got {type(translation_provider).__name__}"
        )
    if not translation_provider.sidecar_path.is_file():
        raise FileNotFoundError(f"Sidecar file not found: {translation_provider.sidecar_path}")
    csha = canonical_sidecar_sha256(translation_provider.sidecar_path)
    if csha.lower() != CANONICAL_RC1_TNEW_SHA256.lower():
        raise AssertionError(
            f"Fail-closed: Sidecar canonical SHA256 mismatch! Expected {CANONICAL_RC1_TNEW_SHA256}, got {csha}"
        )

    # 2. Corpus Invariants
    if len(manifest.videos) != 873:
        raise AssertionError(f"Fail-closed: Expected 873 videos in corpus, found {len(manifest.videos)}")
    if registry.total_rows != 177321:
        raise AssertionError(f"Fail-closed: Expected 177,321 total rows, found {registry.total_rows}")
    if registry.embedding_dimension != 512:
        raise AssertionError(f"Fail-closed: Expected 512 dimensions, found {registry.embedding_dimension}")
    if manifest.fingerprint not in ALLOWED_CORPUS_FINGERPRINTS:
        raise AssertionError(
            f"Fail-closed: Corpus fingerprint mismatch! Expected one of {ALLOWED_CORPUS_FINGERPRINTS}, got {manifest.fingerprint}"
        )

    # 3. Encoder Invariants
    dev = shared_encoder.identifiers.get("device", "unknown")
    if dev != "cpu":
        raise AssertionError(f"Fail-closed: Shared encoder device must be 'cpu', got '{dev}'")
