"""CLI entry point for system_tai long-lived contest operational session."""

from __future__ import annotations

import argparse
import signal
import sys
from pathlib import Path
from typing import Any, TextIO

from system_tai.kis.benchmark import resolve_device
from system_tai.kis.session_engine import OperationalKISRuntime
from system_tai.kis.session_schema import (
    DuplicateRequestIdError,
    HealthRequest,
    InvalidRequestError,
    MalformedRequestError,
    QAQueryRequest,
    QueryRequest,
    SessionConfig,
    SessionProtocolError,
    ShutdownRequest,
    TRAKEQueryRequest,
    UnknownRequestTypeError,
    format_json_response,
    parse_session_request,
)
from system_tai.kis.video_first import KISVideoFirstConfig
from system_tai.refinement.models import (
    CandidateFailurePolicy,
    MissingRawVideoPolicy,
    Q3AnchorRefinementConfig,
    RefinementConfig,
    SelectedVideoTimelineScoutConfig,
    SelectedVideoVisualVerifierConfig,
    VisualVerifierExecutionMode,
    VisualVerifierFailurePolicy,
)
from system_tai.refinement.video import CoarseDecodeStrategy
from system_tai.retrieval.video_restricted import VideoConditionedKeyframeConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a long-lived contest operational KIS session using JSON lines "
            "over stdin/stdout."
        )
    )
    parser.add_argument("--input-root", type=Path, default=Path("/kaggle/input"))
    manifest_source = parser.add_mutually_exclusive_group()
    manifest_source.add_argument("--reuse-manifest", type=Path)
    manifest_source.add_argument("--manifest-cache", type=Path)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/kaggle/working/system_tai_operational_session"),
    )
    parser.add_argument(
        "--profile",
        choices=("legacy", "kis-v2a-rc1-replay"),
        default="legacy",
        help="Named operational profile. 'kis-v2a-rc1-replay' locks the verified RC1 replay baseline.",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=None)
    parser.add_argument("--allow-model-download", action="store_true")
    parser.add_argument("--clip-cache-dir", type=Path)
    parser.add_argument("--enable-dynamic-translation", action="store_true")
    parser.add_argument(
        "--translation-model-name",
        default="vinai/vinai-translate-vi2en-v2",
    )
    parser.add_argument("--translation-cache-dir", type=Path)
    parser.add_argument(
        "--translation-device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument("--translation-allow-model-download", action="store_true")
    parser.add_argument(
        "--translation-revision",
        default="ae7baa85da07dbe8e23ac26a9f5ef560c17e2138",
    )
    parser.add_argument("--translation-max-clip-tokens", type=int, default=75)
    parser.add_argument(
        "--enable-kis-semantic-video-first",
        action="store_true",
        help=(
            "Use VinAI semantic clauses, exact video-level RRF nomination, and "
            "restricted full-keyframe search for KIS requests."
        ),
    )
    parser.add_argument("--kis-selected-video-cap", type=int, default=None)
    parser.add_argument("--kis-video-nomination-depth", type=int, default=None)
    parser.add_argument(
        "--kis-restricted-frames-per-video-per-variant",
        type=int,
        default=None,
    )
    parser.add_argument("--kis-full-query-weight", type=float, default=None)
    parser.add_argument("--kis-primary-scene-weight", type=float, default=None)
    parser.add_argument("--kis-supporting-attribute-weight", type=float, default=None)
    parser.add_argument(
        "--enable-kis-multi-anchor-refinement",
        action="store_true",
        help=(
            "Opt in to semantic-unit keyframe anchors plus bounded Q3 raw-video "
            "refinement. Retrieval and output schema remain unchanged."
        ),
    )
    parser.add_argument("--kis-anchor-video-rank-cap", type=int, default=20)
    parser.add_argument("--kis-anchor-max-videos", type=int, default=5)
    parser.add_argument("--kis-anchors-per-video", type=int, default=6)
    parser.add_argument("--kis-anchor-min-gap-seconds", type=float, default=2.0)
    parser.add_argument("--kis-max-extra-raw-anchors", type=int, default=12)
    parser.add_argument(
        "--enable-kis-selected-video-timeline-scout",
        action="store_true",
        help=(
            "Uniformly scout complete raw timelines of system-nominated videos, "
            "then apply bounded exact-frame refinement to discovered regions."
        ),
    )
    parser.add_argument("--kis-timeline-max-videos", type=int, default=3)
    parser.add_argument("--kis-timeline-sample-stride-seconds", type=float, default=1.0)
    parser.add_argument("--kis-timeline-max-samples-per-video", type=int, default=300)
    parser.add_argument("--kis-timeline-max-regions-per-video", type=int, default=3)
    parser.add_argument("--kis-timeline-min-region-gap-seconds", type=float, default=5.0)
    parser.add_argument(
        "--enable-kis-visual-predicate-verifier",
        action="store_true",
        help=(
            "Use one locally loaded structured VLM to verify a bounded, "
            "coverage-preserving shortlist from the automatic timeline scout."
        ),
    )
    parser.add_argument(
        "--kis-visual-verifier-model",
        default="Qwen/Qwen2.5-VL-3B-Instruct",
    )
    parser.add_argument("--kis-visual-verifier-revision")
    parser.add_argument("--kis-visual-verifier-cache-dir", type=Path)
    parser.add_argument("--kis-visual-verifier-allow-model-download", action="store_true")
    parser.add_argument("--kis-visual-verifier-shortlist-per-video", type=int, default=32)
    parser.add_argument("--kis-visual-verifier-coverage-bins", type=int, default=12)
    parser.add_argument(
        "--kis-visual-verifier-temporal-evidence-window-seconds",
        type=float,
        default=6.0,
    )
    parser.add_argument("--kis-visual-verifier-neighbor-radius", type=int, default=1)
    parser.add_argument("--kis-visual-verifier-max-new-tokens", type=int, default=512)
    parser.add_argument(
        "--kis-visual-verifier-execution-mode",
        choices=tuple(mode.value for mode in VisualVerifierExecutionMode),
        default=VisualVerifierExecutionMode.AUTO.value,
        help=(
            "auto applies a bounded CPU-safe workload on CPU and preserves the "
            "requested workload on CUDA; full always preserves requested values."
        ),
    )
    parser.add_argument(
        "--kis-visual-verifier-failure-policy",
        choices=tuple(policy.value for policy in VisualVerifierFailurePolicy),
        default=VisualVerifierFailurePolicy.FALLBACK_CLIP.value,
    )
    parser.add_argument("--rrf-constant", type=float, default=None)
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--default-top-k-per-variant", type=int, default=None)
    parser.add_argument("--default-output-top-k", type=int, default=None)
    parser.add_argument("--default-refine-top-n", type=int, default=None)
    parser.add_argument("--max-requests", type=int)

    cont_group = parser.add_mutually_exclusive_group()
    cont_group.add_argument(
        "--continue-on-request-error",
        action="store_true",
        default=True,
    )
    cont_group.add_argument(
        "--no-continue-on-request-error",
        dest="continue_on_request_error",
        action="store_false",
    )

    parser.add_argument("--fail-fast-protocol", action="store_true")
    parser.add_argument("--session-id")

    # Refinement parameters
    parser.add_argument("--window-before-seconds", type=float, default=5.0)
    parser.add_argument("--window-after-seconds", type=float, default=5.0)
    parser.add_argument("--coarse-stride-frames", type=int, default=15)
    parser.add_argument("--coarse-top-n", type=int, default=3)
    parser.add_argument("--fine-radius-frames", type=int, default=30)
    parser.add_argument("--fine-stride-frames", type=int, default=1)
    parser.add_argument("--image-batch-size", type=int, default=32)
    parser.add_argument("--max-decoded-frames-per-candidate", type=int, default=500)
    parser.add_argument(
        "--missing-raw-video-policy",
        choices=tuple(p.value for p in MissingRawVideoPolicy),
        default=MissingRawVideoPolicy.KEEP_ORIGINAL.value,
    )
    parser.add_argument(
        "--candidate-failure-policy",
        choices=tuple(p.value for p in CandidateFailurePolicy),
        default=CandidateFailurePolicy.KEEP_ORIGINAL.value,
    )
    parser.add_argument(
        "--coarse-decode-strategy",
        choices=("sequential", "sparse-verified"),
        default="sequential",
    )
    return parser


def session_config_from_args(args: argparse.Namespace) -> SessionConfig:
    profile = getattr(args, "profile", "legacy")
    if profile == "kis-v2a-rc1-replay":
        # 1. Condition 3: Check conflicting CLI arguments strictly fail-fast
        if args.device is not None and args.device != "cpu":
            raise ValueError(
                f"--profile kis-v2a-rc1-replay strictly requires --device cpu, got '{args.device}'"
            )
        if getattr(args, "allow_model_download", False):
            raise ValueError("--profile kis-v2a-rc1-replay strictly requires allow_model_download=False")
        if getattr(args, "enable_kis_multi_anchor_refinement", False):
            raise ValueError("--profile kis-v2a-rc1-replay strictly requires local anchor refinement to be OFF")
        if getattr(args, "enable_kis_selected_video_timeline_scout", False):
            raise ValueError("--profile kis-v2a-rc1-replay strictly requires timeline scout to be OFF")
        if getattr(args, "enable_kis_visual_predicate_verifier", False):
            raise ValueError("--profile kis-v2a-rc1-replay strictly requires visual verifier to be OFF")
        if args.kis_selected_video_cap is not None and args.kis_selected_video_cap != 64:
            raise ValueError(f"--profile kis-v2a-rc1-replay strictly requires --kis-selected-video-cap 64, got {args.kis_selected_video_cap}")
        if args.kis_video_nomination_depth is not None and args.kis_video_nomination_depth != 100:
            raise ValueError(f"--profile kis-v2a-rc1-replay strictly requires --kis-video-nomination-depth 100, got {args.kis_video_nomination_depth}")
        if args.kis_restricted_frames_per_video_per_variant is not None and args.kis_restricted_frames_per_video_per_variant != 20:
            raise ValueError(f"--profile kis-v2a-rc1-replay strictly requires --kis-restricted-frames-per-video-per-variant 20, got {args.kis_restricted_frames_per_video_per_variant}")
        if args.kis_full_query_weight is not None and args.kis_full_query_weight != 1.0:
            raise ValueError(f"--profile kis-v2a-rc1-replay strictly requires --kis-full-query-weight 1.0, got {args.kis_full_query_weight}")
        if args.kis_primary_scene_weight is not None and args.kis_primary_scene_weight != 1.0:
            raise ValueError(f"--profile kis-v2a-rc1-replay strictly requires --kis-primary-scene-weight 1.0, got {args.kis_primary_scene_weight}")
        if args.kis_supporting_attribute_weight is not None and args.kis_supporting_attribute_weight != 0.35:
            raise ValueError(f"--profile kis-v2a-rc1-replay strictly requires --kis-supporting-attribute-weight 0.35, got {args.kis_supporting_attribute_weight}")
        if args.rrf_constant is not None and args.rrf_constant != 60.0:
            raise ValueError(f"--profile kis-v2a-rc1-replay strictly requires --rrf-constant 60.0, got {args.rrf_constant}")
        if args.chunk_size is not None and args.chunk_size != 256:
            raise ValueError(f"--profile kis-v2a-rc1-replay strictly requires --chunk-size 256, got {args.chunk_size}")
        if args.default_output_top_k is not None and args.default_output_top_k != 100:
            raise ValueError(f"--profile kis-v2a-rc1-replay strictly requires --default-output-top-k 100, got {args.default_output_top_k}")
        if args.default_top_k_per_variant is not None and args.default_top_k_per_variant != 100:
            raise ValueError(f"--profile kis-v2a-rc1-replay strictly requires --default-top-k-per-variant 100, got {args.default_top_k_per_variant}")
        if args.default_refine_top_n is not None and args.default_refine_top_n != 3:
            raise ValueError(f"--profile kis-v2a-rc1-replay strictly requires --default-refine-top-n 3, got {args.default_refine_top_n}")

        from system_tai.kis.profiles import (
            apply_kis_v2a_rc1_replay_profile,
            validate_kis_v2a_rc1_replay_config,
        )
        base_cfg = SessionConfig(
            input_root=args.input_root,
            reuse_manifest=args.reuse_manifest,
            manifest_cache=args.manifest_cache,
            output_root=args.output_root,
            device="cpu",
            clip_cache_dir=args.clip_cache_dir,
            session_id=args.session_id,
            max_requests=args.max_requests,
        )
        rc1_cfg = apply_kis_v2a_rc1_replay_profile(base_cfg)
        validate_kis_v2a_rc1_replay_config(rc1_cfg)
        return rc1_cfg

    # Legacy profile branch
    device_val = args.device or "auto"
    resolved_device = resolve_device(device_val)
    refine_top_n_val = args.default_refine_top_n if args.default_refine_top_n is not None else 3
    multi_anchor_enabled = getattr(args, "enable_kis_multi_anchor_refinement", False)
    if multi_anchor_enabled and not getattr(
        args, "enable_kis_semantic_video_first", False
    ):
        raise ValueError(
            "--enable-kis-multi-anchor-refinement requires "
            "--enable-kis-semantic-video-first"
        )
    if multi_anchor_enabled and refine_top_n_val <= 0:
        raise ValueError(
            "--enable-kis-multi-anchor-refinement requires "
            "--default-refine-top-n greater than zero"
        )
    timeline_scout_enabled = getattr(
        args, "enable_kis_selected_video_timeline_scout", False
    )
    if timeline_scout_enabled and not getattr(
        args, "enable_kis_semantic_video_first", False
    ):
        raise ValueError(
            "--enable-kis-selected-video-timeline-scout requires "
            "--enable-kis-semantic-video-first"
        )
    if timeline_scout_enabled and refine_top_n_val <= 0:
        raise ValueError(
            "--enable-kis-selected-video-timeline-scout requires "
            "--default-refine-top-n greater than zero"
        )
    visual_verifier_enabled = getattr(
        args, "enable_kis_visual_predicate_verifier", False
    )
    if visual_verifier_enabled and not timeline_scout_enabled:
        raise ValueError(
            "--enable-kis-visual-predicate-verifier requires "
            "--enable-kis-selected-video-timeline-scout"
        )
    # RefinementConfig describes an executable refinement pass and therefore
    # requires at least one candidate.  The session-level value zero is still
    # the supported switch that disables refinement for every default request;
    # keep a valid dormant template without changing that request semantics.
    refinement_template_top_n = max(1, refine_top_n_val)
    ref_config = RefinementConfig(
        device=resolved_device,
        top_candidates_to_refine=refinement_template_top_n,
        window_before_seconds=args.window_before_seconds,
        window_after_seconds=args.window_after_seconds,
        coarse_stride_frames=args.coarse_stride_frames,
        coarse_top_n=args.coarse_top_n,
        fine_radius_frames=args.fine_radius_frames,
        fine_stride_frames=args.fine_stride_frames,
        image_batch_size=args.image_batch_size,
        max_decoded_frames_per_candidate=args.max_decoded_frames_per_candidate,
        missing_raw_video_policy=MissingRawVideoPolicy(args.missing_raw_video_policy),
        candidate_failure_policy=CandidateFailurePolicy(args.candidate_failure_policy),
        allow_model_download=args.allow_model_download,
        clip_cache_dir=args.clip_cache_dir,
        coarse_decode_strategy=CoarseDecodeStrategy(args.coarse_decode_strategy),
    )
    return SessionConfig(
        input_root=args.input_root,
        reuse_manifest=args.reuse_manifest,
        manifest_cache=args.manifest_cache,
        output_root=args.output_root,
        device=device_val,
        allow_model_download=args.allow_model_download,
        clip_cache_dir=args.clip_cache_dir,
        rrf_constant=args.rrf_constant if args.rrf_constant is not None else 60.0,
        chunk_size=args.chunk_size if args.chunk_size is not None else 4096,
        default_top_k_per_variant=(
            args.default_top_k_per_variant if args.default_top_k_per_variant is not None else 100
        ),
        default_output_top_k=(
            args.default_output_top_k if args.default_output_top_k is not None else 100
        ),
        default_refine_top_n=refine_top_n_val,
        max_requests=args.max_requests,
        continue_on_request_error=args.continue_on_request_error,
        fail_fast_protocol=args.fail_fast_protocol,
        session_id=args.session_id,
        refinement_config=ref_config,
        enable_dynamic_translation=(
            getattr(args, "enable_dynamic_translation", False)
            or getattr(args, "enable_kis_semantic_video_first", False)
        ),
        translation_model_name=getattr(
            args,
            "translation_model_name",
            "vinai/vinai-translate-vi2en-v2",
        ),
        translation_cache_dir=getattr(args, "translation_cache_dir", None),
        translation_device=getattr(args, "translation_device", "auto"),
        translation_allow_model_download=getattr(
            args,
            "translation_allow_model_download",
            False,
        ),
        translation_revision=getattr(
            args,
            "translation_revision",
            "ae7baa85da07dbe8e23ac26a9f5ef560c17e2138",
        ),
        translation_max_clip_tokens=getattr(
            args,
            "translation_max_clip_tokens",
            75,
        ),
        kis_video_first_config=KISVideoFirstConfig(
            enabled=getattr(args, "enable_kis_semantic_video_first", False),
            selected_video_cap=(
                args.kis_selected_video_cap if args.kis_selected_video_cap is not None else 32
            ),
            video_nomination_depth=(
                args.kis_video_nomination_depth if args.kis_video_nomination_depth is not None else 100
            ),
            restricted_frames_per_video_per_variant=(
                args.kis_restricted_frames_per_video_per_variant
                if args.kis_restricted_frames_per_video_per_variant is not None
                else 10
            ),
            full_query_weight=(
                args.kis_full_query_weight if args.kis_full_query_weight is not None else 1.0
            ),
            primary_scene_weight=(
                args.kis_primary_scene_weight if args.kis_primary_scene_weight is not None else 1.0
            ),
            supporting_attribute_weight=(
                args.kis_supporting_attribute_weight
                if args.kis_supporting_attribute_weight is not None
                else 0.35
            ),
        ),
        video_conditioned_keyframe_config=VideoConditionedKeyframeConfig(
            enabled=multi_anchor_enabled,
            selected_video_global_rank_cap=getattr(
                args, "kis_anchor_video_rank_cap", 20
            ),
            max_selected_videos=getattr(args, "kis_anchor_max_videos", 5),
            max_anchors_per_video=getattr(args, "kis_anchors_per_video", 6),
            minimum_anchor_gap_seconds=getattr(
                args, "kis_anchor_min_gap_seconds", 2.0
            ),
            semantic_variant_coverage=multi_anchor_enabled,
        ),
        q3_anchor_refinement_config=Q3AnchorRefinementConfig(
            enabled=multi_anchor_enabled,
            max_extra_q3_anchors=getattr(
                args, "kis_max_extra_raw_anchors", 12
            ),
        ),
        selected_video_timeline_scout_config=SelectedVideoTimelineScoutConfig(
            enabled=timeline_scout_enabled,
            max_videos=getattr(args, "kis_timeline_max_videos", 3),
            sample_stride_seconds=getattr(
                args, "kis_timeline_sample_stride_seconds", 1.0
            ),
            max_samples_per_video=getattr(
                args, "kis_timeline_max_samples_per_video", 300
            ),
            max_regions_per_video=getattr(
                args, "kis_timeline_max_regions_per_video", 3
            ),
            minimum_region_gap_seconds=getattr(
                args, "kis_timeline_min_region_gap_seconds", 5.0
            ),
        ),
        selected_video_visual_verifier_config=SelectedVideoVisualVerifierConfig(
            enabled=visual_verifier_enabled,
            model_name=getattr(
                args,
                "kis_visual_verifier_model",
                "Qwen/Qwen2.5-VL-3B-Instruct",
            ),
            model_revision=getattr(args, "kis_visual_verifier_revision", None),
            shortlist_per_video=getattr(
                args, "kis_visual_verifier_shortlist_per_video", 32
            ),
            coverage_bins=getattr(args, "kis_visual_verifier_coverage_bins", 12),
            temporal_evidence_window_seconds=getattr(
                args,
                "kis_visual_verifier_temporal_evidence_window_seconds",
                6.0,
            ),
            neighbor_sample_radius=getattr(
                args, "kis_visual_verifier_neighbor_radius", 1
            ),
            max_new_tokens=getattr(args, "kis_visual_verifier_max_new_tokens", 512),
            device=resolved_device,
            execution_mode=VisualVerifierExecutionMode(
                getattr(
                    args,
                    "kis_visual_verifier_execution_mode",
                    VisualVerifierExecutionMode.AUTO.value,
                )
            ),
            allow_model_download=getattr(
                args, "kis_visual_verifier_allow_model_download", False
            ),
            cache_dir=getattr(args, "kis_visual_verifier_cache_dir", None),
            failure_policy=VisualVerifierFailurePolicy(
                getattr(
                    args,
                    "kis_visual_verifier_failure_policy",
                    VisualVerifierFailurePolicy.FALLBACK_CLIP.value,
                )
            ),
        ),
    )


def run_session(
    args: argparse.Namespace,
    *,
    runtime: OperationalKISRuntime | None = None,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    in_stream = stdin or sys.stdin
    out_stream = stdout or sys.stdout
    err_stream = stderr or sys.stderr

    config = session_config_from_args(args)
    if runtime is None:
        print(f"bootstrapping session at output_root: {config.output_root}", file=err_stream)
        try:
            translation_provider = None
            if getattr(args, "profile", "legacy") == "kis-v2a-rc1-replay":
                from system_tai.kis.profiles import get_kis_v2a_rc1_replay_translation_provider
                translation_provider = get_kis_v2a_rc1_replay_translation_provider()
            runtime = OperationalKISRuntime.bootstrap(config, translation_provider=translation_provider)
            print(f"session bootstrapped: session_id={runtime.session_id}", file=err_stream)
        except Exception as exc:
            print(f"session bootstrap failed: {type(exc).__name__}: {exc}", file=err_stream)
            return 2

    active_runtime = runtime

    def _write_resp(payload: dict[str, Any]) -> None:
        out_stream.write(format_json_response(payload) + "\n")
        out_stream.flush()

    line_number = 0
    shutdown_received = False
    exit_code = 0

    def signal_handler(_signum: int, _frame: Any) -> None:
        nonlocal shutdown_received
        print("interrupt received, closing session...", file=err_stream)
        shutdown_received = True
        active_runtime.close(shutdown_reason="sigint")

    try:
        signal.signal(signal.SIGINT, signal_handler)
    except (ValueError, AttributeError):
        pass

    try:
        while not shutdown_received:
            line = in_stream.readline()
            if not line:
                print("stdin EOF reached, shutting down session", file=err_stream)
                active_runtime.close(shutdown_reason="eof")
                break
            line_number += 1
            if not line.strip():
                continue

            try:
                request = parse_session_request(
                    line,
                    line_number=line_number,
                    default_top_k_per_variant=config.default_top_k_per_variant,
                    default_output_top_k=config.default_output_top_k,
                    default_refine_top_n=config.default_refine_top_n,
                )
            except MalformedRequestError as exc:
                err_resp = active_runtime.handle_error(
                    request_id=None,
                    error_code="MALFORMED_JSON",
                    error_type=type(exc).__name__,
                    message=str(exc),
                    session_continues=(
                        config.continue_on_request_error
                        and not config.fail_fast_protocol
                    ),
                )
                _write_resp(err_resp)
                if config.fail_fast_protocol:
                    exit_code = 1
                    break
                continue
            except (InvalidRequestError, UnknownRequestTypeError, SessionProtocolError) as exc:
                err_code = (
                    "UNKNOWN_REQUEST_TYPE"
                    if isinstance(exc, UnknownRequestTypeError)
                    else "INVALID_REQUEST"
                )
                err_resp = active_runtime.handle_error(
                    request_id=None,
                    error_code=err_code,
                    error_type=type(exc).__name__,
                    message=str(exc),
                    session_continues=(
                        config.continue_on_request_error
                        and not config.fail_fast_protocol
                    ),
                )
                _write_resp(err_resp)
                if config.fail_fast_protocol:
                    exit_code = 1
                    break
                continue

            if isinstance(request, HealthRequest):
                try:
                    resp = active_runtime.handle_health(request)
                    _write_resp(resp)
                except DuplicateRequestIdError as exc:
                    resp = active_runtime.handle_error(
                        request_id=request.request_id,
                        error_code="DUPLICATE_REQUEST_ID",
                        error_type=type(exc).__name__,
                        message=str(exc),
                        session_continues=True,
                    )
                    _write_resp(resp)
            elif isinstance(request, ShutdownRequest):
                try:
                    resp = active_runtime.handle_shutdown(request, shutdown_reason="requested")
                    _write_resp(resp)
                except DuplicateRequestIdError as exc:
                    resp = active_runtime.handle_error(
                        request_id=request.request_id,
                        error_code="DUPLICATE_REQUEST_ID",
                        error_type=type(exc).__name__,
                        message=str(exc),
                        session_continues=True,
                    )
                    _write_resp(resp)
                shutdown_received = True
                break
            elif isinstance(request, QueryRequest):
                try:
                    resp = active_runtime.handle_query(request)
                    _write_resp(resp)
                except DuplicateRequestIdError as exc:
                    resp = active_runtime.handle_error(
                        request_id=request.request_id,
                        error_code="DUPLICATE_REQUEST_ID",
                        error_type=type(exc).__name__,
                        message=str(exc),
                        session_continues=True,
                    )
                    _write_resp(resp)
                except Exception as exc:
                    print(f"query execution error on {request.request_id}: {exc}", file=err_stream)
                    resp = active_runtime.handle_error(
                        request_id=request.request_id,
                        error_code="QUERY_EXECUTION_FAILED",
                        error_type=type(exc).__name__,
                        message=f"{type(exc).__name__}: {exc}",
                        session_continues=(
                            config.continue_on_request_error
                            and not config.fail_fast_protocol
                        ),
                    )
                    _write_resp(resp)
                    if not config.continue_on_request_error or config.fail_fast_protocol:
                        exit_code = 1
                        break
            elif isinstance(request, QAQueryRequest):
                try:
                    resp = active_runtime.handle_qa_query(request)
                    _write_resp(resp)
                except DuplicateRequestIdError as exc:
                    resp = active_runtime.handle_error(
                        request_id=request.request_id,
                        error_code="DUPLICATE_REQUEST_ID",
                        error_type=type(exc).__name__,
                        message=str(exc),
                        session_continues=True,
                    )
                    _write_resp(resp)
                except Exception as exc:
                    print(
                        f"qa query execution error on {request.request_id}: {exc}",
                        file=err_stream,
                    )
                    resp = active_runtime.handle_error(
                        request_id=request.request_id,
                        error_code="QUERY_EXECUTION_FAILED",
                        error_type=type(exc).__name__,
                        message=f"{type(exc).__name__}: {exc}",
                        session_continues=(
                            config.continue_on_request_error
                            and not config.fail_fast_protocol
                        ),
                    )
                    _write_resp(resp)
                    if not config.continue_on_request_error or config.fail_fast_protocol:
                        exit_code = 1
                        break
            elif isinstance(request, TRAKEQueryRequest):
                try:
                    resp = active_runtime.handle_trake_query(request)
                    _write_resp(resp)
                except DuplicateRequestIdError as exc:
                    resp = active_runtime.handle_error(
                        request_id=request.request_id,
                        error_code="DUPLICATE_REQUEST_ID",
                        error_type=type(exc).__name__,
                        message=str(exc),
                        session_continues=True,
                    )
                    _write_resp(resp)
                except Exception as exc:
                    print(
                        f"trake query execution error on {request.request_id}: {exc}",
                        file=err_stream,
                    )
                    resp = active_runtime.handle_error(
                        request_id=request.request_id,
                        error_code="QUERY_EXECUTION_FAILED",
                        error_type=type(exc).__name__,
                        message=f"{type(exc).__name__}: {exc}",
                        session_continues=(
                            config.continue_on_request_error
                            and not config.fail_fast_protocol
                        ),
                    )
                    _write_resp(resp)
                    if not config.continue_on_request_error or config.fail_fast_protocol:
                        exit_code = 1
                        break

            if (
                config.max_requests is not None
                and active_runtime._request_count >= config.max_requests
            ):
                print(
                    f"max_requests limit ({config.max_requests}) reached, closing session",
                    file=err_stream,
                )
                active_runtime.close(shutdown_reason="max_requests_reached")
                break

    except Exception as exc:
        print(f"session loop unexpected failure: {exc}", file=err_stream)
        active_runtime.close(shutdown_reason=f"error: {exc}")
        return 2

    return exit_code


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run_session(args)


if __name__ == "__main__":
    raise SystemExit(main())
