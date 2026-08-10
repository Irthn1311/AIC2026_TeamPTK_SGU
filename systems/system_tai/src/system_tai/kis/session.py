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
from system_tai.refinement.models import (
    CandidateFailurePolicy,
    MissingRawVideoPolicy,
    RefinementConfig,
)
from system_tai.refinement.video import CoarseDecodeStrategy


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
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--allow-model-download", action="store_true")
    parser.add_argument("--clip-cache-dir", type=Path)
    parser.add_argument("--rrf-constant", type=float, default=60.0)
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--default-top-k-per-variant", type=int, default=100)
    parser.add_argument("--default-output-top-k", type=int, default=100)
    parser.add_argument("--default-refine-top-n", type=int, default=3)
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
    resolved_device = resolve_device(args.device)
    ref_config = RefinementConfig(
        device=resolved_device,
        top_candidates_to_refine=args.default_refine_top_n,
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
        device=args.device,
        allow_model_download=args.allow_model_download,
        clip_cache_dir=args.clip_cache_dir,
        rrf_constant=args.rrf_constant,
        chunk_size=args.chunk_size,
        default_top_k_per_variant=args.default_top_k_per_variant,
        default_output_top_k=args.default_output_top_k,
        default_refine_top_n=args.default_refine_top_n,
        max_requests=args.max_requests,
        continue_on_request_error=args.continue_on_request_error,
        fail_fast_protocol=args.fail_fast_protocol,
        session_id=args.session_id,
        refinement_config=ref_config,
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
            runtime = OperationalKISRuntime.bootstrap(config)
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
