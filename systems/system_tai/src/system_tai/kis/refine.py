"""Opt-in raw-video exact-frame refinement for an existing Phase 3 KIS run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from system_tai.kis.benchmark import resolve_device
from system_tai.refinement.models import (
    CandidateFailurePolicy,
    MissingRawVideoPolicy,
    RefinementConfig,
)
from system_tai.refinement.runner import RefinementRunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--top-candidates-to-refine", type=int, default=20)
    parser.add_argument("--window-before-seconds", type=float, default=5.0)
    parser.add_argument("--window-after-seconds", type=float, default=5.0)
    parser.add_argument("--coarse-stride-frames", type=int, default=15)
    parser.add_argument("--coarse-top-n", type=int, default=3)
    parser.add_argument("--fine-radius-frames", type=int, default=30)
    parser.add_argument("--fine-stride-frames", type=int, default=1)
    parser.add_argument("--image-batch-size", type=int, default=32)
    parser.add_argument("--max-decoded-frames-per-candidate", type=int, default=500)
    parser.add_argument("--output-top-k", type=int, default=100)
    parser.add_argument(
        "--missing-raw-video-policy",
        choices=tuple(policy.value for policy in MissingRawVideoPolicy),
        default=MissingRawVideoPolicy.KEEP_ORIGINAL.value,
    )
    parser.add_argument(
        "--candidate-failure-policy",
        choices=tuple(policy.value for policy in CandidateFailurePolicy),
        default=CandidateFailurePolicy.KEEP_ORIGINAL.value,
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--allow-model-download", action="store_true")
    parser.add_argument("--clip-cache-dir", type=Path)
    parser.add_argument("--rrf-constant", type=float, default=60.0)
    failures = parser.add_mutually_exclusive_group()
    failures.add_argument("--fail-fast", action="store_true")
    failures.add_argument("--continue-on-query-error", action="store_true")
    parser.add_argument("--contact-sheet", action="store_true")
    return parser


def run(args: argparse.Namespace, *, runner: RefinementRunner | None = None) -> int:
    device = resolve_device(args.device)
    config = RefinementConfig(
        top_candidates_to_refine=args.top_candidates_to_refine,
        window_before_seconds=args.window_before_seconds,
        window_after_seconds=args.window_after_seconds,
        coarse_stride_frames=args.coarse_stride_frames,
        coarse_top_n=args.coarse_top_n,
        fine_radius_frames=args.fine_radius_frames,
        fine_stride_frames=args.fine_stride_frames,
        image_batch_size=args.image_batch_size,
        max_decoded_frames_per_candidate=args.max_decoded_frames_per_candidate,
        output_top_k=args.output_top_k,
        device=device,
        missing_raw_video_policy=MissingRawVideoPolicy(args.missing_raw_video_policy),
        candidate_failure_policy=CandidateFailurePolicy(args.candidate_failure_policy),
        allow_model_download=args.allow_model_download,
        clip_cache_dir=args.clip_cache_dir,
        rrf_constant=args.rrf_constant,
    )
    print(f"refinement device selected: {device}")
    print("GPU changes latency only; refinement ranking semantics are unchanged")
    outcome = (runner or RefinementRunner()).run(
        run_directory=args.run_directory,
        output_directory=args.output_directory,
        config=config,
        continue_on_query_error=args.continue_on_query_error,
        create_contact_sheet=args.contact_sheet,
    )
    print(
        json.dumps(
            {
                "status": "PASS" if outcome.exit_code == 0 else "FAILED",
                "successful_query_ids": outcome.successful_query_ids,
                "failed_query_ids": [item[0] for item in outcome.failed_queries],
                "validator_valid": outcome.validation.valid,
                "generated_files": [
                    str(path.relative_to(args.output_directory)).replace("\\", "/")
                    for path in outcome.output_files
                ],
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )
    return outcome.exit_code


def main(argv: list[str] | None = None) -> int:
    try:
        return run(build_parser().parse_args(argv))
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"KIS refinement failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
