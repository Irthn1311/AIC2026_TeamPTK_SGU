#!/usr/bin/env python3
"""Prepare the bounded MB1 v0.2 model-free candidate and AI-QC bundle."""

from __future__ import annotations

import argparse
import logging
import subprocess
from pathlib import Path

from triage_eg.experiments.mb1_v02 import (
    MB1V02Config,
    create_mb1_v02_bundle,
    preflight_mb1_v02,
    prepare_mb1_v02_candidates,
)


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() or None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--prior-candidate-manifest", type=Path, required=True)
    parser.add_argument("--prior-candidate-qc", type=Path, required=True)
    parser.add_argument("--rt2-benchmark", type=Path)
    parser.add_argument("--prior-selection", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--zip-path", type=Path)
    parser.add_argument("--jpeg-quality", type=int, default=88)
    parser.add_argument("--build-git-commit")
    args = parser.parse_args()
    config = MB1V02Config(
        dataset_root=args.dataset_root,
        prior_candidate_manifest_path=args.prior_candidate_manifest,
        prior_candidate_qc_path=args.prior_candidate_qc,
        rt2_benchmark_path=args.rt2_benchmark,
        prior_selection_path=args.prior_selection,
        output_root=args.output_root,
        jpeg_quality=args.jpeg_quality,
        build_git_commit=args.build_git_commit or _git_commit(),
    )
    try:
        ready = preflight_mb1_v02(config)
        print("source videos:", ready["source_video_count"])
        print("maximum possible candidates:", ready["maximum_possible_with_pool_and_cap"])
        result = prepare_mb1_v02_candidates(config)
        zip_path = args.zip_path or (
            args.output_root.parent / "triage_eg_mb1_v02_candidates.zip"
        )
        create_mb1_v02_bundle(args.output_root, zip_path)
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
        logging.error("%s", error)
        return 2
    print("final candidates:", result["selection"]["final_candidate_count"])
    print("DOWNLOAD ZIP:", zip_path)
    print("MB1_V02_REAL_STATUS = COMPLETE")
    print("MB1_V02_AI_QC_STATUS = WAITING_FOR_AI")
    print("M3_IMPLEMENTATION_STATUS = NOT_STARTED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
