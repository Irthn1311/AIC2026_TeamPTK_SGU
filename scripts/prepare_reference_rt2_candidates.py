#!/usr/bin/env python3
"""Prepare deterministic RT2 chronological candidate contact sheets."""

from __future__ import annotations

import argparse
import logging
import subprocess
from pathlib import Path

from triage_eg.experiments.reference_rt2 import (
    create_candidate_bundle,
    load_rt2_settings,
    prepare_benchmark_candidates,
)


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() or None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--candidate-count", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--frames-per-sheet", type=int)
    parser.add_argument(
        "--experiment-config",
        type=Path,
        default=Path("configs/experiments/reference_rt2.yaml"),
    )
    parser.add_argument("--build-git-commit")
    parser.add_argument("--zip-path", type=Path)
    args = parser.parse_args()
    try:
        settings = load_rt2_settings(args.experiment_config)
        summary = prepare_benchmark_candidates(
            args.stage1_root,
            args.dataset_root,
            args.output_root,
            candidate_count=args.candidate_count or settings.candidate_count,
            seed=args.seed if args.seed is not None else settings.seed,
            frames_per_sheet=args.frames_per_sheet or settings.frames_per_sheet,
            build_git_commit=args.build_git_commit or _git_commit(),
        )
        zip_path = args.zip_path or (
            args.output_root.parent / "triage_eg_rt2_benchmark_candidates.zip"
        )
        create_candidate_bundle(args.output_root, zip_path)
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
        logging.error("%s", error)
        return 2
    print("candidate sheets:", summary["selected_video_count"])
    print("DOWNLOAD ZIP:", zip_path)
    print("RT2_CANDIDATE_PACK_STATUS = READY")
    print("RT2_BENCHMARK_STATUS = WAITING_FOR_AI_CURATED_LABELS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
