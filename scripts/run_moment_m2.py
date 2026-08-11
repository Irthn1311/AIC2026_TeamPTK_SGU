#!/usr/bin/env python3
"""Run bounded M2 moment-type-aware temporal plateau evaluation."""

from __future__ import annotations

import argparse
import logging
import subprocess
from pathlib import Path

from triage_eg.experiments.moment_m2 import M2Config, create_m2_bundle, preflight_m2, run_m2


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() or None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--stage1b-root", type=Path, required=True)
    parser.add_argument("--clip-asset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--build-git-commit")
    parser.add_argument("--zip-path", type=Path)
    args = parser.parse_args()
    config = M2Config(
        dataset_root=args.dataset_root,
        candidate_manifest_path=args.candidate_manifest,
        annotation_path=args.annotation,
        stage1b_root=args.stage1b_root,
        clip_asset_root=args.clip_asset_root,
        output_root=args.output_root,
        device=args.device,
        batch_size=args.batch_size,
        build_git_commit=args.build_git_commit or _git_commit(),
    )
    try:
        preflight_m2(config)
        result = run_m2(config)
        zip_path = args.zip_path or (
            args.output_root.parent / "triage_eg_moment_m2_bundle.zip"
        )
        create_m2_bundle(args.output_root, zip_path)
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
        logging.error("%s", error)
        return 2
    print("dense frames encoded:", result["summary"]["dense_frames_encoded"])
    print("DOWNLOAD ZIP:", zip_path)
    print("M2_REAL_STATUS = COMPLETE")
    print("M2_QUALITY_DECISION = NOT_EVALUATED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
