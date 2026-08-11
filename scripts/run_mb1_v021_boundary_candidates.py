#!/usr/bin/env python3
"""Run MB1 v0.2.1 supplementary boundary-rich candidate mining."""

from __future__ import annotations

import argparse
import logging
import subprocess
from pathlib import Path

from triage_eg.experiments.mb1_v021 import (
    MB1V021Config,
    create_mb1_v021_bundle,
    preflight_mb1_v021,
    prepare_mb1_v021_candidates,
)


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() or None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--old-candidate-manifest", type=Path, required=True)
    parser.add_argument("--old-candidate-diagnostics", type=Path, required=True)
    parser.add_argument("--old-candidate-selection", type=Path, required=True)
    parser.add_argument("--ai-qc", type=Path, required=True)
    parser.add_argument("--ai-qc-summary", type=Path, required=True)
    parser.add_argument("--rt2-benchmark", type=Path, required=True)
    parser.add_argument("--prior-rt2-selection", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--zip-path", type=Path)
    parser.add_argument("--jpeg-quality", type=int, default=88)
    parser.add_argument("--build-git-commit")
    args = parser.parse_args()
    config = MB1V021Config(
        dataset_root=args.dataset_root,
        old_candidate_manifest_path=args.old_candidate_manifest,
        old_candidate_diagnostics_path=args.old_candidate_diagnostics,
        old_candidate_selection_path=args.old_candidate_selection,
        ai_qc_path=args.ai_qc,
        ai_qc_summary_path=args.ai_qc_summary,
        rt2_benchmark_path=args.rt2_benchmark,
        prior_rt2_selection_path=args.prior_rt2_selection,
        output_root=args.output_root,
        jpeg_quality=args.jpeg_quality,
        build_git_commit=args.build_git_commit or _git_commit(),
    )
    try:
        ready = preflight_mb1_v021(config)
        print("frozen seeds:", ready["frozen_seed_count"])
        print("source videos:", ready["source_video_count"])
        result = prepare_mb1_v021_candidates(config)
        zip_path = args.zip_path or (
            args.output_root.parent / "triage_eg_mb1_v021_candidates.zip"
        )
        create_mb1_v021_bundle(args.output_root, zip_path)
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
        logging.error("%s", error)
        return 2
    audit = result["audit"]
    print("OLD_HARD_CUT_RECALL =", audit["OLD_HARD_CUT_RECALL"])
    print("OLD_USABLE_FALSE_VETO_RATE =", audit["OLD_USABLE_FALSE_VETO_RATE"])
    print(
        "retained NEW candidates:",
        result["selection"]["retained_NEW_candidate_count"],
    )
    print("DOWNLOAD ZIP:", zip_path)
    print("MB1_V021_REAL_STATUS = COMPLETE")
    print("MB1_V021_AI_QC_STATUS = WAITING_FOR_AI")
    print("M3_IMPLEMENTATION_STATUS = NOT_STARTED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
