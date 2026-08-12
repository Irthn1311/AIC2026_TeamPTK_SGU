#!/usr/bin/env python3
"""Run MB1 v0.2.2 trusted-source continuity repair."""

from __future__ import annotations

import argparse
import logging
import subprocess
from pathlib import Path

from triage_eg.experiments.mb1_v022 import (
    MB1V022Config,
    create_mb1_v022_bundle,
    preflight_mb1_v022,
    prepare_mb1_v022_candidates,
)


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() or None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--v021-seed-manifest", type=Path, required=True)
    parser.add_argument("--v021-candidate-manifest", type=Path, required=True)
    parser.add_argument("--v021-candidate-diagnostics", type=Path, required=True)
    parser.add_argument("--v021-selection", type=Path, required=True)
    parser.add_argument("--v021-cut-audit", type=Path, required=True)
    parser.add_argument("--old-v02-candidate-manifest", type=Path, required=True)
    parser.add_argument("--ai-qc", type=Path, required=True)
    parser.add_argument("--rt2-benchmark", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--zip-path", type=Path)
    parser.add_argument("--jpeg-quality", type=int, default=88)
    parser.add_argument("--build-git-commit")
    args = parser.parse_args()
    config = MB1V022Config(
        dataset_root=args.dataset_root,
        v021_seed_manifest_path=args.v021_seed_manifest,
        v021_candidate_manifest_path=args.v021_candidate_manifest,
        v021_candidate_diagnostics_path=args.v021_candidate_diagnostics,
        v021_selection_path=args.v021_selection,
        v021_cut_audit_path=args.v021_cut_audit,
        old_v02_candidate_manifest_path=args.old_v02_candidate_manifest,
        ai_qc_path=args.ai_qc,
        rt2_benchmark_path=args.rt2_benchmark,
        output_root=args.output_root,
        jpeg_quality=args.jpeg_quality,
        build_git_commit=args.build_git_commit or _git_commit(),
    )
    try:
        ready = preflight_mb1_v022(config)
        print("frozen seeds:", ready["frozen_seed_count"])
        print("trusted source videos:", ready["trusted_source_count"])
        result = prepare_mb1_v022_candidates(config)
        zip_path = args.zip_path or (args.output_root.parent / "triage_eg_mb1_v022_candidates.zip")
        create_mb1_v022_bundle(args.output_root, zip_path)
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
        logging.error("%s", error)
        return 2
    print("OLD_HARD_CUT_RECALL =", result["old_qc_audit"]["OLD_HARD_CUT_RECALL"])
    print(
        "OLD_USABLE_FALSE_VETO_RATE =",
        result["old_qc_audit"]["OLD_USABLE_FALSE_VETO_RATE"],
    )
    print(
        "V021_REJECTED_BY_CONTEXT_GEOMETRY =",
        result["v021_geometry_audit"]["V021_REJECTED_BY_CONTEXT_GEOMETRY"],
    )
    print("retained NEW:", result["selection"]["retained_NEW_candidates"])
    print("DOWNLOAD ZIP:", zip_path)
    print("MB1_V022_REAL_STATUS = COMPLETE")
    print("MB1_V022_AI_QC_STATUS = WAITING_FOR_AI")
    print("M3_IMPLEMENTATION_STATUS = NOT_STARTED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
