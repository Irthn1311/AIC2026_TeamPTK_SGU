#!/usr/bin/env python3
"""Run bounded Stage 1B encoder compatibility validation without model downloads."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from triage_eg.retrieval.stage1b import Stage1BConfig, run_stage1b
from triage_eg.retrieval.stage1b.writers import create_stage1b_report_bundle


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--stage0-root", type=Path, required=True)
    parser.add_argument("--stage1-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--candidate-config", type=Path, required=True)
    parser.add_argument("--queries", type=Path)
    parser.add_argument("--candidate-ids", nargs="*", default=())
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--reuse-results", action="store_true")
    parser.add_argument("--strict-root", action="store_true")
    parser.add_argument("--skip-text-smoke", action="store_true")
    parser.add_argument("--build-git-commit")
    parser.add_argument("--zip-path", type=Path)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    try:
        result = run_stage1b(
            Stage1BConfig(
                repo_root=args.repo_root,
                dataset_root=args.dataset_root,
                stage0_root=args.stage0_root,
                stage1_root=args.stage1_root,
                output_root=args.output_root,
                candidate_config=args.candidate_config,
                smoke_queries=args.queries,
                sample_size=args.sample_size,
                seed=args.seed,
                candidate_ids=tuple(args.candidate_ids),
                overwrite=args.overwrite,
                reuse_results=args.reuse_results,
                strict_root=args.strict_root,
                run_text_smoke=not args.skip_text_smoke,
                build_git_commit=args.build_git_commit,
            )
        )
        if args.zip_path:
            create_stage1b_report_bundle(result.output_root, args.zip_path)
    except (
        FileExistsError,
        FileNotFoundError,
        ImportError,
        OSError,
        TypeError,
        ValueError,
    ) as error:
        logging.error("%s", error)
        return 2
    print("output:", result.output_root)
    print("encoder compatibility:", result.summary["readiness"]["encoder_compatibility"])
    print("text retrieval:", result.summary["readiness"]["text_retrieval"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
