#!/usr/bin/env python3
"""Ingest Stage 1D AI review evidence and freeze the internal language path."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from triage_eg.retrieval.stage1e import run_stage1e_language_path_freeze


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1d-root", type=Path, required=True)
    parser.add_argument("--ai-review-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--build-git-commit")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    try:
        result = run_stage1e_language_path_freeze(
            args.stage1d_root,
            args.ai_review_root,
            args.output_root,
            build_git_commit=args.build_git_commit,
        )
    except (
        FileExistsError,
        FileNotFoundError,
        KeyError,
        OSError,
        RuntimeError,
        ValueError,
    ) as error:
        logging.error("%s", error)
        return 2
    print("STAGE1E_EXECUTION =", result["stage1e_execution"])
    print("AI_REVIEW_STATUS =", result["ai_review_status"])
    print("HUMAN_REVIEW_STATUS =", result["human_review_status"])
    print("LANGUAGE_PATH_STATUS =", result["language_path_status"])
    print("STAGE2_READINESS =", result["stage2_readiness"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
