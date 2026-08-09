#!/usr/bin/env python3
"""Validate and score a filled Stage 1D blinded human-review CSV."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from triage_eg.retrieval.stage1d import score_stage1d_review


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    roots = parser.add_mutually_exclusive_group(required=True)
    roots.add_argument("--stage1d-root", type=Path)
    roots.add_argument("--bundle-dir", type=Path)
    parser.add_argument("--review-csv", type=Path, required=True)
    parser.add_argument("--review-key", type=Path)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    try:
        metrics = score_stage1d_review(
            args.stage1d_root or args.bundle_dir,
            args.review_csv,
            args.review_key,
        )
    except (
        FileNotFoundError,
        KeyError,
        OSError,
        TypeError,
        ValueError,
    ) as error:
        logging.error("%s", error)
        return 2
    print("human review status:", metrics["human_review_status"])
    print("language bridge quality:", metrics["language_bridge_quality_status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

