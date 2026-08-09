#!/usr/bin/env python3
"""Validate and score a filled Stage 1C human-review CSV without model or index access."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from triage_eg.retrieval.stage1c import score_human_review


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    roots = parser.add_mutually_exclusive_group(required=True)
    roots.add_argument("--stage1c-root", type=Path)
    roots.add_argument("--bundle-dir", type=Path)
    parser.add_argument("--review-csv", type=Path, required=True)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    try:
        metrics = score_human_review(args.stage1c_root or args.bundle_dir, args.review_csv)
    except (FileNotFoundError, KeyError, OSError, TypeError, ValueError) as error:
        logging.error("%s", error)
        return 2
    print("human review status:", metrics["human_review_status"])
    print("retrieval quality:", metrics["retrieval_quality_status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

