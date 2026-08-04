#!/usr/bin/env python3
"""Build or safely reuse the Stage 1 BTC global exact-search index."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from triage_eg.retrieval.stage1 import Stage1BuildConfig, build_index


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage0-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--backend", choices=("numpy_exact", "faiss_flat_ip"), default="numpy_exact"
    )
    parser.add_argument("--metric", choices=("cosine", "dot"), default="cosine")
    parser.add_argument("--search-chunk-rows", type=int, default=16_384)
    parser.add_argument("--expected-rows", type=int, default=177_321)
    parser.add_argument("--expected-videos", type=int, default=873)
    parser.add_argument("--self-queries", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--reuse-index", action="store_true")
    parser.add_argument("--strict-root", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    try:
        result = build_index(
            Stage1BuildConfig(
                stage0_root=args.stage0_root,
                dataset_root=args.dataset_root,
                output_root=args.output_root,
                backend=args.backend,
                metric=args.metric,
                search_chunk_rows=args.search_chunk_rows,
                expected_rows=args.expected_rows,
                expected_videos=args.expected_videos,
                self_queries=args.self_queries,
                overwrite=args.overwrite,
                reuse_index=args.reuse_index,
                strict_root=args.strict_root,
            )
        )
    except (
        FileNotFoundError,
        FileExistsError,
        ImportError,
        OSError,
        TypeError,
        ValueError,
    ) as error:
        logging.error("%s", error)
        return 2
    print(result.output_root)
    print("reused:", result.reused)
    print("self_retrieval:", result.summary["self_retrieval_status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
