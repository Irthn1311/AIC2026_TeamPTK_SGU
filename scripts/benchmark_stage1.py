#!/usr/bin/env python3
"""Run Stage 1 self-integrity and exact-search latency benchmarks."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from triage_eg.retrieval.stage1.benchmark import run_benchmark, run_self_retrieval


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-root", type=Path, required=True)
    parser.add_argument("--random-queries", type=int, default=50)
    parser.add_argument("--self-queries", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--backend", choices=("numpy_exact", "faiss_flat_ip"), default="numpy_exact"
    )
    parser.add_argument("--metric", choices=("cosine", "dot"), default="cosine")
    parser.add_argument("--search-chunk-rows", type=int, default=16_384)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    if args.backend != "numpy_exact":
        logging.error("faiss_flat_ip is optional and not enabled by this baseline CLI")
        return 2
    try:
        self_report = run_self_retrieval(
            args.stage1_root,
            samples=args.self_queries,
            top_k=5,
            seed=args.seed,
            chunk_rows=args.search_chunk_rows,
        )
        report = run_benchmark(
            args.stage1_root,
            random_queries=args.random_queries,
            self_queries=args.self_queries,
            top_k=args.top_k,
            seed=args.seed,
            chunk_rows=args.search_chunk_rows,
            metric=args.metric,
        )
    except (FileNotFoundError, OSError, TypeError, ValueError) as error:
        logging.error("%s", error)
        return 2
    print("self_retrieval:", self_report["status"])
    print("latency:", report["latency_seconds"])
    return 0 if self_report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
