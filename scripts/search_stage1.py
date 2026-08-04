#!/usr/bin/env python3
"""Search Stage 1 by vector or gated text query."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from triage_eg.retrieval.stage1.contracts import SearchConfig
from triage_eg.retrieval.stage1.encoder import (
    compatibility_gate,
    load_encoder_contract,
    load_text_encoder,
)
from triage_eg.retrieval.stage1.runner import load_query_vector, search_text, search_vector


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-root", type=Path, required=True)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--query-vector", type=Path)
    modes.add_argument("--query-text")
    modes.add_argument("--queries-jsonl", type=Path)
    parser.add_argument("--query-id", default="query")
    parser.add_argument("--encoder-config", type=Path)
    parser.add_argument("--allow-unverified-encoder", action="store_true")
    parser.add_argument(
        "--backend", choices=("numpy_exact", "faiss_flat_ip"), default="numpy_exact"
    )
    parser.add_argument("--metric", choices=("cosine", "dot"), default="cosine")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--max-predictions", type=int, default=100)
    parser.add_argument("--video-grouping", choices=("max", "mean_top_k"), default="max")
    parser.add_argument("--csv-header", choices=("yes", "no"), default="yes")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    if args.backend != "numpy_exact":
        logging.error("faiss_flat_ip runtime is optional and not enabled by this baseline CLI")
        return 2

    def config(query_id: str) -> SearchConfig:
        return SearchConfig(
            args.stage1_root,
            query_id,
            top_k=args.top_k,
            metric=args.metric,
            video_grouping=args.video_grouping,
            max_predictions=args.max_predictions,
            csv_header=args.csv_header == "yes",
        )

    try:
        if args.query_vector:
            _, paths = search_vector(load_query_vector(args.query_vector), config(args.query_id))
            print(paths)
            return 0
        contract = load_encoder_contract(args.encoder_config)
        compatibility_gate(contract, allow_unverified=args.allow_unverified_encoder)
        encoder = load_text_encoder(contract)
        queries = (
            [{"query_id": args.query_id, "text": args.query_text}]
            if args.query_text
            else [
                json.loads(line)
                for line in args.queries_jsonl.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        )
        if not queries:
            raise ValueError("Batch query file must contain at least one query")
        query_ids = [str(query["query_id"]) for query in queries]
        if len(set(query_ids)) != len(query_ids):
            raise ValueError("Batch query_id values must be unique")
        for query in queries:
            _, paths = search_text(
                str(query["text"]),
                config(str(query["query_id"])),
                contract,
                encoder,
                allow_unverified=args.allow_unverified_encoder,
            )
            print(paths)
    except (
        FileNotFoundError,
        ImportError,
        KeyError,
        PermissionError,
        OSError,
        TypeError,
        ValueError,
    ) as error:
        logging.error("%s", error)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
