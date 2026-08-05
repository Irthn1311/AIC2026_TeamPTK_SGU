#!/usr/bin/env python3
"""Run verified-only Stage 1B text smoke against an existing Stage 1A index."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from triage_eg.retrieval.stage1b.assets import load_multimodal_encoder
from triage_eg.retrieval.stage1b.contracts import CandidateContract
from triage_eg.retrieval.stage1b.smoke import run_text_smoke
from triage_eg.retrieval.stage1b.writers import write_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-root", type=Path, required=True)
    parser.add_argument("--stage1b-root", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()
    try:
        selected = json.loads(
            (args.stage1b_root / "encoder/selected_encoder_contract.json").read_text(
                encoding="utf-8"
            )
        )
        candidate = CandidateContract.from_dict(selected)
        if candidate.compatibility_status != "VERIFIED":
            raise PermissionError("TEXT_ENCODER_BLOCKED: selected encoder is not VERIFIED")
        queries = [
            json.loads(line)
            for line in args.queries.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        encoder = load_multimodal_encoder(candidate)
        results, status = run_text_smoke(
            candidate,
            encoder,
            queries,
            args.stage1_root,
            args.stage1b_root,
            args.top_k,
        )
        write_jsonl(args.stage1b_root / "smoke/smoke_queries.jsonl", queries)
        write_jsonl(args.stage1b_root / "smoke/smoke_results.jsonl", results)
    except (
        AttributeError,
        FileNotFoundError,
        ImportError,
        OSError,
        PermissionError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        logging.error("%s", error)
        return 2
    print("text smoke:", status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
