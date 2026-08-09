#!/usr/bin/env python3
"""Run one query through the frozen Stage 2A operational retrieval runtime."""

from __future__ import annotations

import argparse
import hashlib
import logging
from pathlib import Path

from triage_eg.retrieval.stage2 import (
    OperationalRetrievalRuntime,
    QueryRequest,
    Stage2RuntimeError,
    config_from_yaml,
    create_stage2_report_bundle,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-root", type=Path, required=True)
    parser.add_argument("--stage1b-root", type=Path, required=True)
    parser.add_argument("--stage1e-root", type=Path, required=True)
    parser.add_argument("--clip-asset-root", type=Path, required=True)
    parser.add_argument("--translator-asset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--language", choices=("en", "vi", "auto"))
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--query-id")
    parser.add_argument(
        "--runtime-config",
        type=Path,
        default=Path("configs/retrieval/stage2_operational_runtime.yaml"),
    )
    parser.add_argument(
        "--stage1d-config",
        type=Path,
        default=Path("configs/retrieval/stage1d_translation_ablation.yaml"),
    )
    parser.add_argument("--build-git-commit")
    parser.add_argument("--zip-path", type=Path)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    query_id = args.query_id or f"query_{hashlib.sha256(args.text.encode()).hexdigest()[:12]}"
    try:
        config = config_from_yaml(
            args.runtime_config,
            stage1_root=args.stage1_root,
            stage1b_root=args.stage1b_root,
            stage1e_root=args.stage1e_root,
            clip_asset_root=args.clip_asset_root,
            translator_asset_root=args.translator_asset_root,
            output_root=args.output_root,
            stage1d_config=args.stage1d_config,
            build_git_commit=args.build_git_commit,
        )
        request = QueryRequest(
            query_id,
            args.text,
            args.language or config.default_language,
            args.top_k if args.top_k is not None else config.default_top_k,
        )
        runtime = OperationalRetrievalRuntime(config)
        try:
            runtime.load()
            result = runtime.search_one(request)
        finally:
            runtime.close()
        zip_path = args.zip_path or (
            args.output_root.parent / "triage_eg_stage2a_operational_runtime_reports.zip"
        )
        create_stage2_report_bundle(args.output_root, zip_path)
    except (
        FileExistsError,
        FileNotFoundError,
        ImportError,
        OSError,
        RuntimeError,
        Stage2RuntimeError,
        ValueError,
    ) as error:
        logging.error("%s", error)
        return 2
    print("query:", result.query_id)
    print("resolved language:", result.language_resolution["resolved_language"])
    print("translation applied:", result.encoding["translation_applied"])
    print("raw results:", len(result.ranked_frames))
    print("DOWNLOAD ZIP:", zip_path)
    print("OPERATIONAL_RETRIEVAL_RUNTIME_STATUS = READY_FOR_REAL_SMOKE")
    print("RANKING_QUALITY_STATUS = UNCHANGED_FROM_FROZEN_BASELINE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
