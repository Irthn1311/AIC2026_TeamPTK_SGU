#!/usr/bin/env python3
"""Run Stage 1C raw qualitative text retrieval with the verified Stage 1B encoder."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from triage_eg.retrieval.stage1c import Stage1CConfig, create_stage1c_bundle, run_stage1c


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--stage0-root", type=Path, required=True)
    parser.add_argument("--stage1-root", type=Path, required=True)
    parser.add_argument("--stage1b-root", type=Path, required=True)
    parser.add_argument("--encoder-asset-root", type=Path, required=True)
    parser.add_argument("--query-suite", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--frame-top-k", type=int, default=50)
    parser.add_argument("--kis-top-k", type=int, default=100)
    parser.add_argument("--review-top-k", type=int, default=10)
    parser.add_argument("--contact-sheet-top-k", type=int, default=20)
    parser.add_argument("--query-ids", nargs="*", default=())
    parser.add_argument("--languages", nargs="*", default=())
    parser.add_argument("--categories", nargs="*", default=())
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "cuda:0"), default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--reuse-results", action="store_true")
    parser.add_argument("--strict-root", action="store_true")
    parser.add_argument("--skip-contact-sheets", action="store_true")
    parser.add_argument("--build-git-commit")
    parser.add_argument("--zip-path", type=Path)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    try:
        result = run_stage1c(
            Stage1CConfig(
                repo_root=args.repo_root,
                dataset_root=args.dataset_root,
                stage0_root=args.stage0_root,
                stage1_root=args.stage1_root,
                stage1b_root=args.stage1b_root,
                encoder_asset_root=args.encoder_asset_root,
                query_suite=args.query_suite,
                output_root=args.output_root,
                frame_top_k=args.frame_top_k,
                kis_top_k=args.kis_top_k,
                review_top_k=args.review_top_k,
                contact_sheet_top_k=args.contact_sheet_top_k,
                query_ids=tuple(args.query_ids),
                languages=tuple(args.languages),
                categories=tuple(args.categories),
                device=args.device,
                batch_size=args.batch_size,
                overwrite=args.overwrite,
                reuse_results=args.reuse_results,
                strict_root=args.strict_root,
                skip_contact_sheets=args.skip_contact_sheets,
                build_git_commit=args.build_git_commit,
            )
        )
        if args.zip_path:
            create_stage1c_bundle(result.output_root, args.zip_path)
    except (
        FileExistsError,
        FileNotFoundError,
        ImportError,
        OSError,
        RuntimeError,
        ValueError,
    ) as error:
        logging.error("%s", error)
        return 2
    print("output:", result.output_root)
    print("evaluation:", result.summary["evaluation_status"])
    print("retrieval quality:", result.summary["retrieval_quality_status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
