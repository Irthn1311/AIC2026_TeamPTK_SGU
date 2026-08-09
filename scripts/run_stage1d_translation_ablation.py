#!/usr/bin/env python3
"""Run the Stage 1D frozen-baseline Vietnamese translation-bridge ablation."""

from __future__ import annotations

import argparse
import logging
from dataclasses import replace
from pathlib import Path

from triage_eg.retrieval.stage1d import (
    Stage1DConfig,
    create_stage1d_bundle,
    resolve_input_root,
    run_stage1d,
    settings_from_yaml,
)
from triage_eg.retrieval.stage1d.inputs import STAGE1C_REQUIRED, TRANSLATOR_REQUIRED


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--stage0-root", type=Path, required=True)
    parser.add_argument("--stage1-root", type=Path, required=True)
    parser.add_argument("--stage1b-root", type=Path, required=True)
    parser.add_argument("--stage1c-root", type=Path, required=True)
    parser.add_argument("--clip-asset-root", type=Path, required=True)
    parser.add_argument("--translator-asset-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--pair-ids", nargs="*", default=())
    parser.add_argument(
        "--translator-device", choices=("auto", "cpu", "cuda", "cuda:0")
    )
    parser.add_argument("--clip-device", choices=("auto", "cpu", "cuda", "cuda:0"))
    parser.add_argument("--translator-batch-size", type=int)
    parser.add_argument("--reuse-translations", action="store_true")
    parser.add_argument("--reuse-results", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--strict-root", action="store_true")
    parser.add_argument("--skip-contact-sheets", action="store_true")
    parser.add_argument("--build-git-commit")
    parser.add_argument("--zip-path", type=Path)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    try:
        translator, generation, retrieval, review = settings_from_yaml(args.config)
        if args.translator_device:
            translator = replace(translator, device=args.translator_device)
        if args.translator_batch_size:
            translator = replace(translator, batch_size=args.translator_batch_size)
        search_root = Path("/kaggle/input") if Path("/kaggle/input").is_dir() else None
        stage1c_root, stage1c_mode = resolve_input_root(
            args.stage1c_root,
            required=STAGE1C_REQUIRED,
            search_root=search_root,
            materialize_root=args.output_root.parent / "triage_eg_stage1c_frozen_runtime",
            archive_keyword="stage1c",
        )
        translator_root, translator_mode = resolve_input_root(
            args.translator_asset_root,
            required=TRANSLATOR_REQUIRED,
            search_root=search_root,
            materialize_root=args.output_root.parent / "triage_eg_opus_mt_vi_en_runtime",
            archive_keyword="opus-mt-vi-en",
        )
        result = run_stage1d(
            Stage1DConfig(
                repo_root=args.repo_root,
                dataset_root=args.dataset_root,
                stage0_root=args.stage0_root,
                stage1_root=args.stage1_root,
                stage1b_root=args.stage1b_root,
                stage1c_root=stage1c_root,
                clip_asset_root=args.clip_asset_root,
                translator_asset_root=translator_root,
                output_root=args.output_root,
                translator=translator,
                generation=generation,
                retrieval=retrieval,
                review=review,
                pair_ids=tuple(args.pair_ids),
                clip_device=args.clip_device or "auto",
                overwrite=args.overwrite,
                reuse_translations=args.reuse_translations,
                reuse_results=args.reuse_results,
                strict_root=args.strict_root,
                skip_contact_sheets=args.skip_contact_sheets,
                build_git_commit=args.build_git_commit,
                stage1c_materialization=stage1c_mode,
                translator_asset_materialization=translator_mode,
            )
        )
        if args.zip_path:
            create_stage1d_bundle(result.output_root, args.zip_path)
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
    print("stage1d execution:", result.summary["execution_status"])
    print("translator:", result.summary["translator"]["asset_status"])
    print("translated retrieval:", result.summary["retrieval"]["translated_queries_completed"])
    print("human review:", result.summary["human_review"]["status"])
    print("language bridge quality:", result.summary["language_bridge_quality_status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

