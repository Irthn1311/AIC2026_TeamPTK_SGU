#!/usr/bin/env python3
"""Run bounded RT1 whole-query, unordered-event, and DANTE reference arms."""

from __future__ import annotations

import argparse
import logging
import subprocess
from dataclasses import replace
from pathlib import Path

from triage_eg.experiments.reference_rt1 import (
    RT1RunnerConfig,
    create_rt1_bundle,
    load_rt1_queries,
    load_rt1_settings,
    run_reference_rt1,
)
from triage_eg.retrieval.stage2 import config_from_yaml


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() or None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-root", type=Path, required=True)
    parser.add_argument("--stage1b-root", type=Path, required=True)
    parser.add_argument("--stage1e-root", type=Path, required=True)
    parser.add_argument("--clip-asset-root", type=Path, required=True)
    parser.add_argument("--translator-asset-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--query-suite",
        type=Path,
        default=Path("configs/experiments/reference_rt1_queries.jsonl"),
    )
    parser.add_argument(
        "--experiment-config",
        type=Path,
        default=Path("configs/experiments/reference_rt1.yaml"),
    )
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
    parser.add_argument("--dante-lambda", type=float)
    parser.add_argument("--build-git-commit")
    parser.add_argument("--zip-path", type=Path)
    args = parser.parse_args()
    try:
        settings = load_rt1_settings(args.experiment_config)
        if args.dante_lambda is not None:
            settings = replace(settings, dante_lambda=args.dante_lambda)
        commit = args.build_git_commit or _git_commit()
        stage2 = config_from_yaml(
            args.runtime_config,
            stage1_root=args.stage1_root,
            stage1b_root=args.stage1b_root,
            stage1e_root=args.stage1e_root,
            clip_asset_root=args.clip_asset_root,
            translator_asset_root=args.translator_asset_root,
            output_root=args.output_root / "_stage2_control",
            stage1d_config=args.stage1d_config,
            build_git_commit=commit,
        )
        queries = load_rt1_queries(args.query_suite)
        summary = run_reference_rt1(
            RT1RunnerConfig(
                stage2,
                args.dataset_root,
                args.query_suite,
                args.output_root,
                settings,
            ),
            queries,
        )
        zip_path = args.zip_path or (args.output_root.parent / "triage_eg_reference_rt1_bundle.zip")
        create_rt1_bundle(args.output_root, zip_path)
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
        logging.error("%s", error)
        return 2
    print("queries completed:", len(summary["queries"]))
    print("DOWNLOAD ZIP:", zip_path)
    print("RT1_REAL_EXPERIMENT_STATUS = COMPLETE")
    print("DANTE_REFERENCE_STATUS = REAL_EXPERIMENT_COMPLETE")
    print("QUALITY_DECISION = EXPLORATORY_NO_GT")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
