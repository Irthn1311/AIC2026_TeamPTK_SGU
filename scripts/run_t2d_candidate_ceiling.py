#!/usr/bin/env python3
"""Run the bounded T2-D known-source-video ceiling diagnostic."""

from __future__ import annotations

import argparse
import logging
import subprocess
from pathlib import Path

from triage_eg.experiments.reference_rt2 import load_rt2_benchmark
from triage_eg.experiments.t2d_ceiling import (
    T2DRunnerConfig,
    create_t2d_bundle,
    preflight_t2d,
    run_t2d,
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
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
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
    args = parser.parse_args()
    try:
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
        config = T2DRunnerConfig(stage2, args.benchmark, args.output_root)
        preflight_t2d(config)
        result = run_t2d(config, load_rt2_benchmark(args.benchmark))
        zip_path = args.zip_path or (
            args.output_root.parent / "triage_eg_t2d_ceiling_bundle.zip"
        )
        create_t2d_bundle(args.output_root, zip_path)
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
        logging.error("%s", error)
        return 2
    print("K5 failures:", result["summary"]["k5_failure_count"])
    print("DOWNLOAD ZIP:", zip_path)
    print("T2D_REAL_STATUS = COMPLETE")
    print("ROOT_CAUSE_DECISION = NOT_EVALUATED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
