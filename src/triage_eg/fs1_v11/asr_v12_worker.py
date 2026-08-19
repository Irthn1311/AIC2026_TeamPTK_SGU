"""Subprocess entry point for one isolated Whisper replica and one ASR shard."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .asr_v12 import worker_run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--progress", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, choices=(4, 8, 12, 16), required=True)
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()
    report = worker_run(
        args.manifest,
        args.checkpoint,
        args.progress,
        args.asset_root,
        args.batch_size,
        benchmark=args.benchmark,
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
