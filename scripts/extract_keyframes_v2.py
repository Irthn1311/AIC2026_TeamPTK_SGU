from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from src.preprocessing.keyframe_v2.pipeline import run_keyframe_v2


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract KEYFRAME V2 for exactly one video.")
    parser.add_argument("--video", required=True, help="Path to one input video.")
    parser.add_argument("--config", default="configs/keyframe_v2.yaml")
    parser.add_argument("--output", default="outputs/keyframe_v2_test")
    parser.add_argument("--validate-btc-mapping", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    summary = run_keyframe_v2(
        video_path=Path(args.video),
        config_path=Path(args.config),
        output_root=Path(args.output),
        validate_btc_mapping=bool(args.validate_btc_mapping),
        debug=bool(args.debug),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
