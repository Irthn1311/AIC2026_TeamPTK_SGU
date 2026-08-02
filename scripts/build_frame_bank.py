"""Build shot and frame metadata from a video manifest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from triage_eg.common.config import load_yaml_config
from triage_eg.common.run_context import create_run_context
from triage_eg.common.schemas import save_jsonl
from triage_eg.data.manifest import read_video_manifest_csv
from triage_eg.frame_bank.dummy_shot_detector import DummyShotDetector
from triage_eg.frame_bank.pipeline import FrameBankPipeline
from triage_eg.frame_bank.selectors import CenterFrameSelector


def main() -> int:
    """Run the configured v0.1 frame-bank baseline."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    try:
        config = load_yaml_config(args.config)
        records = read_video_manifest_csv(args.manifest)
        data_version = records[0].dataset_version if records else "UNKNOWN"
        context = create_run_context(
            artifact_name="frame_bank",
            config_path=args.config,
            config=config,
            data_version=data_version,
            command=(
                f"python scripts/build_frame_bank.py --config {args.config} "
                f"--manifest {args.manifest}"
            ),
            output_root=str(config.get("output_root", "artifacts")),
        )
        config["output_path"] = str(context.artifact_dir)
        detector_config = config.get("detector", {})
        selector_config = config.get("selector", {})
        result = FrameBankPipeline(
            DummyShotDetector(
                detector_name=str(detector_config.get("name", "dummy")),
                detector_version=str(detector_config.get("version", "0.1")),
            ),
            CenterFrameSelector(str(selector_config.get("extraction_version", "0.1"))),
        ).run(records)
        output = Path(config.get("output_path", "artifacts/frame_bank/latest"))
        save_jsonl(result.shots, output / "shots.jsonl")
        save_jsonl(result.frames, output / "frames.jsonl")
        context.write_manifest("COMPLETED")
    except (FileNotFoundError, KeyError, TypeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(f"Wrote {len(result.frames)} frames to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
