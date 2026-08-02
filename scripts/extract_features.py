"""Extract deterministic v0.1 features for frame metadata."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from triage_eg.common.config import load_yaml_config
from triage_eg.common.run_context import create_run_context
from triage_eg.common.schemas import FrameRecord, load_jsonl
from triage_eg.features.dummy_encoder import DeterministicDummyEncoder
from triage_eg.features.extractor import extract_frame_features
from triage_eg.features.feature_store import save_feature_store


def main() -> int:
    """Encode frame records and persist a feature store."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--frames", required=True)
    args = parser.parse_args()
    try:
        config = load_yaml_config(args.config)
        encoder_config = config.get("encoder", config.get("features", {}))
        frames = load_jsonl(args.frames, FrameRecord)
        data_version = frames[0].dataset_version if frames else "UNKNOWN"
        context = create_run_context(
            artifact_name="features",
            config_path=args.config,
            config=config,
            data_version=data_version,
            command=(
                f"python scripts/extract_features.py --config {args.config} --frames {args.frames}"
            ),
            output_root=str(config.get("output_root", "artifacts")),
        )
        config["output_path"] = str(context.artifact_dir)
        encoder = DeterministicDummyEncoder(
            int(encoder_config.get("dimension", 32)), str(encoder_config.get("version", "0.1"))
        )
        output = Path(config.get("output_path", "artifacts/features/latest"))
        vectors, records = extract_frame_features(frames, encoder, output)
        save_feature_store(output, vectors, records)
        context.write_manifest("COMPLETED")
    except (FileNotFoundError, KeyError, TypeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(f"Wrote {len(records)} feature rows to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
