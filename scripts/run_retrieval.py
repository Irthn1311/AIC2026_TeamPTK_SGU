"""Run NumPy cosine retrieval against a persisted feature store."""

from __future__ import annotations

import argparse
import sys

from triage_eg.common.config import load_yaml_config, validate_required_keys
from triage_eg.common.run_context import create_run_context
from triage_eg.common.schemas import FrameRecord, load_jsonl, save_jsonl
from triage_eg.features.dummy_encoder import DeterministicDummyEncoder
from triage_eg.features.feature_store import load_feature_store
from triage_eg.retrieval.numpy_index import NumPyFlatCosineIndex
from triage_eg.retrieval.search import RetrievalEngine


def main() -> int:
    """Search features using the configured dummy text encoder."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", required=True, type=int)
    args = parser.parse_args()
    try:
        config = load_yaml_config(args.config)
        validate_required_keys(config, ["feature_store", "frames"])
        vectors, feature_records = load_feature_store(config["feature_store"])
        frames = load_jsonl(config["frames"], FrameRecord)
        encoder = DeterministicDummyEncoder(
            vectors.shape[1], feature_records[0].model_version if feature_records else "0.1"
        )
        index = NumPyFlatCosineIndex()
        index.build(vectors, [record.frame_uid for record in feature_records])
        candidates = RetrievalEngine(encoder, index, frames).search(args.query, args.top_k)
        data_version = frames[0].dataset_version if frames else "UNKNOWN"
        context = create_run_context(
            artifact_name="retrieval",
            config_path=args.config,
            config=config,
            data_version=data_version,
            command=(
                f"python scripts/run_retrieval.py --config {args.config} "
                f"--query {args.query!r} --top-k {args.top_k}"
            ),
            output_root=str(config.get("output_root", "artifacts")),
        )
        config["results_path"] = str(context.artifact_dir / "retrieval_results.jsonl")
        output_path = config.get("results_path", "artifacts/retrieval/latest/results.jsonl")
        save_jsonl(candidates, output_path)
        context.write_manifest("COMPLETED")
    except (FileNotFoundError, KeyError, RuntimeError, TypeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(f"Wrote {len(candidates)} candidates to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
