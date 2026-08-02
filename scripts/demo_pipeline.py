"""Run the complete v0.1 metadata-to-retrieval demonstration."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from triage_eg.common.config import load_yaml_config, validate_required_keys
from triage_eg.common.run_context import create_run_context
from triage_eg.common.schemas import VideoRecord, save_jsonl
from triage_eg.features.dummy_encoder import DeterministicDummyEncoder
from triage_eg.features.extractor import extract_frame_features
from triage_eg.features.feature_store import save_feature_store
from triage_eg.frame_bank.dummy_shot_detector import DummyShotDetector
from triage_eg.frame_bank.pipeline import FrameBankPipeline
from triage_eg.frame_bank.selectors import CenterFrameSelector
from triage_eg.retrieval.grouping import group_candidates_by_video
from triage_eg.retrieval.numpy_index import NumPyFlatCosineIndex
from triage_eg.retrieval.search import RetrievalEngine

WARNING = "Dummy encoder is not a semantic model. Results only validate the software pipeline."


def _synthetic_videos(data_version: str) -> list[VideoRecord]:
    return [
        VideoRecord(
            video_id="demo_video_001",
            relative_path="synthetic/demo_video_001.mp4",
            batch_id="demo",
            fps=25.0,
            total_frames=250,
            duration_ms=10_000,
            width=1920,
            height=1080,
            has_audio=True,
            dataset_version=data_version,
        ),
        VideoRecord(
            video_id="demo_video_002",
            relative_path="synthetic/demo_video_002.mp4",
            batch_id="demo",
            fps=30.0,
            total_frames=360,
            duration_ms=12_000,
            width=1280,
            height=720,
            has_audio=False,
            dataset_version=data_version,
        ),
    ]


def run_demo(config_path: str | Path, output_root: str | Path | None = None) -> Path:
    """Execute the synthetic pipeline and return the created artifact directory."""

    config = load_yaml_config(config_path)
    validate_required_keys(
        config,
        [
            "data.dataset_version",
            "features.dimension",
            "features.encoder_version",
            "retrieval.top_k",
            "artifact.name",
            "query",
        ],
    )
    data_version = str(config["data"]["dataset_version"])
    artifact_config = config["artifact"]
    selected_output_root = output_root or str(artifact_config.get("output_root", "artifacts"))
    command = f"python scripts/demo_pipeline.py --config {config_path}"
    context = create_run_context(
        artifact_name=str(artifact_config["name"]),
        config_path=config_path,
        config=config,
        data_version=data_version,
        command=command,
        output_root=selected_output_root,
    )

    detector = DummyShotDetector(
        detector_name=str(config["frame_bank"].get("detector_name", "dummy")),
        detector_version=str(config["frame_bank"].get("detector_version", "0.1")),
    )
    selector = CenterFrameSelector(
        extraction_version=str(config["frame_bank"].get("extraction_version", "0.1"))
    )
    frame_bank = FrameBankPipeline(detector, selector).run(_synthetic_videos(data_version))
    save_jsonl(frame_bank.frames, context.artifact_dir / "frames.jsonl")

    encoder = DeterministicDummyEncoder(
        dimension=int(config["features"]["dimension"]),
        model_version=str(config["features"]["encoder_version"]),
    )
    vectors, feature_records = extract_frame_features(
        frame_bank.frames, encoder, context.artifact_dir / "features"
    )
    save_feature_store(context.artifact_dir / "features", vectors, feature_records)

    index = NumPyFlatCosineIndex()
    index.build(vectors, [frame.frame_uid for frame in frame_bank.frames])
    engine = RetrievalEngine(encoder, index, frame_bank.frames)
    candidates = engine.search(str(config["query"]), int(config["retrieval"]["top_k"]))
    save_jsonl(candidates, context.artifact_dir / "retrieval_results.jsonl")
    context.write_manifest("COMPLETED")

    print(WARNING)
    print(f"Artifact: {context.artifact_dir}")
    print(
        f"Frame bank: {frame_bank.report.total_shots} shots, "
        f"{frame_bank.report.total_selected_frames} selected frames"
    )
    for video_id, video_candidates in group_candidates_by_video(candidates).items():
        best = video_candidates[0]
        print(
            f"#{best.rank} {video_id} frame={best.frame_id} "
            f"timestamp_ms={best.timestamp_ms} score={best.score:.4f}"
        )
    return context.artifact_dir


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Experiment YAML path")
    parser.add_argument("--output-root", help="Optional artifact root override")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""

    args = parse_args(argv)
    try:
        run_demo(args.config, args.output_root)
    except (FileNotFoundError, KeyError, TypeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
